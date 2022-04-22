# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import pathlib
import sys
from unicodedata import name
import pytest
import numpy as np
import logging
import onnx
import json

import tvm.testing
from tvm import te
from tvm import relay
from tvm.relay.backend import Executor, Runtime
from tvm.contrib import utils, ndk
from tvm.runtime import ndarray
from tvm.contrib.hexagon.build import HexagonLauncher
import tvm.contrib.hexagon as hexagon

from .conftest import requires_hexagon_toolchain


@requires_hexagon_toolchain
def test_add(hexagon_session):
    dtype = "int8"
    A = tvm.te.placeholder((2,), dtype=dtype)
    B = tvm.te.placeholder((1,), dtype=dtype)
    C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name="C")
    sched = tvm.te.create_schedule(C.op)

    target_hexagon = tvm.target.hexagon("v68", link_params=True)
    func = tvm.build(
        sched, [A, B, C], tvm.target.Target(target_hexagon, host=target_hexagon), name="add"
    )

    if hexagon_session is None:
        pytest.skip(msg="Skip hardware test, ANDROID_SERIAL_NUMBER is not set.")

    mod = hexagon_session.load_module(func)

    A_data = tvm.nd.array(np.array([2, 3], dtype=dtype), device=hexagon_session.device)
    assert (A_data.numpy() == np.array([2, 3])).all()
    B_data = tvm.nd.array(np.array([4], dtype=dtype), device=hexagon_session.device)
    assert (B_data.numpy() == np.array([4])).all()
    C_data = tvm.nd.array(np.array([0, 0], dtype=dtype), device=hexagon_session.device)
    assert (C_data.numpy() == np.array([0, 0])).all()
    mod["add"](A_data, B_data, C_data)
    assert (C_data.numpy() == np.array([6, 7])).all()


@requires_hexagon_toolchain
def test_add_vtcm(hexagon_session):
    dtype = "int8"
    A = tvm.te.placeholder((2,), dtype=dtype)
    B = tvm.te.placeholder((1,), dtype=dtype)
    C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name="C")
    sched = tvm.te.create_schedule(C.op)

    target_hexagon = tvm.target.hexagon("v68", link_params=True)
    func = tvm.build(
        sched, [A, B, C], tvm.target.Target(target_hexagon, host=target_hexagon), name="add"
    )

    if hexagon_session is None:
        pytest.skip(msg="Skip hardware test, ANDROID_SERIAL_NUMBER is not set.")

    mod = hexagon_session.load_module(func)
    A_data = tvm.nd.empty(A.shape, A.dtype, hexagon_session.device, "global.vtcm")
    A_data.copyfrom(np.array([2, 3]))

    B_data = tvm.nd.empty(B.shape, B.dtype, hexagon_session.device, "global.vtcm")
    B_data.copyfrom(np.array([4]))

    C_data = tvm.nd.empty(C.shape, C.dtype, hexagon_session.device, "global.vtcm")
    C_data.copyfrom(np.array([0, 0]))

    mod["add"](A_data, B_data, C_data)
    result = C_data.numpy()
    assert (result == np.array([6, 7])).all()


class TestMatMul:
    M = tvm.testing.parameter(32)
    N = tvm.testing.parameter(32)
    K = tvm.testing.parameter(32)

    @requires_hexagon_toolchain
    def test_matmul(self, hexagon_session, M, N, K):
        X = te.placeholder((M, K), dtype="float32")
        Y = te.placeholder((K, N), dtype="float32")
        k1 = te.reduce_axis((0, K), name="k1")
        Z = te.compute((M, N), lambda i, j: te.sum(X[i, k1] * Y[k1, j], axis=[k1]))
        schedule = te.create_schedule(Z.op)

        target_hexagon = tvm.target.hexagon("v68", link_params=True)
        func = tvm.build(
            schedule, [X, Y, Z], tvm.target.Target(target_hexagon, host=target_hexagon)
        )

        if hexagon_session is None:
            pytest.skip(msg="Skip hardware test, ANDROID_SERIAL_NUMBER is not set.")

        mod = hexagon_session.load_module(func)

        x = np.random.uniform(size=[i.value for i in X.shape]).astype(X.dtype)
        y = np.random.uniform(size=[i.value for i in Y.shape]).astype(Y.dtype)
        z = np.zeros([i.value for i in Z.shape], dtype=Z.dtype)

        xt = tvm.nd.array(x, device=hexagon_session.device)
        yt = tvm.nd.array(y, device=hexagon_session.device)
        zt = tvm.nd.array(z, device=hexagon_session.device)
        mod(xt, yt, zt)

        target_llvm = tvm.target.Target("llvm")
        mod = tvm.build(schedule, [X, Y, Z], tvm.target.Target(target_llvm, host=target_llvm))
        device = tvm.cpu(0)
        xtcpu = tvm.nd.array(x, device)
        ytcpu = tvm.nd.array(y, device)
        ztcpu = tvm.nd.array(z, device)
        mod(xtcpu, ytcpu, ztcpu)

        tvm.testing.assert_allclose(zt.numpy(), ztcpu.numpy(), rtol=1e-4)


@requires_hexagon_toolchain
def test_graph_executor(hexagon_session):
    dtype = "float32"
    data = relay.var("data", relay.TensorType((1, 64, 64, 3), dtype))
    weight = relay.var("weight", relay.TensorType((5, 5, 3, 8), dtype))
    y = relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    f = relay.Function([data, weight], y)
    relay_mod = tvm.IRModule.from_expr(f)
    relay_mod = relay.transform.InferType()(relay_mod)

    target_hexagon = tvm.target.hexagon("v68")
    runtime = Runtime("cpp")
    executor = Executor("graph")

    weight_in = np.random.rand(5, 5, 3, 8).astype(dtype=dtype)
    data_in = np.random.rand(1, 64, 64, 3).astype(dtype=dtype)
    params = {"weight": weight_in}
    inputs = {"data": data_in}

    with tvm.transform.PassContext(opt_level=3):
        lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_hexagon, host=target_hexagon),
            runtime=runtime,
            executor=executor,
        )

    if hexagon_session is None:
        pytest.skip(msg="Skip hardware test since ANDROID_SERIAL_NUMBER is not set.")

    graph_mod = hexagon_session.get_executor_from_factory(lowered)
    graph_mod.set_input(**params)
    graph_mod.run(**inputs)
    hexagon_output = graph_mod.get_output(0).numpy()

    target_llvm = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            runtime=runtime,
            executor=executor,
        )
    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    llvm_graph_mod.set_input(**params)
    llvm_graph_mod.run(**inputs)
    expected_output = llvm_graph_mod.get_output(0).numpy()

    tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-4, atol=1e-5)


@requires_hexagon_toolchain
def test_graph_executor_multiple_conv2d(hexagon_session):
    dtype = "float32"
    input_shape = (1, 8, 8, 3)
    w1_shape = (5, 5, 3, 1)
    w2_shape = (5, 5, 1, 3)
    data = relay.var("data", relay.TensorType(input_shape, dtype))
    weight1 = relay.var("weight1", relay.TensorType(w1_shape, dtype))
    weight2 = relay.var("weight2", relay.TensorType(w2_shape, dtype))
    y1 = relay.nn.conv2d(
        data,
        weight1,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    y2 = relay.nn.conv2d(
        y1,
        weight2,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    f = relay.Function([data, weight1, weight2], y2)
    relay_mod = tvm.IRModule.from_expr(f)
    relay_mod = relay.transform.InferType()(relay_mod)

    target_hexagon = tvm.target.hexagon("v68")
    runtime = Runtime("cpp")
    executor = Executor("graph")

    with tvm.transform.PassContext(opt_level=3):
        lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_hexagon, host=target_hexagon),
            runtime=runtime,
            executor=executor,
        )

    if hexagon_session is None:
        pytest.skip(msg="Skip hardware test since ANDROID_SERIAL_NUMBER is not set.")

    weight1_data = np.random.rand(w1_shape[0], w1_shape[1], w1_shape[2], w1_shape[3]).astype(
        dtype=dtype
    )
    weight2_data = np.random.rand(w2_shape[0], w2_shape[1], w2_shape[2], w2_shape[3]).astype(
        dtype=dtype
    )
    input_data = np.random.rand(
        input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    ).astype(dtype=dtype)

    params = {"weight1": weight1_data, "weight2": weight2_data}
    inputs = {"data": input_data}

    graph_mod = hexagon_session.get_executor_from_factory(lowered)
    graph_mod.set_input(**params)
    graph_mod.run(**inputs)
    hexagon_output = graph_mod.get_output(0).numpy()

    target_llvm = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            runtime=runtime,
            executor=executor,
        )
    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    llvm_graph_mod.set_input(**params)
    llvm_graph_mod.run(**inputs)
    expected_output = llvm_graph_mod.get_output(0).numpy()

    tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-4, atol=1e-5)

def create_args(func):
    args = {
        k: tvm.nd.array(np.ones([x.value for x in v.shape]).astype(v.dtype))
        for k, v in func.preflattened_buffer_map.items()
    }
    return [args[p] for p in func.params]

def create_args_hexagon(func):
    args = {
        k: tvm.nd.array(np.ones([x.value for x in v.shape]).astype(v.dtype))
        for k, v in func.preflattened_buffer_map.items()
    }
    return args, func.params

repeat = tvm.testing.parameter(0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,)
@requires_hexagon_toolchain
def test_graph_executor_debug(hexagon_session):
    with open("/home/mhessar/work/tvm/hexagon_output/relay_1.log", "r") as f:
        RELAY_MODEL = f.read()
    dtype = "float32"
    input_shape = (1, 3, 224, 224)
    low_range = 0.01
    high_range = 0.05

    # w1_shape = (32, 3, 3, 3)
    # bias1_shape = (32,)
    # data = relay.var("data", relay.TensorType(input_shape, dtype))
    # weight1 = relay.var("weight1", relay.TensorType(w1_shape, dtype))
    # bias1 = relay.var("bias1", relay.TensorType(bias1_shape, dtype))
    # y1 = relay.nn.conv2d(
    #     data,
    #     weight1,
    #     padding=(1, 1),
    #     kernel_size=(3, 3),
    #     strides=(2, 2),
    #     data_layout="NCHW",
    #     kernel_layout="OIHW",
    #     out_dtype="float32",
    # )
    # b1 = relay.nn.bias_add(y1, bias1)
    # c1 = relay.clip(b1, 0.0, 6.0)

    # w2_shape = (32, 32, 3, 3)
    # bias2_shape = (32,)
    # weight2 = relay.var("weight2", relay.TensorType(w2_shape, dtype))
    # bias2 = relay.var("bias2", relay.TensorType(bias2_shape, dtype))
    # y2 = relay.nn.conv2d(
    #     c1,
    #     weight2,
    #     padding=(1, 1),
    #     kernel_size=(3, 3),
    #     strides=(1, 1),
    #     data_layout="NCHW",
    #     kernel_layout="OIHW",
    #     out_dtype="float32",
    # )
    # b2 = relay.nn.bias_add(y2, bias2)
    # c2 = relay.clip(b2, 0.0, 6.0)

    # f = relay.Function([data, weight1, bias1, weight2, bias2], c2)
    # relay_mod = tvm.IRModule.from_expr(f)
    # relay_mod = relay.transform.InferType()(relay_mod)
    
    # weight1_data = np.random.uniform(low=low_range, high=high_range, size=w1_shape).astype(
    #     dtype=dtype
    # )
    # weight2_data = np.random.uniform(low=low_range, high=high_range, size=w2_shape).astype(
    #     dtype=dtype
    # )
    # bias1_data = np.random.uniform(low=low_range, high=high_range, size=bias1_shape).astype(
    #     dtype=dtype
    # )
    # bias2_data = np.random.uniform(low=low_range, high=high_range, size=bias1_shape).astype(
    #     dtype=dtype
    # )
    # params = {"weight1": weight1_data, "weight2": weight2_data, "bias1": bias1_data, "bias2": bias2_data}
    # params = {}

    relay_mod = tvm.parser.fromtext(RELAY_MODEL)
    import pdb; pdb.set_trace()

    target_hexagon = tvm.target.hexagon("v68")
    target_llvm = tvm.target.Target("llvm")
    runtime = Runtime("cpp")
    executor = Executor("graph", {"link-params": True})

    input_data = np.random.uniform(low=low_range, high=high_range, size=input_shape).astype(dtype=dtype)
    # inputs = {"data": input_data}
    inputs = {"input": input_data}

    # with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
    #     comp = tvm.relay.vm.VMCompiler()
    #     omod, oparams = comp.optimize(relay_mod, target=target_hexagon)

    #     for f in omod.functions.values():
    #         logging.debug(f)
    #         if "global_symbol" in f.attrs.keys():
    #             logging.debug(f.attrs["global_symbol"])
    #             hexagon_func = tvm.build(f, target=tvm.target.Target(target_hexagon, host=target_hexagon), name=f.attrs["global_symbol"])
    #             llvm_func = tvm.build(f, target=target_llvm)
                
    #             # get parameters and load to device for hexagon
    #             hexagon_args, hexagon_func_params = create_args_hexagon(f)
    #             hexagon_mod = hexagon_session.load_module(hexagon_func)
    #             hexagon_func_call_args = []
    #             for item in hexagon_func_params:
    #                 item_numpy = hexagon_args[item].asnumpy()
    #                 item_data = tvm.nd.array(item_numpy, device=hexagon_session.device)
    #                 assert (item_data.numpy() == item_numpy).all()
    #                 hexagon_func_call_args.append(item_data)

    #             # run on hexagon
    #             hexagon_mod[f.attrs["global_symbol"]](hexagon_func_call_args[0], hexagon_func_call_args[1], hexagon_func_call_args[2])
                
    #             # get args and run on LLVM
    #             llvm_args = create_args(f)
    #             llvm_func(*llvm_args)
    #             import pdb; pdb.set_trace()
    #             for a1, a2 in zip(hexagon_args, llvm_args):
    #                 assert a1 == a2
    #             print("Ran!")
        
    with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
        hexagon_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_hexagon, host=target_hexagon),
            runtime=runtime,
            executor=executor,
            # params=params,
        )

        llvm_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            runtime=runtime,
            executor=executor,
            # params=params,
        )

    graph_mod = hexagon_session.get_executor_from_factory(hexagon_lowered)
    graph_mod.set_input(**inputs)
    graph_mod.run()
    hexagon_output = graph_mod.get_output(0).numpy()

    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    llvm_graph_mod.set_input(**inputs)
    llvm_graph_mod.run()
    expected_output = llvm_graph_mod.get_output(0).numpy()

    # import pdb; pdb.set_trace()
    logging.debug(hexagon_output)
    logging.debug(expected_output)
    import pdb; pdb.set_trace()
    tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-6, atol=1e-5)

def _workaround_create_aot_shared():
    # The C codegen uses TVM/RT functions directly. On Hexagon it should use
    # functions pointers via __TVMxyz variables. This workaround makes the
    # runtime symbols visible to the compiled shared library.
    extra_link_flags = os.environ.get("HEXAGON_SHARED_LINK_FLAGS")
    extra_options = str(extra_link_flags).split() if extra_link_flags else []
    return lambda so_name, files, hexagon_arch, options: hexagon.create_aot_shared(
        so_name, files, hexagon_arch, options=extra_options + options
    )


@requires_hexagon_toolchain
def test_aot_executor(hexagon_session):
    dtype = "float32"
    input_shape = (1, 128, 128, 3)
    w_shape = (5, 5, 3, 8)
    data = relay.var("data", relay.TensorType(input_shape, dtype))
    weight = relay.var("weight", relay.TensorType(w_shape, dtype))
    y = relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    f = relay.Function([data, weight], y)
    relay_mod = tvm.IRModule.from_expr(f)
    relay_mod = relay.transform.InferType()(relay_mod)

    target_hexagon = tvm.target.hexagon("v68")

    weight_data = np.random.rand(w_shape[0], w_shape[1], w_shape[2], w_shape[3]).astype(dtype=dtype)
    input_data = np.random.rand(
        input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    ).astype(dtype=dtype)

    params = {"weight": weight_data}
    inputs = {"data": input_data}

    with tvm.transform.PassContext(opt_level=3):
        lowered = tvm.relay.build(
            relay_mod,
            params=params,
            target=tvm.target.Target(target_hexagon, host="c"),
            runtime=Runtime("cpp"),
            executor=Executor("aot", {"unpacked-api": False, "interface-api": "c"}),
        )

    if hexagon_session is None:
        pytest.skip(msg="Skip hardware test, ANDROID_SERIAL_NUMBER is not set.")

    aot_mod = hexagon_session.get_executor_from_factory(lowered)
    aot_mod.set_input(**inputs)
    aot_mod.run()
    hexagon_output = aot_mod.get_output(0).numpy()

    target_llvm = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            runtime=Runtime("cpp"),
            executor=Executor("graph"),
        )

    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    llvm_graph_mod.set_input(**params)
    llvm_graph_mod.run(**inputs)
    expected_output = llvm_graph_mod.get_output(0).numpy()

    tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-4, atol=1e-5)


@requires_hexagon_toolchain
def test_aot_executor_multiple_conv2d(hexagon_session):
    dtype = "float32"
    input_shape = (1, 8, 8, 3)
    w1_shape = (5, 5, 3, 1)
    w2_shape = (5, 5, 1, 3)
    data = relay.var("data", relay.TensorType(input_shape, dtype))
    weight1 = relay.var("weight1", relay.TensorType(w1_shape, dtype))
    weight2 = relay.var("weight2", relay.TensorType(w2_shape, dtype))
    y1 = relay.nn.conv2d(
        data,
        weight1,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    y2 = relay.nn.conv2d(
        y1,
        weight2,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    f = relay.Function([data, weight1, weight2], y2)
    relay_mod = tvm.IRModule.from_expr(f)
    relay_mod = relay.transform.InferType()(relay_mod)

    target_hexagon = tvm.target.hexagon("v68")

    weight1_data = np.random.rand(w1_shape[0], w1_shape[1], w1_shape[2], w1_shape[3]).astype(
        dtype=dtype
    )
    weight2_data = np.random.rand(w2_shape[0], w2_shape[1], w2_shape[2], w2_shape[3]).astype(
        dtype=dtype
    )
    input_data = np.random.rand(
        input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    ).astype(dtype=dtype)

    params = {"weight1": weight1_data, "weight2": weight2_data}
    inputs = {"data": input_data}

    with tvm.transform.PassContext(opt_level=3):
        lowered = tvm.relay.build(
            relay_mod,
            params=params,
            target=tvm.target.Target(target_hexagon, host="c"),
            runtime=Runtime("cpp"),
            executor=Executor("aot", {"unpacked-api": False, "interface-api": "c"}),
        )

    if hexagon_session is None:
        pytest.skip(msg="Skip hardware test, ANDROID_SERIAL_NUMBER is not set.")

    aot_mod = hexagon_session.get_executor_from_factory(lowered)
    aot_mod.set_input(**inputs)
    aot_mod.run()
    hexagon_output = aot_mod.get_output(0).numpy()

    target_llvm = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            runtime=Runtime("cpp"),
            executor=Executor("graph"),
        )

    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    llvm_graph_mod.set_input(**params)
    llvm_graph_mod.run(**inputs)
    expected_output = llvm_graph_mod.get_output(0).numpy()

    tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-4, atol=1e-5)

@requires_hexagon_toolchain
def test_mnist(hexagon_launcher, hexagon_session):
    dtype = "float32"
    model_url = (
        "https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-8.onnx"
    )
    model_path = tvm.contrib.download.download_testdata(model_url, "mnist-8.onnx", module="onnx")
    onnx_model = onnx.load(model_path)

    target_hexagon = tvm.target.hexagon("v68")
    runtime = Runtime("cpp")
    executor = Executor("graph", {"link-params": True})

    data_in = np.random.rand(1, 1, 28, 28).astype(dtype=dtype)
    
    input_name = "Input3"
    shape_dict = {input_name: data_in.shape}
    relay_mod, params= relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=False)
    inputs = {input_name: data_in}

    temp = utils.tempdir()
    dso_binary = "test_binary.so"
    dso_binary_path = temp.relpath(dso_binary)

    with tvm.transform.PassContext(opt_level=3):
        lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_hexagon, host=target_hexagon),
            runtime=runtime,
            executor=executor,
            params=params,
        )
        lowered.get_lib().save(dso_binary_path)

    if hexagon_session is None:
        pytest.skip(msg="Skip hardware test since ANDROID_SERIAL_NUMBER is not set.")

    hexagon_launcher.upload(dso_binary_path, dso_binary)

    hexagon_mod = hexagon_launcher.get_graph_executor(
        lowered.get_graph_json(), dso_binary, hexagon_session
    )
    hexagon_mod.set_input(**inputs)
    hexagon_mod.run()
    hexagon_output = hexagon_mod.get_output(0).numpy()

    target_llvm = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            runtime=runtime,
            executor=executor,
            params=params,
        )
    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    llvm_graph_mod.set_input(**inputs)
    import pdb; pdb.set_trace()
    llvm_graph_mod.run()
    expected_output = llvm_graph_mod.get_output(0).numpy()

    tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-4, atol=1e-5)

@requires_hexagon_toolchain
def test_mobilenet(hexagon_launcher, hexagon_session):
    dtype = "float32"
    model_url = (
        "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
    )
    model_path = tvm.contrib.download.download_testdata(model_url, "mobilenetv2-7.onnx", module="onnx")
    onnx_model = onnx.load(model_path)

    target_hexagon = tvm.target.hexagon("v68")
    runtime = Runtime("cpp")
    executor = Executor("graph", {"link-params": True})

    data_in = np.random.rand(1, 3, 224, 224).astype(dtype=dtype)
    
    input_name = "input"
    shape_dict = {input_name: data_in.shape}
    relay_mod, params= relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    import pdb; pdb.set_trace()
    inputs = {input_name: data_in}

    temp = utils.tempdir()
    dso_binary = "test_binary.so"
    dso_binary_path = temp.relpath(dso_binary)

    # maybe add config={"tir.disable_vectorize": True}
    with tvm.transform.PassContext(opt_level=3):
        lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_hexagon, host=target_hexagon),
            runtime=runtime,
            executor=executor,
            params=params,
        )
        lowered.get_lib().save(dso_binary_path)

    if hexagon_session is None:
        pytest.skip(msg="Skip hardware test since ANDROID_SERIAL_NUMBER is not set.")

    hexagon_launcher.upload(dso_binary_path, dso_binary)

    hexagon_mod = hexagon_launcher.get_graph_executor(
        lowered.get_graph_json(), dso_binary, hexagon_session
    )
    hexagon_mod.set_input(**inputs)
    hexagon_mod.run()
    hexagon_output = hexagon_mod.get_output(0).numpy()

    target_llvm = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            runtime=runtime,
            executor=executor,
            params=params,
        )
    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    llvm_graph_mod.set_input(**inputs)
    llvm_graph_mod.run()
    expected_output = llvm_graph_mod.get_output(0).numpy()

    # file1 = "/home/mhessar/work/tvm/output2.npy"
    # np.save(file1, hexagon_output)
    import pdb; pdb.set_trace()
    tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-4, atol=1e-5)


@requires_hexagon_toolchain
def test_mobilenet_debug(hexagon_launcher, hexagon_session):
    dtype = "float32"
    model_url = (
        "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
    )
    model_path = tvm.contrib.download.download_testdata(model_url, "mobilenetv2-7.onnx", module="onnx")
    onnx_model = onnx.load(model_path)

    target_hexagon = tvm.target.hexagon("v68")
    runtime = Runtime("cpp")
    executor = Executor("graph", {"link-params": True})

    data_in = np.random.rand(1, 3, 224, 224).astype(dtype=dtype)
    
    input_name = "input"
    shape_dict = {input_name: data_in.shape}
    relay_mod, params= relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    
    # with open("/home/mhessar/work/tvm/hexagon_output/params.bin", "wb") as f_params:
    #     f_params.write(tvm.runtime.save_param_dict(params))
    # import pdb; pdb.set_trace()

    # param_base_dir = pathlib.Path("/home/mhessar/work/tvm/hexagon_output/params")
    # for name,val in params.items():
    #     np.save(param_base_dir / name, val.numpy())
    # import pdb; pdb.set_trace()

    inputs = {input_name: data_in}
    with open("/home/mhessar/work/tvm/hexagon_output/relay_mod.log", "w") as f:
        f.write(relay_mod.astext())
    import pdb; pdb.set_trace()

    temp = utils.tempdir()
    dso_binary = "test_binary.so"
    dso_binary_path = temp.relpath(dso_binary)

    with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
        lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_hexagon, host=target_hexagon),
            runtime=runtime,
            executor=executor,
            params=params,
        )
        lowered.get_lib().save(dso_binary_path)
    hexagon_launcher.upload(dso_binary_path, dso_binary)

    base_dir = pathlib.Path("/home/mhessar/work/tvm/hexagon_output/")
    hexagon_debug_mod = hexagon_launcher.get_graph_debug_executor(
        lowered.get_graph_json(), dso_binary, hexagon_session, dump_root= str(base_dir / "debug_output")
    )
    hexagon_graph_json_obj = json.loads(lowered.get_graph_json())
    hexagon_nodes = hexagon_graph_json_obj["nodes"]

    llvm_base_dir = pathlib.Path("/home/mhessar/work/tvm/llvm_output/")
    target_llvm = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
        llvm_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            runtime=runtime,
            executor=executor,
            params=params,
        )
    device = tvm.cpu(0)
    llvm_debug_mod = tvm.contrib.debugger.debug_executor.GraphModuleDebug(
        llvm_lowered["debug_create"]("default", device),
        [device],
        llvm_lowered.get_graph_json(),
        str(llvm_base_dir),
    )
    llvm_graph_json_obj = json.loads(llvm_lowered.get_graph_json())
    llvm_nodes = llvm_graph_json_obj["nodes"]
    with open("/home/mhessar/work/tvm/hexagon_output/graph_llvm.log", "w") as llvm_graph:
        llvm_graph.write(llvm_lowered.get_graph_json())

    graph_debug_json_str = open("/home/mhessar/work/tvm/hexagon_output/_tvmdbg_graph_dump.json", "r")
    graph_debug_json = json.loads(graph_debug_json_str.read())
    debug_nodes = graph_debug_json["nodes"]
    llvm_debug_mod.set_input(**inputs)
    hexagon_debug_mod.set_input(**inputs)
    with open("/home/mhessar/work/tvm/hexagon_output/llvm_funcs.log", "w") as f:
        for node in llvm_nodes:
            if node["op"] != "tvm_op":
                continue
            f.write(f"{node['name']}\n")

            # if node["name"] == "tvmgen_default_fused_nn_conv2d_add_clip":
            #     continue

            for debug_node in debug_nodes:
                if debug_node["name"] == node["name"]:
                    llvm_output_shape = tuple(debug_node["shape"])

            out_sample = np.zeros(shape=llvm_output_shape).astype(dtype)
            logging.debug(f'trying name: {node["name"]}, out_shape: {out_sample.shape}')
            
            import pdb; pdb.set_trace()

            llvm_output = tvm.nd.array(out_sample, device=device)
            hexagon_output = tvm.nd.array(out_sample, device=hexagon_session.device)
            
            llvm_debug_mod.debug_get_output(node["name"], llvm_output)
            hexagon_debug_mod.debug_get_output(node["name"], hexagon_output)
            try:
                logging.debug(f'name: {node["name"]}')
                tvm.testing.assert_allclose(hexagon_output.numpy(), llvm_output.numpy(), rtol=1e-4, atol=1e-5)
            except:
                import pdb; pdb.set_trace()
                print(f'name: {node["name"]}')
            del(llvm_output)
            del(hexagon_output)

    # llvm_debug_mod.run(**inputs)

    fund_ind = 0
    with open("/home/mhessar/work/tvm/hexagon_output/hexagon_funcs.log", "w") as f:
        for node in hexagon_nodes:
            if node["op"] != "tvm_op":
                continue
            fund_ind += 1
            f.write(f"{node['name']}\n")
            hexagon_output = hexagon_debug_mod.run_individual(fund_ind)
            llvm_output = llvm_debug_mod.run_individual(fund_ind)
            import pdb; pdb.set_trace()
            tvm.testing.assert_allclose(hexagon_output, llvm_output, rtol=1e-4, atol=1e-5)
    # hexagon_debug_mod.run(**inputs)

    # hexagon_debug_mod.set_input(**inputs)
    # dtype = "float32"
    # for node in hexagon_nodes:
    #     if node["op"] != "tvm_op":
    #         continue
    #     import pdb; pdb.set_trace()
    #   hexagon_output = tvm.nd.array(, dtype=dtype), device=hexagon_session.device)
    #     llvm_output = None
    #     hexagon_debug_mod.debug_get_output(node["name"], hexagon_output)
    #     # llvm_debug_mod.debug_get_output(node["name"], llvm_output)
    #     import pdb; pdb.set_trace()
    #     print("mehrdad")

    
    # import pdb; pdb.set_trace()
    # hexagon_mod.set_input(**params)
    # import pdb; pdb.set_trace()
    # hexagon_mod.run(**inputs)
    # import pdb; pdb.set_trace()
    # hexagon_output = hexagon_mod.get_output(0).numpy()

    # target_llvm = tvm.target.Target("llvm")
    # with tvm.transform.PassContext(opt_level=3):
    #     llvm_lowered = tvm.relay.build(
    #         relay_mod,
    #         tvm.target.Target(target_llvm, host=target_llvm),
    #         runtime=runtime,
    #         executor=executor,
    #     )
    # llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    # llvm_graph_mod.set_input(**params)
    # llvm_graph_mod.run(**inputs)
    # expected_output = llvm_graph_mod.get_output(0).numpy()

    # tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-4, atol=1e-5)

if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
