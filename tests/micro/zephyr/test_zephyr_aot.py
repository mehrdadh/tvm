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
import io
import logging
import os
import sys
import logging
import pathlib
import tarfile
import tempfile

import pytest
import numpy as np

import tvm
import tvm.testing
from tvm.micro.project_api import server
import tvm.relay as relay
from tvm.relay.backend import Executor, Runtime

from tvm.contrib.download import download_testdata
from tvm.micro.testing import aot_transport_init_wait, aot_transport_find_message

try:
    from tvm.relay.op.contrib import cmsisnn
except ImportError:
    pass

import test_utils


@tvm.testing.requires_micro
def test_tflite(temp_dir, board, west_cmd, tvm_debug):
    """Testing a TFLite model."""
    model = test_utils.ZEPHYR_BOARDS[board]
    input_shape = (1, 49, 10, 1)
    output_shape = (1, 12)
    build_config = {"debug": tvm_debug}

    model_url = "https://github.com/tlc-pack/web-data/raw/25fe99fb00329a26bd37d3dca723da94316fd34c/testdata/microTVM/model/keyword_spotting_quant.tflite"
    model_path = download_testdata(model_url, "keyword_spotting_quant.tflite", module="model")

    # Import TFLite model
    tflite_model_buf = open(model_path, "rb").read()
    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    # Load TFLite model and convert to Relay
    relay_mod, params = relay.frontend.from_tflite(
        tflite_model, shape_dict={"input_1": input_shape}, dtype_dict={"input_1 ": "int8"}
    )

    target = tvm.target.target.micro(model)
    executor = Executor(
        "aot", {"unpacked-api": True, "interface-api": "c", "workspace-byte-alignment": 4}
    )
    runtime = Runtime("crt")
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(relay_mod, target, params=params, runtime=runtime, executor=executor)

    sample_url = "https://github.com/tlc-pack/web-data/raw/967fc387dadb272c5a7f8c3461d34c060100dbf1/testdata/microTVM/data/keyword_spotting_int8_6.pyc.npy"
    sample_path = download_testdata(sample_url, "keyword_spotting_int8_6.pyc.npy", module="data")
    sample = np.load(sample_path)

    project, _ = test_utils.generate_project(
        temp_dir,
        board,
        west_cmd,
        lowered,
        build_config,
        sample,
        output_shape,
        "int8",
        load_cmsis=False,
    )

    result, time = test_utils.run_model(project)
    assert result == 6


@tvm.testing.requires_micro
def test_qemu_make_fail(temp_dir, board, west_cmd, tvm_debug):
    """Testing QEMU make fail."""
    if board not in ["qemu_x86", "mps2_an521"]:
        pytest.skip(msg="Only for QEMU targets.")

    model = test_utils.ZEPHYR_BOARDS[board]
    build_config = {"debug": tvm_debug}
    shape = (10,)
    dtype = "float32"

    # Construct Relay program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    xx = relay.multiply(x, x)
    z = relay.add(xx, relay.const(np.ones(shape=shape, dtype=dtype)))
    func = relay.Function([x], z)
    ir_mod = tvm.IRModule.from_expr(func)

    target = tvm.target.target.micro(model)
    executor = Executor("aot")
    runtime = Runtime("crt")
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(ir_mod, target, executor=executor, runtime=runtime)

    sample = np.zeros(shape=shape, dtype=dtype)
    project, project_dir = test_utils.generate_project(
        temp_dir, board, west_cmd, lowered, build_config, sample, shape, dtype, load_cmsis=False
    )

    file_path = (
        pathlib.Path(project_dir) / "build" / "zephyr" / "CMakeFiles" / "run.dir" / "build.make"
    )
    assert file_path.is_file(), f"[{file_path}] does not exist."

    # Remove a file to create make failure.
    os.remove(file_path)
    project.flash()
    with pytest.raises(server.JSONRPCError) as excinfo:
        project.transport().open()
    assert "QEMU setup failed" in str(excinfo.value)

# @tvm.testing.requires_micro
# @tvm.testing.requires_cmsisnn
# def test_cmsis_nn(temp_dir, board, west_cmd, tvm_debug):
#     model = test_utils.ZEPHYR_BOARDS[board]
#     build_config = {"debug": tvm_debug}

#     data_shape = (1, 3, 16, 16)
#     weight_shape = (8, 3, 5, 5)
#     data = relay.var("input_1", relay.TensorType(data_shape, "float32"))
#     weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))
#     y = relay.nn.conv2d(
#         data,
#         weight,
#         padding=(2, 2),
#         kernel_size=(5, 5),
#         kernel_layout="OIHW",
#         out_dtype="float32",
#     )
#     output_shape = (1, 8, 16, 16)

#     f = relay.Function([data, weight], y)
#     mod = tvm.IRModule.from_expr(f)
#     mod = relay.transform.InferType()(mod)

#     weight_sample = np.random.rand(
#         weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]
#     ).astype("float32")
#     params = {mod["main"].params[1].name_hint: weight_sample}
#     cmsisnn_mod = cmsisnn.partition_for_cmsisnn(mod, params)
    
#     target = tvm.target.target.micro(model)
#     executor = Executor(
#         "aot", {"unpacked-api": True, "interface-api": "c", "workspace-byte-alignment": 4}
#     )
#     runtime = Runtime("crt")
#     with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
#         lowered = relay.build(cmsisnn_mod, target, params=params, runtime=runtime, executor=executor)
    
#     data_sample = np.random.rand(data_shape[0], data_shape[1], data_shape[2], data_shape[3]).astype(
#         "float32"
#     )
#     project, _ = test_utils.generate_project(
#         temp_dir,
#         board,
#         west_cmd,
#         lowered,
#         build_config,
#         data_sample,
#         output_shape,
#         "float32",
#         load_cmsis=False,
#     )

@tvm.testing.requires_micro
@tvm.testing.requires_cmsisnn
def test_cmsis_nn(temp_dir, board, west_cmd, tvm_debug):
    model = test_utils.ZEPHYR_BOARDS[board]
    input_shape = (1, 49, 10, 1)
    output_shape = (1, 12)
    build_config = {"debug": tvm_debug}

    model_url = "https://github.com/tlc-pack/web-data/raw/25fe99fb00329a26bd37d3dca723da94316fd34c/testdata/microTVM/model/keyword_spotting_quant.tflite"
    model_path = download_testdata(model_url, "keyword_spotting_quant.tflite", module="model")

    # Import TFLite model
    tflite_model_buf = open(model_path, "rb").read()
    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    # Load TFLite model and convert to Relay
    relay_mod, params = relay.frontend.from_tflite(
        tflite_model, shape_dict={"input_1": input_shape}, dtype_dict={"input_1 ": "int8"}
    )

    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(relay_mod, params)
    
    target = tvm.target.target.micro(model)
    executor = Executor(
        "aot", {"unpacked-api": True, "interface-api": "c", "workspace-byte-alignment": 4}
    )
    runtime = Runtime("crt")
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(cmsisnn_mod, target, params=params, runtime=runtime, executor=executor)
    
    sample_url = "https://github.com/tlc-pack/web-data/raw/967fc387dadb272c5a7f8c3461d34c060100dbf1/testdata/microTVM/data/keyword_spotting_int8_6.pyc.npy"
    sample_path = download_testdata(sample_url, "keyword_spotting_int8_6.pyc.npy", module="data")
    sample = np.load(sample_path)

    project, _ = test_utils.generate_project(
        temp_dir,
        board,
        west_cmd,
        lowered,
        build_config,
        sample,
        output_shape,
        "float32",
        load_cmsis=True,
    )

if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
