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
# pylint: disable=line-too-long,missing-class-docstring,missing-module-docstring,missing-function-docstring,no-self-argument,unused-argument,invalid-name
from __future__ import annotations  # must import to defer parsing of annotations

import tvm
import tvm.testing
from tvm.ir import assert_structural_equal
import tvm.script
from tvm.script import tir as T, relax as R
from tvm.relax.backend.aot import AOTLowerMain


def _assert_lowered_main(mod, main_func, print_script=False):
    host_target = tvm.target.Target("llvm")
    prim_target = tvm.target.Target("llvm", host=host_target)
    ctxt = tvm.transform.PassContext()
    config = tvm.target.make_compilation_config(ctxt, prim_target)
    mod = AOTLowerMain("test_mod", config)(mod)
    if print_script:
        print(mod["__tvm_main__"].script())

    assert_structural_equal(mod["__tvm_main__"], main_func)


def test_single_call():
    @tvm.script.ir_module
    class SingleCall:
        @R.function
        def main(a: R.Tensor((5, 7), "float32"), output: R.Tensor((5, 7), "float32")):
            R.func_attr({"input_vars": [a], "output_vars": [output]})
            alloc = output
            _ = R.call_packed("identity", a, alloc, type_args=R.Tensor(ndim=2, dtype="float32"))
            output_1 = alloc
            return ()

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "input_vars": [a], "output_vars": [output]})
        a_buffer = T.match_buffer(a, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        # body
        T.evaluate(T.tvm_call_cpacked("identity", a_buffer.data, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    _assert_lowered_main(SingleCall, func)


def test_constant():
    @tvm.script.ir_module
    class Constant:
        @R.function
        def main(a: R.Tensor((2, 2), "float32"), output: R.Tensor((2, 2), "float32")):
            R.func_attr({"input_vars": [a], "output_vars": [output]})
            const = R.const([[1, 2], [3, 4]], dtype="float32")
            alloc = output
            _ = R.call_packed("add", a, const, alloc, type_args=R.Tensor(ndim=2, dtype="int32"))
            output_1 = alloc
            return ()

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "input_vars": [a], "output_vars": [output]})
        a_buffer = T.match_buffer(a, [T.int64(2), T.int64(2)], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [T.int64(2), T.int64(2)], dtype="float32", align=16)
        # body
        constant_0 = T.allocate_const([1, 2, 3, 4], "float32", [2, 2])
        T.evaluate(T.tvm_call_cpacked("add", a_buffer.data, constant_0, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    _assert_lowered_main(Constant, func)


def test_two_calls():
    @tvm.script.ir_module
    class TwoCalls:
        @R.function
        def main(a: R.Tensor((5, 7), "float32"), output: R.Tensor((5, 7), "float32")):
            R.func_attr({"input_vars": [a], "output_vars": [output]})
            alloc_0 = R.memory.alloc_storage(140, 0, "global", "uint8", "global.workspace")
            tid_0 = R.memory.alloc_tensor(alloc_0, (5, 7), offset=0, dtype="float32")
            _ = R.call_packed("identity", a, tid_0, type_args=R.Tensor(ndim=2, dtype="float32"))
            tid_1 = output
            _ = R.call_packed("identity", tid_0, tid_1, type_args=R.Tensor(ndim=2, dtype="float32"))
            return ()

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output: T.handle):
        # function attr dict
        T.func_attr({"runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "global_symbol": "test_mod___tvm_main__", "input_vars": [a], "output_vars": [output]})
        a_buffer = T.match_buffer(a, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        # body
        sid_0 = T.allocate([140], "uint8", "global.workspace")
        sid_0_1 = T.buffer_decl([140], dtype="uint8", data=sid_0, strides=[1], scope="global.workspace", align=16)
        tid_0: T.Ptr[T.float32, "global.workspace"] = T.address_of(sid_0_1[0], dtype="handle")
        T.tvm_call_cpacked("identity", a_buffer.data, tid_0, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32")
        T.tvm_call_cpacked("identity", tid_0, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32")
    # fmt: on

    _assert_lowered_main(TwoCalls, func)


def test_multi_input():
    @tvm.script.ir_module
    class MultiInput:
        @R.function
        def main(
            a: R.Tensor((5, 7), "float32"),
            b: R.Tensor((5, 7), "float32"),
            output: R.Tensor((5, 7), "float32"),
        ):
            R.func_attr({"input_vars": [a, b], "output_vars": [output]})
            tid_0 = output
            _ = R.call_packed("add", a, b, tid_0, type_args=R.Tensor(ndim=2, dtype="float32"))
            return ()

    # fmt: off
    @T.prim_func
    def func(a: T.handle, b: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "input_vars": [a, b], "output_vars": [output]})
        a_buffer = T.match_buffer(a, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        b_buffer = T.match_buffer(b, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        # body
        T.evaluate(T.tvm_call_cpacked("add", a_buffer.data, b_buffer.data, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    _assert_lowered_main(MultiInput, func)


def test_multi_output():
    @tvm.script.ir_module
    class MultiOutput:
        @R.function
        def main(
            a: R.Tensor((5, 7), "float32"),
            output_0: R.Tensor((5, 7), "float32"),
            output_1: R.Tensor((5, 7), "float32"),
        ):
            R.func_attr({"input_vars": [a], "output_vars": [output_0, output_1]})
            tid_0 = output_0
            tid_1 = output_1
            _ = R.call_packed(
                "duplicate", a, tid_0, tid_1, type_args=R.Tensor(ndim=2, dtype="float32")
            )
            return ()

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output_0: T.handle, output_1: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "input_vars": [a], "output_vars": [output_0, output_1]})
        a_buffer = T.match_buffer(a, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        output_0_buffer = T.match_buffer(output_0, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        output_1_buffer = T.match_buffer(output_1, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        # body
        T.evaluate(T.tvm_call_cpacked("duplicate", a_buffer.data, output_0_buffer.data, output_1_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    _assert_lowered_main(MultiOutput, func)


def test_tuple():
    @tvm.script.ir_module
    class Tuple:
        @R.function
        def main(a: R.Tensor((5, 7), "float32"), output: R.Tensor((5, 7), "float32")):
            R.func_attr({"input_vars": [a], "output_vars": [output]})
            tup = (a, a)
            _ = R.call_packed("add", tup, output, type_args=R.Tensor(ndim=2, dtype="float32"))
            return ()

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "input_vars": [a], "output_vars": [output]})
        a_buffer = T.match_buffer(a, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        # body
        T.evaluate(T.tvm_call_cpacked("add", a_buffer.data, a_buffer.data, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    _assert_lowered_main(Tuple, func)


def test_tuple_get_item():
    @tvm.script.ir_module
    class TupleGetItem:
        @R.function
        def main(a: R.Tensor((5, 7), "float32"), output: R.Tensor((5, 7), "float32")):
            R.func_attr({"input_vars": [a], "output_vars": [output]})
            tup = (a, a)
            _ = R.call_packed(
                "identity", tup[1], output, type_args=R.Tensor(ndim=2, dtype="float32")
            )
            return ()

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "input_vars": [a], "output_vars": [output]})
        a_buffer = T.match_buffer(a, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        # body
        T.evaluate(T.tvm_call_cpacked("identity", a_buffer.data, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    _assert_lowered_main(TupleGetItem, func)


def test_branch():
    @tvm.script.ir_module
    class Branch:
        @R.function
        def main(a: R.Tensor((5, 7), "float32"), output: R.Tensor((5, 7), "float32")):
            R.func_attr({"input_vars": [a], "output_vars": [output]})
            alloc_0 = R.memory.alloc_storage(140, 0, "global", "uint8", "global.workspace")
            tid_0 = R.memory.alloc_tensor(alloc_0, (5, 7), offset=0, dtype="float32")
            _ = R.call_packed("identity", a, tid_0, type_args=R.Tensor(ndim=2, dtype="float32"))
            alloc_1 = R.memory.alloc_storage(140, 0, "global", "uint8", "global.workspace")
            tid_1 = R.memory.alloc_tensor(alloc_1, (5, 7), offset=0, dtype="float32")
            _ = R.call_packed("identity", tid_0, tid_1, type_args=R.Tensor(ndim=2, dtype="float32"))
            alloc_2 = R.memory.alloc_storage(140, 0, "global", "uint8", "global.workspace")
            tid_2 = R.memory.alloc_tensor(alloc_2, (5, 7), offset=0, dtype="float32")
            _ = R.call_packed("identity", tid_0, tid_2, type_args=R.Tensor(ndim=2, dtype="float32"))
            tid_3 = output
            _ = R.call_packed(
                "add", tid_1, tid_2, tid_3, type_args=R.Tensor(ndim=2, dtype="float32")
            )
            return ()

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output: T.handle):
        # function attr dict
        T.func_attr({"runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "global_symbol": "test_mod___tvm_main__", "input_vars": [a], "output_vars": [output]})
        a_buffer = T.match_buffer(a, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        # body
        sid_2 = T.allocate([140], "uint8", "global.workspace")
        sid_2_1 = T.buffer_decl([140], dtype="uint8", data=sid_2, strides=[1], scope="global.workspace", align=16)
        sid_1 = T.allocate([140], "uint8", "global.workspace")
        sid_1_1 = T.buffer_decl([140], dtype="uint8", data=sid_1, strides=[1], scope="global.workspace", align=16)
        sid_0 = T.allocate([140], "uint8", "global.workspace")
        sid_0_1 = T.buffer_decl([140], dtype="uint8", data=sid_0, strides=[1], scope="global.workspace", align=16)
        tid_2: T.Ptr[T.float32, "global.workspace"] = T.address_of(sid_2_1[0], dtype="handle")
        tid_1: T.Ptr[T.float32, "global.workspace"] = T.address_of(sid_1_1[0], dtype="handle")
        tid_0: T.Ptr[T.float32, "global.workspace"] = T.address_of(sid_0_1[0], dtype="handle")
        T.tvm_call_cpacked("identity", a_buffer.data, tid_0, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32")
        T.tvm_call_cpacked("identity", tid_0, tid_1, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32")
        T.tvm_call_cpacked("identity", tid_0, tid_2, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32")
        T.tvm_call_cpacked("add", tid_1, tid_2, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32")
    # fmt: on

    _assert_lowered_main(Branch, func)


def test_device_hooks():
    @tvm.script.ir_module
    class DeviceHooks:
        @T.prim_func
        def identity(
            a: T.handle, output: T.handle, device_context_example_target_hook: T.handle
        ) -> None:
            # function attr dict
            T.func_attr(
                {
                    "global_symbol": "identity",
                    "runner_function": True,
                    "target": T.target({"kind": "llvm", "tag": "", "keys": ["cpu"]}),
                }
            )
            a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
            output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
            # body
            T.evaluate(0)

        @R.function
        def main(a: R.Tensor((5, 7), "float32"), output: R.Tensor((5, 7), "float32")):
            R.func_attr({"input_vars": [a], "output_vars": [output]})
            alloc_0 = R.memory.alloc_storage(140, 0, "global", "uint8", "global.workspace")
            tid_0 = R.memory.alloc_tensor(alloc_0, (5, 7), offset=0, dtype="float32")
            _ = R.call_packed("identity", a, tid_0, type_args=R.Tensor(ndim=2, dtype="float32"))
            tid_1 = output
            _ = R.call_packed("identity", tid_0, tid_1, type_args=R.Tensor(ndim=2, dtype="float32"))
            return ()

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output: T.handle, device_context_example_target_hook: T.handle):
        # function attr dict
        T.func_attr({"runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "global_symbol": "test_mod___tvm_main__", "input_vars": [a], "output_vars": [output]})
        a_buffer = T.match_buffer(a, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [T.int64(5), T.int64(7)], dtype="float32", align=16)
        # body
        T.tvm_check_return(0, -1, T.call_extern("TVMDeviceExampleTargetHookActivate", device_context_example_target_hook, dtype="int32"), dtype="int32")
        with T.allocate([140], "uint8", "global.workspace") as sid_0:
            sid_0_1 = T.buffer_decl([140], dtype="uint8", data=sid_0, strides=[1], scope="global.workspace", align=16)
            tid_0: T.Ptr[T.float32, "global.workspace"] = T.address_of(sid_0_1[0], dtype="handle")
            T.tvm_check_return(0, -1, T.call_extern("TVMDeviceExampleTargetHookOpen", device_context_example_target_hook, dtype="int32"), dtype="int32")
            T.tvm_call_cpacked("identity", a_buffer.data, tid_0, device_context_example_target_hook, dtype="int32")
            T.tvm_check_return(0, -1, T.call_extern("TVMDeviceExampleTargetHookClose", device_context_example_target_hook, dtype="int32"), dtype="int32")
            T.tvm_check_return(0, -1, T.call_extern("TVMDeviceExampleTargetHookOpen", device_context_example_target_hook, dtype="int32"), dtype="int32")
            T.tvm_call_cpacked("identity", tid_0, output_buffer.data, device_context_example_target_hook, dtype="int32")
            T.tvm_check_return(0, -1, T.call_extern("TVMDeviceExampleTargetHookClose", device_context_example_target_hook, dtype="int32"), dtype="int32")
        T.tvm_check_return(0, -1, T.call_extern("TVMDeviceExampleTargetHookDeactivate", device_context_example_target_hook, dtype="int32"), dtype="int32")
    # fmt: on

    mod = DeviceHooks
    device_contexts = {}
    for gv in mod.get_global_vars():
        device_contexts[gv] = "example_target_hook"

    mod = mod.with_attr("device_contexts", device_contexts)

    _assert_lowered_main(mod, func)


if __name__ == "__main__":
    tvm.testing.main()
