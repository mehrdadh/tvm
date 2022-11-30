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
from __future__ import annotations  # must import to defer parsing of annotations
from typing import Optional, Callable

import sys
import pytest
import tvm
import tvm.script
import tvm.testing
from tvm.relax.testing import dump_ast
from tvm import (
    relax,
    rpc,
    te,
    tir,
    topi,
    TVMError,
    cpu,
    WorkspacePoolInfo,
    ConstantPoolInfo,
    PoolInfoProperties,
    IRModule,
)
from tvm.relax import expr_functor, PyExprVisitor, Expr
from tvm.relay.expr import GlobalVar
from tvm.script import relax as R, tir as T


def _check_no_duplicate_calls(mod):
    @relax.expr_functor.visitor
    class RelaxCheckCalls(PyExprVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.conflict_map = {}

        def visit_span(self, span: Span) -> None:
            pass

        def visit_call_(self, op: tvm.relax.Call) -> None:
            call = op
            if isinstance(call.op, (relax.ExternFunc, GlobalVar)):
                if isinstance(call.op, relax.ExternFunc):
                    func_name = str(call.op.global_symbol)
                else:
                    func_name = str(call.op.name_hint)
                assert func_name not in self.conflict_map.keys(), (
                    "Found duplicate call to function " + func_name
                )
                self.conflict_map[func_name] = True

    relax_visitor = RelaxCheckCalls()
    relax_visitor.visit_expr(mod["main"])


# fmt: off
@tvm.script.ir_module
class TestDuplicateGVCall:

    @T.prim_func
    def tvmgen_default_fused_nn_max_pool2d_cast(placeholder_28: T.handle, T_cast_6: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_max_pool2d_cast", "tir.noalias": True})
        placeholder_29 = T.match_buffer(placeholder_28, [802816], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        T_cast_7 = T.match_buffer(T_cast_6, [177], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        tensor_2 = T.decl_buffer([200704], "uint8")
        for ax0_ax1_fused_4 in T.serial(0, 56):
            for ax2_4 in T.serial(0, 56):
                for ax3_init in T.serial(0, 64):
                    tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_init)] = T.uint8(0)
                for rv0_rv1_fused_1, ax3_2 in T.grid(9, 64):
                    tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2)] = T.max(tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2)], T.if_then_else(((((ax0_ax1_fused_4*2) + T.floordiv(rv0_rv1_fused_1, 3)) < 112) and (((ax2_4*2) + T.floormod(rv0_rv1_fused_1, 3)) < 112)), placeholder_29[(((((ax0_ax1_fused_4*14336) + (T.floordiv(rv0_rv1_fused_1, 3)*7168)) + (ax2_4*128)) + (T.floormod(rv0_rv1_fused_1, 3)*64)) + ax3_2)], T.uint8(0), dtype="uint8"))
        for ax0_ax1_fused_5 in T.serial(0, 56):
            for ax2_5, ax3_3 in T.grid(56, 64):
                T_cast_7[(((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3)] = T.cast(tensor_2[(((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3)], "int16")

    @T.prim_func
    def tvmgen_default_fused_cast_subtract(placeholder_2: T.handle, placeholder_3: T.handle, T_subtract: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast_subtract", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder_2, [150528], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_3, [1], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        T_subtract_1 = T.match_buffer(T_subtract, [452], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        for ax0_ax1_fused_1 in T.serial(0, 224):
            for ax2_1, ax3_inner_1 in T.grid(224, 3):
                T_subtract_1[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)] = (T.cast(placeholder_4[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)], "int16") - placeholder_5[0])

    @R.function
    def main(input: R.Tensor((16, 16), "uint8")):
        # block 0
        tsid_10 = R.builtin.alloc_tensor((1, 1), dtype="int16", runtime_device_index=0)
        tsid_11 = R.builtin.alloc_tensor((9408, 1), dtype="int16", runtime_device_index=0)
        alloc = R.builtin.alloc_tensor((301056, 1), dtype="int32", runtime_device_index=0)
        _ = tvmgen_default_fused_cast_subtract(input, tsid_10, alloc)
        lv0 = alloc
        alloc1 = R.builtin.alloc_tensor((802816, 1), dtype="int32", runtime_device_index=0)
        _1 = tvmgen_default_fused_cast_subtract(lv0, tsid_11, alloc1)
        lv1 = alloc1
        alloc2 = R.builtin.alloc_tensor((16, 16), dtype="int32", runtime_device_index=0)
        _2 = R.call_packed("tvmgen_default_fused_nn_max_pool2d_cast", lv1, alloc2, type_args=(R.Tensor(ndim=2, dtype="int32")))
        output = alloc2
        return output
# fmt: on


def test_duplicate_gv_call():
    before_mod = TestDuplicateGVCall
    after_mod = tvm.relax.transform.UniqueTIRFunctions()(before_mod)
    _check_no_duplicate_calls(after_mod)


# fmt: off
@tvm.script.ir_module
class TestDuplicateExternCall:

    @T.prim_func
    def tvmgen_default_fused_nn_max_pool2d_cast(placeholder_28: T.handle, T_cast_6: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_max_pool2d_cast", "tir.noalias": True})
        placeholder_29 = T.match_buffer(placeholder_28, [802816], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        T_cast_7 = T.match_buffer(T_cast_6, [177], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        tensor_2 = T.decl_buffer([200704], "uint8")
        for ax0_ax1_fused_4 in T.serial(0, 56):
            for ax2_4 in T.serial(0, 56):
                for ax3_init in T.serial(0, 64):
                    tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_init)] = T.uint8(0)
                for rv0_rv1_fused_1, ax3_2 in T.grid(9, 64):
                    tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2)] = T.max(tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2)], T.if_then_else(((((ax0_ax1_fused_4*2) + T.floordiv(rv0_rv1_fused_1, 3)) < 112) and (((ax2_4*2) + T.floormod(rv0_rv1_fused_1, 3)) < 112)), placeholder_29[(((((ax0_ax1_fused_4*14336) + (T.floordiv(rv0_rv1_fused_1, 3)*7168)) + (ax2_4*128)) + (T.floormod(rv0_rv1_fused_1, 3)*64)) + ax3_2)], T.uint8(0), dtype="uint8"))
        for ax0_ax1_fused_5 in T.serial(0, 56):
            for ax2_5, ax3_3 in T.grid(56, 64):
                T_cast_7[(((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3)] = T.cast(tensor_2[(((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3)], "int16")

    @T.prim_func
    def tvmgen_default_fused_cast_subtract(placeholder_2: T.handle, placeholder_3: T.handle, T_subtract: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast_subtract", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder_2, [150528], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_3, [1], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        T_subtract_1 = T.match_buffer(T_subtract, [452], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        for ax0_ax1_fused_1 in T.serial(0, 224):
            for ax2_1, ax3_inner_1 in T.grid(224, 3):
                T_subtract_1[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)] = (T.cast(placeholder_4[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)], "int16") - placeholder_5[0])

    @R.function
    def main(input: R.Tensor((16, 16), "uint8")):
        # block 0
        tsid_10 = R.builtin.alloc_tensor((1, 1), dtype="int16", runtime_device_index=0)
        tsid_11 = R.builtin.alloc_tensor((9408, 1), dtype="int16", runtime_device_index=0)
        alloc = R.builtin.alloc_tensor((301056, 1), dtype="int32", runtime_device_index=0)
        _ = R.call_packed("tvmgen_default_fused_cast_subtract", input, tsid_10, alloc, type_args=(R.Tensor(ndim=2, dtype="int32")))
        lv0 = alloc
        alloc1 = R.builtin.alloc_tensor((802816, 1), dtype="int32", runtime_device_index=0)
        _1 = R.call_packed("tvmgen_default_fused_cast_subtract", lv0, tsid_11, alloc1, type_args=(R.Tensor(ndim=2, dtype="int32")))
        lv1 = alloc1
        alloc2 = R.builtin.alloc_tensor((16, 16), dtype="int32", runtime_device_index=0)
        _2 = R.call_packed("tvmgen_default_fused_nn_max_pool2d_cast", lv1, alloc2, type_args=(R.Tensor(ndim=2, dtype="int32")))
        output = alloc2
        return output
# fmt: on


def test_duplicate_extern_call():
    before_mod = TestDuplicateExternCall
    after_mod = tvm.relax.transform.UniqueTIRFunctions()(before_mod)
    _check_no_duplicate_calls(after_mod)


# fmt: off
@tvm.script.ir_module
class TestDuplicateMixedCalls:

    @T.prim_func
    def tvmgen_default_fused_nn_max_pool2d_cast(placeholder_28: T.handle, T_cast_6: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_max_pool2d_cast", "tir.noalias": True})
        placeholder_29 = T.match_buffer(placeholder_28, [802816], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        T_cast_7 = T.match_buffer(T_cast_6, [177], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        tensor_2 = T.decl_buffer([200704], "uint8")
        for ax0_ax1_fused_4 in T.serial(0, 56):
            for ax2_4 in T.serial(0, 56):
                for ax3_init in T.serial(0, 64):
                    tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_init)] = T.uint8(0)
                for rv0_rv1_fused_1, ax3_2 in T.grid(9, 64):
                    tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2)] = T.max(tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2)], T.if_then_else(((((ax0_ax1_fused_4*2) + T.floordiv(rv0_rv1_fused_1, 3)) < 112) and (((ax2_4*2) + T.floormod(rv0_rv1_fused_1, 3)) < 112)), placeholder_29[(((((ax0_ax1_fused_4*14336) + (T.floordiv(rv0_rv1_fused_1, 3)*7168)) + (ax2_4*128)) + (T.floormod(rv0_rv1_fused_1, 3)*64)) + ax3_2)], T.uint8(0), dtype="uint8"))
        for ax0_ax1_fused_5 in T.serial(0, 56):
            for ax2_5, ax3_3 in T.grid(56, 64):
                T_cast_7[(((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3)] = T.cast(tensor_2[(((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3)], "int16")

    @T.prim_func
    def tvmgen_default_fused_cast_subtract(placeholder_2: T.handle, placeholder_3: T.handle, T_subtract: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast_subtract", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder_2, [150528], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_3, [1], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        T_subtract_1 = T.match_buffer(T_subtract, [452], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        for ax0_ax1_fused_1 in T.serial(0, 224):
            for ax2_1, ax3_inner_1 in T.grid(224, 3):
                T_subtract_1[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)] = (T.cast(placeholder_4[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)], "int16") - placeholder_5[0])

    @R.function
    def main(input: R.Tensor((16, 16), "uint8")):
        # block 0
        tsid_10 = R.builtin.alloc_tensor((1, 1), dtype="int16", runtime_device_index=0)
        tsid_11 = R.builtin.alloc_tensor((9408, 1), dtype="int16", runtime_device_index=0)
        alloc = R.builtin.alloc_tensor((301056, 1), dtype="int32", runtime_device_index=0)
        _ = R.call_packed("tvmgen_default_fused_cast_subtract", input, tsid_10, alloc, type_args=(R.Tensor(ndim=2, dtype="int32")))
        lv0 = alloc
        alloc1 = R.builtin.alloc_tensor((802816, 1), dtype="int32", runtime_device_index=0)
        _1 = R.call_packed("tvmgen_default_fused_cast_subtract", lv0, tsid_11, alloc1, type_args=(R.Tensor(ndim=2, dtype="int32")))
        lv1 = alloc1
        alloc2 = R.builtin.alloc_tensor((16, 16), dtype="int32", runtime_device_index=0)
        _2 = R.call_packed("tvmgen_default_fused_nn_max_pool2d_cast", lv1, alloc2, type_args=(R.Tensor(ndim=2, dtype="int32")))
        _3 = tvmgen_default_fused_cast_subtract(lv0, tsid_11, alloc1)
        _4 = tvmgen_default_fused_nn_max_pool2d_cast(lv0, alloc1)
        _5 = R.call_packed("tvmgen_default_fused_nn_max_pool2d_cast", lv0, tsid_11, type_args=(R.Tensor(ndim=2, dtype="int32")))
        output = alloc2
        return output
# fmt: on


def test_mixed_calls():
    before_mod = TestDuplicateMixedCalls
    after_mod = tvm.relax.transform.UniqueTIRFunctions()(before_mod)
    _check_no_duplicate_calls(after_mod)


if __name__ == "__main__":
    pytest.main([__file__] + sys.argv[1:])
