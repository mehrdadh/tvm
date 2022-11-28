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
from tvm.ir import Span
from tvm.relax import expr_functor, PyExprVisitor, PyExprMutator, Expr
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
from tvm.script import relax as R, tir as T
from tvm.target import Target


def _assign_poolinfos_to_allocates_in_primfuncs(func, pool_infos, constant_pool_infos):
    """helper to assing poolinfos to allocate nodes in a tir.PrimFunc"""

    def set_poolinfos(stmt):
        if isinstance(stmt, tvm.tir.Allocate):
            return tvm.tir.Allocate(
                buffer_var=stmt.buffer_var,
                dtype=stmt.dtype,
                extents=stmt.extents,
                condition=stmt.condition,
                body=stmt.body,
                annotations={tvm.tir.usmp.utils.CANDIDATE_MEMORY_POOL_ATTR: pool_infos},
            )
        elif isinstance(stmt, tvm.tir.AllocateConst):
            return tvm.tir.AllocateConst(
                buffer_var=stmt.buffer_var,
                dtype=stmt.dtype,
                extents=stmt.extents,
                data_or_idx=stmt.data,
                body=stmt.body,
                annotations={tvm.tir.usmp.utils.CANDIDATE_MEMORY_POOL_ATTR: constant_pool_infos},
            )

    return func.with_body(tvm.tir.stmt_functor.ir_transform(func.body, None, set_poolinfos))


def _append_type_args(mod, dtypes):
    # Can not express relax.DynTensorType in tvm script and get structural equal to work.
    # Manually add the right type args to call packed functions in the expected module.
    @relax.expr_functor.mutator
    class RelaxAddTypeArgs(PyExprMutator):
        def __init__(self, mod: Optional[IRModule] = None) -> None:
            super().__init__(mod)
            self.index = 0

        def visit_span(self, span: Span) -> Span:
            pass

        def visit_call_(self, op: tvm.relax.Call) -> Expr:
            call = op
            if isinstance(call.op, relax.ExternFunc):
                type_args = [relax.DynTensorType(ndim=2, dtype=dtypes[self.index])]
                self.index += 1
                return tvm.relax.Call(call.op, call.args, call.attrs, type_args, call.span)
            return super().visit_call_(op)

    relax_visitor = RelaxAddTypeArgs()
    mod["main"] = relax_visitor.visit_expr(mod["main"])
    return mod


def _assign_poolinfos_to_allocates_in_irmodule(mod, pool_infos, constant_pool_infos=None):
    """helper to assign poolinfos to allocate nodes in a IRModule"""

    @relax.expr_functor.mutator
    class RelaxFuncAnnotate(PyExprMutator):
        def visit_span(self, span: Span) -> Span:
            pass

        def visit_call_(self, op: tvm.relax.Call) -> Expr:
            call = op
            if "relax.builtin.alloc_tensor" == str(call.op):
                attrs = tvm.ir.attrs.make_node(
                    "relax.attrs.AllocTensorAttrs",
                    dtype=call.attrs["dtype"],
                    runtime_device_index=call.attrs["runtime_device_index"],
                    candidate_memory_pools=pool_infos,
                )
                return tvm.relax.Call(call.op, call.args, attrs, call.type_args, call.span)
            return super().visit_call_(op)

    relax_visitor = RelaxFuncAnnotate()
    mod["main"] = relax_visitor.visit_expr(mod["main"])

    ret = tvm.IRModule()
    for global_var, basefunc in mod.functions.items():
        if isinstance(basefunc, tvm.tir.PrimFunc):
            ret[global_var] = _assign_poolinfos_to_allocates_in_primfuncs(
                basefunc, pool_infos, constant_pool_infos
            )
        else:
            ret[global_var] = basefunc
    return ret


def _assign_targets_to_relaxfuncs_irmodule(mod, target):
    """helper to assign target for PrimFunc in a IRModule"""
    ret = tvm.IRModule()
    for global_var, basefunc in mod.functions.items():
        if isinstance(basefunc, (tvm.relax.Function, tvm.tir.PrimFunc)):
            ret[global_var] = basefunc.with_attr("target", target)
    return ret


# fmt: off
@tvm.script.ir_module
class LinearStructure:
    @T.prim_func
    def tvmgen_default_fused_cast_subtract(placeholder_2: T.handle, placeholder_3: T.handle, T_subtract: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast_subtract", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder_2, [150528], dtype="uint8", elem_offset=0, align=64, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_3, [1], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        T_subtract_1 = T.match_buffer(T_subtract, [452], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        # body
        for ax0_ax1_fused_1 in T.serial(0, 224):
            for ax2_1, ax3_inner_1 in T.grid(224, 3):
                T_subtract_1[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)] = (T.cast(placeholder_4[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)], "int16") - placeholder_5[0])

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast(placeholder_62: T.handle, placeholder_63: T.handle, placeholder_64: T.handle, T_cast_20: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast", "tir.noalias": True})
        placeholder_65 = T.match_buffer(placeholder_62, [150528], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        placeholder_66 = T.match_buffer(placeholder_63, [9408], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        placeholder_67 = T.match_buffer(placeholder_64, [64], dtype="int32", elem_offset=0, align=64, offset_factor=1)
        T_cast_21 = T.match_buffer(T_cast_20, [289], dtype="uint8", elem_offset=0, align=64, offset_factor=1)
        # body
        PaddedInput_7_data = T.allocate([157323], "int16", "global")
        PaddedInput_7 = T.buffer_decl(shape=[157323], dtype="int16", data=PaddedInput_7_data)
        for i0_i1_fused_7 in T.serial(0, 229):
            for i2_7, i3_7 in T.grid(229, 3):
                PaddedInput_7[(((i0_i1_fused_7*687) + (i2_7*3)) + i3_7)] = T.if_then_else(((((2 <= i0_i1_fused_7) and (i0_i1_fused_7 < 226)) and (2 <= i2_7)) and (i2_7 < 226)), placeholder_65[((((i0_i1_fused_7*672) + (i2_7*3)) + i3_7) - 1350)], T.int16(0), dtype="int16")
        for ax0_ax1_fused_ax2_fused_7 in T.serial(0, 12544):
            Conv2dOutput_7_data = T.allocate([64], "int32", "global")
            Conv2dOutput_7 = T.buffer_decl(shape=[64], dtype="int32", data=Conv2dOutput_7_data)
            for ff_3 in T.serial(0, 64):
                Conv2dOutput_7[ff_3] = 0
                for ry_2, rx_2, rc_7 in T.grid(7, 7, 3):
                    Conv2dOutput_7[ff_3] = (Conv2dOutput_7[ff_3] + (T.cast(PaddedInput_7[(((((T.floordiv(ax0_ax1_fused_ax2_fused_7, 112)*1374) + (ry_2*687)) + (T.floormod(ax0_ax1_fused_ax2_fused_7, 112)*6)) + (rx_2*3)) + rc_7)], "int32")*T.cast(placeholder_66[((((ry_2*1344) + (rx_2*192)) + (rc_7*64)) + ff_3)], "int32")))
            for ax3_inner_7 in T.serial(0, 64):
                T_cast_21[((ax0_ax1_fused_ax2_fused_7*64) + ax3_inner_7)] = T.cast(T.max(T.min(T.q_multiply_shift((Conv2dOutput_7[ax3_inner_7] + placeholder_67[ax3_inner_7]), 1939887962, 31, -9, dtype="int32"), 255), 0), "uint8")

    @T.prim_func
    def tvmgen_default_fused_nn_max_pool2d_cast(placeholder_28: T.handle, T_cast_6: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_max_pool2d_cast", "tir.noalias": True})
        placeholder_29 = T.match_buffer(placeholder_28, [802816], dtype="uint8", elem_offset=0, align=64, offset_factor=1)
        T_cast_7 = T.match_buffer(T_cast_6, [177], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        # body
        tensor_2_data = T.allocate([200704], "uint8", "global")
        tensor_2 = T.buffer_decl(shape=[200704], dtype="uint8", data=tensor_2_data)
        for ax0_ax1_fused_4 in T.serial(0, 56):
            for ax2_4 in T.serial(0, 56):
                for ax3_init in T.serial(0, 64):
                    tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_init)] = T.uint8(0)
                for rv0_rv1_fused_1, ax3_2 in T.grid(9, 64):
                    tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2)] = T.max(tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2)], T.if_then_else(((((ax0_ax1_fused_4*2) + T.floordiv(rv0_rv1_fused_1, 3)) < 112) and (((ax2_4*2) + T.floormod(rv0_rv1_fused_1, 3)) < 112)), placeholder_29[(((((ax0_ax1_fused_4*14336) + (T.floordiv(rv0_rv1_fused_1, 3)*7168)) + (ax2_4*128)) + (T.floormod(rv0_rv1_fused_1, 3)*64)) + ax3_2)], T.uint8(0), dtype="uint8"))
        for ax0_ax1_fused_5 in T.serial(0, 56):
            for ax2_5, ax3_3 in T.grid(56, 64):
                T_cast_7[(((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3)] = T.cast(tensor_2[(((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3)], "int16")

    @R.function
    def main(input: R.Tensor((16, 16), "uint8")) -> R.Tensor:
        tsid_10 = R.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int16")
        tsid_11 = R.builtin.alloc_tensor((9408, 1), runtime_device_index=0, dtype="int16")
        tsid_12 = R.builtin.alloc_tensor((64, 1), runtime_device_index=0, dtype="int32")

        lv0 = relax.call_tir(tvmgen_default_fused_cast_subtract, (input, tsid_10), (301056, 1), dtype="int32")
        lv1 = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast", (lv0, tsid_11, tsid_12), (802816, 1), dtype="int32")
        output = relax.call_tir(tvmgen_default_fused_nn_max_pool2d_cast, (lv1), (16, 16), dtype="int32")
        return output
# fmt: on


# fmt: off
@tvm.script.ir_module
class LinearStructurePlanned:
    @R.function
    def main(input: R.Tensor((16, 16), "uint8"), output: R.Tensor((16, 16), "int32"), fast_memory_0_pool: R.Object, slow_memory_1_pool: R.Object) -> R.Tuple():
        # block 0
        tsid_10: R.Tensor((1, 1), "int16") = R.memory.alloc_tensor(fast_memory_0_pool, (1, 1), offset=0, dtype="int16")
        tsid_11: R.Tensor((9408, 1), "int16") = R.memory.alloc_tensor(fast_memory_0_pool, (9408, 1), offset=0, dtype="int16")
        tsid_12: R.Tensor((64, 1), "int32") = R.memory.alloc_tensor(fast_memory_0_pool, (64, 1), offset=18816, dtype="int32")
        alloc: R.Tensor((301056, 1), "int32") = R.memory.alloc_tensor(slow_memory_1_pool, (301056, 1), offset=0, dtype="int32")
        _ = tvmgen_default_fused_cast_subtract(input, tsid_10, alloc, fast_memory_0_pool, slow_memory_1_pool, type_args=(R.Tensor(ndim=2, dtype="int32")))
        lv0: R.Tensor((301056, 1), "int32") = alloc
        alloc1: R.Tensor((802816, 1), "int32") = R.memory.alloc_tensor(slow_memory_1_pool, (802816, 1), offset=0, dtype="int32")
        _1: R.Tensor(_, "int32", ndim = 2) = R.call_packed("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast", lv0, tsid_11, tsid_12, alloc1, fast_memory_0_pool, slow_memory_1_pool, type_args=(R.Tensor(ndim=2, dtype="int32")))
        lv1: R.Tensor((802816, 1), "int32") = alloc1
        _2 = tvmgen_default_fused_nn_max_pool2d_cast(lv1, output, fast_memory_0_pool, slow_memory_1_pool, type_args=(R.Tensor(ndim=2, dtype="int32")))
        return R.Tuple()

    @T.prim_func
    def tvmgen_default_fused_nn_max_pool2d_cast(placeholder_28: T.handle, T_cast_6: T.handle, fast_memory_6_var: T.Ptr[T.uint8], slow_memory_7_var: T.Ptr[T.uint8]) -> None:
        placeholder_29 = T.match_buffer(placeholder_28, [802816], dtype="uint8")
        T_cast_7 = T.match_buffer(T_cast_6, [177], dtype="int16")
        fast_memory_6_buffer_var = T.match_buffer(fast_memory_6_var, [200704], dtype="uint8", strides=[1], elem_offset=0, align=16)
        slow_memory_7_buffer_var = T.match_buffer(slow_memory_7_var, [3525910], dtype="uint8", strides=[1], elem_offset=0, align=16)
        # body
        tensor_2_let = T.buffer_decl([200704], dtype="uint8")
        with T.let(tensor_2_let.data, T.address_of(fast_memory_6_buffer_var[0], dtype="handle")):
            for ax0_ax1_fused_4, ax2_4 in T.grid(56, 56):
                for ax3_init in T.serial(0, 64):
                    tensor_2_let[ax0_ax1_fused_4 * 3584 + ax2_4 * 64 + ax3_init] = T.uint8(0)
                for rv0_rv1_fused_1, ax3_2 in T.grid(9, 64):
                    tensor_2_let[ax0_ax1_fused_4 * 3584 + ax2_4 * 64 + ax3_2] = T.max(tensor_2_let[ax0_ax1_fused_4 * 3584 + ax2_4 * 64 + ax3_2], T.if_then_else(ax0_ax1_fused_4 * 2 + rv0_rv1_fused_1 // 3 < 112 and ax2_4 * 2 + rv0_rv1_fused_1 % 3 < 112, placeholder_29[ax0_ax1_fused_4 * 14336 + rv0_rv1_fused_1 // 3 * 7168 + ax2_4 * 128 + rv0_rv1_fused_1 % 3 * 64 + ax3_2], T.uint8(0), dtype="uint8"))
            for ax0_ax1_fused_5, ax2_5, ax3_3 in T.grid(56, 56, 64):
                T_cast_7[ax0_ax1_fused_5 * 3584 + ax2_5 * 64 + ax3_3] = T.cast(tensor_2_let[ax0_ax1_fused_5 * 3584 + ax2_5 * 64 + ax3_3], "int16")

    @T.prim_func
    def tvmgen_default_fused_cast_subtract(placeholder_2: T.handle, placeholder_3: T.handle, T_subtract: T.handle, fast_memory_2_var: T.Ptr[T.uint8], slow_memory_3_var: T.Ptr[T.uint8]) -> None:
        placeholder_4 = T.match_buffer(placeholder_2, [150528], dtype="uint8")
        placeholder_5 = T.match_buffer(placeholder_3, [1], dtype="int16")
        T_subtract_1 = T.match_buffer(T_subtract, [452], dtype="int16")
        fast_memory_2_buffer_var = T.match_buffer(fast_memory_2_var, [200704], dtype="uint8", strides=[1], elem_offset=0, align=16)
        slow_memory_3_buffer_var = T.match_buffer(slow_memory_3_var, [3525910], dtype="uint8", strides=[1], elem_offset=0, align=16)
        # body
        for ax0_ax1_fused_1, ax2_1, ax3_inner_1 in T.grid(224, 224, 3):
            T_subtract_1[ax0_ax1_fused_1 * 672 + ax2_1 * 3 + ax3_inner_1] = T.cast(placeholder_4[ax0_ax1_fused_1 * 672 + ax2_1 * 3 + ax3_inner_1], "int16") - placeholder_5[0]

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast(placeholder_62: T.handle, placeholder_63: T.handle, placeholder_64: T.handle, T_cast_20: T.handle, fast_memory_4_var: T.Ptr[T.uint8], slow_memory_5_var: T.Ptr[T.uint8]) -> None:
        placeholder_65 = T.match_buffer(placeholder_62, [150528], dtype="int16")
        placeholder_66 = T.match_buffer(placeholder_63, [9408], dtype="int16")
        placeholder_67 = T.match_buffer(placeholder_64, [64], dtype="int32")
        T_cast_21 = T.match_buffer(T_cast_20, [289], dtype="uint8")
        fast_memory_4_buffer_var = T.match_buffer(fast_memory_4_var, [200704], dtype="uint8", strides=[1], elem_offset=0, align=16)
        slow_memory_5_buffer_var = T.match_buffer(slow_memory_5_var, [3525910], dtype="uint8", strides=[1], elem_offset=0, align=16)
        # body
        PaddedInput_7_let = T.buffer_decl([157323], "int16")
        with T.let(PaddedInput_7_let.data, T.address_of(slow_memory_5_buffer_var[3211264], dtype="handle")):
            for i0_i1_fused_7, i2_7, i3_7 in T.grid(229, 229, 3):
                PaddedInput_7_let[i0_i1_fused_7 * 687 + i2_7 * 3 + i3_7] = T.if_then_else(2 <= i0_i1_fused_7 and i0_i1_fused_7 < 226 and 2 <= i2_7 and i2_7 < 226, placeholder_65[i0_i1_fused_7 * 672 + i2_7 * 3 + i3_7 - 1350], T.int16(0), dtype="int16")
            for ax0_ax1_fused_ax2_fused_7 in T.serial(0, 12544):
                Conv2dOutput_7_let = T.buffer_decl([64], "int32")
                with T.let(Conv2dOutput_7_let.data, T.address_of(fast_memory_4_buffer_var[19072], dtype="handle")):
                    for ff_3 in T.serial(0, 64):
                        Conv2dOutput_7_let[ff_3] = 0
                        for ry_2, rx_2, rc_7 in T.grid(7, 7, 3):
                            Conv2dOutput_7_let[ff_3] = Conv2dOutput_7_let[ff_3] + T.cast(PaddedInput_7_let[ax0_ax1_fused_ax2_fused_7 // 112 * 1374 + ry_2 * 687 + ax0_ax1_fused_ax2_fused_7 % 112 * 6 + rx_2 * 3 + rc_7], "int32") * T.cast(placeholder_66[ry_2 * 1344 + rx_2 * 192 + rc_7 * 64 + ff_3], "int32")
                    for ax3_inner_7 in T.serial(0, 64):
                        T_cast_21[ax0_ax1_fused_ax2_fused_7 * 64 + ax3_inner_7] = T.cast(T.max(T.min(T.q_multiply_shift(Conv2dOutput_7_let[ax3_inner_7] + placeholder_67[ax3_inner_7], 1939887962, 31, -9, dtype="int32"), 255), 0), "uint8")
# fmt: on


def test_mobilenet_subgraph():
    target = Target("c")
    relax_mod = LinearStructure
    passes = [
        relax.transform.ToNonDataflow(),
        relax.transform.CallTIRRewrite(),
        relax.transform.ConvertRelaxMainToDPS(attach_io_to_attrs=False),
    ]
    seq = tvm.transform.Sequential(passes)
    relax_mod = seq(relax_mod)

    fast_memory_pool = WorkspacePoolInfo(
        "fast_memory",
        [target],
        PoolInfoProperties(size_hint_bytes=200704),
    )
    slow_memory_pool = WorkspacePoolInfo(
        "slow_memory",
        [target],
    )
    relax_mod = _assign_targets_to_relaxfuncs_irmodule(relax_mod, target)

    relax_mod = _assign_poolinfos_to_allocates_in_irmodule(
        relax_mod, [fast_memory_pool, slow_memory_pool]
    )
    main_func = relax_mod["main"]
    buffer_analysis = tvm.relax.analysis.extract_buffer_info(main_func, relax_mod)
    buffer_info_map = buffer_analysis.buffer_info_stmts

    fcreate_array_bi = tvm.get_global_func("tir.usmp.CreateArrayBufferInfo")
    buffer_info_arr = fcreate_array_bi(buffer_info_map)
    fusmp_algo_greedy_by_size = tvm.get_global_func("tir.usmp.algo.greedy_by_size")
    buffer_pool_allocations = fusmp_algo_greedy_by_size(
        buffer_info_arr, buffer_analysis.memory_pressure
    )
    fassign_stmt_pool_allocations = tvm.get_global_func("tir.usmp.AssignStmtPoolAllocations")
    pool_allocations = fassign_stmt_pool_allocations(buffer_info_map, buffer_pool_allocations)
    tir_mod_with_offsets = tvm.relax.transform.ConvertPoolAllocationsToOffsets(
        pool_allocations, emit_tvmscript_printable=True, insert_storage_allocations=False
    )(relax_mod)

    tir_mod_with_offsets_ref = LinearStructurePlanned
    tir_mod_with_offsets_ref = _append_type_args(
        tir_mod_with_offsets_ref, ["int32", "int32", "int32"]
    )

    for gv, ref_func in tir_mod_with_offsets_ref.functions.items():
        actual_func = tir_mod_with_offsets[gv.name_hint]
        tvm.ir.assert_structural_equal(actual_func, ref_func)


# fmt: off
@tvm.script.ir_module
class ResnetStructure:
    @T.prim_func
    def tvmgen_default_fused_cast_subtract_fixed_point_multiply_add_clip_cast_cast(placeholder: T.handle, placeholder_1: T.handle, T_cast: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast_subtract_fixed_point_multiply_add_clip_cast_cast", "tir.noalias": True})
        placeholder_2 = T.match_buffer(placeholder, [360000], dtype="uint8")
        placeholder_3 = T.match_buffer(placeholder_1, [64], dtype="int32")
        T_cast_1 = T.match_buffer(T_cast, [215], dtype="int16")
        # body
        for ax0_ax1_fused, ax2, ax3_outer, ax3_inner in T.grid(75, 75, 4, 16):
            T_cast_1[ax0_ax1_fused * 4800 + ax2 * 64 + ax3_outer * 16 + ax3_inner] = T.cast(T.cast(T.max(T.min(T.q_multiply_shift(T.cast(placeholder_2[ax0_ax1_fused * 4800 + ax2 * 64 + ax3_outer * 16 + ax3_inner], "int32") - 94, 1843157232, 31, 1, dtype="int32") + placeholder_3[ax3_outer * 16 + ax3_inner], 255), 0), "uint8"), "int16")

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1(placeholder_10: T.handle, placeholder_11: T.handle, placeholder_12: T.handle, T_cast_4: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1", "tir.noalias": True})
        placeholder_13 = T.match_buffer(placeholder_10, [360000], dtype="int16")
        placeholder_14 = T.match_buffer(placeholder_11, [36864], dtype="int16")
        placeholder_15 = T.match_buffer(placeholder_12, [64], dtype="int32")
        T_cast_5 = T.match_buffer(T_cast_4, [215], dtype="int16")
        # body
        PaddedInput_1_data = T.allocate([379456], "int16", "global")
        PaddedInput_1 = T.buffer_decl(shape=[379456], dtype="int16", data=PaddedInput_1_data)
        for i0_i1_fused_1, i2_1, i3_1 in T.grid(77, 77, 64):
            PaddedInput_1[i0_i1_fused_1 * 4928 + i2_1 * 64 + i3_1] = T.if_then_else(1 <= i0_i1_fused_1 and i0_i1_fused_1 < 76 and 1 <= i2_1 and i2_1 < 76, placeholder_13[i0_i1_fused_1 * 4800 + i2_1 * 64 + i3_1 - 4864], T.int16(0), dtype="int16")
        for ax0_ax1_fused_ax2_fused_1 in T.serial(0, 5625):
            Conv2dOutput_1_data = T.allocate([64], "int32", "global")
            Conv2dOutput_1 = T.buffer_decl(shape=[64], dtype="int32", data=Conv2dOutput_1_data)
            for ff_1 in T.serial(0, 64):
                Conv2dOutput_1[ff_1] = 0
                for ry, rx, rc_1 in T.grid(3, 3, 64):
                    Conv2dOutput_1[ff_1] = Conv2dOutput_1[ff_1] + T.cast(PaddedInput_1[T.floordiv(ax0_ax1_fused_ax2_fused_1, 75) * 4928 + ry * 4928 + rx * 64 + T.floormod(ax0_ax1_fused_ax2_fused_1, 75) * 64 + rc_1], "int32") * T.cast(placeholder_14[ry * 12288 + rx * 4096 + rc_1 * 64 + ff_1], "int32")
            for ax3_inner_2 in T.serial(0, 64):
                T_cast_5[ax0_ax1_fused_ax2_fused_1 * 64 + ax3_inner_2] = T.cast(T.cast(T.max(T.min(T.q_multiply_shift(Conv2dOutput_1[ax3_inner_2] + placeholder_15[ax3_inner_2], 1608879842, 31, -7, dtype="int32"), 255), 0), "uint8"), "int16")

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_15934180698220515269_(placeholder_16: T.handle, placeholder_17: T.handle, placeholder_18: T.handle, T_add: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_15934180698220515269_", "tir.noalias": True})
        placeholder_19 = T.match_buffer(placeholder_16, [360000], dtype="int16")
        placeholder_20 = T.match_buffer(placeholder_17, [16384], dtype="int16")
        placeholder_21 = T.match_buffer(placeholder_18, [256], dtype="int32")
        T_add_1 = T.match_buffer(T_add, [407], dtype="int32")
        # body
        PaddedInput_2_data = T.allocate([360000], "int16", "global")
        PaddedInput_2 = T.buffer_decl(shape=[360000], dtype="int16", data=PaddedInput_2_data)
        for i0_i1_fused_2, i2_2, i3_2 in T.grid(75, 75, 64):
            PaddedInput_2[i0_i1_fused_2 * 4800 + i2_2 * 64 + i3_2] = placeholder_19[i0_i1_fused_2 * 4800 + i2_2 * 64 + i3_2]
        for ax0_ax1_fused_ax2_fused_2 in T.serial(0, 5625):
            Conv2dOutput_2_data = T.allocate([64], "int32", "global")
            Conv2dOutput_2 = T.buffer_decl(shape=[64], dtype="int32", data=Conv2dOutput_2_data)
            for ax3_outer_1 in T.serial(0, 4):
                for ff_2 in T.serial(0, 64):
                    Conv2dOutput_2[ff_2] = 0
                    for rc_2 in T.serial(0, 64):
                        Conv2dOutput_2[ff_2] = Conv2dOutput_2[ff_2] + T.cast(PaddedInput_2[ax0_ax1_fused_ax2_fused_2 * 64 + rc_2], "int32") * T.cast(placeholder_20[rc_2 * 256 + ax3_outer_1 * 64 + ff_2], "int32")
                for ax3_inner_3 in T.serial(0, 64):
                    T_add_1[ax0_ax1_fused_ax2_fused_2 * 256 + ax3_outer_1 * 64 + ax3_inner_3] = T.q_multiply_shift(T.cast(T.cast(T.max(T.min(T.q_multiply_shift(Conv2dOutput_2[ax3_inner_3] + placeholder_21[ax3_outer_1 * 64 + ax3_inner_3], 1711626602, 31, -8, dtype="int32") + 132, 255), 0), "uint8"), "int32") - 132, 2094289803, 31, -2, dtype="int32") + 136

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_4200876283395191415_(placeholder_22: T.handle, placeholder_23: T.handle, placeholder_24: T.handle, placeholder_25: T.handle, T_cast_6: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_4200876283395191415_", "tir.noalias": True})
        placeholder_29 = T.match_buffer(placeholder_22, [360000], dtype="int16")
        placeholder_27 = T.match_buffer(placeholder_23, [16384], dtype="int16")
        placeholder_26 = T.match_buffer(placeholder_24, [256], dtype="int32")
        placeholder_28 = T.match_buffer(placeholder_25, [1440000], dtype="int32")
        T_cast_7 = T.match_buffer(T_cast_6, [407], dtype="uint8")
        # body
        PaddedInput_3_data = T.allocate([360000], "int16", "global")
        PaddedInput_3 = T.buffer_decl(shape=[360000], dtype="int16", data=PaddedInput_3_data)
        for i0_i1_fused_3, i2_3, i3_3 in T.grid(75, 75, 64):
            PaddedInput_3[i0_i1_fused_3 * 4800 + i2_3 * 64 + i3_3] = placeholder_29[i0_i1_fused_3 * 4800 + i2_3 * 64 + i3_3]
        for ax0_ax1_fused_ax2_fused_3 in T.serial(0, 5625):
            Conv2dOutput_3_data = T.allocate([64], "int32", "global")
            Conv2dOutput_3 = T.buffer_decl(shape=[64], dtype="int32", data=Conv2dOutput_3_data)
            for ax3_outer_2 in T.serial(0, 4):
                for ff_3 in T.serial(0, 64):
                    Conv2dOutput_3[ff_3] = 0
                    for rc_3 in T.serial(0, 64):
                        Conv2dOutput_3[ff_3] = Conv2dOutput_3[ff_3] + T.cast(PaddedInput_3[ax0_ax1_fused_ax2_fused_3 * 64 + rc_3], "int32") * T.cast(placeholder_27[rc_3 * 256 + ax3_outer_2 * 64 + ff_3], "int32")
                for ax3_inner_4 in T.serial(0, 64):
                    T_cast_7[ax0_ax1_fused_ax2_fused_3 * 256 + ax3_outer_2 * 64 + ax3_inner_4] = T.cast(T.max(T.min(T.q_multiply_shift(T.cast(T.cast(T.max(T.min(T.q_multiply_shift(Conv2dOutput_3[ax3_inner_4] + placeholder_26[ax3_outer_2 * 64 + ax3_inner_4], 1343014664, 31, -8, dtype="int32") + 136, 255), 0), "uint8"), "int32") - 136, 1073903788, 31, 1, dtype="int32") + placeholder_28[ax0_ax1_fused_ax2_fused_3 * 256 + ax3_outer_2 * 64 + ax3_inner_4], 255), 0), "uint8")

    @R.function
    def main(input: R.Tensor((16, 16), "uint8")) -> R.Tensor:
        param_p0 = R.builtin.alloc_tensor((64, 1), runtime_device_index=0, dtype="int32")
        param_p3 = R.builtin.alloc_tensor((4096, 1), runtime_device_index=0, dtype="int16")
        param_p4 = R.builtin.alloc_tensor((64, 1), runtime_device_index=0, dtype="int32")
        param_p5 = R.builtin.alloc_tensor((36864, 1), runtime_device_index=0, dtype="int16")
        param_p6 = R.builtin.alloc_tensor((64, 1), runtime_device_index=0, dtype="int32")
        param_p7 = R.builtin.alloc_tensor((16384, 1), runtime_device_index=0, dtype="int16")
        param_p8 = R.builtin.alloc_tensor((256, 1), runtime_device_index=0, dtype="int32")
        param_p1 = R.builtin.alloc_tensor((16384, 1), runtime_device_index=0, dtype="int16")
        param_p2 = R.builtin.alloc_tensor((256, 1), runtime_device_index=0, dtype="int32")

        sid_2 = relax.call_tir("tvmgen_default_fused_cast_subtract_fixed_point_multiply_add_clip_cast_cast", (input, param_p0), (720000, 1), dtype="int8")
        sid_8 = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast", (sid_2, param_p3, param_p4), (720000, 1), dtype="int8")
        sid_7 = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1", (sid_8, param_p5, param_p6), (720000, 1), dtype="int8")
        sid_6 = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_15934180698220515269_", (sid_7, param_p7, param_p8), (5760000, 1), dtype="int8")
        output = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_4200876283395191415_", (sid_2, param_p1, param_p2, sid_6), (16, 16), dtype="int32")
        return output

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast(placeholder_4: T.handle, placeholder_5: T.handle, placeholder_6: T.handle, T_cast_2: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast", "tir.noalias": True})
        placeholder_7 = T.match_buffer(placeholder_4, [360000], dtype="int16")
        placeholder_8 = T.match_buffer(placeholder_5, [4096], dtype="int16")
        placeholder_9 = T.match_buffer(placeholder_6, [64], dtype="int32")
        T_cast_3 = T.match_buffer(T_cast_2, [215], dtype="int16")
        # body
        PaddedInput_data = T.allocate([360000], "int16", "global")
        PaddedInput = T.buffer_decl([360000], "int16", data=PaddedInput_data)
        for i0_i1_fused, i2, i3 in T.grid(75, 75, 64):
            PaddedInput[i0_i1_fused * 4800 + i2 * 64 + i3] = placeholder_7[i0_i1_fused * 4800 + i2 * 64 + i3]
        for ax0_ax1_fused_ax2_fused in T.serial(0, 5625):
            Conv2dOutput_data = T.allocate([64], "int32", "global")
            Conv2dOutput = T.buffer_decl([64], "int32", data=Conv2dOutput_data)
            for ff in T.serial(0, 64):
                Conv2dOutput[ff] = 0
                for rc in T.serial(0, 64):
                    Conv2dOutput[ff] = Conv2dOutput[ff] + T.cast(PaddedInput[ax0_ax1_fused_ax2_fused * 64 + rc], "int32") * T.cast(placeholder_8[rc * 64 + ff], "int32")
            for ax3_inner_1 in T.serial(0, 64):
                T_cast_3[ax0_ax1_fused_ax2_fused * 64 + ax3_inner_1] = T.cast(T.cast(T.max(T.min(T.q_multiply_shift(Conv2dOutput[ax3_inner_1] + placeholder_9[ax3_inner_1], 1843106743, 31, -6, dtype="int32"), 255), 0), "uint8"), "int16")
# fmt: on


# fmt: off
@tvm.script.ir_module
class ResnetStructurePlanned:
    @T.prim_func
    def tvmgen_default_fused_cast_subtract_fixed_point_multiply_add_clip_cast_cast(placeholder: T.handle, placeholder_1: T.handle, T_cast: T.handle, global_workspace_1_pool: T.Ptr[T.uint8]) -> None:
        placeholder_2 = T.match_buffer(placeholder, [360000], dtype="uint8")
        placeholder_3 = T.match_buffer(placeholder_1, [64], dtype="int32")
        T_cast_1 = T.match_buffer(T_cast, [215], dtype="int16")
        global_workspace_1_buffer_var = T.match_buffer(global_workspace_1_pool, [7954048], dtype="uint8", strides=[1], elem_offset=0, align=16)
        # body
        for ax0_ax1_fused, ax2, ax3_outer, ax3_inner in T.grid(75, 75, 4, 16):
            T_cast_1[ax0_ax1_fused * 4800 + ax2 * 64 + ax3_outer * 16 + ax3_inner] = T.cast(T.cast(T.max(T.min(T.q_multiply_shift(T.cast(placeholder_2[ax0_ax1_fused * 4800 + ax2 * 64 + ax3_outer * 16 + ax3_inner], "int32") - 94, 1843157232, 31, 1, dtype="int32") + placeholder_3[ax3_outer * 16 + ax3_inner], 255), 0), "uint8"), "int16")

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_4200876283395191415_(placeholder_22: T.handle, placeholder_23: T.handle, placeholder_24: T.handle, placeholder_25: T.handle, T_cast_6: T.handle, global_workspace_5_var: T.Ptr[T.uint8]) -> None:
        placeholder_29 = T.match_buffer(placeholder_22, [360000], dtype="int16")
        placeholder_27 = T.match_buffer(placeholder_23, [16384], dtype="int16")
        placeholder_26 = T.match_buffer(placeholder_24, [256], dtype="int32")
        placeholder_28 = T.match_buffer(placeholder_25, [1440000], dtype="int32")
        T_cast_7 = T.match_buffer(T_cast_6, [407], dtype="uint8")
        global_workspace_5_buffer_var = T.match_buffer(global_workspace_5_var, [7954048], dtype="uint8", strides=[1], elem_offset=0, align=16)
        # body
        PaddedInput_3_let = T.buffer_decl([360000], 'int16')
        with T.let(PaddedInput_3_let.data, T.address_of(global_workspace_5_buffer_var[6480000], dtype="handle")):
            for i0_i1_fused_3, i2_3, i3_3 in T.grid(75, 75, 64):
                PaddedInput_3_let[i0_i1_fused_3 * 4800 + i2_3 * 64 + i3_3] = placeholder_29[i0_i1_fused_3 * 4800 + i2_3 * 64 + i3_3]
            for ax0_ax1_fused_ax2_fused_3 in T.serial(0, 5625):
                Conv2dOutput_3_let = T.buffer_decl([64], 'int32')
                with T.let(Conv2dOutput_3_let.data, T.address_of(global_workspace_5_buffer_var[7233792], dtype="handle")):
                    for ax3_outer_2 in T.serial(0, 4):
                        for ff_3 in T.serial(0, 64):
                            Conv2dOutput_3_let[ff_3] = 0
                            for rc_3 in T.serial(0, 64):
                                Conv2dOutput_3_let[ff_3] = Conv2dOutput_3_let[ff_3] + T.cast(PaddedInput_3_let[ax0_ax1_fused_ax2_fused_3 * 64 + rc_3], "int32") * T.cast(placeholder_27[rc_3 * 256 + ax3_outer_2 * 64 + ff_3], "int32")
                        for ax3_inner_4 in T.serial(0, 64):
                            T_cast_7[ax0_ax1_fused_ax2_fused_3 * 256 + ax3_outer_2 * 64 + ax3_inner_4] = T.cast(T.max(T.min(T.q_multiply_shift(T.cast(T.cast(T.max(T.min(T.q_multiply_shift(Conv2dOutput_3_let[ax3_inner_4] + placeholder_26[ax3_outer_2 * 64 + ax3_inner_4], 1343014664, 31, -8, dtype="int32") + 136, 255), 0), "uint8"), "int32") - 136, 1073903788, 31, 1, dtype="int32") + placeholder_28[ax0_ax1_fused_ax2_fused_3 * 256 + ax3_outer_2 * 64 + ax3_inner_4], 255), 0), "uint8")

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_15934180698220515269_(placeholder_16: T.handle, placeholder_17: T.handle, placeholder_18: T.handle, T_add: T.handle, global_workspace_4_var: T.Ptr[T.uint8]) -> None:
        placeholder_19 = T.match_buffer(placeholder_16, [360000], dtype="int16")
        placeholder_20 = T.match_buffer(placeholder_17, [16384], dtype="int16")
        placeholder_21 = T.match_buffer(placeholder_18, [256], dtype="int32")
        T_add_1 = T.match_buffer(T_add, [407], dtype="int32")
        global_workspace_4_buffer_var = T.match_buffer(global_workspace_4_var, [7954048], dtype="uint8", strides=[1], elem_offset=0, align=16)
        # body
        PaddedInput_2_let = T.buffer_decl([360000], "int16")
        with T.let(PaddedInput_2_let.data, T.address_of(global_workspace_4_buffer_var[7200000], dtype="handle")):
            for i0_i1_fused_2, i2_2, i3_2 in T.grid(75, 75, 64):
                PaddedInput_2_let[i0_i1_fused_2 * 4800 + i2_2 * 64 + i3_2] = placeholder_19[i0_i1_fused_2 * 4800 + i2_2 * 64 + i3_2]
            for ax0_ax1_fused_ax2_fused_2 in T.serial(0, 5625):
                Conv2dOutput_2_let = T.buffer_decl([64], 'int32')
                with T.let(Conv2dOutput_2_let.data, T.address_of(global_workspace_4_buffer_var[7953792], dtype="handle")):
                    for ax3_outer_1 in T.serial(0, 4):
                        for ff_2 in T.serial(0, 64):
                            Conv2dOutput_2_let[ff_2] = 0
                            for rc_2 in T.serial(0, 64):
                                Conv2dOutput_2_let[ff_2] = Conv2dOutput_2_let[ff_2] + T.cast(PaddedInput_2_let[ax0_ax1_fused_ax2_fused_2 * 64 + rc_2], "int32") * T.cast(placeholder_20[rc_2 * 256 + ax3_outer_1 * 64 + ff_2], "int32")
                        for ax3_inner_3 in T.serial(0, 64):
                            T_add_1[ax0_ax1_fused_ax2_fused_2 * 256 + ax3_outer_1 * 64 + ax3_inner_3] = T.q_multiply_shift(T.cast(T.cast(T.max(T.min(T.q_multiply_shift(Conv2dOutput_2_let[ax3_inner_3] + placeholder_21[ax3_outer_1 * 64 + ax3_inner_3], 1711626602, 31, -8, dtype="int32") + 132, 255), 0), "uint8"), "int32") - 132, 2094289803, 31, -2, dtype="int32") + 136

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast(placeholder_4: T.handle, placeholder_5: T.handle, placeholder_6: T.handle, T_cast_2: T.handle, global_workspace_2_var: T.Ptr[T.uint8]) -> None:
        placeholder_7 = T.match_buffer(placeholder_4, [360000], dtype="int16")
        placeholder_8 = T.match_buffer(placeholder_5, [4096], dtype="int16")
        placeholder_9 = T.match_buffer(placeholder_6, [64], dtype="int32")
        T_cast_3 = T.match_buffer(T_cast_2, [215], dtype="int16")
        global_workspace_2_buffer_var = T.match_buffer(global_workspace_2_var, [7954048], dtype="uint8", strides=[1], elem_offset=0, align=16)
        # body
        PaddedInput_let = T.buffer_decl([360000], "int16")
        with T.let(PaddedInput_let.data, T.address_of(global_workspace_2_buffer_var[7200000], dtype="handle")):
            for i0_i1_fused, i2, i3 in T.grid(75, 75, 64):
                PaddedInput_let[i0_i1_fused * 4800 + i2 * 64 + i3] = placeholder_7[i0_i1_fused * 4800 + i2 * 64 + i3]
            for ax0_ax1_fused_ax2_fused in T.serial(0, 5625):
                Conv2dOutput_let = T.buffer_decl([64], "int32")
                with T.let(Conv2dOutput_let.data, T.address_of(global_workspace_2_buffer_var[7928448], dtype="handle")):
                    for ff in T.serial(0, 64):
                        Conv2dOutput_let[ff] = 0
                        for rc in T.serial(0, 64):
                            Conv2dOutput_let[ff] = Conv2dOutput_let[ff] + T.cast(PaddedInput_let[ax0_ax1_fused_ax2_fused * 64 + rc], "int32") * T.cast(placeholder_8[rc * 64 + ff], "int32")
                    for ax3_inner_1 in T.serial(0, 64):
                        T_cast_3[ax0_ax1_fused_ax2_fused * 64 + ax3_inner_1] = T.cast(T.cast(T.max(T.min(T.q_multiply_shift(Conv2dOutput_let[ax3_inner_1] + placeholder_9[ax3_inner_1], 1843106743, 31, -6, dtype="int32"), 255), 0), "uint8"), "int16")

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1(placeholder_10: T.handle, placeholder_11: T.handle, placeholder_12: T.handle, T_cast_4: T.handle, global_workspace_3_var: T.Ptr[T.uint8]) -> None:
        placeholder_13 = T.match_buffer(placeholder_10, [360000], dtype="int16")
        placeholder_14 = T.match_buffer(placeholder_11, [36864], dtype="int16")
        placeholder_15 = T.match_buffer(placeholder_12, [64], dtype="int32")
        T_cast_5 = T.match_buffer(T_cast_4, [215], dtype="int16")
        global_workspace_3_buffer_var = T.match_buffer(global_workspace_3_var, [7954048], dtype="uint8", strides=[1], elem_offset=0, align=16)
        # body
        PaddedInput_1_let = T.buffer_decl([379456], "int16")
        with T.let(PaddedInput_1_let.data, T.address_of(global_workspace_3_buffer_var[0], dtype="handle")):
            for i0_i1_fused_1, i2_1, i3_1 in T.grid(77, 77, 64):
                PaddedInput_1_let[i0_i1_fused_1 * 4928 + i2_1 * 64 + i3_1] = T.if_then_else(1 <= i0_i1_fused_1 and i0_i1_fused_1 < 76 and 1 <= i2_1 and i2_1 < 76, placeholder_13[i0_i1_fused_1 * 4800 + i2_1 * 64 + i3_1 - 4864], T.int16(0), dtype="int16")
            for ax0_ax1_fused_ax2_fused_1 in T.serial(0, 5625):
                Conv2dOutput_1_let = T.buffer_decl([64], "int32")
                with T.let(Conv2dOutput_1_let.data, T.address_of(global_workspace_3_buffer_var[7273984], dtype="handle")):
                    for ff_1 in T.serial(0, 64):
                        Conv2dOutput_1_let[ff_1] = 0
                        for ry, rx, rc_1 in T.grid(3, 3, 64):
                            Conv2dOutput_1_let[ff_1] = Conv2dOutput_1_let[ff_1] + T.cast(PaddedInput_1_let[ax0_ax1_fused_ax2_fused_1 // 75 * 4928 + ry * 4928 + rx * 64 + ax0_ax1_fused_ax2_fused_1 % 75 * 64 + rc_1], "int32") * T.cast(placeholder_14[ry * 12288 + rx * 4096 + rc_1 * 64 + ff_1], "int32")
                    for ax3_inner_2 in T.serial(0, 64):
                        T_cast_5[ax0_ax1_fused_ax2_fused_1 * 64 + ax3_inner_2] = T.cast(T.cast(T.max(T.min(T.q_multiply_shift(Conv2dOutput_1_let[ax3_inner_2] + placeholder_15[ax3_inner_2], 1608879842, 31, -7, dtype="int32"), 255), 0), "uint8"), "int16")

    @R.function
    def main(input: R.Tensor((16, 16), "uint8"), output: R.Tensor((16, 16), "int32"), global_workspace_0_pool: R.Object) -> R.Tuple():
        # block 0
        param_p0: R.Tensor((64, 1), "int32") = R.memory.alloc_tensor(global_workspace_0_pool, (64, 1), offset=6480000, dtype="int32")
        param_p3: R.Tensor((4096, 1), "int16") = R.memory.alloc_tensor(global_workspace_0_pool, (4096, 1), offset=7920000, dtype="int16")
        param_p4: R.Tensor((64, 1), "int32") = R.memory.alloc_tensor(global_workspace_0_pool, (64, 1), offset=7928192, dtype="int32")
        param_p5: R.Tensor((36864, 1), "int16") = R.memory.alloc_tensor(global_workspace_0_pool, (36864, 1), offset=7200000, dtype="int16")
        param_p6: R.Tensor((64, 1), "int32") = R.memory.alloc_tensor(global_workspace_0_pool, (64, 1), offset=7273728, dtype="int32")
        param_p7: R.Tensor((16384, 1), "int16") = R.memory.alloc_tensor(global_workspace_0_pool, (16384, 1), offset=7920000, dtype="int16")
        param_p8: R.Tensor((256, 1), "int32") = R.memory.alloc_tensor(global_workspace_0_pool, (256, 1), offset=7952768, dtype="int32")
        param_p1: R.Tensor((16384, 1), "int16") = R.memory.alloc_tensor(global_workspace_0_pool, (16384, 1), offset=7200000, dtype="int16")
        param_p2: R.Tensor((256, 1), "int32") = R.memory.alloc_tensor(global_workspace_0_pool, (256, 1), offset=7232768, dtype="int32")
        alloc: R.Tensor((720000, 1), "int8") = R.memory.alloc_tensor(global_workspace_0_pool, (720000, 1), offset=5760000, dtype="int8")
        _: R.Tensor(_, "int8", ndim = 2) = R.call_packed("tvmgen_default_fused_cast_subtract_fixed_point_multiply_add_clip_cast_cast", input, param_p0, alloc, global_workspace_0_pool, type_args=(Tensor(ndim=2, dtype="int8")))
        sid_2: R.Tensor((720000, 1), "int8") = alloc
        alloc1: R.Tensor((720000, 1), "int8") = R.memory.alloc_tensor(global_workspace_0_pool, (720000, 1), offset=6480000, dtype="int8")
        _1: R.Tensor(_, "int8", ndim = 2) = R.call_packed("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast", sid_2, param_p3, param_p4, alloc1, global_workspace_0_pool, type_args=(R.Tensor(ndim=2, dtype="int8")))
        sid_8: R.Tensor((720000, 1), "int8") = alloc1
        alloc2: R.Tensor((720000, 1), "int8") = R.memory.alloc_tensor(global_workspace_0_pool, (720000, 1), offset=6480000, dtype="int8")
        _2: R.Tensor(_, "int8", ndim = 2) = R.call_packed("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1", sid_8, param_p5, param_p6, alloc2, global_workspace_0_pool, type_args=(R.Tensor(ndim=2, dtype="int8")))
        sid_7: R.Tensor((720000, 1), "int8") = alloc2
        alloc3: R.Tensor((5760000, 1), "int8") = R.memory.alloc_tensor(global_workspace_0_pool, (5760000, 1), offset=0, dtype="int8")
        _3: R.Tensor(_, "int8", ndim = 2) = R.call_packed("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_15934180698220515269_", sid_7, param_p7, param_p8, alloc3, global_workspace_0_pool, type_args=(R.Tensor(ndim=2, dtype="int8")))
        sid_6: R.Tensor((5760000, 1), "int8") = alloc3
        _4: R.Tensor(_, "int32", ndim = 2) = R.call_packed("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_4200876283395191415_", sid_2, param_p1, param_p2, sid_6, output, global_workspace_0_pool, type_args=(R.Tensor(ndim=2, dtype="int32")))
        return R.Tuple()
# fmt: on


def test_resnet_subgraph():
    target = Target("c")
    relax_mod = ResnetStructure
    passes = [
        relax.transform.ToNonDataflow(),
        relax.transform.CallTIRRewrite(),
        relax.transform.ConvertRelaxMainToDPS(attach_io_to_attrs=False),
    ]
    seq = tvm.transform.Sequential(passes)
    relax_mod = seq(relax_mod)

    global_workspace_pool = WorkspacePoolInfo(
        "global_workspace",
        [target],
    )
    relax_mod = _assign_targets_to_relaxfuncs_irmodule(relax_mod, target)

    relax_mod = _assign_poolinfos_to_allocates_in_irmodule(relax_mod, [global_workspace_pool])
    main_func = relax_mod["main"]
    buffer_analysis = tvm.relax.analysis.extract_buffer_info(main_func, relax_mod)
    buffer_info_map = buffer_analysis.buffer_info_stmts

    fcreate_array_bi = tvm.get_global_func("tir.usmp.CreateArrayBufferInfo")
    buffer_info_arr = fcreate_array_bi(buffer_info_map)
    fusmp_algo_greedy_by_size = tvm.get_global_func("tir.usmp.algo.greedy_by_size")
    buffer_pool_allocations = fusmp_algo_greedy_by_size(
        buffer_info_arr, buffer_analysis.memory_pressure
    )
    fassign_stmt_pool_allocations = tvm.get_global_func("tir.usmp.AssignStmtPoolAllocations")
    pool_allocations = fassign_stmt_pool_allocations(buffer_info_map, buffer_pool_allocations)
    tir_mod_with_offsets = tvm.relax.transform.ConvertPoolAllocationsToOffsets(
        pool_allocations, emit_tvmscript_printable=True, insert_storage_allocations=False
    )(relax_mod)

    tir_mod_with_offsets_ref = ResnetStructurePlanned
    tir_mod_with_offsets_ref = _append_type_args(
        tir_mod_with_offsets_ref, ["int8", "int8", "int8", "int8", "int32"]
    )

    for gv, ref_func in tir_mod_with_offsets_ref.functions.items():
        actual_func = tir_mod_with_offsets[gv.name_hint]
        tvm.ir.assert_structural_equal(actual_func, ref_func)


# fmt: off
@tvm.script.ir_module
class TensorIntrinStructure:
    @T.prim_func
    def tensor_intrin_primfunc(output: T.handle) -> None:
        dense_data = T.allocate([10], "int32", "global")
        T.evaluate(
            T.call_extern(
                "intrin_function",
                T.tvm_access_ptr(
                    T.type_annotation(dtype="int32"), dense_data, 0, 1, 2, dtype="handle"
                ),
                dtype="int32",
            )
        )
        dense = T.buffer_decl([10], "int32", data=dense_data)
        dense[0] = T.q_multiply_shift(dense[0], 1608879842, 31, -7, dtype="int32")

    @R.function
    def main(input: R.Tensor((1, 1), "uint8")) -> R.Tensor:
        _ = relax.call_tir("tensor_intrin_primfunc", (), (1, 1), dtype="int32")
        output = R.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int8")
        return output


@tvm.script.ir_module
class TensorIntrinStructurePlanned:
    @T.prim_func
    def tensor_intrin_primfunc(output: T.handle, global_workspace_1_pool: T.Ptr[T.uint8]) -> None:
        global_workspace_1_buffer_var = T.match_buffer(
            global_workspace_1_pool, [40], dtype="uint8", strides=[1], elem_offset=0, align=16
        )
        dense_let = T.buffer_decl([10], "int32")
        with T.let(dense_let.data, T.address_of(global_workspace_1_buffer_var[0], dtype="handle")):
            T.evaluate(
                T.call_extern(
                    "intrin_function",
                    T.tvm_access_ptr(
                        T.type_annotation(dtype="int32"), dense_let.data, 0, 1, 2, dtype="handle"
                    ),
                    dtype="int32",
                )
            )
            dense_let[0] = T.q_multiply_shift(dense_let[0], 1608879842, 31, -7, dtype="int32")

    @R.function
    def main(input: R.Tensor((1, 1), "uint8"), output: R.Tensor((1, 1), "int8"), global_workspace_0_pool: R.Object) -> R.Tuple():
        # block 0
        alloc: R.Tensor((1, 1), "int32") = R.memory.alloc_tensor(global_workspace_0_pool, (1, 1), offset=0, dtype="int32")
        _: R.Tensor(_, "int32", ndim = 2) = R.call_packed("tensor_intrin_primfunc", (), alloc, global_workspace_0_pool, type_args=(R.Tensor(ndim=2, dtype="int32")))
        _1: R.Tensor((1, 1), "int32") = alloc
        return R.Tuple()

# fmt: on


def test_tensor_intrin():
    target = Target("c")
    relax_mod = TensorIntrinStructure
    passes = [
        relax.transform.ToNonDataflow(),
        relax.transform.CallTIRRewrite(),
        relax.transform.ConvertRelaxMainToDPS(attach_io_to_attrs=False),
    ]
    seq = tvm.transform.Sequential(passes)
    relax_mod = seq(relax_mod)

    global_workspace_pool = WorkspacePoolInfo(
        "global_workspace",
        [target],
    )
    relax_mod = _assign_targets_to_relaxfuncs_irmodule(relax_mod, target)

    relax_mod = _assign_poolinfos_to_allocates_in_irmodule(relax_mod, [global_workspace_pool])
    main_func = relax_mod["main"]
    buffer_analysis = tvm.relax.analysis.extract_buffer_info(main_func, relax_mod)
    buffer_info_map = buffer_analysis.buffer_info_stmts

    fcreate_array_bi = tvm.get_global_func("tir.usmp.CreateArrayBufferInfo")
    buffer_info_arr = fcreate_array_bi(buffer_info_map)
    fusmp_algo_greedy_by_size = tvm.get_global_func("tir.usmp.algo.greedy_by_size")
    buffer_pool_allocations = fusmp_algo_greedy_by_size(
        buffer_info_arr, buffer_analysis.memory_pressure
    )
    fassign_stmt_pool_allocations = tvm.get_global_func("tir.usmp.AssignStmtPoolAllocations")
    pool_allocations = fassign_stmt_pool_allocations(buffer_info_map, buffer_pool_allocations)
    tir_mod_with_offsets = tvm.relax.transform.ConvertPoolAllocationsToOffsets(
        pool_allocations, emit_tvmscript_printable=True, insert_storage_allocations=False
    )(relax_mod)

    tir_mod_with_offsets_ref = TensorIntrinStructurePlanned
    tir_mod_with_offsets_ref = _append_type_args(tir_mod_with_offsets_ref, ["int32"])

    for gv, ref_func in tir_mod_with_offsets_ref.functions.items():
        actual_func = tir_mod_with_offsets[gv.name_hint]
        tvm.ir.assert_structural_equal(actual_func, ref_func)


def test_simple():
    target = Target("c")
    relax_mod = LinearStructure
    passes = [
        relax.transform.ToNonDataflow(),
        relax.transform.CallTIRRewrite(),
        relax.transform.ConvertRelaxMainToDPS(attach_io_to_attrs=False),
    ]
    seq = tvm.transform.Sequential(passes)
    relax_mod = seq(relax_mod)

    global_workspace_pool = WorkspacePoolInfo(
        "global_workspace",
        [target],
    )
    relax_mod = _assign_targets_to_relaxfuncs_irmodule(relax_mod, target)
    relax_mod = _assign_poolinfos_to_allocates_in_irmodule(relax_mod, [global_workspace_pool])
    relax_mod = relax_mod.with_attr("executor", tvm.relay.backend.Executor("aot"))
    relax_mod = relax_mod.with_attr("runtime", tvm.relay.backend.Runtime("crt"))

    after_mod = tvm.relax.transform.UnifiedStaticMemoryPlanner()(relax_mod)
    print(after_mod)


if __name__ == "__main__":
    pytest.main([__file__] + sys.argv[1:])
