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
    mod["run_model"] = relax_visitor.visit_expr(mod["run_model"])
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
    mod["run_model"] = relax_visitor.visit_expr(mod["run_model"])

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

# def test_tensor_intrin():
#     target = Target("c")
#     relax_mod = TensorIntrinStructure
#     passes = [relax.transform.ToNonDataflow(), relax.transform.CallTIRRewrite()]
#     seq = tvm.transform.Sequential(passes)
#     relax_mod = seq(relax_mod)
#
#     global_workspace_pool = WorkspacePoolInfo(
#         "global_workspace",
#         [target],
#     )
#     relax_mod = _assign_targets_to_relaxfuncs_irmodule(relax_mod, target)
#
#     relax_mod = _assign_poolinfos_to_allocates_in_irmodule(relax_mod, [global_workspace_pool])
#     main_func = relax_mod["run_model"]
#     buffer_analysis = tvm.relax.analysis.extract_buffer_info(main_func, relax_mod)
#     buffer_info_map = buffer_analysis.buffer_info_stmts
#
#     fcreate_array_bi = tvm.get_global_func("tir.usmp.CreateArrayBufferInfo")
#     buffer_info_arr = fcreate_array_bi(buffer_info_map)
#     fusmp_algo_greedy_by_size = tvm.get_global_func("tir.usmp.algo.greedy_by_size")
#     buffer_pool_allocations = fusmp_algo_greedy_by_size(
#         buffer_info_arr, buffer_analysis.memory_pressure
#     )
#     fassign_stmt_pool_allocations = tvm.get_global_func("tir.usmp.AssignStmtPoolAllocations")
#     pool_allocations = fassign_stmt_pool_allocations(buffer_info_map, buffer_pool_allocations)
#     tir_mod_with_offsets = tvm.relax.transform.ConvertPoolAllocationsToOffsets(
#         pool_allocations, emit_tvmscript_printable=True, insert_storage_allocations=False
#     )(relax_mod)
#
#     tir_mod_with_offsets_ref = TensorIntrinStructurePlanned
#     tir_mod_with_offsets_ref = _append_type_args(tir_mod_with_offsets_ref, ["int32"])
#
#     for gv, ref_func in tir_mod_with_offsets_ref.functions.items():
#         actual_func = tir_mod_with_offsets[gv.name_hint]
#         tvm.ir.assert_structural_equal(actual_func, ref_func)


# fmt: off
@tvm.script.ir_module
class DummyClass:
    # @R.function
    # def main(x: Tensor((5, 7), "float32")) -> Tensor:
    #     lv0 = relax.call_tir("tir_func", (x), (5, 7), dtype="float32")
    #     return lv0

    # @R.function
    # def run_model(x: Tensor((5, 7), "float32")) -> Tensor(None, "float32", ndim = 2):
    #     # block 0
    #     alloc: Tensor((5, 7), "float32") = relax.builtin.alloc_tensor((5, 7), dtype="float32", runtime_device_index=0, attrs_type_key="relax.attrs.AllocTensorAttrs")
    #     _: Tensor(_, "float32", ndim = 2) = R.call_packed("tir_func", x, alloc, type_args=(Tensor(ndim=2, dtype="float32")))
    #     lv0: Tensor((5, 7), "float32") = alloc
    #     return lv0

    # @R.function
    # def run_model(x: Tensor((5, 7), "float32")):
    #     block 0
        # alloc: Tensor((5, 7), "float32") = relax.builtin.alloc_tensor((5, 7), dtype="float32", runtime_device_index=0, attrs_type_key="relax.attrs.AllocTensorAttrs")
        # _: Tensor(_, "float32", ndim = 2) = R.call_packed("tir_func", x, alloc, type_args=(Tensor(ndim=2, dtype="float32")))
        # lv0: Tensor((5, 7), "float32") = alloc
        # return lv0
        # return relax.const(0)
        # return relax.Tuple((lv0, x))

    @R.function
    def run_model(input: Tensor((16, 16), "uint8")) -> Tensor:
        tsid_10 = relax.builtin.alloc_tensor((1, 1), runtime_device_index=0, dtype="int16")
        tsid_11 = relax.builtin.alloc_tensor((9408, 1), runtime_device_index=0, dtype="int16")
        tsid_12 = relax.builtin.alloc_tensor((64, 1), runtime_device_index=0, dtype="int32")

        lv0 = relax.call_tir("tvmgen_default_fused_cast_subtract", (input, tsid_10), (301056, 1), dtype="int32")
        lv1 = relax.call_tir("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast", (lv0, tsid_11, tsid_12), (802816, 1), dtype="int32")
        output = relax.call_tir("tvmgen_default_fused_nn_max_pool2d_cast", (lv1), (16, 16), dtype="int32")
        return output


# @R.function
# def main(x: Tensor((5, 7), "int32")):
#     return relax.const(0)

# fmt: on

def test_dummy_test():
    target = Target("c")
    relax_mod = DummyClass
    # passes = [relax.transform.ToNonDataflow()]
    passes = [relax.transform.ToNonDataflow(), relax.transform.CallTIRRewrite()]
    seq = tvm.transform.Sequential(passes)
    relax_mod = seq(relax_mod)

    relax_mod = tvm.relax.transform.ConvertRelaxToDPS()(relax_mod)
    print(dump_ast(relax_mod["run_model"]))
    print(relax_mod)


if __name__ == "__main__":
    pytest.main([__file__] + sys.argv[1:])
