/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file src/relax/backend/aot/aot_memory_lower.cc
 * \brief Perform memory lowering. Lowers the relax.builtin.alloc_tensor intrinsic to
 * relax.memory.alloc_storage + relax.memory.alloc_tensor.
 */
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/backend.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>

#include "../../../relay/transforms/pattern_utils.h"

namespace tvm {
namespace relax {

// ==================
// MemLowerMutator
// Lower the relax.builtin.alloc_tensor op to relax.memory builtin functions.
// Example:
// x = relax.builtin.alloc_tensor((m, n), relax.attrs.AllocTensorAttrs)
// -->
// gv0 = relax.memory.alloc_storage(m * n * dtype, relax.attrs.MemAllocStorageAttrs)
// gv1 = relax.memory.alloc_tensor(gv0, (m, n), relax.attrs.MemAllocTensorAttrs)

class AOTMemLowerMutator : public ExprMutator {

  // TODO(gigiblender): Dedup this function with the one in VMMemoryLower.
  Expr ComputeStorageSize(const Expr& shape, const DataType& dtype) const {
    // Question: what if the dtype of tensor_type is unknown?
    // Symbolic/static shape case
    if (auto* shape_expr = shape.as<ShapeExprNode>()) {
      PrimExpr num = PrimExpr(dtype.bits()) * PrimExpr(dtype.lanes());
      PrimExpr add = num + 7;
      PrimExpr ret = 1;
      for (PrimExpr dim : shape_expr->values) {
        ret = ret * dim;
      }
      ret = ret * (add / PrimExpr(8));
      return ShapeExpr({ret});
    }
    // Fully dynamic shape case will need to dedup with ComputeStorageInRelay when we upstream
    Expr prod = relay::Prod(shape, Array<Integer>(nullptr), false, false);
    Expr num = relay::MakeConstantScalar(DataType::Int(64), dtype.bits() * dtype.lanes());
    Expr add = relay::Add(num, relay::MakeConstantScalar(DataType::Int(64), 7));
    Expr div = relay::MakeConstantScalar(DataType::Int(64), 8);
    Expr ret = relay::Multiply(prod, relay::Divide(add, div));
    return ret;
  }

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* call) override {
    // post-order mutation
    Expr expr = VisitExprPostOrder_(call);
    call = expr.as<CallNode>();

    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    static const Op& memory_alloc_storage_op = Op::Get("relax.memory.alloc_storage");
    static const Op& memory_alloc_tensor_op = Op::Get("relax.memory.alloc_tensor");
    if (call->op == alloc_tensor_op) {
      ShapeExpr output_shape = Downcast<ShapeExpr>(call->args[0]);
      auto alloc_attrs = call->attrs.as<AllocTensorAttrs>();
      ICHECK(alloc_attrs != nullptr) << "must be AllocTensorAttrs";
      DataType dtype = alloc_attrs->dtype;
      Expr storage_size = ComputeStorageSize(output_shape, dtype);
      auto storage_attr = make_object<MemAllocStorageAttrs>();
      storage_attr->dtype = dtype;

      Var storage =
          builder_->Emit(Call(memory_alloc_storage_op, {storage_size}, Attrs(storage_attr)),
                         "storage");
      auto tensor_attr = make_object<MemAllocTensorAttrs>();
      tensor_attr->offset = 0;
      tensor_attr->dtype = dtype;
      Expr shape = call->args[0];
      return Call(memory_alloc_tensor_op, {storage, shape}, Attrs(tensor_attr));
    }

    return GetRef<Expr>(call);
  }
};

Expr AOTMemoryLower(const Expr& e) { return AOTMemLowerMutator().VisitExpr(e); }

namespace transform {

Pass AOTMemoryLower() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(AOTMemoryLower(f)); };
  return CreateFunctionPass(pass_func, 0, "AOTMemoryLower", {});
}

TVM_REGISTER_GLOBAL("relax.transform.AOTMemoryLower").set_body_typed(AOTMemoryLower);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
