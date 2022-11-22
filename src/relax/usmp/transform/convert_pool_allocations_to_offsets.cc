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
 * \file tir/analysis/usmp/transform/convert_pool_allocations_to_offsets.cc
 * \brief This pass would convert the pool allocations to offsets from pools
 */

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/usmp/transform.h>
#include <tvm/tir/usmp/utils.h>

#include <stack>

#include "tvm/relax/attrs/memory.h"
#include "tvm/relax/expr_functor.h"
#include "tvm/relax/usmp/utils.h"
#include "tvm/tir/usmp/utils.h"

namespace tvm {

namespace tir::usmp {
class TIRPoolAllocationToOffsetConverter;
}

namespace relax::usmp {
class RelaxPoolAllocationToOffsetConverter;
class RelaxPoolAllocationInserter;
}  // namespace relax::usmp

class PoolAllocationToOffsetConverter;

namespace usmp {

class PoolAllocationsToOffsetsPassData {
  /*! \brief This is a structure where the modified function
   * signature is kept while body of the function is mutated
   */
  struct ScopeInfo {
    // tir::Var or relax::Var.
    Array<BaseExpr> params;
    // tir::Var or relax::Var.
    // Can point to either a tir parameter or a relax var bound to a relax.memory.alloc_storage
    Map<PoolInfo, BaseExpr> pools_to_var;
    Array<tir::usmp::AllocatedPoolInfo> allocated_pools;
    // Only used in TIR.
    Map<tir::Var, tir::Buffer> buffer_map;
  };

  /*! \brief The function scope information that are needed
   * in the mutation of the function need to be stacked and
   * popped when each function is entered/exited in the
   * mutation process.
   */
  std::stack<ScopeInfo> scope_stack;

  /*! \brief The IRModule being constructed/mutated */
  IRModule module_;
  /*! \brief The input allocate node to PoolAllocation map. Can be tir::Stmt or Relax::Expr */
  Map<runtime::ObjectRef, tir::usmp::PoolAllocation> pool_allocations_;
  /*! \brief The set of ordered pools to ensure an unique order of args for functions */
  std::vector<tir::usmp::AllocatedPoolInfo> allocated_pool_ordering_;
  /*! \brief The storage of calculated pool size at init */
  std::unordered_map<PoolInfo, int, ObjectPtrHash, ObjectPtrEqual> all_pools_sizes_;
  /*! \brief After mutation, each allocate buffer is replaced with tir::Var that is let bounded
   * to position from a pool as designated by a PoolAllocation. Only used by the TIR visitor.
   */
  Map<tir::Var, tir::Var> allocate_var_to_let_var_;
  /*! \brief A map from the original buffer object
   *
   * Each key-value pair in this map satisfies
   * ``allocate_buf_to_let_var[key->data] = value->data``.  However,
   * since more than one `tir::Buffer` may use the same Var, they must
   * be tracked separately.
   * Only used by the TIR visitor.
   */
  Map<tir::Buffer, tir::Buffer> original_buf_to_let_buf_;

  Map<String, Bool> signature_has_device_context_;
  /*! \brief A counter to give references to pools a reproducible unique set of names */
  int pool_var_count_ = 0;
  /*! \brief This toggles to remove non tvmscript printable items for IRModule for unit tests */
  bool emit_tvmscript_printable_ = false;

  /*! \brief This controls if pool vars are passed as parameters or allocated with alloc_storage */
  bool insert_storage_allocations_ = true;

  std::unordered_set<BaseFunc, ObjectPtrHash, ObjectPtrEqual> visited_funcs;

  Map<PoolInfo, Array<ConstantInfo>> pool_initializations_;

  void AppdendConstInitializationData(ScopeInfo si) {
    for (tir::usmp::AllocatedPoolInfo api : si.allocated_pools) {
      const auto& it = pool_initializations_.find(api->pool_info);
      if (it != pool_initializations_.end()) {
        auto* pi = const_cast<ConstantPoolInfoNode*>(api->pool_info.as<ConstantPoolInfoNode>());
        pi->constant_info_array = (*it).second;
      }
    }
  }

  friend class tvm::PoolAllocationToOffsetConverter;
  friend class tir::usmp::TIRPoolAllocationToOffsetConverter;
  friend class relax::usmp::RelaxPoolAllocationToOffsetConverter;
  friend class relax::usmp::RelaxPoolAllocationInserter;
};

class PoolAllocationInserterPassData {
 public:
  explicit PoolAllocationInserterPassData(
      const Array<tir::usmp::AllocatedPoolInfo>& allocated_pools)
      : allocated_pools_(allocated_pools) {}

  Array<tir::usmp::AllocatedPoolInfo> allocated_pools_;

  friend class relax::usmp::RelaxPoolAllocationToOffsetConverter;
  friend class relax::usmp::RelaxPoolAllocationInserter;
};

}  // namespace usmp

namespace tir {
namespace usmp {

/*!
 * \brief The StmtExpr mutator class to replace allocate nodes
 * with offsets within memory pools
 *
 * This mutator class will add Pool variables recursively to every PrimFunc
 * starting from the main PrimFunc. For all allocate nodes, that have been
 * memory planned, will be mutated into an offset using a Let binding.
 */
class TIRPoolAllocationToOffsetConverter : public StmtExprMutator {
  using PoolAllocationsToOffsetsPassData = tvm::usmp::PoolAllocationsToOffsetsPassData;
  using ScopeInfo = PoolAllocationsToOffsetsPassData::ScopeInfo;

 public:
  explicit TIRPoolAllocationToOffsetConverter(const PoolAllocationsToOffsetsPassData& pass_data)
      : pass_data_(pass_data) {}

 private:
  PrimExpr VisitExpr_(const CallNode* op) override;
  Stmt VisitStmt_(const AllocateNode* op) override;
  PrimExpr VisitExpr_(const VarNode* op) override;
  PrimExpr VisitExpr_(const BufferLoadNode* op) override;
  Stmt VisitStmt_(const BufferStoreNode* op) override;

  Stmt VisitStmt_(const AllocateConstNode* op) override;
  LetStmt ToLetStmt(const PoolAllocation& pool_allocation, const Var& buffer_var, const Stmt& body);
  /*! \brief Each PrimFunc signature needs to be updated
   * with pool variables. This is a helper function to
   * capture the updated information to ScopeInfo object.
   */
  ScopeInfo UpdateFunctionScopeInfo(const PrimFunc& original_func);
  /*! \brief This is a helper to create the PrimFunc with
   * pool variables that calls the UpdateFunctionScopeInfo
   * inside of it.
   */
  PrimFunc CreatePrimFuncWithPoolParams(const PrimFunc& original_primfunc);
  /*! \brief This is a helper to append the pool args to
   * the callsite of the function.
   */
  Array<PrimExpr> AppendPoolParamsToArgs(Array<PrimExpr> args, bool has_device_context);
  /*! \brief Some arguments that used to be Allocate nodes
   * should be replaced by Let nodes in the pass that loads
   * the space from a pool variable.
   */
  Array<PrimExpr> ReplaceAllocateArgsWithLetArgs(const Array<PrimExpr>& args);
  /*! \brief Obtain a resource handle if its there
   */
  Optional<Var> GetResourceHandle(const PrimFunc& func);
  /*! \brief Get the Buffer object representing the mapped access into
   *  the pool.
   */
  Buffer GetRemappedBuffer(Buffer buf);

  PoolAllocationsToOffsetsPassData pass_data_;

  friend class relax::usmp::RelaxPoolAllocationToOffsetConverter;
};

Optional<Var> TIRPoolAllocationToOffsetConverter::GetResourceHandle(const PrimFunc& func) {
  if (!func->params.empty() &&
      func->buffer_map.find(func->params.back()) == func->buffer_map.end()) {
    return func->params.back();
  }
  return Optional<Var>();
}

tvm::usmp::PoolAllocationsToOffsetsPassData::ScopeInfo
TIRPoolAllocationToOffsetConverter::UpdateFunctionScopeInfo(const PrimFunc& original_func) {
  ScopeInfo si;

  Optional<Var> resource_handle = GetResourceHandle(original_func);
  si.params = Array<BaseExpr>(original_func->params.begin(), original_func->params.end());
  if (resource_handle) {
    si.params.pop_back();
    ICHECK(si.params.size() == original_func->params.size() - 1);
  }
  si.buffer_map = original_func->buffer_map;
  Map<tir::Var, PoolInfo> ret;
  for (const AllocatedPoolInfo& allocated_pool_info : pass_data_.allocated_pool_ordering_) {
    PoolInfo pool_info = allocated_pool_info->pool_info;
    String pool_ref_name =
        pool_info->pool_name + "_" + std::to_string(pass_data_.pool_var_count_++);
    String var_name = pool_ref_name + "_pool";
    DataType elem_dtype = DataType::UInt(8);
    Var buffer_var(var_name, PointerType(PrimType(elem_dtype), "global"));
    Var pool_var = Var(var_name, PointerType(PrimType(elem_dtype), "global"));
    si.params.push_back(pool_var);
    si.pools_to_var.Set(pool_info, pool_var);
    si.allocated_pools.push_back(AllocatedPoolInfo(
        allocated_pool_info->pool_info, allocated_pool_info->allocated_size, si.params.size() - 1));

    int pool_size = pass_data_.all_pools_sizes_[pool_info];
    String buffer_var_name = pool_ref_name + "_buffer_var";
    si.buffer_map.Set(pool_var,
                      Buffer(buffer_var /* data */, elem_dtype /* dtype */, {pool_size} /* shape */,
                             {1} /* strides */, 0 /* elem_offset */, buffer_var_name /* name */,
                             16 /* data_alignment */, 1 /* offset_factor */,
                             BufferType::kDefault /* buffer-type */));
  }
  if (resource_handle) {
    si.params.push_back(resource_handle.value());
  }
  return si;
}

PrimFunc TIRPoolAllocationToOffsetConverter::CreatePrimFuncWithPoolParams(
    const PrimFunc& original_primfunc) {
  // Only create the new function if it was not modified with pool params
  if (pass_data_.visited_funcs.find(original_primfunc) == pass_data_.visited_funcs.end()) {
    ScopeInfo si = UpdateFunctionScopeInfo(original_primfunc);
    pass_data_.scope_stack.push(si);
    Stmt new_body = this->VisitStmt(original_primfunc->body);
    pass_data_.scope_stack.pop();
    DictAttrs original_attrs = original_primfunc->attrs;
    // We dont need attrs of PrimFunc that might include non printable attrs such as target
    // for unit tests where emit_tvmscript_printable_ is to be used.
    if (pass_data_.emit_tvmscript_printable_) {
      original_attrs = DictAttrs();
    }
    Array<Var> params;
    for (BaseExpr base_expr : si.params) {
      params.push_back(runtime::Downcast<Var>(base_expr));
    }
    PrimFunc ret = PrimFunc(params, new_body, original_primfunc->ret_type, si.buffer_map,
                            original_attrs);
    if (!pass_data_.emit_tvmscript_printable_) {
      ret = WithAttr(ret, tvm::attr::kPoolArgs, si.allocated_pools);
    }
    pass_data_.visited_funcs.insert(ret);
    return ret;
  }
  return original_primfunc;
}

Array<PrimExpr> TIRPoolAllocationToOffsetConverter::AppendPoolParamsToArgs(
    Array<PrimExpr> args, bool has_device_context) {
  Array<PrimExpr> new_args;
  PrimExpr resource_handle_arg;
  // name, params...params[, context]
  if (has_device_context) {
    resource_handle_arg = args.back();
    args.pop_back();
  }
  for (const auto& arg : args) {
    new_args.push_back(VisitExpr(arg));
  }
  ScopeInfo top_scope = pass_data_.scope_stack.top();
  for (const auto& pools_vars : top_scope.pools_to_var) {
    Var pool_var = runtime::Downcast<Var>(pools_vars.second);
    Buffer buffer_var = top_scope.buffer_map[pool_var];
    new_args.push_back(buffer_var->data);
  }
  if (resource_handle_arg.defined()) {
    new_args.push_back(resource_handle_arg);
  }
  return new_args;
}

Array<PrimExpr> TIRPoolAllocationToOffsetConverter::ReplaceAllocateArgsWithLetArgs(
    const Array<PrimExpr>& args) {
  Array<PrimExpr> ret;
  for (const PrimExpr& arg : args) {
    if (arg->IsInstance<VarNode>() && pass_data_.allocate_var_to_let_var_.find(Downcast<Var>(
                                          arg)) != pass_data_.allocate_var_to_let_var_.end()) {
      ret.push_back(
          runtime::Downcast<PrimExpr>(pass_data_.allocate_var_to_let_var_[Downcast<Var>(arg)]));
    } else {
      ret.push_back(VisitExpr(arg));
    }
  }
  return ret;
}

PrimExpr TIRPoolAllocationToOffsetConverter::VisitExpr_(const CallNode* op) {
  if (op->op.same_as(builtin::call_extern()) || op->op.same_as(builtin::tvm_call_cpacked())) {
    String func_name = Downcast<StringImm>(op->args[0])->value;
    Array<PrimExpr> new_args;
    if (pass_data_.module_->ContainGlobalVar(func_name) &&
        pass_data_.module_->Lookup(func_name)->IsInstance<PrimFuncNode>()) {
      GlobalVar gv = pass_data_.module_->GetGlobalVar(func_name);
      PrimFunc func = Downcast<PrimFunc>(pass_data_.module_->Lookup(gv));

      if (!pass_data_.signature_has_device_context_.count(func_name)) {
        if (op->args.size() == func->params.size() + 2) {
          pass_data_.signature_has_device_context_.Set(func_name, Bool(true));
        } else {
          pass_data_.signature_has_device_context_.Set(func_name, Bool(false));
        }
      }

      PrimFunc prim_func = CreatePrimFuncWithPoolParams(func);
      pass_data_.module_->Update(gv, prim_func);
      new_args =
          AppendPoolParamsToArgs(op->args, pass_data_.signature_has_device_context_[func_name]);
      new_args = ReplaceAllocateArgsWithLetArgs(new_args);
    } else {
      new_args = ReplaceAllocateArgsWithLetArgs(op->args);
    }
    return Call(op->dtype, op->op, new_args);
  }
  if (op->op->IsInstance<PrimFuncNode>()) {
    String func_name = Downcast<StringImm>(op->args[0])->value;
    PrimFunc func = Downcast<PrimFunc>(op->op);
    PrimFunc prim_func = CreatePrimFuncWithPoolParams(func);
    Array<PrimExpr> new_args =
        AppendPoolParamsToArgs(op->args, pass_data_.signature_has_device_context_[func_name]);
    new_args = ReplaceAllocateArgsWithLetArgs(new_args);
    return Call(op->dtype, prim_func, new_args);
  }
  return StmtExprMutator::VisitExpr_(op);
}

LetStmt TIRPoolAllocationToOffsetConverter::ToLetStmt(const PoolAllocation& pool_allocation,
                                                      const Var& buffer_var, const Stmt& body) {
  ScopeInfo scope_info = pass_data_.scope_stack.top();
  Var param = runtime::Downcast<Var>(scope_info.pools_to_var[pool_allocation->pool_info]);
  BufferLoad load_node = BufferLoad(scope_info.buffer_map[param], {pool_allocation->byte_offset});
  Call address_of_load = Call(DataType::Handle(), builtin::address_of(), {load_node});

  Type let_var_type = buffer_var->type_annotation;
  if (pass_data_.emit_tvmscript_printable_) {
    // Strip the storage_scope from the variable type, as TVMScript
    // doesn't parsethe scoped pointers (e.g. ``T.Ptr[global T.int32]``)
    // correctly.
    let_var_type = PointerType(Downcast<PointerType>(let_var_type)->element_type);
  }
  Var let_var(buffer_var->name_hint + "_let", let_var_type);
  pass_data_.allocate_var_to_let_var_.Set(buffer_var, let_var);
  Stmt new_body = VisitStmt(body);
  pass_data_.allocate_var_to_let_var_.erase(buffer_var);
  return LetStmt(let_var, address_of_load, new_body);
}

Stmt TIRPoolAllocationToOffsetConverter::VisitStmt_(const AllocateNode* op) {
  if (pass_data_.pool_allocations_.count(GetRef<Allocate>(op))) {
    return ToLetStmt(pass_data_.pool_allocations_[GetRef<Stmt>(op)], op->buffer_var, op->body);
  }
  return StmtExprMutator::VisitStmt_(op);
}

Stmt TIRPoolAllocationToOffsetConverter::VisitStmt_(const AllocateConstNode* op) {
  if (pass_data_.pool_allocations_.count(GetRef<AllocateConst>(op))) {
    const auto& result =
        ToLetStmt(pass_data_.pool_allocations_[GetRef<Stmt>(op)], op->buffer_var, op->body);

    PoolInfo pool_info = pass_data_.pool_allocations_[GetRef<Stmt>(op)]->pool_info;
    if (pass_data_.pool_initializations_.find(pool_info) ==
        pass_data_.pool_initializations_.end()) {
      pass_data_.pool_initializations_.Set(pool_info, {});
    }

    auto consts = pass_data_.pool_initializations_[pool_info];
    consts.push_back({result->var->name_hint,
                      pass_data_.pool_allocations_[GetRef<Stmt>(op)]->byte_offset,
                      op->data.value()});

    pass_data_.pool_initializations_.Set(pool_info, consts);
    return result;
  }
  return StmtExprMutator::VisitStmt_(op);
}

Stmt TIRPoolAllocationToOffsetConverter::VisitStmt_(const BufferStoreNode* op) {
  BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));

  Buffer remapped = GetRemappedBuffer(store->buffer);
  if (!op->buffer.same_as(remapped)) {
    store.CopyOnWrite()->buffer = remapped;
  }
  return std::move(store);
}

PrimExpr TIRPoolAllocationToOffsetConverter::VisitExpr_(const BufferLoadNode* op) {
  BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));

  Buffer remapped = GetRemappedBuffer(load->buffer);
  if (!op->buffer.same_as(remapped)) {
    load.CopyOnWrite()->buffer = remapped;
  }
  return std::move(load);
}

PrimExpr TIRPoolAllocationToOffsetConverter::VisitExpr_(const VarNode* op) {
  auto it = pass_data_.allocate_var_to_let_var_.find(GetRef<Var>(op));
  if (it != pass_data_.allocate_var_to_let_var_.end()) {
    return (*it).second;
  }

  return StmtExprMutator::VisitExpr_(op);
}

Buffer TIRPoolAllocationToOffsetConverter::GetRemappedBuffer(Buffer original) {
  {
    auto it = pass_data_.original_buf_to_let_buf_.find(original);
    if (it != pass_data_.original_buf_to_let_buf_.end()) {
      return (*it).second;
    }
  }

  Buffer remapped = original;

  auto it = pass_data_.allocate_var_to_let_var_.find(original->data);
  if (it != pass_data_.allocate_var_to_let_var_.end()) {
    Var var = runtime::Downcast<Var>((*it).second);
    remapped =
        Buffer(var, original->dtype, original->shape, original->strides, original->elem_offset,
               original->name, original->data_alignment, original->offset_factor,
               original->buffer_type, original->axis_separators, original->span);
  }

  pass_data_.original_buf_to_let_buf_.Set(original, remapped);
  return remapped;
}

}  // namespace usmp
}  // namespace tir

namespace relax {
namespace usmp {

class RelaxPoolAllocationToOffsetConverter : public relax::ExprMutator {
  using PoolAllocationsToOffsetsPassData = tvm::usmp::PoolAllocationsToOffsetsPassData;
  using ScopeInfo = PoolAllocationsToOffsetsPassData::ScopeInfo;

 public:
  explicit RelaxPoolAllocationToOffsetConverter(const PoolAllocationsToOffsetsPassData& pass_data)
      : pass_data_(pass_data) {}

  std::pair<IRModule, tvm::usmp::PoolAllocationInserterPassData> operator()();

 private:
  Expr VisitExpr_(const CallNode* op) override;
  void VisitBinding_(const VarBindingNode* binding);

  /*! \brief Each Relax function signature needs to be updated
   * with pool variables. This is a helper function to
   * capture the updated information to ScopeInfo object.
   */
  ScopeInfo UpdateFunctionScopeInfo(const Function& original_func);
  /*! \brief This is a helper to append the pool args to
   * the callsite of the function.
   */
  Array<Expr> AppendPoolParamsToArgs(Array<Expr> args);

  // Call to bound var map used to find the PoolAllocations in pool_allocations_.
  Map<tvm::relay::Call, tvm::relax::Var> call_to_bound_var_;

  PoolAllocationsToOffsetsPassData pass_data_;
};

Array<Expr> RelaxPoolAllocationToOffsetConverter::AppendPoolParamsToArgs(Array<Expr> args) {
  Array<Expr> new_args;
  for (const auto& arg : args) {
    new_args.push_back(VisitExpr(arg));
  }
  ScopeInfo top_scope = pass_data_.scope_stack.top();
  for (const auto& pools_vars : top_scope.pools_to_var) {
    Var pool_var = runtime::Downcast<Var>(pools_vars.second);
    new_args.push_back(pool_var);
  }
  return new_args;
}

Expr RelaxPoolAllocationToOffsetConverter::VisitExpr_(const CallNode* op) {
  auto node = GetRef<Call>(op);
  static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
  if (op->op == alloc_tensor_op) {
    if (pass_data_.pool_allocations_.count(call_to_bound_var_.Get(node).value())) {
      auto pool_allocation = pass_data_.pool_allocations_[call_to_bound_var_.Get(node).value()];
      static const Op& memory_alloc_tensor_op = Op::Get("relax.memory.alloc_tensor");
      auto attrs = make_object<MemAllocTensorAttrs>();
      attrs->offset = pool_allocation->byte_offset->value;
      attrs->dtype = op->attrs.as<AllocTensorAttrs>()->dtype;
      auto scope_info = pass_data_.scope_stack.top();
      auto storage = scope_info.pools_to_var[pool_allocation->pool_info];
      return Call(
          memory_alloc_tensor_op,
          {GetRef<Var>(storage.as<VarNode>()), call_to_bound_var_.Get(node).value()->shape()},
          Attrs(attrs), {});
    }
    return ExprMutator::VisitExpr_(op);
  }

  // Rewrite call to TIR PrimFunc
  if (op->op->IsInstance<ExternFuncNode>()) {
    String func_name = runtime::Downcast<ExternFunc>(op->op)->global_symbol;
    Array<Expr> new_args;
    if (pass_data_.module_->ContainGlobalVar(func_name) &&
        pass_data_.module_->Lookup(func_name)->IsInstance<tir::PrimFuncNode>()) {
      GlobalVar gv = pass_data_.module_->GetGlobalVar(func_name);
      tir::PrimFunc prim_func = runtime::Downcast<tir::PrimFunc>(pass_data_.module_->Lookup(gv));

      tir::usmp::TIRPoolAllocationToOffsetConverter tir_offset_converter =
          tir::usmp::TIRPoolAllocationToOffsetConverter(pass_data_);
      tir::PrimFunc new_prim_func = tir_offset_converter.CreatePrimFuncWithPoolParams(prim_func);
      pass_data_.module_->Update(gv, new_prim_func);

      new_args = AppendPoolParamsToArgs(op->args);
    } else {
      new_args = Array<Expr>(op->args.begin(), op->args.end());
    }
    return Call(op->op, new_args, op->attrs, op->type_args, op->span);
  }
  if (op->op->IsInstance<relax::FunctionNode>()) {
    auto func = Downcast<relax::Function>(op->op);
    ICHECK(false) << "Calls to Relax functions are not supported." << PrettyPrint(func);
  }
  if (op->op->IsInstance<GlobalVarNode>()) {
    auto global_var = Downcast<GlobalVar>(op->op);
    ICHECK(false) << "Calls to Relax functions are not supported: " << global_var->name_hint;
  }
  return ExprMutator::VisitExpr_(op);
}

void RelaxPoolAllocationToOffsetConverter::VisitBinding_(const VarBindingNode* binding) {
  auto node = GetRef<VarBinding>(binding);
  if (node->value->IsInstance<CallNode>()) {
    auto call_node = runtime::Downcast<Call>(node->value);
    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    if (call_node->op == alloc_tensor_op) {
      call_to_bound_var_.Set(call_node, binding->var);
    }
  }
  ExprMutator::VisitBinding_(binding);
}

tvm::usmp::PoolAllocationsToOffsetsPassData::ScopeInfo
RelaxPoolAllocationToOffsetConverter::UpdateFunctionScopeInfo(const Function& original_func) {
  ScopeInfo si;
  si.params = Array<BaseExpr>(original_func->params.begin(), original_func->params.end());
  using AllocatedPoolInfo = tir::usmp::AllocatedPoolInfo;
  for (const AllocatedPoolInfo& allocated_pool_info : pass_data_.allocated_pool_ordering_) {
    PoolInfo pool_info = allocated_pool_info->pool_info;
    int pool_size = pass_data_.all_pools_sizes_[pool_info];
    String pool_ref_name =
        pool_info->pool_name + "_" + std::to_string(pass_data_.pool_var_count_++);
    String var_name = pool_ref_name + "_pool";
    DataType elem_dtype = DataType::Int(64);
    IntImm shape_value = IntImm(elem_dtype, pool_size);
    Var pool_var = Var(var_name, {}, ObjectType());
    si.params.push_back(pool_var);
    si.pools_to_var.Set(pool_info, pool_var);
    si.allocated_pools.push_back(AllocatedPoolInfo(
        allocated_pool_info->pool_info, allocated_pool_info->allocated_size, si.params.size() - 1));
  }
  return si;
}

std::pair<IRModule, tvm::usmp::PoolAllocationInserterPassData>
RelaxPoolAllocationToOffsetConverter::operator()() {
  GlobalVar gv = pass_data_.module_->GetGlobalVar("run_model");
  auto main_func = Downcast<relax::Function>(pass_data_.module_->Lookup(gv));
  ScopeInfo si = UpdateFunctionScopeInfo(main_func);
  pass_data_.scope_stack.push(si);
  Expr main_func_body = this->VisitExpr(main_func->body);
  pass_data_.scope_stack.pop();
  pass_data_.AppdendConstInitializationData(si);
  Array<Var> params;
  for (BaseExpr base_expr : si.params) {
    params.push_back(runtime::Downcast<Var>(base_expr));
  }
  // We dont need attrs of PrimFunc that might include non printable attrs such as target
  // for unit tests where emit_tvmscript_printable_ is to be used.
  if (!pass_data_.emit_tvmscript_printable_) {
    main_func = Function(params, main_func_body, main_func->ret_type, main_func->ret_shape,
                         main_func->attrs, main_func->span);
    main_func = WithAttr(main_func, tvm::attr::kPoolArgs, si.allocated_pools);
  } else {
    main_func = Function(params, main_func_body, main_func->ret_type, main_func->ret_shape,
                         DictAttrs(), main_func->span);
    main_func = WithAttr(main_func, tvm::attr::kGlobalSymbol, String("run_model"));
  }
  pass_data_.module_->Update(gv, main_func);
  if (!pass_data_.emit_tvmscript_printable_) {
    return {WithAttr(pass_data_.module_, tvm::attr::kPoolArgs, si.allocated_pools),
            tvm::usmp::PoolAllocationInserterPassData(si.allocated_pools)};
  }
  return {pass_data_.module_, tvm::usmp::PoolAllocationInserterPassData(si.allocated_pools)};
}

class RelaxPoolAllocationInserter : public relax::ExprMutator {
  using PoolAllocationsToOffsetsPassData = tvm::usmp::PoolAllocationsToOffsetsPassData;
  using PoolAllocationInserterPassData = tvm::usmp::PoolAllocationInserterPassData;

 public:
  explicit RelaxPoolAllocationInserter(
      const PoolAllocationsToOffsetsPassData& pass_data,
      const PoolAllocationInserterPassData& pool_allocations_to_offsets_pass_data)
      : pass_data_(pass_data),
        pool_allocation_inserter_pass_data_(pool_allocations_to_offsets_pass_data) {}

  IRModule operator()() {
    GlobalVar gv = pass_data_.module_->GetGlobalVar("run_model");
    auto main_func = Downcast<relax::Function>(pass_data_.module_->Lookup(gv));

    for (const tir::usmp::AllocatedPoolInfo& allocated_pool_info :
         pool_allocation_inserter_pass_data_.allocated_pools_) {
      ICHECK(allocated_pool_info->pool_var_idx.defined())
          << "The pool var parameter index should be defined at this point.";
      pool_params_.push_back(
          main_func->params[allocated_pool_info->pool_var_idx.value().IntValue()]);
    }
    Array<Var> func_params;
    for (const Var param : main_func->params) {
      if (std::find(this->pool_params_.begin(), this->pool_params_.end(), param) ==
          this->pool_params_.end()) {
        func_params.push_back(param);
      }
    }

    Expr main_func_body = this->VisitExpr(main_func->body);

    main_func = Function(func_params, main_func_body, main_func->ret_type, main_func->ret_shape,
                         main_func->attrs, main_func->span);
    main_func = WithAttr(main_func, tvm::attr::kPoolArgs, {});
    pass_data_.module_->Update(gv, main_func);
    return WithAttr(pass_data_.module_, tvm::attr::kPoolArgs, {});
  }

 private:
  Expr VisitExpr_(const SeqExprNode* op) override {
    auto allocated_pool_infos = pool_allocation_inserter_pass_data_.allocated_pools_;
    if (!allocated_pool_infos.empty()) {
      Array<BindingBlock> blocks;
      builder_->BeginBindingBlock();
      int index = 0;
      for (const tir::usmp::AllocatedPoolInfo& allocated_pool_info : allocated_pool_infos) {
        Call alloc_storage_call = build_alloc_storage(allocated_pool_info);
        Var pool_var = pool_params_[index];
        builder_->Emit(VarBinding(pool_var, alloc_storage_call));
        index++;
      }
      blocks.push_back(builder_->EndBlock());
      blocks.insert(blocks.end(), op->blocks.begin(), op->blocks.end());
      return SeqExpr(blocks, op->body, op->span);
    }
    return runtime::GetRef<Expr>(op);
  }

  Call build_alloc_storage(const tir::usmp::AllocatedPoolInfo& allocated_pool_info) const {
    static const Op& alloc_storage_op = Op::Get("relax.memory.alloc_storage");
    auto attrs = runtime::make_object<MemAllocStorageAttrs>();
    attrs->dtype = DataType::UInt(8);
    attrs->pool_info_name = allocated_pool_info->pool_info->pool_name;
    int32_t pool_size = allocated_pool_info->allocated_size.IntValue();
    Call alloc_storage_call = Call(alloc_storage_op, {ShapeExpr({ PrimExpr(pool_size) })}, Attrs(attrs), {}, Span());
    return alloc_storage_call;
  }

  Array<Var> pool_params_;
  PoolAllocationsToOffsetsPassData pass_data_;
  PoolAllocationInserterPassData pool_allocation_inserter_pass_data_;
};

}  // namespace usmp
}  // namespace relax

class PoolAllocationToOffsetConverter {
  using PoolAllocationsToOffsetsPassData = usmp::PoolAllocationsToOffsetsPassData;
  using PoolAllocation = tir::usmp::PoolAllocation;
  using AllocatedPoolInfo = tir::usmp::AllocatedPoolInfo;

 public:
  explicit PoolAllocationToOffsetConverter(
      const IRModule& module,
      const Map<runtime::ObjectRef, tir::usmp::PoolAllocation>& pool_allocations,
      bool emit_tvmscript_printable = false, bool insert_storage_allocations = true) {
    pass_data_.pool_allocations_ = pool_allocations;
    pass_data_.emit_tvmscript_printable_ = emit_tvmscript_printable;
    pass_data_.insert_storage_allocations_ = insert_storage_allocations;
    pass_data_.module_ = module->ShallowCopy();
    for (const auto& kv : pass_data_.pool_allocations_) {
      size_t extent_size = -1;
      if (kv.first->IsInstance<relax::VarNode>()) {
        relax::Var var_node = Downcast<relax::Var>(kv.first);
        ICHECK(var_node->checked_type()->IsInstance<relax::DynTensorTypeNode>())
            << "Expected a dynamic tensor type object";
        auto dyn_tensor_type = runtime::Downcast<relax::DynTensorType>(var_node->checked_type());
        ICHECK(var_node->shape()->IsInstance<relax::ShapeExprNode>()) << "Expected a ShapeExpr";
        auto shape_expr = runtime::Downcast<relax::ShapeExpr>(var_node->shape());
        extent_size =
            relax::usmp::CalculateRelaxExtentsSize(dyn_tensor_type->dtype, shape_expr->values)
                .IntValue();
      } else if (kv.first->IsInstance<tir::AllocateNode>()) {
        tir::Allocate allocate_node = Downcast<tir::Allocate>(kv.first);
        extent_size = tir::usmp::CalculateExtentsSize(allocate_node.operator->()).IntValue();
      } else if (kv.first->IsInstance<tir::AllocateConstNode>()) {
        tir::AllocateConst allocate_const_node = Downcast<tir::AllocateConst>(kv.first);
        extent_size = tir::usmp::CalculateExtentsSize(allocate_const_node.operator->()).IntValue();
      } else {
        ICHECK(false) << "Not supported node type " << kv.first->GetTypeKey();
      }
      PoolAllocation pool_allocation = kv.second;
      PoolInfo pool_info = pool_allocation->pool_info;
      int byte_pool_offset = pool_allocation->byte_offset->value;
      int required_pool_size_for_allocation = byte_pool_offset + extent_size;
      if (pass_data_.all_pools_sizes_.find(pool_info) == pass_data_.all_pools_sizes_.end()) {
        pass_data_.all_pools_sizes_[pool_info] = required_pool_size_for_allocation;
      } else {
        int prev_required_pool_size = pass_data_.all_pools_sizes_[pool_info];
        if (prev_required_pool_size < required_pool_size_for_allocation) {
          pass_data_.all_pools_sizes_[pool_info] = required_pool_size_for_allocation;
        }
      }
    }

    for (const auto& kv : pass_data_.all_pools_sizes_) {
      PoolInfo pi = kv.first;
      int allocated_size = kv.second;
      pass_data_.allocated_pool_ordering_.push_back(AllocatedPoolInfo(pi, allocated_size));
    }
    std::sort(pass_data_.allocated_pool_ordering_.begin(),
              pass_data_.allocated_pool_ordering_.end(),
              [](const AllocatedPoolInfo& lhs, const AllocatedPoolInfo& rhs) {
                if (lhs->pool_info->pool_name < rhs->pool_info->pool_name) {
                  return true;
                }
                return false;
              });
  }

  IRModule operator()();

 private:
  PoolAllocationsToOffsetsPassData pass_data_;
};

IRModule PoolAllocationToOffsetConverter::operator()() {
  relax::usmp::RelaxPoolAllocationToOffsetConverter relax_pool_allocation =
      relax::usmp::RelaxPoolAllocationToOffsetConverter(pass_data_);
  std::pair<IRModule, tvm::usmp::PoolAllocationInserterPassData> pair = relax_pool_allocation();

  if (pass_data_.insert_storage_allocations_) {
    relax::usmp::RelaxPoolAllocationInserter relax_pool_allocation_inserter =
        relax::usmp::RelaxPoolAllocationInserter(pass_data_, pair.second);
    IRModule module = relax_pool_allocation_inserter();
    return module;
  }
  return pair.first;
}

namespace transform {

tvm::transform::Pass ConvertPoolAllocationsToOffsets(
    const Map<runtime::ObjectRef, tir::usmp::PoolAllocation>& pool_allocations,
    Bool emit_tvmscript_printable, Bool insert_storage_allocations) {
  auto pass_func = [=](IRModule m, tvm::transform::PassContext ctx) {
    return Downcast<IRModule>(
        PoolAllocationToOffsetConverter(m, pool_allocations, emit_tvmscript_printable->value != 0,
                                        insert_storage_allocations->value != 0)());
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "relax.usmp.ConvertPoolAllocationsToOffsets", {});
}

TVM_REGISTER_GLOBAL("relax.transform.ConvertPoolAllocationsToOffsets")
    .set_body_typed(ConvertPoolAllocationsToOffsets);

}  // namespace transform
}  // namespace tvm
