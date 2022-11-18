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

#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/usmp/utils.h>

#include <string>
#include <utility>

#include "tvm/relax/attrs/memory.h"
#include "tvm/relax/expr_functor.h"

namespace tvm {

/*! \brief Assign PoolInfo objects to allocate that does not have any.
* The schedulers have the oppurtunity to assign PoolInfo objects to
* allocate nodes. However, each allocate node is expected to have
* at least one PoolInfo node assigned to it. If it was not the case,
* this Pass will assign all PoolInfo objects that the target could
* access.*/

namespace relax {
namespace usmp {

class ConvertRelaxToDPS : public ExprMutator {
 public:
  explicit ConvertRelaxToDPS(IRModule mod) : mod_(mod) {}

  IRModule operator()() {
    GlobalVar gv = mod_->GetGlobalVar("run_model");
    auto main_func = Downcast<relax::Function>(mod_->Lookup(gv));
    ICHECK(main_func.defined()) << "main function is not in the module";

    func_return = get_func_return(main_func);
    ICHECK(func_return.as<VarNode>() != nullptr) << "Only support Var returns for now.";
    {
      auto func_return_var = runtime::Downcast<Var>(func_return);
      output_vars.push_back(func_return_var);
      alias_[func_return_var] = func_return_var;
    }

    Array<Var> input_vars = main_func->params;
    Expr new_body = this->VisitExpr(main_func->body);

    Array<Var> new_params = input_vars;
    new_params.insert(input_vars.end(), output_vars.begin(), output_vars.end());

    Function new_func = Function(new_params, new_body,
                                 DynTensorType(0, DataType::Int(32)),
                                 RuntimeDepShape(), main_func->attrs);
    new_func = WithAttr(new_func, "input_vars", input_vars);
    new_func = WithAttr(new_func, "output_vars", output_vars);

    mod_->Update(gv, new_func);
    return mod_;
  }

 private:
  Expr VisitExpr_(const CallNode* op) override {
    if (op->op->IsInstance<ExternFuncNode>()) {
      // This is a call_packed call.
      Array<Expr> new_args;
      for (Expr arg : op->args) {
        if (arg->IsInstance<VarNode>() && alias_.count(runtime::Downcast<Var>(arg)) > 0) {
            new_args.push_back(alias_[runtime::Downcast<Var>(arg)]);
          } else {
            new_args.push_back(arg);
          }
      }
      return Call(op->op, new_args, op->attrs, op->type_args, op->span);
    }
    return runtime::GetRef<Call>(op);
  }

  BindingBlock VisitBindingBlock(const BindingBlock& block) override {
    builder_->BeginBindingBlock();
    Array<VarBinding> bindings_reverse;
    for (auto iter = block->bindings.rbegin(); iter != block->bindings.rend(); iter++) {
      Binding binding = *iter;
      if (const auto* var_binding = binding.as<VarBindingNode>()) {
        if (var_binding->var.same_as(func_return) && var_binding->value->IsInstance<VarNode>()) {
          // Alias. Update alias map and do not emit binding.
          alias_[runtime::Downcast<Var>(var_binding->value)] = alias_[var_binding->var];
          continue;
        }

        if (var_binding->value->IsInstance<CallNode>()) {
          static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
          Call call = runtime::Downcast<Call>(var_binding->value);
          if (call->op == alloc_tensor_op && alias_.count(var_binding->var) > 0 &&
              alias_[var_binding->var].same_as(func_return)) {
            // This is an alloc tensor for the output. Do not emit the binding.
            continue;
          }
        }

        Expr new_value = this->VisitExpr(var_binding->value);
        Var new_var = this->VisitVarDef(var_binding->var);
        Var temp = WithShapeAndType(new_var, new_value->shape_, new_value->checked_type_);
        if (!temp.same_as(new_var)) {
          new_var = temp;
          this->var_remap_[var_binding->var->vid] = new_var;
        }

        bindings_reverse.push_back(VarBinding(new_var, new_value));
      } else if (const auto* node = binding.as<MatchShapeNode>()) {
        LOG(FATAL) << "MatchShape bindings are not supporter for now.";
      } else {
        LOG(FATAL) << "TypeError: Invalid type: " << binding->GetTypeKey();
      }
    }

    for (auto iter = bindings_reverse.rbegin(); iter != bindings_reverse.rend(); iter++) {
      this->builder_->Emit(*iter);
    }

    return builder_->EndBlock();
  }

  Expr VisitExpr_(const SeqExprNode* op) override {
    if (!visited_func_body) {
      visited_func_body = true;

      Array<BindingBlock> reverse_blocks;
      for (auto iter = op->blocks.rbegin(); iter != op->blocks.rend(); iter++) {
        BindingBlock new_block = this->VisitBindingBlock(*iter);
        if (!new_block->bindings.empty()) {
          reverse_blocks.push_back(new_block);
        }
      }
      Array<BindingBlock> blocks;
      for (auto iter = reverse_blocks.rbegin(); iter != reverse_blocks.rend(); iter++) {
        blocks.push_back(*iter);
      }

//      builder_->BeginBindingBlock();
      Expr body = build_return_const();
//      BindingBlock prologue = builder_->EndBlock();
//      if (!prologue->bindings.empty()) {
//        blocks.push_back(prologue);
//      }
      return SeqExpr(blocks, body);
    }
    return runtime::GetRef<SeqExpr>(op);
  }


  Expr get_func_return(Function func) {
    const SeqExprNode* seq_expr = func->body.as<SeqExprNode>();
    ICHECK(seq_expr != nullptr) << "Expecting a function with a SeqExpr body";
    return seq_expr->body;
  }

  relay::Constant build_return_const() {
    auto value = runtime::NDArray::Empty({}, DataType::Int(32), {kDLCPU, 0});
    auto zero_value = 0;
    value.CopyFromBytes(&zero_value, sizeof(0));
    auto constant = relay::Constant(value);
    return constant;
  }

  bool visited_func_body = false;

  Expr func_return;
  Array<Var> output_vars;

  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> alias_;

  IRModule mod_;
};

}  // namespace usmp
}  // namespace relax

namespace transform {

tvm::transform::Pass ConvertRelaxToDPS() {
 auto pass_func = [=](IRModule m, tvm::transform::PassContext ctx) {
   return relax::usmp::ConvertRelaxToDPS(m)();
 };
 return tvm::transform::CreateModulePass(pass_func, 0, "relax.usmp.ConvertRelaxToDPS", {});
}

TVM_REGISTER_GLOBAL("relax.transform.ConvertRelaxToDPS").set_body_typed(ConvertRelaxToDPS);

}  // namespace transform
}  // namespace tvm
