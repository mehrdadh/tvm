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

#include "tvm/relax/expr_functor.h"

namespace tvm {

namespace relax {
namespace usmp {

class ConvertRelaxMainToDPS : public ExprMutator {
 public:
  explicit ConvertRelaxMainToDPS(IRModule mod, bool attach_io_to_attrs)
      : mod_(mod), attach_io_to_attrs_(attach_io_to_attrs) {}

  IRModule operator()() {
    GlobalVar gv = mod_->GetGlobalVar("main");
    auto main_func = Downcast<relax::Function>(mod_->Lookup(gv));
    ICHECK(main_func.defined()) << "main function is not in the module";

    input_vars = main_func->params;

    Expr func_return = get_func_return(main_func);
    if (func_return->IsInstance<relay::TupleNode>()) {
      bool empty = runtime::Downcast<relay::Tuple>(func_return)->fields.empty();
      if (empty) {
        VLOG(0) << "Function " << gv->name_hint << " is already in DPS or returns empty Tuple.";
        return mod_;
      }
    }
    ICHECK(func_return->IsInstance<VarNode>() || func_return->IsInstance<relay::TupleNode>())
        << "Only support Var or Tuple returns for now.";
    {
      if (const VarNode* node = func_return.as<VarNode>()) {
        auto return_var = runtime::GetRef<Var>(node);
        if (node->checked_type()->IsInstance<DynTensorTypeNode>() &&
            !var_in_array(input_vars, return_var)) {
          output_vars.push_back(return_var);
        }
        return_alias_[return_var] = return_var;
      }
      if (const relay::TupleNode* node = func_return.as<relay::TupleNode>()) {
        auto tuple = runtime::GetRef<relay::Tuple>(node);
        return_alias_[tuple] = tuple;
        for (Expr expr : node->fields) {
          ICHECK(expr->checked_type()->IsInstance<DynTensorTypeNode>())
              << "Only support Tuple containing Tensors but got Tuple containing "
              << PrettyPrint(expr->checked_type());
          auto var = runtime::Downcast<Var>(expr);
          if (!var_in_array(input_vars, var)) {
            // If var is not already in the input.
            output_vars.push_back(var);
          }
          return_alias_[var] = var;
        }
      }
    }

    Expr new_body = this->VisitExpr(main_func->body);

    Array<Var> new_params = input_vars;
    new_params.insert(input_vars.end(), output_vars.begin(), output_vars.end());

    Function new_func =
        Function(new_params, new_body, TupleType({}, Span()), RuntimeDepShape(), main_func->attrs);
    if (attach_io_to_attrs_) {
      new_func = WithAttr(new_func, "input_vars", input_vars);
      new_func = WithAttr(new_func, "output_vars", output_vars);
    }

    mod_->Update(gv, new_func);
    return mod_;
  }

 private:
  Expr VisitExpr_(const CallNode* op) override {
    if (op->op->IsInstance<ExternFuncNode>() || op->op->IsInstance<GlobalVarNode>()) {
      // This must be a call to a TIR func.
      Array<Expr> new_args;
      for (Expr arg : op->args) {
        if (arg->IsInstance<VarNode>() && return_alias_.count(runtime::Downcast<Var>(arg)) > 0) {
          new_args.push_back(return_alias_[runtime::Downcast<Var>(arg)]);
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
        if (var_binding->value->IsInstance<VarNode>() &&
            return_alias_.count(var_binding->var) > 0) {
          // Alias. Update alias map and do not emit binding.
          return_alias_[runtime::Downcast<Var>(var_binding->value)] =
              return_alias_[var_binding->var];
          continue;
        }

        if (var_binding->value->IsInstance<relay::TupleNode>() &&
            return_alias_.count(var_binding->var) > 0) {
          // Defining the returned tuple.
          auto tuple = runtime::Downcast<relay::Tuple>(var_binding->value);
          return_alias_[tuple] = return_alias_[var_binding->var];
          for (Expr expr : tuple->fields) {
            ICHECK(expr->IsInstance<VarNode>()) << "Only support Tuple containing Vars but got "
                                                   "Tuple containing "
                                                << PrettyPrint(expr->checked_type());
            auto var = runtime::Downcast<Var>(expr);
            if (!var_in_array(input_vars, var)) {
              output_vars.push_back(var);
            }
            return_alias_[var] = var;
          }
          // Do not emit the tuple declaration.
          continue;
        }

        if (var_binding->value->IsInstance<CallNode>()) {
          static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
          Call call = runtime::Downcast<Call>(var_binding->value);
          if (call->op == alloc_tensor_op && return_alias_.count(var_binding->var) > 0) {
            // This is an alloc tensor for the output. Do not emit the binding.
            continue;
          }
          if (return_alias_.count(var_binding->var) > 0 &&
              !call->op->IsInstance<ExternFuncNode>()) {
            LOG(FATAL) << "Binding the output \"" << var_binding->var->name_hint()
                       << "\" to "
                          "a var allocated outside the function.";
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
      } else if (binding->IsInstance<MatchShapeNode>()) {
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

      // Return of function is empty tuple.
      Expr body = relay::Tuple({}, Span());
      return SeqExpr(blocks, body);
    }
    return runtime::GetRef<SeqExpr>(op);
  }

  static bool var_in_array(const Array<Var>& arr, const Var& var) {
    return (std::find(arr.begin(), arr.end(), var) != arr.end());
  }

  static Expr get_func_return(Function func) {
    const SeqExprNode* seq_expr = func->body.as<SeqExprNode>();
    ICHECK(seq_expr != nullptr) << "Expecting a function with a SeqExpr body";
    return seq_expr->body;
  }

  bool visited_func_body = false;

  Array<Var> input_vars;
  Array<Var> output_vars;
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> return_alias_;

  IRModule mod_;
  bool attach_io_to_attrs_;
};

}  // namespace usmp
}  // namespace relax

namespace transform {

tvm::transform::Pass ConvertRelaxMainToDPS(Bool attach_io_to_attrs) {
  auto pass_func = [=](IRModule m, tvm::transform::PassContext ctx) {
    return relax::usmp::ConvertRelaxMainToDPS(m, attach_io_to_attrs->value != 0)();
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "relax.usmp.ConvertRelaxMainToDPS", {});
}

TVM_REGISTER_GLOBAL("relax.transform.ConvertRelaxMainToDPS").set_body_typed(ConvertRelaxMainToDPS);

}  // namespace transform
}  // namespace tvm
