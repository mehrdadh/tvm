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

#include <tvm/ir/global_var_supply.h>
#include <tvm/ir/name_supply.h>
#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/usmp/utils.h>

#include <string>

#include "tvm/relax/expr_functor.h"

namespace tvm {

/*! \brief Make sure that there are not multiple calls that call into the same TIR function.
 * If such a situation occurs, this is fixed by duplicating the TIR function in the IRModule
 * and then patching the call site.
 */

namespace relax {
namespace usmp {

class UniqueTIRFunctions : public ExprMutator {
 public:
  explicit UniqueTIRFunctions(IRModule mod) : mod_(mod) {}

  IRModule operator()() {
    GlobalVar gv = mod_->GetGlobalVar("main");
    auto main_func = Downcast<relax::Function>(mod_->Lookup(gv));
    ICHECK(main_func.defined()) << "Main function is not in the module";

    for (auto pair : mod_->functions) {
      if (pair.second->IsInstance<tvm::tir::PrimFuncNode>()) {
        supply_->ReserveGlobalVar(pair.first);
      }
    }

    Expr new_body = this->VisitExpr(main_func->body);

    Function new_func = Function(main_func->params, new_body, main_func->ret_type,
                                 main_func->ret_shape, main_func->attrs);

    mod_->Update(gv, new_func);
    return mod_;
  }

 private:
  Expr VisitExpr_(const CallNode* op) override {
    if (op->op->IsInstance<ExternFuncNode>() || op->op->IsInstance<GlobalVarNode>()) {
      String func_name = op->op->IsInstance<ExternFuncNode>()
                             ? runtime::Downcast<ExternFunc>(op->op)->global_symbol
                             : runtime::Downcast<GlobalVar>(op->op)->name_hint;
      if (!conflict_map.count(func_name)) {
        // There is a single call to this function so far, so leave it unchanged.
        conflict_map[func_name] = true;
        return runtime::GetRef<Call>(op);
      }

      GlobalVar new_global = supply_->FreshGlobal(func_name, false);
      tir::PrimFunc prev_func;
      for (auto pair : mod_->functions) {
        if (pair.first->name_hint.compare(func_name) == 0) {
          ICHECK(pair.second->IsInstance<tir::PrimFuncNode>()) << "Expecting only PrimFuncs at "
                                                                  "this stage.";
          prev_func = runtime::Downcast<tir::PrimFunc>(pair.second);
          break;
        }
      }
      ICHECK(prev_func.defined()) << "Could not find matching PrimFunc with name " << func_name
                                  << " in the IRModule";
      tir::PrimFunc clone_func =
          tir::PrimFunc(prev_func->params, prev_func->body, prev_func->ret_type,
                        prev_func->buffer_map, prev_func->attrs, prev_func->span);
      // Overwrite the global_symbol attribute of the function with the new name.
      clone_func = WithAttr(std::move(clone_func), "global_symbol", new_global->name_hint);

      mod_->Add(new_global, clone_func);

      if (op->op->IsInstance<GlobalVarNode>()) {
        return Call(new_global, op->args, op->attrs, op->type_args, op->span);
      } else if (auto* prev_extern_func = op->op.as<ExternFuncNode>()) {
        return Call(ExternFunc(new_global->name_hint, prev_extern_func->span), op->args, op->attrs,
                    op->type_args, op->span);
      } else {
        ICHECK(false) << "Unexpected call type to PrimFunc.";
      }
    }
    return runtime::GetRef<Call>(op);
  }

  std::unordered_map<std::string, bool> conflict_map;
  GlobalVarSupply supply_ = GlobalVarSupply(NameSupply(""));
  IRModule mod_;
};

}  // namespace usmp
}  // namespace relax

namespace transform {

tvm::transform::Pass UniqueTIRFunctions() {
  auto pass_func = [=](IRModule m, tvm::transform::PassContext ctx) {
    return relax::usmp::UniqueTIRFunctions(m)();
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "relax.usmp.UniqueTIRFunctions", {});
}

TVM_REGISTER_GLOBAL("relax.transform.UniqueTIRFunctions").set_body_typed(UniqueTIRFunctions);

}  // namespace transform
}  // namespace tvm
