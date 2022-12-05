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
 * \file src/relax/backend/aot/tir_func_rename.cc
 * \brief Mangles TIR function names to avoid symbol conflicts.
 * Appends "_tvm_gen" to all function names in the IRModule.
 */

#include <utility>

#include "tvm/ir/name_supply.h"
#include "tvm/ir/transform.h"
#include "tvm/tir/builtin.h"
#include "tvm/tir/stmt_functor.h"

namespace tvm {
namespace tir {
namespace aot {

class TIRMangleFuncName : public StmtExprMutator {

 public:
  explicit TIRMangleFuncName(IRModule mod) : mod_(std::move(mod)) {
    ICHECK(mod_->ContainGlobalVar(runtime::symbol::tvm_module_main)) << "Expecting module to have"
                              << " symbol " << runtime::symbol::tvm_module_main << " attached.";
    auto main_func_gv = mod_->GetGlobalVar(runtime::symbol::tvm_module_main);
    NameSupply name_supply = NameSupply("_tvm_gen");
    for (auto pair : mod_->functions) {
      if (pair.first.same_as(main_func_gv)) {
        // Ignore the main function.
        continue;
      }
      auto prim_func = runtime::Downcast<PrimFunc>(pair.second);
      auto func_name = prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
      ICHECK(func_name.defined()) << "Expecting global_symbol attribute to be attached to the"
                                     " function";
      name_map_[func_name.value()] = name_supply->FreshName(func_name.value());
    }
  }

  IRModule operator()() {
    auto main_func_gv = mod_->GetGlobalVar(runtime::symbol::tvm_module_main);

    Map<GlobalVar, BaseFunc> func_map = Map<GlobalVar, BaseFunc>();
    for (auto pair : mod_->functions) {
      auto prim_func = runtime::Downcast<PrimFunc>(pair.second);
      auto func_name = prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol);

      Stmt new_body = this->VisitStmt(prim_func->body);
      if (pair.first.same_as(main_func_gv)) {
        // No need to set a new global var and global symbol for the main function.
        func_map.Set(pair.first, PrimFunc(prim_func->params, new_body, prim_func->ret_type,
                                      prim_func->buffer_map, prim_func->attrs, prim_func->span));
      } else {
        ICHECK(name_map_.count(func_name.value()) > 0) << "Expecting new name in name_map_ at "
                                                          "this stage.";
        GlobalVar new_var = GlobalVar(name_map_[func_name.value()]);
        PrimFunc new_func = PrimFunc(prim_func->params, new_body, prim_func->ret_type,
                                     prim_func->buffer_map, prim_func->attrs, prim_func->span);
        new_func = WithAttr(new_func, tvm::attr::kGlobalSymbol,
                            String(name_map_[func_name.value()]));
        func_map.Set(new_var, new_func);
      }
    }

    IRModule new_mod = IRModule(func_map, mod_->type_definitions, mod_->Imports(),
                                mod_->source_map, mod_->attrs);
    return new_mod;
  }

 private:
  PrimExpr VisitExpr_(const CallNode* op) override {
    String func_name;
    if (op->op.same_as(builtin::call_extern()) || op->op.same_as(builtin::tvm_call_cpacked())) {
      func_name = Downcast<StringImm>(op->args[0])->value;
    }
    if (op->op->IsInstance<PrimFuncNode>()) {
      func_name = Downcast<StringImm>(op->args[0])->value;
    }
    if (func_name.defined() && mod_->ContainGlobalVar(func_name) &&
        mod_->Lookup(func_name)->IsInstance<PrimFuncNode>()) {
      ICHECK(name_map_.count(func_name) > 0) << "Name map should contain a name.";
      StringImm new_name = StringImm(name_map_[func_name]);
      Array<PrimExpr> new_args = { new_name };
      new_args.insert(new_args.end(), op->args.begin() + 1, op->args.end());
      return Call(op->dtype, op->op, new_args, op->span);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  std::unordered_map<std::string, std::string> name_map_;
  IRModule mod_;
};

}  // namespace aot

namespace transform {

tvm::transform::Pass TIRFuncRename() {
  auto pass_func = [=](IRModule m, tvm::transform::PassContext ctx) {
    return runtime::Downcast<IRModule>(tvm::tir::aot::TIRMangleFuncName(m)());
  };

  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tir.transform.TIRFuncRename", {});
}

}  // namespace transform
}  // namespace tir
}  // namespace tvm