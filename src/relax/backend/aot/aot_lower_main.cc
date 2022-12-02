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
 * \file src/relax/backend/aot/aot_lower_main.cc
 * \brief Lower the Relax main func into an AOT TIR main func.
 */
#include "./aot_lower_main.h"

#include <tvm/runtime/name_transforms.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>
#include <tvm/relay/op_attr_types.h>

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>
#include <tvm/relax/attrs/memory.h>

#include "../../../relay/backend/name_transforms.h"
#include "../../../runtime/meta_data.h"
// #include "../utils.h"

namespace tvm {
namespace relax {
namespace aot {

struct TensorInfo {
  TensorInfo(const tir::Var& var, const tir::Var& storage, int offset) : var(var), storage(storage), offset(offset) {}

  tir::Var var;
  tir::Var storage;
  int offset;
};

class AOTMainLowerer : public ExprVisitor {
 public:
  using ExprVisitor::VisitExpr_;
  AOTMainLowerer(tvm::CompilationConfig config)
      : config_(config) {}

  IRModule Lower(IRModule mod, String mod_name) {
    IRModule lowered_mod = GetRef<IRModule>(mod.CopyOnWrite());

    auto lowered_main = lowered_mod->Lookup("main");
    auto lowered_main_func = GetRef<Function>(lowered_main.as<FunctionNode>());

    Map<relax::Var, tir::Var> io_map;
    for (auto input : lowered_main_func->params) {
      std::string input_name = runtime::SanitizeName(input->name_hint());
      // We don't want the compiler changing input names in the
      // event of a sanitization collision. Therefore, enforcing
      // the var created to use the input_name strictly.
      tir::Var tvar = CreateIOVar(input, input_name, /*use_unique_name = */ false);
      io_map.Set(input, tvar);
    }

    CollectDeviceVariables(mod->GetAttr<Map<GlobalVar, String>>("device_contexts")
                               .value_or(Map<GlobalVar, String>()));

    VisitExpr(lowered_main_func);

    // Remove the Relay main and replace it with the lowered TIR version
    mod->Remove(lowered_mod->GetGlobalVar("main"));
    auto tir_main_func = CreateMainFunc(mod_name);

    auto input_var_attr = lowered_main_func->GetAttr<Array<relax::Var>>("input_vars");
    ICHECK(input_var_attr.defined()) << "Main function is missing the input_vars attr";
    auto relax_input_vars = input_var_attr.value();
    Array<tir::Var> tir_input_vars;
    for (const auto& rvar : relax_input_vars) {
      tir_input_vars.push_back(io_map.at(rvar));
    }
    auto output_var_attr = lowered_main_func->GetAttr<Array<relax::Var>>("output_vars");
    ICHECK(output_var_attr.defined()) << "Main function is missing the output_vars attr";
    auto relax_output_vars = output_var_attr.value();
    Array<tir::Var> tir_output_vars;
    for (const auto& rvar : relax_output_vars) {
      tir_output_vars.push_back(io_map.at(rvar));
    }
    tir_main_func = WithAttr(tir_main_func, "input_vars", tir_input_vars);
    tir_main_func = WithAttr(tir_main_func, "output_vars", tir_output_vars);

    lowered_mod->Update(GlobalVar(runtime::symbol::tvm_module_main), tir_main_func);
    lowered_mod = tir::transform::RemoveNoOp()(lowered_mod);
    return lowered_mod;
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    static const Op& alloc_storage_op = Op::Get("relax.memory.alloc_storage");
    static const Op& alloc_tensor_op = Op::Get("relax.memory.alloc_tensor");

    if (const auto* vn = binding->value.as<VarNode>()) {
      var_map_[binding->var.get()] = var_map_.at(vn);
    }

    if (const auto* tn = binding->value.as<TupleNode>()) {
      Array<tir::Var> tvars;
      for (const auto& field : tn->fields) {
        const auto* vn = field.as<VarNode>();
        tvars.insert(tvars.end(), var_map_.at(vn).begin(), var_map_.at(vn).end());
      }
      var_map_[binding->var.get()] = tvars;
    }

    // TODO(@mbaret) This logic won't work for nested tuples
    if (const auto* tgn = binding->value.as<TupleGetItemNode>()) {
      if (const auto* vn = tgn->tuple.as<VarNode>()) {
        var_map_[binding->var.get()] = {var_map_.at(vn)[tgn->index]};
      } else {
        // TupleGetItem with a Tuple is not put into ANF
        const auto* tn = tgn->tuple.as<TupleNode>();
        ICHECK(tn != nullptr);
        var_map_[binding->var.get()] = var_map_.at(tn->fields[tgn->index].as<VarNode>());
      }
    }

    if (const auto* cn = binding->value.as<ConstantNode>()) {
      tir::Var constant_var(MakeString("constant_", alloc_constants_.size()), PointerType(PrimType(DataType(cn->data->dtype))));
      var_map_[binding->var.get()] = {constant_var};
      alloc_constants_.emplace_back(constant_var, GetRef<Constant>(cn));
    }

    if (const auto* cn = binding->value.as<CallNode>()) {
      if (cn->op == alloc_storage_op) {
        auto attrs = cn->attrs.as<MemAllocStorageAttrs>();
        tir::Var buffer_var(MakeString("sid_", allocs_.size()),
                            PointerType(PrimType(attrs->dtype), "global.workspace"));
        var_map_[binding->var.get()] = {buffer_var};
        int alloc_size;
        if (const auto* shape_expr = cn->args[0].as<ShapeExprNode>()) {
          alloc_size = shape_expr->values[0].as<IntImmNode>()->value;
        } else {
          alloc_size = static_cast<int64_t*>(cn->args[0].as<ConstantNode>()->data->data)[0];
        }
        auto buffer = tir::Buffer(buffer_var, attrs->dtype, {alloc_size}, {1}, 0, buffer_var->name_hint, 16, 1, tir::BufferType::kDefault);
        allocs_.push_back(buffer);
        alloc_buffers_.Set(buffer_var, buffer);
      } else if (cn->op == alloc_tensor_op) {
        auto attrs = cn->attrs.as<MemAllocTensorAttrs>();
        tir::Var tensor_var(MakeString("tid_", tensors_.size()),
                            PointerType(PrimType(attrs->dtype), "global.workspace"));
        var_map_[binding->var.get()] = {tensor_var};
        tensors_.emplace_back(tensor_var, var_map_.at(cn->args[0].as<VarNode>())[0], attrs->offset);
      } else if (const auto* gv = cn->op.as<GlobalVarNode>()) {
        CreateFuncCall(gv->name_hint, cn->args);
      } else if (const auto * ef = cn->op.as<ExternFuncNode>()) {
        CreateFuncCall(ef->global_symbol, cn->args);
      } else {
        LOG(FATAL) << "Unsupported op";
      }
    }
  }

 private:
  /*!
   * \brief Create the main PrimFunc to execute the graph.
   * \note The packed function calls don't pack their arguments. The AOT
   * runner function needs to be legalized by the LegalizePackedCalls pass.
   */
  tir::PrimFunc CreateMainFunc(String mod_name) {
    tir::Stmt body = tir::SeqStmt(stmts_);

    // Bind tensor values
    for (auto tensor_info : tensors_) {
      auto load_node = tir::BufferLoad(alloc_buffers_[tensor_info.storage], {tensor_info.offset});
      auto address_of_load = tir::Call(DataType::Handle(), tir::builtin::address_of(), {load_node});
      body = tir::LetStmt(tensor_info.var, address_of_load, body);
    }
    // Allocates and DeclBuffers
    for (const auto& buffer : allocs_) {
      body = tir::Allocate(buffer->data, buffer->dtype, buffer->shape, tir::const_true(), body);
    }
    // AllocateConsts
    for (const auto& it : alloc_constants_) {
      relay::Shape shape;
      for (int i=0;i<it.second->data->ndim;i++) {
        shape.push_back(Integer(it.second->data->shape[i]));
      }
      body = tir::AllocateConst(it.first, DataType(it.second->data->dtype), shape, it.second->data, body);
    }

    // Define the PrimFunc attributes
    Map<String, ObjectRef> dict_attrs;
    String run_func_name = runtime::get_name_mangled(mod_name, runtime::symbol::tvm_module_main);
    dict_attrs.Set("global_symbol", run_func_name);
    dict_attrs.Set("runner_function", Bool(true));
    dict_attrs.Set(tvm::attr::kTarget, config_->host_target);

    tir::Stmt device_activations = GenerateAllDeviceHook("Activate");
    tir::Stmt device_deactivations = GenerateAllDeviceHook("Deactivate");
    tir::Stmt final_body = tir::SeqStmt({device_activations, body, device_deactivations});

    // Make the PrimFunc
    return tir::PrimFunc(main_signature_, final_body, VoidType(), main_buffer_map_, DictAttrs(dict_attrs));
  }

  /*!
   * \brief Collects device context variables for passing to operators
   */
  void CollectDeviceVariables(const Map<GlobalVar, String>& device_contexts) {
    Map<TargetKind, tir::Var> target_contexts;
    TargetKindAttrMap<Bool> target_attr_map = tvm::TargetKind::GetAttrMap<Bool>("use_device_api");

    for (const auto& it : device_contexts) {
      const GlobalVar& global_var = it.first;
      const std::string device_context_name = it.second;

      Optional<TargetKind> target_kind = tvm::TargetKind::Get(device_context_name);
      if (!target_kind || !target_attr_map.count(target_kind.value())) {
        return;
      }
      if (target_attr_map[target_kind.value()]) {
        std::string context_name = runtime::SanitizeName(device_context_name);
        tir::Var device_context_var("device_context_" + context_name, DataType::Handle());

        auto pair = target_contexts.find(target_kind.value());
        if (pair != target_contexts.end()) {
          device_context_var = (*pair).second;
        } else {
          main_signature_.push_back(device_context_var);
          devices_.Set(context_name, device_context_var);
          target_contexts.Set(target_kind.value(), device_context_var);
        }

        device_contexts_.Set(global_var->name_hint, device_context_var);
      }
    }
  }

  /*!
   * \brief Create tir::Var for input/output while updating the buffer_maps.
   * \param type The type of the IO var.
   * \param shape The shape of the IO var.
   * \param original_name The name of the tir::Var.
   * \param use_unique_name Whether to generate a new unique name where a name conflicts.
   */
  tir::Var CreateIOVar(const relax::Var& var, const std::string& original_name,
                   bool use_unique_name = true) {
    auto shape = var->shape_;
    auto type = var->checked_type();
    std::string name = original_name;
    if (use_unique_name) {
      name = GetUniqueIOVarName(original_name);
    }
    tir::Var tvar = tir::Var(name, DataType::Handle());
    main_signature_.push_back(tvar);
    auto tensor_type = type.as<DynTensorTypeNode>();
    ICHECK(tensor_type) << "Expected DynTensorType node but was " << type->GetTypeKey();
    DataType elem_type = tensor_type->dtype;
    auto tensor_shape = shape.as<ShapeExprNode>()->values;
    tir::Var buffer_var =
        tir::Var(name + "_buffer_var", PointerType(PrimType(elem_type), "global"));
    tir::Buffer buffer = tir::Buffer(buffer_var, elem_type, tensor_shape, {}, IntImm(DataType::Int(64), 0),
                                      name + "_buffer", 16, 1, tir::BufferType::kDefault);
    main_buffer_map_.Set(tvar, buffer);
    var_map_[var.get()] = {buffer_var};
    return tvar;
  }

  /*!
   * \brief Create a unique name for I/O Var
   */
  std::string GetUniqueIOVarName(std::string name) {
    if (io_var_names_.find(name) == io_var_names_.end()) {
      io_var_names_[name] = 1;
      return name + std::to_string(io_var_names_[name] - 1);
    } else {
      io_var_names_[name] = io_var_names_[name] + 1;
      return name + std::to_string(io_var_names_[name] - 1);
    }
  }

  /*!
   * \brief Wraps a call_extern with a tvm_check_return annotation if required otherwise
   * returns the passed Call
   */
  tir::Call AddCheckReturn(tir::Call existing_call) {
    Array<PrimExpr> args = {tir::make_const(DataType::Int(32, 1), 0, Span()),
                            tir::make_const(DataType::Int(32, 1), -1, Span()), existing_call};
    return tir::Call(DataType::Int(32), tir::builtin::tvm_check_return(), args);
  }

  /*!
   * \brief Create a function call
   * \param call_lowered_props The lowered function and the arguments to call it with
   */
  void CreateFuncCall(std::string func_name, const Array<Expr>& func_args) {
    tvm::Array<PrimExpr> args{tvm::tir::StringImm(func_name)};
    std::vector<tir::Stmt> create_func_call_stmts;

    // Pack the inputs
    for (const Expr& arg : func_args) {
      if (const auto* cn = arg.as<ConstantNode>()) {
        tir::Var constant_var(MakeString("constant_", alloc_constants_.size()), PointerType(PrimType(DataType(cn->data->dtype))));
        alloc_constants_.emplace_back(constant_var, GetRef<Constant>(cn));
        args.push_back(constant_var);
      } else {
        const auto* vn = arg.as<VarNode>();
        ICHECK(vn != nullptr);
        Array<tir::Var> vars = var_map_.at(vn);
        args.insert(args.end(), vars.begin(), vars.end());
      }
    }

    bool has_c_device_api_context = device_contexts_.count(func_name) != 0;
    tir::Var device_context;
    tir::Stmt func_call;
    if (has_c_device_api_context) {
      device_context = device_contexts_.Get(func_name).value();
      args.push_back(device_context);
    } else {
      // NOTE: LowerTVMBuiltin expects some device_context placeholder.
      args.push_back(tir::make_zero(DataType::Handle()));
    }
    func_call = tir::Evaluate(
        tvm::tir::Call(DataType::Int(32), tvm::tir::builtin::tvm_call_cpacked(), args));
    create_func_call_stmts.push_back(func_call);

    if (has_c_device_api_context) {
      func_call = tir::SeqStmt(Array<tir::Stmt>({
          GenerateDeviceHook(device_context, "Open"),
          func_call,
          GenerateDeviceHook(device_context, "Close"),
      }));
    }

    tir::Stmt body = tir::SeqStmt({func_call});
    stmts_.push_back(body);
  }

  /*!
   * \brief Generates a call to a given hook for a single Device function
   * \param context Device context to call hook on
   * \param hook Name of hook to generate statements for
   * \return Statement with function call to Device API
   */
  tir::Stmt GenerateDeviceHook(const tir::Var& context, const String& hook) {
    const auto& it = std::find_if(std::begin(devices_), std::end(devices_), [&](const auto& it) {
      return it.second->name_hint == context->name_hint;
    });
    const String& device_name = (*it).first;
    Array<String> sections = {"Device", device_name, hook};
    String device_hook = relay::backend::ToCFunctionStyle(relay::backend::PrefixName(sections));

    return tir::Evaluate(
        AddCheckReturn(tir::Call(DataType::Int(32), tvm::tir::builtin::call_extern(),
                                 {tvm::tir::StringImm(device_hook), context})));
  }

  /*!
   * \brief Generates a call to a given hook for all Devices found for C Device API
   * \param hook Name of hook to generate statements for
   * \return Statement with function calls for each device
   */
  tir::Stmt GenerateAllDeviceHook(const String& hook) {
    std::vector<tir::Stmt> device_hooks;
    for (const auto& it : devices_) {
      const String& device_name = it.first;
      const tir::Var& context = it.second;
      Array<String> sections = {"Device", device_name, hook};
      String device_hook_name = relay::backend::ToCFunctionStyle(relay::backend::PrefixName(sections));

      tir::Evaluate device_hook(
          AddCheckReturn(tvm::tir::Call(DataType::Int(32), tvm::tir::builtin::call_extern(),
                                        {tvm::tir::StringImm(device_hook_name), context})));
      device_hooks.push_back(device_hook);
    }
    return tir::SeqStmt(device_hooks);
  }

  /*!
   * \brief Utility function to string together different arguments
   */
  template <typename... Args>
  std::string MakeString(Args const&... args) {
    std::ostringstream ss;
    using List = int[];
    (void)List{0, ((void)(ss << args), 0)...};

    return ss.str();
  }

  /*! \brief All available targets. */
  CompilationConfig config_;
  /*! \brief list of input expressions (i.e., variable passed by the user) */
  std::vector<Var> input_vars_;
  /*! \brief map of device contexts variables */
  Map<String, tir::Var> devices_;
  /*! \brief map of GlobalVars to C Device API contexts */
  Map<String, tir::Var> device_contexts_;
  /*! \brief input and output variables belonging to the main function signature */
  Array<tir::Var> main_signature_;
  /*! \brief input and output variables belonging to the main function signature */
  Map<tir::Var, tir::Buffer> main_buffer_map_;
  /*! \brief This is per IO var name counter to aid the generating unique names */
  std::unordered_map<std::string, int> io_var_names_;
  std::unordered_map<const relax::VarNode*, Array<tir::Var>> var_map_;
  std::vector<tir::Buffer> allocs_;
  std::vector<TensorInfo> tensors_;
  Map<tir::Var, tir::Buffer> alloc_buffers_;
  Map<relax::Var, relax::Expr> alias_map_;
  /*! \brief the set of statements that make the program */
  std::vector<tir::Stmt> stmts_;
  std::vector<std::pair<tir::Var, relax::Constant>> alloc_constants_;
};

transform::Pass AOTLowerMain(String mod_name, tvm::CompilationConfig config) {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule module, transform::PassContext ctx) {
        return AOTMainLowerer(config).Lower(module, mod_name);
      };

  return tvm::transform::CreateModulePass(pass_func, 0, "relax.aot.AOTLowerMain", {"InferType"});
}

TVM_REGISTER_GLOBAL("relax.aot.AOTLowerMain")
    .set_body_typed([](const String& mod_name, const tvm::CompilationConfig& config) {
      return AOTLowerMain(mod_name, config);
    });

}  // namespace aot
}  // namespace relax
}  // namespace tvm
