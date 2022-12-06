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
 * \file src/relax/aot/codegen_aot.cc
 * \brief AOT codegen
 */

#include <tvm/ir/module.h>
#include <tvm/driver/driver_api.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/attrs/call.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/runtime.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/backend.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/usmp/utils.h>
#include <tvm/relax/usmp/utils.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/runtime.h>

#include <algorithm>
#include <list>
#include <string>
#include <vector>

#include "./aot_lower_main.h"
#include "../../../relay/backend/te_compiler.h"
#include "../../../relay/backend/aot/create_executor_metadata.h"
#include "../../../target/metadata_module.h"
#include "../../../driver/internal_driver_api.h"

namespace tvm {
namespace relax {
namespace aot {

runtime::Module Build(IRModule mod, String mod_name, CompilationConfig config, relay::Executor executor,
                   relay::Runtime runtime) {
  Integer workspace_byte_alignment =
      executor->GetAttr<Integer>("workspace-byte-alignment").value_or(16);
  Integer constant_byte_alignment =
      executor->GetAttr<Integer>("constant-byte-alignment").value_or(16);

  transform::PassContext pass_ctx = transform::PassContext::Current();
  bool enable_usmp = pass_ctx->GetConfig<Bool>(kUSMPRelaxEnableOption, Bool(false)).value();

  mod = LowerModule(mod);
  if (enable_usmp) {
    mod = relax::transform::UnifiedStaticMemoryPlanner()(mod);
  } else {
    mod = relax::transform::AOTMemoryLower()(mod);
  }
  mod = AOTLowerMain(mod_name, config)(mod);

  mod = tir::transform::LegalizePackedCalls()(mod);
  mod = tir::transform::TIRFuncRename()(mod);

  auto lowered_funcs = tvm::relay::tec::GetPerTargetModules(mod);
  auto exec_metadata = tvm::relay::backend::aot::CreateExecutorMetadata(mod, mod_name, executor, workspace_byte_alignment,
                                        constant_byte_alignment);

  const Target& host_target = config->host_virtual_device->target;
  auto rt_mod = tvm::TIRToRuntime(lowered_funcs, host_target);
  return tvm::codegen::CreateMetadataModule({}, rt_mod, {}, host_target, runtime, executor, exec_metadata);
}

TVM_REGISTER_GLOBAL("relax.aot.build")
    .set_body_typed([](IRModule mod, String mod_name, CompilationConfig config, relay::Executor executor, relay::Runtime runtime) {
      return Build(mod, mod_name, config, executor, runtime);
    });

}  // namespace aot
}  // namespace relax
}  // namespace tvm
