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
 * \file relax/usmp/unified_static_memory_planner.cc
 * \brief This is the pass that integrates the USMP passes to
 * a single composite pass.
 */

#include <tvm/relax/usmp/analysis.h>
#include <tvm/relax/usmp/transform.h>
#include <tvm/relax/usmp/utils.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/runtime.h>
#include <tvm/target/target.h>
#include <tvm/tir/usmp/algorithms.h>

#include <algorithm>
#include <string>

#include "tvm/tir/usmp/utils.h"

namespace tvm {

TVM_REGISTER_PASS_CONFIG_OPTION(kUSMPRelaxEnableOption, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kUSMPRelaxAlgorithmOption, String);
TVM_REGISTER_PASS_CONFIG_OPTION(kUSMPRelaxUseWorkspaceIO, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kUSMPRelaxCustomAlgorithmOption, String);

namespace relax {
namespace usmp {

static constexpr const char* kDefaultAlgo = "greedy_by_size";

using BufferInfo = tvm::tir::usmp::BufferInfo;
using PoolAllocation = tvm::tir::usmp::PoolAllocation;

static std::unordered_map<String, std::function<Map<BufferInfo, PoolAllocation>(
                                      const Array<BufferInfo>&, const Integer&)>>
    algorithms{{"greedy_by_size", tir::usmp::algo::GreedyBySize},
               {"greedy_by_conflicts", tir::usmp::algo::GreedyByConflicts},
               {"hill_climb", tir::usmp::algo::HillClimb}};

IRModule PlanMemory(const IRModule& mod, String algo, bool use_workspace_io,
                    Optional<String> opt_custom_algo) {
  IRModule module = mod->ShallowCopy();
  // TODO(gigiblender): Add support for use_workspace_io.
  if (use_workspace_io) {
    LOG(FATAL) << "No support for use_workspace_io at the moment.";
  }
  module = tvm::transform::AssignPoolInfo()(module);
  auto main_func = Downcast<relax::Function>(module->Lookup("main"));
  tir::usmp::BufferInfoAnalysis buffer_info_analysis = tvm::ExtractBufferInfo(main_func, module);
  Array<BufferInfo> buffer_info_arr =
      ConvertToArrayOfBufferInfo(buffer_info_analysis->buffer_info_stmts);
  decltype(algorithms)::mapped_type algorithm;
  if (opt_custom_algo) {
    String algo_func_name = "tir.usmp.algo." + opt_custom_algo.value();
    const runtime::PackedFunc* pfAlgo = runtime::Registry::Get(algo_func_name);
    CHECK(pfAlgo) << "The selected custom USMP algorithm : " << opt_custom_algo.value()
                  << " is not defined. Please register it as " << algo_func_name;
    algorithm = *pfAlgo;
  } else {
    CHECK(algorithms.count(algo))
        << "The selected USMP algorithm : " << algo
        << " is not defined. Please define it in the above algorithms map.";
    algorithm = algorithms[algo];
  }
  Map<BufferInfo, PoolAllocation> buffer_info_pool_allocations =
      algorithm(buffer_info_arr, buffer_info_analysis->memory_pressure);

  Map<runtime::ObjectRef, PoolAllocation> pool_allocations = AssignStmtPoolAllocations(
      buffer_info_analysis->buffer_info_stmts, buffer_info_pool_allocations);

  module = tvm::transform::ConvertPoolAllocationsToOffsets(pool_allocations,
                                                           Bool(false),
                                                           Bool(true))(module);
  if (use_workspace_io) {
    // TODO(gigiblender): Add support for use_workspace_io.
    LOG(FATAL) << "No support for use_workspace_io at the moment.";
  }
  return module;
}

}  // namespace usmp

namespace transform {

tvm::transform::Pass UnifiedStaticMemoryPlanner() {
  auto usmp_main_pass_func = [=](IRModule m, tvm::transform::PassContext ctx) {
    auto algorithm_str = ctx->GetConfig(kUSMPRelaxAlgorithmOption, String(usmp::kDefaultAlgo));
    auto use_workspace_io = ctx->GetConfig(kUSMPRelaxUseWorkspaceIO, Bool(false));
    auto custom_algorithm_str = ctx->GetConfig<String>(kUSMPRelaxCustomAlgorithmOption);
    tvm::relay::Executor executor_config =
        m->GetAttr<tvm::relay::Executor>(tvm::attr::kExecutor).value();
    String interface_api = executor_config->GetAttr<String>("interface-api").value_or("packed");
    tvm::relay::Runtime runtime_config =
        m->GetAttr<tvm::relay::Runtime>(tvm::attr::kRuntime).value();
    if (use_workspace_io.value()) {
      CHECK(interface_api == "c") << kUSMPRelaxUseWorkspaceIO
                                  << " option is only compatible with interface_api c.\n"
                                  << "Please use interface_api c to be able to enable "
                                  << kUSMPRelaxUseWorkspaceIO << "\n";
    }
    return Downcast<IRModule>(
        usmp::PlanMemory(m, algorithm_str.value_or(String(usmp::kDefaultAlgo)),
                         use_workspace_io.value_or(Bool(false)), custom_algorithm_str));
  };

  return tvm::transform::CreateModulePass(usmp_main_pass_func, 0,
                                          "relax.transform.UnifiedStaticMemoryPlanner", {});
}

TVM_REGISTER_GLOBAL("relax.transform.UnifiedStaticMemoryPlanner")
    .set_body_typed(UnifiedStaticMemoryPlanner);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
