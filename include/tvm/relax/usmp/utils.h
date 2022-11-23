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
 * \file relax/usmp/utils.h
 * \brief Utilities for Relax Unified Static Memory Planner
 */

#ifndef TVM_RELAX_USMP_UTILS_H_
#define TVM_RELAX_USMP_UTILS_H_

#include "tvm/ir/expr.h"

namespace tvm {

/*!
 * \brief PassContext option to enable the USMP
 */
constexpr const char* kUSMPRelaxEnableOption = "relax.usmp.enable";
/*!
 * \brief PassContext option to select the memory planning algorithm in USMP
 */
constexpr const char* kUSMPRelaxAlgorithmOption = "relax.usmp.algorithm";
/*!
 * \brief PassContext option to enable placing I/O tensors in the workspace
 */
constexpr const char* kUSMPRelaxUseWorkspaceIO = "relax.usmp.use_workspace_io";
/*!
 * \brief PassContext option to specify a custom memory planning algorithm in USMP.
 * The algorithm should be provided as registered PackedFunc with the name tir.usmp.algorithm.NAME
 */
constexpr const char* kUSMPRelaxCustomAlgorithmOption = "relax.usmp.custom_algorithm";

namespace relax {
namespace usmp {

Integer CalculateRelaxExtentsSize(const DataType& dtype, const Array<PrimExpr>& extents);

}  // namespace usmp
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_USMP_UTILS_H_
