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
 * \file tir/usmp/transform.h
 * \brief The transform passes for TIR-based Unified Static Memory Planner
 */

#ifndef TVM_RELAX_USMP_TRANSFORM_H_
#define TVM_RELAX_USMP_TRANSFORM_H_

#include <tvm/ir/transform.h>
#include <tvm/relax/usmp/utils.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/usmp/utils.h>

namespace tvm {
namespace transform {

using Pass = tvm::transform::Pass;

/*!
 * \brief Convert the analyzed PoolAllocation to offsets from pool variables
 *
 * This pass would convert the main function to accept pool variables as an input
 * that get passed onto the operator PrimFuncs. Furthermore, the static allocations
 * will be converted to offsets within the pool variable.
 *
 * \return the pass
 */
TVM_DLL Pass ConvertPoolAllocationsToOffsets(
    const Map<runtime::ObjectRef, tir::usmp::PoolAllocation>& pool_allocations,
    Bool emit_tvmscript_printable = Bool(false), Bool insert_storage_allocations = Bool(true));

/*!
 * \brief Assign PoolInfo objects to tir.allocate nodes depending on the PrimFunc's target
 *
 * This pass would assign default PoolInfo objects to allocate nodes that are not otherwise
 * annotated, depending on pool info supplied for each target.
 *
 * \return the pass
 */
TVM_DLL Pass AssignPoolInfo();

}  // namespace transform
}  // namespace tvm

#endif  // TVM_RELAX_USMP_TRANSFORM_H_
