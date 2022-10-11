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
 * \file relax/usmp/utils.cc
 * \brief Utilities for Relax Unified Static Memory Planner
 */

#include <tvm/relax/usmp/utils.h>

namespace tvm {
namespace relax {
namespace usmp {

Integer CalculateRelaxExtentsSize(const DataType& dtype, const Array<PrimExpr>& extents) {
  size_t element_size_bytes = dtype.bytes();
  size_t num_elements = 1;
  for (const auto& ext : extents) {
    if (ext->IsInstance<IntImmNode>()) {
      num_elements *= Downcast<IntImm>(ext)->value;
    }
  }
  return Integer(num_elements * element_size_bytes);
}

}  // namespace usmp
}  // namespace relax
}  // namespace tvm