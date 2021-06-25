# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_CMSIS STREQUAL "ON")
  #find_library(BLAS_LIBRARY openblas)
  #list(APPEND TVM_RUNTIME_LINKER_LIBS ${BLAS_LIBRARY})
  list(APPEND RUNTIME_SRCS src/runtime/contrib/cmsis/cmsis.cc)
  message(STATUS "Using CMSIS library")
elseif(USE_CMSIS STREQUAL "OFF")
  # pass
else()
  message(FATAL_ERROR "Invalid option: USE_CMSIS=" ${USE_BLAS})
endif()

