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

#######################################################
# AArch64 Launcher
#
cmake_minimum_required(VERSION 3.2)

project(aarch64_launcher)

include(ExternalProject)

set(TVM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../../../..")
set(LAUNCHER_LIB_DIR "${CMAKE_SOURCE_DIR}/../../../../build/launchers/aarch64")

find_program(C_COMPILER aarch64-linux-gnu-gcc REQUIRED)
find_program(CXX_COMPILER aarch64-linux-gnu-g++ REQUIRED)

# Build AArch64 binaries:
# - libtvm_runtime.so
# - tvm_rpc for AArch64
ExternalProject_Add(aarch64_tvm_runtime_rpc
  SOURCE_DIR "${TVM_SOURCE_DIR}"
  BUILD_COMMAND $(MAKE) runtime tvm_rpc
  CMAKE_ARGS
    "-DUSE_GRAPH_EXECUTOR=ON"
    "-DUSE_LIBBACKTRACE=OFF"
    "-DUSE_LLVM=OFF"
    "-DUSE_PROFILER=OFF"
    "-DUSE_RPC=ON"
    "-DUSE_CPP_RPC=ON"
    "-DCMAKE_C_COMPILER=${C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}"
    "-DUSE_ALTERNATIVE_LINKER=OFF"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
)

file(MAKE_DIRECTORY ${LAUNCHER_LIB_DIR})
ExternalProject_Get_Property(aarch64_tvm_runtime_rpc BINARY_DIR)
ExternalProject_Add_Step(aarch64_tvm_runtime_rpc copy_runtime
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BINARY_DIR}/libtvm_runtime.so
    ${LAUNCHER_LIB_DIR}
  DEPENDEES install
)
ExternalProject_Add_Step(aarch64_tvm_runtime_rpc copy_rpc_server
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BINARY_DIR}/tvm_rpc
    ${LAUNCHER_LIB_DIR}
  DEPENDEES install
)

configure_file("${CMAKE_SOURCE_DIR}/../rpc_server.sh.template"
  ${LAUNCHER_LIB_DIR} COPYONLY)
