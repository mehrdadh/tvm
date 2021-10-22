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
 * \file rpc_server.cc
 * \brief MicroTVM RPC Server
 */

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

// NOTE: dmlc/base.h contains some declarations that are incompatible with some C embedded
// toolchains. Just pull the bits we need for this file.
#define DMLC_CMAKE_LITTLE_ENDIAN DMLC_IO_USE_LITTLE_ENDIAN
#define DMLC_LITTLE_ENDIAN true

#include <tvm/runtime/crt/crt.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/microtvm_aot_server.h>
#include <tvm/runtime/crt/rpc_common/transport.h>

#include "./aot_server.h"
#include "crt_config.h"

namespace tvm {
namespace runtime {
namespace micro_rpc {

class MicroAOTServer : public MicroTransport {
 public:
  MicroAOTServer(uint8_t* receive_storage, size_t receive_storage_size_bytes,
                 microtvm_rpc_channel_write_t write_func, void* write_func_ctx)
      : MicroTransport(receive_storage, receive_storage_size_bytes, write_func, write_func_ctx),
        aot_server_{GetIOHandler()} {}

 private:
  AOTServer<MicroIOHandler> aot_server_;

  void HandleCompleteMessage(MessageType message_type, FrameBuffer* buf) {
    if (message_type != MessageType::kNormal) {
      return;
    }

    SetRunning(aot_server_.ProcessOnePacket());
    Session* session = GetSession();
    session->ClearReceiveBuffer();
  }
};

}  // namespace micro_rpc
}  // namespace runtime
}  // namespace tvm

void* operator new[](size_t count, void* ptr) noexcept { return ptr; }

extern "C" {

static microtvm_aot_server_t g_aot_server = nullptr;

microtvm_aot_server_t MicroTVMAOTServerInit(microtvm_rpc_channel_write_t write_func,
                                            void* write_func_ctx) {
  tvm::runtime::micro_rpc::g_write_func = write_func;
  tvm::runtime::micro_rpc::g_write_func_ctx = write_func_ctx;

  // tvm_crt_error_t err = TVMInitializeRuntime();
  // if (err != kTvmErrorNoError) {
  //   TVMPlatformAbort(err);
  // }

  DLDevice dev = {kDLCPU, 0};
  void* receive_buffer_memory;
  tvm_crt_error_t err = TVMPlatformMemoryAllocate(TVM_CRT_MAX_PACKET_SIZE_BYTES, dev, &receive_buffer_memory);
  if (err != kTvmErrorNoError) {
    TVMPlatformAbort(err);
  }
  auto receive_buffer = new (receive_buffer_memory) uint8_t[TVM_CRT_MAX_PACKET_SIZE_BYTES];
  void* rpc_server_memory;
  err = TVMPlatformMemoryAllocate(sizeof(tvm::runtime::micro_rpc::MicroAOTServer), dev,
                                  &rpc_server_memory);
  if (err != kTvmErrorNoError) {
    TVMPlatformAbort(err);
  }
  auto aot_server = new (rpc_server_memory) tvm::runtime::micro_rpc::MicroAOTServer(
      receive_buffer, TVM_CRT_MAX_PACKET_SIZE_BYTES, write_func, write_func_ctx);
  g_aot_server = static_cast<microtvm_aot_server_t>(aot_server);
  aot_server->Initialize();
  return g_aot_server;
}

void TVMLogf(const char* format, ...) {
  va_list args;
  char log_buffer[256];
  va_start(args, format);
  size_t num_bytes_logged = TVMPlatformFormatMessage(log_buffer, sizeof(log_buffer), format, args);
  va_end(args);

  // Most header-based logging frameworks tend to insert '\n' at the end of the log message.
  // Remove that for remote logging, since the remote logger will do the same.
  if (num_bytes_logged > 0 && log_buffer[num_bytes_logged - 1] == '\n') {
    log_buffer[num_bytes_logged - 1] = 0;
    num_bytes_logged--;
  }

  if (g_aot_server != nullptr) {
    static_cast<tvm::runtime::micro_rpc::MicroAOTServer*>(g_aot_server)
        ->Log(reinterpret_cast<uint8_t*>(log_buffer), num_bytes_logged);
  } 
  // else
  //   tvm::runtime::micro_rpc::SerialWriteStream write_stream;
  //   tvm::runtime::micro_rpc::Framer framer{&write_stream};
  //   tvm::runtime::micro_rpc::Session session{&framer, nullptr, nullptr, nullptr};
  //   tvm_crt_error_t err =
  //       session.SendMessage(tvm::runtime::micro_rpc::MessageType::kLog,
  //                           reinterpret_cast<uint8_t*>(log_buffer), num_bytes_logged);
  //   if (err != kTvmErrorNoError) {
  //     TVMPlatformAbort(err);
  //   }
  }

tvm_crt_error_t MicroTVMAOTServerLoop(microtvm_aot_server_t server_ptr, uint8_t** new_data,
                                      size_t* new_data_size_bytes) {
  tvm::runtime::micro_rpc::MicroAOTServer* server =
      static_cast<tvm::runtime::micro_rpc::MicroAOTServer*>(server_ptr);
  return server->Loop(new_data, new_data_size_bytes);
}

}  // extern "C"
