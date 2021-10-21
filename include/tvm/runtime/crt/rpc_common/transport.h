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
 * \file transport.h
 * \brief RPC Session
 */

#ifndef TVM_RUNTIME_CRT_RPC_COMMON_TRANSPORT_H_
#define TVM_RUNTIME_CRT_RPC_COMMON_TRANSPORT_H_

#include <stdlib.h>
#include <sys/types.h>

// NOTE: dmlc/base.h contains some declarations that are incompatible with some C embedded
// toolchains. Just pull the bits we need for this file.
#define DMLC_CMAKE_LITTLE_ENDIAN DMLC_IO_USE_LITTLE_ENDIAN
#define DMLC_LITTLE_ENDIAN true
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/crt.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/microtvm_rpc_server.h>
#include <tvm/runtime/crt/module.h>
#include <tvm/runtime/crt/page_allocator.h>
#include <tvm/runtime/crt/platform.h>
#include <tvm/runtime/crt/rpc_common/frame_buffer.h>
#include <tvm/runtime/crt/rpc_common/framing.h>
#include <tvm/runtime/crt/rpc_common/session.h>

#include "crt_config.h"

namespace tvm {
namespace runtime {
namespace micro_rpc {

class MicroIOHandler {
 public:
  MicroIOHandler(Session* session, FrameBuffer* receive_buffer)
      : session_{session}, receive_buffer_{receive_buffer} {}

  void MessageStart(size_t message_size_bytes) {
    session_->StartMessage(MessageType::kNormal, message_size_bytes + 8);
  }

  ssize_t PosixWrite(const uint8_t* buf, size_t buf_size_bytes) {
    int to_return = session_->SendBodyChunk(buf, buf_size_bytes);
    if (to_return < 0) {
      return to_return;
    }
    return buf_size_bytes;
  }

  void MessageDone() { CHECK_EQ(session_->FinishMessage(), kTvmErrorNoError, "FinishMessage"); }

  ssize_t PosixRead(uint8_t* buf, size_t buf_size_bytes) {
    return receive_buffer_->Read(buf, buf_size_bytes);
  }

  void Close() {}

  void Exit(int code) {
    for (;;) {
    }
  }

 private:
  Session* session_;
  FrameBuffer* receive_buffer_;
};

namespace {
// Stored as globals so that they can be used to report initialization errors.
microtvm_rpc_channel_write_t g_write_func = nullptr;
void* g_write_func_ctx = nullptr;
}  // namespace

class SerialWriteStream : public WriteStream {
 public:
  SerialWriteStream() {}
  virtual ~SerialWriteStream() {}

  ssize_t Write(const uint8_t* data, size_t data_size_bytes) override {
    return g_write_func(g_write_func_ctx, data, data_size_bytes);
  }

  void PacketDone(bool is_valid) override {}

 private:
  void operator delete(void*) noexcept {}  // NOLINT(readability/casting)
};

class MicroTransport {
 public:
  MicroTransport(uint8_t* receive_storage, size_t receive_storage_size_bytes,
                 microtvm_rpc_channel_write_t write_func, void* write_func_ctx)
      : receive_buffer_{receive_storage, receive_storage_size_bytes},
        framer_{&send_stream_},
        session_{&framer_, &receive_buffer_, &HandleCompleteMessageCb, this},
        io_{&session_, &receive_buffer_},
        unframer_{session_.Receiver()},
        is_running_{true} {}

  void* operator new(size_t count, void* ptr) { return ptr; }

  void Initialize() {
    uint8_t initial_session_nonce = Session::kInvalidNonce;
    tvm_crt_error_t error =
        TVMPlatformGenerateRandom(&initial_session_nonce, sizeof(initial_session_nonce));
    CHECK_EQ(kTvmErrorNoError, error, "generating random session id");
    CHECK_EQ(kTvmErrorNoError, session_.Initialize(initial_session_nonce), "rpc server init");
  }

  /*! \brief Process one message from the receive buffer, if possible.
   *
   * \param new_data If not nullptr, a pointer to a buffer pointer, which should point at new input
   *     data to process. On return, updated to point past data that has been consumed.
   * \param new_data_size_bytes Points to the number of valid bytes in `new_data`. On return,
   *     updated to the number of unprocessed bytes remaining in `new_data` (usually 0).
   * \return an error code indicating the outcome of the processing loop.
   */
  tvm_crt_error_t Loop(uint8_t** new_data, size_t* new_data_size_bytes) {
    if (!is_running_) {
      return kTvmErrorPlatformShutdown;
    }

    tvm_crt_error_t err = kTvmErrorNoError;
    if (new_data != nullptr && new_data_size_bytes != nullptr && *new_data_size_bytes > 0) {
      size_t bytes_consumed;
      err = unframer_.Write(*new_data, *new_data_size_bytes, &bytes_consumed);
      *new_data += bytes_consumed;
      *new_data_size_bytes -= bytes_consumed;
    }

    if (err == kTvmErrorNoError && !is_running_) {
      err = kTvmErrorPlatformShutdown;
    }

    return err;
  }

  void Log(const uint8_t* message, size_t message_size_bytes) {
    tvm_crt_error_t to_return =
        session_.SendMessage(MessageType::kLog, message, message_size_bytes);
    if (to_return != 0) {
      TVMPlatformAbort(to_return);
    }
  }

 private:
  FrameBuffer receive_buffer_;
  SerialWriteStream send_stream_;
  Framer framer_;
  Unframer unframer_;

 protected:
  MicroIOHandler io_;
  bool is_running_;
  Session session_;

  // void HandleCompleteMessage(MessageType message_type, FrameBuffer* buf) {
  //   if (message_type != MessageType::kNormal) {
  //     return;
  //   }

  //   is_running_ = rpc_server_.ProcessOnePacket();
  //   session_.ClearReceiveBuffer();
  // }

  virtual void HandleCompleteMessage(MessageType message_type, FrameBuffer* buf) = 0;

  static void HandleCompleteMessageCb(void* context, MessageType message_type, FrameBuffer* buf) {
    static_cast<MicroTransport*>(context)->HandleCompleteMessage(message_type, buf);
  }
};

}  // namespace micro_rpc
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CRT_RPC_COMMON_TRANSPORT_H_