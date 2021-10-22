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
 * \file aot_server.h
 * \brief AOT server implementation XXX
 *
 * \note This file do not depend on c++ std or c std,
 *       and only depends on TVM's C runtime API.
 */

#ifndef TVM_RUNTIME_MINRPC_AOT_SERVER_H_
#define TVM_RUNTIME_MINRPC_AOT_SERVER_H_

#define DMLC_LITTLE_ENDIAN true
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>

#include "../../../support/generic_arena.h"
#include "../../minrpc/rpc_reference.h"

namespace tvm {
namespace runtime {

/*!
 * \brief A minimum RPC server that only depends on the tvm C runtime..
 *
 *  All the dependencies are provided by the io arguments.
 *
 * \tparam TIOHandler IO provider to provide io handling.
 *         An IOHandler needs to provide the following functions:
 *         - PosixWrite, PosixRead, Close: posix style, read, write, close API.
 *         - MessageStart(num_bytes), MessageDone(): framing APIs.
 *         - Exit: exit with status code.
 */
template <typename TIOHandler>
class AOTServer {
 public:
  /*!
   * \brief Constructor.
   * \param io The IO handler.
   */
  explicit AOTServer(TIOHandler* io) : io_(io), arena_(PageAllocator(io)) {}

  /*! \brief Process a single request.
   *
   * \return true when the server should continue processing requests. false when it should be
   *  shutdown.
   */
  bool ProcessOnePacket() {
    // RPCCode code;
    uint64_t packet_len;

    arena_.RecycleAll();
    allow_clean_shutdown_ = true;

    this->Read(&packet_len);
    if (packet_len == 0) return true;
    TVMLogf("mehrdad: %d\n", (int)packet_len);

    // this->Read(&code);

    // allow_clean_shutdown_ = false;

    // if (code >= RPCCode::kSyscallCodeStart) {
    //   this->HandleSyscallFunc(code);
    // } else {
    //   switch (code) {
    //     case RPCCode::kCallFunc: {
    //       HandleNormalCallFunc();
    //       break;
    //     }
    //     case RPCCode::kInitServer: {
    //       HandleInitServer();
    //       break;
    //     }
    //     case RPCCode::kCopyFromRemote: {
    //       HandleCopyFromRemote();
    //       break;
    //     }
    //     case RPCCode::kCopyToRemote: {
    //       HandleCopyToRemote();
    //       break;
    //     }
    //     case RPCCode::kShutdown: {
    //       this->Shutdown();
    //       return false;
    //     }
    //     default: {
    //       this->ThrowError(RPCServerStatus::kUnknownRPCCode);
    //       break;
    //     }
    //   }
    // }

    return true;
  }

  template <typename T>
  void Read(T* data) {
    static_assert(std::is_pod<T>::value, "need to be trival");
    this->ReadRawBytes(data, sizeof(T));
  }
    void Shutdown() {
    arena_.FreeAll();
    io_->Close();
  }
  //   void ThrowError(RPCServerStatus code, RPCCode info = RPCCode::kNone) {
  //   io_->Exit(static_cast<int>(code));
  // }
 private:
 
      void ReadRawBytes(void* data, size_t size) {
    uint8_t* buf = reinterpret_cast<uint8_t*>(data);
    size_t ndone = 0;
    while (ndone < size) {
      ssize_t ret = io_->PosixRead(buf, size - ndone);
      if (ret == 0) {
        if (allow_clean_shutdown_) {
          this->Shutdown();
          io_->Exit(0);
        } else {
          // this->ThrowError(RPCServerStatus::kReadError);
        }
      }
      if (ret == -1) {
        // this->ThrowError(RPCServerStatus::kReadError);
      }
      ndone += ret;
      buf += ret;
    }
  }

      

   // Internal allocator that redirects alloc to TVM's C API.
  class PageAllocator {
   public:
    using ArenaPageHeader = tvm::support::ArenaPageHeader;

    explicit PageAllocator(TIOHandler* io) : io_(io) {}

    ArenaPageHeader* allocate(size_t min_size) {
      size_t npages = ((min_size + kPageSize - 1) / kPageSize);
      void* data;

      if (TVMDeviceAllocDataSpace(DLDevice{kDLCPU, 0}, npages * kPageSize, kPageAlign,
                                  DLDataType{kDLInt, 1, 1}, &data) != 0) {
        // io_->Exit(static_cast<int>(RPCServerStatus::kAllocError));
      }

      ArenaPageHeader* header = static_cast<ArenaPageHeader*>(data);
      header->size = npages * kPageSize;
      header->offset = sizeof(ArenaPageHeader);
      return header;
    }

    void deallocate(ArenaPageHeader* page) {
      if (TVMDeviceFreeDataSpace(DLDevice{kDLCPU, 0}, page) != 0) {
        // io_->Exit(static_cast<int>(RPCServerStatus::kAllocError));
      }
    }

    static const constexpr int kPageSize = 2 << 10;
    static const constexpr int kPageAlign = 8;

   private:
    TIOHandler* io_;
  };

  /*! \brief IO handler. */
  TIOHandler* io_;
    /*! \brief internal arena. */
  support::GenericArena<PageAllocator> arena_;
      bool allow_clean_shutdown_{true};
};

}  // namespace runtime
}  // namespace tvm
#endif  //TVM_RUNTIME_MINRPC_AOT_SERVER_H_