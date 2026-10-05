// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstring>

namespace qsb {
// Non-owning cache for one slot's fixed-size input. The caller must wait for
// the slot's completion before enqueue(), including before a cache-hit check:
// the pinned mirror can still be a source for a previous asynchronous copy.
class SlotInputCache {
 public:
  cudaError_t init(void* device, void* pinned_host, size_t bytes) {
    if (device_ || host_ || !device || !pinned_host || !bytes)
      return cudaErrorInvalidValue;
    device_ = device;
    host_ = pinned_host;
    bytes_ = bytes;
    return cudaSuccess;
  }

  cudaError_t enqueue(const void* input, size_t bytes, cudaStream_t stream) {
    if (!device_ || !host_ || !input || bytes != bytes_)
      return cudaErrorInvalidValue;
    // Never compare against the initially uninitialized pinned allocation.
    if (uploaded_ && std::memcmp(host_, input, bytes_) == 0)
      return cudaSuccess;
    uploaded_ = false;
    std::memcpy(host_, input, bytes_);
    cudaError_t e = cudaMemcpyAsync(device_, host_, bytes_, cudaMemcpyHostToDevice, stream);
    if (e == cudaSuccess) uploaded_ = true;
    return e;
  }

 private:
  void* device_ = nullptr;
  void* host_ = nullptr;
  size_t bytes_ = 0;
  bool uploaded_ = false;
};
}  // namespace qsb
