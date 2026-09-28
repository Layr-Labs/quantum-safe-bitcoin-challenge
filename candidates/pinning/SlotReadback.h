// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace qsb {

// One counter plus the unchanged 1024 device hit slots. Only the counter and
// first 64 hits cross to the host, in one transfer on the completion stream.
class SlotReadback {
 public:
  SlotReadback() : device_(nullptr), host_(nullptr) {}
  SlotReadback(const SlotReadback&) = delete;
  SlotReadback& operator=(const SlotReadback&) = delete;

  ~SlotReadback() {
    if (device_) cudaFree(device_);
    if (host_) cudaFreeHost(host_);
  }

  cudaError_t init() {
    if (device_ || host_) return cudaErrorInvalidValue;
    cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&device_),
                               (1 + 1024) * sizeof(uint32_t));
    if (e == cudaSuccess)
      e = cudaHostAlloc(reinterpret_cast<void**>(&host_),
                        (1 + 64) * sizeof(uint32_t), cudaHostAllocDefault);
    return e;
  }

  uint32_t* device_count() const { return device_; }
  uint32_t* device_indices() const { return device_ ? device_ + 1 : nullptr; }

  // The caller must successfully synchronize the recorded completion event
  // before reading these values or reusing this slot's buffers.
  uint32_t count() const { return host_[0]; }
  const uint32_t* indices() const { return host_ ? host_ + 1 : nullptr; }

  cudaError_t enqueue(cudaStream_t stream, cudaEvent_t done) {
    if (!device_ || !host_) return cudaErrorInvalidValue;
    cudaError_t e = cudaMemcpyAsync(host_, device_, (1 + 64) * sizeof(uint32_t),
                                    cudaMemcpyDeviceToHost, stream);
    if (e == cudaSuccess) e = cudaEventRecord(done, stream);
    return e;
  }

 private:
  uint32_t *device_, *host_;
};

}  // namespace qsb
