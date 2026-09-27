// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <cuda_runtime.h>

namespace qsb {

// Host scheduling only. Events establish dependencies; stream priority is
// merely a scheduling hint and is never relied on for correctness.
class CompletionLane {
 public:
  enum Mode { SameStream = 0, RootsPriority = 1, TailPriority = 2, SplitControl = 3 };

  CompletionLane() : prepare_(nullptr), auxiliary_(nullptr),
                     prepared_(nullptr), roots_ready_(nullptr), mode_(SameStream) {}
  CompletionLane(const CompletionLane&) = delete;
  CompletionLane& operator=(const CompletionLane&) = delete;

  ~CompletionLane() {
    if (roots_ready_) cudaEventDestroy(roots_ready_);
    if (prepared_) cudaEventDestroy(prepared_);
    if (auxiliary_) cudaStreamDestroy(auxiliary_);
  }

  cudaError_t init(cudaStream_t prepare, int mode) {
    if (mode < SameStream || mode > SplitControl) return cudaErrorInvalidValue;
    prepare_ = prepare;
    mode_ = static_cast<Mode>(mode);
    if (mode_ == SameStream) return cudaSuccess;
    int least = 0, greatest = 0;
    cudaError_t e = cudaDeviceGetStreamPriorityRange(&least, &greatest);
    if (e != cudaSuccess) return e;
    const int priority = mode_ == SplitControl ? 0 : greatest;
    e = cudaStreamCreateWithPriority(&auxiliary_, cudaStreamNonBlocking, priority);
    if (e == cudaSuccess)
      e = cudaEventCreateWithFlags(&prepared_, cudaEventDisableTiming);
    if (e == cudaSuccess && mode_ == RootsPriority)
      e = cudaEventCreateWithFlags(&roots_ready_, cudaEventDisableTiming);
    return e;
  }

  cudaError_t begin_roots(cudaStream_t& stream) {
    if (mode_ == SameStream) return cudaSuccess;
    cudaError_t e = cudaEventRecord(prepared_, prepare_);
    if (e == cudaSuccess) e = cudaStreamWaitEvent(auxiliary_, prepared_, 0);
    if (e == cudaSuccess) stream = auxiliary_;
    return e;
  }

  cudaError_t end_roots(cudaStream_t& stream) {
    if (mode_ != RootsPriority) return cudaSuccess;
    cudaError_t e = cudaEventRecord(roots_ready_, auxiliary_);
    if (e == cudaSuccess) e = cudaStreamWaitEvent(prepare_, roots_ready_, 0);
    if (e == cudaSuccess) stream = prepare_;
    return e;
  }

  cudaStream_t completion_stream() const {
    return mode_ == TailPriority || mode_ == SplitControl ? auxiliary_ : prepare_;
  }

 private:
  cudaStream_t prepare_, auxiliary_;
  cudaEvent_t prepared_, roots_ready_;
  Mode mode_;
};

}  // namespace qsb
