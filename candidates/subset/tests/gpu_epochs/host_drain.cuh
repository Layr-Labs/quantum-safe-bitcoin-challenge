// Host-drain: stream-ordered occupancy memset + result copy. No extra
// device synchronize before the hit-buffer memcpy, and no per-hit stdout.
// The consumer publishes hits and the occupancy word through the same
// stream that launched it; a second full-device barrier only parks the
// host (and the next launch) until every other stream drains.
//
// One mechanism. Kernel arithmetic, inverse, SHA, and XYZZ are untouched.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Zero the device occupancy word on `stream`. Replaces a producer-side
// scalar store and a synchronous cudaMemset that implied a null-stream
// drain before launch.
inline cudaError_t host_drain_memset_u32(uint32_t *dst, uint32_t n, cudaStream_t stream)
{
    return cudaMemsetAsync(dst, 0, (size_t)n * sizeof(uint32_t), stream);
}

// Pull the occupancy word (and the packed first-hit records that sit
// immediately after it) back on the same stream. The caller must not
// cudaDeviceSynchronize() before this copy: the memcpy is already
// ordered after the kernel that filled the buffers. A later
// cudaStreamSynchronize(stream) is the only wait.
inline cudaError_t host_drain_copy_u32(
    uint32_t *host,
    const uint32_t *device,
    uint32_t n,
    cudaStream_t stream)
{
    return cudaMemcpyAsync(
        host,
        device,
        (size_t)n * sizeof(uint32_t),
        cudaMemcpyDeviceToHost,
        stream);
}
