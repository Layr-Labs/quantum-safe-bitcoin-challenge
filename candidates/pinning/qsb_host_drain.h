#pragma once

/* Host-drain of the scored search loop. Device kernels, algebra, L2
 * persist, and the 32 KiB stack ceiling stay at HEAD. */

#include <cstdint>
#include <cuda_runtime.h>

static inline cudaError_t qsb_host_drain_reset(uint32_t *d_hit_cnt) {
    return cudaMemset(d_hit_cnt, 0, sizeof(uint32_t));
}

static inline cudaError_t qsb_host_drain_counter(uint32_t *h_hit, const uint32_t *d_hit_cnt) {
    cudaError_t err = cudaMemcpy(h_hit, d_hit_cnt, sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost);
    if (err == cudaSuccess)
        err = cudaGetLastError();
    return err;
}
