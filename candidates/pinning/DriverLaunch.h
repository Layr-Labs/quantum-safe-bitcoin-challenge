// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <stdio.h>
#include <stdlib.h>

// Resolve the v4000 launch ABI once. Only non-null carrier streams use it;
// startup/default-stream launches retain the original runtime semantics.
namespace qsb_driver {
static PFN_cuLaunchKernel_v4000 launch = nullptr;
static void init() {
    void *entry = nullptr;
    cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
    cudaError_t error = cudaGetDriverEntryPointByVersion(
        "cuLaunchKernel", &entry, 4000, cudaEnableLegacyStream, &status);
    if (error == cudaSuccess && status == cudaDriverEntryPointSuccess && entry)
        launch = reinterpret_cast<PFN_cuLaunchKernel_v4000>(entry);
    else {
        launch = nullptr;
        (void)cudaGetLastError();
    }
    printf("  Carrier non-default launches: %s\n", launch ? "direct driver" : "runtime fallback");
}
static cudaError_t enqueue(cudaKernel_t kernel, dim3 grid, dim3 block,
                           void **args, cudaStream_t stream) {
    if (!launch || !stream)
        return cudaLaunchKernel((const void *)kernel, grid, block, args, 0, stream);
    // Runtime cudaKernel_t and driver CUkernel are interchangeable. The
    // context-less handle takes its execution context from this exact stream.
    CUfunction function = reinterpret_cast<CUfunction>(reinterpret_cast<CUkernel>(kernel));
    CUresult error = launch(function, grid.x, grid.y, grid.z,
                            block.x, block.y, block.z, 0,
                            reinterpret_cast<CUstream>(stream), args, nullptr);
    if (error != CUDA_SUCCESS) {
        fprintf(stderr, "Native carrier direct launch failed (CUDA driver %d)\n", (int)error);
        exit(2);
    }
    return cudaSuccess;
}
} // namespace qsb_driver
