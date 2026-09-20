// Native diagnostic for the shipped header. Run only under the shared GPU lease.
// Compile: nvcc -O3 audit_final_fold.cu -o audit_final_fold
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include "GPUMath.h"

__global__ void audit_final_fold(const uint64_t *in, uint64_t *out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t a[4], b[4], r[4];
    for (int j = 0; j < 4; ++j) {
        a[j] = in[8ull*i+j]; b[j] = in[8ull*i+4+j];
    }
    _ModMultCore(r, a, b);
    for (int j = 0; j < 4; ++j) out[20ull*i+j] = r[j];
    for (int j = 0; j < 4; ++j) r[j] = a[j];
    _ModMultCore(r, r, b);
    for (int j = 0; j < 4; ++j) out[20ull*i+4+j] = r[j];
    for (int j = 0; j < 4; ++j) r[j] = b[j];
    _ModMultCore(r, a, r);
    for (int j = 0; j < 4; ++j) out[20ull*i+8+j] = r[j];
    _ModSqr(r, a);
    for (int j = 0; j < 4; ++j) out[20ull*i+12+j] = r[j];
    for (int j = 0; j < 4; ++j) r[j] = a[j];
    _ModSqr(r, r);
    for (int j = 0; j < 4; ++j) out[20ull*i+16+j] = r[j];
}

#define CUDA_CHECK(expr) do { \
    cudaError_t error = (expr); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(error)); return 3; \
    } \
} while (0)

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    FILE *file = fopen(argv[1], "rb");
    if (!file) return 2;
    unsigned n;
    if (fread(&n, 4, 1, file) != 1 || !n || n > 1000000) return 2;
    std::vector<uint64_t> inputs(8ull*n), outputs(20ull*n);
    if (fread(inputs.data(), 64, n, file) != n) return 2;
    fclose(file);
    uint64_t *device_inputs, *device_outputs;
    CUDA_CHECK(cudaMalloc(&device_inputs, inputs.size()*8));
    CUDA_CHECK(cudaMalloc(&device_outputs, outputs.size()*8));
    CUDA_CHECK(cudaMemcpy(device_inputs, inputs.data(), inputs.size()*8, cudaMemcpyHostToDevice));
    audit_final_fold<<<(n+127)/128,128>>>(device_inputs, device_outputs, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(outputs.data(), device_outputs, outputs.size()*8, cudaMemcpyDeviceToHost));
    file = fopen(argv[2], "wb");
    if (!file) return 2;
    bool ok = fwrite(outputs.data(), 160, n, file) == n;
    ok = (fclose(file) == 0) && ok;
    CUDA_CHECK(cudaFree(device_inputs));
    CUDA_CHECK(cudaFree(device_outputs));
    return ok ? 0 : 2;
}
