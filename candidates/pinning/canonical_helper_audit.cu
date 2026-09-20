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
    uint64_t ac[4],bc[4];
    for(int j=0;j<4;++j){ac[j]=a[j];bc[j]=b[j];}
    for(int t=0;t<2;++t){uint64_t *v=t?bc:ac;
        if(v[3]==UINT64_MAX&&v[2]==UINT64_MAX&&v[1]==UINT64_MAX&&v[0]>=0xFFFFFFFEFFFFFC2FULL){
            v[0]-=0xFFFFFFFEFFFFFC2FULL;v[1]=v[2]=v[3]=0;}}
    _ModAddCanonicalPair(r,ac,bc);
    for(int j=0;j<4;++j)out[36ull*i+0+j]=r[j];
    for(int j=0;j<4;++j)r[j]=ac[j];
    _ModAddCanonicalPair(r,r,bc);
    for(int j=0;j<4;++j)out[36ull*i+4+j]=r[j];
    for(int j=0;j<4;++j)r[j]=bc[j];
    _ModAddCanonicalPair(r,ac,r);
    for(int j=0;j<4;++j)out[36ull*i+8+j]=r[j];
    _ModSubCanonicalRhs(r,a,bc);
    for(int j=0;j<4;++j)out[36ull*i+12+j]=r[j];
    for(int j=0;j<4;++j)r[j]=a[j];
    _ModSubCanonicalRhs(r,r,bc);
    for(int j=0;j<4;++j)out[36ull*i+16+j]=r[j];
    for(int j=0;j<4;++j)r[j]=bc[j];
    _ModSubCanonicalRhs(r,a,r);
    for(int j=0;j<4;++j)out[36ull*i+20+j]=r[j];
    _ModAddLazy(r,a,b);
    for(int j=0;j<4;++j)out[36ull*i+24+j]=r[j];
    for(int j=0;j<4;++j)r[j]=a[j];
    _ModAddLazy(r,r,b);
    for(int j=0;j<4;++j)out[36ull*i+28+j]=r[j];
    for(int j=0;j<4;++j)r[j]=b[j];
    _ModAddLazy(r,a,r);
    for(int j=0;j<4;++j)out[36ull*i+32+j]=r[j];
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
    std::vector<uint64_t> inputs(8ull*n), outputs(36ull*n);
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
    bool ok = fwrite(outputs.data(), 288, n, file) == n;
    ok = (fclose(file) == 0) && ok;
    CUDA_CHECK(cudaFree(device_inputs));
    CUDA_CHECK(cudaFree(device_outputs));
    return ok ? 0 : 2;
}
