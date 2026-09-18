// Static code-generation screen only. No benchmark or production integration.
#include <cuda_runtime.h>
#include <stdint.h>
#include "fragment.cuh"
#include "control_math.h"

extern "C" __global__ void tensor_coefficients(
    const uint32_t *a,const uint32_t *b,uint32_t *out,unsigned n
) {
    const unsigned global=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned item=global>>5,lane=global&31;
    if(item>=n)return; // Whole-warp decision; launch blockDim must be multiple32.
    const uint32_t aw=lane<8?a[item*8+lane]:0;
    const uint32_t bw=lane<8?b[item*8+lane]:0;
    uint32_t c0,c1;
    qsb_tensor_coefficients(aw,bw,lane,c0,c1);
    out[item*64+lane*2]=c0;out[item*64+lane*2+1]=c1;
}

extern "C" __global__ void tensor_full_product(
    const uint32_t *a,const uint32_t *b,uint32_t *out,unsigned n
) {
    const unsigned global=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned item=global>>5,lane=global&31;
    if(item>=n)return;
    const uint32_t aw=lane<8?a[item*8+lane]:0;
    const uint32_t bw=lane<8?b[item*8+lane]:0;
    const uint32_t word=qsb_tensor_product_word(aw,bw,lane);
    if(lane<16)out[item*16+lane]=word;
}

extern "C" __global__ void scalar_field_control(
    const uint64_t *a,const uint64_t *b,uint64_t *out,unsigned n
) {
    const unsigned item=blockIdx.x*blockDim.x+threadIdx.x;
    if(item>=n)return;
    uint64_t aa[4],bb[4],rr[4];
    #pragma unroll
    for(int i=0;i<4;i++){aa[i]=a[item*4+i];bb[i]=b[item*4+i];}
    _ModMultCore(rr,aa,bb);
    #pragma unroll
    for(int i=0;i<4;i++)out[item*4+i]=rr[i];
}

int main(){return 0;}
