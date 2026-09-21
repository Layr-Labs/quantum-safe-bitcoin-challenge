// Recovery helper for a positively identified scalar-zero lane, not arbitrary singular lanes.
#pragma once
__device__ __forceinline__ uint32_t qsb_serial16_identity_finish(
    const uint64_t *a,const uint64_t *b,uint64_t *x1,uint64_t *x2) {
    #pragma unroll
    for(int i=0;i<4;i++){x1[i]=a[i];x2[i]=a[i];}
    const uint32_t parity=uint32_t(b[0])&1u;
    return parity|((parity^1u)<<1);
}
