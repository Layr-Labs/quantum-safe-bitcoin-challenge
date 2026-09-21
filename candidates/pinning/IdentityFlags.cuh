// Identity protocol: one explicit bit per candidate, no in-band field-value sentinel.
// Allocate ceil(n/32) uint32 words per stream. Producer and consumer use the same stream.
#pragma once
__device__ __forceinline__ void qsb_s16_publish_identity(
    uint32_t *flags,unsigned idx,unsigned n,unsigned special) {
    // All lanes of every full prepare warp must reach this before any special return.
    const uint32_t bits=__ballot_sync(0xffffffffu,idx<n && special==1u);
    if((threadIdx.x&31u)==0u && idx<n)flags[idx>>5]=bits;
}
__device__ __forceinline__ bool qsb_s16_identity_flag(
    const uint32_t *flags,unsigned idx) {
    // Called only for active finish lanes in the existing zero-tbar branch.
    return ((flags[idx>>5]>>(idx&31u))&1u)!=0u;
}
