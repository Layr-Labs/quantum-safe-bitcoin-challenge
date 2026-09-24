#pragma once
#include <stdint.h>

/* Research handoff: geometry only, no field arithmetic or GPU implementation.
 * First table record i represents (bias+i)A, where
 * bias=(42639944 << 101)-(1 << 23). Other records represent
 * (2*i+1)*2^(shift(c)-1)A. A=neg_r_inv*G, not A/2.
 * Exactly 138760484 records / 8880670976 bytes. */
#ifdef __CUDACC__
#define GLV10_HD __host__ __device__ __forceinline__
#else
#define GLV10_HD static inline
#endif

GLV10_HD unsigned glv10_entries(int c) {
    return c==0 ? 16777216u : c==4 ? 21319972u : 33554432u;
}
GLV10_HD unsigned glv10_offset(int c) {
    return c==0 ? 0u : c==1 ? 16777216u : c==2 ? 50331648u :
           c==3 ? 83886080u : 117440512u;
}
GLV10_HD unsigned glv10_shift(int c) {
    return c==0 ? 0u : c==1 ? 24u : c==2 ? 50u : c==3 ? 76u : 102u;
}
GLV10_HD uint32_t glv10_code(const uint64_t mag[2], unsigned sign, int c) {
    const unsigned shift=glv10_shift(c);
    uint64_t wide;
    if (shift<64u) {
        wide=mag[0]>>shift;
        if (shift) wide|=mag[1]<<(64u-shift);
    } else wide=mag[1]>>(shift-64u);
    uint32_t f=(uint32_t)wide,idx,negative;
    if (c==0) {
        idx=f&0xffffffu; negative=0;
    } else if (c==4) {
        const int32_t d=(int32_t)(2u*f)-42639943;
        negative=(uint32_t)d>>31;
        const uint32_t magnitude=((uint32_t)d^(0u-negative))+negative;
        idx=(magnitude-1u)>>1;
    } else {
        f&=0x3ffffffu;
        negative=1u-(f>>25);
        idx=(f^(0u-negative))&0x1ffffffu;
    }
    return (glv10_offset(c)+idx)|((negative^sign)<<31);
}
#undef GLV10_HD
