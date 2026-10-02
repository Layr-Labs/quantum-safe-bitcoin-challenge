// Research-only dense byte convolution. All 32 lanes must participate.
// Each operand is distributed one little-endian uint32 per lane0..7;
// lanes8..31 must contain zero. Output is two raw coefficients per lane,
// not a normalized integer or reduced field value.
#pragma once
#include <stdint.h>

__device__ __forceinline__ uint32_t qsb_tensor_window(uint32_t word, int byte) {
    const int q=byte>>2;
    const uint32_t lo=__shfl_sync(0xffffffffu,word,q&31);
    const uint32_t hi=__shfl_sync(0xffffffffu,word,(q+1)&31);
    return __funnelshift_r(lo,hi,(byte&3)*8);
}

__device__ __forceinline__ void qsb_tensor_coefficients(
    uint32_t aw, uint32_t bw, int lane, uint32_t &c0, uint32_t &c1
) {
    const int g=lane>>2, t=lane&3;
    uint32_t d0=0,d1=0,d2=0,d3=0;
    #pragma unroll
    for(int phase=0;phase<64;phase+=32) {
        const int k=phase+4*t;
        const uint32_t a0=qsb_tensor_window(aw,k-g);
        const uint32_t a1=qsb_tensor_window(aw,k-g-8);
        const uint32_t a2=qsb_tensor_window(aw,k+16-g);
        const uint32_t a3=qsb_tensor_window(aw,k+8-g);
        const uint32_t b0=__byte_perm(qsb_tensor_window(bw,4+8*g-k),0,0x0123);
        const uint32_t b1=__byte_perm(qsb_tensor_window(bw,4+8*g-k-16),0,0x0123);
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    c0=d0; c1=d1;
}

__device__ __forceinline__ uint32_t qsb_tensor_get_coefficient(
    uint32_t c0, uint32_t c1, int coefficient
) {
    const int column=coefficient>>3;
    const int owner=4*(7-(coefficient&7))+(column>>1);
    const uint32_t a=__shfl_sync(0xffffffffu,c0,owner);
    const uint32_t b=__shfl_sync(0xffffffffu,c1,owner);
    return (column&1)?b:a;
}

// Normalize one uint64 raw limb per lane into base2^32 words. Used for
// sixteen product limbs; lanes16..31 are zero. A predecessor's upper half
// is added first; only a one-bit ripple remains. Two ballots locate the
// latest generate/stop before each lane. No divergent shuffle or ballot.
__device__ __forceinline__ uint32_t qsb_tensor_normalize(uint64_t raw, int lane) {
    uint32_t hi=(uint32_t)(raw>>32);
    const uint32_t previous=__shfl_up_sync(0xffffffffu,hi,1);
    const uint64_t adjusted=(uint64_t)(uint32_t)raw+(lane?previous:0);
    const uint32_t lo=(uint32_t)adjusted;
    const uint32_t generates=__ballot_sync(0xffffffffu,(adjusted>>32)!=0);
    const uint32_t propagates=__ballot_sync(0xffffffffu,lo==0xffffffffu);
    const uint32_t before=(1u<<lane)-1u;
    const uint32_t prior_g=generates&before;
    const uint32_t prior_stop=(~propagates)&before;
    const uint32_t carry=(prior_g!=0 && __clz(prior_g)<=__clz(prior_stop));
    return lo+carry;
}

__device__ __forceinline__ uint32_t qsb_tensor_product_word(
    uint32_t aw,uint32_t bw,int lane
) {
    uint32_t c0,c1;
    qsb_tensor_coefficients(aw,bw,lane,c0,c1);
    // Lanes16..31 still execute all collectives, then contribute zero.
    const int base=4*(lane&15);
    const uint32_t t0=qsb_tensor_get_coefficient(c0,c1,base);
    const uint32_t t1=qsb_tensor_get_coefficient(c0,c1,base+1);
    const uint32_t t2=qsb_tensor_get_coefficient(c0,c1,base+2);
    const uint32_t t3=qsb_tensor_get_coefficient(c0,c1,base+3);
    uint64_t raw=(uint64_t)t0+((uint64_t)t1<<8)+((uint64_t)t2<<16)+((uint64_t)t3<<24);
    if(lane>=16) raw=0;
    return qsb_tensor_normalize(raw,lane);
}
