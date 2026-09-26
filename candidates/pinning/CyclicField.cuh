// SPDX-License-Identifier: GPL-3.0-only
// Python-validated integration; no local native compilation or GPU execution.
// Four independent secp256k1 products per full warp, eight lanes/product.
// A and B each supply one 32-bit limb per lane. Return one noncanonical limb.
// All 32 lanes participate in EVERY shuffle and ballot. No divergent entry.
#pragma once
namespace qsb_cyclic_field_research {
constexpr unsigned full=0xffffffffu;

__device__ __forceinline__ uint32_t normalize8(
    uint64_t column,unsigned lane,uint32_t &top){
    const unsigned d=lane&7u,group=lane&24u;
    const uint32_t lo=(uint32_t)column,hi=(uint32_t)(column>>32);
    const uint32_t prev0=__shfl_up_sync(full,hi,1,8);
    const uint32_t sum=lo+(d?prev0:0u);
    const unsigned gen=(__ballot_sync(full,sum<lo)>>group)&255u;
    const unsigned prop=(__ballot_sync(full,sum==0xffffffffu)>>group)&255u;
    const unsigned carry=(prop+(gen<<1))^prop;
    top=__shfl_sync(full,hi,7,8)+((carry>>8)&1u);
    return sum+((carry>>d)&1u);
}

__device__ __forceinline__ uint32_t multiply8(uint32_t a,uint32_t b,unsigned lane){
    const unsigned d=lane&7u;
    uint64_t l0=0,l1=0,h0=0,h1=0;
    #pragma unroll
    for(unsigned i=0;i<8;++i){
        const uint32_t av=__shfl_sync(full,a,i,8);
        const uint32_t bv=__shfl_sync(full,b,(d-i)&7u,8);
        const uint64_t product=(uint64_t)av*bv;
        const uint32_t lo=(uint32_t)product,hi=(uint32_t)(product>>32);
        // Cyclic convolution: every physical product is useful. Its true
        // exponent is d when d>=i, or d+8 otherwise. Retain both halves.
        l0+=d>=i?lo:0u; l1+=d>=i?hi:0u;
        h0+=d<i?lo:0u;  h1+=d<i?hi:0u;
    }
    const uint64_t lprev0=__shfl_up_sync(full,l1,1,8);
    const uint64_t hprev0=__shfl_up_sync(full,h1,1,8);
    const uint64_t llast=__shfl_sync(full,l1,7,8);
    const uint64_t low_column=l0+(d?lprev0:0ull);
    const uint64_t high_column=h0+(d?hprev0:llast);
    const uint64_t prev_high0=__shfl_up_sync(full,high_column,1,8);
    const uint64_t extra=__shfl_sync(full,high_column,7,8);
    // Fold unnormalized product columns LINEARLY. All column sums fit <2^46.
    // Thus the 512-bit product does not need a separate carry normalization.
    uint64_t column=low_column+977ull*high_column+(d?prev_high0:0ull);
    uint32_t top;
    uint32_t t=normalize8(column,lane,top);
    const uint64_t high=extra+(uint64_t)top; // may exceed 2^32; do not truncate
    const uint32_t q0=(uint32_t)high,q1=(uint32_t)(high>>32);
    column=t;
    if(d==0)column+=977ull*q0;
    if(d==1)column+=(uint64_t)q0+977ull*q1;
    if(d==2)column+=q1;
    t=normalize8(column,lane,top);
    const uint32_t cf=top; // <=1, retained even on exceptional dense inputs
    column=t;
    if(d==0)column+=977ull*cf;
    if(d==1)column+=cf;
    t=normalize8(column,lane,top); // mathematically top==0
    return t;
}
} // namespace qsb_cyclic_field_research
