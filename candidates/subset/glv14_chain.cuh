// GPL-3.0-only. GLV14 table walk for the subset filter and its exact replay.
// Grouped one-beta architecture: public GLV40 4b77964f. Exact split constants
// and arithmetic retain bitcoin-core/secp256k1's MIT notice in GLVScalar.cuh.
// The seven-window bounded-top recoder and shared-tree overlay are new here.
#pragma once
#include "GLVScalar.cuh"

// 14 code planes require 14 KiB. The later inverse uses these same 16 KiB
// as four product rows; its independent 8 KiB inverse rows remain unchanged.
// A and B overwrite only their own per-thread columns, so they need no
// intervening block barrier. The inverse inserts a barrier BEFORE leaf writes.
__device__ __forceinline__ uint64_t (*qsb_glv14_tree_products())[512] {
    __shared__ uint64_t products[4][512];
    return products;
}
__device__ __forceinline__ uint64_t qsb_glv14_extract(const uint64_t m[2], unsigned shift) {
    if(shift<64u) {
        uint64_t v=m[0]>>shift;
        if(shift) v|=m[1]<<(64u-shift);
        return v;
    }
    return m[1]>>(shift-64u);
}
__device__ __forceinline__ uint32_t qsb_glv14_code(
    const uint64_t mag[2][2], const unsigned sign[2], unsigned term) {
    const unsigned side=term<7u?1u:0u, ch=term<7u?term:term-7u;
    uint32_t f=(uint32_t)qsb_glv14_extract(mag[side],gt_shift(ch));
    uint32_t idx,neg;
    if(ch==0u) {
        idx=f&((1u<<18)-1u); neg=0;
    } else if(ch==6u) {
        const int32_t d=(int32_t)(2u*f)-333125;
        neg=(uint32_t)d>>31;
        const uint32_t ad=((uint32_t)d^(0u-neg))+neg;
        idx=(ad-1u)>>1;
    } else {
        const unsigned bits=ch==1u?19u:18u;
        f&=(1u<<bits)-1u;
        neg=1u-(f>>(bits-1u));
        idx=(f^(0u-neg))&((1u<<(bits-1u))-1u);
    }
    return (gt_offset(ch)+idx)|((neg^sign[side])<<31);
}
__device__ __forceinline__ unsigned qsb_glv14_decode(const uint64_t k[4]) {
    uint64_t mag[2][2]; unsigned sign[2];
    q9_glv_split(k,mag[0],mag[1],&sign[0],&sign[1]);
    volatile uint32_t *codes=(volatile uint32_t*)qsb_glv14_tree_products();
    #pragma unroll
    for(unsigned term=0;term<14u;term++)
        codes[(size_t)term*256u+threadIdx.x]=qsb_glv14_code(mag,sign,term);
    return ((mag[1][0]|mag[1][1])!=0) |
           (((mag[0][0]|mag[0][1])!=0)<<1);
}
__device__ __forceinline__ void qsb_glv14_load_filter(
    const uint8_t *table, unsigned term, uint64_t *x, uint64_t *y) {
    volatile uint32_t *codes=(volatile uint32_t*)qsb_glv14_tree_products();
    const uint32_t code=codes[(size_t)term*256u+threadIdx.x];
    gt_load_signed_flat_f(table,0u,code&0x1fffffu,(uint64_t)(code>>31),x,y);
}
__device__ __forceinline__ void qsb_glv14_beta(uint64_t *X) {
    uint64_t beta[4]={0xC1396C28719501EEULL,0x9CF0497512F58995ULL,
                     0x6E64479EAC3434E9ULL,0x7AE96A2B657C0710ULL};
    _ModMult(X,X,beta);
}
__device__ __forceinline__ void qsb_glv14_infinity(
    uint64_t *X,uint64_t *Y,uint64_t *ZZ,uint64_t *ZZZ) {
    #pragma unroll
    for(int i=0;i<4;i++) {X[i]=ZZ[i]=ZZZ[i]=0;Y[i]=(i==0);}
}
__device__ __forceinline__ void qsb_glv14_filter_chain(
    uint64_t *X,uint64_t *Y,uint64_t *ZZ,uint64_t *ZZZ,
    const uint64_t k[4],const uint8_t *table,uint32_t &bad) {
    const unsigned nonzero=qsb_glv14_decode(k);
    if(!nonzero) {qsb_glv14_infinity(X,Y,ZZ,ZZZ);return;}
    const unsigned first=(nonzero&1u)?0u:7u;
    uint64_t x0[4],y0[4],x1[4],y1[4];
    qsb_glv14_load_filter(table,first,x0,y0);
    qsb_glv14_load_filter(table,first+1u,x1,y1);
    qsb_filter_point_seed(X,Y,ZZ,ZZZ,x0,y0,x1,y1,bad);
    #pragma unroll 1
    for(unsigned term=first+2u;term<13u;term++) {
        if(term==7u)qsb_glv14_beta(X);
        qsb_glv14_load_filter(table,term,x1,y1);
        qsb_filter_point_add<true>(X,Y,ZZ,ZZZ,x1,y1,y0,bad);
#if !QSB_CHAIN_ANCHOR_UPDATE
        Load256(y0,y1);
#endif
    }
    qsb_glv14_load_filter(table,13u,x1,y1);
    // Subset's helper restores POSITIVE actual Y for its existing recovery.
    // Pinning instead carries negative actual Y; its finish cannot be copied.
    qsb_filter_last_add(X,Y,ZZ,ZZZ,x1,y1,y0,bad);
}

// Exact verification uses no shared digit arena and no new barriers. Hit-check
// lanes may return independently. Full-precision signed loads, point helpers
// and complete final addition remain separate from the speculative filter.
__device__ __forceinline__ void qsb_glv14_exact_chain(
    uint64_t *X,uint64_t *Y,uint64_t *ZZ,uint64_t *ZZZ,
    const uint64_t k[4],const uint8_t *table) {
    uint64_t mag[2][2]; unsigned sign[2];
    q9_glv_split(k,mag[0],mag[1],&sign[0],&sign[1]);
    const bool q_nonzero=(mag[1][0]|mag[1][1])!=0;
    if(!q_nonzero && !(mag[0][0]|mag[0][1])) {
        qsb_glv14_infinity(X,Y,ZZ,ZZZ);return;
    }
    const unsigned first=q_nonzero?0u:7u;
    uint64_t x0[4],y0[4],x1[4],y1[4];
    uint32_t code=qsb_glv14_code(mag,sign,first);
    gt_load_signed_flat(table,0u,code&0x1fffffu,code>>31,x0,y0);
    code=qsb_glv14_code(mag,sign,first+1u);
    gt_load_signed_flat(table,0u,code&0x1fffffu,code>>31,x1,y1);
    _PointAddXYZZ_mm_def(X,Y,ZZ,ZZZ,x0,y0,x1,y1);
    #pragma unroll 1
    for(unsigned term=first+2u;term<13u;term++) {
        if(term==7u)qsb_glv14_beta(X);
        code=qsb_glv14_code(mag,sign,term);
        gt_load_signed_flat(table,0u,code&0x1fffffu,code>>31,x1,y1);
        _PointAddXYZZ_def<true>(X,Y,ZZ,ZZZ,x1,y1,y0);
        Load256(y0,y1);
    }
    code=qsb_glv14_code(mag,sign,13u);
    gt_load_signed_flat(table,0u,code&0x1fffffu,code>>31,x1,y1);
    qsb_complete_last_add(X,Y,ZZ,ZZZ,x1,y1,y0);
}
