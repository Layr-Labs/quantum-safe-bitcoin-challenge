/* SPDX-License-Identifier: GPL-3.0-only */
/* Four independent warp trees with register-resident upper nodes and full carries. */
#pragma once
#include "WarpInverse.cuh"
#include "CyclicField.cuh"
#include "PrefixCyclicField.cuh"
static_assert(QSB_RF_LANES==128 && QSB_SUBPIPE==131072,
              "register roots require promoted 128-lane / 1024-root shape");
__device__ __forceinline__ bool qbw_root_load(
    uint64_t x[5],const uint64_t *roots,unsigned i,unsigned count) {
    x[0]=1;x[1]=x[2]=x[3]=x[4]=0;
    if(i>=count)return false;
    #pragma unroll
    for(int k=0;k<4;++k)x[k]=roots[(size_t)i*4u+k];
    qsb_field_normalize(x);
    const bool nz=(x[0]|x[1]|x[2]|x[3])!=0;
    if(!nz)x[0]=1;
    return nz;
}
__device__ __forceinline__ void qbw_root_store(
    uint64_t *roots,unsigned count,unsigned i,uint64_t x[5],bool nonzero) {
    if(i>=count)return;
    qsb_field_normalize(x);
    if(!nonzero)x[0]=x[1]=x[2]=x[3]=0;
    #pragma unroll
    for(int k=0;k<4;++k)roots[(size_t)i*4u+k]=x[k];
    uint64_t b[5]={
#if QSB_ISO_XR
        pin_iso_u2ry_words[0],pin_iso_u2ry_words[1],
        pin_iso_u2ry_words[2],pin_iso_u2ry_words[3],0
#else
        pin_u2ry_words[0],pin_u2ry_words[1],
        pin_u2ry_words[2],pin_u2ry_words[3],0
#endif
    };
    uint64_t weighted[5];qsb_field_mul(weighted,x,b);
    #pragma unroll
    for(int k=0;k<4;++k)roots[((size_t)count+i)*4u+k]=weighted[k];
}


__device__ __forceinline__ void qbw_scratch_put(
    uint64_t *roots,unsigned count,unsigned row,const uint64_t v[5]) {
    volatile uint64_t *p=roots+(count+row)*4u;
    #pragma unroll
    for(unsigned k=0;k<4;++k)p[k]=v[k];
}
__device__ __forceinline__ void qbw_scratch_get(
    uint64_t v[5],const uint64_t *roots,unsigned count,unsigned row) {
    const volatile uint64_t *p=roots+(count+row)*4u;
    #pragma unroll
    for(unsigned k=0;k<4;++k)v[k]=p[k];
    v[4]=0;
}

/* QSB_RROOT_WIDE (kill switch, default 0: 4090 screen 04f6673f9a7e read +0.33% ±0.72 vs crown
 * on a +0.50% base, not a measured gain): qsb_root_register runs as one CTA of 256 lanes
 * (eight independent warp trees) instead of 128. Lane t owns roots t, t+256, t+512, t+768
 * (one quartet) instead of two strided quartets, so its serial field-product chain around the
 * warp inverses drops from 7 up + 14 down to 3 up + 6 down, and each SM scheduler holds two
 * warps of the latency-bound tree instead of one. Every root still gets its own inverse
 * (a field inverse is unique) and qbw_root_store normalises it before the weighted product,
 * so every stored word is the one the 128-lane shape stores. 0 restores the 128-lane shape. */
#ifndef QSB_RROOT_WIDE
#define QSB_RROOT_WIDE 0
#endif
#define QSB_RROOT_LANES (QSB_RROOT_WIDE ? 256 : 128)
template<int N>
__device__ __forceinline__ void qsb_block_inverse_register_n(uint64_t *value){
    static_assert(N==128 || N==256,"four- or eight-warp shape");
    /* 56 product / 28 inverse rows per warp. Plane padding rotates limb banks. */
    __shared__ uint64_t products[4][(N/32)*56+4];
    __shared__ uint64_t inverses[4][(N/32)*28+4];
    const unsigned tid=threadIdx.x,lane=tid&31u,warp=tid>>5;
    const unsigned pb=warp*56u,ib=warp*28u,d=lane&7u,group=lane>>3;
    #pragma unroll
    for(int k=0;k<4;++k)products[k][pb+lane]=value[k];
    __syncwarp(0xffffffffu);
    unsigned offset=0;
    #pragma unroll 1
    for(unsigned count=32;count>8;count>>=1){
        unsigned half=count>>1;
        if(lane<half){
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;++k){a[k]=products[k][pb+offset+lane];b[k]=products[k][pb+offset+half+lane];}
            a[4]=b[4]=0;QSB_RF_MUL(out,a,b);
            #pragma unroll
            for(int k=0;k<4;++k)products[k][pb+offset+count+lane]=out[k];
        }
        offset+=count;__syncwarp(0xffffffffu);
    }
    /* Eight remaining nodes at [48,56). Four eight-lane fields live in registers. */
    const uint32_t a=(uint32_t)(products[d>>1][pb+48+group]>>(32*(d&1u)));
    const uint32_t b=(uint32_t)(products[d>>1][pb+52+group]>>(32*(d&1u)));
    const uint32_t u4=qsb_prefix_cyclic_research::multiply8(a,b,lane);
    const uint32_t u2=qsb_prefix_cyclic_research::multiply8(
        __shfl_sync(0xffffffffu,u4,8*(group&1u)+d),
        __shfl_sync(0xffffffffu,u4,8*((group&1u)+2u)+d),lane);
    const uint32_t u1=qsb_prefix_cyclic_research::multiply8(
        __shfl_sync(0xffffffffu,u2,d),
        __shfl_sync(0xffffffffu,u2,8+d),lane);
    uint64_t root[5];
    #pragma unroll
    for(unsigned k=0;k<4;++k){
        const uint32_t lo=__shfl_sync(0xffffffffu,u1,2*k);
        const uint32_t hi=__shfl_sync(0xffffffffu,u1,2*k+1);
        root[k]=(uint64_t)lo|((uint64_t)hi<<32);
    }
    root[4]=0;qsb_field_normalize(root);
    qsb_warp_research::qwr_inverse_scaled(root,lane);
    uint32_t inverse_word=0;
    #pragma unroll
    for(unsigned k=0;k<4;++k)
        if((d>>1)==k)inverse_word=(uint32_t)(root[k]>>(32*(d&1u)));
    const uint32_t v2=qsb_prefix_cyclic_research::multiply8(inverse_word,
        __shfl_sync(0xffffffffu,u2,8*((group&1u)^1u)+d),lane);
    const uint32_t v4=qsb_prefix_cyclic_research::multiply8(
        __shfl_sync(0xffffffffu,v2,8*(group&1u)+d),
        __shfl_sync(0xffffffffu,u4,8*(group^2u)+d),lane);
    const uint32_t next=__shfl_down_sync(0xffffffffu,v4,1,8);
    if(!(d&1u))inverses[d>>1][ib+24+group]=(uint64_t)v4|((uint64_t)next<<32);
    __syncwarp(0xffffffffu);
    offset=48;
    #pragma unroll 1
    for(unsigned count=8;count<32;count<<=1){
        unsigned half=count>>1;
        if(lane<count){
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;++k){a[k]=inverses[k][ib+offset+count-32+(lane&(half-1))];b[k]=products[k][pb+offset+(lane^half)];}
            a[4]=b[4]=0;QSB_RF_MUL(out,a,b);
            #pragma unroll
            for(int k=0;k<4;++k)inverses[k][ib+offset-32+lane]=out[k];
        }
        offset-=count<<1;__syncwarp(0xffffffffu);
    }
    uint64_t aa[5],bb[5];
    #pragma unroll
    for(int k=0;k<4;++k){aa[k]=inverses[k][ib+(lane&15u)];bb[k]=products[k][pb+(lane^16u)];}
    aa[4]=bb[4]=0;QSB_RF_MUL(value,aa,bb);qsb_field_normalize(value);
}
// Launch exactly <<<1,QSB_RROOT_LANES>>> with 1<=count<=1024.
/* Physical capacity is 2048 four-word rows even for a partial final tile. */
#if QSB_RROOT_WIDE == 2
/* QSB_RROOT_WIDE 2: the 256-lane shape with its four normalised roots, their nonzero flags and
 * the pair products p01, p23 held in registers across the block inverse instead of a volatile
 * scratch round trip and a second load + normalise of each root. The block inverse only reads
 * and writes shared memory and `total`, so the held values are the ones the scratch rows and
 * the reloads would return (qbw_root_load is a pure function of the unchanged input rows):
 * every stored word is bit-identical to QSB_RROOT_WIDE 1. No scratch row is written.
 * 4090 screen 05248a8b46c0 read -0.89% vs crown on the +0.50% base: not a gain, off by default. */
__global__ void __launch_bounds__(256,1) qsb_root_register(uint64_t *roots,int count) {
    if (count<=0 || count>1024) return; // uniform, before any block barrier
    const unsigned n=(unsigned)count;
    const unsigned lane=threadIdx.x;
    uint64_t r0[5],r1[5],r2[5],r3[5],p01[5],p23[5],total[5];
    const bool n0=qbw_root_load(r0,roots,lane,n);
    const bool n1=qbw_root_load(r1,roots,lane+256u,n);
    const bool n2=qbw_root_load(r2,roots,lane+512u,n);
    const bool n3=qbw_root_load(r3,roots,lane+768u,n);
    qsb_field_mul(p01,r0,r1);p01[4]=0;
    qsb_field_mul(p23,r2,r3);p23[4]=0;
    qsb_field_mul(total,p01,p23);total[4]=0;
    qsb_block_inverse_register_n<256>(total);
    uint64_t ip01[5],ip23[5],ia[5],ib[5];
    qsb_field_mul(ip01,total,p23);ip01[4]=0;
    qsb_field_mul(ip23,total,p01);ip23[4]=0;
    qsb_field_mul(ia,ip01,r1);ia[4]=0;
    qsb_field_mul(ib,ip01,r0);ib[4]=0;
    qbw_root_store(roots,n,lane,ia,n0);
    qbw_root_store(roots,n,lane+256u,ib,n1);
    qsb_field_mul(ia,ip23,r3);ia[4]=0;
    qsb_field_mul(ib,ip23,r2);ib[4]=0;
    qbw_root_store(roots,n,lane+512u,ia,n2);
    qbw_root_store(roots,n,lane+768u,ib,n3);
}
#elif QSB_RROOT_WIDE
/* Lane t: p01 = r_t*r_{t+256}, p23 = r_{t+512}*r_{t+768} in scratch rows t and t+256, the
 * warp tree inverts p01*p23, and the down sweep returns 1/r for its four roots. Scratch row
 * count+r and weighted-output row count+i are only touched by lane (r mod 256) = (i mod 256),
 * and each lane reads both scratch rows before its first store, so no row is read after
 * another lane or a later statement overwrote it. */
__global__ void __launch_bounds__(256,1) qsb_root_register(uint64_t *roots,int count) {
    if (count<=0 || count>1024) return; // uniform, before any block barrier
    const unsigned n=(unsigned)count;
    const unsigned lane=threadIdx.x;
    uint64_t total[5];
    {
        uint64_t p01[5],p23[5],a[5],b[5];
        qbw_root_load(a,roots,lane,n);
        qbw_root_load(b,roots,lane+256u,n);
        qsb_field_mul(p01,a,b);p01[4]=0;
        qbw_root_load(a,roots,lane+512u,n);
        qbw_root_load(b,roots,lane+768u,n);
        qsb_field_mul(p23,a,b);p23[4]=0;
        qbw_scratch_put(roots,n,lane,p01);
        qbw_scratch_put(roots,n,lane+256u,p23);
        qsb_field_mul(total,p01,p23);total[4]=0;
    }
    qsb_block_inverse_register_n<256>(total);
    uint64_t p01[5],p23[5],ip01[5],ip23[5];
    qbw_scratch_get(p01,roots,n,lane);
    qbw_scratch_get(p23,roots,n,lane+256u);
    qsb_field_mul(ip01,total,p23);ip01[4]=0;
    qsb_field_mul(ip23,total,p01);ip23[4]=0;
    #pragma unroll
    for(unsigned pair=0;pair<2;++pair) {
        unsigned j=lane+pair*512u;
        uint64_t a[5],b[5],ia[5],ib[5];
        bool na=qbw_root_load(a,roots,j,n);
        bool nb=qbw_root_load(b,roots,j+256u,n);
        uint64_t *pinv=pair?ip23:ip01;
        qsb_field_mul(ia,pinv,b);ia[4]=0;
        qsb_field_mul(ib,pinv,a);ib[4]=0;
        qbw_root_store(roots,n,j,ia,na);
        qbw_root_store(roots,n,j+256u,ib,nb);
    }
}
#else
__global__ void __launch_bounds__(128,1) qsb_root_register(uint64_t *roots,int count) {
    if (count<=0 || count>1024) return; // uniform, before any block barrier
    const unsigned n=(unsigned)count;
    const unsigned lane=threadIdx.x;
    uint64_t total[5];
    {
        uint64_t q[2][5];
        #pragma unroll 1
        for(unsigned quartet=0;quartet<2;++quartet) {
            uint64_t p01[5],p23[5],a[5],b[5];
            unsigned i=lane+quartet*512u;
            qbw_root_load(a,roots,i,n);
            qbw_root_load(b,roots,i+128u,n);
            qsb_field_mul(p01,a,b);p01[4]=0;
            qbw_root_load(a,roots,i+256u,n);
            qbw_root_load(b,roots,i+384u,n);
            qsb_field_mul(p23,a,b);p23[4]=0;
            /* Slots 2,3 hold first quartet pairs; 4,5 hold second pairs. */
            qbw_scratch_put(roots,n,lane+(2u+2u*quartet)*128u,p01);
            qbw_scratch_put(roots,n,lane+(3u+2u*quartet)*128u,p23);
            qsb_field_mul(q[quartet],p01,p23);q[quartet][4]=0;
            qbw_scratch_put(roots,n,lane+quartet*128u,q[quartet]);
        }
        qsb_field_mul(total,q[0],q[1]);total[4]=0;
    }
    /* No local subtree field is needed across this collective. Volatile */
    /* scratch accesses require reloading instead of forwarding old values. */
    qsb_block_inverse_register_n<128>(total);
    uint64_t iq0[5],iq1[5];
    {
        uint64_t q0[5],q1[5];
        qbw_scratch_get(q0,roots,n,lane);
        qbw_scratch_get(q1,roots,n,lane+128u);
        qsb_field_mul(iq0,total,q1);iq0[4]=0;
        qsb_field_mul(iq1,total,q0);iq1[4]=0;
    }
    /* Fetch BOTH pair products before writing a quartet's weighted outputs. */
    /* Both quartet orders are safe with this rule. This descending order */
    /* consumes 4,5 before writing them, then consumes 2,3 before writing 0..3. */
    #pragma unroll 1
    for(int quartet=1;quartet>=0;--quartet) {
        uint64_t p01[5],p23[5],ip01[5],ip23[5];
        qbw_scratch_get(p01,roots,n,lane+(2u+2u*quartet)*128u);
        qbw_scratch_get(p23,roots,n,lane+(3u+2u*quartet)*128u);
        uint64_t *parent=quartet?iq1:iq0;
        qsb_field_mul(ip01,parent,p23);ip01[4]=0;
        qsb_field_mul(ip23,parent,p01);ip23[4]=0;
        #pragma unroll
        for(unsigned pair=0;pair<2;++pair) {
            unsigned j=lane+(unsigned)quartet*512u+pair*256u;
            uint64_t a[5],b[5],ia[5],ib[5];
            bool na=qbw_root_load(a,roots,j,n);
            bool nb=qbw_root_load(b,roots,j+128u,n);
            uint64_t *pinv=pair?ip23:ip01;
            qsb_field_mul(ia,pinv,b);ia[4]=0;
            qsb_field_mul(ib,pinv,a);ib[4]=0;
            qbw_root_store(roots,n,j,ia,na);
            qbw_root_store(roots,n,j+128u,ib,nb);
        }
    }
}
#endif
