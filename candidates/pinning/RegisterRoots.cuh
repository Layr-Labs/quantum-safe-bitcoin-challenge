// SPDX-License-Identifier: GPL-3.0-only
// Four independent warp trees with register-resident upper nodes and full carries.
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

template<int N>
__device__ __forceinline__ void qsb_block_inverse_register_n(uint64_t *value){
    static_assert(N==128,"research fixed four-warp shape");
    // 56 product / 28 inverse rows per warp. Plane padding rotates limb banks.
    __shared__ uint64_t products[4][4*56+4];
    __shared__ uint64_t inverses[4][4*28+4];
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
    // Eight remaining nodes at [48,56). Four eight-lane fields live in registers.
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
// Launch exactly <<<1,128>>> with 1<=count<=1024.
// Physical capacity is 2048 four-word rows even for a partial final tile.
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
            // Slots 2,3 hold first quartet pairs; 4,5 hold second pairs.
            qbw_scratch_put(roots,n,lane+(2u+2u*quartet)*128u,p01);
            qbw_scratch_put(roots,n,lane+(3u+2u*quartet)*128u,p23);
            qsb_field_mul(q[quartet],p01,p23);q[quartet][4]=0;
            qbw_scratch_put(roots,n,lane+quartet*128u,q[quartet]);
        }
        qsb_field_mul(total,q[0],q[1]);total[4]=0;
    }
    // No local subtree field is needed across this collective. Volatile
    // scratch accesses require reloading instead of forwarding old values.
    qsb_block_inverse_register_n<128>(total);
    uint64_t iq0[5],iq1[5];
    {
        uint64_t q0[5],q1[5];
        qbw_scratch_get(q0,roots,n,lane);
        qbw_scratch_get(q1,roots,n,lane+128u);
        qsb_field_mul(iq0,total,q1);iq0[4]=0;
        qsb_field_mul(iq1,total,q0);iq1[4]=0;
    }
    // Fetch BOTH pair products before writing a quartet's weighted outputs.
    // Both quartet orders are safe with this rule. This descending order
    // consumes 4,5 before writing them, then consumes 2,3 before writing 0..3.
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
