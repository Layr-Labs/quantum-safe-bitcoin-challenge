// SPDX-License-Identifier: GPL-3.0-only
// Original-order top-32 traversal. Each immutable internal node has one owner;
// unlike duplicated-node top schedules, no field product is recomputed.
#pragma once

__device__ __forceinline__ void qsb_top_gather(
    uint64_t *dst, const uint64_t *src, int owner) {
    #pragma unroll
    for(int k=0;k<4;k++)dst[k]=__shfl_sync(0xffffffffu,src[k],owner);
    dst[4]=0;
}

template<int N> __device__ __forceinline__ void qsb_resident_top32(
    uint64_t *roots,uint64_t (*products)[2*N],uint64_t (*excluded)[N]) {
    const int lane=threadIdx.x;
    uint64_t leaf[5],node[5]={0,0,0,0,0},ex[5]={0,0,0,0,0};
    uint64_t a[5],b[5];
    #pragma unroll
    for(int k=0;k<4;k++)leaf[k]=products[k][2*N-64+lane];
    leaf[4]=0;

    // Owners 0..15: pairs; 16..23: quarters; 24..27: eighths;
    // 28..29: halves. All 32 lanes execute every gather before predicates.
    #pragma unroll 1
    for(int width=16;width>=2;width>>=1) {
        const int dest=32-2*width;
        const int src=32-4*width;
        const int i=lane&(width-1);
        if(width==16) {
            Load256(a,leaf);
            a[4]=0;
            qsb_top_gather(b,leaf,lane^16);
        } else {
            qsb_top_gather(a,node,src+i);
            qsb_top_gather(b,node,src+width+i);
        }
        if(lane>=dest && lane<dest+width)qsb_field_mul_sc(node,a,b);
    }

    // Preserve the promoted TOP2 operand order, including the root in lane 4.
    // The immutable bank survives while the exclusion bank is expanded.
    #pragma unroll 1
    for(int count=4;count<=32;count<<=1) {
        if(count==4) {
            qsb_top_gather(a,node,lane<4 ? 28+((lane&1)^1) : 28);
            qsb_top_gather(b,node,lane<4 ? 24+(lane^2) : 29);
        } else {
            qsb_top_gather(a,ex,lane&((count/2)-1));
            if(count==32)qsb_top_gather(b,leaf,lane^16);
            else qsb_top_gather(b,node,32-2*count+((lane&(count-1))^(count/2)));
        }
        if(lane<count || (count==4 && lane==4))qsb_field_mul_sc(ex,a,b);
        if(count==4 && lane==4) {
            #pragma unroll
            for(int k=0;k<4;k++)roots[(size_t)blockIdx.x*4+k]=ex[k];
        }
    }
    #pragma unroll
    for(int k=0;k<4;k++)excluded[k][N-64+lane]=ex[k];
}
