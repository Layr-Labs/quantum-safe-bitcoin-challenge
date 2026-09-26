// SPDX-License-Identifier: GPL-3.0-only
// New ordinary-module root kernel for promoted e892e6e Pinning green pipeline.
// Balanced 8-root local tree, dead weighted-output scratch, promoted Subset
// cooperative inverse. Preserves full-carry roots, ISO scale, weighted ABI.
// Include after qsb_root_fused, before host launch helpers.
#pragma once
#include "WarpInverse.cuh"
static_assert(QSB_RF_LANES==128 && QSB_SUBPIPE==131072,
              "balanced root path requires promoted 128-lane / 1024-root shape");
template<int N>
__device__ __forceinline__ void qsb_block_inverse_warp_n(uint64_t *value) {
    __shared__ uint64_t products[4][2*N];
    __shared__ uint64_t inverses[4][N];
    int tid=threadIdx.x;
    #pragma unroll
    for(int k=0;k<4;k++)products[k][tid]=value[k];
    __syncthreads();
    int offset=0;
    #pragma unroll 1
    for(int count=N;count>1;count>>=1){
        int half=count>>1;
        if(tid<half){
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++){ a[k]=products[k][offset+tid]; b[k]=products[k][offset+half+tid]; }
            a[4]=b[4]=0;
            QSB_RF_MUL(out,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)products[k][offset+count+tid]=out[k];
        }
        offset+=count;
        __syncthreads();
    }
    if(__all_sync(0xffffffffu,tid<32)){
        uint64_t root[5];
        #pragma unroll
        for(int k=0;k<4;k++)root[k]=products[k][2*N-2];
        root[4]=0;
        qsb_field_normalize(root);
        qsb_warp_research::qwr_inverse_scaled(root,tid);
        if(tid==0){
        #pragma unroll
        for(int k=0;k<4;k++)inverses[k][N-2]=root[k];
        }
    }
    __syncthreads();
    offset=2*N-4;
    #pragma unroll 1
    for(int count=2;count<N;count<<=1){
        int half=count>>1;
        if(tid<count){
            int local_parent=tid&(half-1);
            uint64_t parent_inv[5],sibling[5],child_inv[5];
            #pragma unroll
            for(int k=0;k<4;k++){
                parent_inv[k]=inverses[k][offset+count-N+local_parent];
                sibling[k]=products[k][offset+(tid^half)];
            }
            parent_inv[4]=sibling[4]=0;
            QSB_RF_MUL(child_inv,parent_inv,sibling);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-N+tid]=child_inv[k];
        }
        offset-=count<<1;
        __syncthreads();
    }
    uint64_t parent_inv[5],sibling[5];
    #pragma unroll
    for(int k=0;k<4;k++){
        parent_inv[k]=inverses[k][tid&(N/2-1)];
        sibling[k]=products[k][tid^(N/2)];
    }
    parent_inv[4]=sibling[4]=0;
    QSB_RF_MUL(value,parent_inv,sibling);
    qsb_field_normalize(value);
}

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

// Launch exactly <<<1,128>>> with 1<=count<=1024.
// Physical capacity is 2048 four-word rows even for a partial final tile.
__global__ void __launch_bounds__(128,1) qsb_root_balanced_warp(uint64_t *roots,int count) {
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
    qsb_block_inverse_warp_n<128>(total);
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
