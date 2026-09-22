#pragma once
#include "CoopContiguous.cuh"
#include "CoopEight.cuh"
template<int G> struct qsb_tree_coop_math {
    __device__ static __forceinline__ void run(const unsigned *a,const unsigned *b,unsigned *out,unsigned mask) {
        qsb_coop_raw_contiguous<G>(a,b,out,mask);
    }
};
template<> struct qsb_tree_coop_math<8> {
    __device__ static __forceinline__ void run(const unsigned *a,const unsigned *b,unsigned *out,unsigned mask) {
        out[0]=qsb_coop8_raw_preloaded(a,b[0],mask);
    }
};
template<int G> __device__ __forceinline__ void qsb_tree_coop_product(
    const uint64_t *ap,int as,const uint64_t *bp,int bs,uint64_t *op,int os,unsigned mask) {
    constexpr int H=8/G;const unsigned lane=threadIdx.x&(G-1);
    unsigned a[8],b[H],out[H];
    #pragma unroll
    for(int k=0;k<4;k++) {uint64_t v=ap[k*as];a[2*k]=(unsigned)v;a[2*k+1]=(unsigned)(v>>32);}
    #pragma unroll
    for(int p=0;p<H;p++) {unsigned word=lane*H+p;uint64_t v=bp[(word/2)*bs];b[p]=(unsigned)(v>>(32*(word&1)));}
    qsb_tree_coop_math<G>::run(a,b,out,mask);
    if(G==8) {
        unsigned other=__shfl_xor_sync(mask,out[0],1,G);
        if(!(lane&1)) op[(lane/2)*os]=(uint64_t)out[0]|((uint64_t)other<<32);
    } else {
        #pragma unroll
        for(int p=0;p<H/2;p++) op[(lane*(H/2)+p)*os]=(uint64_t)out[2*p]|((uint64_t)out[2*p+1]<<32);
    }
}
template<int N,int G> __device__ __forceinline__ void qsb_tree_coop_up(
    uint64_t (*products)[2*N],int offset,int count) {
    unsigned tid=threadIdx.x;int half=count>>1;
    if(tid<32) {
        unsigned mask=__ballot_sync(0xffffffffu,tid<(unsigned)(half*G));
        if(tid<(unsigned)(half*G)) {
            int job=tid/G;
            qsb_tree_coop_product<G>(&products[0][offset+job],2*N,
                &products[0][offset+half+job],2*N,&products[0][offset+count+job],2*N,mask);
        }
    }
}
template<int N,int G> __device__ __forceinline__ void qsb_tree_coop_down(
    uint64_t (*products)[2*N],uint64_t (*excluded)[N],int offset,int count) {
    unsigned tid=threadIdx.x;int half=count>>1;
    if(tid<32) {
        unsigned mask=__ballot_sync(0xffffffffu,tid<(unsigned)(count*G));
        if(tid<(unsigned)(count*G)) {
            int job=tid/G;
            qsb_tree_coop_product<G>(&excluded[0][offset+count-N+(job&(half-1))],N,
                &products[0][offset+(job^half)],2*N,&excluded[0][offset-N+job],N,mask);
        }
    }
}
