// A[8] is replicated; B/out hold H=8/G consecutive words owned by this lane.
#pragma once
__device__ __forceinline__ unsigned qsb_cc_comp(unsigned f,unsigned g) {
    return ((f&1u)|(((f>>1)&1u)&(g&1u)))|(f&g&2u);
}
__device__ __forceinline__ unsigned qsb_cc_apply(unsigned f,unsigned c) {
    return (f&1u)|(((f>>1)&1u)&c);
}
template<int G,int D>
__device__ __forceinline__ void qsb_cc_prev_pair(
    const unsigned *lo,const unsigned *hi,unsigned *pl,unsigned *ph,unsigned mask) {
    constexpr int H=8/G;const unsigned lane=threadIdx.x&(G-1);
    unsigned bl[D],bh[D];
    #pragma unroll
    for(int p=0;p<D;p++) {
        bl[p]=__shfl_sync(mask,lo[H-D+p],(lane+G-1)&(G-1),G);
        unsigned v=lane==G-1?lo[H-D+p]:hi[H-D+p];
        bh[p]=__shfl_sync(mask,v,(lane+G-1)&(G-1),G);
    }
    #pragma unroll
    for(int p=0;p<H;p++) {
        pl[p]=p>=D?lo[p-D]:(lane?bl[p]:0);
        ph[p]=p>=D?hi[p-D]:bh[p];
    }
}
template<int G>
__device__ __forceinline__ void qsb_cc_prev(const unsigned *v,unsigned *out,unsigned mask) {
    constexpr int H=8/G;const unsigned lane=threadIdx.x&(G-1);
    unsigned edge=__shfl_up_sync(mask,v[H-1],1,G);
    #pragma unroll
    for(int p=0;p<H;p++) out[p]=p?v[p-1]:(lane?edge:0);
}
template<int G>
__device__ __forceinline__ unsigned qsb_cc_normalize(
    const unsigned long long *z,unsigned *out,unsigned seed,unsigned mask) {
    constexpr int H=8/G;const unsigned lane=threadIdx.x&(G-1);
    unsigned f=2; // identity carry map
    #pragma unroll
    for(int p=0;p<H;p++) {
        unsigned m=(unsigned)(z[p]>>32)|(((unsigned)z[p]==0xffffffffu)?2u:0u);
        f=qsb_cc_comp(m,f);
    }
    #pragma unroll
    for(int d=1;d<G;d<<=1) {
        unsigned prev=__shfl_up_sync(mask,f,d,G);
        if(lane>=(unsigned)d) f=qsb_cc_comp(f,prev);
    }
    unsigned prev=__shfl_up_sync(mask,f,1,G);
    unsigned c=lane?qsb_cc_apply(prev,seed):seed;
    #pragma unroll
    for(int p=0;p<H;p++) {unsigned long long t=z[p]+c;out[p]=(unsigned)t;c=(unsigned)(t>>32);}
    return qsb_cc_apply(f,seed);
}
template<int G>
__device__ __forceinline__ void qsb_coop_raw_contiguous(
    const unsigned *a,const unsigned *b,unsigned *out,unsigned mask) {
    static_assert(G==2||G==4,"two or four contiguous lanes");
    constexpr int H=8/G;const unsigned lane=threadIdx.x&(G-1);
    unsigned long long lo[H]={},hi[H]={};unsigned lx[H]={},hx[H]={};
    #pragma unroll
    for(unsigned j=0;j<8;j++) {
        #pragma unroll
        for(int p=0;p<H;p++) {
            unsigned k=lane*H+p,bi=(k-j)&7u;
            unsigned bj=__shfl_sync(mask,b[(p-j)&(H-1)],bi/H,G);
            unsigned long long t=(unsigned long long)a[j]*bj;
            unsigned long long tl=j<=k?t:0,th=j>k?t:0;
            unsigned long long nl=lo[p]+tl,nh=hi[p]+th;
            lx[p]+=(nl<lo[p]);hx[p]+=(nh<hi[p]);lo[p]=nl;hi[p]=nh;
        }
    }
    unsigned ml[H],mh[H],pl[H],ph[H],p2l[H],p2h[H];
    #pragma unroll
    for(int p=0;p<H;p++) {ml[p]=(unsigned)(lo[p]>>32);mh[p]=(unsigned)(hi[p]>>32);}
    qsb_cc_prev_pair<G,1>(ml,mh,pl,ph,mask);
    qsb_cc_prev_pair<G,2>(lx,hx,p2l,p2h,mask);
    unsigned long long zl[H],zh[H];unsigned il[H],ih[H];
    #pragma unroll
    for(int p=0;p<H;p++) {
        zl[p]=(unsigned long long)(unsigned)lo[p]+pl[p]+p2l[p];
        zh[p]=(unsigned long long)(unsigned)hi[p]+ph[p]+p2h[p];
        il[p]=(unsigned)(zl[p]>>32);ih[p]=(unsigned)(zh[p]>>32);
    }
    qsb_cc_prev_pair<G,1>(il,ih,pl,ph,mask);
    #pragma unroll
    for(int p=0;p<H;p++) {zl[p]=(unsigned long long)(unsigned)zl[p]+pl[p];zh[p]=(unsigned long long)(unsigned)zh[p]+ph[p];}
    unsigned dl[H],dh[H];
    unsigned seed=qsb_cc_normalize<G>(zl,dl,0,mask);
    seed=__shfl_sync(mask,seed,G-1,G);
    qsb_cc_normalize<G>(zh,dh,seed,mask);
    unsigned prev[H];qsb_cc_prev<G>(dh,prev,mask);
    unsigned long long fold[H];unsigned fh[H],pfh[H];
    #pragma unroll
    for(int p=0;p<H;p++) {fold[p]=(unsigned long long)dl[p]+977ULL*dh[p]+prev[p];fh[p]=(unsigned)(fold[p]>>32);}
    qsb_cc_prev<G>(fh,pfh,mask);
    #pragma unroll
    for(int p=0;p<H;p++) fold[p]=(unsigned long long)(unsigned)fold[p]+pfh[p];
    unsigned carry=qsb_cc_normalize<G>(fold,out,0,mask);
    unsigned h=dh[H-1]+fh[H-1]+carry;
    h=__shfl_sync(mask,h,G-1,G);
    unsigned long long t0=(unsigned long long)out[0]+977ULL*h;
    unsigned long long t1=(unsigned long long)out[1]+h+(t0>>32);
    if(lane==0) {out[0]=(unsigned)t0;out[1]=(unsigned)t1;}
    if(G==2) {
        if(lane==0) out[2]+=(unsigned)(t1>>32);
    } else {
        unsigned c1=__shfl_sync(mask,(unsigned)(t1>>32),0,G);
        if(lane==1) out[0]+=c1;
    }
}
