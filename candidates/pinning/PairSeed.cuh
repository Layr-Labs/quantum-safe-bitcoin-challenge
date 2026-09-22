// SPDX-License-Identifier: GPL-3.0-only
// Two independent affine pair sums share one CTA-wide denominator inverse.
// Include after qsb_field_mul/qsb_field_normalize and the signed-table helpers.
#pragma once

template<int N>
__device__ __forceinline__ void qsb_pair_product_inverse(
    uint64_t *inverse, const uint64_t *d0, const uint64_t *d1,
    bool singular) {
    static_assert(N==128 || N==256,"pair-seed CTA width");
    __shared__ uint64_t heap[4][2*N];
    const unsigned tid=threadIdx.x;
    uint64_t a[5],b[5],v[5];
    Load256(a,d0);Load256(b,d1);a[4]=b[4]=0;
    qsb_field_mul(v,a,b);
    if(singular){v[0]=1;v[1]=v[2]=v[3]=0;}
    #pragma unroll
    for(int k=0;k<4;k++)heap[k][N+tid]=v[k];
    __syncthreads();
    #pragma unroll 1
    for(unsigned width=N/2;width; width>>=1) {
        if(tid<width) {
            const unsigned node=width+tid;
            #pragma unroll
            for(int k=0;k<4;k++) {
                a[k]=heap[k][2*node];b[k]=heap[k][2*node+1];
            }
            qsb_field_mul(v,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)heap[k][node]=v[k];
        }
        __syncthreads();
    }
    if(tid==0) {
        #pragma unroll
        for(int k=0;k<4;k++)v[k]=heap[k][1];
        v[4]=0;qsb_field_normalize(v);_ModInv(v);
        #pragma unroll
        for(int k=0;k<4;k++)heap[k][1]=v[k];
    }
    __syncthreads();
    // Each parent owns both children, so replacing products with inverses
    // cannot race with a different parent. The barrier separates levels.
    #pragma unroll 1
    for(unsigned width=1;width<N;width<<=1) {
        if(tid<width) {
            const unsigned node=width+tid;
            uint64_t parent[5];
            #pragma unroll
            for(int k=0;k<4;k++) {
                parent[k]=heap[k][node];
                a[k]=heap[k][2*node];b[k]=heap[k][2*node+1];
            }
            parent[4]=0;
            qsb_field_mul(v,parent,b);
            #pragma unroll
            for(int k=0;k<4;k++)heap[k][2*node]=v[k];
            qsb_field_mul(v,parent,a);
            #pragma unroll
            for(int k=0;k<4;k++)heap[k][2*node+1]=v[k];
        }
        __syncthreads();
    }
    #pragma unroll
    for(int k=0;k<4;k++)inverse[k]=heap[k][N+tid];
    // The heap is private to this helper. No later collective reuses it.
}

__device__ __forceinline__ void qsb_pair_affine_finish(
    uint64_t *x, uint64_t *y, const uint64_t *dx, const uint64_t *slope) {
    uint64_t xx[4],t[4];
    _ModSqr(xx,const_cast<uint64_t*>(slope));
    _ModSub256(xx,xx,x);_ModSub256(xx,xx,x);_ModSub256(xx,xx,dx);
    _ModSub256(t,x,xx);_ModMult(t,t,const_cast<uint64_t*>(slope));
#if QSB_YOFF
    // The input is y+c, c=(2^32+976)/2. Recover y before the affine
    // subtraction, then canonicalize the new y and add c without reduction.
    // This preserves the exact signed-table offset representation.
    qsb_yoff_to_y(y);
#endif
    _ModSub256(y,t,y);
    qsb_field_normalize(y);
#if QSB_YOFF
    asm("add.cc.u64 %0,%0,0x800001E8;\n"
        "addc.cc.u64 %1,%1,0;\naddc.cc.u64 %2,%2,0;\naddc.u64 %3,%3,0;"
        : "+l"(y[0]),"+l"(y[1]),"+l"(y[2]),"+l"(y[3]));
#endif
    Load256(x,xx);
}

template<bool RELOAD_ANCHORS=true>
__device__ __forceinline__ void qsb_pairseed_scalar(
    uint64_t *X,uint64_t *Y,uint64_t *U,uint64_t *V,
    const uint64_t k[4],const uint8_t *table) {
    qsb_decode_to_shared(k);
    uint64_t x0[4],y0[4],d0[4],s0[4];
    uint64_t x1[4],y1[4],d1[4],s1[4],h[4];
    qsb_load_decoded(table,0,gt_offset(0),x0,y0);
    qsb_load_decoded(table,1,gt_offset(1),d0,s0);
    _ModSub256(d0,d0,x0);_ModSub256(s0,s0,y0);
    qsb_load_decoded(table,2,gt_offset(2),x1,y1);
    qsb_load_decoded(table,3,gt_offset(3),d1,s1);
    _ModSub256(d1,d1,x1);_ModSub256(s1,s1,y1);
    qsb_field_normalize(d0);qsb_field_normalize(d1);
    const bool singular=(d0[0]|d0[1]|d0[2]|d0[3])==0 ||
                        (d1[0]|d1[1]|d1[2]|d1[3])==0;
    qsb_pair_product_inverse<QSB_TREE_N>(h,d0,d1,singular);
    // Reload these two already-read table records after the collective.
    // Their first values are dead after the differences above. This avoids
    // keeping 16 field limbs live through the product tree and root inverse.
    if(RELOAD_ANCHORS) {
        qsb_load_decoded(table,0,gt_offset(0),x0,y0);
        qsb_load_decoded(table,2,gt_offset(2),x1,y1);
    }
    if(singular) {
        // Preserve the inherited chain's degenerate behavior and isolate
        // the denominator from all other lanes in this collective.
        _FixedBaseSignedXYZZScalar(X,Y,U,V,k,table);
        return;
    }
    _ModMult(s0,s0,h);_ModMult(s0,s0,d1);
    _ModMult(s1,s1,h);_ModMult(s1,s1,d0);
    qsb_pair_affine_finish(x0,y0,d0,s0);
    qsb_pair_affine_finish(x1,y1,d1,s1);
    _PointAddXYZZ_mm(X,Y,U,V,x0,y0,x1,y1);
    unsigned base=gt_offset(4);
    #pragma unroll 1
    for(int c=4;c<GT_CHUNKS;c++) {
        qsb_load_decoded(table,c,base,x1,y1);
        _PointAddXYZZT<true>(X,Y,U,V,x1,y1,y0);
        Load256(y0,y1);base+=1u<<16;
    }
#if QSB_YOFF
    qsb_yoff_to_y(y0);
#endif
    _ModMult(x1,y0,V);_ModSub256(Y,Y,x1);
}
