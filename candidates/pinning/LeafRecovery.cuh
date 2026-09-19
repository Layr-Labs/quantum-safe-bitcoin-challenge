// SPDX-License-Identifier: GPL-3.0-only
// Uses the VanitySearch-derived field primitives in GPUMath.h.
// Squaring-free recovery identity: odinfree, public submission e00f5566.
// 128-leaf pipeline geometry follows 0xCramJam, public submission 1a228081.
// The scaled sibling checkpoint and truncated inverse expansion are new here.
#pragma once

#ifndef QSB_RECOVERY_N
#define QSB_RECOVERY_N 128
#endif
static_assert(QSB_RECOVERY_N==128 || QSB_RECOVERY_N==256,"supported tree width");

// The tree multiplier retains the final reduction carry. Normalize at these
// recovery boundaries, where additions, zero tests and parity require [0,p).
__device__ __forceinline__ void qsb_recovery_mul(
    uint64_t *out, const uint64_t *a, const uint64_t *b) {
    uint64_t tmp[5];
    qsb_field_mul(tmp, const_cast<uint64_t *>(a), const_cast<uint64_t *>(b));
    qsb_field_normalize(tmp);
    Load256(out,tmp);
}

__device__ __forceinline__ void qsb_recovery_denominator(
    uint64_t *X, uint64_t *U, uint64_t *Y, uint64_t *V,
    const uint64_t *a, uint64_t *W) {
    // U, Y and V are consumed only by carry-complete full-width multiplies
    // in this cofactor path. Keep X canonical for the subtraction; a*U may be raw.
    qsb_field_normalize(X);
    uint64_t d[4];
    // X<p, while the exact product d may use any256-bit representative.
    // Thus d-X>-p. A borrow-corrected subtraction stays congruent and fits.
    // The next exact full-width multiply accepts d without normalization.
    uint64_t raw[5];qsb_field_mul(raw,const_cast<uint64_t*>(a),U);
    Load256(d,raw);
    _ModSub256(d,X);
    qsb_recovery_mul(W,V,d);       // W = V*(a*U-X), with U=ZZ and V=ZZZ.
    W[4]=0;
}

// Preserve Y,V before this call. Instead of saving W, save H_i=U_i*W_sibling.
// The finish stops at the pair inverse: H_i/(W_i*W_sibling)=U_i/W_i.
// Inactive/singular lanes enter with W=1 and U=0; H=0 retains their mask.
__device__ __forceinline__ void qsb_recovery_product_checkpoint(
    const uint64_t *value, const uint64_t *U, ulonglong2 *saved,
    int batch_size, bool active, uint64_t *roots, uint64_t *checkpoint) {
    __shared__ uint64_t products[4][2*QSB_RECOVERY_N];
    const int tid=threadIdx.x;
    const size_t block_base=(size_t)blockIdx.x*4u*QSB_RECOVERY_N;
    #pragma unroll
    for(int k=0;k<4;k++)products[k][tid]=value[k];
    __syncthreads();
    uint64_t sibling[4],H[4];
    #pragma unroll
    for(int k=0;k<4;k++)sibling[k]=products[k][tid^(QSB_RECOVERY_N/2)];
    qsb_recovery_mul(H,U,sibling);
    if(active){
        const size_t idx=(size_t)blockIdx.x*QSB_RECOVERY_N+tid;
        const size_t stride=(size_t)batch_size;
        saved[4u*stride+idx]=make_ulonglong2(H[0],H[1]);
        saved[5u*stride+idx]=make_ulonglong2(H[2],H[3]);
    }

    int offset=0;
    #pragma unroll 1
    for(int count=QSB_RECOVERY_N;count>1;count>>=1){
        const int half=count>>1;
        if(tid<half){
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++){
                a[k]=products[k][offset+tid];
                b[k]=products[k][offset+half+tid];
            }
            a[4]=b[4]=0;
            qsb_field_mul(out,a,b);
            const int node=offset+count+tid;
            #pragma unroll
            for(int k=0;k<4;k++){
                products[k][node]=out[k];
                if(node<2*QSB_RECOVERY_N-2)
                    checkpoint[block_base+(size_t)k*QSB_RECOVERY_N+node-QSB_RECOVERY_N]=out[k];
            }
        }
        offset+=count;
        if(count>2)__syncthreads();
    }
    if(tid==0){
        #pragma unroll
        for(int k=0;k<4;k++)roots[(size_t)blockIdx.x*4u+k]=products[k][2*QSB_RECOVERY_N-2];
    }
}

// Only immutable INTERNAL products are required. The final leaf expansion is
// folded into the saved H, saving one N-element shared product plane per limb.
__device__ __forceinline__ void qsb_recovery_pair_inverse(
    uint64_t *value, const uint64_t *roots, const uint64_t *checkpoint) {
    __shared__ uint64_t products[4][QSB_RECOVERY_N];
    __shared__ uint64_t inverses[4][QSB_RECOVERY_N];
    const int tid=threadIdx.x;
    const size_t block_base=(size_t)blockIdx.x*4u*QSB_RECOVERY_N;
    #pragma unroll
    for(int k=0;k<4;k++){
        if(tid<QSB_RECOVERY_N-2)
            products[k][tid]=checkpoint[block_base+(size_t)k*QSB_RECOVERY_N+tid];
        if(tid==0)inverses[k][QSB_RECOVERY_N-2]=roots[(size_t)blockIdx.x*4u+k];
    }
    __syncthreads();
    int offset=2*QSB_RECOVERY_N-4;
    #pragma unroll 1
    for(int count=2;count<QSB_RECOVERY_N;count<<=1){
        const int half=count>>1;
        if(tid<count){
            uint64_t parent[5],sibling[5],child[5];
            #pragma unroll
            for(int k=0;k<4;k++){
                parent[k]=inverses[k][offset+count-QSB_RECOVERY_N+(tid&(half-1))];
                sibling[k]=products[k][offset+(tid^half)-QSB_RECOVERY_N];
            }
            parent[4]=sibling[4]=0;
            qsb_field_mul(child,parent,sibling);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-QSB_RECOVERY_N+tid]=child[k];
        }
        offset-=count<<1;
        __syncthreads();
    }
    #pragma unroll
    for(int k=0;k<4;k++)value[k]=inverses[k][tid&(QSB_RECOVERY_N/2-1)];
    value[4]=0;
}

// P=(X/U,Y/V), V^2=U^3, R=(a,b), c=3*a^2/(2*b).
// h=H/pair_product=U/[V*(a*U-X)] gives slopes l=(b*V-Y)*h,
// m=(b*V+Y)*h. On the curve, x(P+R)=(l+m)*(l-c)+a,
// x(P-R)=(l+m)*(m-c)+a. No field square is needed.
__device__ __forceinline__ uint32_t qsb_recovery_finish(
    const uint64_t *Y, const uint64_t *V, const uint64_t *H,
    const uint64_t *pair_inv, uint64_t *a, uint64_t *b, uint64_t *c,
    uint64_t *x1, uint64_t *x2) {
    uint64_t h[4],yb[4],l[4],m[4],sum[4],t[4],s[4];
    qsb_recovery_mul(h,H,pair_inv);
    qsb_recovery_mul(yb,V,b);
    _ModSub256(l,yb,const_cast<uint64_t *>(Y));
    qsb_recovery_mul(l,l,h);
    _ModAdd256(m,yb,const_cast<uint64_t *>(Y));
    qsb_recovery_mul(m,m,h);
    _ModAdd256(sum,l,m);
    _ModSub256(t,l,c);
    qsb_recovery_mul(x1,sum,t);
    _ModAdd256(x1,x1,a);
    _ModSub256(t,m,c);
    qsb_recovery_mul(x2,sum,t);
    _ModAdd256(x2,x2,a);
    _ModSub256(t,a,x1);
    qsb_recovery_mul(s,l,t);
    _ModSub256(s,b);
    uint32_t parity=(uint32_t)(s[0]&1u);
    _ModSub256(t,a,x2);
    qsb_recovery_mul(s,m,t);
    _ModSub256(s,b,s);
    return parity|((uint32_t)(s[0]&1u)<<1);
}
