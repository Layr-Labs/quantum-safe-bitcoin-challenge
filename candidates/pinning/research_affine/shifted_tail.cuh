// SPDX-License-Identifier: GPL-3.0-only
// Shift the recovery pair into the final fixed-base window, in original curve
// coordinates. Research primitive: the caller owns zero-denominator fallback.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#ifdef __CUDACC__
#define QSB_TAIL_INLINE __host__ __device__ __forceinline__
#else
#define QSB_TAIL_INLINE inline
#endif

struct QsbShiftedTailPoint { uint64_t x[4], y[4]; };
struct QsbShiftedTailRecord { QsbShiftedTailPoint arm[2]; };
static_assert(sizeof(QsbShiftedTailRecord)==128,"two canonical affine points");

QSB_TAIL_INLINE bool qsb_shifted_tail_zero(const uint64_t a[4]) {
    return (a[0]|a[1]|a[2]|a[3])==0;
}
QSB_TAIL_INLINE bool qsb_shifted_tail_infinity(const QsbShiftedTailPoint& a) {
    return qsb_shifted_tail_zero(a.x)&&qsb_shifted_tail_zero(a.y);
}

// Field policy F supplies exact/canonical sub,mul,sqr,neg plus
// parity_product(a,b,beta,neg)=parity((-1)^neg*(a*b+beta) mod p).
// Inputs and outputs are little-endian 4x64-bit field arrays.
template<class F>
QSB_TAIL_INLINE void qsb_shifted_tail_load(
    QsbShiftedTailPoint out[2], const QsbShiftedTailRecord *table,
    uint32_t index, uint32_t negative) {
    const uint32_t sign=negative&1u;
#ifdef __CUDACC__
    #pragma unroll
#endif
    for(unsigned r=0;r<2;r++) {
        const QsbShiftedTailPoint &a=table[index].arm[r^sign];
#ifdef __CUDACC__
        #pragma unroll
#endif
        for(unsigned k=0;k<4;k++){out[r].x[k]=a.x[k];out[r].y[k]=a.y[k];}
        if(sign)F::neg(out[r].y,out[r].y);
    }
}

template<class F>
QSB_TAIL_INLINE bool qsb_shifted_tail_prepare(
    uint64_t denominator[4], uint64_t delta[2][4],uint64_t numerator[2][4],
    const QsbShiftedTailPoint& prefix,const QsbShiftedTailPoint tail[2]) {
    if(qsb_shifted_tail_infinity(prefix)||qsb_shifted_tail_infinity(tail[0])||
       qsb_shifted_tail_infinity(tail[1]))return false;
#ifdef __CUDACC__
    #pragma unroll
#endif
    for(unsigned r=0;r<2;r++) {
        F::sub(delta[r],tail[r].x,prefix.x);
        F::sub(numerator[r],tail[r].y,prefix.y);
        if(qsb_shifted_tail_zero(delta[r]))return false;
    }
    F::mul(denominator,delta[0],delta[1]);
    return true;
}

template<class F>
QSB_TAIL_INLINE uint32_t qsb_shifted_tail_finish(
    uint64_t out_x[2][4],const uint64_t inverse_denominator[4],
    const uint64_t delta[2][4],const uint64_t numerator[2][4],
    const QsbShiftedTailPoint& prefix,const QsbShiftedTailPoint tail[2]) {
    uint32_t parities=0;
#ifdef __CUDACC__
    #pragma unroll
#endif
    for(unsigned r=0;r<2;r++) {
        uint64_t slope[4],offset[4];
        F::mul(slope,numerator[r],delta[r^1u]);
        F::mul(slope,slope,inverse_denominator);
        F::sqr(out_x[r],slope);
        F::sub(out_x[r],out_x[r],prefix.x);
        F::sub(out_x[r],out_x[r],tail[r].x);
        F::sub(offset,out_x[r],tail[r].x);
        // y = slope*(tail.x-out_x)-tail.y; same sign for both recids.
        const uint32_t parity=F::parity_product(slope,offset,tail[r].y,1u);
        parities|=(parity&1u)<<r;
    }
    return parities;
}

#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <vector>

// Entry j represents L_j=(2j+1)*2^239*(neg_r_inv/2)*G.
// arm[0]=L_j+R; arm[1]=L_j-R. Infinity is encoded by all-zero x,y.
// Return false for allocation/EC failures. Range slices support cheap audits.
static inline bool qsb_build_shifted_tail(
    QsbShiftedTailRecord *out,const uint8_t neg_r_inv_le[32],
    const uint8_t recovery_x_le[32],const uint8_t recovery_y_le[32],
    uint32_t first_index=0,uint32_t count=65536) {
    if(first_index>65536u||count>65536u-first_index)return false;
    if(count==0)return true;
    EC_GROUP *g=EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx=BN_CTX_new();
    BIGNUM *n=BN_new(),*s=BN_new(),*k=BN_new(),*x=BN_new(),*y=BN_new();
    EC_POINT *base=g?EC_POINT_new(g):nullptr,*step=g?EC_POINT_new(g):nullptr;
    EC_POINT *cur=g?EC_POINT_new(g):nullptr,*r=g?EC_POINT_new(g):nullptr;
    EC_POINT *minus_r=g?EC_POINT_new(g):nullptr;
    bool ok=g&&ctx&&n&&s&&k&&x&&y&&base&&step&&cur&&r&&minus_r;
    if(ok)ok=EC_GROUP_get_order(g,n,ctx)==1 && BN_lebin2bn(neg_r_inv_le,32,s);
    // 2^239*(A/2) = 2^238*A in the prime-order group.
    if(ok)ok=BN_lshift(s,s,238)==1 && BN_nnmod(s,s,n,ctx)==1;
    if(ok)ok=EC_POINT_mul(g,base,s,nullptr,nullptr,ctx)==1 &&
             EC_POINT_dbl(g,step,base,ctx)==1;
    if(ok)ok=BN_set_word(k,2u*first_index+1u)==1 &&
             EC_POINT_mul(g,cur,nullptr,base,k,ctx)==1;
    if(ok)ok=BN_lebin2bn(recovery_x_le,32,x) && BN_lebin2bn(recovery_y_le,32,y) &&
             EC_POINT_set_affine_coordinates(g,r,x,y,ctx)==1 &&
             EC_POINT_copy(minus_r,r)==1 && EC_POINT_invert(g,minus_r,ctx)==1;
    const uint32_t chunk=1024;
    for(uint32_t offset=0;ok&&offset<count;offset+=chunk) {
        const uint32_t take=(count-offset<chunk)?count-offset:chunk;
        std::vector<EC_POINT*> points(2u*take,nullptr);
        for(uint32_t j=0;ok&&j<take;j++) {
            points[2*j]=EC_POINT_new(g);points[2*j+1]=EC_POINT_new(g);
            ok=points[2*j]&&points[2*j+1] &&
               EC_POINT_add(g,points[2*j],cur,r,ctx)==1 &&
               EC_POINT_add(g,points[2*j+1],cur,minus_r,ctx)==1 &&
               EC_POINT_add(g,cur,cur,step,ctx)==1;
        }
        if(ok)ok=EC_POINTs_make_affine(g,points.size(),points.data(),ctx)==1;
        for(uint32_t j=0;ok&&j<take;j++)for(unsigned arm=0;arm<2;arm++) {
            QsbShiftedTailPoint &dst=out[offset+j].arm[arm];
            EC_POINT *pt=points[2*j+arm];
            if(EC_POINT_is_at_infinity(g,pt)==1){memset(&dst,0,sizeof(dst));continue;}
            ok=EC_POINT_get_affine_coordinates(g,pt,x,y,ctx)==1 &&
               BN_bn2lebinpad(x,(uint8_t*)dst.x,32)==32 &&
               BN_bn2lebinpad(y,(uint8_t*)dst.y,32)==32;
            if(!ok)break;
        }
        for(EC_POINT *pt:points)EC_POINT_free(pt);
    }
    EC_POINT_free(base);EC_POINT_free(step);EC_POINT_free(cur);
    EC_POINT_free(r);EC_POINT_free(minus_r);EC_GROUP_free(g);BN_CTX_free(ctx);
    BN_free(n);BN_free(s);BN_free(k);BN_free(x);BN_free(y);
    return ok;
}
#undef QSB_TAIL_INLINE
