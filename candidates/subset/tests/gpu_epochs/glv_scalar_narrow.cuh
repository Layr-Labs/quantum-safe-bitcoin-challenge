// SPDX-License-Identifier: GPL-3.0-only
// Exact truncated GLV scalar arithmetic adapted from promoted pinning
// GLVScalar.cuh, public source b59484345df5208f5caffc82c25a4a3b50cbe523.
// That source credits may93182 for scalar support. Constants and lattice
// bounds originate in bitcoin-core/secp256k1 (Pieter Wuille, MIT), whose
// license is supplied in COPYING-secp256k1. Existing GPL notices apply.
// This header follows glv_round384's full-product reference definition.
#pragma once
#ifndef QSB_GLV_HIGH15
#define QSB_GLV_HIGH15 1
#endif
#ifndef QSB_GLV_RESIDUAL129
#define QSB_GLV_RESIDUAL129 1
#endif
#if (QSB_GLV_HIGH15 != 0 && QSB_GLV_HIGH15 != 1) || (QSB_GLV_RESIDUAL129 != 0 && QSB_GLV_RESIDUAL129 != 1)
#error GLV arithmetic switches must be 0 or 1
#endif
#ifdef __CUDACC__
#define GLV_NOINLINE __host__ __device__ __noinline__
#else
#define GLV_NOINLINE __attribute__((noinline))
#endif
struct glv_coeff_pair { uint64_t x,y; };
template<int WHICH>
GLV_NOINLINE glv_coeff_pair glv_coeff_fallback(uint64_t k0,uint64_t k1,uint64_t k2,uint64_t k3) {
    const uint64_t k[4]={k0,k1,k2,k3};
    const uint64_t g1[4]={0xe893209a45dbb031ULL,0x3daa8a1471e8ca7fULL,
                          0xe86c90e49284eb15ULL,0x3086d221a7d46bcdULL};
    const uint64_t g2[4]={0x1571b4ae8ac47f71ULL,0x221208ac9df506c6ULL,
                          0x6f547fa90abfe4c4ULL,0xe4437ed6010e8828ULL};
    uint64_t c[2];
    if(WHICH==1)glv_round384(c,k,g1);else glv_round384(c,k,g2);
    glv_coeff_pair r={c[0],c[1]};return r;
}
GLV_INLINE void glv_high15_add(uint64_t *acc,uint32_t *overflow,uint64_t product){
    uint64_t before=*acc;*acc=before+product;*overflow+=(uint32_t)(*acc<before);
}

#ifndef QSB_GLV_COEFF_BOUNDS
#define QSB_GLV_COEFF_BOUNDS 1
#endif
#if QSB_GLV_COEFF_BOUNDS != 0 && QSB_GLV_COEFF_BOUNDS != 1
#error QSB_GLV_COEFF_BOUNDS must be 0 or 1
#endif

/* For the two fixed reciprocals, b7+b6 is respectively0xd85b3dee and
 * 0xe55206fe. Every incoming high15 carry is below3*2^32. Thus the first
 * two products of each diagonal10..13, plus carry, fit64bits:
 * (2^32-1)*(b7+b6)+(3*2^32-1) < 2^64.
 * Later products retain their full overflow accounting. */
GLV_INLINE void glv_high15_begin(uint64_t *acc,uint32_t *overflow,
        uint64_t carry,uint64_t first,uint64_t second){
#if QSB_GLV_COEFF_BOUNDS
    *acc=carry+first+second;*overflow=0;
#else
    *acc=carry;*overflow=0;
    glv_high15_add(acc,overflow,first);
    glv_high15_add(acc,overflow,second);
#endif
}

template<int WHICH,uint32_t FALLBACK_WORD>
GLV_INLINE void glv_coeff_high15(uint64_t out[2],const uint64_t k[4],const uint64_t g[4]){
    const uint32_t a3=(uint32_t)(k[1]>>32);
    const uint32_t a4=(uint32_t)k[2],a5=(uint32_t)(k[2]>>32);
    const uint32_t a6=(uint32_t)k[3],a7=(uint32_t)(k[3]>>32);
    const uint32_t b3=(uint32_t)(g[1]>>32),b4=(uint32_t)g[2];
    const uint32_t b5=(uint32_t)(g[2]>>32),b6=(uint32_t)g[3],b7=(uint32_t)(g[3]>>32);
    uint64_t carry=0,acc;uint32_t overflow,w10,w11,w12,w13,w14,w15;

    /* Diagonal 10: (3,7)..(7,3). A 64-bit sum is insufficient for five
     * products, so overflow counts its lost 2^64 units explicitly. */
    glv_high15_begin(&acc,&overflow,carry,(uint64_t)a3*b7,(uint64_t)a4*b6);
    glv_high15_add(&acc,&overflow,(uint64_t)a5*b5);
    glv_high15_add(&acc,&overflow,(uint64_t)a6*b4);
    glv_high15_add(&acc,&overflow,(uint64_t)a7*b3);
    w10=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

    glv_high15_begin(&acc,&overflow,carry,(uint64_t)a4*b7,(uint64_t)a5*b6);
    glv_high15_add(&acc,&overflow,(uint64_t)a6*b5);
    glv_high15_add(&acc,&overflow,(uint64_t)a7*b4);
    w11=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

    glv_high15_begin(&acc,&overflow,carry,(uint64_t)a5*b7,(uint64_t)a6*b6);
    glv_high15_add(&acc,&overflow,(uint64_t)a7*b5);
    w12=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

    glv_high15_begin(&acc,&overflow,carry,(uint64_t)a6*b7,(uint64_t)a7*b6);
    w13=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

    acc=carry+(uint64_t)a7*b7;
    w14=(uint32_t)acc;w15=(uint32_t)(acc>>32);
    (void)w10;

    if(w11<FALLBACK_WORD || w11>=0x80000000U){
        uint64_t lo=(uint64_t)w12|((uint64_t)w13<<32);
        uint64_t hi=(uint64_t)w14|((uint64_t)w15<<32);
        const uint64_t round=(uint64_t)(w11>>31);
        uint64_t rounded=lo+round;out[0]=rounded;out[1]=hi+(uint64_t)(rounded<lo);
    }else{
        glv_coeff_pair r=glv_coeff_fallback<WHICH>(k[0],k[1],k[2],k[3]);
        out[0]=r.x;out[1]=r.y;
    }
}

struct glv_u129 { uint64_t lo,hi;uint32_t top; };

GLV_INLINE glv_u129 glv_product129(const uint64_t x[2],const uint32_t d[4]) {
    const uint32_t x0=(uint32_t)x[0],x1=(uint32_t)(x[0]>>32);
    const uint32_t x2=(uint32_t)x[1],x3=(uint32_t)(x[1]>>32);
    uint64_t t=(uint64_t)x0*d[0];const uint32_t w0=(uint32_t)t;uint64_t carry=t>>32;
    t=(uint64_t)x0*d[1]+carry;uint32_t w1=(uint32_t)t;carry=t>>32;
    t=(uint64_t)x0*d[2]+carry;uint32_t w2=(uint32_t)t;carry=t>>32;
    t=(uint64_t)x0*d[3]+carry;uint32_t w3=(uint32_t)t;uint32_t top=(uint32_t)(t>>32);
    t=(uint64_t)x1*d[0]+w1;w1=(uint32_t)t;carry=t>>32;
    t=(uint64_t)x1*d[1]+w2+carry;w2=(uint32_t)t;carry=t>>32;
    t=(uint64_t)x1*d[2]+w3+carry;w3=(uint32_t)t;top^=(uint32_t)(t>>32);
    t=(uint64_t)x2*d[0]+w2;w2=(uint32_t)t;carry=t>>32;
    t=(uint64_t)x2*d[1]+w3+carry;w3=(uint32_t)t;top^=(uint32_t)(t>>32);
    t=(uint64_t)x3*d[0]+w3;w3=(uint32_t)t;top^=(uint32_t)(t>>32);
    top^=(x1&d[3])^(x2&d[2])^(x3&d[1]);
    glv_u129 r={(uint64_t)w0|((uint64_t)w1<<32),
                 (uint64_t)w2|((uint64_t)w3<<32),top&1U};
    return r;
}

// Portable closed carry chains: host and CUDA compile the same arithmetic.
GLV_INLINE glv_u129 glv_sub129(glv_u129 a,glv_u129 b) {
    const __uint128_t t0=(__uint128_t)a.lo-b.lo;
    const __uint128_t t1=(__uint128_t)a.hi-b.hi-(uint64_t)(t0>>127);
    const uint32_t top=(a.top-b.top-(uint32_t)(t1>>127))&1U;
    glv_u129 r={(uint64_t)t0,(uint64_t)t1,top};return r;
}

// Let a=a1, b=-b1 and c=a2=a+b. Then P=a*(c1+c2), Q=b*c2,
// R=c*c1 give residuals k-P-Q and R-P. Only their low 129 bits are
// required because the exact rounded residuals lie strictly in (-2^128,2^128).
// The final odd-basis adjustment is performed by the caller in 192 bits.
GLV_INLINE void glv_residual129(uint64_t (&u)[3],uint64_t (&v)[3],
        const uint64_t (&k)[4],const uint64_t (&c1)[2],const uint64_t (&c2)[2]) {
    const uint32_t a[4]={0x9284eb15U,0xe86c90e4U,0xa7d46bcdU,0x3086d221U};
    const uint32_t b[4]={0x0abfe4c3U,0x6f547fa9U,0x010e8828U,0xe4437ed6U};
    const uint32_t c[4]={0x9d44cfd8U,0x57c1108dU,0xa8e2f3f6U,0x14ca50f7U};
    const __uint128_t s0=(__uint128_t)c1[0]+c2[0];
    const __uint128_t s1=(__uint128_t)c1[1]+c2[1]+(uint64_t)(s0>>64);
    const uint64_t sum[2]={(uint64_t)s0,(uint64_t)s1};
    glv_u129 p=glv_product129(sum,a);
    p.top^=(uint32_t)(s1>>64)&(a[0]&1U);
    const glv_u129 q=glv_product129(c2,b);
    glv_u129 r=glv_product129(c1,c);
    r.top^=(uint32_t)(c1[0]&1ULL); // implicit bit 128 of c
    const glv_u129 kk={k[0],k[1],(uint32_t)(k[2]&1ULL)};
    const glv_u129 z1=glv_sub129(glv_sub129(kk,p),q),z2=glv_sub129(r,p);
    u[0]=z1.lo;u[1]=z1.hi;u[2]=0ULL-(uint64_t)z1.top;
    v[0]=z2.lo;v[1]=z2.hi;v[2]=0ULL-(uint64_t)z2.top;
}
