#pragma once
#include <stdint.h>

#ifndef QSB_BIGTBL
#define QSB_BIGTBL 1
#endif
#if QSB_BIGTBL != 0 && QSB_BIGTBL != 1
#error QSB_BIGTBL must be 0 or 1
#endif

#if QSB_BIGTBL
// BEGIN QSB_BIGTBL_HOST_EXACT
/* Six terms per signed GLV component. The split below is unchanged. Its
 * rounded reciprocal error gives |r_i| <
 * 0xa2a8918ca85bafe22016d0b917e4dd77 (libsecp256k1's (a1+a2+1)/2).
 * Shifts 0,18,37,55,73,100; widths 18 unsigned, 19, 18, 18, 27 signed.
 * At shift 100 the largest top field is 170559768, so T=170559769 (the
 * smallest odd T >= it) makes d_top=2*f-T odd, nonzero and in [-T,T]
 * for every f <= T: magnitudes up to (T+1)*2^100-1 decode, 1.2*2^100 above
 * the bound. No residual truncation is used. The segment-0 bias is
 * K=(T+1)*2^99-2^17=170559770*2^99-2^17: the middle and top biases
 * telescope, so the six digits sum exactly to the magnitude.
 * Physical order 0,1,2,3,4,5 keeps the four small segments (48 MiB) first.
 * These portable helpers are also compiled verbatim by check_bigtable.py. */
__host__ __device__ __forceinline__ unsigned q9_bigtbl_entries(int c) {
    return c<2 ? 262144u : (c<4 ? 131072u : (c==4 ? 67108864u : 85279885u));
}
__host__ __device__ __forceinline__ unsigned q9_bigtbl_offset(int c) {
    return c==0?0u:c==1?262144u:c==2?524288u:c==3?655360u:
           c==4?786432u:67895296u;
}
__host__ __device__ __forceinline__ unsigned q9_bigtbl_shift(int c) {
    return c==0?0u:c==1?18u:c==2?37u:c==3?55u:c==4?73u:100u;
}
__host__ __device__ __forceinline__ uint32_t q9_bigtbl_code(
    const uint64_t mag[2],unsigned sign,int c) {
    const unsigned shift=q9_bigtbl_shift(c);
    uint64_t wide;
    if(shift<64u) {
        wide=mag[0]>>shift;
        if(shift) wide|=mag[1]<<(64u-shift);
    } else wide=mag[1]>>(shift-64u);
    uint32_t f=(uint32_t)wide,idx,neg_digit;
    if(c==0) {
        idx=f&((1u<<18)-1u);neg_digit=0;
    } else if(c==5) {
        const int32_t d=(int32_t)(2u*f)-170559769;
        neg_digit=(uint32_t)d>>31;
        const uint32_t ad=((uint32_t)d^(0u-neg_digit))+neg_digit;
        idx=(ad-1u)>>1;
    } else {
        const unsigned bits=c==1?19u:(c==4?27u:18u);
        f&=(1u<<bits)-1u;
        neg_digit=1u-(f>>(bits-1u));
        idx=(f^(0u-neg_digit))&((1u<<(bits-1u))-1u);
    }
    return (q9_bigtbl_offset(c)+idx)|((neg_digit^sign)<<31);
}
#if QSB_GLV11
__host__ __device__ __forceinline__ uint32_t q11_bigtbl_code(const uint64_t mag[2],unsigned sign,int c) {
#if QSB_GLV11_P18
 const unsigned shift=c==0?0u:c==1?18u:c==2?45u:c==3?73u:100u;
 uint64_t wide=shift<64 ? (mag[0]>>shift)|(shift ? mag[1]<<(64-shift) : 0) : mag[1]>>(shift-64);
 uint32_t f=(uint32_t)wide,idx,neg;
 if(c==0) {idx=f&0x3ffffu;neg=0;}
 else if(c==4) {uint32_t d=2u*f-170559769u;neg=d>>31;idx=(((d^(0u-neg))+neg)-1u)>>1;}
 else {unsigned width=c==2?28u:27u;f&=(1u<<width)-1;neg=1u-(f>>(width-1));idx=(f^(0u-neg))&((1u<<(width-1))-1);}
 unsigned off=c==0?0u:c==1?153175181u:c==2?220284045u:c==3?786432u:67895296u;
#else
 const unsigned shift=c==0?0u:c==1?23u:c==2?48u:c==3?73u:100u;
 uint64_t wide=shift<64 ? (mag[0]>>shift)|(shift ? mag[1]<<(64-shift) : 0) : mag[1]>>(shift-64);
 uint32_t f=(uint32_t)wide,idx,neg;
 if(c==0) {idx=f&0x7fffffu;neg=0;}
 else if(c==4) {uint32_t d=2u*f-170559769u;neg=d>>31;idx=(((d^(0u-neg))+neg)-1u)>>1;}
 else {unsigned width=c==3?27u:25u;f&=(1u<<width)-1;neg=1u-(f>>(width-1));idx=(f^(0u-neg))&((1u<<(width-1))-1);}
 unsigned off=c==0?153175181u:c==1?161563789u:c==2?178341005u:c==3?786432u:67895296u;
#endif
 return (off+idx)|((neg^sign)<<31);
}
#endif
// END QSB_BIGTBL_HOST_EXACT
#endif

// QSB/VanitySearch GPLv3 exact wide-product schedule, without field reduction.
__device__ __forceinline__ void q9_wide(uint64_t out[8],const uint64_t a[4],const uint64_t b[4]){
    uint64_t r0,r1,r2,r3,r4,r5,r6,r7;
    asm(
        "{\n"
        "\t.reg .u32 a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7;\n"
        "\t.reg .u64 e0,e1,e2,e3,e4,e5,e6,e7,o0,o1,o2,o3,o4,o5,o6,t,lc;\n"
        "\t.reg .u32 cy,o15;\n"
        "\t.reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n"
        "\t.reg .u32 y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14;\n"
        "\tmov.b64 {a0,a1}, %8;\n"
        "\tmov.b64 {a2,a3}, %9;\n"
        "\tmov.b64 {a4,a5}, %10;\n"
        "\tmov.b64 {a6,a7}, %11;\n"
        "\tmov.b64 {b0,b1}, %12;\n"
        "\tmov.b64 {b2,b3}, %13;\n"
        "\tmov.b64 {b4,b5}, %14;\n"
        "\tmov.b64 {b6,b7}, %15;\n"
        "\t.reg .u64 odd_t,odd_lc; .reg .u32 odd_cy;\n"
        "mul.wide.u32 e0, a0, b0;\n"
        "mul.wide.u32 o0, a0, b1;\n"
        "mul.wide.u32 e1, a0, b2;\n"
        "mul.wide.u32 o1, a0, b3;\n"
        "mul.wide.u32 e2, a0, b4;\n"
        "mul.wide.u32 o2, a0, b5;\n"
        "mul.wide.u32 e3, a0, b6;\n"
        "mul.wide.u32 o3, a0, b7;\n"
        "mul.wide.u32 t, a1, b1;\n"
        "mul.wide.u32 odd_t, a1, b0;\n"
        "add.cc.u64 e1, e1, t;\n"
        "mul.wide.u32 t, a1, b3;\n"
        "addc.cc.u64 e2, e2, t;\n"
        "mul.wide.u32 t, a1, b5;\n"
        "addc.cc.u64 e3, e3, t;\n"
        "mul.wide.u32 t, a1, b7;\n"
        "addc.u64 e4, t, 0;\n"
        "add.cc.u64 o0, o0, odd_t;\n"
        "mul.wide.u32 odd_t, a1, b2;\n"
        "addc.cc.u64 o1, o1, odd_t;\n"
        "mul.wide.u32 odd_t, a1, b4;\n"
        "addc.cc.u64 o2, o2, odd_t;\n"
        "mul.wide.u32 odd_t, a1, b6;\n"
        "addc.cc.u64 o3, o3, odd_t;\n"
        "addc.u32 odd_cy, 0, 0;\n"
        "mul.wide.u32 t, a2, b0;\n"
        "cvt.u64.u32 odd_lc, odd_cy;\n"
        "add.cc.u64 e1, e1, t;\n"
        "mul.wide.u32 t, a2, b2;\n"
        "addc.cc.u64 e2, e2, t;\n"
        "mul.wide.u32 t, a2, b4;\n"
        "addc.cc.u64 e3, e3, t;\n"
        "mul.wide.u32 t, a2, b6;\n"
        "addc.cc.u64 e4, e4, t;\n"
        "addc.u32 cy, 0, 0;\n"
        "mul.wide.u32 odd_t, a2, b1;\n"
        "cvt.u64.u32 lc, cy;\n"
        "add.cc.u64 o1, o1, odd_t;\n"
        "mul.wide.u32 odd_t, a2, b3;\n"
        "addc.cc.u64 o2, o2, odd_t;\n"
        "mul.wide.u32 odd_t, a2, b5;\n"
        "addc.cc.u64 o3, o3, odd_t;\n"
        "mul.wide.u32 odd_t, a2, b7;\n"
        "addc.u64 o4, odd_t, odd_lc;\n"
        "mul.wide.u32 t, a3, b1;\n"
        "mul.wide.u32 odd_t, a3, b0;\n"
        "add.cc.u64 e2, e2, t;\n"
        "mul.wide.u32 t, a3, b3;\n"
        "addc.cc.u64 e3, e3, t;\n"
        "mul.wide.u32 t, a3, b5;\n"
        "addc.cc.u64 e4, e4, t;\n"
        "mul.wide.u32 t, a3, b7;\n"
        "addc.u64 e5, t, lc;\n"
        "add.cc.u64 o1, o1, odd_t;\n"
        "mul.wide.u32 odd_t, a3, b2;\n"
        "addc.cc.u64 o2, o2, odd_t;\n"
        "mul.wide.u32 odd_t, a3, b4;\n"
        "addc.cc.u64 o3, o3, odd_t;\n"
        "mul.wide.u32 odd_t, a3, b6;\n"
        "addc.cc.u64 o4, o4, odd_t;\n"
        "addc.u32 odd_cy, 0, 0;\n"
        "mul.wide.u32 t, a4, b0;\n"
        "cvt.u64.u32 odd_lc, odd_cy;\n"
        "add.cc.u64 e2, e2, t;\n"
        "mul.wide.u32 t, a4, b2;\n"
        "addc.cc.u64 e3, e3, t;\n"
        "mul.wide.u32 t, a4, b4;\n"
        "addc.cc.u64 e4, e4, t;\n"
        "mul.wide.u32 t, a4, b6;\n"
        "addc.cc.u64 e5, e5, t;\n"
        "addc.u32 cy, 0, 0;\n"
        "mul.wide.u32 odd_t, a4, b1;\n"
        "cvt.u64.u32 lc, cy;\n"
        "add.cc.u64 o2, o2, odd_t;\n"
        "mul.wide.u32 odd_t, a4, b3;\n"
        "addc.cc.u64 o3, o3, odd_t;\n"
        "mul.wide.u32 odd_t, a4, b5;\n"
        "addc.cc.u64 o4, o4, odd_t;\n"
        "mul.wide.u32 odd_t, a4, b7;\n"
        "addc.u64 o5, odd_t, odd_lc;\n"
        "mul.wide.u32 t, a5, b1;\n"
        "mul.wide.u32 odd_t, a5, b0;\n"
        "add.cc.u64 e3, e3, t;\n"
        "mul.wide.u32 t, a5, b3;\n"
        "addc.cc.u64 e4, e4, t;\n"
        "mul.wide.u32 t, a5, b5;\n"
        "addc.cc.u64 e5, e5, t;\n"
        "mul.wide.u32 t, a5, b7;\n"
        "addc.u64 e6, t, lc;\n"
        "add.cc.u64 o2, o2, odd_t;\n"
        "mul.wide.u32 odd_t, a5, b2;\n"
        "addc.cc.u64 o3, o3, odd_t;\n"
        "mul.wide.u32 odd_t, a5, b4;\n"
        "addc.cc.u64 o4, o4, odd_t;\n"
        "mul.wide.u32 odd_t, a5, b6;\n"
        "addc.cc.u64 o5, o5, odd_t;\n"
        "addc.u32 odd_cy, 0, 0;\n"
        "mul.wide.u32 t, a6, b0;\n"
        "cvt.u64.u32 odd_lc, odd_cy;\n"
        "add.cc.u64 e3, e3, t;\n"
        "mul.wide.u32 t, a6, b2;\n"
        "addc.cc.u64 e4, e4, t;\n"
        "mul.wide.u32 t, a6, b4;\n"
        "addc.cc.u64 e5, e5, t;\n"
        "mul.wide.u32 t, a6, b6;\n"
        "addc.cc.u64 e6, e6, t;\n"
        "addc.u32 cy, 0, 0;\n"
        "mul.wide.u32 odd_t, a6, b1;\n"
        "cvt.u64.u32 lc, cy;\n"
        "add.cc.u64 o3, o3, odd_t;\n"
        "mul.wide.u32 odd_t, a6, b3;\n"
        "addc.cc.u64 o4, o4, odd_t;\n"
        "mul.wide.u32 odd_t, a6, b5;\n"
        "addc.cc.u64 o5, o5, odd_t;\n"
        "mul.wide.u32 odd_t, a6, b7;\n"
        "addc.u64 o6, odd_t, odd_lc;\n"
        "mul.wide.u32 t, a7, b1;\n"
        "mul.wide.u32 odd_t, a7, b0;\n"
        "add.cc.u64 e4, e4, t;\n"
        "mul.wide.u32 t, a7, b3;\n"
        "addc.cc.u64 e5, e5, t;\n"
        "mul.wide.u32 t, a7, b5;\n"
        "addc.cc.u64 e6, e6, t;\n"
        "mul.wide.u32 t, a7, b7;\n"
        "addc.u64 e7, t, lc;\n"
        "add.cc.u64 o3, o3, odd_t;\n"
        "mul.wide.u32 odd_t, a7, b2;\n"
        "addc.cc.u64 o4, o4, odd_t;\n"
        "mul.wide.u32 odd_t, a7, b4;\n"
        "addc.cc.u64 o5, o5, odd_t;\n"
        "mul.wide.u32 odd_t, a7, b6;\n"
        "addc.cc.u64 o6, o6, odd_t;\n"
        "addc.u32 o15, 0, 0;\n"
        "mov.b64 {x0,x1}, e0;\n"
        "\tmov.b64 {x2,x3}, e1;\n"
        "\tmov.b64 {x4,x5}, e2;\n"
        "\tmov.b64 {x6,x7}, e3;\n"
        "\tmov.b64 {x8,x9}, e4;\n"
        "\tmov.b64 {x10,x11}, e5;\n"
        "\tmov.b64 {x12,x13}, e6;\n"
        "\tmov.b64 {x14,x15}, e7;\n"
        "\tmov.b64 {y1,y2}, o0;\n"
        "\tmov.b64 {y3,y4}, o1;\n"
        "\tmov.b64 {y5,y6}, o2;\n"
        "\tmov.b64 {y7,y8}, o3;\n"
        "\tmov.b64 {y9,y10}, o4;\n"
        "\tmov.b64 {y11,y12}, o5;\n"
        "\tmov.b64 {y13,y14}, o6;\n"
        "\tadd.cc.u32 x1, x1, y1;\n"
        "\taddc.cc.u32 x2, x2, y2;\n"
        "\taddc.cc.u32 x3, x3, y3;\n"
        "\taddc.cc.u32 x4, x4, y4;\n"
        "\taddc.cc.u32 x5, x5, y5;\n"
        "\taddc.cc.u32 x6, x6, y6;\n"
        "\taddc.cc.u32 x7, x7, y7;\n"
        "\taddc.cc.u32 x8, x8, y8;\n"
        "\taddc.cc.u32 x9, x9, y9;\n"
        "\taddc.cc.u32 x10, x10, y10;\n"
        "\taddc.cc.u32 x11, x11, y11;\n"
        "\taddc.cc.u32 x12, x12, y12;\n"
        "\taddc.cc.u32 x13, x13, y13;\n"
        "\taddc.cc.u32 x14, x14, y14;\n"
        "\taddc.u32 x15, x15, o15;\n"
        "\t\n"
        "mov.b64 %0, {x0,x1};\n"
        "mov.b64 %1, {x2,x3};\n"
        "mov.b64 %2, {x4,x5};\n"
        "mov.b64 %3, {x6,x7};\n"
        "mov.b64 %4, {x8,x9};\n"
        "mov.b64 %5, {x10,x11};\n"
        "mov.b64 %6, {x12,x13};\n"
        "mov.b64 %7, {x14,x15};\n"
        "}\n"

        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3),"=l"(r4),"=l"(r5),"=l"(r6),"=l"(r7)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    out[0]=r0;out[1]=r1;out[2]=r2;out[3]=r3;out[4]=r4;out[5]=r5;out[6]=r6;out[7]=r7;
}
// GLV lattice and rounded-reciprocal constants from bitcoin-core/secp256k1
// v0.6.0 scalar_impl.h, Copyright (c) 2014 Pieter Wuille, MIT.
// The original MIT license is supplied as COPYING-secp256k1.
#ifndef QSB_GLV_HIGH15
#define QSB_GLV_HIGH15 1
#endif
#if QSB_GLV_HIGH15 != 0 && QSB_GLV_HIGH15 != 1
#error QSB_GLV_HIGH15 must be 0 or 1
#endif

/* Exact original reference and rare out-of-line wrapper. q9_coeff_high15 computes only product
 * diagonals 10..14. The fixed omitted low part is too small to change the
 * bit-383 rounding decision except in FALLBACK_WORD..0x7fffffff. Keeping this
 * path out of line prevents its full 64-product register set from becoming
 * live in the ordinary path. */
__device__ __forceinline__ void q9_coeff_reference(uint64_t out[2],const uint64_t k[4],const uint64_t g[4]){
    uint64_t p[8];q9_wide(p,k,g);
    __uint128_t t=(__uint128_t)p[6]+(p[5]>>63);out[0]=(uint64_t)t;out[1]=p[7]+(uint64_t)(t>>64);
}
/* QSB_GLV_FALLBACK_INLINE (subset): 1 = the rare exact-rounding path is inlined into its (rarely
 * taken) branch. Subset's chain runs inside a __noinline__ front function; a call from there is a nested
 * call, which makes ptxas keep the return address on the stack (STACK 16, spills around both front
 * calls). 0 = pinning's out-of-line form. */
#ifndef QSB_GLV_FALLBACK_INLINE
#define QSB_GLV_FALLBACK_INLINE 0
#endif
#if QSB_GLV_FALLBACK_INLINE
#define QSB_GLV_FALLBACK_ATTR __forceinline__
#else
#define QSB_GLV_FALLBACK_ATTR __noinline__
#endif
template<int WHICH>
__device__ QSB_GLV_FALLBACK_ATTR ulonglong2 q9_coeff_fallback(uint64_t k0,uint64_t k1,
                                                      uint64_t k2,uint64_t k3){
    const uint64_t k[4]={k0,k1,k2,k3};
    const uint64_t g1[4]={0xE893209A45DBB031ULL,0x3DAA8A1471E8CA7FULL,
                          0xE86C90E49284EB15ULL,0x3086D221A7D46BCDULL};
    const uint64_t g2[4]={0x1571B4AE8AC47F71ULL,0x221208AC9DF506C6ULL,
                          0x6F547FA90ABFE4C4ULL,0xE4437ED6010E8828ULL};
    uint64_t out[2];q9_coeff_reference(out,k,WHICH==1?g1:g2);
    ulonglong2 r;r.x=out[0];r.y=out[1];return r;
}

#ifndef QSB_GLV_LEAN
#define QSB_GLV_LEAN 1
#endif
#if QSB_GLV_LEAN != 0 && QSB_GLV_LEAN != 1
#error QSB_GLV_LEAN must be 0 or 1
#endif
#if QSB_GLV_LEAN
__device__ __forceinline__ uint64_t q9_mulw(uint32_t a,uint32_t b){
#ifdef __CUDA_ARCH__
    uint64_t r;asm("mul.wide.u32 %0,%1,%2;":"=l"(r):"r"(a),"r"(b));return r;
#else
    return (uint64_t)a*b;
#endif
}
__device__ __forceinline__ uint64_t q9_madw(uint32_t a,uint32_t b,uint64_t c){
#ifdef __CUDA_ARCH__
    uint64_t r;asm("mad.wide.u32 %0,%1,%2,%3;":"=l"(r):"r"(a),"r"(b),"l"(c));return r;
#else
    return (uint64_t)a*b+c;
#endif
}
#define QSB_GLV_PRODUCT(a,b) q9_mulw(a,b)
#else
#define QSB_GLV_PRODUCT(a,b) ((uint64_t)(a)*(b))
#endif

__device__ __forceinline__ void q9_high15_add(uint64_t *acc,uint32_t *overflow,uint64_t product){
#if QSB_GLV_LEAN && defined(__CUDA_ARCH__)
    /* The same sum and lost-2^64 count, taken from the add's carry flag. */
    asm("{add.cc.u64 %0,%0,%2; addc.u32 %1,%1,0;}":"+l"(*acc),"+r"(*overflow):"l"(product));
#else
    uint64_t before=*acc;*acc=before+product;*overflow+=(uint32_t)(*acc<before);
#endif
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
__device__ __forceinline__ void q9_high15_begin(uint64_t *acc,uint32_t *overflow,
        uint64_t carry,uint64_t first,uint64_t second){
#if QSB_GLV_COEFF_BOUNDS
    *acc=carry+first+second;*overflow=0;
#else
    *acc=carry;*overflow=0;
    q9_high15_add(acc,overflow,first);
    q9_high15_add(acc,overflow,second);
#endif
}

#ifndef QSB_GLV_ROUND_CC
#define QSB_GLV_ROUND_CC 1
#endif
#if QSB_GLV_ROUND_CC != 0 && QSB_GLV_ROUND_CC != 1
#error "QSB_GLV_ROUND_CC must be 0 or 1"
#endif
__device__ __forceinline__ void q9_round_coeff(uint64_t out[2],uint64_t lo,uint64_t hi,uint64_t round) {
#if QSB_GLV_ROUND_CC && defined(__CUDA_ARCH__)
    asm("{add.cc.u64 %0,%2,%4; addc.u64 %1,%3,0;}"
        : "=&l"(out[0]),"=l"(out[1]) : "l"(lo),"l"(hi),"l"(round));
#else
    const uint64_t rounded=lo+round;
    out[0]=rounded;out[1]=hi+(uint64_t)(rounded<lo);
#endif
}

/* Exact high-half diagonal10: the discarded carry can change word11 by at
 * most8 (g1) or7 (g2); the coefficient wrappers widen the exact fallback band. */
#ifndef QSB_GLV_HIGH15_HI
#define QSB_GLV_HIGH15_HI 1
#endif
#if QSB_GLV_HIGH15_HI != 0 && QSB_GLV_HIGH15_HI != 1
#error "QSB_GLV_HIGH15_HI must be 0 or 1"
#endif

/* QSB_GLV_RND (port of our pinning tree's QSB_GLV_GLUE bit 4, "rounding takes bit 31"): the same rounded
 * coefficient c = floor(P'/2^384) + bit 383 of P' mod 2^128 (P' = the high15 approximation of k*g, words
 * w11..w15 above bit 352) in fewer instructions; the exact-fallback band is unchanged.
 *  1: pinning's form. The rounding bit is the carry out of w11 + 2^31, fed straight into the 128-bit add
 *     of (w12..w15): no shift and mask of w11.
 *  2: the 2^31 rides in the diagonal-10 carry, so diagonal 11 accumulates w11 + 2^31 and its carry into
 *     diagonal 12 already holds the rounding bit: w12..w15 ARE floor((P' + 2^383)/2^384) and no rounding
 *     add is left. The band test w11 in [FALLBACK_WORD, 2^31) becomes w11' = w11 + 2^31 mod 2^32 in
 *     [2^31 + FALLBACK_WORD, 2^32): one unsigned compare. Diagonal 11 cannot overflow its 64-bit sum (the
 *     carry grows by 2^31 < 2^32; COEFF_BOUNDS needs carry < (2^32 - b7 - b6) * 2^32).
 * (ptxas places the 2^31 as one IADD3 + IMAD.X inside diagonal 11 and adds 3 moves at the fallback
 * join, so form 2 nets -3 slots per candidate, form 1 -2; writing the diagonal-10 sum as mad.hi + 32-bit
 * adds so the constant rides in an existing IADD3 gave -1 and +1)
 * Both forms give floor((P' + 2^383)/2^384) mod 2^128 on the same P' and the same band, so every
 * coefficient is bit-identical to the base for every k. 0 = q9_round_coeff byte for byte. */
#ifndef QSB_GLV_RND
#define QSB_GLV_RND 2
#endif
#if QSB_GLV_RND < 0 || QSB_GLV_RND > 2
#error "QSB_GLV_RND must be 0, 1 or 2"
#endif

template<int WHICH,uint32_t FALLBACK_WORD>
__device__ __forceinline__ void q9_coeff_high15(uint64_t out[2],const uint64_t k[4],const uint64_t g[4]){
    const uint32_t a3=(uint32_t)(k[1]>>32);
    const uint32_t a4=(uint32_t)k[2],a5=(uint32_t)(k[2]>>32);
    const uint32_t a6=(uint32_t)k[3],a7=(uint32_t)(k[3]>>32);
    const uint32_t b3=(uint32_t)(g[1]>>32),b4=(uint32_t)g[2];
    const uint32_t b5=(uint32_t)(g[2]>>32),b6=(uint32_t)g[3],b7=(uint32_t)(g[3]>>32);
    uint64_t carry=0,acc;uint32_t overflow,w10,w11,w12,w13,w14,w15;

#if QSB_GLV_HIGH15_HI
    /* Only the carry into diagonal11 is observed. Retain each product's
     * high32 and bound the omitted sum of five low32 halves separately. */
    carry=(uint64_t)__umulhi(a3,b7)+(uint64_t)__umulhi(a4,b6)
         +(uint64_t)__umulhi(a5,b5)+(uint64_t)__umulhi(a6,b4)
         +(uint64_t)__umulhi(a7,b3);
#else
    /* Diagonal 10: (3,7)..(7,3). A 64-bit sum is insufficient for five
     * products, so overflow counts its lost 2^64 units explicitly. */
    q9_high15_begin(&acc,&overflow,carry,QSB_GLV_PRODUCT(a3,b7),QSB_GLV_PRODUCT(a4,b6));
    q9_high15_add(&acc,&overflow,QSB_GLV_PRODUCT(a5,b5));
    q9_high15_add(&acc,&overflow,QSB_GLV_PRODUCT(a6,b4));
    q9_high15_add(&acc,&overflow,QSB_GLV_PRODUCT(a7,b3));
    w10=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

#endif
#if QSB_GLV_RND == 2
    carry+=0x80000000ULL;   /* + 2^383: diagonal 11 now accumulates w11 + 2^31 */
#endif

    q9_high15_begin(&acc,&overflow,carry,QSB_GLV_PRODUCT(a4,b7),QSB_GLV_PRODUCT(a5,b6));
    q9_high15_add(&acc,&overflow,QSB_GLV_PRODUCT(a6,b5));
    q9_high15_add(&acc,&overflow,QSB_GLV_PRODUCT(a7,b4));
    w11=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

    q9_high15_begin(&acc,&overflow,carry,QSB_GLV_PRODUCT(a5,b7),QSB_GLV_PRODUCT(a6,b6));
    q9_high15_add(&acc,&overflow,QSB_GLV_PRODUCT(a7,b5));
    w12=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

    q9_high15_begin(&acc,&overflow,carry,QSB_GLV_PRODUCT(a6,b7),QSB_GLV_PRODUCT(a7,b6));
    w13=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

    #if QSB_GLV_LEAN
    acc=q9_madw(a7,b7,carry);
#else
    acc=carry+QSB_GLV_PRODUCT(a7,b7);
#endif
    w14=(uint32_t)acc;w15=(uint32_t)(acc>>32);
    (void)w10;

#if QSB_GLV_RND == 2
    /* w11 here is w11 + 2^31 mod 2^32 and w12..w15 already carry the rounding bit. */
    if(__builtin_expect(w11<0x80000000U+FALLBACK_WORD,1)){
        out[0]=(uint64_t)w12|((uint64_t)w13<<32);
        out[1]=(uint64_t)w14|((uint64_t)w15<<32);
    }else{
#else
    if(w11<FALLBACK_WORD || w11>=0x80000000U){
        uint64_t lo=(uint64_t)w12|((uint64_t)w13<<32);
        uint64_t hi=(uint64_t)w14|((uint64_t)w15<<32);
#if QSB_GLV_RND == 1 && defined(__CUDA_ARCH__)
        /* (hi:lo) + (w11 >> 31) mod 2^128: the carry out of w11 + 2^31 is bit 31 of w11. */
        asm("{\n\t.reg .u32 t;\n\t"
            "add.cc.u32 t,%4,0x80000000;\n\t"
            "addc.cc.u64 %0,%2,0;\n\t"
            "addc.u64 %1,%3,0;\n\t}"
            : "=l"(out[0]),"=l"(out[1]) : "l"(lo),"l"(hi),"r"(w11));
#else
        const uint64_t round=(uint64_t)(w11>>31);
        q9_round_coeff(out,lo,hi,round);
#endif
    }else{
#endif
        ulonglong2 r=q9_coeff_fallback<WHICH>(k[0],k[1],k[2],k[3]);
        out[0]=r.x;out[1]=r.y;
    }
}

__device__ __forceinline__ void q9_coeff_g1(uint64_t out[2],const uint64_t k[4],const uint64_t g[4]){
#if QSB_GLV_HIGH15
#if QSB_GLV_HIGH15_HI
    q9_coeff_high15<1,0x7ffffff8U>(out,k,g);
#else
    q9_coeff_high15<1,0x7ffffffcU>(out,k,g);
#endif
#else
    q9_coeff_reference(out,k,g);
#endif
}
__device__ __forceinline__ void q9_coeff_g2(uint64_t out[2],const uint64_t k[4],const uint64_t g[4]){
#if QSB_GLV_HIGH15
#if QSB_GLV_HIGH15_HI
    q9_coeff_high15<2,0x7ffffff9U>(out,k,g);
#else
    q9_coeff_high15<2,0x7ffffffdU>(out,k,g);
#endif
#else
    q9_coeff_reference(out,k,g);
#endif
}
__device__ __forceinline__ void q9_sub4(uint64_t out[4],const uint64_t a[4],const uint64_t b[4]){
    uint64_t r0,r1,r2,r3;
    asm("{sub.cc.u64 %0,%4,%8;subc.cc.u64 %1,%5,%9;subc.cc.u64 %2,%6,%10;subc.u64 %3,%7,%11;}"
        :"=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        :"l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    out[0]=r0;out[1]=r1;out[2]=r2;out[3]=r3;
}
template<int W> __device__ __forceinline__ void q9_small_product(uint64_t out[4],const uint64_t a[2],const uint32_t b[W]){
    uint32_t aa[4]={(uint32_t)a[0],(uint32_t)(a[0]>>32),(uint32_t)a[1],(uint32_t)(a[1]>>32)};
    uint32_t rr[9]={0,0,0,0,0,0,0,0,0};
    #pragma unroll
    for(int i=0;i<4;i++){
        uint64_t carry=0;
        #pragma unroll
        for(int j=0;j<W;j++){
            uint64_t t=(uint64_t)aa[i]*b[j]+rr[i+j]+carry;
            rr[i+j]=(uint32_t)t;carry=t>>32;
        }
        rr[i+W]=(uint32_t)carry;
    }
    #pragma unroll
    for(int j=0;j<4;j++)out[j]=(uint64_t)rr[2*j]|((uint64_t)rr[2*j+1]<<32);
}
__device__ __forceinline__ void q9_abs128(uint64_t out[2],unsigned *negative,const uint64_t in[4]){
    unsigned s=(unsigned)(in[3]>>63);uint64_t m=0ULL-s;
    __uint128_t t=(__uint128_t)(in[0]^m)+s;out[0]=(uint64_t)t;out[1]=(in[1]^m)+(uint64_t)(t>>64);*negative=s;
}

#ifndef QSB_GLV_RESIDUAL3
#define QSB_GLV_RESIDUAL3 1
#endif
#if QSB_GLV_RESIDUAL3 != 0 && QSB_GLV_RESIDUAL3 != 1
#error QSB_GLV_RESIDUAL3 must be 0 or 1
#endif
#ifndef QSB_GLV_RESIDUAL129
#define QSB_GLV_RESIDUAL129 1
#endif
#if QSB_GLV_RESIDUAL129 != 0 && QSB_GLV_RESIDUAL129 != 1
#error QSB_GLV_RESIDUAL129 must be 0 or 1
#endif

/* Original four-product residual schedule, retained as the exact OFF path. */
__device__ __forceinline__ void q9_glv_residual_reference(
    const uint64_t k[4],const uint64_t c1[2],const uint64_t c2[2],
    const uint32_t a1[4],const uint32_t a2[5],const uint32_t b1[4],
    uint64_t r1[2],uint64_t r2[2],unsigned *s1,unsigned *s2) {
    uint64_t p[4],q[4],z[4];
    q9_small_product<4>(p,c1,a1);q9_small_product<5>(q,c2,a2);
    q9_sub4(z,k,p);q9_sub4(z,z,q);q9_abs128(r1,s1,z);
    q9_small_product<4>(p,c1,b1);q9_small_product<4>(q,c2,a1);
    q9_sub4(z,p,q);q9_abs128(r2,s2,z);
}

__device__ __forceinline__ void q9_add_upper128(uint64_t x[4],uint64_t lo,uint64_t hi) {
    uint64_t r2,r3;
    asm("{add.cc.u64 %0,%2,%4;addc.u64 %1,%3,%5;}"
        : "=l"(r2),"=l"(r3) : "l"(x[2]),"l"(x[3]),"l"(lo),"l"(hi));
    x[2]=r2;x[3]=r3;
}

/* Three-product residual identity. With c=a+b:
 *   P=a*(c1+c2), Q=b*c2, R=c*c1;
 *   z1=k-P-Q, z2=R-P.
 * c1+c2 is explicitly 129 bits. The carry contributes (a<<128) modulo
 * 2^256; c's implicit top word one contributes (c1<<128). */
__device__ __forceinline__ void q9_glv_residual3(
    const uint64_t k[4],const uint64_t c1[2],const uint64_t c2[2],
    const uint32_t a1[4],const uint32_t a2[5],const uint32_t b1[4],
    uint64_t r1[2],uint64_t r2[2],unsigned *s1,unsigned *s2) {
    uint64_t sum0,sum1;uint32_t sum2;
    asm("{add.cc.u64 %0,%3,%5;addc.cc.u64 %1,%4,%6;addc.u32 %2,0,0;}"
        : "=l"(sum0),"=l"(sum1),"=r"(sum2)
        : "l"(c1[0]),"l"(c1[1]),"l"(c2[0]),"l"(c2[1]));
    const uint64_t sum[2]={sum0,sum1};
    const uint64_t a_lo=(uint64_t)a1[0]|((uint64_t)a1[1]<<32);
    const uint64_t a_hi=(uint64_t)a1[2]|((uint64_t)a1[3]<<32);
    const uint64_t carry_mask=0ULL-(uint64_t)sum2;
    uint64_t p[4],q[4],rr[4],z[4];
    q9_small_product<4>(p,sum,a1);
    q9_add_upper128(p,a_lo&carry_mask,a_hi&carry_mask);
    q9_small_product<4>(q,c2,b1);
    q9_small_product<4>(rr,c1,a2); /* a2[0..3] is c mod 2^128. */
    q9_add_upper128(rr,c1[0],c1[1]);
    q9_sub4(z,k,p);q9_sub4(z,z,q);q9_abs128(r1,s1,z);
    q9_sub4(z,rr,p);q9_abs128(r2,s2,z);
}

/* Exact modulo-2^129 arithmetic is sufficient here: the rounded GLV
 * coefficients guarantee both signed residuals have magnitude below 2^128.
 * The product helper retains words 0..3 and bit 128. Each truncated row's
 * carry and the parity of diagonal four both contribute to that top bit. */
struct q9_u129 { uint64_t lo,hi;uint32_t top; };

/* QSB_GLV_EO (from our pinning tree, QSB_GLV_GLUE bit 1): the same x*d mod 2^129 from column-pair
 * accumulators instead of row-wise multiply-adds, fewer carry-propagation instructions per product
 * (three products per GLV split). Bit-identical result. 0 = the row-wise form byte for byte. */
#ifndef QSB_GLV_EO
#define QSB_GLV_EO 1
#endif
#if QSB_GLV_EO != 0 && QSB_GLV_EO != 1
#error "QSB_GLV_EO must be 0 or 1"
#endif
/* QSB_DECODE_CUT (bit mask; after HY16's QSB_DECODE_CUT in the pinning record b9736ce1 by kaankolcu,
 * GLVScalar.cuh:53-72 there): fewer ALU instructions on the GLV residual path.
 *  bit 2: q9_product129_rx without materialised carries: the column-3 carry of O0 and the two column-4 carries
 *         of E1 are added into O1 (words 3..4) straight off the carry flag as they are produced, instead of being
 *         captured into registers (cc3, cc4) and added in the final word chain. O1 is kept mod 2^64 and word 4
 *         matters only for bit 0, so the result is the same x*d mod 2^129 (bit 0 of top) for every input:
 *         bit-identical. The asm is the record's, which starts from the same QSB_GLV_EO form as this tree.
 *  bit 1: q9_zwalk_value forms v = z - s*(1 + D*2^100) mod 2^128 as two 32-bit adds, one on word 0 (z - s) and
 *         one on word 3 (-s*16D mod 2^32 = 0xA2A891A0 & -s), instead of a 128-bit add with carries. It differs from
 *         the exact v only when s = 1 and word 0 of z is 0 (the borrow of z - 1 into word 1 is dropped: v comes out
 *         2^32 too large), about 2^-33 per residual and 2^-32 per candidate. Such a candidate walks a wrong but
 *         in-range scalar (only the field containing bit 32 moves, by one; the top field moves only if words 0..2
 *         of z are all 0) and is lost; the exact host gate re-derives every nomination, so it can never publish
 * a wrong hit.
 * Every value is in QSB_CARRIER_KNOBS. 0 = the previous q9_product129_rx and q9_zwalk_value byte for byte.
 * Default 2 (gate): bit 2 is -6 slots per candidate (census_b8, 20,554.2 against 20,560.2)
 * and bit-identical; bit 1 in this tree's walker form is +3 slots (ptxas splits the 64-bit words it had fused with
 * the walker's field extraction: +6 LOP3, +4 IMAD.IADD against -8 IADD3/IADD3.X), so it stays off. */
#ifndef QSB_DECODE_CUT
#if QSB_LOCAL_SM86
#define QSB_DECODE_CUT 0   /* dev-only sm_86 rig: needs the ZDEC walker */
#else
#define QSB_DECODE_CUT 2
#endif
#endif
#if QSB_DECODE_CUT < 0 || QSB_DECODE_CUT > 3
#error "QSB_DECODE_CUT is a mask of bits 1 and 2"
#endif
#if QSB_GLV_EO
/* x*d mod 2^129 from column-pair accumulators: E0 = x0d0 (words 0-1), O0 = x0d1+x1d0
 * (words 1-2, carry cc3 into word 3), E1 = x0d2+x1d1+x2d0 (words 2-3, carries cc4 into
 * word 4), O1 = x0d3+x1d2+x2d1+x3d0 mod 2^64 (words 3-4). Word 4 only matters for its
 * parity (bit 0 of top is bit 128); X is added into it: the parity terms, which include the
 * low bits of x1*d3, x2*d2 and x3*d1 (bit 0 of each is x_i & d_j & 1, so the words x_i with
 * odd d_j are added whole; their bits above bit 0 only reach unused bits of word 4). */
__device__ __forceinline__ q9_u129 q9_product129_rx(const uint64_t x[2],const uint32_t d[4],uint32_t X) {
    uint32_t w0,w1,w2,w3,top;
#if QSB_DECODE_CUT & 2
    /* QSB_DECODE_CUT bit 2: O1 first; cc3 (bit 0 of O1's word 3 slot) and the two cc4 carries (word 4) are
     * added into O1 straight off the carry flag. */
    asm("{\n\t"
        ".reg .u32 x0,x1,x2,x3,a,b,c,e;\n\t"
        ".reg .u64 E0,O0,E1,O1,m;\n\t"
        "mov.b64 {x0,x1},%5;\n\t"
        "mov.b64 {x2,x3},%6;\n\t"
        "mul.wide.u32 O1,x0,%10;\n\t"
        "mad.wide.u32 O1,x1,%9,O1;\n\t"
        "mad.wide.u32 O1,x2,%8,O1;\n\t"
        "mad.wide.u32 O1,x3,%7,O1;\n\t"
        "mov.b64 {c,e},O1;\n\t"
        "mul.wide.u32 E0,x0,%7;\n\t"
        "mul.wide.u32 O0,x0,%8;\n\t"
        "mul.wide.u32 m,x1,%7;\n\t"
        "add.cc.u64 O0,O0,m;\n\t"
        "addc.cc.u32 c,c,0;\n\t"
        "addc.u32 e,e,0;\n\t"
        "mul.wide.u32 E1,x0,%9;\n\t"
        "mul.wide.u32 m,x1,%8;\n\t"
        "add.cc.u64 E1,E1,m;\n\t"
        "addc.u32 e,e,0;\n\t"
        "mul.wide.u32 m,x2,%7;\n\t"
        "add.cc.u64 E1,E1,m;\n\t"
        "addc.u32 e,e,0;\n\t"
        "mov.b64 {%0,a},E0;\n\t"
        "mov.b64 {b,x0},O0;\n\t"
        "add.cc.u32 %1,a,b;\n\t"
        "mov.b64 {a,b},E1;\n\t"
        "addc.cc.u32 %2,a,x0;\n\t"
        "addc.cc.u32 %3,b,c;\n\t"
        "addc.u32 %4,e,%11;\n\t"
        "}"
        : "=r"(w0),"=r"(w1),"=r"(w2),"=r"(w3),"=r"(top)
        : "l"(x[0]),"l"(x[1]),"r"(d[0]),"r"(d[1]),"r"(d[2]),"r"(d[3]),"r"(X));
#else
    asm("{\n\t"
        ".reg .u32 x0,x1,x2,x3,a,b,c,e,t,cc3,cc4;\n\t"
        ".reg .u64 E0,O0,E1,O1,m;\n\t"
        "mov.b64 {x0,x1},%5;\n\t"
        "mov.b64 {x2,x3},%6;\n\t"
        "mul.wide.u32 E0,x0,%7;\n\t"
        "mul.wide.u32 O0,x0,%8;\n\t"
        "mul.wide.u32 m,x1,%7;\n\t"
        "add.cc.u64 O0,O0,m;\n\t"
        "addc.u32 cc3,0,0;\n\t"
        "mul.wide.u32 E1,x0,%9;\n\t"
        "mul.wide.u32 m,x1,%8;\n\t"
        "add.cc.u64 E1,E1,m;\n\t"
        "addc.u32 cc4,0,0;\n\t"
        "mul.wide.u32 m,x2,%7;\n\t"
        "add.cc.u64 E1,E1,m;\n\t"
        "addc.u32 cc4,cc4,0;\n\t"
        "mul.wide.u32 O1,x0,%10;\n\t"
        "mad.wide.u32 O1,x1,%9,O1;\n\t"
        "mad.wide.u32 O1,x2,%8,O1;\n\t"
        "mad.wide.u32 O1,x3,%7,O1;\n\t"
        "mov.b64 {%0,a},E0;\n\t"
        "mov.b64 {b,c},O0;\n\t"
        "add.cc.u32 %1,a,b;\n\t"
        "mov.b64 {a,b},E1;\n\t"
        "addc.cc.u32 %2,a,c;\n\t"
        "mov.b64 {c,e},O1;\n\t"
        "addc.cc.u32 %3,b,c;\n\t"
        "addc.u32 t,e,cc4;\n\t"
        "add.cc.u32 %3,%3,cc3;\n\t"
        "addc.u32 %4,t,%11;\n\t"
        "}"
        : "=r"(w0),"=r"(w1),"=r"(w2),"=r"(w3),"=r"(top)
        : "l"(x[0]),"l"(x[1]),"r"(d[0]),"r"(d[1]),"r"(d[2]),"r"(d[3]),"r"(X));
#endif
    q9_u129 r={(uint64_t)w0|((uint64_t)w1<<32),
               (uint64_t)w2|((uint64_t)w3<<32),top};   /* bit 0 of top is bit 128 */
    return r;
}
__device__ __forceinline__ q9_u129 q9_product129(const uint64_t x[2],const uint32_t d[4]) {
    const uint32_t x1=(uint32_t)(x[0]>>32),x2=(uint32_t)x[1],x3=(uint32_t)(x[1]>>32);
    q9_u129 r=q9_product129_rx(x,d,((d[3]&1U)?x1:0U)+((d[2]&1U)?x2:0U)+((d[1]&1U)?x3:0U));
    r.top&=1U;return r;
}
#else
__device__ __forceinline__ q9_u129 q9_product129(const uint64_t x[2],const uint32_t d[4]) {
    const uint32_t x0=(uint32_t)x[0],x1=(uint32_t)(x[0]>>32);
    const uint32_t x2=(uint32_t)x[1],x3=(uint32_t)(x[1]>>32);
#if QSB_GLV_LEAN
    uint64_t t=q9_mulw(x0,d[0]);const uint32_t w0=(uint32_t)t;uint64_t carry=t>>32;
    t=q9_madw(x0,d[1],carry);uint32_t w1=(uint32_t)t;carry=t>>32;
    t=q9_madw(x0,d[2],carry);uint32_t w2=(uint32_t)t;carry=t>>32;
    t=q9_madw(x0,d[3],carry);uint32_t w3=(uint32_t)t;uint32_t top=(uint32_t)(t>>32);
    t=q9_madw(x1,d[0],w1);w1=(uint32_t)t;carry=t>>32;
    t=q9_madw(x1,d[1],(uint64_t)w2+carry);w2=(uint32_t)t;carry=t>>32;
    t=q9_madw(x1,d[2],(uint64_t)w3+carry);w3=(uint32_t)t;top^=(uint32_t)(t>>32);
    t=q9_madw(x2,d[0],w2);w2=(uint32_t)t;carry=t>>32;
    t=q9_madw(x2,d[1],(uint64_t)w3+carry);w3=(uint32_t)t;top^=(uint32_t)(t>>32);
    t=q9_madw(x3,d[0],w3);w3=(uint32_t)t;top^=(uint32_t)(t>>32);
#else
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
#endif
    top^=(x1&d[3])^(x2&d[2])^(x3&d[1]);
    q9_u129 r={(uint64_t)w0|((uint64_t)w1<<32),
                 (uint64_t)w2|((uint64_t)w3<<32),top&1U};
    return r;
}
#endif

__device__ __forceinline__ q9_u129 q9_sub129(q9_u129 a,q9_u129 b) {
    q9_u129 r;uint32_t top;
    asm("{sub.cc.u64 %0,%3,%6;subc.cc.u64 %1,%4,%7;subc.u32 %2,%5,%8;}"
        : "=l"(r.lo),"=l"(r.hi),"=r"(top)
        : "l"(a.lo),"l"(a.hi),"r"(a.top),"l"(b.lo),"l"(b.hi),"r"(b.top));
    r.top=top&1U;return r;
}

__device__ __forceinline__ void q9_abs129(uint64_t out[2],unsigned *negative,q9_u129 in) {
    const unsigned s=in.top&1U;const uint64_t m=0ULL-(uint64_t)s;
    const __uint128_t t=(__uint128_t)(in.lo^m)+s;
    out[0]=(uint64_t)t;out[1]=(in.hi^m)+(uint64_t)(t>>64);*negative=s;
}

__device__ __forceinline__ void q9_glv_residual129(
    const uint64_t k[4],const uint64_t c1[2],const uint64_t c2[2],
    const uint32_t a1[4],const uint32_t a2[5],const uint32_t b1[4],
    uint64_t r1[2],uint64_t r2[2],unsigned *s1,unsigned *s2) {
    uint64_t sum0,sum1;uint32_t sum2;
    asm("{add.cc.u64 %0,%3,%5;addc.cc.u64 %1,%4,%6;addc.u32 %2,0,0;}"
        : "=l"(sum0),"=l"(sum1),"=r"(sum2)
        : "l"(c1[0]),"l"(c1[1]),"l"(c2[0]),"l"(c2[1]));
    const uint64_t sum[2]={sum0,sum1};
    q9_u129 p=q9_product129(sum,a1);
    p.top^=(sum2&(a1[0]&1U));
    const q9_u129 q=q9_product129(c2,b1);
    q9_u129 rr=q9_product129(c1,a2); /* a2[0..3] is c modulo 2^128. */
    rr.top^=(uint32_t)(c1[0]&1ULL);   /* c has one implicit bit at 128. */
    const q9_u129 kk={k[0],k[1],(uint32_t)(k[2]&1ULL)};
    const q9_u129 z1=q9_sub129(q9_sub129(kk,p),q);
    const q9_u129 z2=q9_sub129(rr,p);
    q9_abs129(r1,s1,z1);q9_abs129(r2,s2,z2);
}

/* QSB_GLV_NO_KRED (kill switch): q9_glv_split's only caller is the subset filter chain, whose scalar
 * is the SHA-256d message digest z, uniform on [0,2^256). z >= n needs z[3] == 2^64-1 and
 * z[2] >= 2^64-2, probability (2^256-n)/2^256 < 2^-127 per candidate; on such a z the split may
 * leave the walker's 128-bit fields and that one candidate recovers a wrong key, which the host's
 * exact publication gate (OpenSSL re-derivation of every hit) rejects: a lost candidate, never a
 * false hit. 1 drops the conditional subtraction; 0 keeps it. */
#ifndef QSB_GLV_NO_KRED
#define QSB_GLV_NO_KRED 1
#endif
#if QSB_GLV_NO_KRED != 0 && QSB_GLV_NO_KRED != 1
#error "QSB_GLV_NO_KRED must be 0 or 1"
#endif
__device__ __forceinline__ void q9_glv_split(const uint64_t input[4],uint64_t r1[2],uint64_t r2[2],unsigned *s1,unsigned *s2){
    uint64_t k[4]={input[0],input[1],input[2],input[3]};
#if !QSB_GLV_NO_KRED
    const uint64_t n[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
    if(k[3]==n[3]&&(k[2]>n[2]||(k[2]==n[2]&&(k[1]>n[1]||(k[1]==n[1]&&k[0]>=n[0])))))q9_sub4(k,k,n);
#endif
    const uint64_t g1[4]={0xE893209A45DBB031ULL,0x3DAA8A1471E8CA7FULL,0xE86C90E49284EB15ULL,0x3086D221A7D46BCDULL};
    const uint64_t g2[4]={0x1571B4AE8AC47F71ULL,0x221208AC9DF506C6ULL,0x6F547FA90ABFE4C4ULL,0xE4437ED6010E8828ULL};
    const uint32_t a1[4]={0x9284eb15,0xe86c90e4,0xa7d46bcd,0x3086d221};
    const uint32_t a2[5]={0x9d44cfd8,0x57c1108d,0xa8e2f3f6,0x14ca50f7,1};
    const uint32_t b1[4]={0x0abfe4c3,0x6f547fa9,0x010e8828,0xe4437ed6};
    uint64_t c1[2],c2[2];q9_coeff_g1(c1,k,g1);q9_coeff_g2(c2,k,g2);
#if QSB_GLV_RESIDUAL129
    q9_glv_residual129(k,c1,c2,a1,a2,b1,r1,r2,s1,s2);
#elif QSB_GLV_RESIDUAL3
    q9_glv_residual3(k,c1,c2,a1,a2,b1,r1,r2,s1,s2);
#else
    q9_glv_residual_reference(k,c1,c2,a1,a2,b1,r1,r2,s1,s2);
#endif
}

/* QSB_GLV_ZDEC (analog of our pinning tree's QSB_GLV_GLUE bit 2, signed-residual decode; walker side in
 * tests/gpu_epochs/tree.cu, qsb_s3_code_z): the split hands the chain the signed residuals instead of
 * (|r|, sign), and the walker decodes the same table codes from them. With s = bit 128 of the 129-bit
 * residual z and M = -s, the magnitude is |r| = (z - s) ^ M on 128 bits. Every centred segment field of
 * (z - s) is the magnitude's field complemented when s = 1, which negates its odd digit, and the digit's
 * negation cancels the component sign in the code's bit 31: those fields decode with no sign at all. The
 * two uncentred segments need their own handling: the top field (bits 100..127, centre C = 170559770)
 * of a negative component is pre-shifted by D = 2^28 - C here (it then decodes as the negated digit with
 * the same centre; the complemented field is >= D + 1 because |r| < 0xa2a8918c... gives a top field
 * <= C - 2, so the shift never borrows), and segment 0 (unsigned, bias only) takes its digit centre 2^19
 * instead of 0 when s = 1 (the chain passes 1 - (M & 2^19) as that term's centre). So
 *   v = z - s * (1 + D * 2^100) mod 2^128,
 * one 128-bit add of a masked constant in place of q9_abs129's mask, four XORs and 128-bit increment; the
 * word-4 parity fix-ups of the three residual products (sum2 and c's implicit bit 128) fold into the
 * product's word-4 add (as pinning's q9_glv_split_z), and the subtractions keep no masks (only bit 0 of
 * every top word is read). Same scalar lattice, coefficients and residuals as q9_glv_split; every table
 * code the chain gathers is identical (qsb_s3_selfcheck replays the z walker against q9_bigtbl_code /
 * q11_bigtbl_code). 0 = q9_glv_split and the (|r|, sign) walker byte for byte. */
#ifndef QSB_GLV_ZDEC
#if QSB_LOCAL_SM86
#define QSB_GLV_ZDEC 0   /* dev-only sm_86 rig: needs the GLV11/Q_MIX walk */
#else
#define QSB_GLV_ZDEC 1
#endif
#endif
#if QSB_GLV_ZDEC != 0 && QSB_GLV_ZDEC != 1
#error "QSB_GLV_ZDEC must be 0 or 1"
#endif
/* Top word of -(1 + D*2^100) mod 2^128, D = 2^28 - 170559770 = 97875686: 0xFFFFFFFF - 16*D. */
#define QSB_ZDEC_TOPWORD 0xA2A8919Fu
#if QSB_DECODE_CUT && !(QSB_GLV_ZDEC && QSB_GLV_EO)
#error "QSB_DECODE_CUT is written for the QSB_GLV_EO product and the QSB_GLV_ZDEC walker value"
#endif
#if QSB_GLV_ZDEC
#if !QSB_GLV_RESIDUAL129 || !QSB_GLV_EO || !QSB_GLV_NO_KRED
#error "QSB_GLV_ZDEC is written for the RESIDUAL129 / GLV_EO / GLV_NO_KRED split"
#endif
/* Only bit 0 of every top word is meaningful on this path (bit 128); no masks. */
__device__ __forceinline__ q9_u129 q9_sub129_z(q9_u129 a,q9_u129 b) {
    q9_u129 r;
    asm("{sub.cc.u64 %0,%3,%6;subc.cc.u64 %1,%4,%7;subc.u32 %2,%5,%8;}"
        : "=l"(r.lo),"=l"(r.hi),"=r"(r.top)
        : "l"(a.lo),"l"(a.hi),"r"(a.top),"l"(b.lo),"l"(b.hi),"r"(b.top));
    return r;
}
/* v = z - s*(1 + D*2^100) mod 2^128 and m = -s, s = bit 0 of z.top (bit 128 of z). */
__device__ __forceinline__ void q9_zwalk_value(uint64_t v[2],uint32_t *m32,q9_u129 z) {
    uint32_t m;
#if QSB_DECODE_CUT & 1
    /* QSB_DECODE_CUT bit 1: z - s on word 0 only and -s*16D on word 3 only (0xA2A891A0 = 2^32 - 16D =
     * QSB_ZDEC_TOPWORD + 1); the borrow of z - 1 into word 1 (s = 1, word 0 of z = 0: 2^-33 per residual)
     * is dropped, a lost candidate behind the exact host gate. */
    static_assert(QSB_ZDEC_TOPWORD + 1u == 0xA2A891A0u, "QSB_DECODE_CUT bit 1: word-3 constant is 2^32 - 16D");
    asm("{\n\t.reg .u32 t,a,b,c,e;\n\t"
        "bfe.s32 %2,%5,0,1;\n\t"
        "and.b32 t,%2,0xA2A891A0;\n\t"
        "mov.b64 {a,b},%3;\n\t"
        "mov.b64 {c,e},%4;\n\t"
        "add.u32 a,a,%2;\n\t"
        "add.u32 e,e,t;\n\t"
        "mov.b64 %0,{a,b};\n\t"
        "mov.b64 %1,{c,e};\n\t}"
        : "=l"(v[0]),"=l"(v[1]),"=&r"(m) : "l"(z.lo),"l"(z.hi),"r"(z.top));
#else
    asm("{\n\t.reg .u32 t;\n\t.reg .u64 M,H;\n\t"
        "bfe.s32 %2,%5,0,1;\n\t"
        "and.b32 t,%2,0xA2A8919F;\n\t"
        "mov.b64 M,{%2,%2};\n\t"
        "mov.b64 H,{%2,t};\n\t"
        "add.cc.u64 %0,%3,M;\n\t"
        "addc.u64 %1,%4,H;\n\t}"
        : "=l"(v[0]),"=l"(v[1]),"=&r"(m) : "l"(z.lo),"l"(z.hi),"r"(z.top));
#endif
    *m32=m;
}
/* v1/m1: component r1 (P), v2/m2: component r2 (Q), in q9_glv_split's order. */
__device__ __forceinline__ void q9_glv_split_z(const uint64_t input[4],uint64_t v1[2],uint64_t v2[2],
                                               uint32_t *m1,uint32_t *m2){
    const uint64_t k[4]={input[0],input[1],input[2],input[3]};
    const uint64_t g1[4]={0xE893209A45DBB031ULL,0x3DAA8A1471E8CA7FULL,0xE86C90E49284EB15ULL,0x3086D221A7D46BCDULL};
    const uint64_t g2[4]={0x1571B4AE8AC47F71ULL,0x221208AC9DF506C6ULL,0x6F547FA90ABFE4C4ULL,0xE4437ED6010E8828ULL};
    const uint32_t a1[4]={0x9284eb15,0xe86c90e4,0xa7d46bcd,0x3086d221};
    const uint32_t a2[5]={0x9d44cfd8,0x57c1108d,0xa8e2f3f6,0x14ca50f7,1};
    const uint32_t b1[4]={0x0abfe4c3,0x6f547fa9,0x010e8828,0xe4437ed6};
    uint64_t c1[2],c2[2];q9_coeff_g1(c1,k,g1);q9_coeff_g2(c2,k,g2);
    /* sum = c1 + c2 (129 bits) and P = sum*a1's word-4 parity addend: sum2 (a1[0] odd adds sum2 at bit
     * 128) plus sum words 1 and 2 (a1[3], a1[2] odd, a1[1] even), folded into the carry capture. */
    uint64_t sum0,sum1;uint32_t xp;
    asm("{\n\t.reg .u32 a,b,c,d,t;\n\t"
        "add.cc.u64 %0,%3,%5;\n\t"
        "addc.cc.u64 %1,%4,%6;\n\t"
        "mov.b64 {a,b},%0;\n\t"
        "mov.b64 {c,d},%1;\n\t"
        "add.u32 t,b,c;\n\t"
        "addc.u32 %2,t,0;\n\t}"
        : "=l"(sum0),"=l"(sum1),"=r"(xp)
        : "l"(c1[0]),"l"(c1[1]),"l"(c2[0]),"l"(c2[1]));
    const uint64_t sum[2]={sum0,sum1};
    static_assert((0x9284eb15u&1u)==1u && (0xa7d46bcdu&1u)==1u && (0x3086d221u&1u)==1u && (0xe86c90e4u&1u)==0u,
                  "parity addend of P assumes a1 words 0, 2, 3 odd and word 1 even");
    static_assert((0x0abfe4c3u&1u)==1u && (0x6f547fa9u&1u)==1u && (0x010e8828u&1u)==0u && (0xe4437ed6u&1u)==0u,
                  "parity addend of Q assumes b1 words 0, 1 odd and words 2, 3 even");
    static_assert((0x57c1108du&1u)==1u && (0xa8e2f3f6u&1u)==0u && (0x14ca50f7u&1u)==1u,
                  "parity addend of R assumes a2 words 1, 3 odd and word 2 even");
    const q9_u129 p=q9_product129_rx(sum,a1,xp);
    const q9_u129 q=q9_product129_rx(c2,b1,(uint32_t)(c2[1]>>32));                     /* b1[1] odd: x3 */
    const q9_u129 rr=q9_product129_rx(c1,a2,(uint32_t)c1[0]+(uint32_t)(c1[0]>>32)
                                       +(uint32_t)(c1[1]>>32));  /* c's bit 128 adds c1; a2[3], a2[1] odd: x1, x3 */
    const q9_u129 kk={k[0],k[1],(uint32_t)k[2]};
    const q9_u129 z1=q9_sub129_z(q9_sub129_z(kk,p),q);
    const q9_u129 z2=q9_sub129_z(rr,p);
    q9_zwalk_value(v1,m1,z1);q9_zwalk_value(v2,m2,z2);
}
#endif
