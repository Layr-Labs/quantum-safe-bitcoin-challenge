// SPDX-License-Identifier: GPL-3.0-or-later
// Isolated weak-field prototype; no production integration or GPU validation.
// Field p=2^256-(2^32+977); inputs/outputs in [0,2^256), little-endian limbs.
// Four uint64_t caller limbs correspond to eight physical32-bit field words.
// All inputs are captured before output writes: exact and partial limb overlap safe.
#pragma once
#include <stdint.h>
#if defined(__CUDACC__)
#define QSB_WEAK_INLINE __host__ __device__ __forceinline__
#else
#define QSB_WEAK_INLINE inline
#endif

QSB_WEAK_INLINE void qsb_weak_add(uint64_t *r, const uint64_t *a, const uint64_t *b) {
#if defined(__CUDA_ARCH__)
    uint64_t o0,o1,o2,o3;
    asm(
        "{\n"
        ".reg .u32 a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7,f,lo,hi,tmp;\n"
        "mov.b64 {a0,a1}, %4;\n"
        "mov.b64 {a2,a3}, %5;\n"
        "mov.b64 {a4,a5}, %6;\n"
        "mov.b64 {a6,a7}, %7;\n"
        "mov.b64 {b0,b1}, %8;\n"
        "mov.b64 {b2,b3}, %9;\n"
        "mov.b64 {b4,b5}, %10;\n"
        "mov.b64 {b6,b7}, %11;\n"
        "add.cc.u32 a0, a0, b0;\n"
        "addc.cc.u32 a1, a1, b1;\n"
        "addc.cc.u32 a2, a2, b2;\n"
        "addc.cc.u32 a3, a3, b3;\n"
        "addc.cc.u32 a4, a4, b4;\n"
        "addc.cc.u32 a5, a5, b5;\n"
        "addc.cc.u32 a6, a6, b6;\n"
        "addc.cc.u32 a7, a7, b7;\n"
        "addc.u32 f, 0, 0;\n"
        "mul.lo.u32 lo, f, 977;\n"
        "mov.u32 hi, f;\n"
        "add.cc.u32 a0, a0, lo;\n"
        "addc.cc.u32 a1, a1, hi;\n"
        "addc.cc.u32 a2, a2, 0;\n"
        "addc.cc.u32 a3, a3, 0;\n"
        "addc.cc.u32 a4, a4, 0;\n"
        "addc.cc.u32 a5, a5, 0;\n"
        "addc.cc.u32 a6, a6, 0;\n"
        "addc.cc.u32 a7, a7, 0;\n"
        "addc.u32 f, 0, 0;\n"
        "mul.lo.u32 lo, f, 977;\n"
        "mov.u32 hi, f;\n"
        "add.cc.u32 a0, a0, lo;\n"
        "addc.u32 a1, a1, hi;\n"
        "mov.b64 %0, {a0,a1};\n"
        "mov.b64 %1, {a2,a3};\n"
        "mov.b64 %2, {a4,a5};\n"
        "mov.b64 %3, {a6,a7};\n"
        "}\n"
        : "=l"(o0), "=l"(o1), "=l"(o2), "=l"(o3)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
          "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]));
    r[0]=o0;r[1]=o1;r[2]=o2;r[3]=o3;
#else
    uint32_t x[8],y[8];
    for(unsigned i=0;i<8;i++){x[i]=(uint32_t)(a[i/2]>>(32*(i&1)));y[i]=(uint32_t)(b[i/2]>>(32*(i&1)));}
    uint64_t carry=0;
    for(unsigned i=0;i<8;i++){uint64_t t=(uint64_t)x[i]+y[i]+carry;x[i]=(uint32_t)t;carry=t>>32;}
    uint64_t fold=carry;carry=0;
    for(unsigned i=0;i<8;i++){uint64_t t=(uint64_t)x[i]+(i==0?977*fold:i==1?fold:0)+carry;x[i]=(uint32_t)t;carry=t>>32;}
    // If this carried, the remaining value is <=C-2, so +C fits in64 bits.
    uint64_t low=((uint64_t)x[1]<<32)|x[0];low+=carry*UINT64_C(0x1000003d1);
    x[0]=(uint32_t)low;x[1]=(uint32_t)(low>>32);
    for(unsigned i=0;i<4;i++)r[i]=(uint64_t)x[2*i]|((uint64_t)x[2*i+1]<<32);
#endif
}

QSB_WEAK_INLINE void qsb_weak_sub(uint64_t *r, const uint64_t *a, const uint64_t *b) {
#if defined(__CUDA_ARCH__)
    uint64_t o0,o1,o2,o3;
    asm(
        "{\n"
        ".reg .u32 a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7,f,lo,hi,tmp;\n"
        "mov.b64 {a0,a1}, %4;\n"
        "mov.b64 {a2,a3}, %5;\n"
        "mov.b64 {a4,a5}, %6;\n"
        "mov.b64 {a6,a7}, %7;\n"
        "mov.b64 {b0,b1}, %8;\n"
        "mov.b64 {b2,b3}, %9;\n"
        "mov.b64 {b4,b5}, %10;\n"
        "mov.b64 {b6,b7}, %11;\n"
        "sub.cc.u32 a0, a0, b0;\n"
        "subc.cc.u32 a1, a1, b1;\n"
        "subc.cc.u32 a2, a2, b2;\n"
        "subc.cc.u32 a3, a3, b3;\n"
        "subc.cc.u32 a4, a4, b4;\n"
        "subc.cc.u32 a5, a5, b5;\n"
        "subc.cc.u32 a6, a6, b6;\n"
        "subc.cc.u32 a7, a7, b7;\n"
        "subc.u32 f, 0, 0;\n"
        "and.b32 lo, f, 977;\n"
        "and.b32 hi, f, 1;\n"
        "sub.cc.u32 a0, a0, lo;\n"
        "subc.cc.u32 a1, a1, hi;\n"
        "subc.cc.u32 a2, a2, 0;\n"
        "subc.cc.u32 a3, a3, 0;\n"
        "subc.cc.u32 a4, a4, 0;\n"
        "subc.cc.u32 a5, a5, 0;\n"
        "subc.cc.u32 a6, a6, 0;\n"
        "subc.cc.u32 a7, a7, 0;\n"
        "subc.u32 f, 0, 0;\n"
        "and.b32 lo, f, 977;\n"
        "and.b32 hi, f, 1;\n"
        "sub.cc.u32 a0, a0, lo;\n"
        "subc.u32 a1, a1, hi;\n"
        "mov.b64 %0, {a0,a1};\n"
        "mov.b64 %1, {a2,a3};\n"
        "mov.b64 %2, {a4,a5};\n"
        "mov.b64 %3, {a6,a7};\n"
        "}\n"
        : "=l"(o0), "=l"(o1), "=l"(o2), "=l"(o3)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
          "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]));
    r[0]=o0;r[1]=o1;r[2]=o2;r[3]=o3;
#else
    uint32_t x[8],y[8];
    for(unsigned i=0;i<8;i++){x[i]=(uint32_t)(a[i/2]>>(32*(i&1)));y[i]=(uint32_t)(b[i/2]>>(32*(i&1)));}
    uint64_t borrow=0;
    for(unsigned i=0;i<8;i++){uint64_t rhs=(uint64_t)y[i]+borrow;uint64_t lhs=x[i];x[i]=(uint32_t)(lhs-rhs);borrow=lhs<rhs;}
    uint64_t fold=borrow;borrow=0;
    for(unsigned i=0;i<8;i++){uint64_t rhs=(i==0?977*fold:i==1?fold:0)+borrow;uint64_t lhs=x[i];x[i]=(uint32_t)(lhs-rhs);borrow=lhs<rhs;}
    // If this borrowed, low64>=2^64-C+1>C: -C cannot borrow into word2.
    uint64_t low=((uint64_t)x[1]<<32)|x[0];low-=borrow*UINT64_C(0x1000003d1);
    x[0]=(uint32_t)low;x[1]=(uint32_t)(low>>32);
    for(unsigned i=0;i<4;i++)r[i]=(uint64_t)x[2*i]|((uint64_t)x[2*i+1]<<32);
#endif
}

QSB_WEAK_INLINE void qsb_weak_normalize(uint64_t *r, const uint64_t *a) {
#if defined(__CUDA_ARCH__)
    uint64_t o0,o1,o2,o3;
    asm(
        "{\n"
        ".reg .u32 a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7,f,lo,hi,tmp;\n"
        "mov.b64 {a0,a1}, %4;\n"
        "mov.b64 {a2,a3}, %5;\n"
        "mov.b64 {a4,a5}, %6;\n"
        "mov.b64 {a6,a7}, %7;\n"
        "add.cc.u32 b0, a0, 977;\n"
        "addc.cc.u32 b1, a1, 1;\n"
        "addc.cc.u32 b2, a2, 0;\n"
        "addc.cc.u32 b3, a3, 0;\n"
        "addc.cc.u32 b4, a4, 0;\n"
        "addc.cc.u32 b5, a5, 0;\n"
        "addc.cc.u32 b6, a6, 0;\n"
        "addc.cc.u32 b7, a7, 0;\n"
        "addc.u32 f, 0, 0;\n"
        "sub.u32 f, 0, f;\n"
        "xor.b32 tmp, a0, b0;\n"
        "and.b32 tmp, tmp, f;\n"
        "xor.b32 a0, a0, tmp;\n"
        "xor.b32 tmp, a1, b1;\n"
        "and.b32 tmp, tmp, f;\n"
        "xor.b32 a1, a1, tmp;\n"
        "xor.b32 tmp, a2, b2;\n"
        "and.b32 tmp, tmp, f;\n"
        "xor.b32 a2, a2, tmp;\n"
        "xor.b32 tmp, a3, b3;\n"
        "and.b32 tmp, tmp, f;\n"
        "xor.b32 a3, a3, tmp;\n"
        "xor.b32 tmp, a4, b4;\n"
        "and.b32 tmp, tmp, f;\n"
        "xor.b32 a4, a4, tmp;\n"
        "xor.b32 tmp, a5, b5;\n"
        "and.b32 tmp, tmp, f;\n"
        "xor.b32 a5, a5, tmp;\n"
        "xor.b32 tmp, a6, b6;\n"
        "and.b32 tmp, tmp, f;\n"
        "xor.b32 a6, a6, tmp;\n"
        "xor.b32 tmp, a7, b7;\n"
        "and.b32 tmp, tmp, f;\n"
        "xor.b32 a7, a7, tmp;\n"
        "mov.b64 %0, {a0,a1};\n"
        "mov.b64 %1, {a2,a3};\n"
        "mov.b64 %2, {a4,a5};\n"
        "mov.b64 %3, {a6,a7};\n"
        "}\n"
        : "=l"(o0), "=l"(o1), "=l"(o2), "=l"(o3)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]));
    r[0]=o0;r[1]=o1;r[2]=o2;r[3]=o3;
#else
    uint32_t x[8];
    for(unsigned i=0;i<8;i++){x[i]=(uint32_t)(a[i/2]>>(32*(i&1)));}
    uint32_t t[8];uint64_t carry=0;
    for(unsigned i=0;i<8;i++){uint64_t v=(uint64_t)x[i]+(i==0?977u:i==1?1u:0u)+carry;t[i]=(uint32_t)v;carry=v>>32;}
    if(carry)for(unsigned i=0;i<8;i++)x[i]=t[i];
    for(unsigned i=0;i<4;i++)r[i]=(uint64_t)x[2*i]|((uint64_t)x[2*i+1]<<32);
#endif
}

QSB_WEAK_INLINE void qsb_weak_normalize(uint64_t *r) {qsb_weak_normalize(r,r);}
QSB_WEAK_INLINE bool qsb_weak_is_zero(const uint64_t *a) {
    const uint64_t a0=a[0],a1=a[1],a2=a[2],a3=a[3];
    return ((a0|a1|a2|a3)==0) ||
        (a0==UINT64_C(0xfffffffefffffc2f) && a1==UINT64_MAX && a2==UINT64_MAX && a3==UINT64_MAX);
}
#undef QSB_WEAK_INLINE
