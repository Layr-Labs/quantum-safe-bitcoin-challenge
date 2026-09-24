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
 * rounded reciprocal error gives |r_i| <=
 * 0xa2a8918ca85bafe22016d0b917e4dd77. At shift 104 the largest top
 * field is 10659985, an odd number. Thus d_top=2*f-10659985 is odd,
 * nonzero and in [-10659985,10659985]. No residual truncation is used.
 * Physical order 0,1,2,5,3,4 keeps the three 16 MiB segments first.
 * These portable helpers are also compiled verbatim by check_bigtable.py. */
__host__ __device__ __forceinline__ unsigned q9_bigtbl_entries(int c) {
    return c<3 ? 262144u : (c<5 ? 8388608u : 5329993u);
}
__host__ __device__ __forceinline__ unsigned q9_bigtbl_offset(int c) {
    return c==0?0u:c==1?262144u:c==2?524288u:c==3?6116425u:
           c==4?14505033u:786432u;
}
__host__ __device__ __forceinline__ unsigned q9_bigtbl_shift(int c) {
    return c==0?0u:c==1?18u:c==2?37u:c==3?56u:c==4?80u:104u;
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
        const int32_t d=(int32_t)(2u*f)-10659985;
        neg_digit=(uint32_t)d>>31;
        const uint32_t ad=((uint32_t)d^(0u-neg_digit))+neg_digit;
        idx=(ad-1u)>>1;
    } else {
        const unsigned bits=c<3?19u:24u;
        f&=(1u<<bits)-1u;
        neg_digit=1u-(f>>(bits-1u));
        idx=(f^(0u-neg_digit))&((1u<<(bits-1u))-1u);
    }
    return (q9_bigtbl_offset(c)+idx)|((neg_digit^sign)<<31);
}
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
template<int WHICH>
__device__ __noinline__ ulonglong2 q9_coeff_fallback(uint64_t k0,uint64_t k1,
                                                      uint64_t k2,uint64_t k3){
    const uint64_t k[4]={k0,k1,k2,k3};
    const uint64_t g1[4]={0xE893209A45DBB031ULL,0x3DAA8A1471E8CA7FULL,
                          0xE86C90E49284EB15ULL,0x3086D221A7D46BCDULL};
    const uint64_t g2[4]={0x1571B4AE8AC47F71ULL,0x221208AC9DF506C6ULL,
                          0x6F547FA90ABFE4C4ULL,0xE4437ED6010E8828ULL};
    uint64_t out[2];q9_coeff_reference(out,k,WHICH==1?g1:g2);
    ulonglong2 r;r.x=out[0];r.y=out[1];return r;
}

__device__ __forceinline__ void q9_high15_add(uint64_t *acc,uint32_t *overflow,uint64_t product){
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

template<int WHICH,uint32_t FALLBACK_WORD>
__device__ __forceinline__ void q9_coeff_high15(uint64_t out[2],const uint64_t k[4],const uint64_t g[4]){
    const uint32_t a3=(uint32_t)(k[1]>>32);
    const uint32_t a4=(uint32_t)k[2],a5=(uint32_t)(k[2]>>32);
    const uint32_t a6=(uint32_t)k[3],a7=(uint32_t)(k[3]>>32);
    const uint32_t b3=(uint32_t)(g[1]>>32),b4=(uint32_t)g[2];
    const uint32_t b5=(uint32_t)(g[2]>>32),b6=(uint32_t)g[3],b7=(uint32_t)(g[3]>>32);
    uint64_t carry=0,acc;uint32_t overflow,w10,w11,w12,w13,w14,w15;

    /* Diagonal 10: (3,7)..(7,3). A 64-bit sum is insufficient for five
     * products, so overflow counts its lost 2^64 units explicitly. */
    q9_high15_begin(&acc,&overflow,carry,(uint64_t)a3*b7,(uint64_t)a4*b6);
    q9_high15_add(&acc,&overflow,(uint64_t)a5*b5);
    q9_high15_add(&acc,&overflow,(uint64_t)a6*b4);
    q9_high15_add(&acc,&overflow,(uint64_t)a7*b3);
    w10=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

    q9_high15_begin(&acc,&overflow,carry,(uint64_t)a4*b7,(uint64_t)a5*b6);
    q9_high15_add(&acc,&overflow,(uint64_t)a6*b5);
    q9_high15_add(&acc,&overflow,(uint64_t)a7*b4);
    w11=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

    q9_high15_begin(&acc,&overflow,carry,(uint64_t)a5*b7,(uint64_t)a6*b6);
    q9_high15_add(&acc,&overflow,(uint64_t)a7*b5);
    w12=(uint32_t)acc;carry=(acc>>32)|((uint64_t)overflow<<32);

    q9_high15_begin(&acc,&overflow,carry,(uint64_t)a6*b7,(uint64_t)a7*b6);
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
        ulonglong2 r=q9_coeff_fallback<WHICH>(k[0],k[1],k[2],k[3]);
        out[0]=r.x;out[1]=r.y;
    }
}

__device__ __forceinline__ void q9_coeff_g1(uint64_t out[2],const uint64_t k[4],const uint64_t g[4]){
#if QSB_GLV_HIGH15
    q9_coeff_high15<1,0x7ffffffcU>(out,k,g);
#else
    q9_coeff_reference(out,k,g);
#endif
}
__device__ __forceinline__ void q9_coeff_g2(uint64_t out[2],const uint64_t k[4],const uint64_t g[4]){
#if QSB_GLV_HIGH15
    q9_coeff_high15<2,0x7ffffffdU>(out,k,g);
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

__device__ __forceinline__ q9_u129 q9_product129(const uint64_t x[2],const uint32_t d[4]) {
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
    q9_u129 r={(uint64_t)w0|((uint64_t)w1<<32),
                 (uint64_t)w2|((uint64_t)w3<<32),top&1U};
    return r;
}

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

__device__ __forceinline__ void q9_glv_split(const uint64_t input[4],uint64_t r1[2],uint64_t r2[2],unsigned *s1,unsigned *s2){
    const uint64_t n[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
    uint64_t k[4]={input[0],input[1],input[2],input[3]};
    if(k[3]==n[3]&&(k[2]>n[2]||(k[2]==n[2]&&(k[1]>n[1]||(k[1]==n[1]&&k[0]>=n[0])))))q9_sub4(k,k,n);
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
