#pragma once
#include <stdint.h>
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
__device__ __forceinline__ void q9_coeff(uint64_t out[2],const uint64_t k[4],const uint64_t g[4]){
    uint64_t p[8];q9_wide(p,k,g);
    __uint128_t t=(__uint128_t)p[6]+(p[5]>>63);out[0]=(uint64_t)t;out[1]=p[7]+(uint64_t)(t>>64);
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
__device__ __forceinline__ void q9_glv_split(const uint64_t input[4],uint64_t r1[2],uint64_t r2[2],unsigned *s1,unsigned *s2){
    const uint64_t n[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
    uint64_t k[4]={input[0],input[1],input[2],input[3]};
    if(k[3]==n[3]&&(k[2]>n[2]||(k[2]==n[2]&&(k[1]>n[1]||(k[1]==n[1]&&k[0]>=n[0])))))q9_sub4(k,k,n);
    const uint64_t g1[4]={0xE893209A45DBB031ULL,0x3DAA8A1471E8CA7FULL,0xE86C90E49284EB15ULL,0x3086D221A7D46BCDULL};
    const uint64_t g2[4]={0x1571B4AE8AC47F71ULL,0x221208AC9DF506C6ULL,0x6F547FA90ABFE4C4ULL,0xE4437ED6010E8828ULL};
    const uint32_t a1[4]={0x9284eb15,0xe86c90e4,0xa7d46bcd,0x3086d221};
    const uint32_t a2[5]={0x9d44cfd8,0x57c1108d,0xa8e2f3f6,0x14ca50f7,1};
    const uint32_t b1[4]={0x0abfe4c3,0x6f547fa9,0x010e8828,0xe4437ed6};
    uint64_t c1[2],c2[2],p[4],q[4],z[4];q9_coeff(c1,k,g1);q9_coeff(c2,k,g2);
    q9_small_product<4>(p,c1,a1);q9_small_product<5>(q,c2,a2);
    q9_sub4(z,k,p);q9_sub4(z,z,q);q9_abs128(r1,s1,z);
    q9_small_product<4>(p,c1,b1);q9_small_product<4>(q,c2,a1);
    q9_sub4(z,p,q);q9_abs128(r2,s2,z);
}
