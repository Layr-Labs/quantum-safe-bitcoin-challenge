#include <stdint.h>
#define __device__
#define __forceinline__ inline
__device__ __forceinline__ void _ModMultCore(uint64_t *r, const uint64_t *a, const uint64_t *b)
{
#ifdef __CUDA_ARCH__
    uint64_t r0,r1,r2,r3; uint32_t carry;
    asm( "{\n\t.reg .u32 a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7;\n\t.reg .u64 e0,e1,e2,e3,e4,e5,e6,e7,o0,o1,o2,o3,o4,o5,o6,t,lc;\n\t.reg .u32 cy,o15;\n\t.reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n\t.reg .u32 y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14;\n\tmov.b64 {a0,a1}, %5;\n\tmov.b64 {a2,a3}, %6;\n\tmov.b64 {a4,a5}, %7;\n\tmov.b64 {a6,a7}, %8;\n\tmov.b64 {b0,b1}, %9;\n\tmov.b64 {b2,b3}, %10;\n\tmov.b64 {b4,b5}, %11;\n\tmov.b64 {b6,b7}, %12;\n\tmul.wide.u32 e0, a0, b0; mul.wide.u32 e1, a0, b2; mul.wide.u32 e2, a0, b4; mul.wide.u32 e3, a0, b6;\n\tmul.wide.u32 t, a1, b1; add.cc.u64 e1, e1, t;\n\tmul.wide.u32 t, a1, b3; addc.cc.u64 e2, e2, t;\n\tmul.wide.u32 t, a1, b5; addc.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, a1, b7; addc.u64 e4, t, 0;\n\tmul.wide.u32 t, a2, b0; add.cc.u64 e1, e1, t;\n\tmul.wide.u32 t, a2, b2; addc.cc.u64 e2, e2, t;\n\tmul.wide.u32 t, a2, b4; addc.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, a2, b6; addc.cc.u64 e4, e4, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, a3, b1; add.cc.u64 e2, e2, t;\n\tmul.wide.u32 t, a3, b3; addc.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, a3, b5; addc.cc.u64 e4, e4, t;\n\tmul.wide.u32 t, a3, b7; addc.u64 e5, t, lc;\n\tmul.wide.u32 t, a4, b0; add.cc.u64 e2, e2, t;\n\tmul.wide.u32 t, a4, b2; addc.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, a4, b4; addc.cc.u64 e4, e4, t;\n\tmul.wide.u32 t, a4, b6; addc.cc.u64 e5, e5, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, a5, b1; add.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, a5, b3; addc.cc.u64 e4, e4, t;\n\tmul.wide.u32 t, a5, b5; addc.cc.u64 e5, e5, t;\n\tmul.wide.u32 t, a5, b7; addc.u64 e6, t, lc;\n\tmul.wide.u32 t, a6, b0; add.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, a6, b2; addc.cc.u64 e4, e4, t;\n\tmul.wide.u32 t, a6, b4; addc.cc.u64 e5, e5, t;\n\tmul.wide.u32 t, a6, b6; addc.cc.u64 e6, e6, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, a7, b1; add.cc.u64 e4, e4, t;\n\tmul.wide.u32 t, a7, b3; addc.cc.u64 e5, e5, t;\n\tmul.wide.u32 t, a7, b5; addc.cc.u64 e6, e6, t;\n\tmul.wide.u32 t, a7, b7; addc.u64 e7, t, lc;\n\tmul.wide.u32 o0, a0, b1; mul.wide.u32 o1, a0, b3; mul.wide.u32 o2, a0, b5; mul.wide.u32 o3, a0, b7;\n\tmul.wide.u32 t, a1, b0; add.cc.u64 o0, o0, t;\n\tmul.wide.u32 t, a1, b2; addc.cc.u64 o1, o1, t;\n\tmul.wide.u32 t, a1, b4; addc.cc.u64 o2, o2, t;\n\tmul.wide.u32 t, a1, b6; addc.cc.u64 o3, o3, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, a2, b1; add.cc.u64 o1, o1, t;\n\tmul.wide.u32 t, a2, b3; addc.cc.u64 o2, o2, t;\n\tmul.wide.u32 t, a2, b5; addc.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, a2, b7; addc.u64 o4, t, lc;\n\tmul.wide.u32 t, a3, b0; add.cc.u64 o1, o1, t;\n\tmul.wide.u32 t, a3, b2; addc.cc.u64 o2, o2, t;\n\tmul.wide.u32 t, a3, b4; addc.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, a3, b6; addc.cc.u64 o4, o4, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, a4, b1; add.cc.u64 o2, o2, t;\n\tmul.wide.u32 t, a4, b3; addc.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, a4, b5; addc.cc.u64 o4, o4, t;\n\tmul.wide.u32 t, a4, b7; addc.u64 o5, t, lc;\n\tmul.wide.u32 t, a5, b0; add.cc.u64 o2, o2, t;\n\tmul.wide.u32 t, a5, b2; addc.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, a5, b4; addc.cc.u64 o4, o4, t;\n\tmul.wide.u32 t, a5, b6; addc.cc.u64 o5, o5, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, a6, b1; add.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, a6, b3; addc.cc.u64 o4, o4, t;\n\tmul.wide.u32 t, a6, b5; addc.cc.u64 o5, o5, t;\n\tmul.wide.u32 t, a6, b7; addc.u64 o6, t, lc;\n\tmul.wide.u32 t, a7, b0; add.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, a7, b2; addc.cc.u64 o4, o4, t;\n\tmul.wide.u32 t, a7, b4; addc.cc.u64 o5, o5, t;\n\tmul.wide.u32 t, a7, b6; addc.cc.u64 o6, o6, t;\n\taddc.u32 o15, 0, 0;\n\tmov.b64 {x0,x1}, e0;\n\tmov.b64 {x2,x3}, e1;\n\tmov.b64 {x4,x5}, e2;\n\tmov.b64 {x6,x7}, e3;\n\tmov.b64 {x8,x9}, e4;\n\tmov.b64 {x10,x11}, e5;\n\tmov.b64 {x12,x13}, e6;\n\tmov.b64 {x14,x15}, e7;\n\tmov.b64 {y1,y2}, o0;\n\tmov.b64 {y3,y4}, o1;\n\tmov.b64 {y5,y6}, o2;\n\tmov.b64 {y7,y8}, o3;\n\tmov.b64 {y9,y10}, o4;\n\tmov.b64 {y11,y12}, o5;\n\tmov.b64 {y13,y14}, o6;\n\tadd.cc.u32 x1, x1, y1;\n\taddc.cc.u32 x2, x2, y2;\n\taddc.cc.u32 x3, x3, y3;\n\taddc.cc.u32 x4, x4, y4;\n\taddc.cc.u32 x5, x5, y5;\n\taddc.cc.u32 x6, x6, y6;\n\taddc.cc.u32 x7, x7, y7;\n\taddc.cc.u32 x8, x8, y8;\n\taddc.cc.u32 x9, x9, y9;\n\taddc.cc.u32 x10, x10, y10;\n\taddc.cc.u32 x11, x11, y11;\n\taddc.cc.u32 x12, x12, y12;\n\taddc.cc.u32 x13, x13, y13;\n\taddc.cc.u32 x14, x14, y14;\n\taddc.u32 x15, x15, o15;\n\t.reg .u64 r0,r1,r2,r3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;\n\t.reg .u32 f8,g8,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,w0,w1,w2,w3,w4,w5,w6,w7,m0,m1,m2;\n\tmov.b64 r0, {x0,x1}; mov.b64 r1, {x2,x3}; mov.b64 r2, {x4,x5}; mov.b64 r3, {x6,x7};\n\tmov.b64 h0, {x8,x9}; mov.b64 h1, {x10,x11}; mov.b64 h2, {x12,x13}; mov.b64 h3, {x14,x15};\n\tmul.wide.u32 t, x8, 977;  add.cc.u64  f0, r0, t;\n\tmul.wide.u32 t, x10, 977; addc.cc.u64 f1, r1, t;\n\tmul.wide.u32 t, x12, 977; addc.cc.u64 f2, r2, t;\n\tmul.wide.u32 t, x14, 977; addc.cc.u64 f3, r3, t;\n\taddc.u32 f8, 0, 0;\n\tmul.wide.u32 t, x9, 977;  add.cc.u64  g0, h0, t;\n\tmul.wide.u32 t, x11, 977; addc.cc.u64 g1, h1, t;\n\tmul.wide.u32 t, x13, 977; addc.cc.u64 g2, h2, t;\n\tmul.wide.u32 t, x15, 977; addc.cc.u64 g3, h3, t;\n\taddc.u32 g8, 0, 0;\n\tmov.b64 {z0,z1}, f0;\n\tmov.b64 {z2,z3}, f1;\n\tmov.b64 {z4,z5}, f2;\n\tmov.b64 {z6,z7}, f3;\n\tmov.b64 {w0,w1}, g0;\n\tmov.b64 {w2,w3}, g1;\n\tmov.b64 {w4,w5}, g2;\n\tmov.b64 {w6,w7}, g3;\n\tadd.cc.u32  z1, z1, w0;\n\taddc.cc.u32 z2, z2, w1;\n\taddc.cc.u32 z3, z3, w2;\n\taddc.cc.u32 z4, z4, w3;\n\taddc.cc.u32 z5, z5, w4;\n\taddc.cc.u32 z6, z6, w5;\n\taddc.cc.u32 z7, z7, w6;\n\taddc.cc.u32 z8, f8, w7;\n\taddc.u32    z9, g8, 0;\n\tmul.wide.u32 t, z8, 977; mov.b64 {m0,m1}, t;\n\tmad.lo.u32 m1, z9, 977, m1;\n\tadd.cc.u32 m1, m1, z8;\n\taddc.u32 m2, z9, 0;\n\tadd.cc.u32 z0, z0, m0; addc.cc.u32 z1, z1, m1; addc.cc.u32 z2, z2, m2;\n\taddc.cc.u32 z3, z3, 0;\n\taddc.cc.u32 z4, z4, 0;\n\taddc.cc.u32 z5, z5, 0;\n\taddc.cc.u32 z6, z6, 0;\n\taddc.cc.u32 z7, z7, 0;\n\taddc.u32 %4, 0, 0;\n\tmov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.b64 %2, {z4,z5}; mov.b64 %3, {z6,z7};\n\t}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3),"=r"(carry)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]) );
    if ((uint32_t)((uint32_t)(r3 >> 32) + 1u) <= 1u) {
        QsbFieldWords v = qsb_field_cold_correction({r0,r1,r2,r3},carry);
        r0=v.a; r1=v.b; r2=v.c; r3=v.d;
    }
    r[0]=r0; r[1]=r1; r[2]=r2; r[3]=r3;
#else
#define QSB_MW(x,y) ((uint64_t)(uint32_t)(x)*(uint32_t)(y))

    uint32_t A[8], B[8];
    for (int i=0;i<4;i++){ A[2*i]=(uint32_t)a[i]; A[2*i+1]=(uint32_t)(a[i]>>32);
                           B[2*i]=(uint32_t)b[i]; B[2*i+1]=(uint32_t)(b[i]>>32); }
    uint64_t e0,e1,e2,e3,e4,e5,e6,e7, o0,o1,o2,o3,o4,o5,o6, cy,lc; __uint128_t s;
    /* even chain */
    e0=QSB_MW(A[0],B[0]); e1=QSB_MW(A[0],B[2]); e2=QSB_MW(A[0],B[4]); e3=QSB_MW(A[0],B[6]);
    s=(__uint128_t)e1+QSB_MW(A[1],B[1]); e1=(uint64_t)s;
    s=(s>>64)+e2+QSB_MW(A[1],B[3]); e2=(uint64_t)s;
    s=(s>>64)+e3+QSB_MW(A[1],B[5]); e3=(uint64_t)s;
    e4=(uint64_t)(s>>64)+QSB_MW(A[1],B[7]);
    s=(__uint128_t)e1+QSB_MW(A[2],B[0]); e1=(uint64_t)s;
    s=(s>>64)+e2+QSB_MW(A[2],B[2]); e2=(uint64_t)s;
    s=(s>>64)+e3+QSB_MW(A[2],B[4]); e3=(uint64_t)s;
    s=(s>>64)+e4+QSB_MW(A[2],B[6]); e4=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)e2+QSB_MW(A[3],B[1]); e2=(uint64_t)s;
    s=(s>>64)+e3+QSB_MW(A[3],B[3]); e3=(uint64_t)s;
    s=(s>>64)+e4+QSB_MW(A[3],B[5]); e4=(uint64_t)s;
    e5=(uint64_t)(s>>64)+QSB_MW(A[3],B[7])+lc;
    s=(__uint128_t)e2+QSB_MW(A[4],B[0]); e2=(uint64_t)s;
    s=(s>>64)+e3+QSB_MW(A[4],B[2]); e3=(uint64_t)s;
    s=(s>>64)+e4+QSB_MW(A[4],B[4]); e4=(uint64_t)s;
    s=(s>>64)+e5+QSB_MW(A[4],B[6]); e5=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)e3+QSB_MW(A[5],B[1]); e3=(uint64_t)s;
    s=(s>>64)+e4+QSB_MW(A[5],B[3]); e4=(uint64_t)s;
    s=(s>>64)+e5+QSB_MW(A[5],B[5]); e5=(uint64_t)s;
    e6=(uint64_t)(s>>64)+QSB_MW(A[5],B[7])+lc;
    s=(__uint128_t)e3+QSB_MW(A[6],B[0]); e3=(uint64_t)s;
    s=(s>>64)+e4+QSB_MW(A[6],B[2]); e4=(uint64_t)s;
    s=(s>>64)+e5+QSB_MW(A[6],B[4]); e5=(uint64_t)s;
    s=(s>>64)+e6+QSB_MW(A[6],B[6]); e6=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)e4+QSB_MW(A[7],B[1]); e4=(uint64_t)s;
    s=(s>>64)+e5+QSB_MW(A[7],B[3]); e5=(uint64_t)s;
    s=(s>>64)+e6+QSB_MW(A[7],B[5]); e6=(uint64_t)s;
    e7=(uint64_t)(s>>64)+QSB_MW(A[7],B[7])+lc;
    /* odd chain */
    o0=QSB_MW(A[0],B[1]); o1=QSB_MW(A[0],B[3]); o2=QSB_MW(A[0],B[5]); o3=QSB_MW(A[0],B[7]);
    s=(__uint128_t)o0+QSB_MW(A[1],B[0]); o0=(uint64_t)s;
    s=(s>>64)+o1+QSB_MW(A[1],B[2]); o1=(uint64_t)s;
    s=(s>>64)+o2+QSB_MW(A[1],B[4]); o2=(uint64_t)s;
    s=(s>>64)+o3+QSB_MW(A[1],B[6]); o3=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)o1+QSB_MW(A[2],B[1]); o1=(uint64_t)s;
    s=(s>>64)+o2+QSB_MW(A[2],B[3]); o2=(uint64_t)s;
    s=(s>>64)+o3+QSB_MW(A[2],B[5]); o3=(uint64_t)s;
    o4=(uint64_t)(s>>64)+QSB_MW(A[2],B[7])+lc;
    s=(__uint128_t)o1+QSB_MW(A[3],B[0]); o1=(uint64_t)s;
    s=(s>>64)+o2+QSB_MW(A[3],B[2]); o2=(uint64_t)s;
    s=(s>>64)+o3+QSB_MW(A[3],B[4]); o3=(uint64_t)s;
    s=(s>>64)+o4+QSB_MW(A[3],B[6]); o4=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)o2+QSB_MW(A[4],B[1]); o2=(uint64_t)s;
    s=(s>>64)+o3+QSB_MW(A[4],B[3]); o3=(uint64_t)s;
    s=(s>>64)+o4+QSB_MW(A[4],B[5]); o4=(uint64_t)s;
    o5=(uint64_t)(s>>64)+QSB_MW(A[4],B[7])+lc;
    s=(__uint128_t)o2+QSB_MW(A[5],B[0]); o2=(uint64_t)s;
    s=(s>>64)+o3+QSB_MW(A[5],B[2]); o3=(uint64_t)s;
    s=(s>>64)+o4+QSB_MW(A[5],B[4]); o4=(uint64_t)s;
    s=(s>>64)+o5+QSB_MW(A[5],B[6]); o5=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)o3+QSB_MW(A[6],B[1]); o3=(uint64_t)s;
    s=(s>>64)+o4+QSB_MW(A[6],B[3]); o4=(uint64_t)s;
    s=(s>>64)+o5+QSB_MW(A[6],B[5]); o5=(uint64_t)s;
    o6=(uint64_t)(s>>64)+QSB_MW(A[6],B[7])+lc;
    s=(__uint128_t)o3+QSB_MW(A[7],B[0]); o3=(uint64_t)s;
    s=(s>>64)+o4+QSB_MW(A[7],B[2]); o4=(uint64_t)s;
    s=(s>>64)+o5+QSB_MW(A[7],B[4]); o5=(uint64_t)s;
    s=(s>>64)+o6+QSB_MW(A[7],B[6]); o6=(uint64_t)s;
    uint32_t o15=(uint32_t)(s>>64);
    /* unpack + merge to 16 u32 limbs */
    uint32_t x[16];
    x[0]=(uint32_t)e0; x[1]=(uint32_t)(e0>>32); x[2]=(uint32_t)e1; x[3]=(uint32_t)(e1>>32);
    x[4]=(uint32_t)e2; x[5]=(uint32_t)(e2>>32); x[6]=(uint32_t)e3; x[7]=(uint32_t)(e3>>32);
    x[8]=(uint32_t)e4; x[9]=(uint32_t)(e4>>32); x[10]=(uint32_t)e5; x[11]=(uint32_t)(e5>>32);
    x[12]=(uint32_t)e6; x[13]=(uint32_t)(e6>>32); x[14]=(uint32_t)e7; x[15]=(uint32_t)(e7>>32);
    uint32_t y[15];
    y[1]=(uint32_t)o0; y[2]=(uint32_t)(o0>>32); y[3]=(uint32_t)o1; y[4]=(uint32_t)(o1>>32);
    y[5]=(uint32_t)o2; y[6]=(uint32_t)(o2>>32); y[7]=(uint32_t)o3; y[8]=(uint32_t)(o3>>32);
    y[9]=(uint32_t)o4; y[10]=(uint32_t)(o4>>32); y[11]=(uint32_t)o5; y[12]=(uint32_t)(o5>>32);
    y[13]=(uint32_t)o6; y[14]=(uint32_t)(o6>>32);
    { uint64_t c=0; for (int k=1;k<=14;k++){ uint64_t t=(uint64_t)x[k]+y[k]+c; x[k]=(uint32_t)t; c=t>>32; }
      x[15]=(uint32_t)((uint64_t)x[15]+o15+c); }
    /* secp256k1 double-fold reduction (identical to mm32) */
    uint64_t r0=x[0]|((uint64_t)x[1]<<32), r1=x[2]|((uint64_t)x[3]<<32),
             r2=x[4]|((uint64_t)x[5]<<32), r3=x[6]|((uint64_t)x[7]<<32);
    uint64_t h0=x[8]|((uint64_t)x[9]<<32), h1=x[10]|((uint64_t)x[11]<<32),
             h2=x[12]|((uint64_t)x[13]<<32), h3=x[14]|((uint64_t)x[15]<<32);
    (void)r0;(void)r1;(void)r2;(void)r3;(void)h0;(void)h1;(void)h2;(void)h3;
    uint64_t f0,f1,f2,f3; uint32_t f8;
    s=(__uint128_t)r0+QSB_MW(x[8],977);  f0=(uint64_t)s;
    s=(s>>64)+r1+QSB_MW(x[10],977); f1=(uint64_t)s;
    s=(s>>64)+r2+QSB_MW(x[12],977); f2=(uint64_t)s;
    s=(s>>64)+r3+QSB_MW(x[14],977); f3=(uint64_t)s;
    f8=(uint32_t)(s>>64);
    uint64_t g0,g1,g2,g3; uint32_t g8;
    s=(__uint128_t)h0+QSB_MW(x[9],977);  g0=(uint64_t)s;
    s=(s>>64)+h1+QSB_MW(x[11],977); g1=(uint64_t)s;
    s=(s>>64)+h2+QSB_MW(x[13],977); g2=(uint64_t)s;
    s=(s>>64)+h3+QSB_MW(x[15],977); g3=(uint64_t)s;
    g8=(uint32_t)(s>>64);
    uint32_t z[10], w[8];
    z[0]=(uint32_t)f0; z[1]=(uint32_t)(f0>>32); z[2]=(uint32_t)f1; z[3]=(uint32_t)(f1>>32);
    z[4]=(uint32_t)f2; z[5]=(uint32_t)(f2>>32); z[6]=(uint32_t)f3; z[7]=(uint32_t)(f3>>32);
    w[0]=(uint32_t)g0; w[1]=(uint32_t)(g0>>32); w[2]=(uint32_t)g1; w[3]=(uint32_t)(g1>>32);
    w[4]=(uint32_t)g2; w[5]=(uint32_t)(g2>>32); w[6]=(uint32_t)g3; w[7]=(uint32_t)(g3>>32);
    { uint64_t c=0,t;
      for (int k=0;k<7;k++){ t=(uint64_t)z[k+1]+w[k]+c; z[k+1]=(uint32_t)t; c=t>>32; }
      t=(uint64_t)f8+w[7]+c; z[8]=(uint32_t)t; c=t>>32;
      z[9]=(uint32_t)((uint64_t)g8+c); }
    /* second fold: c33 = z8 + 2^32 z9; m = c33*977 + c33<<32 */
    uint64_t tt=QSB_MW(z[8],977);
    uint32_t m0=(uint32_t)tt, m1=(uint32_t)(tt>>32), m2;
    m1=(uint32_t)(m1 + (uint32_t)((uint64_t)z[9]*977));
    { uint64_t t=(uint64_t)m1+z[8]; m1=(uint32_t)t; uint64_t c=t>>32; m2=(uint32_t)((uint64_t)z[9]+c); }
    { uint64_t c=0,t;
      t=(uint64_t)z[0]+m0+c; z[0]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[1]+m1+c; z[1]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[2]+m2+c; z[2]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[3]+c; z[3]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[4]+c; z[4]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[5]+c; z[5]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[6]+c; z[6]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[7]+c; z[7]=(uint32_t)t;
      // If carry is set, low < C*C for C=2^32+977; three limbs suffice.
      c=t>>32;
      t=(uint64_t)z[0]+c*977; z[0]=(uint32_t)t;
      t=(uint64_t)z[1]+c+(t>>32); z[1]=(uint32_t)t;
      z[2]+=(uint32_t)(t>>32); }
    r[0]=z[0]|((uint64_t)z[1]<<32); r[1]=z[2]|((uint64_t)z[3]<<32);
    r[2]=z[4]|((uint64_t)z[5]<<32); r[3]=z[6]|((uint64_t)z[7]<<32);
#undef QSB_MW
#endif
#ifndef __CUDA_ARCH__
    // Raw reduction is below 2^256. At most one subtraction yields [0,p).
    // Bounded carry and normalization follow promoted subset 65fb673d.
    if ((r[1] & r[2] & r[3]) == UINT64_MAX &&
        r[0] >= 0xFFFFFFFEFFFFFC2FULL) {
        r[0] -= 0xFFFFFFFEFFFFFC2FULL;
        r[1] = r[2] = r[3] = 0;
    }
#endif
}
__device__ __forceinline__ void _ModSqr(uint64_t r[4], const uint64_t a[4]) {
#ifdef __CUDA_ARCH__
    uint64_t r0,r1,r2,r3; uint32_t carry;
    asm("{\n\t"
        ".reg .u32 a0,a1,a2,a3,a4,a5,a6,a7;\n\t"
        ".reg .u64 e2,e4,e6,e8,e10,e12,o1,o3,o5,o7,o9,o11,o13,t;\n\t"
        ".reg .u32 ecy,ocy,e14,o15;\n\t"
        ".reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n\t"
        ".reg .u32 y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14;\n\t"
        ".reg .u64 d0,d1,d2,d3,d4,d5,d6,d7;\n\t"
        "mov.b64 {a0,a1}, %5;\n\t"
        "mov.b64 {a2,a3}, %6;\n\t"
        "mov.b64 {a4,a5}, %7;\n\t"
        "mov.b64 {a6,a7}, %8;\n\t"

        /* E rows, shortest first. Carries run into the next 64-bit column. */
        "mul.wide.u32 e6, a2, a4;\n\t"
        "mul.wide.u32 e8, a3, a5;\n\t"
        "mul.wide.u32 e4, a1, a3;\n\t"
        "mul.wide.u32 t, a1, a5; add.cc.u64 e6, e6, t;\n\t"
        "mul.wide.u32 t, a2, a6; addc.cc.u64 e8, e8, t;\n\t"
        "mul.wide.u32 t, a4, a6; addc.cc.u64 e10, t, 0;\n\t"
        "addc.u32 ecy, 0, 0; mov.b64 e12, {ecy,0};\n\t"
        "mul.wide.u32 e2, a0, a2;\n\t"
        "mul.wide.u32 t, a0, a4; add.cc.u64 e4, e4, t;\n\t"
        "mul.wide.u32 t, a0, a6; addc.cc.u64 e6, e6, t;\n\t"
        "mul.wide.u32 t, a1, a7; addc.cc.u64 e8, e8, t;\n\t"
        "mul.wide.u32 t, a3, a7; addc.cc.u64 e10, e10, t;\n\t"
        "mul.wide.u32 t, a5, a7; addc.cc.u64 e12, e12, t;\n\t"
        "addc.u32 e14, 0, 0;\n\t"

        /* O rows, shortest first; O7's four products occupy four rows. */
        "mul.wide.u32 o7, a3, a4;\n\t"
        "mul.wide.u32 o5, a2, a3;\n\t"
        "mul.wide.u32 t, a2, a5; add.cc.u64 o7, o7, t;\n\t"
        "mul.wide.u32 t, a4, a5; addc.cc.u64 o9, t, 0;\n\t"
        "addc.u32 ocy, 0, 0; mov.b64 o11, {ocy,0};\n\t"
        "mul.wide.u32 o3, a1, a2;\n\t"
        "mul.wide.u32 t, a1, a4; add.cc.u64 o5, o5, t;\n\t"
        "mul.wide.u32 t, a1, a6; addc.cc.u64 o7, o7, t;\n\t"
        "mul.wide.u32 t, a3, a6; addc.cc.u64 o9, o9, t;\n\t"
        "mul.wide.u32 t, a5, a6; addc.cc.u64 o11, o11, t;\n\t"
        "addc.u32 ocy, 0, 0; mov.b64 o13, {ocy,0};\n\t"
        "mul.wide.u32 o1, a0, a1;\n\t"
        "mul.wide.u32 t, a0, a3; add.cc.u64 o3, o3, t;\n\t"
        "mul.wide.u32 t, a0, a5; addc.cc.u64 o5, o5, t;\n\t"
        "mul.wide.u32 t, a0, a7; addc.cc.u64 o7, o7, t;\n\t"
        "mul.wide.u32 t, a2, a7; addc.cc.u64 o9, o9, t;\n\t"
        "mul.wide.u32 t, a4, a7; addc.cc.u64 o11, o11, t;\n\t"
        "mul.wide.u32 t, a6, a7; addc.cc.u64 o13, o13, t;\n\t"
        "addc.u32 o15, 0, 0;\n\t"

        /* Merge the staggered 64-bit E/O words into X[0..15]. */
        "mov.u32 x0, 0;\n\t"
        "mov.b64 {x2,x3}, e2; mov.b64 {x4,x5}, e4; mov.b64 {x6,x7}, e6;\n\t"
        "mov.b64 {x8,x9}, e8; mov.b64 {x10,x11}, e10; mov.b64 {x12,x13}, e12;\n\t"
        "mov.u32 x14, e14; mov.u32 x15, 0;\n\t"
        "mov.b64 {x1,y2}, o1; mov.b64 {y3,y4}, o3; mov.b64 {y5,y6}, o5;\n\t"
        "mov.b64 {y7,y8}, o7; mov.b64 {y9,y10}, o9; mov.b64 {y11,y12}, o11; mov.b64 {y13,y14}, o13;\n\t"
        "add.cc.u32 x2, x2, y2; addc.cc.u32 x3, x3, y3;\n\t"
        "addc.cc.u32 x4, x4, y4; addc.cc.u32 x5, x5, y5; addc.cc.u32 x6, x6, y6;\n\t"
        "addc.cc.u32 x7, x7, y7; addc.cc.u32 x8, x8, y8; addc.cc.u32 x9, x9, y9;\n\t"
        "addc.cc.u32 x10, x10, y10; addc.cc.u32 x11, x11, y11; addc.cc.u32 x12, x12, y12;\n\t"
        "addc.cc.u32 x13, x13, y13; addc.cc.u32 x14, x14, y14; addc.u32 x15, x15, o15;\n\t"

        /* X = 2*cross. Descending funnel shifts retain the old lower limb. */
        "shf.l.wrap.b32 x15, x14, x15, 1; shf.l.wrap.b32 x14, x13, x14, 1;\n\t"
        "shf.l.wrap.b32 x13, x12, x13, 1; shf.l.wrap.b32 x12, x11, x12, 1;\n\t"
        "shf.l.wrap.b32 x11, x10, x11, 1; shf.l.wrap.b32 x10, x9, x10, 1;\n\t"
        "shf.l.wrap.b32 x9, x8, x9, 1; shf.l.wrap.b32 x8, x7, x8, 1;\n\t"
        "shf.l.wrap.b32 x7, x6, x7, 1; shf.l.wrap.b32 x6, x5, x6, 1;\n\t"
        "shf.l.wrap.b32 x5, x4, x5, 1; shf.l.wrap.b32 x4, x3, x4, 1;\n\t"
        "shf.l.wrap.b32 x3, x2, x3, 1; shf.l.wrap.b32 x2, x1, x2, 1;\n\t"
        "shf.l.wrap.b32 x1, x0, x1, 1;\n\t"
        /* Add A[i]^2 at each 64-bit word 2*i. */
        "mov.b64 d0, {x0,x1}; mov.b64 d1, {x2,x3}; mov.b64 d2, {x4,x5}; mov.b64 d3, {x6,x7};\n\t"
        "mov.b64 d4, {x8,x9}; mov.b64 d5, {x10,x11}; mov.b64 d6, {x12,x13}; mov.b64 d7, {x14,x15};\n\t"
        "mul.wide.u32 t, a0, a0; add.cc.u64 d0, d0, t;\n\t"
        "mul.wide.u32 t, a1, a1; addc.cc.u64 d1, d1, t;\n\t"
        "mul.wide.u32 t, a2, a2; addc.cc.u64 d2, d2, t;\n\t"
        "mul.wide.u32 t, a3, a3; addc.cc.u64 d3, d3, t;\n\t"
        "mul.wide.u32 t, a4, a4; addc.cc.u64 d4, d4, t;\n\t"
        "mul.wide.u32 t, a5, a5; addc.cc.u64 d5, d5, t;\n\t"
        "mul.wide.u32 t, a6, a6; addc.cc.u64 d6, d6, t;\n\t"
        "mul.wide.u32 t, a7, a7; addc.u64 d7, d7, t;\n\t"
        "mov.b64 {x0,x1}, d0; mov.b64 {x2,x3}, d1; mov.b64 {x4,x5}, d2; mov.b64 {x6,x7}, d3;\n\t"
        "mov.b64 {x8,x9}, d4; mov.b64 {x10,x11}, d5; mov.b64 {x12,x13}, d6; mov.b64 {x14,x15}, d7;\n\t"

        /* The multiply core's unchanged secp256k1 double fold. */
        ".reg .u64 fr0,fr1,fr2,fr3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;\n\t"
        ".reg .u32 f8,g8,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,w0,w1,w2,w3,w4,w5,w6,w7,m0,m1,m2;\n\t"
        "mov.b64 fr0, {x0,x1}; mov.b64 fr1, {x2,x3}; mov.b64 fr2, {x4,x5}; mov.b64 fr3, {x6,x7};\n\t"
        "mov.b64 h0, {x8,x9}; mov.b64 h1, {x10,x11}; mov.b64 h2, {x12,x13}; mov.b64 h3, {x14,x15};\n\t"
        "mul.wide.u32 t, x8, 977;  add.cc.u64  f0, fr0, t;\n\t"
        "mul.wide.u32 t, x10, 977; addc.cc.u64 f1, fr1, t;\n\t"
        "mul.wide.u32 t, x12, 977; addc.cc.u64 f2, fr2, t;\n\t"
        "mul.wide.u32 t, x14, 977; addc.cc.u64 f3, fr3, t;\n\t"
        "addc.u32 f8, 0, 0;\n\t"
        "mul.wide.u32 t, x9, 977;  add.cc.u64  g0, h0, t;\n\t"
        "mul.wide.u32 t, x11, 977; addc.cc.u64 g1, h1, t;\n\t"
        "mul.wide.u32 t, x13, 977; addc.cc.u64 g2, h2, t;\n\t"
        "mul.wide.u32 t, x15, 977; addc.cc.u64 g3, h3, t;\n\t"
        "addc.u32 g8, 0, 0;\n\t"
        "mov.b64 {z0,z1}, f0; mov.b64 {z2,z3}, f1; mov.b64 {z4,z5}, f2; mov.b64 {z6,z7}, f3;\n\t"
        "mov.b64 {w0,w1}, g0; mov.b64 {w2,w3}, g1; mov.b64 {w4,w5}, g2; mov.b64 {w6,w7}, g3;\n\t"
        "add.cc.u32 z1, z1, w0; addc.cc.u32 z2, z2, w1; addc.cc.u32 z3, z3, w2;\n\t"
        "addc.cc.u32 z4, z4, w3; addc.cc.u32 z5, z5, w4; addc.cc.u32 z6, z6, w5;\n\t"
        "addc.cc.u32 z7, z7, w6; addc.cc.u32 z8, f8, w7; addc.u32 z9, g8, 0;\n\t"
        "mul.wide.u32 t, z8, 977; mov.b64 {m0,m1}, t;\n\t"
        "mad.lo.u32 m1, z9, 977, m1;\n\t"
        "add.cc.u32 m1, m1, z8; addc.u32 m2, z9, 0;\n\t"
        "add.cc.u32 z0, z0, m0; addc.cc.u32 z1, z1, m1; addc.cc.u32 z2, z2, m2;\n\t"
        "addc.cc.u32 z3, z3, 0; addc.cc.u32 z4, z4, 0; addc.cc.u32 z5, z5, 0;\n\t"
        "addc.cc.u32 z6, z6, 0; addc.cc.u32 z7, z7, 0;\n\taddc.u32 %4, 0, 0;\n\t"
        "mov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.b64 %2, {z4,z5}; mov.b64 %3, {z6,z7};\n\t"
        "}\n\t"
        : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3),"=r"(carry)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]));
    if ((uint32_t)((uint32_t)(r3 >> 32) + 1u) <= 1u) {
        QsbFieldWords v = qsb_field_cold_correction({r0,r1,r2,r3},carry);
        r0=v.a; r1=v.b; r2=v.c; r3=v.d;
    }
    r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
#else
    uint32_t A[8];
    for (int i=0;i<4;i++){ A[2*i]=(uint32_t)a[i]; A[2*i+1]=(uint32_t)(a[i]>>32); }
    #define PP(i,j) ((uint64_t)A[i]*A[j])
    __uint128_t t;
    /* even column words E_c (cols c,c+1), carry chain E_c -> E_{c+2} */
    uint64_t E2,E4,E6,E8,E10,E12,e14;
    t=(__uint128_t)PP(0,2);                          E2=(uint64_t)t;
    t=(t>>64)+PP(0,4)+PP(1,3);                       E4=(uint64_t)t;
    t=(t>>64)+PP(0,6)+PP(1,5)+PP(2,4);               E6=(uint64_t)t;
    t=(t>>64)+PP(1,7)+PP(2,6)+PP(3,5);               E8=(uint64_t)t;
    t=(t>>64)+PP(3,7)+PP(4,6);                       E10=(uint64_t)t;
    t=(t>>64)+PP(5,7);                               E12=(uint64_t)t;
    e14=(uint64_t)(t>>64);
    /* odd column words O_c */
    uint64_t O1,O3,O5,O7,O9,O11,O13,o15;
    t=(__uint128_t)PP(0,1);                          O1=(uint64_t)t;
    t=(t>>64)+PP(0,3)+PP(1,2);                       O3=(uint64_t)t;
    t=(t>>64)+PP(0,5)+PP(1,4)+PP(2,3);               O5=(uint64_t)t;
    t=(t>>64)+PP(0,7)+PP(1,6)+PP(2,5)+PP(3,4);       O7=(uint64_t)t;
    t=(t>>64)+PP(2,7)+PP(3,6)+PP(4,5);               O9=(uint64_t)t;
    t=(t>>64)+PP(4,7)+PP(5,6);                       O11=(uint64_t)t;
    t=(t>>64)+PP(6,7);                               O13=(uint64_t)t;
    o15=(uint64_t)(t>>64);
    #undef PP
    /* merge E (even cols) + O (odd cols) -> cross-sum X[0..15] (u32 limbs) */
    uint32_t X[16]; uint64_t cc;
    X[0]=0;
    cc=(uint64_t)(uint32_t)O1;                                    X[1]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E2 + (uint32_t)(O1>>32);              X[2]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E2>>32) + (uint32_t)O3;              X[3]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E4 + (uint32_t)(O3>>32);             X[4]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E4>>32) + (uint32_t)O5;             X[5]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E6 + (uint32_t)(O5>>32);             X[6]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E6>>32) + (uint32_t)O7;             X[7]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E8 + (uint32_t)(O7>>32);             X[8]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E8>>32) + (uint32_t)O9;             X[9]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E10 + (uint32_t)(O9>>32);            X[10]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E10>>32) + (uint32_t)O11;           X[11]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E12 + (uint32_t)(O11>>32);           X[12]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E12>>32) + (uint32_t)O13;           X[13]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)e14 + (uint32_t)(O13>>32);           X[14]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)o15;                                 X[15]=(uint32_t)cc;
    /* double: 2*cross */
    uint64_t d=0;
    for (int k=0;k<16;k++){ uint64_t v=((uint64_t)X[k]<<1)|d; X[k]=(uint32_t)v; d=v>>32; }
    /* add the 8 diagonal squares A[i]^2 at columns 2i */
    uint64_t carry=0;
    for (int i=0;i<8;i++){
        __uint128_t s=(__uint128_t)X[2*i] + ((uint64_t)X[2*i+1]<<32) + (uint64_t)A[i]*A[i] + carry;
        X[2*i]=(uint32_t)s; X[2*i+1]=(uint32_t)(s>>32); carry=(uint64_t)(s>>64);
    }
    /* secp256k1 double-fold (identical to the multiply's 6.2 reduction) */
    #define MW(x,y) ((uint64_t)(uint32_t)(x)*(uint32_t)(y))
    uint64_t r0=X[0]|((uint64_t)X[1]<<32), r1=X[2]|((uint64_t)X[3]<<32),
             r2=X[4]|((uint64_t)X[5]<<32), r3=X[6]|((uint64_t)X[7]<<32);
    uint64_t h0=X[8]|((uint64_t)X[9]<<32), h1=X[10]|((uint64_t)X[11]<<32),
             h2=X[12]|((uint64_t)X[13]<<32), h3=X[14]|((uint64_t)X[15]<<32);
    __uint128_t s;
    uint64_t f0,f1,f2,f3; uint32_t f8;
    s=(__uint128_t)r0+MW(X[8],977);  f0=(uint64_t)s;
    s=(s>>64)+r1+MW(X[10],977); f1=(uint64_t)s;
    s=(s>>64)+r2+MW(X[12],977); f2=(uint64_t)s;
    s=(s>>64)+r3+MW(X[14],977); f3=(uint64_t)s;
    f8=(uint32_t)(s>>64);
    uint64_t g0,g1,g2,g3; uint32_t g8;
    s=(__uint128_t)h0+MW(X[9],977);  g0=(uint64_t)s;
    s=(s>>64)+h1+MW(X[11],977); g1=(uint64_t)s;
    s=(s>>64)+h2+MW(X[13],977); g2=(uint64_t)s;
    s=(s>>64)+h3+MW(X[15],977); g3=(uint64_t)s;
    g8=(uint32_t)(s>>64);
    uint32_t z[10], w[8];
    z[0]=(uint32_t)f0;z[1]=(uint32_t)(f0>>32);z[2]=(uint32_t)f1;z[3]=(uint32_t)(f1>>32);
    z[4]=(uint32_t)f2;z[5]=(uint32_t)(f2>>32);z[6]=(uint32_t)f3;z[7]=(uint32_t)(f3>>32);
    w[0]=(uint32_t)g0;w[1]=(uint32_t)(g0>>32);w[2]=(uint32_t)g1;w[3]=(uint32_t)(g1>>32);
    w[4]=(uint32_t)g2;w[5]=(uint32_t)(g2>>32);w[6]=(uint32_t)g3;w[7]=(uint32_t)(g3>>32);
    { uint64_t c=0,tt;
      for(int k=0;k<7;k++){ tt=(uint64_t)z[k+1]+w[k]+c; z[k+1]=(uint32_t)tt; c=tt>>32; }
      tt=(uint64_t)f8+w[7]+c; z[8]=(uint32_t)tt; c=tt>>32; z[9]=(uint32_t)((uint64_t)g8+c); }
    uint64_t tt2=MW(z[8],977);
    uint32_t m0=(uint32_t)tt2,m1=(uint32_t)(tt2>>32),m2;
    m1=(uint32_t)(m1+(uint32_t)((uint64_t)z[9]*977));
    { uint64_t tv=(uint64_t)m1+z[8]; m1=(uint32_t)tv; uint64_t c=tv>>32; m2=(uint32_t)((uint64_t)z[9]+c); }
    { uint64_t c=0,tv;
      tv=(uint64_t)z[0]+m0+c; z[0]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[1]+m1+c; z[1]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[2]+m2+c; z[2]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[3]+c; z[3]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[4]+c; z[4]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[5]+c; z[5]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[6]+c; z[6]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[7]+c; z[7]=(uint32_t)tv;
      // If carry is set, low < C*C for C=2^32+977; three limbs suffice.
      c=tv>>32;
      tv=(uint64_t)z[0]+c*977; z[0]=(uint32_t)tv;
      tv=(uint64_t)z[1]+c+(tv>>32); z[1]=(uint32_t)tv;
      z[2]+=(uint32_t)(tv>>32); }
    r[0]=z[0]|((uint64_t)z[1]<<32); r[1]=z[2]|((uint64_t)z[3]<<32);
    r[2]=z[4]|((uint64_t)z[5]<<32); r[3]=z[6]|((uint64_t)z[7]<<32);
    #undef MW
    (void)h0;(void)h1;(void)h2;(void)h3;(void)r0;(void)r1;(void)r2;(void)r3;(void)carry;(void)d;
#endif
#ifndef __CUDA_ARCH__
    // Raw reduction is below 2^256. At most one subtraction yields [0,p).
    // Bounded carry and normalization follow promoted subset 65fb673d.
    if ((r[1] & r[2] & r[3]) == UINT64_MAX &&
        r[0] >= 0xFFFFFFFEFFFFFC2FULL) {
        r[0] -= 0xFFFFFFFEFFFFFC2FULL;
        r[1] = r[2] = r[3] = 0;
    }
#endif
}
extern "C" void mul(uint64_t*r,const uint64_t*a,const uint64_t*b){_ModMultCore(r,a,b);}
extern "C" void sq(uint64_t*r,const uint64_t*a){_ModSqr(r,a);}
