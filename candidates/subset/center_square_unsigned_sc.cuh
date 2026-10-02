#pragma once
/* Experimental unsigned counterpart of center_square_sc.cuh. GPL-3 inherited
 * square core. Host BN prebias is 3p-c^2; high is 1 or 2. Filter-only path. */
__device__ __forceinline__ void qsb_filter_sqr_offset(uint64_t r[4], const uint64_t a[4], const uint64_t b[4], const uint64_t c[4], uint32_t high, uint32_t &bad) {
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
        "mul.wide.u32 t, a4, a6; addc.u64 e10, t, 0;\n\t"
        "mul.wide.u32 e2, a0, a2;\n\t"
        "mul.wide.u32 t, a0, a4; add.cc.u64 e4, e4, t;\n\t"
        "mul.wide.u32 t, a0, a6; addc.cc.u64 e6, e6, t;\n\t"
        "mul.wide.u32 t, a1, a7; addc.cc.u64 e8, e8, t;\n\t"
        "mul.wide.u32 t, a3, a7; addc.cc.u64 e10, e10, t;\n\t"
        "mul.wide.u32 t, a5, a7; addc.u64 e12, t, 0;\n\t"

        /* O rows, shortest first; O7's four products occupy four rows. */
        "mul.wide.u32 o7, a3, a4;\n\t"
        "mul.wide.u32 o5, a2, a3;\n\t"
        "mul.wide.u32 t, a2, a5; add.cc.u64 o7, o7, t;\n\t"
        "mul.wide.u32 t, a4, a5; addc.u64 o9, t, 0;\n\t"
        "mul.wide.u32 o3, a1, a2;\n\t"
        "mul.wide.u32 t, a1, a4; add.cc.u64 o5, o5, t;\n\t"
        "mul.wide.u32 t, a1, a6; addc.cc.u64 o7, o7, t;\n\t"
        "mul.wide.u32 t, a3, a6; addc.cc.u64 o9, o9, t;\n\t"
        "mul.wide.u32 t, a5, a6; addc.u64 o11, t, 0;\n\t"
        "mul.wide.u32 o1, a0, a1;\n\t"
        "mul.wide.u32 t, a0, a3; add.cc.u64 o3, o3, t;\n\t"
        "mul.wide.u32 t, a0, a5; addc.cc.u64 o5, o5, t;\n\t"
        "mul.wide.u32 t, a0, a7; addc.cc.u64 o7, o7, t;\n\t"
        "mul.wide.u32 t, a2, a7; addc.cc.u64 o9, o9, t;\n\t"
        "mul.wide.u32 t, a4, a7; addc.cc.u64 o11, o11, t;\n\t"
        "mul.wide.u32 t, a6, a7; addc.u64 o13, t, 0;\n\t"

        /* Merge the staggered 64-bit E/O words into X[0..15]. */
        "mov.u32 x0, 0;\n\t"
        "mov.b64 {x2,x3}, e2; mov.b64 {x4,x5}, e4; mov.b64 {x6,x7}, e6;\n\t"
        "mov.b64 {x8,x9}, e8; mov.b64 {x10,x11}, e10; mov.b64 {x12,x13}, e12;\n\t"
        "mov.u32 x14, 0; mov.u32 x15, 0;\n\t"
        "mov.b64 {x1,y2}, o1; mov.b64 {y3,y4}, o3; mov.b64 {y5,y6}, o5;\n\t"
        "mov.b64 {y7,y8}, o7; mov.b64 {y9,y10}, o9; mov.b64 {y11,y12}, o11; mov.b64 {y13,y14}, o13;\n\t"
        "add.cc.u32 x2, x2, y2; addc.cc.u32 x3, x3, y3;\n\t"
        "addc.cc.u32 x4, x4, y4; addc.cc.u32 x5, x5, y5; addc.cc.u32 x6, x6, y6;\n\t"
        "addc.cc.u32 x7, x7, y7; addc.cc.u32 x8, x8, y8; addc.cc.u32 x9, x9, y9;\n\t"
        "addc.cc.u32 x10, x10, y10; addc.cc.u32 x11, x11, y11; addc.cc.u32 x12, x12, y12;\n\t"
        "addc.cc.u32 x13, x13, y13; addc.cc.u32 x14, x14, y14; addc.u32 x15, x15, 0;\n\t"

        /* X = 2*cross. Descending funnel shifts retain the old lower limb. */
        "shf.l.wrap.b32 x15, x14, x15, 1; shf.l.wrap.b32 x14, x13, x14, 1;\n\t"
        "shf.l.wrap.b32 x13, x12, x13, 1; shf.l.wrap.b32 x12, x11, x12, 1;\n\t"
        "shf.l.wrap.b32 x11, x10, x11, 1; shf.l.wrap.b32 x10, x9, x10, 1;\n\t"
        "shf.l.wrap.b32 x9, x8, x9, 1; shf.l.wrap.b32 x8, x7, x8, 1;\n\t"
        "shf.l.wrap.b32 x7, x6, x7, 1; shf.l.wrap.b32 x6, x5, x6, 1;\n\t"
        "shf.l.wrap.b32 x5, x4, x5, 1; shf.l.wrap.b32 x4, x3, x4, 1;\n\t"
        "shf.l.wrap.b32 x3, x2, x3, 1; shf.l.wrap.b32 x2, x1, x2, 1;\n\t"
#if QSB_SQR_X0_GLUE
        "shl.b32 x1, x1, 1;\n\t"
#else
        "shf.l.wrap.b32 x1, x0, x1, 1;\n\t"
#endif
        /* Add A[i]^2 at each 64-bit word 2*i. */
#if QSB_SQR_X0_GLUE
        "mov.b64 d1, {x2,x3}; mov.b64 d2, {x4,x5}; mov.b64 d3, {x6,x7};\n\t"
#else
        "mov.b64 d0, {x0,x1}; mov.b64 d1, {x2,x3}; mov.b64 d2, {x4,x5}; mov.b64 d3, {x6,x7};\n\t"
#endif
        "mov.b64 d4, {x8,x9}; mov.b64 d5, {x10,x11}; mov.b64 d6, {x12,x13}; mov.b64 d7, {x14,x15};\n\t"
#if QSB_SQR_X0_GLUE
        "mul.wide.u32 d0, a0, a0; { .reg .u32 q0l, q0h; mov.b64 {q0l,q0h}, d0; add.cc.u32 q0h, q0h, x1; mov.b64 d0, {q0l,q0h}; }\n\t"
#else
        "mul.wide.u32 t, a0, a0; add.cc.u64 d0, d0, t;\n\t"
#endif
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
        "\n\t"
        "mul.wide.u32 t, x9, 977;  add.cc.u64  g0, h0, t;\n\t"
        "mul.wide.u32 t, x11, 977; addc.cc.u64 g1, h1, t;\n\t"
        "mul.wide.u32 t, x13, 977; addc.cc.u64 g2, h2, t;\n\t"
        "mul.wide.u32 t, x15, 977; addc.u64 g3, h3, t;\n\t"
        "mov.b64 {z0,z1}, f0; mov.b64 {z2,z3}, f1; mov.b64 {z4,z5}, f2; mov.b64 {z6,z7}, f3;\n\t"
        "mov.b64 {w0,w1}, g0; mov.b64 {w2,w3}, g1; mov.b64 {w4,w5}, g2; mov.b64 {w6,w7}, g3;\n\t"
        "add.cc.u32 z1, z1, w0; addc.cc.u32 z2, z2, w1; addc.cc.u32 z3, z3, w2;\n\t"
        "addc.cc.u32 z4, z4, w3; addc.cc.u32 z5, z5, w4; addc.cc.u32 z6, z6, w5;\n\t"
        "addc.cc.u32 z7, z7, w6; addc.u32 z8, 0, w7;\n\t"
        /* First-fold square minus two 256-bit residues; signed high fold.
         * Like inherited fused-X3, short final carry is speculative: publication
         * still requires the existing exact host/OpenSSL verification. */
        ".reg .u32 u0,u1,u2,u3,u4,u5,u6,u7,v0,v1,v2,v3,v4,v5,v6,v7,ext,fc,fq,fh;\n\t"
        ".reg .u64 sfz,sft;\n\t"
        "mov.b64 {u0,u1}, %9;\n\t"
        "mov.b64 {u2,u3}, %10;\n\t"
        "mov.b64 {u4,u5}, %11;\n\t"
        "mov.b64 {u6,u7}, %12;\n\t"
        "mov.b64 {v0,v1}, %13;\n\t"
        "mov.b64 {v2,v3}, %14;\n\t"
        "mov.b64 {v4,v5}, %15;\n\t"
        "mov.b64 {v6,v7}, %16;\n\t"
        /* Unsigned first-fold offset (3p-c^2) minus p1. 3p rather than
         * 2p ensures nonnegativity for even noncanonical 256-bit p1.
         * Like inherited unsigned split3p-X3, top-word wrap is deliberately
         * omitted in this speculative filter. No all-domain exact claim. */
        "add.cc.u32 z0,z0,u0;\n\t"
        "addc.cc.u32 z1,z1,u1;\n\t"
        "addc.cc.u32 z2,z2,u2;\n\t"
        "addc.cc.u32 z3,z3,u3;\n\t"
        "addc.cc.u32 z4,z4,u4;\n\t"
        "addc.cc.u32 z5,z5,u5;\n\t"
        "addc.cc.u32 z6,z6,u6;\n\t"
        "addc.cc.u32 z7,z7,u7;\n\t"
        "addc.u32 z8,z8,%17;\n\t"
        "sub.cc.u32 z0,z0,v0;\n\t"
        "subc.cc.u32 z1,z1,v1;\n\t"
        "subc.cc.u32 z2,z2,v2;\n\t"
        "subc.cc.u32 z3,z3,v3;\n\t"
        "subc.cc.u32 z4,z4,v4;\n\t"
        "subc.cc.u32 z5,z5,v5;\n\t"
        "subc.cc.u32 z6,z6,v6;\n\t"
        "subc.cc.u32 z7,z7,v7;\n\t"
        "subc.u32 z8,z8,0;\n\t"
        "mov.b64 sfz,{z0,z8};\n\t"
        "mul.wide.u32 sft,z8,977;\n\t"
        "add.cc.u64 sft,sft,sfz;\n\t"
        "addc.u32 fc,0,0;\n\t"
        "mov.b64 {z0,fh},sft;\n\t"
        "add.cc.u32 z1,z1,fh;\n\t"
        "addc.u32 z2,z2,fc;\n\t"
        "\n\tmov.u32 %4, 0;\n\t"
        "mov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.b64 %2, {z4,z5}; mov.b64 %3, {z6,z7};\n\t"
        "}\n\t"
        : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3),"=r"(carry)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
          "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]),
          "l"(c[0]),"l"(c[1]),"l"(c[2]),"l"(c[3]),"r"(high));

    r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;

#else
    _ModSqr(r,a); _ModAdd256(r,r,(uint64_t*)b); _ModSub256(r,r,(uint64_t*)c);

#endif
}
