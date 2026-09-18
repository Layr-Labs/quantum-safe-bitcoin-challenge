/* Canonical field-transition primitive.
 * Arithmetic schedule derived from VanitySearch GPUMath,
 * Copyright (c) 2019 Jean Luc PONS.
 * GNU General Public License version 3; see candidate COPYING.
 * Computes (a*b-c*d) mod (2^256-2^32-977).
 * Canonical inputs required; canonical output; all aliasing supported.
 * Host fallback retains the checked signed stream.
 */
#pragma once
#include <stdint.h>
__device__ __forceinline__ void qsb_field_transition(uint64_t *out,
    const uint64_t *a,const uint64_t *b,const uint64_t *c,const uint64_t *d){
#ifdef __CUDA_ARCH__
    uint64_t r0,r1,r2,r3;
    asm( "{\n\n\t.reg .u32 a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7;\n\t.reg .u64 e0,e1,e2,e3,e4,e5,e6,e7,o0,o1,o2,o3,o4,o5,o6,t,lc;\n\t.reg .u32 cy,o15;\n\t.reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n\t.reg .u32 y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14;\n\t.reg .u32 c0,c1,c2,c3,c4,c5,c6,c7,d0,d1,d2,d3,d4,d5,d6,d7,saved0,saved1,saved2,saved3,saved4,saved5,saved6,saved7,saved8,saved9,saved10,saved11,saved12,saved13,saved14,saved15,br,sign;\n.reg .pred ge;\nmov.b64 {a0,a1}, %4;\nmov.b64 {a2,a3}, %5;\nmov.b64 {a4,a5}, %6;\nmov.b64 {a6,a7}, %7;\nmov.b64 {b0,b1}, %8;\nmov.b64 {b2,b3}, %9;\nmov.b64 {b4,b5}, %10;\nmov.b64 {b6,b7}, %11;\nmov.b64 {c0,c1}, %12;\nmov.b64 {c2,c3}, %13;\nmov.b64 {c4,c5}, %14;\nmov.b64 {c6,c7}, %15;\nmov.b64 {d0,d1}, %16;\nmov.b64 {d2,d3}, %17;\nmov.b64 {d4,d5}, %18;\nmov.b64 {d6,d7}, %19;\nmul.wide.u32 e0, a0, b0; mul.wide.u32 e1, a0, b2; mul.wide.u32 e2, a0, b4; mul.wide.u32 e3, a0, b6;\n\tmul.wide.u32 t, a1, b1; add.cc.u64 e1, e1, t;\n\tmul.wide.u32 t, a1, b3; addc.cc.u64 e2, e2, t;\n\tmul.wide.u32 t, a1, b5; addc.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, a1, b7; addc.u64 e4, t, 0;\n\tmul.wide.u32 t, a2, b0; add.cc.u64 e1, e1, t;\n\tmul.wide.u32 t, a2, b2; addc.cc.u64 e2, e2, t;\n\tmul.wide.u32 t, a2, b4; addc.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, a2, b6; addc.cc.u64 e4, e4, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, a3, b1; add.cc.u64 e2, e2, t;\n\tmul.wide.u32 t, a3, b3; addc.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, a3, b5; addc.cc.u64 e4, e4, t;\n\tmul.wide.u32 t, a3, b7; addc.u64 e5, t, lc;\n\tmul.wide.u32 t, a4, b0; add.cc.u64 e2, e2, t;\n\tmul.wide.u32 t, a4, b2; addc.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, a4, b4; addc.cc.u64 e4, e4, t;\n\tmul.wide.u32 t, a4, b6; addc.cc.u64 e5, e5, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, a5, b1; add.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, a5, b3; addc.cc.u64 e4, e4, t;\n\tmul.wide.u32 t, a5, b5; addc.cc.u64 e5, e5, t;\n\tmul.wide.u32 t, a5, b7; addc.u64 e6, t, lc;\n\tmul.wide.u32 t, a6, b0; add.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, a6, b2; addc.cc.u64 e4, e4, t;\n\tmul.wide.u32 t, a6, b4; addc.cc.u64 e5, e5, t;\n\tmul.wide.u32 t, a6, b6; addc.cc.u64 e6, e6, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, a7, b1; add.cc.u64 e4, e4, t;\n\tmul.wide.u32 t, a7, b3; addc.cc.u64 e5, e5, t;\n\tmul.wide.u32 t, a7, b5; addc.cc.u64 e6, e6, t;\n\tmul.wide.u32 t, a7, b7; addc.u64 e7, t, lc;\n\tmul.wide.u32 o0, a0, b1; mul.wide.u32 o1, a0, b3; mul.wide.u32 o2, a0, b5; mul.wide.u32 o3, a0, b7;\n\tmul.wide.u32 t, a1, b0; add.cc.u64 o0, o0, t;\n\tmul.wide.u32 t, a1, b2; addc.cc.u64 o1, o1, t;\n\tmul.wide.u32 t, a1, b4; addc.cc.u64 o2, o2, t;\n\tmul.wide.u32 t, a1, b6; addc.cc.u64 o3, o3, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, a2, b1; add.cc.u64 o1, o1, t;\n\tmul.wide.u32 t, a2, b3; addc.cc.u64 o2, o2, t;\n\tmul.wide.u32 t, a2, b5; addc.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, a2, b7; addc.u64 o4, t, lc;\n\tmul.wide.u32 t, a3, b0; add.cc.u64 o1, o1, t;\n\tmul.wide.u32 t, a3, b2; addc.cc.u64 o2, o2, t;\n\tmul.wide.u32 t, a3, b4; addc.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, a3, b6; addc.cc.u64 o4, o4, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, a4, b1; add.cc.u64 o2, o2, t;\n\tmul.wide.u32 t, a4, b3; addc.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, a4, b5; addc.cc.u64 o4, o4, t;\n\tmul.wide.u32 t, a4, b7; addc.u64 o5, t, lc;\n\tmul.wide.u32 t, a5, b0; add.cc.u64 o2, o2, t;\n\tmul.wide.u32 t, a5, b2; addc.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, a5, b4; addc.cc.u64 o4, o4, t;\n\tmul.wide.u32 t, a5, b6; addc.cc.u64 o5, o5, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, a6, b1; add.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, a6, b3; addc.cc.u64 o4, o4, t;\n\tmul.wide.u32 t, a6, b5; addc.cc.u64 o5, o5, t;\n\tmul.wide.u32 t, a6, b7; addc.u64 o6, t, lc;\n\tmul.wide.u32 t, a7, b0; add.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, a7, b2; addc.cc.u64 o4, o4, t;\n\tmul.wide.u32 t, a7, b4; addc.cc.u64 o5, o5, t;\n\tmul.wide.u32 t, a7, b6; addc.cc.u64 o6, o6, t;\n\taddc.u32 o15, 0, 0;\n\tmov.b64 {x0,x1}, e0;\n\tmov.b64 {x2,x3}, e1;\n\tmov.b64 {x4,x5}, e2;\n\tmov.b64 {x6,x7}, e3;\n\tmov.b64 {x8,x9}, e4;\n\tmov.b64 {x10,x11}, e5;\n\tmov.b64 {x12,x13}, e6;\n\tmov.b64 {x14,x15}, e7;\n\tmov.b64 {y1,y2}, o0;\n\tmov.b64 {y3,y4}, o1;\n\tmov.b64 {y5,y6}, o2;\n\tmov.b64 {y7,y8}, o3;\n\tmov.b64 {y9,y10}, o4;\n\tmov.b64 {y11,y12}, o5;\n\tmov.b64 {y13,y14}, o6;\n\tadd.cc.u32 x1, x1, y1;\n\taddc.cc.u32 x2, x2, y2;\n\taddc.cc.u32 x3, x3, y3;\n\taddc.cc.u32 x4, x4, y4;\n\taddc.cc.u32 x5, x5, y5;\n\taddc.cc.u32 x6, x6, y6;\n\taddc.cc.u32 x7, x7, y7;\n\taddc.cc.u32 x8, x8, y8;\n\taddc.cc.u32 x9, x9, y9;\n\taddc.cc.u32 x10, x10, y10;\n\taddc.cc.u32 x11, x11, y11;\n\taddc.cc.u32 x12, x12, y12;\n\taddc.cc.u32 x13, x13, y13;\n\taddc.cc.u32 x14, x14, y14;\n\taddc.u32 x15, x15, o15;\n\nmov.u32 saved0, x0;\nmov.u32 saved1, x1;\nmov.u32 saved2, x2;\nmov.u32 saved3, x3;\nmov.u32 saved4, x4;\nmov.u32 saved5, x5;\nmov.u32 saved6, x6;\nmov.u32 saved7, x7;\nmov.u32 saved8, x8;\nmov.u32 saved9, x9;\nmov.u32 saved10, x10;\nmov.u32 saved11, x11;\nmov.u32 saved12, x12;\nmov.u32 saved13, x13;\nmov.u32 saved14, x14;\nmov.u32 saved15, x15;\nmul.wide.u32 e0, c0, d0; mul.wide.u32 e1, c0, d2; mul.wide.u32 e2, c0, d4; mul.wide.u32 e3, c0, d6;\n\tmul.wide.u32 t, c1, d1; add.cc.u64 e1, e1, t;\n\tmul.wide.u32 t, c1, d3; addc.cc.u64 e2, e2, t;\n\tmul.wide.u32 t, c1, d5; addc.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, c1, d7; addc.u64 e4, t, 0;\n\tmul.wide.u32 t, c2, d0; add.cc.u64 e1, e1, t;\n\tmul.wide.u32 t, c2, d2; addc.cc.u64 e2, e2, t;\n\tmul.wide.u32 t, c2, d4; addc.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, c2, d6; addc.cc.u64 e4, e4, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, c3, d1; add.cc.u64 e2, e2, t;\n\tmul.wide.u32 t, c3, d3; addc.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, c3, d5; addc.cc.u64 e4, e4, t;\n\tmul.wide.u32 t, c3, d7; addc.u64 e5, t, lc;\n\tmul.wide.u32 t, c4, d0; add.cc.u64 e2, e2, t;\n\tmul.wide.u32 t, c4, d2; addc.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, c4, d4; addc.cc.u64 e4, e4, t;\n\tmul.wide.u32 t, c4, d6; addc.cc.u64 e5, e5, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, c5, d1; add.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, c5, d3; addc.cc.u64 e4, e4, t;\n\tmul.wide.u32 t, c5, d5; addc.cc.u64 e5, e5, t;\n\tmul.wide.u32 t, c5, d7; addc.u64 e6, t, lc;\n\tmul.wide.u32 t, c6, d0; add.cc.u64 e3, e3, t;\n\tmul.wide.u32 t, c6, d2; addc.cc.u64 e4, e4, t;\n\tmul.wide.u32 t, c6, d4; addc.cc.u64 e5, e5, t;\n\tmul.wide.u32 t, c6, d6; addc.cc.u64 e6, e6, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, c7, d1; add.cc.u64 e4, e4, t;\n\tmul.wide.u32 t, c7, d3; addc.cc.u64 e5, e5, t;\n\tmul.wide.u32 t, c7, d5; addc.cc.u64 e6, e6, t;\n\tmul.wide.u32 t, c7, d7; addc.u64 e7, t, lc;\n\tmul.wide.u32 o0, c0, d1; mul.wide.u32 o1, c0, d3; mul.wide.u32 o2, c0, d5; mul.wide.u32 o3, c0, d7;\n\tmul.wide.u32 t, c1, d0; add.cc.u64 o0, o0, t;\n\tmul.wide.u32 t, c1, d2; addc.cc.u64 o1, o1, t;\n\tmul.wide.u32 t, c1, d4; addc.cc.u64 o2, o2, t;\n\tmul.wide.u32 t, c1, d6; addc.cc.u64 o3, o3, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, c2, d1; add.cc.u64 o1, o1, t;\n\tmul.wide.u32 t, c2, d3; addc.cc.u64 o2, o2, t;\n\tmul.wide.u32 t, c2, d5; addc.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, c2, d7; addc.u64 o4, t, lc;\n\tmul.wide.u32 t, c3, d0; add.cc.u64 o1, o1, t;\n\tmul.wide.u32 t, c3, d2; addc.cc.u64 o2, o2, t;\n\tmul.wide.u32 t, c3, d4; addc.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, c3, d6; addc.cc.u64 o4, o4, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, c4, d1; add.cc.u64 o2, o2, t;\n\tmul.wide.u32 t, c4, d3; addc.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, c4, d5; addc.cc.u64 o4, o4, t;\n\tmul.wide.u32 t, c4, d7; addc.u64 o5, t, lc;\n\tmul.wide.u32 t, c5, d0; add.cc.u64 o2, o2, t;\n\tmul.wide.u32 t, c5, d2; addc.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, c5, d4; addc.cc.u64 o4, o4, t;\n\tmul.wide.u32 t, c5, d6; addc.cc.u64 o5, o5, t;\n\taddc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n\tmul.wide.u32 t, c6, d1; add.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, c6, d3; addc.cc.u64 o4, o4, t;\n\tmul.wide.u32 t, c6, d5; addc.cc.u64 o5, o5, t;\n\tmul.wide.u32 t, c6, d7; addc.u64 o6, t, lc;\n\tmul.wide.u32 t, c7, d0; add.cc.u64 o3, o3, t;\n\tmul.wide.u32 t, c7, d2; addc.cc.u64 o4, o4, t;\n\tmul.wide.u32 t, c7, d4; addc.cc.u64 o5, o5, t;\n\tmul.wide.u32 t, c7, d6; addc.cc.u64 o6, o6, t;\n\taddc.u32 o15, 0, 0;\n\tmov.b64 {x0,x1}, e0;\n\tmov.b64 {x2,x3}, e1;\n\tmov.b64 {x4,x5}, e2;\n\tmov.b64 {x6,x7}, e3;\n\tmov.b64 {x8,x9}, e4;\n\tmov.b64 {x10,x11}, e5;\n\tmov.b64 {x12,x13}, e6;\n\tmov.b64 {x14,x15}, e7;\n\tmov.b64 {y1,y2}, o0;\n\tmov.b64 {y3,y4}, o1;\n\tmov.b64 {y5,y6}, o2;\n\tmov.b64 {y7,y8}, o3;\n\tmov.b64 {y9,y10}, o4;\n\tmov.b64 {y11,y12}, o5;\n\tmov.b64 {y13,y14}, o6;\n\tadd.cc.u32 x1, x1, y1;\n\taddc.cc.u32 x2, x2, y2;\n\taddc.cc.u32 x3, x3, y3;\n\taddc.cc.u32 x4, x4, y4;\n\taddc.cc.u32 x5, x5, y5;\n\taddc.cc.u32 x6, x6, y6;\n\taddc.cc.u32 x7, x7, y7;\n\taddc.cc.u32 x8, x8, y8;\n\taddc.cc.u32 x9, x9, y9;\n\taddc.cc.u32 x10, x10, y10;\n\taddc.cc.u32 x11, x11, y11;\n\taddc.cc.u32 x12, x12, y12;\n\taddc.cc.u32 x13, x13, y13;\n\taddc.cc.u32 x14, x14, y14;\n\taddc.u32 x15, x15, o15;\n\nsub.cc.u32 x0, saved0, x0;\nsubc.cc.u32 x1, saved1, x1;\nsubc.cc.u32 x2, saved2, x2;\nsubc.cc.u32 x3, saved3, x3;\nsubc.cc.u32 x4, saved4, x4;\nsubc.cc.u32 x5, saved5, x5;\nsubc.cc.u32 x6, saved6, x6;\nsubc.cc.u32 x7, saved7, x7;\nsubc.cc.u32 x8, saved8, x8;\nsubc.cc.u32 x9, saved9, x9;\nsubc.cc.u32 x10, saved10, x10;\nsubc.cc.u32 x11, saved11, x11;\nsubc.cc.u32 x12, saved12, x12;\nsubc.cc.u32 x13, saved13, x13;\nsubc.cc.u32 x14, saved14, x14;\nsubc.cc.u32 x15, saved15, x15;\nsubc.u32 br, 0, 0;\nand.b32 sign, br, 1;\n\t.reg .u64 r0,r1,r2,r3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;\n\t.reg .u32 f8,g8,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,w0,w1,w2,w3,w4,w5,w6,w7,m0,m1,m2;\n\tmov.b64 r0, {x0,x1}; mov.b64 r1, {x2,x3}; mov.b64 r2, {x4,x5}; mov.b64 r3, {x6,x7};\n\tmov.b64 h0, {x8,x9}; mov.b64 h1, {x10,x11}; mov.b64 h2, {x12,x13}; mov.b64 h3, {x14,x15};\n\tmul.wide.u32 t, x8, 977;  add.cc.u64  f0, r0, t;\n\tmul.wide.u32 t, x10, 977; addc.cc.u64 f1, r1, t;\n\tmul.wide.u32 t, x12, 977; addc.cc.u64 f2, r2, t;\n\tmul.wide.u32 t, x14, 977; addc.cc.u64 f3, r3, t;\n\taddc.u32 f8, 0, 0;\n\tmul.wide.u32 t, x9, 977;  add.cc.u64  g0, h0, t;\n\tmul.wide.u32 t, x11, 977; addc.cc.u64 g1, h1, t;\n\tmul.wide.u32 t, x13, 977; addc.cc.u64 g2, h2, t;\n\tmul.wide.u32 t, x15, 977; addc.cc.u64 g3, h3, t;\n\taddc.u32 g8, 0, 0;\n\tmov.b64 {z0,z1}, f0;\n\tmov.b64 {z2,z3}, f1;\n\tmov.b64 {z4,z5}, f2;\n\tmov.b64 {z6,z7}, f3;\n\tmov.b64 {w0,w1}, g0;\n\tmov.b64 {w2,w3}, g1;\n\tmov.b64 {w4,w5}, g2;\n\tmov.b64 {w6,w7}, g3;\n\tadd.cc.u32  z1, z1, w0;\n\taddc.cc.u32 z2, z2, w1;\n\taddc.cc.u32 z3, z3, w2;\n\taddc.cc.u32 z4, z4, w3;\n\taddc.cc.u32 z5, z5, w4;\n\taddc.cc.u32 z6, z6, w5;\n\taddc.cc.u32 z7, z7, w6;\n\taddc.cc.u32 z8, f8, w7;\n\taddc.u32    z9, g8, 0;\nmul.lo.u32 m0, sign, 954529;\nmul.lo.u32 m1, sign, 1954;\nmul.lo.u32 m2, sign, 1;\nsub.cc.u32 z0, z0, m0;\nsubc.cc.u32 z1, z1, m1;\nsubc.cc.u32 z2, z2, m2;\nsubc.cc.u32 z3, z3, 0;\nsubc.cc.u32 z4, z4, 0;\nsubc.cc.u32 z5, z5, 0;\nsubc.cc.u32 z6, z6, 0;\nsubc.cc.u32 z7, z7, 0;\nsubc.cc.u32 z8, z8, 0;\nsubc.cc.u32 z9, z9, 0;\n\n\tmul.wide.u32 t, z8, 977; mov.b64 {m0,m1}, t;\n\tmad.lo.u32 m1, z9, 977, m1;\n\tadd.cc.u32 m1, m1, z8;\n\taddc.u32 m2, z9, 0;\n\tadd.cc.u32 z0, z0, m0; addc.cc.u32 z1, z1, m1; addc.cc.u32 z2, z2, m2;\n\taddc.cc.u32 z3, z3, 0;\n\taddc.cc.u32 z4, z4, 0;\n\taddc.cc.u32 z5, z5, 0;\n\taddc.cc.u32 z6, z6, 0;\n\taddc.cc.u32 z7, z7, 0;\n.reg .u32 cf;\naddc.u32 cf, 0, 0;\nmul.lo.u32 m0, cf, 977;\nadd.cc.u32 z0, z0, m0;\naddc.cc.u32 z1, z1, cf;\naddc.u32 z2, z2, 0;\n\t\nadd.cc.u32 w0, z0, 977;\naddc.cc.u32 w1, z1, 1;\naddc.cc.u32 w2, z2, 0;\naddc.cc.u32 w3, z3, 0;\naddc.cc.u32 w4, z4, 0;\naddc.cc.u32 w5, z5, 0;\naddc.cc.u32 w6, z6, 0;\naddc.cc.u32 w7, z7, 0;\naddc.u32 cf, 0, 0;\nsetp.ne.u32 ge, cf, 0;\nselp.b32 z0, w0, z0, ge;\nselp.b32 z1, w1, z1, ge;\nselp.b32 z2, w2, z2, ge;\nselp.b32 z3, w3, z3, ge;\nselp.b32 z4, w4, z4, ge;\nselp.b32 z5, w5, z5, ge;\nselp.b32 z6, w6, z6, ge;\nselp.b32 z7, w7, z7, ge;\nmov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.b64 %2, {z4,z5}; mov.b64 %3, {z6,z7};\n\t\n}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]),"l"(c[0]),"l"(c[1]),"l"(c[2]),"l"(c[3]),"l"(d[0]),"l"(d[1]),"l"(d[2]),"l"(d[3]) );
    out[0]=r0;out[1]=r1;out[2]=r2;out[3]=r3;
#else

    // Same signed column recurrence, with a wide host accumulator.
    uint32_t inputs[4][8], z[8];
    const uint64_t* pointers[4]={a,b,c,d};
    for(int j=0;j<4;++j)for(int i=0;i<4;++i){
        inputs[j][2*i]=(uint32_t)pointers[j][i];
        inputs[j][2*i+1]=(uint32_t)(pointers[j][i]>>32);
    }
    __int128 acc=0;
    uint64_t fold=0,previous=0;
    const __int128 radix=(__int128)1<<32;
    for(int k=0;k<16;++k){
        for(int i=(k>7?k-7:0);i<=7 && i<=k;++i){
            int j=k-i;
            acc+=(uint64_t)inputs[0][i]*inputs[1][j];
            acc-=(uint64_t)inputs[2][i]*inputs[3][j];
        }
        uint32_t digit=(uint32_t)acc;
        acc=(acc-digit)/radix; // exact signed division; no signed-shift assumption
        if(k<8)z[k]=digit;
        else{
            uint64_t t=(uint64_t)digit*977+z[k-8]+previous+fold;
            z[k-8]=(uint32_t)t;fold=t>>32;previous=digit;
        }
    }
    fold+=previous;
    const uint32_t correction[3]={954529,1954,1};
    uint64_t borrow=0,sign=(acc<0);
    for(int i=0;i<8;++i){
        uint64_t sub=(i<3?(uint64_t)correction[i]*sign:0)+borrow;
        uint64_t old=z[i];z[i]=(uint32_t)(old-sub);borrow=old<sub;
    }
    fold-=borrow; // nonnegative for canonical inputs: see README proof
    uint32_t h0=(uint32_t)fold,h1=(uint32_t)(fold>>32);
    uint64_t t=(uint64_t)z[0]+(uint64_t)h0*977;
    z[0]=(uint32_t)t;fold=t>>32;
    t=(uint64_t)z[1]+(uint64_t)h1*977+h0+fold;
    z[1]=(uint32_t)t;fold=t>>32;
    for(int i=2;i<8;++i){
        t=(uint64_t)z[i]+fold+(i==2?h1:0);
        z[i]=(uint32_t)t;fold=t>>32;
    }
    uint64_t carry=0;
    for(int i=0;i<8;++i){
        t=(uint64_t)z[i]+carry+(i==0?fold*977:i==1?fold:0);
        z[i]=(uint32_t)t;carry=t>>32;
    }
    uint32_t w[8];carry=0;
    for(int i=0;i<8;++i){
        t=(uint64_t)z[i]+carry+(i==0?977:i==1?1:0);
        w[i]=(uint32_t)t;carry=t>>32;
    }
    for(int i=0;i<4;++i){
        const uint32_t* v=carry?w:z;
        out[i]=(uint64_t)v[2*i]|((uint64_t)v[2*i+1]<<32);
    }
#endif
}
