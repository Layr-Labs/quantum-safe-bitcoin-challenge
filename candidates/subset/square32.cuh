// fv_sqrfold: wide-idiom 8x32 square.
// -------------------------------------------------------------------------------------------
// Replaces the column-accumulator product section (mad.lo.cc / madc.hi.cc into a 3-word
// {c0,c1,c2} accumulator, 3 PTX ops per product with NO mul.wide fusion, plus 3 shuffle movs
// per column) with the SAME even/odd mul.wide.u32 column-chain idiom used by _ModMultCore --
// but restricted to the 28 distinct cross products A[i]*A[j] (i<j). The cross sum is then
// doubled and the 8 diagonals A[i]^2 are added, exactly as before, and the SAME sparse-prime
// double-fold tail (identical to _ModMultCore, third fold already dropped) is reused verbatim.
//
// Instruction delta (product generation only, PTX op count):
//   old: 28 cross products * (mad.lo.cc + madc.hi.cc + addc.u32) = 84  +  ~45 column-shuffle
//        movs = ~129 ops, none of them mul.wide-fused.
//   new: 28 cross mul.wide.u32 + ~24 add.cc.u64/addc chain ops + 15 u32 merge = ~67 ops,
//        the 28 multiplies now IMAD.WIDE.U32[.X]-fusable like the mul.
//   The double block (16), the 8-diagonal mad chain (16), and the fold tail are UNCHANGED.
//
// Correctness: the new even/odd cross chains produce the identical 16-limb cross sum x[0..15]
// that the old column accumulator produced (both = sum_{i<j} A[i]A[j]); the reused double +
// diagonal + fold tail is byte-identical to the original. Validated on CPU: a threaded C model
// with the exact add.cc/addc carry semantics of this asm is bit-identical to _ModMultCore(a,a)
// on 200000 random+edge inputs (0,1,p-1,p,p+1,2^256-1,[p,2^256)). See cpuTEST.md.
// Carry semantics: https://docs.nvidia.com/cuda/parallel-thread-execution/#extended-precision-arithmetic-instructions-madc
__device__ __forceinline__ void qsb_square32(uint64_t *out,const uint64_t *a){
#ifdef __CUDA_ARCH__
    uint64_t r0,r1,r2,r3;
    asm(
        "{\n"
        ".reg .u32 a0,a1,a2,a3,a4,a5,a6,a7;\n"
        ".reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n"
        ".reg .u32 y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,o15,cy;\n"
        ".reg .u64 e0,e1,e2,e3,e4,e5,e6,e7,o0,o1,o2,o3,o4,o5,o6,t,lc;\n"
        "mov.b64 {a0,a1}, %4;\n"
        "mov.b64 {a2,a3}, %5;\n"
        "mov.b64 {a4,a5}, %6;\n"
        "mov.b64 {a6,a7}, %7;\n"
        /* ---- even cross chain: columns 2,4,6,8,10,12 (i<j, i+j even) ---- */
        "mov.u64 e0, 0;\n"
        "mul.wide.u32 e1, a0, a2;\n"
        "mul.wide.u32 e2, a0, a4;\n"
        "mul.wide.u32 e3, a0, a6;\n"
        "mul.wide.u32 t, a1, a3; add.cc.u64  e2, e2, t;\n"
        "mul.wide.u32 t, a1, a5; addc.cc.u64 e3, e3, t;\n"
        "mul.wide.u32 t, a1, a7; addc.u64    e4, t, 0;\n"
        "mul.wide.u32 t, a2, a4; add.cc.u64  e3, e3, t;\n"
        "mul.wide.u32 t, a2, a6; addc.cc.u64 e4, e4, t;\n"
        "addc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n"
        "mul.wide.u32 t, a3, a5; add.cc.u64  e4, e4, t;\n"
        "mul.wide.u32 t, a3, a7; addc.u64    e5, t, lc;\n"
        "mul.wide.u32 t, a4, a6; add.cc.u64  e5, e5, t;\n"
        "mul.wide.u32 t, a5, a7; addc.u64    e6, t, 0;\n"
        "mov.u64 e7, 0;\n"
        /* ---- odd cross chain: columns 1,3,5,7,9,11,13 (i<j, i+j odd) ---- */
        "mul.wide.u32 o0, a0, a1;\n"
        "mul.wide.u32 o1, a0, a3;\n"
        "mul.wide.u32 o2, a0, a5;\n"
        "mul.wide.u32 o3, a0, a7;\n"
        "mul.wide.u32 t, a1, a2; add.cc.u64  o1, o1, t;\n"
        "mul.wide.u32 t, a1, a4; addc.cc.u64 o2, o2, t;\n"
        "mul.wide.u32 t, a1, a6; addc.cc.u64 o3, o3, t;\n"
        "addc.u32 cy, 0, 0; cvt.u64.u32 o4, cy;\n"
        "mul.wide.u32 t, a2, a3; add.cc.u64  o2, o2, t;\n"
        "mul.wide.u32 t, a2, a5; addc.cc.u64 o3, o3, t;\n"
        "mul.wide.u32 t, a2, a7; addc.cc.u64 o4, o4, t;\n"
        "addc.u32 cy, 0, 0; cvt.u64.u32 lc, cy;\n"
        "mul.wide.u32 t, a3, a4; add.cc.u64  o3, o3, t;\n"
        "mul.wide.u32 t, a3, a6; addc.cc.u64 o4, o4, t;\n"
        "addc.u64 o5, lc, 0;\n"
        "mul.wide.u32 t, a4, a5; add.cc.u64  o4, o4, t;\n"
        "mul.wide.u32 t, a4, a7; addc.cc.u64 o5, o5, t;\n"
        "addc.u32 cy, 0, 0; cvt.u64.u32 o6, cy;\n"
        "mul.wide.u32 t, a5, a6; add.cc.u64  o5, o5, t;\n"
        "mul.wide.u32 t, a6, a7; addc.cc.u64 o6, o6, t;\n"
        "addc.u32 o15, 0, 0;\n"
        /* ---- unpack e -> x[0..15], o -> y[1..14], merge y into x at 32-bit offset ---- */
        "mov.b64 {x0,x1}, e0;\n"
        "mov.b64 {x2,x3}, e1;\n"
        "mov.b64 {x4,x5}, e2;\n"
        "mov.b64 {x6,x7}, e3;\n"
        "mov.b64 {x8,x9}, e4;\n"
        "mov.b64 {x10,x11}, e5;\n"
        "mov.b64 {x12,x13}, e6;\n"
        "mov.b64 {x14,x15}, e7;\n"
        "mov.b64 {y1,y2}, o0;\n"
        "mov.b64 {y3,y4}, o1;\n"
        "mov.b64 {y5,y6}, o2;\n"
        "mov.b64 {y7,y8}, o3;\n"
        "mov.b64 {y9,y10}, o4;\n"
        "mov.b64 {y11,y12}, o5;\n"
        "mov.b64 {y13,y14}, o6;\n"
        "add.cc.u32 x1, x1, y1;\n"
        "addc.cc.u32 x2, x2, y2;\n"
        "addc.cc.u32 x3, x3, y3;\n"
        "addc.cc.u32 x4, x4, y4;\n"
        "addc.cc.u32 x5, x5, y5;\n"
        "addc.cc.u32 x6, x6, y6;\n"
        "addc.cc.u32 x7, x7, y7;\n"
        "addc.cc.u32 x8, x8, y8;\n"
        "addc.cc.u32 x9, x9, y9;\n"
        "addc.cc.u32 x10, x10, y10;\n"
        "addc.cc.u32 x11, x11, y11;\n"
        "addc.cc.u32 x12, x12, y12;\n"
        "addc.cc.u32 x13, x13, y13;\n"
        "addc.cc.u32 x14, x14, y14;\n"
        "addc.u32 x15, x15, o15;\n"
        /* ---- double the cross sum (x = 2 * sum_{i<j} A[i]A[j]) : UNCHANGED ---- */
        "add.cc.u32 x0,x0,x0;\n"
        "addc.cc.u32 x1,x1,x1;\n"
        "addc.cc.u32 x2,x2,x2;\n"
        "addc.cc.u32 x3,x3,x3;\n"
        "addc.cc.u32 x4,x4,x4;\n"
        "addc.cc.u32 x5,x5,x5;\n"
        "addc.cc.u32 x6,x6,x6;\n"
        "addc.cc.u32 x7,x7,x7;\n"
        "addc.cc.u32 x8,x8,x8;\n"
        "addc.cc.u32 x9,x9,x9;\n"
        "addc.cc.u32 x10,x10,x10;\n"
        "addc.cc.u32 x11,x11,x11;\n"
        "addc.cc.u32 x12,x12,x12;\n"
        "addc.cc.u32 x13,x13,x13;\n"
        "addc.cc.u32 x14,x14,x14;\n"
        "addc.u32 x15,x15,x15;\n"
        /* ---- add the 8 diagonals A[i]^2 at limb 2i (single mad carry chain) : UNCHANGED ---- */
        "mad.lo.cc.u32 x0,a0,a0,x0;\n"
        "madc.hi.cc.u32 x1,a0,a0,x1;\n"
        "madc.lo.cc.u32 x2,a1,a1,x2;\n"
        "madc.hi.cc.u32 x3,a1,a1,x3;\n"
        "madc.lo.cc.u32 x4,a2,a2,x4;\n"
        "madc.hi.cc.u32 x5,a2,a2,x5;\n"
        "madc.lo.cc.u32 x6,a3,a3,x6;\n"
        "madc.hi.cc.u32 x7,a3,a3,x7;\n"
        "madc.lo.cc.u32 x8,a4,a4,x8;\n"
        "madc.hi.cc.u32 x9,a4,a4,x9;\n"
        "madc.lo.cc.u32 x10,a5,a5,x10;\n"
        "madc.hi.cc.u32 x11,a5,a5,x11;\n"
        "madc.lo.cc.u32 x12,a6,a6,x12;\n"
        "madc.hi.cc.u32 x13,a6,a6,x13;\n"
        "madc.lo.cc.u32 x14,a7,a7,x14;\n"
        "madc.hi.u32 x15,a7,a7,x15;\n"
        /* ---- secp256k1 sparse-prime double-fold (identical to _ModMultCore; third fold dropped) ---- */
        ".reg .u64 r0,r1,r2,r3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;\n"
        "\t.reg .u32 f8,g8,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,w0,w1,w2,w3,w4,w5,w6,w7,m0,m1,m2;\n"
        "\tmov.b64 r0, {x0,x1}; mov.b64 r1, {x2,x3}; mov.b64 r2, {x4,x5}; mov.b64 r3, {x6,x7};\n"
        "\tmov.b64 h0, {x8,x9}; mov.b64 h1, {x10,x11}; mov.b64 h2, {x12,x13}; mov.b64 h3, {x14,x15};\n"
        "\tmul.wide.u32 t, x8, 977;  add.cc.u64  f0, r0, t;\n"
        "\tmul.wide.u32 t, x10, 977; addc.cc.u64 f1, r1, t;\n"
        "\tmul.wide.u32 t, x12, 977; addc.cc.u64 f2, r2, t;\n"
        "\tmul.wide.u32 t, x14, 977; addc.cc.u64 f3, r3, t;\n"
        "\taddc.u32 f8, 0, 0;\n"
        "\tmul.wide.u32 t, x9, 977;  add.cc.u64  g0, h0, t;\n"
        "\tmul.wide.u32 t, x11, 977; addc.cc.u64 g1, h1, t;\n"
        "\tmul.wide.u32 t, x13, 977; addc.cc.u64 g2, h2, t;\n"
        "\tmul.wide.u32 t, x15, 977; addc.cc.u64 g3, h3, t;\n"
        "\taddc.u32 g8, 0, 0;\n"
        "\tmov.b64 {z0,z1}, f0;\n"
        "\tmov.b64 {z2,z3}, f1;\n"
        "\tmov.b64 {z4,z5}, f2;\n"
        "\tmov.b64 {z6,z7}, f3;\n"
        "\tmov.b64 {w0,w1}, g0;\n"
        "\tmov.b64 {w2,w3}, g1;\n"
        "\tmov.b64 {w4,w5}, g2;\n"
        "\tmov.b64 {w6,w7}, g3;\n"
        "\tadd.cc.u32  z1, z1, w0;\n"
        "\taddc.cc.u32 z2, z2, w1;\n"
        "\taddc.cc.u32 z3, z3, w2;\n"
        "\taddc.cc.u32 z4, z4, w3;\n"
        "\taddc.cc.u32 z5, z5, w4;\n"
        "\taddc.cc.u32 z6, z6, w5;\n"
        "\taddc.cc.u32 z7, z7, w6;\n"
        "\taddc.cc.u32 z8, f8, w7;\n"
        "\taddc.u32    z9, g8, 0;\n"
        "\tmul.wide.u32 t, z8, 977; mov.b64 {m0,m1}, t;\n"
        "\tmad.lo.u32 m1, z9, 977, m1;\n"
        "\tadd.cc.u32 m1, m1, z8;\n"
        "\taddc.u32 m2, z9, 0;\n"
        "\tadd.cc.u32 z0, z0, m0; addc.cc.u32 z1, z1, m1; addc.cc.u32 z2, z2, m2;\n"
        "\taddc.cc.u32 z3, z3, 0;\n"
        "\taddc.cc.u32 z4, z4, 0;\n"
        "\taddc.cc.u32 z5, z5, 0;\n"
        "\taddc.cc.u32 z6, z6, 0;\n"
        "\taddc.u32 z7, z7, 0;\n"
        "mov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.b64 %2, {z4,z5}; mov.b64 %3, {z6,z7};\n"
        "\t}\n"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]));
    out[0]=r0;out[1]=r1;out[2]=r2;out[3]=r3;
#else
    /* Host mirror of the device schedule: even/odd cross chains -> merge -> double ->
       diagonals -> identical fold. Bit-identical to _ModMultCore(out,a,a). */
#define QSB_MW(x,y) ((uint64_t)(uint32_t)(x)*(uint32_t)(y))
    uint32_t A[8];
    for(int i=0;i<4;i++){A[2*i]=(uint32_t)a[i];A[2*i+1]=(uint32_t)(a[i]>>32);}
    __uint128_t acc; uint64_t E[8],O[7]; uint32_t o_hi;
    acc=0;
    acc+=0;                                                       E[0]=(uint64_t)acc; acc>>=64;
    acc+=(__uint128_t)QSB_MW(A[0],A[2]);                          E[1]=(uint64_t)acc; acc>>=64;
    acc+=(__uint128_t)QSB_MW(A[0],A[4])+QSB_MW(A[1],A[3]);        E[2]=(uint64_t)acc; acc>>=64;
    acc+=(__uint128_t)QSB_MW(A[0],A[6])+QSB_MW(A[1],A[5])+QSB_MW(A[2],A[4]); E[3]=(uint64_t)acc; acc>>=64;
    acc+=(__uint128_t)QSB_MW(A[1],A[7])+QSB_MW(A[2],A[6])+QSB_MW(A[3],A[5]); E[4]=(uint64_t)acc; acc>>=64;
    acc+=(__uint128_t)QSB_MW(A[3],A[7])+QSB_MW(A[4],A[6]);        E[5]=(uint64_t)acc; acc>>=64;
    acc+=(__uint128_t)QSB_MW(A[5],A[7]);                          E[6]=(uint64_t)acc; acc>>=64;
    acc+=0;                                                       E[7]=(uint64_t)acc;
    acc=0;
    acc+=(__uint128_t)QSB_MW(A[0],A[1]);                          O[0]=(uint64_t)acc; acc>>=64;
    acc+=(__uint128_t)QSB_MW(A[0],A[3])+QSB_MW(A[1],A[2]);        O[1]=(uint64_t)acc; acc>>=64;
    acc+=(__uint128_t)QSB_MW(A[0],A[5])+QSB_MW(A[1],A[4])+QSB_MW(A[2],A[3]); O[2]=(uint64_t)acc; acc>>=64;
    acc+=(__uint128_t)QSB_MW(A[0],A[7])+QSB_MW(A[1],A[6])+QSB_MW(A[2],A[5])+QSB_MW(A[3],A[4]); O[3]=(uint64_t)acc; acc>>=64;
    acc+=(__uint128_t)QSB_MW(A[2],A[7])+QSB_MW(A[3],A[6])+QSB_MW(A[4],A[5]); O[4]=(uint64_t)acc; acc>>=64;
    acc+=(__uint128_t)QSB_MW(A[4],A[7])+QSB_MW(A[5],A[6]);        O[5]=(uint64_t)acc; acc>>=64;
    acc+=(__uint128_t)QSB_MW(A[6],A[7]);                          O[6]=(uint64_t)acc; acc>>=64;
    o_hi=(uint32_t)acc;
    uint32_t x[16], y[15];
    for(int k=0;k<8;k++){ x[2*k]=(uint32_t)E[k]; x[2*k+1]=(uint32_t)(E[k]>>32); }
    for(int m=0;m<7;m++){ y[2*m+1]=(uint32_t)O[m]; y[2*m+2]=(uint32_t)(O[m]>>32); }
    { uint64_t c=0; for(int k=1;k<=14;k++){ uint64_t t=(uint64_t)x[k]+y[k]+c; x[k]=(uint32_t)t; c=t>>32; }
      x[15]=(uint32_t)((uint64_t)x[15]+o_hi+c); }
    { uint32_t c=0; for(int k=0;k<16;k++){ uint64_t t=((uint64_t)x[k]<<1)|c; x[k]=(uint32_t)t; c=(uint32_t)(t>>32); } }
    for(int i=0;i<8;i++){ uint64_t d=(uint64_t)A[i]*A[i]; uint64_t c=d;
        for(int k=2*i;k<16 && c;k++){ uint64_t t=(uint64_t)x[k]+(uint32_t)c; x[k]=(uint32_t)t; c=(c>>32)+(t>>32);} }
    __uint128_t s;
    uint64_t r0=x[0]|((uint64_t)x[1]<<32), r1=x[2]|((uint64_t)x[3]<<32),
             r2=x[4]|((uint64_t)x[5]<<32), r3=x[6]|((uint64_t)x[7]<<32);
    uint64_t h0=x[8]|((uint64_t)x[9]<<32), h1=x[10]|((uint64_t)x[11]<<32),
             h2=x[12]|((uint64_t)x[13]<<32), h3=x[14]|((uint64_t)x[15]<<32);
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
      t=(uint64_t)z[7]+c; z[7]=(uint32_t)t; }
    out[0]=z[0]|((uint64_t)z[1]<<32); out[1]=z[2]|((uint64_t)z[3]<<32);
    out[2]=z[4]|((uint64_t)z[5]<<32); out[3]=z[6]|((uint64_t)z[7]<<32);
#undef QSB_MW
#endif
}
