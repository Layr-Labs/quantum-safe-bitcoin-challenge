// QSB fusion integration: audited current interleaved square schedule.
// Derived from the upstream square and xlib fused reduction; see package provenance.
// GPU throughput has not been established by the CPU semantic tests.
/*
* This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
* Copyright (c) 2019 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

// ---------------------------------------------------------------------------------
// 256(+64) bits integer CUDA libray for SECPK1
// ---------------------------------------------------------------------------------


#define GRP_SIZE (1024*2)

#define HSIZE ((GRP_SIZE / 2) - 1)

// 64bits lsb negative inverse of P (mod 2^64)
/* Fused XYZZ squaring reduction: xlib's public f297b0f9 / 8afe4bd lineage.
 * Integration: use the pinned current _ModSqr 512-bit product schedule verbatim,
 * then xlib's original carry-complete r*r+e-2q reduction. The older copied square
 * schedule is no longer used here. Fusion defaults to 1 for this experimental
 * candidate; -DQSB_FUSE_SQRADDSUB2=0 restores the baseline device path.
 * CPU PTX semantic tests passed. No author-side GPU performance claim is made.
 */
#ifndef QSB_FUSE_SQRADDSUB2
#define QSB_FUSE_SQRADDSUB2 1
#endif
#if QSB_FUSE_SQRADDSUB2 != 0 && QSB_FUSE_SQRADDSUB2 != 1
#error QSB_FUSE_SQRADDSUB2 must be 0 or 1
#endif

#define MM64 0xD838091DD2253531ULL


// We need 1 extra block for ModInv
#define NBBLOCK 5
#define BIFULLSIZE 40

// Assembly directives
#define UADDO(c, a, b) asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADDC(c, a, b) asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADD(c, a, b) asm volatile ("addc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define UADDO1(c, a) asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADDC1(c, a) asm volatile ("addc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADD1(c, a) asm volatile ("addc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));

#define USUBO(c, a, b) asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUBC(c, a, b) asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUB(c, a, b) asm volatile ("subc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define USUBO1(c, a) asm volatile ("sub.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUBC1(c, a) asm volatile ("subc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUB1(c, a) asm volatile ("subc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) );

#define UMULLO(lo,a, b) asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
#define UMULHI(hi,a, b) asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
#define MADDO(r,a,b,c) asm volatile ("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADDC(r,a,b,c) asm volatile ("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADD(r,a,b,c) asm volatile ("madc.hi.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
#define MADDS(r,a,b,c) asm volatile ("madc.hi.s64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));

// SECPK1 endomorphism constants
//__device__ __constant__ uint64_t _beta[] = { 0xC1396C28719501EEULL, 0x9CF0497512F58995ULL, 0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL };
//__device__ __constant__ uint64_t _beta2[] = { 0x3EC693D68E6AFA40ULL, 0x630FB68AED0A766AULL, 0x919BB86153CBCB16ULL, 0x851695D49A83F8EFULL };


// ---------------------------------------------------------------------------------------

#define _IsPositive(x) (((int64_t)(x[4]))>=0LL)
#define _IsNegative(x) (((int64_t)(x[4]))<0LL)
#define _IsEqual(a,b)  ((a[4] == b[4]) && (a[3] == b[3]) && (a[2] == b[2]) && (a[1] == b[1]) && (a[0] == b[0]))
#define _IsZero(a)     ((a[4] | a[3] | a[2] | a[1] | a[0]) == 0ULL)
#define _IsOne(a)      ((a[4] == 0ULL) && (a[3] == 0ULL) && (a[2] == 0ULL) && (a[1] == 0ULL) && (a[0] == 1ULL))

#define IDX threadIdx.x

#define __sright128(a,b,n) ((a)>>(n))|((b)<<(64-(n)))
#define __sleft128(a,b,n) ((b)<<(n))|((a)>>(64-(n)))

// ---------------------------------------------------------------------------------------

#define AddP(r) { \
  UADDO1(r[0], 0xFFFFFFFEFFFFFC2FULL); \
  UADDC1(r[1], 0xFFFFFFFFFFFFFFFFULL); \
  UADDC1(r[2], 0xFFFFFFFFFFFFFFFFULL); \
  UADDC1(r[3], 0xFFFFFFFFFFFFFFFFULL); \
  UADD1(r[4], 0ULL);}

// ---------------------------------------------------------------------------------------

#define Add2(r,a,b)  {\
  UADDO(r[0], a[0], b[0]); \
  UADDC(r[1], a[1], b[1]); \
  UADDC(r[2], a[2], b[2]); \
  UADDC(r[3], a[3], b[3]); \
  UADD(r[4], a[4], b[4]);}

// ---------------------------------------------------------------------------------------

#define SubP(r) { \
  USUBO1(r[0], 0xFFFFFFFEFFFFFC2FULL); \
  USUBC1(r[1], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[2], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[3], 0xFFFFFFFFFFFFFFFFULL); \
  USUB1(r[4], 0ULL);}

// ---------------------------------------------------------------------------------------

#define Sub2(r,a,b)  {\
  USUBO(r[0], a[0], b[0]); \
  USUBC(r[1], a[1], b[1]); \
  USUBC(r[2], a[2], b[2]); \
  USUBC(r[3], a[3], b[3]); \
  USUB(r[4], a[4], b[4]);}

// ---------------------------------------------------------------------------------------

#define Sub1(r,a) {\
  USUBO1(r[0], a[0]); \
  USUBC1(r[1], a[1]); \
  USUBC1(r[2], a[2]); \
  USUBC1(r[3], a[3]); \
  USUB1(r[4], a[4]);}

// ---------------------------------------------------------------------------------------

#define Neg(r) {\
USUBO(r[0],0ULL,r[0]); \
USUBC(r[1],0ULL,r[1]); \
USUBC(r[2],0ULL,r[2]); \
USUBC(r[3],0ULL,r[3]); \
USUB(r[4],0ULL,r[4]); }

// ---------------------------------------------------------------------------------------

#define UMult(r, a, b) {\
  UMULLO(r[0],a[0],b); \
  UMULLO(r[1],a[1],b); \
  MADDO(r[1], a[0],b,r[1]); \
  UMULLO(r[2],a[2], b); \
  MADDC(r[2], a[1], b, r[2]); \
  UMULLO(r[3],a[3], b); \
  MADDC(r[3], a[2], b, r[3]); \
  MADD(r[4], a[3], b, 0ULL);}

// ---------------------------------------------------------------------------------------

#define Load(r, a) {\
  (r)[0] = (a)[0]; \
  (r)[1] = (a)[1]; \
  (r)[2] = (a)[2]; \
  (r)[3] = (a)[3]; \
  (r)[4] = (a)[4];}

// ---------------------------------------------------------------------------------------

#define _LoadI64(r, a) {\
  (r)[0] = a; \
  (r)[1] = a>>63; \
  (r)[2] = (r)[1]; \
  (r)[3] = (r)[1]; \
  (r)[4] = (r)[1];}
// ---------------------------------------------------------------------------------------

#define Load256(r, a) {\
  (r)[0] = (a)[0]; \
  (r)[1] = (a)[1]; \
  (r)[2] = (a)[2]; \
  (r)[3] = (a)[3];}

// ---------------------------------------------------------------------------------------

#define Load256A(r, a) {\
  (r)[0] = (a)[IDX]; \
  (r)[1] = (a)[IDX+blockDim.x]; \
  (r)[2] = (a)[IDX+2*blockDim.x]; \
  (r)[3] = (a)[IDX+3*blockDim.x];}

// ---------------------------------------------------------------------------------------

#define Store256A(r, a) {\
  (r)[IDX] = (a)[0]; \
  (r)[IDX+blockDim.x] = (a)[1]; \
  (r)[IDX+2*blockDim.x] = (a)[2]; \
  (r)[IDX+3*blockDim.x] = (a)[3];}

// ---------------------------------------------------------------------------------------

__device__ void _ShiftR62(uint64_t *r)
{

    r[0] = (r[1] << 2) | (r[0] >> 62);
    r[1] = (r[2] << 2) | (r[1] >> 62);
    r[2] = (r[3] << 2) | (r[2] >> 62);
    r[3] = (r[4] << 2) | (r[3] >> 62);
    // With sign extent
    r[4] = (int64_t)(r[4]) >> 62;

}

__device__ void _ShiftR62(uint64_t dest[5], uint64_t r[5], uint64_t carry)
{

    dest[0] = (r[1] << 2) | (r[0] >> 62);
    dest[1] = (r[2] << 2) | (r[1] >> 62);
    dest[2] = (r[3] << 2) | (r[2] >> 62);
    dest[3] = (r[4] << 2) | (r[3] >> 62);
    dest[4] = (carry << 2) | (r[4] >> 62);

}

// ---------------------------------------------------------------------------------------

__device__ void _IMult(uint64_t *r, uint64_t *a, int64_t b)
{

    uint64_t t[NBBLOCK];

    // Make b positive
    if (b < 0) {
        b = -b;
        USUBO(t[0], 0ULL, a[0]);
        USUBC(t[1], 0ULL, a[1]);
        USUBC(t[2], 0ULL, a[2]);
        USUBC(t[3], 0ULL, a[3]);
        USUB(t[4], 0ULL, a[4]);
    } else {
        Load(t, a);
    }

    UMULLO(r[0], t[0], b);
    UMULLO(r[1], t[1], b);
    MADDO(r[1], t[0], b, r[1]);
    UMULLO(r[2], t[2], b);
    MADDC(r[2], t[1], b, r[2]);
    UMULLO(r[3], t[3], b);
    MADDC(r[3], t[2], b, r[3]);
    UMULLO(r[4], t[4], b);
    MADD(r[4], t[3], b, r[4]);

}

__device__ uint64_t _IMultC(uint64_t *r, uint64_t *a, int64_t b)
{

    uint64_t t[NBBLOCK];
    uint64_t carry;

    // Make b positive
    if (b < 0) {
        b = -b;
        USUBO(t[0], 0ULL, a[0]);
        USUBC(t[1], 0ULL, a[1]);
        USUBC(t[2], 0ULL, a[2]);
        USUBC(t[3], 0ULL, a[3]);
        USUB(t[4], 0ULL, a[4]);
    } else {
        Load(t, a);
    }

    UMULLO(r[0], t[0], b);
    UMULLO(r[1], t[1], b);
    MADDO(r[1], t[0], b, r[1]);
    UMULLO(r[2], t[2], b);
    MADDC(r[2], t[1], b, r[2]);
    UMULLO(r[3], t[3], b);
    MADDC(r[3], t[2], b, r[3]);
    UMULLO(r[4], t[4], b);
    MADDC(r[4], t[3], b, r[4]);
    MADDS(carry, t[4], b, 0ULL);

    return carry;

}

// ---------------------------------------------------------------------------------------

__device__ void _MulP(uint64_t *r, uint64_t a)
{

    uint64_t ah;
    uint64_t al;

    UMULLO(al, a, 0x1000003D1ULL);
    UMULHI(ah, a, 0x1000003D1ULL);

    USUBO(r[0], 0ULL, al);
    USUBC(r[1], 0ULL, ah);
    USUBC(r[2], 0ULL, 0ULL);
    USUBC(r[3], 0ULL, 0ULL);
    USUB(r[4], a, 0ULL);

}

// ---------------------------------------------------------------------------------------

__device__ void _ModNeg256(uint64_t *r, uint64_t *a)
{

    uint64_t t[4];
    USUBO(t[0], 0ULL, a[0]);
    USUBC(t[1], 0ULL, a[1]);
    USUBC(t[2], 0ULL, a[2]);
    USUBC(t[3], 0ULL, a[3]);
    UADDO(r[0], t[0], 0xFFFFFFFEFFFFFC2FULL);
    UADDC(r[1], t[1], 0xFFFFFFFFFFFFFFFFULL);
    UADDC(r[2], t[2], 0xFFFFFFFFFFFFFFFFULL);
    UADD(r[3], t[3], 0xFFFFFFFFFFFFFFFFULL);

}

// ---------------------------------------------------------------------------------------

__device__ void _ModNeg256(uint64_t *r)
{

    uint64_t t[4];
    USUBO(t[0], 0ULL, r[0]);
    USUBC(t[1], 0ULL, r[1]);
    USUBC(t[2], 0ULL, r[2]);
    USUBC(t[3], 0ULL, r[3]);
    UADDO(r[0], t[0], 0xFFFFFFFEFFFFFC2FULL);
    UADDC(r[1], t[1], 0xFFFFFFFFFFFFFFFFULL);
    UADDC(r[2], t[2], 0xFFFFFFFFFFFFFFFFULL);
    UADD(r[3], t[3], 0xFFFFFFFFFFFFFFFFULL);

}

__device__ __forceinline__ void _ModAdd256(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t r0,r1,r2,r3;
    asm("{\n.reg .u64 t0,t1,t2,t3,t4,s0,s1,s2,s3,d0,d1,d2,d3,d4,k;\n.reg .pred choose;\nadd.cc.u64 s0,%4,%8;\naddc.cc.u64 s1,%5,%9; addc.cc.u64 s2,%6,%10; addc.cc.u64 s3,%7,%11;\naddc.u64 t4,0,0;\nsub.cc.u64 t0,s0,0xFFFFFFFEFFFFFC2F;\nsubc.cc.u64 t1,s1,0xFFFFFFFFFFFFFFFF;\nsubc.cc.u64 t2,s2,0xFFFFFFFFFFFFFFFF;\nsubc.cc.u64 t3,s3,0xFFFFFFFFFFFFFFFF;\nsubc.u64 t4,t4,0;\nsetp.ge.s64 choose,t4,0;\nselp.u64 %0,t0,s0,choose; selp.u64 %1,t1,s1,choose;\nselp.u64 %2,t2,s2,choose; selp.u64 %3,t3,s3,choose;\n}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}

#ifndef QSB_LAZY
#define QSB_LAZY 1
#endif
#if QSB_LAZY
// r = a - b, plus p when the subtraction borrows. Adding p modulo 2^256 is
// the same as subtracting K = 2^256 - p = 2^32 + 977, so the borrow mask only
// has to select one limb-sized constant instead of four limbs of p. The
// result is bit-identical to the original formulation.
__device__ __forceinline__ void _ModSub256(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t r0,r1,r2,r3;
    asm("{\n.reg .u64 t0,t1,t2,t3,t4,s0,s1,s2,s3,d0,d1,d2,d3,d4,k;\n.reg .pred choose;\nsub.cc.u64 t0,%4,%8;\nsubc.cc.u64 t1,%5,%9; subc.cc.u64 t2,%6,%10; subc.cc.u64 t3,%7,%11;\nsubc.u64 k,0,0; and.b64 k,k,0x1000003D1;\nsub.cc.u64 t0,t0,k; subc.cc.u64 t1,t1,0; subc.cc.u64 t2,t2,0;\nsubc.u64 t3,t3,0;\nmov.u64 %0,t0; mov.u64 %1,t1; mov.u64 %2,t2; mov.u64 %3,t3;\n}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}

__device__ __forceinline__ void _ModSub256(uint64_t *r,uint64_t *b) { _ModSub256(r,r,b); }

// Lazy add: r = a + b reduced only by folding the 2^256 carry as K. The
// result is in [0, 2^256) and congruent mod p, which every consumer in the
// fixed-base chain (_ModMultCore, _ModSqr, _ModSub256, this routine) accepts.
// The fold can carry again only when r >= 2^256 - K after a carry, a 2^-223
// event for field-random inputs, ignored like the existing 2^-224 exposure
// of _ModSub256 to inputs above p.
__device__ __forceinline__ void _ModAddLazy(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t r0,r1,r2,r3;
    asm("{\n.reg .u64 t0,t1,t2,t3,t4,s0,s1,s2,s3,d0,d1,d2,d3,d4,k;\n.reg .pred choose;\nadd.cc.u64 t0,%4,%8;\naddc.cc.u64 t1,%5,%9; addc.cc.u64 t2,%6,%10; addc.cc.u64 t3,%7,%11;\naddc.u64 k,0,0; neg.s64 k,k; and.b64 k,k,0x1000003D1;\nadd.cc.u64 t0,t0,k; addc.cc.u64 t1,t1,0; addc.cc.u64 t2,t2,0;\naddc.u64 t3,t3,0;\nmov.u64 %0,t0; mov.u64 %1,t1; mov.u64 %2,t2; mov.u64 %3,t3;\n}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}

// Fused X3 = a + b - 2c (mod p) for the XYZZ addition (R^2 + PPP - 2V), in
// one carry chain: t = a + b + 2p - 2c lies in [0, 2^258) (the 2^-224 case
// c > p + (a+b)/2 is ignored), and its top two bits h fold as h*K. A second
// carry needs t mod 2^256 >= 2^256 - 2^34, a 2^-222 event, ignored.
__device__ __forceinline__ void _ModX3Fused(uint64_t *r, const uint64_t *a, const uint64_t *b, const uint64_t *c) {
    uint64_t r0,r1,r2,r3;
    asm("{\n.reg .u64 t0,t1,t2,t3,t4,s0,s1,s2,s3,d0,d1,d2,d3,d4,k;\n.reg .pred choose;\nadd.cc.u64 t0,%4,%8;\naddc.cc.u64 t1,%5,%9; addc.cc.u64 t2,%6,%10; addc.cc.u64 t3,%7,%11;\naddc.u64 t4,0,0;\nadd.cc.u64 t0,t0,0xFFFFFFFDFFFFF85E;\naddc.cc.u64 t1,t1,0xFFFFFFFFFFFFFFFF;\naddc.cc.u64 t2,t2,0xFFFFFFFFFFFFFFFF;\naddc.cc.u64 t3,t3,0xFFFFFFFFFFFFFFFF; addc.u64 t4,t4,1;\nshl.b64 d0,%12,1;\nshl.b64 d1,%13,1; shr.u64 k,%12,63; or.b64 d1,d1,k;\nshl.b64 d2,%14,1; shr.u64 k,%13,63; or.b64 d2,d2,k;\nshl.b64 d3,%15,1; shr.u64 k,%14,63; or.b64 d3,d3,k;\nshr.u64 d4,%15,63;\nsub.cc.u64 t0,t0,d0; subc.cc.u64 t1,t1,d1;\nsubc.cc.u64 t2,t2,d2; subc.cc.u64 t3,t3,d3; subc.u64 t4,t4,d4;\nmul.lo.u64 k,t4,0x1000003D1;\nadd.cc.u64 t0,t0,k; addc.cc.u64 t1,t1,0; addc.cc.u64 t2,t2,0;\naddc.u64 t3,t3,0;\nmov.u64 %0,t0; mov.u64 %1,t1; mov.u64 %2,t2; mov.u64 %3,t3;\n}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]),"l"(c[0]),"l"(c[1]),"l"(c[2]),"l"(c[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}
#else
__device__ void _ModSub256(uint64_t *r, uint64_t *a, uint64_t *b)
{
    uint64_t t;
    uint64_t T[4];

    USUBO(r[0], a[0], b[0]);
    USUBC(r[1], a[1], b[1]);
    USUBC(r[2], a[2], b[2]);
    USUBC(r[3], a[3], b[3]);
    USUB(t, 0ULL, 0ULL);

    T[0] = 0xFFFFFFFEFFFFFC2FULL & t;
    T[1] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[2] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[3] = 0xFFFFFFFFFFFFFFFFULL & t;

    UADDO1(r[0], T[0]);
    UADDC1(r[1], T[1]);
    UADDC1(r[2], T[2]);
    UADD1(r[3], T[3]);

}

// ---------------------------------------------------------------------------------------

__device__ void _ModSub256(uint64_t *r, uint64_t *b)
{

    uint64_t t;
    uint64_t T[4];
    USUBO(r[0], r[0], b[0]);
    USUBC(r[1], r[1], b[1]);
    USUBC(r[2], r[2], b[2]);
    USUBC(r[3], r[3], b[3]);
    USUB(t, 0ULL, 0ULL);
    T[0] = 0xFFFFFFFEFFFFFC2FULL & t;
    T[1] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[2] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[3] = 0xFFFFFFFFFFFFFFFFULL & t;
    UADDO1(r[0], T[0]);
    UADDC1(r[1], T[1]);
    UADDC1(r[2], T[2]);
    UADD1(r[3], T[3]);

}
#endif

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t _CTZ(uint64_t x)
{
    uint32_t n;
    asm("{\n\t"
        " .reg .u64 tmp;\n\t"
        " brev.b64 tmp, %1;\n\t"
        " clz.b64 %0, tmp;\n\t"
        "}"
        : "=r"(n) : "l"(x));
    return n;
}

// ---------------------------------------------------------------------------------------
#define SWAP(tmp,x,y) tmp = x; x = y; y = tmp;
#define MSK62 0x3FFFFFFFFFFFFFFF

__device__ void _DivStep62(uint64_t u[5], uint64_t v[5],
                           int32_t *pos,
                           int64_t *uu, int64_t *uv,
                           int64_t *vu, int64_t *vv)
{


    // u' = (uu*u + uv*v) >> bitCount
    // v' = (vu*u + vv*v) >> bitCount
    // Do not maintain a matrix for r and s, the number of
    // 'added P' can be easily calculated

    *uu = 1; *uv = 0;
    *vu = 0; *vv = 1;

    uint32_t bitCount = 62;
    uint32_t zeros;
    uint64_t u0 = u[0];
    uint64_t v0 = v[0];

    // Extract 64 MSB of u and v
    // u and v must be positive
    uint64_t uh, vh;
    int64_t w, x, y, z;
    bitCount = 62;

    while (*pos > 0 && (u[*pos] | v[*pos]) == 0)
        (*pos)--;
    if (*pos == 0) {

        uh = u[0];
        vh = v[0];

    } else {

        uint32_t s = __clzll(u[*pos] | v[*pos]);
        if (s == 0) {
            uh = u[*pos];
            vh = v[*pos];
        } else {
            uh = __sleft128(u[*pos - 1], u[*pos], s);
            vh = __sleft128(v[*pos - 1], v[*pos], s);
        }

    }


    while (true) {

        // Use a sentinel bit to count zeros only up to bitCount
        zeros = _CTZ(v0 | (1ULL << bitCount));

        v0 >>= zeros;
        vh >>= zeros;
        *uu <<= zeros;
        *uv <<= zeros;
        bitCount -= zeros;

        if (bitCount == 0)
            break;

        if (vh < uh) {
            SWAP(w, uh, vh);
            SWAP(x, u0, v0);
            SWAP(y, *uu, *vu);
            SWAP(z, *uv, *vv);
        }

        vh -= uh;
        v0 -= u0;
        *vv -= *uv;
        *vu -= *uu;

    }

}

__device__ void _MatrixVecMulHalf(uint64_t dest[5], uint64_t u[5], uint64_t v[5], int64_t _11, int64_t _12, uint64_t *carry)
{

    uint64_t t1[NBBLOCK];
    uint64_t t2[NBBLOCK];
    uint64_t c1, c2;

    c1 = _IMultC(t1, u, _11);
    c2 = _IMultC(t2, v, _12);

    UADDO(dest[0], t1[0], t2[0]);
    UADDC(dest[1], t1[1], t2[1]);
    UADDC(dest[2], t1[2], t2[2]);
    UADDC(dest[3], t1[3], t2[3]);
    UADDC(dest[4], t1[4], t2[4]);
    UADD(*carry, c1, c2);

}

__device__ void _MatrixVecMul(uint64_t u[5], uint64_t v[5], int64_t _11, int64_t _12, int64_t _21, int64_t _22)
{

    uint64_t t1[NBBLOCK];
    uint64_t t2[NBBLOCK];
    uint64_t t3[NBBLOCK];
    uint64_t t4[NBBLOCK];

    _IMult(t1, u, _11);
    _IMult(t2, v, _12);
    _IMult(t3, u, _21);
    _IMult(t4, v, _22);

    UADDO(u[0], t1[0], t2[0]);
    UADDC(u[1], t1[1], t2[1]);
    UADDC(u[2], t1[2], t2[2]);
    UADDC(u[3], t1[3], t2[3]);
    UADD(u[4], t1[4], t2[4]);

    UADDO(v[0], t3[0], t4[0]);
    UADDC(v[1], t3[1], t4[1]);
    UADDC(v[2], t3[2], t4[2]);
    UADDC(v[3], t3[3], t4[3]);
    UADD(v[4], t3[4], t4[4]);

}

__device__ uint64_t _AddCh(uint64_t r[5], uint64_t a[5], uint64_t carry)
{

    uint64_t carryOut;

    UADDO1(r[0], a[0]);
    UADDC1(r[1], a[1]);
    UADDC1(r[2], a[2]);
    UADDC1(r[3], a[3]);
    UADDC1(r[4], a[4]);
    UADD(carryOut, carry, 0ULL);

    return carryOut;

}

__device__ __noinline__ void _ModInv(uint64_t *R)
{

    // Compute modular inverse of R mod P (using 320bits signed integer)
    // 0 < this < P  , P must be odd
    // Return 0 if no inverse
    // See IntMod.cpp for more info.

    uint64_t u[NBBLOCK];
    uint64_t v[NBBLOCK];
    uint64_t r[NBBLOCK];
    uint64_t s[NBBLOCK];
    uint64_t tr[NBBLOCK];
    uint64_t ts[NBBLOCK];
    uint64_t r0[NBBLOCK];
    uint64_t s0[NBBLOCK];

    int64_t  uu;
    int64_t  uv;
    int64_t  vu;
    int64_t  vv;

    uint64_t mr0;
    uint64_t ms0;

    uint64_t carryR;
    uint64_t carryS;

    int32_t  pos = NBBLOCK - 1;

    u[0] = 0xFFFFFFFEFFFFFC2F;
    u[1] = 0xFFFFFFFFFFFFFFFF;
    u[2] = 0xFFFFFFFFFFFFFFFF;
    u[3] = 0xFFFFFFFFFFFFFFFF;
    u[4] = 0;
    Load(v, R);
    r[0] = 0; s[0] = 1;
    r[1] = 0; s[1] = 0;
    r[2] = 0; s[2] = 0;
    r[3] = 0; s[3] = 0;
    r[4] = 0; s[4] = 0;

    // Delayed right shift 62bits

    // DivStep loop -------------------------------

    while (true) {

        _DivStep62(u, v, &pos, &uu, &uv, &vu, &vv);

        _MatrixVecMul(u, v, uu, uv, vu, vv);

        if (_IsNegative(u)) {
            Neg(u);
            uu = -uu;
            uv = -uv;
        }
        if (_IsNegative(v)) {
            Neg(v);
            vu = -vu;
            vv = -vv;
        }

        _ShiftR62(u);
        _ShiftR62(v);

        // Update r
        _MatrixVecMulHalf(tr, r, s, uu, uv, &carryR);
        mr0 = (tr[0] * MM64) & MSK62;
        _MulP(r0, mr0);
        carryR = _AddCh(tr, r0, carryR);

        if (_IsZero(v)) {

            _ShiftR62(r, tr, carryR);
            break;

        } else {

            // Update s
            _MatrixVecMulHalf(ts, r, s, vu, vv, &carryS);
            ms0 = (ts[0] * MM64) & MSK62;
            _MulP(s0, ms0);
            carryS = _AddCh(ts, s0, carryS);

        }

        _ShiftR62(r, tr, carryR);
        _ShiftR62(s, ts, carryS);

    }

    // u ends with gcd
    if (!_IsOne(u)) {
        // No inverse
        R[0] = 0ULL;
        R[1] = 0ULL;
        R[2] = 0ULL;
        R[3] = 0ULL;
        R[4] = 0ULL;
        return;
    }

    while (_IsNegative(r))
        AddP(r);
    while (!_IsNegative(r))
        SubP(r);
    AddP(r);

    Load(R, r);

}

// ---------------------------------------------------------------------------------------
// secp256k1 field multiply r = a*b mod p, 8x32-bit even/odd column product chains fused
// into IMAD.WIDE.U32[.X], then the sparse double-fold R = L + H*0x1000003D1 (2^256 = 2^32+977
// mod p) applied twice. Output in [0,2^256) (not necessarily < p); the final 2^256 carry is
// dropped -- the SAME convention and the same accepted non-canonical edge behaviour on inputs
// in [p,2^256) as the UMult-based routine this replaces (verified: both agree with the Python
// reference for every input < p, and both are identically non-canonical for the ~2^-224 inputs
// >= p, which never occur in a run).
//
// Device (__CUDA_ARCH__): one non-volatile asm block -- the njuffa/mm32 schedule measured at
// 124 SASS / 73 IMAD.WIDE on Compiler Explorer nvcc 12.9.1 sm_89. %4..%7 = a limbs (LE),
// %8..%11 = b limbs, %0..%3 = result limbs.
// Host (CPU verification): a __uint128_t transcription of the IDENTICAL schedule -- even chain
// e0..e7, odd chain o0..o6, merge to 16 u32 limbs, then the same double fold. Validated against
// harness/crypto.py; the asm<->C line correspondence is documented in
// notes/ce-tools/mm32-cref-map.md so the two cannot silently drift.
__device__ __forceinline__ void _ModMultCore(uint64_t *r, const uint64_t *a, const uint64_t *b)
{
#ifdef __CUDA_ARCH__
    uint64_t r0,r1,r2,r3;
    asm("{\n\t.reg .u32 a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7;\n\t.reg .u64 e0,e1,e2,e3,e4,e5,e6,e7,o0,o1,o2,o3,o4,o5,o6,t,lc;\n\t.reg .u32 cy,o15;\n\t.reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n\t.reg .u32 y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14;\n\tmov.b64 {a0,a1}, %4;\n\tmov.b64 {a2,a3}, %5;\n\tmov.b64 {a4,a5}, %6;\n\tmov.b64 {a6,a7}, %7;\n\tmov.b64 {b0,b1}, %8;\n\tmov.b64 {b2,b3}, %9;\n\tmov.b64 {b4,b5}, %10;\n\tmov.b64 {b6,b7}, %11;\n\t.reg .u64 odd_t,odd_lc; .reg .u32 odd_cy;\nmul.wide.u32 e0, a0, b0;\nmul.wide.u32 o0, a0, b1;\nmul.wide.u32 e1, a0, b2;\nmul.wide.u32 o1, a0, b3;\nmul.wide.u32 e2, a0, b4;\nmul.wide.u32 o2, a0, b5;\nmul.wide.u32 e3, a0, b6;\nmul.wide.u32 o3, a0, b7;\nmul.wide.u32 t, a1, b1;\nmul.wide.u32 odd_t, a1, b0;\nadd.cc.u64 e1, e1, t;\nmul.wide.u32 t, a1, b3;\naddc.cc.u64 e2, e2, t;\nmul.wide.u32 t, a1, b5;\naddc.cc.u64 e3, e3, t;\nmul.wide.u32 t, a1, b7;\naddc.u64 e4, t, 0;\nadd.cc.u64 o0, o0, odd_t;\nmul.wide.u32 odd_t, a1, b2;\naddc.cc.u64 o1, o1, odd_t;\nmul.wide.u32 odd_t, a1, b4;\naddc.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a1, b6;\naddc.cc.u64 o3, o3, odd_t;\naddc.u32 odd_cy, 0, 0;\nmul.wide.u32 t, a2, b0;\ncvt.u64.u32 odd_lc, odd_cy;\nadd.cc.u64 e1, e1, t;\nmul.wide.u32 t, a2, b2;\naddc.cc.u64 e2, e2, t;\nmul.wide.u32 t, a2, b4;\naddc.cc.u64 e3, e3, t;\nmul.wide.u32 t, a2, b6;\naddc.cc.u64 e4, e4, t;\naddc.u32 cy, 0, 0;\nmul.wide.u32 odd_t, a2, b1;\ncvt.u64.u32 lc, cy;\nadd.cc.u64 o1, o1, odd_t;\nmul.wide.u32 odd_t, a2, b3;\naddc.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a2, b5;\naddc.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a2, b7;\naddc.u64 o4, odd_t, odd_lc;\nmul.wide.u32 t, a3, b1;\nmul.wide.u32 odd_t, a3, b0;\nadd.cc.u64 e2, e2, t;\nmul.wide.u32 t, a3, b3;\naddc.cc.u64 e3, e3, t;\nmul.wide.u32 t, a3, b5;\naddc.cc.u64 e4, e4, t;\nmul.wide.u32 t, a3, b7;\naddc.u64 e5, t, lc;\nadd.cc.u64 o1, o1, odd_t;\nmul.wide.u32 odd_t, a3, b2;\naddc.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a3, b4;\naddc.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a3, b6;\naddc.cc.u64 o4, o4, odd_t;\naddc.u32 odd_cy, 0, 0;\nmul.wide.u32 t, a4, b0;\ncvt.u64.u32 odd_lc, odd_cy;\nadd.cc.u64 e2, e2, t;\nmul.wide.u32 t, a4, b2;\naddc.cc.u64 e3, e3, t;\nmul.wide.u32 t, a4, b4;\naddc.cc.u64 e4, e4, t;\nmul.wide.u32 t, a4, b6;\naddc.cc.u64 e5, e5, t;\naddc.u32 cy, 0, 0;\nmul.wide.u32 odd_t, a4, b1;\ncvt.u64.u32 lc, cy;\nadd.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a4, b3;\naddc.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a4, b5;\naddc.cc.u64 o4, o4, odd_t;\nmul.wide.u32 odd_t, a4, b7;\naddc.u64 o5, odd_t, odd_lc;\nmul.wide.u32 t, a5, b1;\nmul.wide.u32 odd_t, a5, b0;\nadd.cc.u64 e3, e3, t;\nmul.wide.u32 t, a5, b3;\naddc.cc.u64 e4, e4, t;\nmul.wide.u32 t, a5, b5;\naddc.cc.u64 e5, e5, t;\nmul.wide.u32 t, a5, b7;\naddc.u64 e6, t, lc;\nadd.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a5, b2;\naddc.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a5, b4;\naddc.cc.u64 o4, o4, odd_t;\nmul.wide.u32 odd_t, a5, b6;\naddc.cc.u64 o5, o5, odd_t;\naddc.u32 odd_cy, 0, 0;\nmul.wide.u32 t, a6, b0;\ncvt.u64.u32 odd_lc, odd_cy;\nadd.cc.u64 e3, e3, t;\nmul.wide.u32 t, a6, b2;\naddc.cc.u64 e4, e4, t;\nmul.wide.u32 t, a6, b4;\naddc.cc.u64 e5, e5, t;\nmul.wide.u32 t, a6, b6;\naddc.cc.u64 e6, e6, t;\naddc.u32 cy, 0, 0;\nmul.wide.u32 odd_t, a6, b1;\ncvt.u64.u32 lc, cy;\nadd.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a6, b3;\naddc.cc.u64 o4, o4, odd_t;\nmul.wide.u32 odd_t, a6, b5;\naddc.cc.u64 o5, o5, odd_t;\nmul.wide.u32 odd_t, a6, b7;\naddc.u64 o6, odd_t, odd_lc;\nmul.wide.u32 t, a7, b1;\nmul.wide.u32 odd_t, a7, b0;\nadd.cc.u64 e4, e4, t;\nmul.wide.u32 t, a7, b3;\naddc.cc.u64 e5, e5, t;\nmul.wide.u32 t, a7, b5;\naddc.cc.u64 e6, e6, t;\nmul.wide.u32 t, a7, b7;\naddc.u64 e7, t, lc;\nadd.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a7, b2;\naddc.cc.u64 o4, o4, odd_t;\nmul.wide.u32 odd_t, a7, b4;\naddc.cc.u64 o5, o5, odd_t;\nmul.wide.u32 odd_t, a7, b6;\naddc.cc.u64 o6, o6, odd_t;\naddc.u32 o15, 0, 0;\nmov.b64 {x0,x1}, e0;\n\tmov.b64 {x2,x3}, e1;\n\tmov.b64 {x4,x5}, e2;\n\tmov.b64 {x6,x7}, e3;\n\tmov.b64 {x8,x9}, e4;\n\tmov.b64 {x10,x11}, e5;\n\tmov.b64 {x12,x13}, e6;\n\tmov.b64 {x14,x15}, e7;\n\tmov.b64 {y1,y2}, o0;\n\tmov.b64 {y3,y4}, o1;\n\tmov.b64 {y5,y6}, o2;\n\tmov.b64 {y7,y8}, o3;\n\tmov.b64 {y9,y10}, o4;\n\tmov.b64 {y11,y12}, o5;\n\tmov.b64 {y13,y14}, o6;\n\tadd.cc.u32 x1, x1, y1;\n\taddc.cc.u32 x2, x2, y2;\n\taddc.cc.u32 x3, x3, y3;\n\taddc.cc.u32 x4, x4, y4;\n\taddc.cc.u32 x5, x5, y5;\n\taddc.cc.u32 x6, x6, y6;\n\taddc.cc.u32 x7, x7, y7;\n\taddc.cc.u32 x8, x8, y8;\n\taddc.cc.u32 x9, x9, y9;\n\taddc.cc.u32 x10, x10, y10;\n\taddc.cc.u32 x11, x11, y11;\n\taddc.cc.u32 x12, x12, y12;\n\taddc.cc.u32 x13, x13, y13;\n\taddc.cc.u32 x14, x14, y14;\n\taddc.u32 x15, x15, o15;\n\t.reg .u64 r0,r1,r2,r3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;\n\t.reg .u32 f8,g8,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,w0,w1,w2,w3,w4,w5,w6,w7,m0,m1,m2;\n\t.reg .pred mm_tail_skip;\n\tmov.b64 r0, {x0,x1}; mov.b64 r1, {x2,x3}; mov.b64 r2, {x4,x5}; mov.b64 r3, {x6,x7};\n\tmov.b64 h0, {x8,x9}; mov.b64 h1, {x10,x11}; mov.b64 h2, {x12,x13}; mov.b64 h3, {x14,x15};\n\tmul.wide.u32 t, x8, 977;  add.cc.u64  f0, r0, t;\n\tmul.wide.u32 t, x10, 977; addc.cc.u64 f1, r1, t;\n\tmul.wide.u32 t, x12, 977; addc.cc.u64 f2, r2, t;\n\tmul.wide.u32 t, x14, 977; addc.cc.u64 f3, r3, t;\n\taddc.u32 f8, 0, 0;\n\tmul.wide.u32 t, x9, 977;  add.cc.u64  g0, h0, t;\n\tmul.wide.u32 t, x11, 977; addc.cc.u64 g1, h1, t;\n\tmul.wide.u32 t, x13, 977; addc.cc.u64 g2, h2, t;\n\tmul.wide.u32 t, x15, 977; addc.cc.u64 g3, h3, t;\n\taddc.u32 g8, 0, 0;\n\tmov.b64 {z0,z1}, f0;\n\tmov.b64 {z2,z3}, f1;\n\tmov.b64 {z4,z5}, f2;\n\tmov.b64 {z6,z7}, f3;\n\tmov.b64 {w0,w1}, g0;\n\tmov.b64 {w2,w3}, g1;\n\tmov.b64 {w4,w5}, g2;\n\tmov.b64 {w6,w7}, g3;\n\tadd.cc.u32  z1, z1, w0;\n\taddc.cc.u32 z2, z2, w1;\n\taddc.cc.u32 z3, z3, w2;\n\taddc.cc.u32 z4, z4, w3;\n\taddc.cc.u32 z5, z5, w4;\n\taddc.cc.u32 z6, z6, w5;\n\taddc.cc.u32 z7, z7, w6;\n\taddc.cc.u32 z8, f8, w7;\n\taddc.u32    z9, g8, 0;\n\tmul.wide.u32 t, z8, 977; mov.b64 {m0,m1}, t;\n\tmad.lo.u32 m1, z9, 977, m1;\n\tadd.cc.u32 m1, m1, z8;\n\taddc.u32 m2, z9, 0;\n\tadd.cc.u32 z0, z0, m0; addc.cc.u32 z1, z1, m1; addc.cc.u32 z2, z2, m2;\n\tsetp.gt.u32 mm_tail_skip, z2, 1;\n\t@mm_tail_skip bra mm_tail_done;\n\taddc.cc.u32 z3, z3, 0;\n\taddc.cc.u32 z4, z4, 0;\n\taddc.cc.u32 z5, z5, 0;\n\taddc.cc.u32 z6, z6, 0;\n\taddc.u32 z7, z7, 0;\n\tmm_tail_done:\n\tmov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.b64 %2, {z4,z5}; mov.b64 %3, {z6,z7};\n\t}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]) );
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
      t=(uint64_t)z[7]+c; z[7]=(uint32_t)t; }
    r[0]=z[0]|((uint64_t)z[1]<<32); r[1]=z[2]|((uint64_t)z[3]<<32);
    r[2]=z[4]|((uint64_t)z[5]<<32); r[3]=z[6]|((uint64_t)z[7]<<32);
#undef QSB_MW
#endif
}

// ---------------------------------------------------------------------------------------
// Compute a*b (mod p). Interface unchanged: uint64_t[4] little-endian, inputs < 2^256.
// ---------------------------------------------------------------------------------------
__device__ void _ModMult(uint64_t *r, uint64_t *a, uint64_t *b)
{
    _ModMultCore(r, a, b);
}

__device__ void _ModMult(uint64_t *r, uint64_t *a)
{
    uint64_t bb[4] = { r[0], r[1], r[2], r[3] };
    _ModMultCore(r, a, bb);
}

// ---------------------------------------------------------------------------------------
// Dedicated secp256k1 square r = a^2 mod p. Triangular 8x32 schedule: 28 off-diagonal
// cross products a_i*a_j (even/odd column chains with multi-bit carries), doubled, plus
// 8 diagonal squares, then the same double-fold as _ModMultCore. 45 IMAD.WIDE/square
// (vs 73 for a*a via _ModMultCore). Output convention identical to _ModMultCore:
// [0,2^256), final 2^256 carry dropped. Device: inline PTX; host: __uint128_t C-ref of
// the IDENTICAL schedule. Independently validated (notes/research/sqr_ptx/VALIDATION.md):
// 10^6 random + boundaries vs crypto.py, PTX row-schedule emulation, CE 45 IMAD.WIDE.
__device__ __forceinline__ void _ModSqr(uint64_t r[4], const uint64_t a[4]) {
#ifdef __CUDA_ARCH__
    uint64_t r0, r1, r2, r3;
    asm("{\n\t.reg .u32 a0,a1,a2,a3,a4,a5,a6,a7;\n\t.reg .u64 e2,e4,e6,e8,e10,e12,o1,o3,o5,o7,o9,o11,o13,t;\n\t.reg .u32 ecy,ocy,e14,o15;\n\t.reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n\t.reg .u32 y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14;\n\t.reg .u64 d0,d1,d2,d3,d4,d5,d6,d7;\n\tmov.b64 {a0,a1}, %4;\n\tmov.b64 {a2,a3}, %5;\n\tmov.b64 {a4,a5}, %6;\n\tmov.b64 {a6,a7}, %7;\n\t.reg .u64 odd_t;\nmul.wide.u32 e6, a2, a4;\nmul.wide.u32 o7, a3, a4;\nmul.wide.u32 e8, a3, a5;\nmul.wide.u32 o5, a2, a3;\nmul.wide.u32 e4, a1, a3;\nmul.wide.u32 odd_t, a2, a5;\nmul.wide.u32 t, a1, a5;\nadd.cc.u64 o7, o7, odd_t;\nmul.wide.u32 odd_t, a4, a5;\naddc.cc.u64 o9, odd_t, 0;\naddc.u32 ocy, 0, 0;\nadd.cc.u64 e6, e6, t;\nmul.wide.u32 t, a2, a6;\naddc.cc.u64 e8, e8, t;\nmul.wide.u32 t, a4, a6;\naddc.cc.u64 e10, t, 0;\naddc.u32 ecy, 0, 0;\nmov.b64 o11, {ocy,0};\nmov.b64 e12, {ecy,0};\nmul.wide.u32 o3, a1, a2;\nmul.wide.u32 e2, a0, a2;\nmul.wide.u32 odd_t, a1, a4;\nmul.wide.u32 t, a0, a4;\nadd.cc.u64 o5, o5, odd_t;\nmul.wide.u32 odd_t, a1, a6;\naddc.cc.u64 o7, o7, odd_t;\nmul.wide.u32 odd_t, a3, a6;\naddc.cc.u64 o9, o9, odd_t;\nmul.wide.u32 odd_t, a5, a6;\naddc.cc.u64 o11, o11, odd_t;\naddc.u32 ocy, 0, 0;\nadd.cc.u64 e4, e4, t;\nmul.wide.u32 t, a0, a6;\naddc.cc.u64 e6, e6, t;\nmul.wide.u32 t, a1, a7;\naddc.cc.u64 e8, e8, t;\nmul.wide.u32 t, a3, a7;\naddc.cc.u64 e10, e10, t;\nmul.wide.u32 t, a5, a7;\naddc.cc.u64 e12, e12, t;\naddc.u32 e14, 0, 0;\nmov.b64 o13, {ocy,0};\nmul.wide.u32 o1, a0, a1;\nmul.wide.u32 odd_t, a0, a3;\nadd.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a0, a5;\naddc.cc.u64 o5, o5, odd_t;\nmul.wide.u32 odd_t, a0, a7;\naddc.cc.u64 o7, o7, odd_t;\nmul.wide.u32 odd_t, a2, a7;\naddc.cc.u64 o9, o9, odd_t;\nmul.wide.u32 odd_t, a4, a7;\naddc.cc.u64 o11, o11, odd_t;\nmul.wide.u32 odd_t, a6, a7;\naddc.cc.u64 o13, o13, odd_t;\naddc.u32 o15, 0, 0;\nmov.u32 x0, 0;\n\tmov.b64 {x2,x3}, e2; mov.b64 {x4,x5}, e4; mov.b64 {x6,x7}, e6;\n\tmov.b64 {x8,x9}, e8; mov.b64 {x10,x11}, e10; mov.b64 {x12,x13}, e12;\n\tmov.u32 x14, e14; mov.u32 x15, 0;\n\tmov.b64 {x1,y2}, o1; mov.b64 {y3,y4}, o3; mov.b64 {y5,y6}, o5;\n\tmov.b64 {y7,y8}, o7; mov.b64 {y9,y10}, o9; mov.b64 {y11,y12}, o11; mov.b64 {y13,y14}, o13;\n\tadd.cc.u32 x2, x2, y2; addc.cc.u32 x3, x3, y3;\n\taddc.cc.u32 x4, x4, y4; addc.cc.u32 x5, x5, y5; addc.cc.u32 x6, x6, y6;\n\taddc.cc.u32 x7, x7, y7; addc.cc.u32 x8, x8, y8; addc.cc.u32 x9, x9, y9;\n\taddc.cc.u32 x10, x10, y10; addc.cc.u32 x11, x11, y11; addc.cc.u32 x12, x12, y12;\n\taddc.cc.u32 x13, x13, y13; addc.cc.u32 x14, x14, y14; addc.u32 x15, x15, o15;\n\tshf.l.wrap.b32 x15, x14, x15, 1; shf.l.wrap.b32 x14, x13, x14, 1;\n\tshf.l.wrap.b32 x13, x12, x13, 1; shf.l.wrap.b32 x12, x11, x12, 1;\n\tshf.l.wrap.b32 x11, x10, x11, 1; shf.l.wrap.b32 x10, x9, x10, 1;\n\tshf.l.wrap.b32 x9, x8, x9, 1; shf.l.wrap.b32 x8, x7, x8, 1;\n\tshf.l.wrap.b32 x7, x6, x7, 1; shf.l.wrap.b32 x6, x5, x6, 1;\n\tshf.l.wrap.b32 x5, x4, x5, 1; shf.l.wrap.b32 x4, x3, x4, 1;\n\tshf.l.wrap.b32 x3, x2, x3, 1; shf.l.wrap.b32 x2, x1, x2, 1;\n\tshf.l.wrap.b32 x1, x0, x1, 1;\n\tmov.b64 d0, {x0,x1}; mov.b64 d1, {x2,x3}; mov.b64 d2, {x4,x5}; mov.b64 d3, {x6,x7};\n\tmov.b64 d4, {x8,x9}; mov.b64 d5, {x10,x11}; mov.b64 d6, {x12,x13}; mov.b64 d7, {x14,x15};\n\tmul.wide.u32 t, a0, a0; add.cc.u64 d0, d0, t;\n\tmul.wide.u32 t, a1, a1; addc.cc.u64 d1, d1, t;\n\tmul.wide.u32 t, a2, a2; addc.cc.u64 d2, d2, t;\n\tmul.wide.u32 t, a3, a3; addc.cc.u64 d3, d3, t;\n\tmul.wide.u32 t, a4, a4; addc.cc.u64 d4, d4, t;\n\tmul.wide.u32 t, a5, a5; addc.cc.u64 d5, d5, t;\n\tmul.wide.u32 t, a6, a6; addc.cc.u64 d6, d6, t;\n\tmul.wide.u32 t, a7, a7; addc.u64 d7, d7, t;\n\tmov.b64 {x0,x1}, d0; mov.b64 {x2,x3}, d1; mov.b64 {x4,x5}, d2; mov.b64 {x6,x7}, d3;\n\tmov.b64 {x8,x9}, d4; mov.b64 {x10,x11}, d5; mov.b64 {x12,x13}, d6; mov.b64 {x14,x15}, d7;\n\t.reg .u64 fr0,fr1,fr2,fr3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;\n\t.reg .u32 f8,g8,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,w0,w1,w2,w3,w4,w5,w6,w7,m0,m1,m2;\n\tmov.b64 fr0, {x0,x1}; mov.b64 fr1, {x2,x3}; mov.b64 fr2, {x4,x5}; mov.b64 fr3, {x6,x7};\n\tmov.b64 h0, {x8,x9}; mov.b64 h1, {x10,x11}; mov.b64 h2, {x12,x13}; mov.b64 h3, {x14,x15};\n\tmul.wide.u32 t, x8, 977;  add.cc.u64  f0, fr0, t;\n\tmul.wide.u32 t, x10, 977; addc.cc.u64 f1, fr1, t;\n\tmul.wide.u32 t, x12, 977; addc.cc.u64 f2, fr2, t;\n\tmul.wide.u32 t, x14, 977; addc.cc.u64 f3, fr3, t;\n\taddc.u32 f8, 0, 0;\n\tmul.wide.u32 t, x9, 977;  add.cc.u64  g0, h0, t;\n\tmul.wide.u32 t, x11, 977; addc.cc.u64 g1, h1, t;\n\tmul.wide.u32 t, x13, 977; addc.cc.u64 g2, h2, t;\n\tmul.wide.u32 t, x15, 977; addc.cc.u64 g3, h3, t;\n\taddc.u32 g8, 0, 0;\n\tmov.b64 {z0,z1}, f0; mov.b64 {z2,z3}, f1; mov.b64 {z4,z5}, f2; mov.b64 {z6,z7}, f3;\n\tmov.b64 {w0,w1}, g0; mov.b64 {w2,w3}, g1; mov.b64 {w4,w5}, g2; mov.b64 {w6,w7}, g3;\n\tadd.cc.u32 z1, z1, w0; addc.cc.u32 z2, z2, w1; addc.cc.u32 z3, z3, w2;\n\taddc.cc.u32 z4, z4, w3; addc.cc.u32 z5, z5, w4; addc.cc.u32 z6, z6, w5;\n\taddc.cc.u32 z7, z7, w6; addc.cc.u32 z8, f8, w7; addc.u32 z9, g8, 0;\n\tmul.wide.u32 t, z8, 977; mov.b64 {m0,m1}, t;\n\tmad.lo.u32 m1, z9, 977, m1;\n\tadd.cc.u32 m1, m1, z8; addc.u32 m2, z9, 0;\n\tadd.cc.u32 z0, z0, m0; addc.cc.u32 z1, z1, m1; addc.cc.u32 z2, z2, m2;\n\taddc.cc.u32 z3, z3, 0; addc.cc.u32 z4, z4, 0; addc.cc.u32 z5, z5, 0;\n\taddc.cc.u32 z6, z6, 0; addc.u32 z7, z7, 0;\n\tmov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.b64 %2, {z4,z5}; mov.b64 %3, {z6,z7};\n\t}\n\t"
        : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]));
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
      tv=(uint64_t)z[7]+c; z[7]=(uint32_t)tv; }
    r[0]=z[0]|((uint64_t)z[1]<<32); r[1]=z[2]|((uint64_t)z[3]<<32);
    r[2]=z[4]|((uint64_t)z[5]<<32); r[3]=z[6]|((uint64_t)z[7]<<32);
    #undef MW
    (void)h0;(void)h1;(void)h2;(void)h3;(void)r0;(void)r1;(void)r2;(void)r3;(void)carry;(void)d;
#endif
}

#if QSB_FUSE_SQRADDSUB2
// Experimental genuine fused r*r+e-2q reduction. The dedicated 8x32
// square schedule is unchanged; 3p+e-2q enters its first 320-bit fold before
// the remaining folds. All four pointers may alias, and inputs may be any
// 256-bit representatives. The result is congruent in [0, 2^256).
__device__ __forceinline__ void _ModSqrAddSub2(uint64_t out[4], const uint64_t a[4], const uint64_t e[4], const uint64_t q[4]) {
#ifdef __CUDA_ARCH__
    uint64_t r0, r1, r2, r3;
    asm(
        /* Final correction needs only three 32-bit limbs. With B=2^256
         * and K=2^32+977, the previous fold is below B+K*(K+4).
         * If it carries, low+K < K*(K+5) < 2^65; otherwise the
         * correction is zero. Upper five limbs remain unchanged. */
        "{\n"
        "\t.reg .u32 a0,a1,a2,a3,a4,a5,a6,a7;\n"
        "\t.reg .u64 e2,e4,e6,e8,e10,e12,o1,o3,o5,o7,o9,o11,o13,t;\n"
        "\t.reg .u32 ecy,ocy,e14,o15;\n"
        "\t.reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n"
        "\t.reg .u32 y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14;\n"
        "\t.reg .u64 d0,d1,d2,d3,d4,d5,d6,d7;\n"
        "\tmov.b64 {a0,a1}, %4;\n"
        "\tmov.b64 {a2,a3}, %5;\n"
        "\tmov.b64 {a4,a5}, %6;\n"
        "\tmov.b64 {a6,a7}, %7;\n"
        "\t.reg .u64 odd_t;\n"
        "mul.wide.u32 e6, a2, a4;\n"
        "mul.wide.u32 o7, a3, a4;\n"
        "mul.wide.u32 e8, a3, a5;\n"
        "mul.wide.u32 o5, a2, a3;\n"
        "mul.wide.u32 e4, a1, a3;\n"
        "mul.wide.u32 odd_t, a2, a5;\n"
        "mul.wide.u32 t, a1, a5;\n"
        "add.cc.u64 o7, o7, odd_t;\n"
        "mul.wide.u32 odd_t, a4, a5;\n"
        "addc.cc.u64 o9, odd_t, 0;\n"
        "addc.u32 ocy, 0, 0;\n"
        "add.cc.u64 e6, e6, t;\n"
        "mul.wide.u32 t, a2, a6;\n"
        "addc.cc.u64 e8, e8, t;\n"
        "mul.wide.u32 t, a4, a6;\n"
        "addc.cc.u64 e10, t, 0;\n"
        "addc.u32 ecy, 0, 0;\n"
        "mov.b64 o11, {ocy,0};\n"
        "mov.b64 e12, {ecy,0};\n"
        "mul.wide.u32 o3, a1, a2;\n"
        "mul.wide.u32 e2, a0, a2;\n"
        "mul.wide.u32 odd_t, a1, a4;\n"
        "mul.wide.u32 t, a0, a4;\n"
        "add.cc.u64 o5, o5, odd_t;\n"
        "mul.wide.u32 odd_t, a1, a6;\n"
        "addc.cc.u64 o7, o7, odd_t;\n"
        "mul.wide.u32 odd_t, a3, a6;\n"
        "addc.cc.u64 o9, o9, odd_t;\n"
        "mul.wide.u32 odd_t, a5, a6;\n"
        "addc.cc.u64 o11, o11, odd_t;\n"
        "addc.u32 ocy, 0, 0;\n"
        "add.cc.u64 e4, e4, t;\n"
        "mul.wide.u32 t, a0, a6;\n"
        "addc.cc.u64 e6, e6, t;\n"
        "mul.wide.u32 t, a1, a7;\n"
        "addc.cc.u64 e8, e8, t;\n"
        "mul.wide.u32 t, a3, a7;\n"
        "addc.cc.u64 e10, e10, t;\n"
        "mul.wide.u32 t, a5, a7;\n"
        "addc.cc.u64 e12, e12, t;\n"
        "addc.u32 e14, 0, 0;\n"
        "mov.b64 o13, {ocy,0};\n"
        "mul.wide.u32 o1, a0, a1;\n"
        "mul.wide.u32 odd_t, a0, a3;\n"
        "add.cc.u64 o3, o3, odd_t;\n"
        "mul.wide.u32 odd_t, a0, a5;\n"
        "addc.cc.u64 o5, o5, odd_t;\n"
        "mul.wide.u32 odd_t, a0, a7;\n"
        "addc.cc.u64 o7, o7, odd_t;\n"
        "mul.wide.u32 odd_t, a2, a7;\n"
        "addc.cc.u64 o9, o9, odd_t;\n"
        "mul.wide.u32 odd_t, a4, a7;\n"
        "addc.cc.u64 o11, o11, odd_t;\n"
        "mul.wide.u32 odd_t, a6, a7;\n"
        "addc.cc.u64 o13, o13, odd_t;\n"
        "addc.u32 o15, 0, 0;\n"
        "mov.u32 x0, 0;\n"
        "\tmov.b64 {x2,x3}, e2; mov.b64 {x4,x5}, e4; mov.b64 {x6,x7}, e6;\n"
        "\tmov.b64 {x8,x9}, e8; mov.b64 {x10,x11}, e10; mov.b64 {x12,x13}, e12;\n"
        "\tmov.u32 x14, e14; mov.u32 x15, 0;\n"
        "\tmov.b64 {x1,y2}, o1; mov.b64 {y3,y4}, o3; mov.b64 {y5,y6}, o5;\n"
        "\tmov.b64 {y7,y8}, o7; mov.b64 {y9,y10}, o9; mov.b64 {y11,y12}, o11; mov.b64 {y13,y14}, o13;\n"
        "\tadd.cc.u32 x2, x2, y2; addc.cc.u32 x3, x3, y3;\n"
        "\taddc.cc.u32 x4, x4, y4; addc.cc.u32 x5, x5, y5; addc.cc.u32 x6, x6, y6;\n"
        "\taddc.cc.u32 x7, x7, y7; addc.cc.u32 x8, x8, y8; addc.cc.u32 x9, x9, y9;\n"
        "\taddc.cc.u32 x10, x10, y10; addc.cc.u32 x11, x11, y11; addc.cc.u32 x12, x12, y12;\n"
        "\taddc.cc.u32 x13, x13, y13; addc.cc.u32 x14, x14, y14; addc.u32 x15, x15, o15;\n"
        "\tshf.l.wrap.b32 x15, x14, x15, 1; shf.l.wrap.b32 x14, x13, x14, 1;\n"
        "\tshf.l.wrap.b32 x13, x12, x13, 1; shf.l.wrap.b32 x12, x11, x12, 1;\n"
        "\tshf.l.wrap.b32 x11, x10, x11, 1; shf.l.wrap.b32 x10, x9, x10, 1;\n"
        "\tshf.l.wrap.b32 x9, x8, x9, 1; shf.l.wrap.b32 x8, x7, x8, 1;\n"
        "\tshf.l.wrap.b32 x7, x6, x7, 1; shf.l.wrap.b32 x6, x5, x6, 1;\n"
        "\tshf.l.wrap.b32 x5, x4, x5, 1; shf.l.wrap.b32 x4, x3, x4, 1;\n"
        "\tshf.l.wrap.b32 x3, x2, x3, 1; shf.l.wrap.b32 x2, x1, x2, 1;\n"
        "\tshf.l.wrap.b32 x1, x0, x1, 1;\n"
        "\tmov.b64 d0, {x0,x1}; mov.b64 d1, {x2,x3}; mov.b64 d2, {x4,x5}; mov.b64 d3, {x6,x7};\n"
        "\tmov.b64 d4, {x8,x9}; mov.b64 d5, {x10,x11}; mov.b64 d6, {x12,x13}; mov.b64 d7, {x14,x15};\n"
        "\tmul.wide.u32 t, a0, a0; add.cc.u64 d0, d0, t;\n"
        "\tmul.wide.u32 t, a1, a1; addc.cc.u64 d1, d1, t;\n"
        "\tmul.wide.u32 t, a2, a2; addc.cc.u64 d2, d2, t;\n"
        "\tmul.wide.u32 t, a3, a3; addc.cc.u64 d3, d3, t;\n"
        "\tmul.wide.u32 t, a4, a4; addc.cc.u64 d4, d4, t;\n"
        "\tmul.wide.u32 t, a5, a5; addc.cc.u64 d5, d5, t;\n"
        "\tmul.wide.u32 t, a6, a6; addc.cc.u64 d6, d6, t;\n"
        "\tmul.wide.u32 t, a7, a7; addc.u64 d7, d7, t;\n"
        "\tmov.b64 {x0,x1}, d0; mov.b64 {x2,x3}, d1; mov.b64 {x4,x5}, d2; mov.b64 {x6,x7}, d3;\n"
        "\tmov.b64 {x8,x9}, d4; mov.b64 {x10,x11}, d5; mov.b64 {x12,x13}, d6; mov.b64 {x14,x15}, d7;\n"
        "\t.reg .u32 u0,u1,u2,u3,u4,u5,u6,u7,v0,v1,v2,v3,v4,v5,v6,v7,cf,k0;\n"
        ".reg .u64 fr0,fr1,fr2,fr3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;\n"
        "\t.reg .u32 f8,g8,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,w0,w1,w2,w3,w4,w5,w6,w7,m0,m1,m2;\n"
        "\tmov.b64 fr0, {x0,x1}; mov.b64 fr1, {x2,x3}; mov.b64 fr2, {x4,x5}; mov.b64 fr3, {x6,x7};\n"
        "\tmov.b64 h0, {x8,x9}; mov.b64 h1, {x10,x11}; mov.b64 h2, {x12,x13}; mov.b64 h3, {x14,x15};\n"
        "\tmul.wide.u32 t, x8, 977;  add.cc.u64  f0, fr0, t;\n"
        "\tmul.wide.u32 t, x10, 977; addc.cc.u64 f1, fr1, t;\n"
        "\tmul.wide.u32 t, x12, 977; addc.cc.u64 f2, fr2, t;\n"
        "\tmul.wide.u32 t, x14, 977; addc.cc.u64 f3, fr3, t;\n"
        "\taddc.u32 f8, 0, 0;\n"
        "\tmul.wide.u32 t, x9, 977;  add.cc.u64  g0, h0, t;\n"
        "\tmul.wide.u32 t, x11, 977; addc.cc.u64 g1, h1, t;\n"
        "\tmul.wide.u32 t, x13, 977; addc.cc.u64 g2, h2, t;\n"
        "\tmul.wide.u32 t, x15, 977; addc.cc.u64 g3, h3, t;\n"
        "\taddc.u32 g8, 0, 0;\n"
        "\tmov.b64 {z0,z1}, f0; mov.b64 {z2,z3}, f1; mov.b64 {z4,z5}, f2; mov.b64 {z6,z7}, f3;\n"
        "\tmov.b64 {w0,w1}, g0; mov.b64 {w2,w3}, g1; mov.b64 {w4,w5}, g2; mov.b64 {w6,w7}, g3;\n"
        "\tadd.cc.u32 z1, z1, w0; addc.cc.u32 z2, z2, w1; addc.cc.u32 z3, z3, w2;\n"
        "\taddc.cc.u32 z4, z4, w3; addc.cc.u32 z5, z5, w4; addc.cc.u32 z6, z6, w5;\n"
        "\taddc.cc.u32 z7, z7, w6; addc.cc.u32 z8, f8, w7; addc.u32 z9, g8, 0;\n"
        "\tmov.b64 {u0,u1}, %8; mov.b64 {u2,u3}, %9;\n"
        "\tmov.b64 {u4,u5}, %10; mov.b64 {u6,u7}, %11;\n"
        "\tmov.b64 {v0,v1}, %12; mov.b64 {v2,v3}, %13;\n"
        "\tmov.b64 {v4,v5}, %14; mov.b64 {v6,v7}, %15;\n"
        "\tadd.cc.u32 z0, z0, 0xfffff48d; addc.cc.u32 z1, z1, 0xfffffffc;\n"
        "\taddc.cc.u32 z2, z2, 0xffffffff; addc.cc.u32 z3, z3, 0xffffffff;\n"
        "\taddc.cc.u32 z4, z4, 0xffffffff; addc.cc.u32 z5, z5, 0xffffffff;\n"
        "\taddc.cc.u32 z6, z6, 0xffffffff; addc.cc.u32 z7, z7, 0xffffffff;\n"
        "\taddc.cc.u32 z8, z8, 2; addc.u32 z9, z9, 0;\n"
        "\tadd.cc.u32 z0, z0, u0; addc.cc.u32 z1, z1, u1;\n"
        "\taddc.cc.u32 z2, z2, u2; addc.cc.u32 z3, z3, u3;\n"
        "\taddc.cc.u32 z4, z4, u4; addc.cc.u32 z5, z5, u5;\n"
        "\taddc.cc.u32 z6, z6, u6; addc.cc.u32 z7, z7, u7;\n"
        "\taddc.cc.u32 z8, z8, 0; addc.u32 z9, z9, 0;\n"
        "\tsub.cc.u32 z0, z0, v0; subc.cc.u32 z1, z1, v1;\n"
        "\tsubc.cc.u32 z2, z2, v2; subc.cc.u32 z3, z3, v3;\n"
        "\tsubc.cc.u32 z4, z4, v4; subc.cc.u32 z5, z5, v5;\n"
        "\tsubc.cc.u32 z6, z6, v6; subc.cc.u32 z7, z7, v7;\n"
        "\tsubc.cc.u32 z8, z8, 0; subc.u32 z9, z9, 0;\n"
        "\tsub.cc.u32 z0, z0, v0; subc.cc.u32 z1, z1, v1;\n"
        "\tsubc.cc.u32 z2, z2, v2; subc.cc.u32 z3, z3, v3;\n"
        "\tsubc.cc.u32 z4, z4, v4; subc.cc.u32 z5, z5, v5;\n"
        "\tsubc.cc.u32 z6, z6, v6; subc.cc.u32 z7, z7, v7;\n"
        "\tsubc.cc.u32 z8, z8, 0; subc.u32 z9, z9, 0;\n"
        "\tmul.wide.u32 t, z8, 977; mov.b64 {m0,m1}, t;\n"
        "\tmad.lo.u32 m1, z9, 977, m1;\n"
        "\tadd.cc.u32 m1, m1, z8; addc.u32 m2, z9, 0;\n"
        "\tadd.cc.u32 z0, z0, m0; addc.cc.u32 z1, z1, m1; addc.cc.u32 z2, z2, m2;\n"
        "\taddc.cc.u32 z3, z3, 0; addc.cc.u32 z4, z4, 0; addc.cc.u32 z5, z5, 0;\n"
        "\taddc.cc.u32 z6, z6, 0; addc.cc.u32 z7, z7, 0;\n"
        "\taddc.u32 cf, 0, 0; mul.lo.u32 k0, cf, 977;\n"
        "\tadd.cc.u32 z0, z0, k0; addc.cc.u32 z1, z1, cf;\n"
        "addc.u32 z2, z2, 0;\n"
        "\tmov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.b64 %2, {z4,z5}; mov.b64 %3, {z6,z7};\n"
        "\t}\n"
        "\t\n"
        : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
          "l"(e[0]), "l"(e[1]), "l"(e[2]), "l"(e[3]),
          "l"(q[0]), "l"(q[1]), "l"(q[2]), "l"(q[3]));
    out[0] = r0; out[1] = r1; out[2] = r2; out[3] = r3;
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

    /* The positive first fold is L+kH+3p+e-2q. For all 256-bit
     * representatives it is in [0,(k+5)*B), so its high word fits 33 bits. */
    {
        const uint32_t bias[8] = {0xfffff48dU,0xfffffffcU,0xffffffffU,0xffffffffU,
                                  0xffffffffU,0xffffffffU,0xffffffffU,0xffffffffU};
        uint64_t carry=0;
        for(int i=0;i<8;i++){
            uint64_t tv=(uint64_t)z[i]+bias[i]+carry;
            z[i]=(uint32_t)tv; carry=tv>>32;
        }
        uint64_t tv=(uint64_t)z[8]+2+carry;
        z[8]=(uint32_t)tv; z[9]+=(uint32_t)(tv>>32);
        carry=0;
        for(int i=0;i<8;i++){
            uint64_t ei=(uint32_t)(e[i/2]>>(32*(i&1)));
            tv=(uint64_t)z[i]+ei+carry;
            z[i]=(uint32_t)tv; carry=tv>>32;
        }
        tv=(uint64_t)z[8]+carry;
        z[8]=(uint32_t)tv; z[9]+=(uint32_t)(tv>>32);
        for(int repeat=0;repeat<2;repeat++){
            uint64_t borrow=0;
            for(int i=0;i<8;i++){
                uint64_t qi=(uint32_t)(q[i/2]>>(32*(i&1)));
                uint64_t sub=qi+borrow;
                borrow=(uint64_t)z[i]<sub;
                z[i]=(uint32_t)((uint64_t)z[i]-sub);
            }
            tv=(uint64_t)z[8]-borrow;
            borrow=(uint64_t)z[8]<borrow;
            z[8]=(uint32_t)tv; z[9]-=(uint32_t)borrow;
        }
    }
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
      tv=(uint64_t)z[7]+c; z[7]=(uint32_t)tv; c=tv>>32;
      /* Fold the final 2^256 carry as k. A second carry is impossible. */
      tv=(uint64_t)z[0]+c*977; z[0]=(uint32_t)tv; c=(tv>>32)+c;
      tv=(uint64_t)z[1]+c; z[1]=(uint32_t)tv; c=tv>>32;
      for(int i=2;i<8;i++){ tv=(uint64_t)z[i]+c; z[i]=(uint32_t)tv; c=tv>>32; }
    }
    out[0]=z[0]|((uint64_t)z[1]<<32); out[1]=z[2]|((uint64_t)z[3]<<32);
    out[2]=z[4]|((uint64_t)z[5]<<32); out[3]=z[6]|((uint64_t)z[7]<<32);
    #undef MW
    (void)h0;(void)h1;(void)h2;(void)h3;(void)r0;(void)r1;(void)r2;(void)r3;(void)carry;(void)d;
#endif
}
#endif  /* QSB_FUSE_SQRADDSUB2 */
//Very efficient way of finding 8-byte target value in global memory buffer (Buffer must be ordered in ascending order)
//Each step it does fast division by half: mid = (hi + lo) >> 1; and checks resulting value
//Worst-case performance is O(log n), and we don't need to calculate any hashes by using this method.
__device__ int _BinarySearch(uint64_t *buffer, int hi, uint64_t target)
{
    int mid;
	int lo = 0;

	while (hi - lo > 1)
	{
		mid = (hi + lo) >> 1;
		if (buffer[mid] == target)
		{
			return mid;
		}
		else if (buffer[mid] < target)
		{
			lo = mid + 1;
		}
		else
		{
			hi = mid;
		}
	}

	if (buffer[lo] == target)
	{
		return lo;
	}
	else if (buffer[hi] == target)
	{
		return hi;
	}
	else
	{
		return -1;
	}
}

//Secp256k1 Point Addition implementation
__device__ void _PointAddSecp256k1(uint64_t *p1x, uint64_t *p1y, uint64_t *p1z, uint64_t *p2x, uint64_t *p2y)
{
  uint64_t u[4];
  uint64_t v[4];

  uint64_t us2[4];
  uint64_t vs2[4];
  uint64_t vs3[4];

  uint64_t a[4];

  uint64_t us2w[4];
  uint64_t vs2v2[4];
  uint64_t vs3u2[4];
  uint64_t _2vs2v2[4];

  _ModMult(u, p2y, p1z);
  _ModMult(v, p2x, p1z);

  _ModSub256(u, u, p1y);
  _ModSub256(v, v, p1x);

  _ModSqr(us2, u);
  _ModSqr(vs2, v);

  _ModMult(vs3, vs2, v);
  _ModMult(us2w, us2, p1z);
  _ModMult(vs2v2, vs2, p1x);

  _ModAdd256(_2vs2v2, vs2v2, vs2v2);

  _ModSub256(a, us2w, vs3);
  _ModSub256(a, _2vs2v2);

  _ModMult(p1x, v, a);
  _ModMult(vs3u2, vs3, p1y);

  _ModSub256(p1y, vs2v2, a);
  _ModMult(p1y, p1y, u);

  _ModSub256(p1y, vs3u2);
  _ModMult(p1z, vs3, p1z);
}

// ---------------------------------------------------------------------------------------
// XYZZ coordinates: x = X/ZZ, y = Y/ZZZ with the invariant ZZ^3 == ZZZ^2 (a = 0 plays no
// part in addition). Same limb convention as _ModMult: values in [0, 2^256), not
// necessarily < p. Outputs must not alias inputs.
//
// EFD "madd-2008-s" -- (X1,Y1,ZZ1,ZZZ1) += affine (X2,Y2) in place, 8M + 2S.
// This equivalent schedule anchors the final y formula at the affine addend:
//   U2 = X2*ZZ1, S2 = Y2*ZZZ1, P = U2-X1, R = S2-Y1, PP = P^2, PPP = P*PP,
//   V = U2*PP, X3 = R^2 + PPP - 2V,
//   Y3 = R*(V-X3) - Y2*ZZZ3, ZZ3 = ZZ1*PP, ZZZ3 = ZZZ1*PPP.
// Y1 may be an affine-anchor-deferred ordinate: Yactual=Y1-Yoff*ZZZ1.
// Adding Yoff to Y2 only for the slope restores the ordinary numerator. If
// defer_y is true, return Ycore=R*(V-X3), making Y2 the next affine anchor;
// otherwise subtract Y2*ZZZ3 and return the exact XYZZ ordinate.
// P == 0 (x1 == x2) gives ZZ3 == ZZZ3 == 0: the point at infinity for P1 == -P2 and, as
// with the homogeneous add this replaces, no valid answer for P1 == P2. Neither occurs in
// the fixed-base multiply, whose table entries are distinct non-opposite multiples of G.
// ---------------------------------------------------------------------------------------
template<bool DEFER_Y>
__device__ __forceinline__ void _PointAddXYZZT(
    uint64_t *X1, uint64_t *Y1, uint64_t *ZZ1, uint64_t *ZZZ1,
    const uint64_t *X2, const uint64_t *Y2, const uint64_t *Yoff);

__device__ void _PointAddXYZZ(uint64_t *X1, uint64_t *Y1, uint64_t *ZZ1, uint64_t *ZZZ1,
                              const uint64_t *X2, const uint64_t *Y2,
                              const uint64_t *Yoff, bool defer_y)
{
  uint64_t U2[4];
  uint64_t S2[4];
  uint64_t P[4];
  uint64_t R[4];
  uint64_t PP[4];
  uint64_t PPP[4];
  uint64_t Q[4];
  uint64_t T[4];

  _ModMult(U2, (uint64_t *)X2, ZZ1);   // U2 = X2*ZZ1
#if QSB_LAZY
  _ModAddLazy(S2, Y2, Yoff);
#else
  _ModAdd256(S2, (uint64_t *)Y2, (uint64_t *)Yoff);
#endif
  _ModMult(S2, ZZZ1);                  // S2 = (Y2+Yoff)*ZZZ1
  _ModSub256(P, U2, X1);               // P  = U2 - X1
  _ModSub256(R, S2, Y1);               // R  = S2 - Y1
  _ModSqr(PP, P);                      // PP = P^2
  _ModMult(PPP, PP, P);                // PPP = P*PP
  _ModMult(Q, U2, PP);                 // V  = U2*PP
  _ModMult(ZZ1, PP);                   // ZZ3; PP dies before the R^2/Y3 tail

#if QSB_FUSE_SQRADDSUB2
  /* xlib f297b0f9: one reduction for R^2 + PPP - 2V. */
  _ModSqrAddSub2(T, R, PPP, Q);        // X3 = R^2 + PPP - 2V
#else
  _ModSqr(T, R);                       // R^2
#if QSB_LAZY
  _ModX3Fused(T, T, PPP, Q);           // X3 = R^2 + PPP - 2V
#else
  _ModAdd256(T, T, PPP);
  _ModSub256(T, T, Q);
  _ModSub256(T, T, Q);                 // X3 = R^2 + PPP - 2V
#endif
#endif

  _ModMult(ZZZ1, PPP);                 // ZZZ3
  _ModSub256(Q, Q, T);                 // V - X3
  _ModMult(Q, R);                      // R*(V - X3)
  if (defer_y) {
    Load256(Y1, Q);                    // actual Y3 = Y1 - Y2*ZZZ3
  } else {
    _ModMult(S2, (uint64_t *)Y2, ZZZ1);// affine Y2*ZZZ3
    _ModSub256(Y1, Q, S2);             // exact Y3
  }

  Load256(X1, T);                      // X3
}

// Compile-time twin of _PointAddXYZZ (delta C, jacklightChen e582bda4): the
// production chain calls <true> twelve times in its rolled loop and <false>
// once for the resolving final addition, so no defer_y branch is in the loop.
template<bool DEFER_Y>
__device__ __forceinline__ void _PointAddXYZZT(
    uint64_t *X1, uint64_t *Y1, uint64_t *ZZ1, uint64_t *ZZZ1,
    const uint64_t *X2, const uint64_t *Y2, const uint64_t *Yoff)
{
  uint64_t U2[4];
  uint64_t S2[4];
  uint64_t P[4];
  uint64_t R[4];
  uint64_t PP[4];
  uint64_t PPP[4];
  uint64_t Q[4];
  uint64_t T[4];

#if QSB_LAZY
  _ModAddLazy(S2, Y2, Yoff);
#else
  _ModAdd256(S2, (uint64_t *)Y2, (uint64_t *)Yoff);
#endif
  _ModMult(S2, ZZZ1);                  // S2 = (Y2+Yoff)*ZZZ1
  _ModSub256(R, S2, Y1);               // R  = S2 - Y1
  _ModMult(U2, (uint64_t *)X2, ZZ1);   // U2 = X2*ZZ1
  _ModSub256(P, U2, X1);               // P  = U2 - X1
  _ModSqr(PP, P);                      // PP = P^2
  _ModMult(PPP, PP, P);                // PPP = P*PP
  _ModMult(Q, U2, PP);                 // V  = U2*PP
  _ModMult(ZZ1, PP);                   // ZZ3; PP dies before the R^2/Y3 tail

#if QSB_FUSE_SQRADDSUB2
  /* xlib f297b0f9: one reduction for R^2 + PPP - 2V. */
  _ModSqrAddSub2(T, R, PPP, Q);        // X3 = R^2 + PPP - 2V
#else
  _ModSqr(T, R);                       // R^2
#if QSB_LAZY
  _ModX3Fused(T, T, PPP, Q);           // X3 = R^2 + PPP - 2V
#else
  _ModAdd256(T, T, PPP);
  _ModSub256(T, T, Q);
  _ModSub256(T, T, Q);                 // X3 = R^2 + PPP - 2V
#endif
#endif

  _ModMult(ZZZ1, PPP);                 // ZZZ3
  _ModSub256(Q, Q, T);                 // V - X3
  _ModMult(Q, R);                      // R*(V - X3)
  if (DEFER_Y) {
    Load256(Y1, Q);                    // actual Y3 = Y1 - Y2*ZZZ3
  } else {
    _ModMult(S2, (uint64_t *)Y2, ZZZ1);// affine Y2*ZZZ3
    _ModSub256(Y1, Q, S2);             // exact Y3
  }

  Load256(X1, T);                      // X3
}

// Direct-three-affine prefix based on EFD "mmadd-2008-s", 3M + 2S. X3,
// ZZ3, and ZZZ3 are the ordinary coordinates of P1+P2, while Y3 deliberately
// holds only R*(Q-X3), omitting -Y1*ZZZ3. The caller adds Y1 only to the third
// affine point's slope input for one affine-anchored madd. Its slope numerator
// is then (Ythird+Y1)*ZZZ3-R*(Q-X3) = Ythird*ZZZ3-Y(P1+P2), while its final
// affine anchor remains Ythird. The combined seed is therefore exact and costs
// 11M+4S rather than 12M+4S.
__device__ void _PointAddXYZZ_mm(uint64_t *X3, uint64_t *Y3, uint64_t *ZZ3, uint64_t *ZZZ3,
                                 const uint64_t *X1, const uint64_t *Y1,
                                 const uint64_t *X2, const uint64_t *Y2)
{
  uint64_t P[4];
  uint64_t R[4];
  uint64_t Q[4];
  uint64_t T[4];

  _ModSub256(P, (uint64_t *)X2, (uint64_t *)X1);   // P = X2 - X1
  _ModSub256(R, (uint64_t *)Y2, (uint64_t *)Y1);   // R = Y2 - Y1
  _ModSqr(ZZ3, P);                                 // ZZ3  = PP  = P^2
  _ModMult(ZZZ3, ZZ3, P);                          // ZZZ3 = PPP = P*PP
  _ModMult(Q, (uint64_t *)X1, ZZ3);                // Q = X1*PP

  _ModSqr(T, R);                                   // R^2
  _ModSub256(T, T, ZZZ3);
  _ModSub256(T, T, Q);
  _ModSub256(T, T, Q);                             // X3 = R^2 - PPP - 2Q

  _ModSub256(Q, Q, T);                             // Q - X3
  _ModMult(Y3, Q, R);                              // deferred R*(Q-X3)
  Load256(X3, T);                                  // X3
}
