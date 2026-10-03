// Constant-bank recovery-point addition derived from public 2f57d80 (fkiene).
// Same speculative qsb_fadd carry semantics; exact replay remains untouched.
#pragma once
#ifndef QSB_POOL_RCONST
#define QSB_POOL_RCONST 0
#endif
#if QSB_POOL_RCONST != 0 && QSB_POOL_RCONST != 1
#error "QSB_POOL_RCONST must be 0 or 1"
#endif
#if QSB_POOL_RCONST
#if !QSB_SHORT_CARRY3 || !QSB_PAIR_SHARED || !ZLAB_DUAL_EPOCH_SHA || !ZLAB_K2S3M || !QSB_NEGFOLD_PARITY || QSB_TAIL_WEAVE || QSB_TAIL_STAGGER
#error "QSB_POOL_RCONST requires original short-carry ISO reload paired finish"
#endif
__device__ __forceinline__ void qsb_fadd_u2rx(uint64_t *r, const uint64_t *a) {
uint64_t r0,r1,r2,r3;
asm("{\n\t.reg .u64 h,t,c0,c1,c2,c3;\n\t"
"ld.const.u64 c0,[QSB_U2R];\n\t"
"ld.const.u64 c1,[QSB_U2R+8];\n\t"
"ld.const.u64 c2,[QSB_U2R+16];\n\t"
"ld.const.u64 c3,[QSB_U2R+24];\n\t"
"add.cc.u64 %0,%4,c0;\n\t"
"addc.cc.u64 %1,%5,c1;\n\t"
"addc.cc.u64 %2,%6,c2;\n\t"
"addc.cc.u64 %3,%7,c3;\n\t"
"addc.u64 h,0,0;\n\t"
"mul.lo.u64 t,h,0x1000003d1;\n\t"
#if QSB_SHORT_CARRY4
"add.u64 %0,%0,t;\n\t}"
#else
"add.cc.u64 %0,%0,t;\n\t"
"addc.u64 %1,%1,0;\n\t}"
#endif
: "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
: "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]));
r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}
#define QSB_FADD_XR(r,a,xr) qsb_fadd_u2rx(r,a)
#else
#define QSB_FADD_XR(r,a,xr) QSB_FADD(r,a,xr)
#endif
