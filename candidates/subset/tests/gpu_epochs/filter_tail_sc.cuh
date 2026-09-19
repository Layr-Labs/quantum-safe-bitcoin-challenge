// Speculative-filter-only field helpers for the finish (pre-inverse, block tree, post-inverse).
// Same bounded carry truncation as the promoted QSB_SHORT_CARRY/QSB_SHORT_CARRY2 chain sites:
// a dropped carry/borrow can only corrupt this candidate's (or, in the block tree, this block's)
// speculative x-coordinates, which loses tentative hits; it can never publish one, because every
// tentative hit is recomputed by kernel_verify_pair_hits with the unchanged exact arithmetic.
// The exact replay chain, qsb_k2s_front_exact/qsb_k2s_post and the verifier never call these.
#pragma once
#ifndef QSB_SHORT_CARRY3
#define QSB_SHORT_CARRY3 1
#endif
#if QSB_SHORT_CARRY3
// r = a - b mod p for a,b < 2^256. The borrow correction subtracts K = 2^32+977 from the
// wrapped difference; this private variant keeps the correction through limb 0 only.
// It differs from the parent when a<b and the wrapped low64 difference is below K.
// No input-distribution probability or absence of lost proposals is assumed.
__device__ __forceinline__ void qsb_fsub(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t r0,r1,r2,r3;
    asm("{\n\t.reg .u64 brw,lo;\n\t"
        "sub.cc.u64 %0,%4,%8;\n\t"
        "subc.cc.u64 %1,%5,%9;\n\t"
        "subc.cc.u64 %2,%6,%10;\n\t"
        "subc.cc.u64 %3,%7,%11;\n\t"
        "subc.u64 brw,0,0;\n\t"
        "and.b64 lo,brw,0x1000003D1;\n\t"
        "sub.u64 %0,%0,lo;\n\t"
        "\n\t}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}
// r = a + b mod p (a,b < 2^256): fold the carry with K; this private variant keeps the correction through limb 0 only.
// It differs from the parent if the high carry is set and the wrapped low64 sum + K overflows.
__device__ __forceinline__ void qsb_fadd(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t r0,r1,r2,r3;
    asm("{\n\t.reg .u64 h,t;\n\t"
        "add.cc.u64 %0,%4,%8;\n\t"
        "addc.cc.u64 %1,%5,%9;\n\t"
        "addc.cc.u64 %2,%6,%10;\n\t"
        "addc.cc.u64 %3,%7,%11;\n\t"
        "addc.u64 h,0,0;\n\t"
        "mul.lo.u64 t,h,0x1000003d1;\n\t"
        "add.u64 %0,%0,t;\n\t"
        "\n\t}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}
__device__ __forceinline__ void qsb_fmul(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint32_t bad=0; qsb_filter_mul(r,a,b,bad);
}
// Block-tree product (5-limb convention of qsb_field_mul_raw: out[4]=0).
__device__ __forceinline__ void qsb_field_mul_tree(uint64_t *out, uint64_t *a, uint64_t *b) {
    uint32_t bad=0; qsb_filter_mul(out,a,b,bad); out[4]=0;
}
#define QSB_FMUL(r,a,b) qsb_fmul(r,a,b)
#define QSB_FSUB(r,a,b) qsb_fsub(r,a,b)
#define QSB_FADD(r,a,b) qsb_fadd(r,a,b)
#define QSB_TREE_MUL(o,a,b) qsb_field_mul_tree(o,a,b)
#define X_FMUL(r,a,b) _ModMult(r,(uint64_t*)(a),(uint64_t*)(b))
#define X_FSUB(r,a,b) _ModSub256(r,(uint64_t*)(a),(uint64_t*)(b))
#define X_FADD(r,a,b) _ModAdd256(r,(uint64_t*)(a),(uint64_t*)(b))
#else
#define QSB_FMUL(r,a,b) _ModMult(r,(uint64_t*)(a),(uint64_t*)(b))
#define QSB_FSUB(r,a,b) _ModSub256(r,(uint64_t*)(a),(uint64_t*)(b))
#define QSB_FADD(r,a,b) _ModAdd256(r,(uint64_t*)(a),(uint64_t*)(b))
#define QSB_TREE_MUL(o,a,b) qsb_field_mul_raw(o,a,b)
#endif
#ifndef X_FMUL   /* exact field ops kept on the pre-inverse side (register pressure: see research notes) */
#define X_FMUL(r,a,b) _ModMult(r,(uint64_t*)(a),(uint64_t*)(b))
#define X_FSUB(r,a,b) _ModSub256(r,(uint64_t*)(a),(uint64_t*)(b))
#define X_FADD(r,a,b) _ModAdd256(r,(uint64_t*)(a),(uint64_t*)(b))
#endif
