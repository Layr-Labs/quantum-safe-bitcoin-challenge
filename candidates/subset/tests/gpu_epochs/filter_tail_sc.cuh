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
/* QSB_SHORT_CARRY4 (kill switch, default off): keep the K-correction's
 * borrow/carry in limb 0 only, dropping the limb-1 propagation that
 * QSB_SHORT_CARRY3 retains. Removes one instruction from each of qsb_fsub and
 * qsb_fadd, which are the pre-inverse/post-inverse tail helpers on the
 * speculative path. Raises the truncation exposure from <= 2^-95 to the
 * 2^-31 class: limb0 must underflow/overflow for the dropped bit to matter.
 * Filter-only, so the same safety argument as the rest of this file holds --
 * a corrupted speculative x-coordinate loses a tentative hit and can never
 * publish one, because kernel_verify_pair_hits recomputes every tentative hit
 * with the unchanged exact arithmetic. Mechanism credit: PR 654 (Meganpark980320),
 * which measured +0.377% over this frontier on a full 1200 s matched pair. */
#ifndef QSB_SHORT_CARRY4
#define QSB_SHORT_CARRY4 1
#endif
/* QSB_LOSS_FINK32 (default 0 = the text above byte for byte): qsb_fsub and qsb_fadd in the chain's K32 half-word form,
 * one correction instruction shorter per site; the wrong class moves from limb 0's 2^-31/2^-33 band to the half-word
 * band (a borrow or carry, then the low half within 977 of its edge: about 2^-23 per site). Every finish site uses
 * them (d, n1, n2, sum, the two t, x1, x2). Loss only: the exact host gate re-derives every nomination. */
#ifndef QSB_LOSS_FINK32
#define QSB_LOSS_FINK32 1
#endif
#if QSB_LOSS_FINK32 && !QSB_SHORT_CARRY4
#error "QSB_LOSS_FINK32 is written for the QSB_SHORT_CARRY4 finish helpers"
#endif
#if QSB_SHORT_CARRY3
// r = a - b mod p for a,b < 2^256. The borrow correction subtracts K = 2^32+977 from the
// wrapped difference; its borrow is kept through limb 1 only. It would have to cross limb 1
// only if limb0 < K and limb1 == 0 after the subtraction: probability <= 2^-95.
__device__ __forceinline__ void qsb_fsub(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t r0,r1,r2,r3;
#if QSB_LOSS_FINK32
    /* QSB_LOSS_FINK32: the chain's K32 short form (qsb_filter_sub_cut): on a borrow, limb 0 minus K = 2^32 + 977 as
     * 32-bit halves, the low half's borrow into the high half dropped (wrong when a borrow meets l < 977). */
    asm("{\n\t.reg .u32 m,l,h;\n\t"
        "sub.cc.u64 %0,%4,%8;\n\t"
        "subc.cc.u64 %1,%5,%9;\n\t"
        "subc.cc.u64 %2,%6,%10;\n\t"
        "subc.cc.u64 %3,%7,%11;\n\t"
        "subc.u32 m,0,0;\n\t"
        "mov.b64 {l,h},%0;\n\t"
        "mad.lo.u32 l,m,977,l;\n\t"
        "add.u32 h,h,m;\n\t"
        "mov.b64 %0,{l,h};\n\t}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
    return;
#endif
    asm("{\n\t.reg .u64 brw,lo;\n\t"
        "sub.cc.u64 %0,%4,%8;\n\t"
        "subc.cc.u64 %1,%5,%9;\n\t"
        "subc.cc.u64 %2,%6,%10;\n\t"
        "subc.cc.u64 %3,%7,%11;\n\t"
        "subc.u64 brw,0,0;\n\t"
        "and.b64 lo,brw,0x1000003D1;\n\t"
#if QSB_SHORT_CARRY4
        "sub.u64 %0,%0,lo;\n\t}"
#else
        "sub.cc.u64 %0,%0,lo;\n\t"
        "subc.u64 %1,%1,0;\n\t}"
#endif
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}
// r = a + b mod p (a,b < 2^256): fold the carry with K; the fold's carry is kept through limb 1
// (crossing it needs limb0 overflow, p <= 2^-31, and limb1 == 2^64-1: <= 2^-95 overall).
__device__ __forceinline__ void qsb_fadd(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t r0,r1,r2,r3;
#if QSB_LOSS_FINK32
    /* QSB_LOSS_FINK32: the anchor sum's K32 fold (QSB_K32_ADDCUT form): on a carry, limb 0 plus K as 32-bit halves,
     * the low half's carry into the high half dropped (wrong when a carry meets l >= 2^32 - 977). */
    asm("{\n\t.reg .u32 m,l,h;\n\t"
        "add.cc.u64 %0,%4,%8;\n\t"
        "addc.cc.u64 %1,%5,%9;\n\t"
        "addc.cc.u64 %2,%6,%10;\n\t"
        "addc.cc.u64 %3,%7,%11;\n\t"
        "addc.u32 m,0,0;\n\t"
        "mov.b64 {l,h},%0;\n\t"
        "mad.lo.u32 l,m,977,l;\n\t"
        "add.u32 h,h,m;\n\t"
        "mov.b64 %0,{l,h};\n\t}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
    return;
#endif
    asm("{\n\t.reg .u64 h,t;\n\t"
        "add.cc.u64 %0,%4,%8;\n\t"
        "addc.cc.u64 %1,%5,%9;\n\t"
        "addc.cc.u64 %2,%6,%10;\n\t"
        "addc.cc.u64 %3,%7,%11;\n\t"
        "addc.u64 h,0,0;\n\t"
        "mul.lo.u64 t,h,0x1000003d1;\n\t"
#if QSB_SHORT_CARRY4
        "add.u64 %0,%0,t;\n\t}"
#else
        "add.cc.u64 %0,%0,t;\n\t"
        "addc.u64 %1,%1,0;\n\t}"
#endif
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
