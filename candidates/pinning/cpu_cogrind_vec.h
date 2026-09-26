/* cpu_cogrind_vec.h -- AVX2 elliptic-curve stage of the pinning co-grinder (4 candidates per
 * vector, one per 64-bit lane). Included once by cpu_cogrind.h inside namespace qcg.
 *
 * Field arithmetic: libsecp256k1's 10x26 representation. v26_mul / v26_sqr (C) and the
 * out-of-line assembly in cg_v26asm.h are secp256k1_fe_mul_inner / secp256k1_fe_sqr_inner
 * (field_10x26_impl.h, Copyright (c) 2013 Pieter Wuille, MIT license -- notice in
 * COPYING-secp256k1) with every 32x32->64 limb product replaced by a lane-wise VPMULUDQ; the
 * reduction schedule and the magnitude bounds are libsecp256k1's, and the assembly produces
 * bit-identical limbs. The normalisation helpers follow libsecp256k1's fe_normalize(_weak).
 * The batch structure (fused backward/forward passes, digit handling, gathers, hashing) is
 * original.
 *
 * One batch (ec_batch): P = T0[d0] (A folded in), then for each signed window s one batched
 * affine addition P += +-T_s[|d_s|] (5M + 1S per candidate, one scalar inversion per batch),
 * then Q- = Q+ + (-2A). Pass s runs backward(s) and forward(s+1) fused, alternating the block
 * order, so each point is loaded once per step. The multiplications are out-of-line assembly
 * calls, which keeps the hot loop small (decoded-icache friendly) and lets the loop body keep
 * its temporaries in one reused block of memory.
 *
 * Every function carries target("avx2") and is only called after __builtin_cpu_supports("avx2").
 */
namespace v4 {
#define QV_INL __attribute__((target("avx2"), always_inline)) inline
#define QV_FN  __attribute__((target("avx2"), noinline))
typedef uint64_t V __attribute__((vector_size(32)));
typedef int64_t VI __attribute__((vector_size(32)));
struct vfe { V n[10]; };

static QV_INL V vs1(uint64_t x) { return (V){x, x, x, x}; }
#define VMUL(a, b) ((V)_mm256_mul_epu32((__m256i)(a), (__m256i)(b)))
/* 64-bit lane times a small constant k < 2^32 (c may exceed 32 bits) */
static QV_INL V vmulk(V c, uint64_t k) { const V kk = vs1(k); return VMUL(c, kk) + (VMUL(c >> 32, kk) << 32); }

/* inputs magnitude <= 8, output magnitude 1 (libsecp256k1 10x26 bounds) */
static QV_INL void v26_mul(vfe *rr, const vfe *aa, const vfe *bb) {
    const V *a = aa->n, *b = bb->n;
    V c, d, u0, u1, u2, u3, u4, u5, u6, u7, u8, t9, t0, t1, t2, t3, t4, t5, t6, t7;
    const V M = vs1(0x3FFFFFFULL), R0 = vs1(0x3D10ULL);
    d  = ((VMUL(a[0], b[9]) + VMUL(a[1], b[8])) + (VMUL(a[2], b[7]) + VMUL(a[3], b[6])))
       + ((VMUL(a[4], b[5]) + VMUL(a[5], b[4])) + (VMUL(a[6], b[3]) + VMUL(a[7], b[2])))
       + (VMUL(a[8], b[1]) + VMUL(a[9], b[0]));
    t9 = d & M; d >>= 26;
    c  = VMUL(a[0], b[0]);
    d += ((VMUL(a[1], b[9]) + VMUL(a[2], b[8])) + (VMUL(a[3], b[7]) + VMUL(a[4], b[6])))
       + ((VMUL(a[5], b[5]) + VMUL(a[6], b[4])) + (VMUL(a[7], b[3]) + VMUL(a[8], b[2]))) + VMUL(a[9], b[1]);
    u0 = d & M; d >>= 26; c += VMUL(u0, R0);
    t0 = c & M; c >>= 26; c += u0 << 10;
    c += VMUL(a[0], b[1]) + VMUL(a[1], b[0]);
    d += ((VMUL(a[2], b[9]) + VMUL(a[3], b[8])) + (VMUL(a[4], b[7]) + VMUL(a[5], b[6])))
       + ((VMUL(a[6], b[5]) + VMUL(a[7], b[4])) + (VMUL(a[8], b[3]) + VMUL(a[9], b[2])));
    u1 = d & M; d >>= 26; c += VMUL(u1, R0);
    t1 = c & M; c >>= 26; c += u1 << 10;
    c += (VMUL(a[0], b[2]) + VMUL(a[1], b[1])) + VMUL(a[2], b[0]);
    d += ((VMUL(a[3], b[9]) + VMUL(a[4], b[8])) + (VMUL(a[5], b[7]) + VMUL(a[6], b[6])))
       + ((VMUL(a[7], b[5]) + VMUL(a[8], b[4])) + VMUL(a[9], b[3]));
    u2 = d & M; d >>= 26; c += VMUL(u2, R0);
    t2 = c & M; c >>= 26; c += u2 << 10;
    c += (VMUL(a[0], b[3]) + VMUL(a[1], b[2])) + (VMUL(a[2], b[1]) + VMUL(a[3], b[0]));
    d += ((VMUL(a[4], b[9]) + VMUL(a[5], b[8])) + (VMUL(a[6], b[7]) + VMUL(a[7], b[6])))
       + (VMUL(a[8], b[5]) + VMUL(a[9], b[4]));
    u3 = d & M; d >>= 26; c += VMUL(u3, R0);
    t3 = c & M; c >>= 26; c += u3 << 10;
    c += ((VMUL(a[0], b[4]) + VMUL(a[1], b[3])) + (VMUL(a[2], b[2]) + VMUL(a[3], b[1]))) + VMUL(a[4], b[0]);
    d += ((VMUL(a[5], b[9]) + VMUL(a[6], b[8])) + (VMUL(a[7], b[7]) + VMUL(a[8], b[6]))) + VMUL(a[9], b[5]);
    u4 = d & M; d >>= 26; c += VMUL(u4, R0);
    t4 = c & M; c >>= 26; c += u4 << 10;
    c += ((VMUL(a[0], b[5]) + VMUL(a[1], b[4])) + (VMUL(a[2], b[3]) + VMUL(a[3], b[2])))
       + (VMUL(a[4], b[1]) + VMUL(a[5], b[0]));
    d += (VMUL(a[6], b[9]) + VMUL(a[7], b[8])) + (VMUL(a[8], b[7]) + VMUL(a[9], b[6]));
    u5 = d & M; d >>= 26; c += VMUL(u5, R0);
    t5 = c & M; c >>= 26; c += u5 << 10;
    c += ((VMUL(a[0], b[6]) + VMUL(a[1], b[5])) + (VMUL(a[2], b[4]) + VMUL(a[3], b[3])))
       + ((VMUL(a[4], b[2]) + VMUL(a[5], b[1])) + VMUL(a[6], b[0]));
    d += (VMUL(a[7], b[9]) + VMUL(a[8], b[8])) + VMUL(a[9], b[7]);
    u6 = d & M; d >>= 26; c += VMUL(u6, R0);
    t6 = c & M; c >>= 26; c += u6 << 10;
    c += ((VMUL(a[0], b[7]) + VMUL(a[1], b[6])) + (VMUL(a[2], b[5]) + VMUL(a[3], b[4])))
       + ((VMUL(a[4], b[3]) + VMUL(a[5], b[2])) + (VMUL(a[6], b[1]) + VMUL(a[7], b[0])));
    d += VMUL(a[8], b[9]) + VMUL(a[9], b[8]);
    u7 = d & M; d >>= 26; c += VMUL(u7, R0);
    t7 = c & M; c >>= 26; c += u7 << 10;
    c += ((VMUL(a[0], b[8]) + VMUL(a[1], b[7])) + (VMUL(a[2], b[6]) + VMUL(a[3], b[5])))
       + ((VMUL(a[4], b[4]) + VMUL(a[5], b[3])) + (VMUL(a[6], b[2]) + VMUL(a[7], b[1]))) + VMUL(a[8], b[0]);
    d += VMUL(a[9], b[9]);
    u8 = d & M; d >>= 26; c += VMUL(u8, R0);
    V *r = rr->n;
    r[3] = t3; r[4] = t4; r[5] = t5; r[6] = t6; r[7] = t7;
    r[8] = c & M; c >>= 26; c += u8 << 10;
    c += VMUL(d, R0) + t9;
    r[9] = c & vs1(0x3FFFFFULL); c >>= 22; c += d << 14;
    d = vmulk(c, 0x3D1ULL) + t0;
    r[0] = d & M; d >>= 26;
    d += (c << 6) + t1;
    r[1] = d & M; d >>= 26;
    d += t2;
    r[2] = d;
}

static QV_INL void v26_sqr(vfe *rr, const vfe *aa) {
    const V *a = aa->n;
    V c, d, u0, u1, u2, u3, u4, u5, u6, u7, u8, t9, t0, t1, t2, t3, t4, t5, t6, t7;
    const V M = vs1(0x3FFFFFFULL), R0 = vs1(0x3D10ULL);
    const V a0x = a[0] << 1, a1x = a[1] << 1, a2x = a[2] << 1, a3x = a[3] << 1, a4x = a[4] << 1,
            a5x = a[5] << 1, a6x = a[6] << 1, a7x = a[7] << 1, a8x = a[8] << 1;
    d  = ((VMUL(a0x, a[9]) + VMUL(a1x, a[8])) + (VMUL(a2x, a[7]) + VMUL(a3x, a[6]))) + VMUL(a4x, a[5]);
    t9 = d & M; d >>= 26;
    c  = VMUL(a[0], a[0]);
    d += ((VMUL(a1x, a[9]) + VMUL(a2x, a[8])) + (VMUL(a3x, a[7]) + VMUL(a4x, a[6]))) + VMUL(a[5], a[5]);
    u0 = d & M; d >>= 26; c += VMUL(u0, R0);
    t0 = c & M; c >>= 26; c += u0 << 10;
    c += VMUL(a0x, a[1]);
    d += (VMUL(a2x, a[9]) + VMUL(a3x, a[8])) + (VMUL(a4x, a[7]) + VMUL(a5x, a[6]));
    u1 = d & M; d >>= 26; c += VMUL(u1, R0);
    t1 = c & M; c >>= 26; c += u1 << 10;
    c += VMUL(a0x, a[2]) + VMUL(a[1], a[1]);
    d += (VMUL(a3x, a[9]) + VMUL(a4x, a[8])) + (VMUL(a5x, a[7]) + VMUL(a[6], a[6]));
    u2 = d & M; d >>= 26; c += VMUL(u2, R0);
    t2 = c & M; c >>= 26; c += u2 << 10;
    c += VMUL(a0x, a[3]) + VMUL(a1x, a[2]);
    d += (VMUL(a4x, a[9]) + VMUL(a5x, a[8])) + VMUL(a6x, a[7]);
    u3 = d & M; d >>= 26; c += VMUL(u3, R0);
    t3 = c & M; c >>= 26; c += u3 << 10;
    c += (VMUL(a0x, a[4]) + VMUL(a1x, a[3])) + VMUL(a[2], a[2]);
    d += (VMUL(a5x, a[9]) + VMUL(a6x, a[8])) + VMUL(a[7], a[7]);
    u4 = d & M; d >>= 26; c += VMUL(u4, R0);
    t4 = c & M; c >>= 26; c += u4 << 10;
    c += (VMUL(a0x, a[5]) + VMUL(a1x, a[4])) + VMUL(a2x, a[3]);
    d += VMUL(a6x, a[9]) + VMUL(a7x, a[8]);
    u5 = d & M; d >>= 26; c += VMUL(u5, R0);
    t5 = c & M; c >>= 26; c += u5 << 10;
    c += (VMUL(a0x, a[6]) + VMUL(a1x, a[5])) + (VMUL(a2x, a[4]) + VMUL(a[3], a[3]));
    d += VMUL(a7x, a[9]) + VMUL(a[8], a[8]);
    u6 = d & M; d >>= 26; c += VMUL(u6, R0);
    t6 = c & M; c >>= 26; c += u6 << 10;
    c += (VMUL(a0x, a[7]) + VMUL(a1x, a[6])) + (VMUL(a2x, a[5]) + VMUL(a3x, a[4]));
    d += VMUL(a8x, a[9]);
    u7 = d & M; d >>= 26; c += VMUL(u7, R0);
    t7 = c & M; c >>= 26; c += u7 << 10;
    c += ((VMUL(a0x, a[8]) + VMUL(a1x, a[7])) + (VMUL(a2x, a[6]) + VMUL(a3x, a[5]))) + VMUL(a[4], a[4]);
    d += VMUL(a[9], a[9]);
    u8 = d & M; d >>= 26; c += VMUL(u8, R0);
    V *r = rr->n;
    r[3] = t3; r[4] = t4; r[5] = t5; r[6] = t6; r[7] = t7;
    r[8] = c & M; c >>= 26; c += u8 << 10;
    c += VMUL(d, R0) + t9;
    r[9] = c & vs1(0x3FFFFFULL); c >>= 22; c += d << 14;
    d = vmulk(c, 0x3D1ULL) + t0;
    r[0] = d & M; d >>= 26;
    d += (c << 6) + t1;
    r[1] = d & M; d >>= 26;
    d += t2;
    r[2] = d;
}

#ifndef QCG_V26_ASM
#define QCG_V26_ASM 1
#endif
#if QCG_V26_ASM
alignas(32) static const uint64_t V26K[16] = {0x3FFFFFFULL, 0x3FFFFFFULL, 0x3FFFFFFULL, 0x3FFFFFFULL, 0x3D10ULL, 0x3D10ULL, 0x3D10ULL, 0x3D10ULL,
                                              0x3FFFFFULL, 0x3FFFFFULL, 0x3FFFFFULL, 0x3FFFFFULL, 0x3D1ULL, 0x3D1ULL, 0x3D1ULL, 0x3D1ULL};
/* out-of-line asm versions (same arithmetic as v26_mul / v26_sqr); r must not alias a or b */
static QV_FN void v26_mul_o(vfe *r, const vfe *a, const vfe *b) {
    __asm__ volatile(QCG_V26_MUL_ASM : : [r] "r"(r), [a] "r"(a), [b] "r"(b), [k] "r"(V26K)
        : "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15", "memory");
}
static QV_FN void v26_sqr_o(vfe *r, const vfe *a) {
    __asm__ volatile(QCG_V26_SQR_ASM : : [r] "r"(r), [a] "r"(a), [k] "r"(V26K)
        : "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15", "memory");
}
#else
static QV_FN void v26_mul_o(vfe *r, const vfe *a, const vfe *b) { v26_mul(r, a, b); }
static QV_FN void v26_sqr_o(vfe *r, const vfe *a) { v26_sqr(r, a); }
#endif
/* r = a + b (magnitudes add) */
static QV_INL void v26_add(vfe *r, const vfe *a, const vfe *b) { for (int k = 0; k < 10; k++) r->n[k] = a->n[k] + b->n[k]; }
/* r = a + (2(m+1)p - b): a - b with b of magnitude <= m; result magnitude mag(a) + m + 1 */
static QV_INL void v26_subm(vfe *r, const vfe *a, const vfe *b, int m) {
    const uint64_t f = 2 * (uint64_t)(m + 1);
    r->n[0] = a->n[0] + (vs1(0x3FFFC2FULL * f) - b->n[0]);
    r->n[1] = a->n[1] + (vs1(0x3FFFFBFULL * f) - b->n[1]);
    for (int k = 2; k < 9; k++) r->n[k] = a->n[k] + (vs1(0x3FFFFFFULL * f) - b->n[k]);
    r->n[9] = a->n[9] + (vs1(0x03FFFFFULL * f) - b->n[9]);
}
/* r = -a, a of magnitude <= m; result magnitude m + 1 */
static QV_INL void v26_neg(vfe *r, const vfe *a, int m) {
    const uint64_t f = 2 * (uint64_t)(m + 1);
    r->n[0] = vs1(0x3FFFC2FULL * f) - a->n[0];
    r->n[1] = vs1(0x3FFFFBFULL * f) - a->n[1];
    for (int k = 2; k < 9; k++) r->n[k] = vs1(0x3FFFFFFULL * f) - a->n[k];
    r->n[9] = vs1(0x03FFFFFULL * f) - a->n[9];
}
/* magnitude -> 1 (libsecp256k1 fe_normalize_weak) */
static QV_INL void v26_nweak(vfe *r) {
    const V M = vs1(0x3FFFFFFULL);
    V t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4], t5 = r->n[5], t6 = r->n[6], t7 = r->n[7], t8 = r->n[8], t9 = r->n[9];
    V x = t9 >> 22; t9 &= vs1(0x03FFFFFULL);
    t0 += VMUL(x, vs1(0x3D1ULL)); t1 += (x << 6);
    t1 += (t0 >> 26); t0 &= M;
    t2 += (t1 >> 26); t1 &= M;
    t3 += (t2 >> 26); t2 &= M;
    t4 += (t3 >> 26); t3 &= M;
    t5 += (t4 >> 26); t4 &= M;
    t6 += (t5 >> 26); t5 &= M;
    t7 += (t6 >> 26); t6 &= M;
    t8 += (t7 >> 26); t7 &= M;
    t9 += (t8 >> 26); t8 &= M;
    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4; r->n[5] = t5; r->n[6] = t6; r->n[7] = t7; r->n[8] = t8; r->n[9] = t9;
}
/* canonical value in [0, p) (libsecp256k1 fe_normalize) */
static QV_INL void v26_norm(vfe *r) {
    const V M = vs1(0x3FFFFFFULL), one = vs1(1);
    V t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4], t5 = r->n[5], t6 = r->n[6], t7 = r->n[7], t8 = r->n[8], t9 = r->n[9];
    V m;
    V x = t9 >> 22; t9 &= vs1(0x03FFFFFULL);
    t0 += VMUL(x, vs1(0x3D1ULL)); t1 += (x << 6);
    t1 += (t0 >> 26); t0 &= M;
    t2 += (t1 >> 26); t1 &= M;
    t3 += (t2 >> 26); t2 &= M; m = t2;
    t4 += (t3 >> 26); t3 &= M; m &= t3;
    t5 += (t4 >> 26); t4 &= M; m &= t4;
    t6 += (t5 >> 26); t5 &= M; m &= t5;
    t7 += (t6 >> 26); t6 &= M; m &= t6;
    t8 += (t7 >> 26); t7 &= M; m &= t7;
    t9 += (t8 >> 26); t8 &= M; m &= t8;
    x = (t9 >> 22) | ((V)(t9 == vs1(0x03FFFFFULL)) & (V)(m == M)
        & (V)((t1 + vs1(0x40ULL) + ((t0 + vs1(0x3D1ULL)) >> 26)) > M) & one);
    t0 += VMUL(x, vs1(0x3D1ULL)); t1 += (x << 6);
    t1 += (t0 >> 26); t0 &= M;
    t2 += (t1 >> 26); t1 &= M;
    t3 += (t2 >> 26); t2 &= M;
    t4 += (t3 >> 26); t3 &= M;
    t5 += (t4 >> 26); t4 &= M;
    t6 += (t5 >> 26); t5 &= M;
    t7 += (t6 >> 26); t6 &= M;
    t8 += (t7 >> 26); t7 &= M;
    t9 += (t8 >> 26); t8 &= M;
    t9 &= vs1(0x03FFFFFULL);
    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4; r->n[5] = t5; r->n[6] = t6; r->n[7] = t7; r->n[8] = t8; r->n[9] = t9;
}
/* r = mask ? a : b (per lane, mask all-ones or zero) */
static QV_INL void v26_sel(vfe *r, V mask, const vfe *a, const vfe *b) {
    for (int k = 0; k < 10; k++) r->n[k] = (V)_mm256_blendv_epi8((__m256i)b->n[k], (__m256i)a->n[k], (__m256i)mask);
}
static QV_INL void v26_one(vfe *r) { r->n[0] = vs1(1); for (int k = 1; k < 10; k++) r->n[k] = vs1(0); }
/* 4 little-endian 64-bit words per lane -> 10x26 (input < 2^256) */
static QV_INL void v26_from_w(vfe *r, V w0, V w1, V w2, V w3) {
    const V M = vs1(0x3FFFFFFULL);
    r->n[0] = w0 & M;
    r->n[1] = (w0 >> 26) & M;
    r->n[2] = ((w0 >> 52) | (w1 << 12)) & M;
    r->n[3] = (w1 >> 14) & M;
    r->n[4] = ((w1 >> 40) | (w2 << 24)) & M;
    r->n[5] = (w2 >> 2) & M;
    r->n[6] = (w2 >> 28) & M;
    r->n[7] = ((w2 >> 54) | (w3 << 10)) & M;
    r->n[8] = (w3 >> 16) & M;
    r->n[9] = w3 >> 42;
}
/* normalized 10x26 -> 4 words per lane */
static QV_INL void v26_to_w(V w[4], const vfe *a) {
    const V *n = a->n;
    w[0] = n[0] | (n[1] << 26) | (n[2] << 52);
    w[1] = (n[2] >> 12) | (n[3] << 14) | (n[4] << 40);
    w[2] = (n[4] >> 24) | (n[5] << 2) | (n[6] << 28) | (n[7] << 54);
    w[3] = (n[7] >> 10) | (n[8] << 16) | (n[9] << 42);
}
/* broadcast a 4x64 constant */
static QV_INL void v26_set_w(vfe *r, const uint64_t *w) { v26_from_w(r, vs1(w[0]), vs1(w[1]), vs1(w[2]), vs1(w[3])); }
/* r lane l = a lane (l ^ k) */
static QV_INL void v26_permx(vfe *r, const vfe *a, int k) {
    for (int j = 0; j < 10; j++)
        r->n[j] = (V)(k == 1 ? _mm256_permute4x64_epi64((__m256i)a->n[j], 0xB1) : _mm256_permute4x64_epi64((__m256i)a->n[j], 0x4E));
}

/* gather x (half 0) or y (half 1) words of 4 table entries, transposed to one V per word */
static QV_INL void tr4(V w[4], const tentry *e0, const tentry *e1, const tentry *e2, const tentry *e3, int half) {
    __m256i r0 = _mm256_load_si256((const __m256i *)((const uint8_t *)e0 + 32 * half));
    __m256i r1 = _mm256_load_si256((const __m256i *)((const uint8_t *)e1 + 32 * half));
    __m256i r2 = _mm256_load_si256((const __m256i *)((const uint8_t *)e2 + 32 * half));
    __m256i r3 = _mm256_load_si256((const __m256i *)((const uint8_t *)e3 + 32 * half));
    __m256i t0 = _mm256_unpacklo_epi64(r0, r1), t1 = _mm256_unpackhi_epi64(r0, r1);
    __m256i t2 = _mm256_unpacklo_epi64(r2, r3), t3 = _mm256_unpackhi_epi64(r2, r3);
    w[0] = (V)_mm256_permute2x128_si256(t0, t2, 0x20); w[2] = (V)_mm256_permute2x128_si256(t0, t2, 0x31);
    w[1] = (V)_mm256_permute2x128_si256(t1, t3, 0x20); w[3] = (V)_mm256_permute2x128_si256(t1, t3, 0x31);
}
static QV_INL void gather_x(vfe *x, const tentry *const *e) { V w[4]; tr4(w, e[0], e[1], e[2], e[3], 0); v26_from_w(x, w[0], w[1], w[2], w[3]); }
static QV_INL void gather_y(vfe *y, const tentry *const *e) { V w[4]; tr4(w, e[0], e[1], e[2], e[3], 1); v26_from_w(y, w[0], w[1], w[2], w[3]); }

/* per-worker vector state */
struct bst;
struct vstate;

/* inv lane l = 1 / m lane l, for all 4 lanes, with one scalar inversion:
 * q = product of the 4 lanes (butterfly), oth = product of the other 3 lanes, inv = oth / q */
static QV_INL void vinv(vfe *inv, const vfe *m) {
    vfe s, q1, q2, oth, t;
    v26_permx(&s, m, 1); oth = s; v26_mul_o(&q1, m, &s);          /* q1 lane l = m_l m_(l^1) */
    v26_permx(&s, &q1, 2); v26_mul_o(&t, &oth, &s); v26_mul_o(&q2, &q1, &s);
    oth = t;
    v26_norm(&q2);
    V w[4]; v26_to_w(w, &q2);
    uint64_t tw[4] = {w[0][0], w[1][0], w[2][0], w[3][0]}, iw[4];
    scalar_inv(iw, tw);
    v26_set_w(&t, iw);
    v26_mul_o(inv, &t, &oth);
}

/* the 8-lane pubkey SHA of Q+ (lanes 0-3) and Q- (lanes 4-7) of one block; returns an 8-bit
 * mask of lanes whose H0 passes the leading-zero prefilter */
static QV_INL unsigned hash_block(const vfe *xp, const vfe *yp, const vfe *xm, const vfe *ym, int use_ni) {
    using namespace qcg_sha;
    V wp[4], wm[4];
    v26_to_w(wp, xp); v26_to_w(wm, xm);
    /* 32-bit words X0 (least significant) .. X7 of x, lanes 0-3 = Q+, 4-7 = Q- */
    const __m256i idx_lo = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6), idx_hi = _mm256_setr_epi32(1, 3, 5, 7, 1, 3, 5, 7);
    v8u X[8];
    for (int k = 0; k < 4; k++) {
        __m256i pl = _mm256_permutevar8x32_epi32((__m256i)wp[k], idx_lo), ml = _mm256_permutevar8x32_epi32((__m256i)wm[k], idx_lo);
        __m256i ph = _mm256_permutevar8x32_epi32((__m256i)wp[k], idx_hi), mh = _mm256_permutevar8x32_epi32((__m256i)wm[k], idx_hi);
        X[2 * k] = _mm256_blend_epi32(pl, ml, 0xF0);
        X[2 * k + 1] = _mm256_blend_epi32(ph, mh, 0xF0);
    }
    /* parity of y */
    __m256i par = _mm256_blend_epi32(_mm256_permutevar8x32_epi32((__m256i)yp->n[0], idx_lo),
                                     _mm256_permutevar8x32_epi32((__m256i)ym->n[0], idx_lo), 0xF0);
    par = _mm256_and_si256(par, _mm256_set1_epi32(1));
    v8u W[16];
    W[0] = _mm256_or_si256(_mm256_slli_epi32(_mm256_or_si256(par, _mm256_set1_epi32(2)), 24), _mm256_srli_epi32(X[7], 8));
    for (int j = 1; j < 8; j++) W[j] = _mm256_or_si256(_mm256_slli_epi32(X[8 - j], 24), _mm256_srli_epi32(X[7 - j], 8));
    W[8] = _mm256_or_si256(_mm256_slli_epi32(X[0], 24), _mm256_set1_epi32(0x00800000));
    for (int j = 9; j < 15; j++) W[j] = _mm256_setzero_si256();
    W[15] = _mm256_set1_epi32(264);
    __m256i h0;
    if (use_ni) {
        alignas(32) uint32_t Wt[9][8], hh[8];
        for (int j = 0; j < 9; j++) _mm256_store_si256((__m256i *)Wt[j], W[j]);
        pub_hash8_shani_wm(Wt, hh);
        h0 = _mm256_load_si256((const __m256i *)hh);
    } else {
        v8u st[8];
        for (int j = 0; j < 8; j++) st[j] = _mm256_set1_epi32((int)IV256[j]);
        s8_compress_full(st, W);
        h0 = st[0];
    }
#if QSB_ZEROS_N >= 32
    __m256i ok = _mm256_cmpeq_epi32(h0, _mm256_setzero_si256());
#else
    __m256i ok = _mm256_cmpeq_epi32(_mm256_srli_epi32(h0, 32 - QSB_ZEROS_N), _mm256_setzero_si256());
#endif
    return (unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(ok));
}

static QV_INL V zmask4(const uint32_t *d4) {
    V m; for (int l = 0; l < 4; l++) m[l] = QCG_ZERO(d4[l]) ? ~0ULL : 0; return m;
}
static QV_INL V negmask4(const uint32_t *d4) {
    const __m128i c = _mm_loadu_si128((const __m128i *)d4);
    return (V)_mm256_cvtepi32_epi64(_mm_srai_epi32(c, 31));      /* bit 31 -> all ones */
}
static QV_INL void entries4(const tentry **e, const tentry *Tj, const uint32_t *d4) {
    e[0] = Tj + (d4[0] & QCG_IDXM); e[1] = Tj + (d4[1] & QCG_IDXM); e[2] = Tj + (d4[2] & QCG_IDXM); e[3] = Tj + (d4[3] & QCG_IDXM);
}
static QV_INL void prefetch4(const tentry *Tj, const uint32_t *d4) {
    for (int l = 0; l < 4; l++) _mm_prefetch((const char *)(Tj + (d4[l] & QCG_IDXM)), _MM_HINT_T0);
}

/* per-block temporaries of one addition step (memory operands of the out-of-line multiplies) */
struct bst {
    vfe xT, dx, dy, ik, lam, l2, t, y3, dxn;
};
/* backward part before the multiplications: gather T_s; xT and dx = xT - px (1 on zero lanes);
 * dy = +-yT - py */
static QV_INL void bk_pre(bst &c, const tentry *Tj, const uint32_t *d4, uint8_t zf, const vfe &px, const vfe &py, const vfe &one) {
    const tentry *e[4]; entries4(e, Tj, d4);
    V wx[4], wy[4];
    tr4(wx, e[0], e[1], e[2], e[3], 0); tr4(wy, e[0], e[1], e[2], e[3], 1);
    v26_from_w(&c.xT, wx[0], wx[1], wx[2], wx[3]);
    v26_subm(&c.dx, &c.xT, &px, 1);                                     /* mag 3 */
    if (__builtin_expect(zf, 0)) v26_sel(&c.dx, zmask4(d4), &one, &c.dx);
    vfe yT, ny; v26_from_w(&yT, wy[0], wy[1], wy[2], wy[3]);
    const V sm = negmask4(d4);
    v26_neg(&ny, &yT, 1);
    for (int k = 0; k < 10; k++) {                                      /* dy = (neg ? -yT : yT) - py, mag <= 4 */
        const V y = (V)_mm256_blendv_epi8((__m256i)yT.n[k], (__m256i)ny.n[k], (__m256i)sm);
        c.dy.n[k] = y;
    }
    v26_subm(&c.dy, &c.dy, &py, 1);
}
/* x3 = l2 - px - xT (normalised weakly, written to px unless a zero digit keeps the old point);
 * t = px_old - x3 */
static QV_INL void bk_x3(bst &c, vfe &px, const uint32_t *d4, uint8_t zf) {
    vfe x3;
    v26_subm(&x3, &c.l2, &px, 1); v26_subm(&x3, &x3, &c.xT, 1);         /* mag 5 */
    v26_nweak(&x3);                                                      /* mag 1 */
    v26_subm(&c.t, &px, &x3, 1);                                         /* mag 3 */
    if (__builtin_expect(zf, 0)) v26_sel(&x3, zmask4(d4), &px, &x3);
    for (int k = 0; k < 10; k++) px.n[k] = x3.n[k];
}
/* y3 = lam t - py (weakly normalised) -> py, keeping the old y on zero lanes */
static QV_INL void bk_y3(bst &c, vfe &py, const uint32_t *d4, uint8_t zf) {
    vfe y3;
    v26_subm(&y3, &c.y3, &py, 1);
    v26_nweak(&y3);
    if (__builtin_expect(zf, 0)) v26_sel(&y3, zmask4(d4), &py, &y3);
    for (int k = 0; k < 10; k++) py.n[k] = y3.n[k];
}
/* forward part of the next step: dxn = xN - x (1 on zero-digit lanes) */
static QV_INL void fw_next(bst &c, const tentry *Tn, const uint32_t *dn4, uint8_t zfn, const vfe &x, const vfe &one) {
    const tentry *e[4]; entries4(e, Tn, dn4);
    vfe xN; gather_x(&xN, e);
    v26_subm(&c.dxn, &xN, &x, 1);
    if (__builtin_expect(zfn, 0)) { const V m = zmask4(dn4); v26_sel(&c.dxn, m, &one, &c.dxn); }
}

struct vstate {
    vfe px[QSB_CG_BMAX / 4 + 4], py[QSB_CG_BMAX / 4 + 4], c[QSB_CG_BMAX / 4 + 4];
    bst tmp;
};

/* One batch: Q+ = z B + A, Q- = Q+ - 2A for the w->n candidates, hash, publish.
 * One product chain over the 4-candidate blocks; pass s runs backward(s) and forward(s+1)
 * fused, alternating the block order. */
static QV_FN void ec_batch(worker_t *w, vstate *vs) {
    shared_t *S = g_cg;
    const int use_ni = S->sha_mode.load(std::memory_order_relaxed) == 2;
    const layout_t &L = S->lay;
    const int n = w->n;
    const int nb = (n + 3) >> 2;
    const tentry *T = S->table;
    vfe *px = vs->px, *py = vs->py, *cc = vs->c;
    vfe one; v26_one(&one);
    const int nw = L.nwin;
    {
        const uint32_t *dg = w->dig[0];
        const tentry *T0 = T + L.off[0];
        for (int b = 0; b < nb; b++) {
            const tentry *e[4] = {T0 + dg[4 * b], T0 + dg[4 * b + 1], T0 + dg[4 * b + 2], T0 + dg[4 * b + 3]};
            if (b + QSB_CG_PF < nb) for (int l = 0; l < 4; l++) _mm_prefetch((const char *)(T0 + dg[4 * (b + QSB_CG_PF) + l]), _MM_HINT_T0);
            gather_x(&px[b], e); gather_y(&py[b], e);
        }
    }
    vfe acc, inv;
    {
        const uint32_t *dg = w->dig[1];
        const uint8_t *zf = w->zf[1];
        const tentry *Tj = T + L.off[1];
        for (int b = 0; b < nb; b++) {
            if (b + QSB_CG_PF < nb) prefetch4(Tj, dg + 4 * (b + QSB_CG_PF));
            const tentry *e[4]; entries4(e, Tj, dg + 4 * b);
            vfe xT, dx; gather_x(&xT, e);
            v26_subm(&dx, &xT, &px[b], 1);
            if (zf[b]) v26_sel(&dx, zmask4(dg + 4 * b), &one, &dx);
            v26_mul_o(&cc[b], b ? &cc[b - 1] : &one, &dx);
        }
        acc = cc[nb - 1];
    }
    vfe xD, yD; v26_set_w(&xD, S->dx_w); v26_set_w(&yD, S->dy_w);
    int asc = 1;
    for (int s = 1; s < nw; s++) {
        vinv(&inv, &acc);
        vfe ubuf[2]; ubuf[0] = inv; int ui = 0;
        const vfe *accp = &one;
        const uint32_t *dg = w->dig[s];
        const uint8_t *zf = w->zf[s];
        const tentry *Tj = T + L.off[s];
        const int last = (s + 1 == nw);
        const uint32_t *dgn = last ? NULL : w->dig[s + 1];
        const uint8_t *zfn = last ? NULL : w->zf[s + 1];
        const tentry *Tn = last ? NULL : T + L.off[s + 1];
        const int step = asc ? -1 : 1;
        int b = asc ? nb - 1 : 0;
        for (int it = 0; it < nb; it++, b += step) {
            const int bp = b + step;
            const int bf = b + step * QSB_CG_PF;
            if (bf >= 0 && bf < nb) { prefetch4(Tj, dg + 4 * bf); if (!last) prefetch4(Tn, dgn + 4 * bf); }
            bst &c = vs->tmp;
            bk_pre(c, Tj, dg + 4 * b, zf[b], px[b], py[b], one);
            const vfe *ikp;
            if (it + 1 < nb) { v26_mul_o(&c.ik, &ubuf[ui], &cc[bp]); v26_mul_o(&ubuf[ui ^ 1], &ubuf[ui], &c.dx); ui ^= 1; ikp = &c.ik; }
            else ikp = &ubuf[ui];
            v26_mul_o(&c.lam, &c.dy, ikp);
            v26_sqr_o(&c.l2, &c.lam);
            bk_x3(c, px[b], dg + 4 * b, zf[b]);
            v26_mul_o(&c.y3, &c.lam, &c.t);
            bk_y3(c, py[b], dg + 4 * b, zf[b]);
            if (!last) fw_next(c, Tn, dgn + 4 * b, zfn[b], px[b], one);
            else v26_subm(&c.dxn, &xD, &px[b], 1);
            v26_mul_o(&cc[b], accp, &c.dxn);
            accp = &cc[b];
        }
        acc = *accp;
        asc = !asc;
    }
    vinv(&inv, &acc);
    {
        vfe ubuf[2]; ubuf[0] = inv; int ui = 0;
        const int step = asc ? -1 : 1;
        int b = asc ? nb - 1 : 0;
        for (int it = 0; it < nb; it++, b += step) {
            const int bp = b + step;
            vfe dx, dy, ik, lam, l2, xm, t, ym;
            v26_subm(&dx, &xD, &px[b], 1);
            const vfe *ikp;
            if (it + 1 < nb) { v26_mul_o(&ik, &ubuf[ui], &cc[bp]); v26_mul_o(&ubuf[ui ^ 1], &ubuf[ui], &dx); ui ^= 1; ikp = &ik; }
            else ikp = &ubuf[ui];
            v26_subm(&dy, &yD, &py[b], 1);
            v26_mul_o(&lam, &dy, ikp);
            v26_sqr_o(&l2, &lam);
            v26_subm(&xm, &l2, &px[b], 1); v26_subm(&xm, &xm, &xD, 1);
            v26_subm(&t, &px[b], &xm, 5);
            v26_mul_o(&ym, &lam, &t);
            v26_subm(&ym, &ym, &py[b], 1);
            vfe xp = px[b], yp = py[b];
            v26_norm(&xp); v26_norm(&yp); v26_norm(&xm); v26_norm(&ym);
#ifdef QCG_EC_HOOK
            { V a4[4], b4[4], c4[4], d4[4]; v26_to_w(a4, &xp); v26_to_w(b4, &yp); v26_to_w(c4, &xm); v26_to_w(d4, &ym);
              for (int l = 0; l < 4; l++) { uint64_t X0[4], Y0[4], X1[4], Y1[4];
                  for (int q = 0; q < 4; q++) { X0[q] = a4[q][l]; Y0[q] = b4[q][l]; X1[q] = c4[q][l]; Y1[q] = d4[q][l]; }
                  if (4 * b + l < n) QCG_EC_HOOK(w, 4 * b + l, X0, Y0, X1, Y1); } }
#endif
            const unsigned hm = hash_block(&xp, &yp, &xm, &ym, use_ni);
            if (hm) {
                for (int l = 0; l < 8; l++) if (hm >> l & 1) {
                    const int ci = 4 * b + (l & 3);
                    if (ci < n) publish(w, ci, l >> 2);
                }
            }
        }
    }
}
#undef VMUL
#undef QV_INL
#undef QV_FN
} /* namespace v4 */
