/* cpu_cogrind_vec.h -- AVX2 elliptic-curve stage of the pinning co-grinder (4 candidates per
 * vector, one per 64-bit lane). Included once by cpu_cogrind.h inside namespace qcg.
 *
 * Field arithmetic: radix 2^29, 9 limbs per element (limb i at bit 29 i), one element per 64-bit
 * lane. v29_mul / v29_sqr are generated straight-line programs (cg_v29asm.h, gen/gen_v29.py):
 * 81 (45) VPMULUDQ limb products, then a libsecp256k1-style interleaved reduction with
 * 2^261 = 2^37 + 31264 (mod p). The generator proves the lane bounds (no 64-bit overflow, every
 * multiplier operand < 2^32) for the input classes used below and self-tests the programs.
 * Value classes (limb bounds, limbs 0..7 / limb 8):
 *   NORM  mul/sqr output, weakly normalised value, table coordinate: <= 2^29 + 2^10 / 2^24 + 2^10
 *   LAZY  f p + x - y (f <= 2) or f p - y + f p - z of NORM values:     <= 5 * 2^29 / 2^27
 * A multiplication takes one NORM and one LAZY operand (either order); a square a NORM operand.
 * The normalisation helpers follow libsecp256k1's fe_normalize(_weak) (Copyright (c) 2013 Pieter
 * Wuille, MIT license -- notice in COPYING-secp256k1) adapted to radix 2^29.
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
struct vfe { V n[9]; };

static QV_INL V vs1(uint64_t x) { return (V){x, x, x, x}; }
#define VMUL(a, b) ((V)_mm256_mul_epu32((__m256i)(a), (__m256i)(b)))
#define QCG_V29_WANT_C
#include "cg_v29asm.h"
#undef QCG_V29_WANT_C

#define QV_M29 0x1FFFFFFFULL
#define QV_M24 0xFFFFFFULL

#ifndef QCG_V29_ASM
#define QCG_V29_ASM 1
#endif
#if QCG_V29_ASM
/* constant table of the generated programs (cg_v29asm.h) */
alignas(32) static const uint64_t V29K[QCG_V29_KTAB_QWORDS] = {QCG_V29_KTAB_INIT};
/* out-of-line asm (same programs as v29_mul_c / v29_sqr_c); r must not alias b */
static QV_FN void v29_mul_o(vfe *r, const vfe *a, const vfe *b) {
    __asm__ volatile(QCG_V29_MUL_ASM : : [r] "r"(r), [a] "r"(a), [b] "r"(b), [k] "r"(V29K)
        : "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15", "memory");
}
static QV_FN void v29_sqr_o(vfe *r, const vfe *a) {
    __asm__ volatile(QCG_V29_SQR_ASM : : [r] "r"(r), [a] "r"(a), [k] "r"(V29K)
        : "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15", "memory");
}
#else
static QV_FN void v29_mul_o(vfe *r, const vfe *a, const vfe *b) { v29_mul_c(r, a, b); }
static QV_FN void v29_sqr_o(vfe *r, const vfe *a) { v29_sqr_c(r, a); }
#endif
/* radix-2^29 digits of p: p0, p1, p2..p7, p8 */
#define QV_P0 0x1FFFFC2FULL
#define QV_P1 0x1FFFFFF7ULL
/* r = a + (f p - b); needs f p_i >= b_i (f = 2 covers every NORM b) */
static QV_INL void v29_subm(vfe *r, const vfe *a, const vfe *b, int f) {
    const uint64_t ff = (uint64_t)f;
    r->n[0] = a->n[0] + (vs1(QV_P0 * ff) - b->n[0]);
    r->n[1] = a->n[1] + (vs1(QV_P1 * ff) - b->n[1]);
    for (int k = 2; k < 8; k++) r->n[k] = a->n[k] + (vs1(QV_M29 * ff) - b->n[k]);
    r->n[8] = a->n[8] + (vs1(QV_M24 * ff) - b->n[8]);
}
/* r = a + (4 p - b - c) = a - b - c; needs 4 p_i >= b_i + c_i (two NORM operands) */
static QV_INL void v29_sub2(vfe *r, const vfe *a, const vfe *b, const vfe *c) {
    r->n[0] = a->n[0] + ((vs1(QV_P0 * 4) - b->n[0]) - c->n[0]);
    r->n[1] = a->n[1] + ((vs1(QV_P1 * 4) - b->n[1]) - c->n[1]);
    for (int k = 2; k < 8; k++) r->n[k] = a->n[k] + ((vs1(QV_M29 * 4) - b->n[k]) - c->n[k]);
    r->n[8] = a->n[8] + ((vs1(QV_M24 * 4) - b->n[8]) - c->n[8]);
}
/* r = f p - a */
static QV_INL void v29_neg(vfe *r, const vfe *a, int f) {
    const uint64_t ff = (uint64_t)f;
    r->n[0] = vs1(QV_P0 * ff) - a->n[0];
    r->n[1] = vs1(QV_P1 * ff) - a->n[1];
    for (int k = 2; k < 8; k++) r->n[k] = vs1(QV_M29 * ff) - a->n[k];
    r->n[8] = vs1(QV_M24 * ff) - a->n[8];
}
/* weak normalisation: limbs 0..7 <= 2^29 - 1, limb 8 <= 2^24 + small (libsecp256k1 fe_normalize_weak) */
static QV_INL void v29_nweak(vfe *r) {
    const V M = vs1(QV_M29);
    V t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4], t5 = r->n[5], t6 = r->n[6], t7 = r->n[7], t8 = r->n[8];
    const V x = t8 >> 24; t8 &= vs1(QV_M24);
    t0 += VMUL(x, vs1(977ULL)); t1 += (x << 3);          /* 2^256 = 2^32 + 977, 2^32 = 8 * 2^29 */
    t1 += (t0 >> 29); t0 &= M;
    t2 += (t1 >> 29); t1 &= M;
    t3 += (t2 >> 29); t2 &= M;
    t4 += (t3 >> 29); t3 &= M;
    t5 += (t4 >> 29); t4 &= M;
    t6 += (t5 >> 29); t5 &= M;
    t7 += (t6 >> 29); t6 &= M;
    t8 += (t7 >> 29); t7 &= M;
    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4; r->n[5] = t5; r->n[6] = t6; r->n[7] = t7; r->n[8] = t8;
}
/* Exact v29_mul_o(one, denominator) specialisation for the LAZY
 * denominator bounds (low limbs < 3*2^29, top < 3*2^24).
 * Carry the radix-29 input first, then reproduce the generated multiplier's
 * final fold into limbs 0..2. This preserves its exact redundant limbs. */
static QV_INL void v29_seed_product(vfe *r, const vfe *d) {
    const V M = vs1(QV_M29);
    V t = d->n[0];
    for (int k = 0; k < 8; k++) {
        r->n[k] = t & M;
        t = d->n[k + 1] + (t >> 29);
    }
    r->n[8] = t & vs1(QV_M24);
    const V high = t >> 24;
    t = r->n[0] + VMUL(high, vs1(977));
    r->n[0] = t & M;
    t = r->n[1] + (t >> 29) + (high << 3);
    r->n[1] = t & M;
    r->n[2] += t >> 29;
}
/* canonical value in [0, p) (libsecp256k1 fe_normalize) */
static QV_INL void v29_norm(vfe *r) {
    const V M = vs1(QV_M29), one = vs1(1);
    V t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4], t5 = r->n[5], t6 = r->n[6], t7 = r->n[7], t8 = r->n[8];
    V m;
    V x = t8 >> 24; t8 &= vs1(QV_M24);
    t0 += VMUL(x, vs1(977ULL)); t1 += (x << 3);
    t1 += (t0 >> 29); t0 &= M;
    t2 += (t1 >> 29); t1 &= M;
    t3 += (t2 >> 29); t2 &= M; m = t2;
    t4 += (t3 >> 29); t3 &= M; m &= t3;
    t5 += (t4 >> 29); t4 &= M; m &= t4;
    t6 += (t5 >> 29); t5 &= M; m &= t5;
    t7 += (t6 >> 29); t6 &= M; m &= t6;
    t8 += (t7 >> 29); t7 &= M; m &= t7;
    /* value >= p: an overflow bit at 2^256, or limbs 2..8 all ones and the low 58 bits >= p's */
    x = (t8 >> 24) | ((V)(t8 == vs1(QV_M24)) & (V)(m == M)
        & (V)((t1 + vs1(8ULL) + ((t0 + vs1(977ULL)) >> 29)) > M) & one);
    t0 += VMUL(x, vs1(977ULL)); t1 += (x << 3);
    t1 += (t0 >> 29); t0 &= M;
    t2 += (t1 >> 29); t1 &= M;
    t3 += (t2 >> 29); t2 &= M;
    t4 += (t3 >> 29); t3 &= M;
    t5 += (t4 >> 29); t4 &= M;
    t6 += (t5 >> 29); t5 &= M;
    t7 += (t6 >> 29); t6 &= M;
    t8 += (t7 >> 29); t7 &= M;
    t8 &= vs1(QV_M24);
    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4; r->n[5] = t5; r->n[6] = t6; r->n[7] = t7; r->n[8] = t8;
}
/* Weakly normalised inputs need at most one subtraction of p. */
static QV_INL V v29_ge_p(const vfe *a) {
    const V M = vs1(QV_M29), one = vs1(1);
    const V *t = a->n;
    V m = t[2]; for (int k = 3; k < 8; k++) m &= t[k];
    return (t[8] >> 24) | ((V)(t[8] == vs1(QV_M24)) & (V)(m == M)
        & (V)((t[1] + vs1(8ULL) + ((t[0] + vs1(977ULL)) >> 29)) > M) & one);
}
static QV_INL void v29_norm_weak(vfe *r) {
    const V high = (r->n[8] + vs1(1)) >> 24;          /* top < 2^24 - 1 already implies < p */
    if (__builtin_expect(_mm256_testz_si256((__m256i)high, (__m256i)high), 1)) return;
    const V M = vs1(QV_M29), x = v29_ge_p(r);
    V t0 = r->n[0] + VMUL(x, vs1(977ULL)), t1 = r->n[1] + (x << 3);
    V t2 = r->n[2], t3 = r->n[3], t4 = r->n[4], t5 = r->n[5], t6 = r->n[6], t7 = r->n[7], t8 = r->n[8];
    t1 += (t0 >> 29); t0 &= M;
    t2 += (t1 >> 29); t1 &= M;
    t3 += (t2 >> 29); t2 &= M;
    t4 += (t3 >> 29); t3 &= M;
    t5 += (t4 >> 29); t4 &= M;
    t6 += (t5 >> 29); t5 &= M;
    t7 += (t6 >> 29); t6 &= M;
    t8 += (t7 >> 29); t7 &= M;
    t8 &= vs1(QV_M24);
    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4; r->n[5] = t5; r->n[6] = t6; r->n[7] = t7; r->n[8] = t8;
}
/* Subtracting odd p flips bit 0; hashing needs no other bits of canonical y. */
static QV_INL V v29_parity(const vfe *a) { return (a->n[0] ^ v29_ge_p(a)) & vs1(1); }

/* Hashing needs only canonical y parity: yp is weakly normalised and ymul
 * is the NORM product before subtracting yp. For d = ymul + 2p - yp,
 * limbs 0..7 < 3*2^29 + 2^10 and limb 8 < 3*2^24 + 2^10.
 * Folding x = d8 >> 24 (x <= 3) leaves every carry <= 3. Thus a
 * masked top limb < 2^24 - 4 guarantees the weak result is < p.
 * Its parity is (ymul0 ^ yp0 ^ x) & 1, since 2p is even and 977 odd. */
static QV_INL void v29_pub_parity(V *pp, V *pm, const vfe *yp, const vfe *ymul) {
    const V top = ymul->n[8] + (vs1(2 * QV_M24) - yp->n[8]);
    const V high = ((yp->n[8] + vs1(1)) >> 24)
        | (((top & vs1(QV_M24)) + vs1(4)) >> 24);
    if (__builtin_expect(_mm256_testz_si256((__m256i)high, (__m256i)high), 1)) {
        *pp = yp->n[0] & vs1(1);
        *pm = (ymul->n[0] ^ yp->n[0] ^ (top >> 24)) & vs1(1);
        return;
    }
    /* Rare boundary lanes: preserve the original subtraction and normalisation. */
    *pp = v29_parity(yp);
    vfe ym;
    v29_subm(&ym, ymul, yp, 2);
    v29_nweak(&ym);
    *pm = v29_parity(&ym);
}
/* r = mask ? a : b (per lane, mask all-ones or zero) */
static QV_INL void v29_sel(vfe *r, V mask, const vfe *a, const vfe *b) {
    for (int k = 0; k < 9; k++) r->n[k] = (V)_mm256_blendv_epi8((__m256i)b->n[k], (__m256i)a->n[k], (__m256i)mask);
}
static QV_INL void v29_one(vfe *r) { r->n[0] = vs1(1); for (int k = 1; k < 9; k++) r->n[k] = vs1(0); }
/* 4 little-endian 64-bit words per lane -> 9 x 29 (input < 2^256) */
static QV_INL void v29_from_w(vfe *r, V w0, V w1, V w2, V w3) {
    const V M = vs1(QV_M29);
    r->n[0] = w0 & M;
    r->n[1] = (w0 >> 29) & M;
    r->n[2] = ((w0 >> 58) | (w1 << 6)) & M;
    r->n[3] = (w1 >> 23) & M;
    r->n[4] = ((w1 >> 52) | (w2 << 12)) & M;
    r->n[5] = (w2 >> 17) & M;
    r->n[6] = ((w2 >> 46) | (w3 << 18)) & M;
    r->n[7] = (w3 >> 11) & M;
    r->n[8] = w3 >> 40;
}
/* normalized 9 x 29 -> 4 words per lane */
static QV_INL void v29_to_w(V w[4], const vfe *a) {
    const V *n = a->n;
    w[0] = n[0] | (n[1] << 29) | (n[2] << 58);
    w[1] = (n[2] >> 6) | (n[3] << 23) | (n[4] << 52);
    w[2] = (n[4] >> 12) | (n[5] << 17) | (n[6] << 46);
    w[3] = (n[6] >> 18) | (n[7] << 11) | (n[8] << 40);
}
/* broadcast a 4x64 constant */
static QV_INL void v29_set_w(vfe *r, const uint64_t *w) { v29_from_w(r, vs1(w[0]), vs1(w[1]), vs1(w[2]), vs1(w[3])); }
/* r lane l = a lane (l ^ k) */
static QV_INL void v29_permx(vfe *r, const vfe *a, int k) {
    for (int j = 0; j < 9; j++)
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
static QV_INL void gather_x(vfe *x, const tentry *const *e) { V w[4]; tr4(w, e[0], e[1], e[2], e[3], 0); v29_from_w(x, w[0], w[1], w[2], w[3]); }
static QV_INL void gather_y(vfe *y, const tentry *const *e) { V w[4]; tr4(w, e[0], e[1], e[2], e[3], 1); v29_from_w(y, w[0], w[1], w[2], w[3]); }

/* per-worker vector state */
struct bst;
struct vstate;

/* inv lane l = 1 / m lane l, for all 4 lanes, with one scalar inversion:
 * q = product of the 4 lanes (butterfly), oth = product of the other 3 lanes, inv = oth / q */
static QV_INL void vinv(vfe *inv, const vfe *m) {
    vfe s, q1, q2, oth, t;
    v29_permx(&s, m, 1); oth = s; v29_mul_o(&q1, m, &s);          /* q1 lane l = m_l m_(l^1) */
    v29_permx(&s, &q1, 2); v29_mul_o(&t, &oth, &s); v29_mul_o(&q2, &q1, &s);
    oth = t;
    v29_norm(&q2);
    V w[4]; v29_to_w(w, &q2);
    uint64_t tw[4] = {w[0][0], w[1][0], w[2][0], w[3][0]}, iw[4];
    scalar_inv(iw, tw);
    v29_set_w(&t, iw);
    v29_mul_o(inv, &t, &oth);
}

/* Low 32 bits of p and m, in lane order p0,p1,p2,p3,m0,m1,m2,m3. */
static QV_INL __m256i hash_pack32(V p, V m) {
    const __m256i t = _mm256_castps_si256(_mm256_shuffle_ps(
        _mm256_castsi256_ps((__m256i)p), _mm256_castsi256_ps((__m256i)m), 0x88));
    return _mm256_permute4x64_epi64(t, 0xD8);
}

/* Pubkey-only compression: retain W, add K at round consumption instead of
 * materialising and rereading a second 64-vector KW array. */
static QV_INL __m256i hash_pub8_single_schedule(const qcg_sha::v8u *Wv) {
    using namespace qcg_sha;
    const s8_plan &P = S8_PLAN_PUBKEY;
    constexpr uint64_t V = s8_varmask(0x1FFu);
    v8u W[64];
    for (int j = 0; j < 9; j++) W[j] = Wv[j];
    for (int j = 9; j < 15; j++) W[j] = _mm256_setzero_si256();
    W[15] = s8_set1(264);
#define QV_PUB_W(t) do { \
        v8u acc_; int has_ = 0; \
        if ((V >> ((t) - 2)) & 1) { acc_ = s8_s1(W[(t) - 2]); has_ = 1; } \
        if ((V >> ((t) - 7)) & 1) { acc_ = has_ ? s8_add(acc_, W[(t) - 7]) : W[(t) - 7]; has_ = 1; } \
        if ((V >> ((t) - 15)) & 1) { const v8u s_ = s8_s0(W[(t) - 15]); acc_ = has_ ? s8_add(acc_, s_) : s_; has_ = 1; } \
        if ((V >> ((t) - 16)) & 1) { acc_ = has_ ? s8_add(acc_, W[(t) - 16]) : W[(t) - 16]; has_ = 1; } \
        if (((V >> ((t) - 2)) & (V >> ((t) - 7)) & (V >> ((t) - 15)) & (V >> ((t) - 16)) & 1) == 0) acc_ = s8_add(acc_, s8_set1(P.kc[t])); \
        W[t] = acc_; } while (0)
    QV_PUB_W(16); QV_PUB_W(17); QV_PUB_W(18); QV_PUB_W(19); QV_PUB_W(20); QV_PUB_W(21); QV_PUB_W(22); QV_PUB_W(23);
    QV_PUB_W(24); QV_PUB_W(25); QV_PUB_W(26); QV_PUB_W(27); QV_PUB_W(28); QV_PUB_W(29); QV_PUB_W(30); QV_PUB_W(31);
#undef QV_PUB_W
    for (int t = 32; t < 64; t++)
        W[t] = s8_add(s8_add(s8_s1(W[t - 2]), W[t - 7]), s8_add(s8_s0(W[t - 15]), W[t - 16]));
    v8u a = s8_set1(P.st0[0]), b = s8_set1(P.st0[1]), c = s8_set1(P.st0[2]);
    v8u d = s8_add(W[0], s8_set1(P.e1c)), e = s8_set1(P.st0[4]);
    v8u f = s8_set1(P.st0[5]), g = s8_set1(P.st0[6]), h = s8_add(W[0], s8_set1(P.a1c));
    v8u bc = s8_set1(P.st0[0] ^ P.st0[1]);
#define QV_PUB_ROUND(a, b, c, d, e, f, g, h, kw, bc) do { \
        v8u t1_ = s8_add(s8_add(h, s8_S1(e)), s8_add(s8_xor(g, s8_and(e, s8_xor(f, g))), (kw))); \
        v8u ab_ = s8_xor(a, b); \
        v8u t2_ = s8_add(s8_S0(a), s8_xor(s8_and(ab_, bc), b)); \
        d = s8_add(d, t1_); h = s8_add(t1_, t2_); bc = ab_; \
    } while (0)
#define QV_PUB_KW(t) s8_add(s8_set1(K256[t]), W[t])
    QV_PUB_ROUND(h, a, b, c, d, e, f, g, QV_PUB_KW(1), bc);
    QV_PUB_ROUND(g, h, a, b, c, d, e, f, QV_PUB_KW(2), bc);
    QV_PUB_ROUND(f, g, h, a, b, c, d, e, QV_PUB_KW(3), bc);
    QV_PUB_ROUND(e, f, g, h, a, b, c, d, QV_PUB_KW(4), bc);
    QV_PUB_ROUND(d, e, f, g, h, a, b, c, QV_PUB_KW(5), bc);
    QV_PUB_ROUND(c, d, e, f, g, h, a, b, QV_PUB_KW(6), bc);
    QV_PUB_ROUND(b, c, d, e, f, g, h, a, QV_PUB_KW(7), bc);
#define QV_PUB_R8(t) \
    QV_PUB_ROUND(a, b, c, d, e, f, g, h, QV_PUB_KW((t) + 0), bc); \
    QV_PUB_ROUND(h, a, b, c, d, e, f, g, QV_PUB_KW((t) + 1), bc); \
    QV_PUB_ROUND(g, h, a, b, c, d, e, f, QV_PUB_KW((t) + 2), bc); \
    QV_PUB_ROUND(f, g, h, a, b, c, d, e, QV_PUB_KW((t) + 3), bc); \
    QV_PUB_ROUND(e, f, g, h, a, b, c, d, QV_PUB_KW((t) + 4), bc); \
    QV_PUB_ROUND(d, e, f, g, h, a, b, c, QV_PUB_KW((t) + 5), bc); \
    QV_PUB_ROUND(c, d, e, f, g, h, a, b, QV_PUB_KW((t) + 6), bc); \
    QV_PUB_ROUND(b, c, d, e, f, g, h, a, QV_PUB_KW((t) + 7), bc);
    for (int t = 8; t < 64; t += 8) {
        QV_PUB_R8(t)
    }
#undef QV_PUB_R8
#undef QV_PUB_KW
#undef QV_PUB_ROUND
    return s8_add(s8_set1(P.st0[0]), a);
}

/* the 8-lane pubkey SHA of Q+ (lanes 0-3) and Q- (lanes 4-7) of one block; returns an 8-bit
 * mask of lanes whose H0 passes the leading-zero prefilter */
static QV_INL unsigned hash_block(const vfe *xp, V yp, const vfe *xm, V ym, int use_ni) {
    using namespace qcg_sha;
    /* Canonical limbs fit in 32 bits. Pack first, then form the big-endian
     * 32-bit words of the 33-byte compressed pubkey without a 4x64 detour. */
    v8u W[16];
    v8u lo = hash_pack32(xp->n[0], xm->n[0]);
    W[8] = _mm256_or_si256(_mm256_slli_epi32(lo, 24), _mm256_set1_epi32(0x00800000));
    v8u hi = hash_pack32(xp->n[1], xm->n[1]);
    W[7] = _mm256_or_si256(_mm256_srli_epi32(lo, 8), _mm256_slli_epi32(hi, 21));
    lo = hi; hi = hash_pack32(xp->n[2], xm->n[2]);
    W[6] = _mm256_or_si256(_mm256_srli_epi32(lo, 11), _mm256_slli_epi32(hi, 18));
    lo = hi; hi = hash_pack32(xp->n[3], xm->n[3]);
    W[5] = _mm256_or_si256(_mm256_srli_epi32(lo, 14), _mm256_slli_epi32(hi, 15));
    lo = hi; hi = hash_pack32(xp->n[4], xm->n[4]);
    W[4] = _mm256_or_si256(_mm256_srli_epi32(lo, 17), _mm256_slli_epi32(hi, 12));
    lo = hi; hi = hash_pack32(xp->n[5], xm->n[5]);
    W[3] = _mm256_or_si256(_mm256_srli_epi32(lo, 20), _mm256_slli_epi32(hi, 9));
    lo = hi; hi = hash_pack32(xp->n[6], xm->n[6]);
    W[2] = _mm256_or_si256(_mm256_srli_epi32(lo, 23), _mm256_slli_epi32(hi, 6));
    lo = hi; hi = hash_pack32(xp->n[7], xm->n[7]);
    W[1] = _mm256_or_si256(_mm256_srli_epi32(lo, 26), _mm256_slli_epi32(hi, 3));
    const v8u par = hash_pack32(yp, ym);   /* Both inputs are exactly 0 or 1. */
    W[0] = _mm256_or_si256(hash_pack32(xp->n[8], xm->n[8]),
        _mm256_or_si256(_mm256_slli_epi32(par, 24), _mm256_set1_epi32(0x02000000)));
    for (int j = 9; j < 15; j++) W[j] = _mm256_setzero_si256();
    W[15] = _mm256_set1_epi32(264);
    __m256i h0;
    if (use_ni) {
        alignas(32) uint32_t Wt[9][8], hh[8];
        for (int j = 0; j < 9; j++) _mm256_store_si256((__m256i *)Wt[j], W[j]);
        pub_hash8_shani_wm(Wt, hh);
        h0 = _mm256_load_si256((const __m256i *)hh);
    } else {
        h0 = hash_pub8_single_schedule(W);
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
    vfe dy, ik, lam, l2, t, y3;
};
/* backward preparation: dx and packed yT are retained from the forward pass.
 * dy = +-yT - py */
static QV_INL void bk_pre(bst &c, const V wy[4], const uint32_t *d4, const vfe &py) {
    vfe yT; v29_from_w(&yT, wy[0], wy[1], wy[2], wy[3]);
    const V sm = negmask4(d4), m29 = sm & vs1(QV_M29), m24 = sm & vs1(QV_M24);
    const V b29 = vs1(2 * QV_M29) + m29, b24 = vs1(2 * QV_M24) + m24;
    /* yT is packed: (y ^ M) = M - y. Bias preserves the original dy limbs exactly. */
    c.dy.n[0] = (yT.n[0] ^ m29) + (vs1(2 * QV_P0) + (sm & vs1(2 * QV_P0 - QV_M29))) - py.n[0];
    c.dy.n[1] = (yT.n[1] ^ m29) + (vs1(2 * QV_P1) + (sm & vs1(2 * QV_P1 - QV_M29))) - py.n[1];
    for (int k = 2; k < 8; k++) c.dy.n[k] = (yT.n[k] ^ m29) + b29 - py.n[k];
    c.dy.n[8] = (yT.n[8] ^ m24) + b24 - py.n[8];
}
/* x3 = l2 - px - xT (normalised weakly, written to px unless a zero digit keeps the old point);
 * t = px_old - x3 */
static QV_INL void bk_x3(bst &c, vfe &px, const vfe &dx, const uint32_t *d4, uint8_t zf) {
    vfe x3;
    /* dx = xT + 2p - px on nonzero lanes, hence x3 = l2 + 6p - 2px - dx.
     * Zero lanes have dx = 1; their raw limbs remain < 2^32 and are discarded below. */
    x3.n[0] = c.l2.n[0] + ((vs1(6 * QV_P0) - (px.n[0] << 1)) - dx.n[0]);
    x3.n[1] = c.l2.n[1] + ((vs1(6 * QV_P1) - (px.n[1] << 1)) - dx.n[1]);
    for (int k = 2; k < 8; k++) x3.n[k] = c.l2.n[k] + ((vs1(6 * QV_M29) - (px.n[k] << 1)) - dx.n[k]);
    x3.n[8] = c.l2.n[8] + ((vs1(6 * QV_M24) - (px.n[8] << 1)) - dx.n[8]);
    v29_nweak(&x3);                                                      /* NORM */
    v29_subm(&c.t, &px, &x3, 2);                                         /* LAZY */
    if (__builtin_expect(zf, 0)) v29_sel(&x3, zmask4(d4), &px, &x3);
    for (int k = 0; k < 9; k++) px.n[k] = x3.n[k];
}
/* y3 = lam t - py (weakly normalised) -> py, keeping the old y on zero lanes */
static QV_INL void bk_y3(bst &c, vfe &py, const uint32_t *d4, uint8_t zf) {
    vfe y3;
    v29_subm(&y3, &c.y3, &py, 2);
    v29_nweak(&y3);
    if (__builtin_expect(zf, 0)) v29_sel(&y3, zmask4(d4), &py, &y3);
    for (int k = 0; k < 9; k++) py.n[k] = y3.n[k];
}
/* forward part: retain packed yN; dxn = xN - x (1 on zero-digit lanes) */
static QV_INL void fw_next(vfe &dxn, V yn[4], const tentry *Tn, const uint32_t *dn4, uint8_t zfn, const vfe &x, const vfe &one) {
    const tentry *e[4]; entries4(e, Tn, dn4);
    vfe xN; gather_x(&xN, e);
    tr4(yn, e[0], e[1], e[2], e[3], 1);
    v29_subm(&dxn, &xN, &x, 2);
    if (__builtin_expect(zfn, 0)) { const V m = zmask4(dn4); v29_sel(&dxn, m, &one, &dxn); }
}

struct vstate {
    vfe px[QSB_CG_BMAX / 4 + 4], py[QSB_CG_BMAX / 4 + 4], c[QSB_CG_BMAX / 4 + 4];
    /* Each denominator is consumed before its block is prepared for the next pass. */
    vfe den[QSB_CG_BMAX / 4 + 4];
    /* Packed, transposed unsigned yT; consumed before reuse for the next window. */
    V yt[QSB_CG_BMAX / 4 + 4][4];
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
    vfe *px = vs->px, *py = vs->py, *cc = vs->c, *den = vs->den;
    V (*yt)[4] = vs->yt;
    vfe one; v29_one(&one);
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
            vfe xT; gather_x(&xT, e);
            tr4(yt[b], e[0], e[1], e[2], e[3], 1);
            v29_subm(&den[b], &xT, &px[b], 2);
            if (zf[b]) v29_sel(&den[b], zmask4(dg + 4 * b), &one, &den[b]);
            if (b == 0) v29_seed_product(&cc[b], &den[b]);
            else v29_mul_o(&cc[b], &cc[b - 1], &den[b]);
        }
        acc = cc[nb - 1];
    }
    vfe xD, yD; v29_set_w(&xD, S->dx_w); v29_set_w(&yD, S->dy_w);
    int asc = 1;
    for (int s = 1; s < nw; s++) {
        vinv(&inv, &acc);
        vfe ubuf[2]; ubuf[0] = inv; int ui = 0;
        const vfe *accp = &one;
        const uint32_t *dg = w->dig[s];
        const uint8_t *zf = w->zf[s];
        const int last = (s + 1 == nw);
        const uint32_t *dgn = last ? NULL : w->dig[s + 1];
        const uint8_t *zfn = last ? NULL : w->zf[s + 1];
        const tentry *Tn = last ? NULL : T + L.off[s + 1];
        const int step = asc ? -1 : 1;
        int b = asc ? nb - 1 : 0;
        for (int it = 0; it < nb; it++, b += step) {
            const int bp = b + step;
            const int bf = b + step * QSB_CG_PF;
            bst &c = vs->tmp;
            const vfe &dx = den[b];
            bk_pre(c, yt[b], dg + 4 * b, py[b]);
            const vfe *ikp;
            if (it + 1 < nb) { v29_mul_o(&c.ik, &ubuf[ui], &cc[bp]); v29_mul_o(&ubuf[ui ^ 1], &ubuf[ui], &dx); ui ^= 1; ikp = &c.ik; }
            else ikp = &ubuf[ui];
            v29_mul_o(&c.lam, &c.dy, ikp);
            v29_sqr_o(&c.l2, &c.lam);
            bk_x3(c, px[b], dx, dg + 4 * b, zf[b]);
            v29_mul_o(&c.y3, &c.lam, &c.t);
            bk_y3(c, py[b], dg + 4 * b, zf[b]);
            if (!last) {
                if (bf >= 0 && bf < nb) prefetch4(Tn, dgn + 4 * bf);
                fw_next(den[b], yt[b], Tn, dgn + 4 * b, zfn[b], px[b], one);
            } else v29_subm(&den[b], &xD, &px[b], 2);
            if (it == 0) v29_seed_product(&cc[b], &den[b]);
            else v29_mul_o(&cc[b], accp, &den[b]);
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
            const vfe &dx = den[b];
            vfe dy, ik, lam, l2, xm, t, ym;
            const vfe *ikp;
            if (it + 1 < nb) { v29_mul_o(&ik, &ubuf[ui], &cc[bp]); v29_mul_o(&ubuf[ui ^ 1], &ubuf[ui], &dx); ui ^= 1; ikp = &ik; }
            else ikp = &ubuf[ui];
            v29_subm(&dy, &yD, &py[b], 2);
            v29_mul_o(&lam, &dy, ikp);
            v29_sqr_o(&l2, &lam);
            v29_sub2(&xm, &l2, &px[b], &xD);                              /* < 5 * 2^29 */
            v29_nweak(&xm);                                               /* NORM */
            v29_subm(&t, &px[b], &xm, 2);
            v29_mul_o(&ym, &lam, &t);
            /* This block's EC arithmetic is finished; canonicalise x in place. */
            vfe &xp = px[b];
            v29_norm_weak(&xp); v29_norm_weak(&xm);
            V pp, pm; v29_pub_parity(&pp, &pm, &py[b], &ym);
#ifdef QCG_EC_HOOK
            v29_subm(&ym, &ym, &py[b], 2);
            v29_nweak(&ym);
            vfe yp = py[b]; v29_norm(&yp); v29_norm(&ym);
            { V a4[4], b4[4], c4[4], d4[4]; v29_to_w(a4, &xp); v29_to_w(b4, &yp); v29_to_w(c4, &xm); v29_to_w(d4, &ym);
              for (int l = 0; l < 4; l++) { uint64_t X0[4], Y0[4], X1[4], Y1[4];
                  for (int q = 0; q < 4; q++) { X0[q] = a4[q][l]; Y0[q] = b4[q][l]; X1[q] = c4[q][l]; Y1[q] = d4[q][l]; }
                  if (4 * b + l < n) QCG_EC_HOOK(w, 4 * b + l, X0, Y0, X1, Y1); } }
#endif
            const unsigned hm = hash_block(&xp, pp, &xm, pm, use_ni);
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
#undef QV_M29
#undef QV_M24
#undef QV_P0
#undef QV_P1
} /* namespace v4 */
