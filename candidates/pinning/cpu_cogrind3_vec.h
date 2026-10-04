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

/* the 8-lane pubkey SHA of Q+ (lanes 0-3) and Q- (lanes 4-7) of one block; returns an 8-bit
 * mask of lanes whose H0 passes the leading-zero prefilter */
static QV_INL unsigned hash_block(const vfe *xp, const vfe *yp, const vfe *xm, const vfe *ym, int use_ni) {
    using namespace qcg_sha;
    V wp[4], wm[4];
    v29_to_w(wp, xp); v29_to_w(wm, xm);
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
        s8_compress_plan<0x1FFu, 1>(st, W, S8_PLAN_PUBKEY);       /* only H0 */
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
    for (int l = 0; l < 4; l++) _mm_prefetch((const char *)(Tj + (d4[l] & QCG_IDXM)), QCG_PF_HINT);
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
    v29_from_w(&c.xT, wx[0], wx[1], wx[2], wx[3]);
    v29_subm(&c.dx, &c.xT, &px, 2);                                     /* LAZY (< 3 * 2^29) */
    if (__builtin_expect(zf, 0)) v29_sel(&c.dx, zmask4(d4), &one, &c.dx);
    vfe yT, ny; v29_from_w(&yT, wy[0], wy[1], wy[2], wy[3]);
    const V sm = negmask4(d4);
    v29_neg(&ny, &yT, 2);
    for (int k = 0; k < 9; k++) {                                       /* dy = (neg ? -yT : yT) - py, LAZY (< 4 * 2^29) */
        const V y = (V)_mm256_blendv_epi8((__m256i)yT.n[k], (__m256i)ny.n[k], (__m256i)sm);
        c.dy.n[k] = y;
    }
    v29_subm(&c.dy, &c.dy, &py, 2);
}
/* x3 = l2 - px - xT (normalised weakly, written to px unless a zero digit keeps the old point);
 * t = px_old - x3 */
static QV_INL void bk_x3(bst &c, vfe &px, const uint32_t *d4, uint8_t zf) {
    vfe x3;
    v29_sub2(&x3, &c.l2, &px, &c.xT);                                    /* < 5 * 2^29 */
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
/* forward part of the next step: dxn = xN - x (1 on zero-digit lanes) */
static QV_INL void fw_next(bst &c, const tentry *Tn, const uint32_t *dn4, uint8_t zfn, const vfe &x, const vfe &one) {
    const tentry *e[4]; entries4(e, Tn, dn4);
    vfe xN; gather_x(&xN, e);
    v29_subm(&c.dxn, &xN, &x, 2);
    if (__builtin_expect(zfn, 0)) { const V m = zmask4(dn4); v29_sel(&c.dxn, m, &one, &c.dxn); }
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
    vfe one; v29_one(&one);
    const int nw = L.nwin;
    {
        const uint32_t *dg = w->dig[0];
        const tentry *T0 = T + L.off[0];
#if QSB_CG_PF_PROLOGUE
        for (int p = 0; p < nb && p < QSB_CG_PF; p++)
            for (int l = 0; l < 4; l++) _mm_prefetch((const char *)(T0 + dg[4 * p + l]), QCG_PF_HINT);
#endif
        for (int b = 0; b < nb; b++) {
            const tentry *e[4] = {T0 + dg[4 * b], T0 + dg[4 * b + 1], T0 + dg[4 * b + 2], T0 + dg[4 * b + 3]};
            if (b + QSB_CG_PF < nb) for (int l = 0; l < 4; l++) _mm_prefetch((const char *)(T0 + dg[4 * (b + QSB_CG_PF) + l]), QCG_PF_HINT);
            gather_x(&px[b], e); gather_y(&py[b], e);
        }
    }
    vfe acc, inv;
    {
        const uint32_t *dg = w->dig[1];
        const uint8_t *zf = w->zf[1];
        const tentry *Tj = T + L.off[1];
#if QSB_CG_PF_PROLOGUE
        for (int p = 0; p < nb && p < QSB_CG_PF; p++) prefetch4(Tj, dg + 4 * p);
#endif
        for (int b = 0; b < nb; b++) {
            if (b + QSB_CG_PF < nb) prefetch4(Tj, dg + 4 * (b + QSB_CG_PF));
            const tentry *e[4]; entries4(e, Tj, dg + 4 * b);
            vfe xT, dx; gather_x(&xT, e);
            v29_subm(&dx, &xT, &px[b], 2);
            if (zf[b]) v29_sel(&dx, zmask4(dg + 4 * b), &one, &dx);
            v29_mul_o(&cc[b], b ? &cc[b - 1] : &one, &dx);
        }
        acc = cc[nb - 1];
    }
    vfe xD, yD; v29_set_w(&xD, S->dx_w); v29_set_w(&yD, S->dy_w);
    int asc = 1;
    for (int s = 1; s < nw; s++) {
#if QSB_CG_PF_PROLOGUE
        /* Issue the first current/next-window lines before the scalar batch inverse. */
        for (int p = 0; p < nb && p < QSB_CG_PF; p++) {
            const int pb = asc ? nb - 1 - p : p;
            prefetch4(T + L.off[s], w->dig[s] + 4 * pb);
            if (s + 1 < nw) prefetch4(T + L.off[s + 1], w->dig[s + 1] + 4 * pb);
        }
#endif
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
            if (it + 1 < nb) { v29_mul_o(&c.ik, &ubuf[ui], &cc[bp]); v29_mul_o(&ubuf[ui ^ 1], &ubuf[ui], &c.dx); ui ^= 1; ikp = &c.ik; }
            else ikp = &ubuf[ui];
            v29_mul_o(&c.lam, &c.dy, ikp);
            v29_sqr_o(&c.l2, &c.lam);
            bk_x3(c, px[b], dg + 4 * b, zf[b]);
            /* Next X gather is independent of current Y; start it before the Y product. */
            if (!last) fw_next(c, Tn, dgn + 4 * b, zfn[b], px[b], one);
            else v29_subm(&c.dxn, &xD, &px[b], 2);
            v29_mul_o(&c.y3, &c.lam, &c.t);
            bk_y3(c, py[b], dg + 4 * b, zf[b]);
            v29_mul_o(&cc[b], accp, &c.dxn);
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
            v29_subm(&dx, &xD, &px[b], 2);
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
            v29_subm(&ym, &ym, &py[b], 2);
            vfe xp = px[b], yp = py[b];
            v29_norm(&xp); v29_norm(&yp); v29_norm(&xm); v29_norm(&ym);
#ifdef QCG_EC_HOOK
            { V a4[4], b4[4], c4[4], d4[4]; v29_to_w(a4, &xp); v29_to_w(b4, &yp); v29_to_w(c4, &xm); v29_to_w(d4, &ym);
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
#undef QV_M29
#undef QV_M24
#undef QV_P0
#undef QV_P1
} /* namespace v4 */
