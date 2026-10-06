/* cpu_cogrind3_ifma.h -- AVX-512 IFMA elliptic-curve stage of the V3 co-grinder. Included once by
 * cpu_cogrind3.h inside namespace qcg, after cpu_cogrind3_vec.h (the AVX2 stage, namespace v4).
 *
 * Same batch as v4::ec_batch (one product chain over 4-candidate blocks, backward(s) and
 * forward(s+1) fused, Q- = Q+ + (-2A), 8-lane pubkey hash, exact gate), with the field in radix
 * 2^52: 5 limbs per 64-bit lane, 4 lanes per 256-bit vector (AVX-512VL), limb products by
 * VPMADD52LUQ / VPMADD52HUQ. A 4x4-limb product is 25 + 25 fused multiply-adds instead of the
 * 10x26 field's 100 VPMULUDQ plus its adds. 256-bit vectors keep the block structure of v4 and
 * avoid the 512-bit frequency licence on Intel parts; Zen 4 runs them at full rate.
 *
 * Invariants (all arithmetic is exact; every bound below is checked by cg3_field_test.cpp):
 *   W form: limbs n0..n3 < 2^52, n4 < 2^49. IFMA reads only bits 51:0 of each operand, so every
 *           multiplication input is in W form. fmul/fsqr return W form with n4 <= 2^48.
 *   fwk(r) maps any limbs < 2^62 to W form (n4 < 2^48 + 2^10), value unchanged mod p.
 *   fsub(r, a, b) = a + 2p - b needs b in W form (2p's limbs dominate b's); limbs of r < 2^55.
 *   fnorm(r) maps limbs < 2^62 to the canonical value in [0, p) (libsecp256k1 fe_normalize).
 * Every function carries target(avx512f,avx512vl,avx512ifma) and is only called after
 * __builtin_cpu_supports reports all three (which includes the OS's AVX-512 state check).
 */
namespace v4i {
#define QI_TGT "avx2,avx512f,avx512vl,avx512ifma"
#define QI_INL __attribute__((target(QI_TGT), always_inline)) inline
#define QI_FN  __attribute__((target(QI_TGT), noinline))
typedef uint64_t V __attribute__((vector_size(32)));
struct vfe { V n[5]; };

static QI_INL V vs1(uint64_t x) { return (V){x, x, x, x}; }
#define QI_LO(acc, a, b) ((V)_mm256_madd52lo_epu64((__m256i)(acc), (__m256i)(a), (__m256i)(b)))
#define QI_HI(acc, a, b) ((V)_mm256_madd52hi_epu64((__m256i)(acc), (__m256i)(a), (__m256i)(b)))
#define QI_M52 0xFFFFFFFFFFFFFULL
#define QI_M48 0xFFFFFFFFFFFFULL
#define QI_C   0x1000003D1ULL                 /* 2^256 mod p */
#define QI_R   0x1000003D10ULL                /* 2^260 mod p */
/* 2p in radix 2^52 */
#define QI_2P0 (2 * 0xFFFFEFFFFFC2FULL)
#define QI_2P1 (2 * 0xFFFFFFFFFFFFFULL)
#define QI_2P4 (2 * 0x0FFFFFFFFFFFFULL)

/* r = a b mod p (W form in, W form out, n4 <= 2^48); r may alias a or b */
static QI_INL void fmul(vfe *r, const vfe *A, const vfe *B) {
    const V a0 = A->n[0], a1 = A->n[1], a2 = A->n[2], a3 = A->n[3], a4 = A->n[4];
    const V b0 = B->n[0], b1 = B->n[1], b2 = B->n[2], b3 = B->n[3], b4 = B->n[4];
    const V z = vs1(0), M = vs1(QI_M52), R = vs1(QI_R), C = vs1(QI_C);
    /* columns: c_k = sum lo(a_i b_j) over i+j = k, plus sum hi(a_i b_j) over i+j = k-1; each < 10 * 2^52 */
    V c0 = QI_LO(z, a0, b0);
    V c1 = QI_LO(QI_LO(QI_HI(z, a0, b0), a0, b1), a1, b0);
    V c2 = QI_LO(QI_LO(QI_LO(QI_HI(QI_HI(z, a0, b1), a1, b0), a0, b2), a1, b1), a2, b0);
    V c3 = QI_LO(QI_LO(QI_LO(QI_LO(QI_HI(QI_HI(QI_HI(z, a0, b2), a1, b1), a2, b0), a0, b3), a1, b2), a2, b1), a3, b0);
    V c4 = QI_HI(QI_HI(QI_HI(QI_HI(z, a0, b3), a1, b2), a2, b1), a3, b0);
    c4 = QI_LO(QI_LO(QI_LO(QI_LO(QI_LO(c4, a0, b4), a1, b3), a2, b2), a3, b1), a4, b0);
    V c5 = QI_HI(QI_HI(QI_HI(QI_HI(QI_HI(z, a0, b4), a1, b3), a2, b2), a3, b1), a4, b0);
    c5 = QI_LO(QI_LO(QI_LO(QI_LO(c5, a1, b4), a2, b3), a3, b2), a4, b1);
    V c6 = QI_HI(QI_HI(QI_HI(QI_HI(z, a1, b4), a2, b3), a3, b2), a4, b1);
    c6 = QI_LO(QI_LO(QI_LO(c6, a2, b4), a3, b3), a4, b2);
    V c7 = QI_HI(QI_HI(QI_HI(z, a2, b4), a3, b3), a4, b2);
    c7 = QI_LO(QI_LO(c7, a3, b4), a4, b3);
    V c8 = QI_LO(QI_HI(QI_HI(z, a3, b4), a4, b3), a4, b4);
    V c9 = QI_HI(z, a4, b4);                                   /* < 2^46 (a4, b4 < 2^49) */
    /* high columns to 52-bit limbs (c9 stays < 2^52) */
    c6 += c5 >> 52; c5 &= M;
    c7 += c6 >> 52; c6 &= M;
    c8 += c7 >> 52; c7 &= M;
    c9 += c8 >> 52; c8 &= M;
    /* fold 2^(52k) = 2^(52(k-5)) 2^260, 2^260 = R mod p: lo(c_k R) -> column k-5, hi -> k-4 */
    V d0 = QI_LO(c0, c5, R);
    V d1 = QI_LO(QI_HI(c1, c5, R), c6, R);
    V d2 = QI_LO(QI_HI(c2, c6, R), c7, R);
    V d3 = QI_LO(QI_HI(c3, c7, R), c8, R);
    V d4 = QI_LO(QI_HI(c4, c8, R), c9, R);
    const V e5 = QI_HI(z, c9, R);                              /* weight 2^260, < 2^32 */
    d4 += d3 >> 52; d3 &= M;
    const V top = (d4 >> 48) + (e5 << 4);                      /* weight 2^256, < 2^37 */
    d4 &= vs1(QI_M48);
    d0 = QI_LO(d0, top, C);
    d1 = QI_HI(d1, top, C);
    d1 += d0 >> 52; d0 &= M;
    d2 += d1 >> 52; d1 &= M;
    d3 += d2 >> 52; d2 &= M;
    d4 += d3 >> 52; d3 &= M;
    r->n[0] = d0; r->n[1] = d1; r->n[2] = d2; r->n[3] = d3; r->n[4] = d4;
}
static QI_INL void fsqr(vfe *r, const vfe *a) { fmul(r, a, a); }

/* any limbs < 2^62 -> W form (libsecp256k1 fe_normalize_weak, radix 2^52) */
static QI_INL void fwk(vfe *r) {
    const V M = vs1(QI_M52);
    V t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4];
    const V x = t4 >> 48; t4 &= vs1(QI_M48);
    t0 = QI_LO(t0, x, vs1(QI_C));                               /* x < 2^14: x C < 2^47 */
    t1 += t0 >> 52; t0 &= M;
    t2 += t1 >> 52; t1 &= M;
    t3 += t2 >> 52; t2 &= M;
    t4 += t3 >> 52; t3 &= M;
    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4;
}
/* canonical value in [0, p) from limbs < 2^62 (libsecp256k1 fe_normalize, radix 2^52) */
static QI_INL void fnorm(vfe *r) {
    const V M = vs1(QI_M52);
    V t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4], m;
    V x = t4 >> 48; t4 &= vs1(QI_M48);
    t0 = QI_LO(t0, x, vs1(QI_C));
    t1 += t0 >> 52; t0 &= M;
    t2 += t1 >> 52; t1 &= M; m = t1;
    t3 += t2 >> 52; t2 &= M; m &= t2;
    t4 += t3 >> 52; t3 &= M; m &= t3;
    /* value < 2p here; subtract p once if value >= p (limbs < 2^53: signed compares are exact) */
    const V ge = (V)_mm256_cmpgt_epi64((__m256i)t0, (__m256i)vs1(0xFFFFEFFFFFC2FULL - 1));
    x = (t4 >> 48) | ((V)_mm256_cmpeq_epi64((__m256i)t4, (__m256i)vs1(QI_M48)) & (V)_mm256_cmpeq_epi64((__m256i)m, (__m256i)M) & ge & vs1(1));
    t0 += (vs1(0) - x) & vs1(QI_C);                             /* x in {0, 1} */
    t1 += t0 >> 52; t0 &= M;
    t2 += t1 >> 52; t1 &= M;
    t3 += t2 >> 52; t2 &= M;
    t4 += t3 >> 52; t3 &= M;
    t4 &= vs1(QI_M48);
    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4;
}
/* Selected EC outputs are already weak-normalized: n0..n3 < 2^52 and
 * n4 < 2^48 + 2^10.  Thus n4>>48 is only zero or one and the remaining
 * 256-bit value needs at most the single secp256k1 p-boundary correction.
 * This skips fnorm's general first fold/carry pass without weakening the
 * canonical result. */
static QI_INL void fnorm_weak(vfe *r) {
    const V M = vs1(QI_M52);
    V t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4];
    V x = t4 >> 48; t4 &= vs1(QI_M48);                         /* x in {0, 1} */
    const V ge = (V)_mm256_cmpgt_epi64((__m256i)t0, (__m256i)vs1(0xFFFFEFFFFFC2FULL - 1));
    x |= (V)_mm256_cmpeq_epi64((__m256i)t4, (__m256i)vs1(QI_M48))
       & (V)_mm256_cmpeq_epi64((__m256i)t3, (__m256i)M)
       & (V)_mm256_cmpeq_epi64((__m256i)t2, (__m256i)M)
       & (V)_mm256_cmpeq_epi64((__m256i)t1, (__m256i)M)
       & ge & vs1(1);
    t0 += (vs1(0) - x) & vs1(QI_C);
    t1 += t0 >> 52; t0 &= M;
    t2 += t1 >> 52; t1 &= M;
    t3 += t2 >> 52; t2 &= M;
    t4 += t3 >> 52; t3 &= M;
    t4 &= vs1(QI_M48);
    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4;
}
static QI_INL V fparity_weak(const vfe *r) {
    const V M = vs1(QI_M52);
    const V t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3];
    V t4 = r->n[4], x = t4 >> 48; t4 &= vs1(QI_M48);
    const V ge = (V)_mm256_cmpgt_epi64((__m256i)t0, (__m256i)vs1(0xFFFFEFFFFFC2FULL - 1));
    x |= (V)_mm256_cmpeq_epi64((__m256i)t4, (__m256i)vs1(QI_M48))
       & (V)_mm256_cmpeq_epi64((__m256i)t3, (__m256i)M)
       & (V)_mm256_cmpeq_epi64((__m256i)t2, (__m256i)M)
       & (V)_mm256_cmpeq_epi64((__m256i)t1, (__m256i)M)
       & ge & vs1(1);
    return (t0 ^ x) & vs1(1);
}

/* QCG_Y_PARITY_PASS (kill switch, default 1): hash_block reads only bit 0 of a y's limb 0 (the
 * 02/03 prefix byte). fparity returns that bit of fnorm's canonical result without the second
 * carry pass: fnorm's final t0 is (t0 + x*C) mod 2^52 with x in {0, 1} its >= p flag and
 * C = 2^256 - p odd, so bit 0 is (t0 ^ x) & 1, and the later carries and masks never touch
 * bit 0 of limb 0. Same inputs (limbs < 2^62), same parity for every value. */
#ifndef QCG_Y_PARITY_PASS
#define QCG_Y_PARITY_PASS 1
#endif
static QI_INL V fparity(const vfe *r) {
    const V M = vs1(QI_M52);
    V t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4], m;
    V x = t4 >> 48; t4 &= vs1(QI_M48);
    t0 = QI_LO(t0, x, vs1(QI_C));
    t1 += t0 >> 52; t0 &= M;
    t2 += t1 >> 52; t1 &= M; m = t1;
    t3 += t2 >> 52; t2 &= M; m &= t2;
    t4 += t3 >> 52; t3 &= M; m &= t3;
    const V ge = (V)_mm256_cmpgt_epi64((__m256i)t0, (__m256i)vs1(0xFFFFEFFFFFC2FULL - 1));
    x = (t4 >> 48) | ((V)_mm256_cmpeq_epi64((__m256i)t4, (__m256i)vs1(QI_M48)) & (V)_mm256_cmpeq_epi64((__m256i)m, (__m256i)M) & ge & vs1(1));
    return (t0 ^ x) & vs1(1);
}
/* r = a + 2p - b (b in W form); limbs of r < a's + 2^54 */
static QI_INL void fsub(vfe *r, const vfe *a, const vfe *b) {
    r->n[0] = a->n[0] + (vs1(QI_2P0) - b->n[0]);
    r->n[1] = a->n[1] + (vs1(QI_2P1) - b->n[1]);
    r->n[2] = a->n[2] + (vs1(QI_2P1) - b->n[2]);
    r->n[3] = a->n[3] + (vs1(QI_2P1) - b->n[3]);
    r->n[4] = a->n[4] + (vs1(QI_2P4) - b->n[4]);
}
/* r = 2p - a (a in W form) */
static QI_INL void fneg(vfe *r, const vfe *a) {
    r->n[0] = vs1(QI_2P0) - a->n[0];
    r->n[1] = vs1(QI_2P1) - a->n[1];
    r->n[2] = vs1(QI_2P1) - a->n[2];
    r->n[3] = vs1(QI_2P1) - a->n[3];
    r->n[4] = vs1(QI_2P4) - a->n[4];
}
/* r = mask ? a : b (per lane, mask all-ones or zero) */
static QI_INL void fsel(vfe *r, V mask, const vfe *a, const vfe *b) {
    for (int k = 0; k < 5; k++) r->n[k] = (V)_mm256_blendv_epi8((__m256i)b->n[k], (__m256i)a->n[k], (__m256i)mask);
}
static QI_INL void fone(vfe *r) { r->n[0] = vs1(1); for (int k = 1; k < 5; k++) r->n[k] = vs1(0); }
/* 4 little-endian 64-bit words per lane (< 2^256) -> radix 2^52 (limbs < 2^52, n4 < 2^48) */
static QI_INL void from_w(vfe *r, V w0, V w1, V w2, V w3) {
    const V M = vs1(QI_M52);
    r->n[0] = w0 & M;
    r->n[1] = ((w0 >> 52) | (w1 << 12)) & M;
    r->n[2] = ((w1 >> 40) | (w2 << 24)) & M;
    r->n[3] = ((w2 >> 28) | (w3 << 36)) & M;
    r->n[4] = w3 >> 16;
}
/* canonical radix 2^52 -> 4 words per lane */
static QI_INL void to_w(V w[4], const vfe *a) {
    const V *n = a->n;
    w[0] = n[0] | (n[1] << 52);
    w[1] = (n[1] >> 12) | (n[2] << 40);
    w[2] = (n[2] >> 24) | (n[3] << 28);
    w[3] = (n[3] >> 36) | (n[4] << 16);
}
static QI_INL void set_w(vfe *r, const uint64_t *w) { from_w(r, vs1(w[0]), vs1(w[1]), vs1(w[2]), vs1(w[3])); }
/* r lane l = a lane (l ^ k), k = 1 or 2 */
static QI_INL void fpermx(vfe *r, const vfe *a, int k) {
    for (int j = 0; j < 5; j++)
        r->n[j] = (V)(k == 1 ? _mm256_permute4x64_epi64((__m256i)a->n[j], 0xB1) : _mm256_permute4x64_epi64((__m256i)a->n[j], 0x4E));
}
/* gather x (half 0) or y (half 1) words of 4 table entries, transposed to one V per word */
static QI_INL void tr4(V w[4], const tentry *e0, const tentry *e1, const tentry *e2, const tentry *e3, int half) {
    __m256i r0 = _mm256_load_si256((const __m256i *)((const uint8_t *)e0 + 32 * half));
    __m256i r1 = _mm256_load_si256((const __m256i *)((const uint8_t *)e1 + 32 * half));
    __m256i r2 = _mm256_load_si256((const __m256i *)((const uint8_t *)e2 + 32 * half));
    __m256i r3 = _mm256_load_si256((const __m256i *)((const uint8_t *)e3 + 32 * half));
    __m256i t0 = _mm256_unpacklo_epi64(r0, r1), t1 = _mm256_unpackhi_epi64(r0, r1);
    __m256i t2 = _mm256_unpacklo_epi64(r2, r3), t3 = _mm256_unpackhi_epi64(r2, r3);
    w[0] = (V)_mm256_permute2x128_si256(t0, t2, 0x20); w[2] = (V)_mm256_permute2x128_si256(t0, t2, 0x31);
    w[1] = (V)_mm256_permute2x128_si256(t1, t3, 0x20); w[3] = (V)_mm256_permute2x128_si256(t1, t3, 0x31);
}
static QI_INL void gather_x(vfe *x, const tentry *const *e) { V w[4]; tr4(w, e[0], e[1], e[2], e[3], 0); from_w(x, w[0], w[1], w[2], w[3]); }
static QI_INL void gather_y(vfe *y, const tentry *const *e) { V w[4]; tr4(w, e[0], e[1], e[2], e[3], 1); from_w(y, w[0], w[1], w[2], w[3]); }

/* inv lane l = 1 / m lane l (m in W form) with one scalar inversion */
static QI_INL void vinv(vfe *inv, const vfe *m) {
    vfe s, q1, q2, oth, t;
    fpermx(&s, m, 1); oth = s; fmul(&q1, m, &s);
    fpermx(&s, &q1, 2); fmul(&t, &oth, &s); fmul(&q2, &q1, &s);
    oth = t;
    fnorm(&q2);
    V w[4]; to_w(w, &q2);
    uint64_t tw[4] = {w[0][0], w[1][0], w[2][0], w[3][0]}, iw[4];
    scalar_inv(iw, tw);
    set_w(&t, iw);
    fmul(inv, &t, &oth);
}

/* 8-lane pubkey SHA of Q+ (lanes 0-3) and Q- (lanes 4-7); canonical inputs; returns the
 * 8-bit mask of lanes whose H0 passes the leading-zero prefilter (v4::hash_block's layout) */
static QI_INL unsigned hash_block(const vfe *xp, const vfe *yp, const vfe *xm, const vfe *ym, int use_ni) {
    using namespace qcg_sha;
    V wp[4], wm[4];
    to_w(wp, xp); to_w(wm, xm);
    const __m256i idx_lo = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6), idx_hi = _mm256_setr_epi32(1, 3, 5, 7, 1, 3, 5, 7);
    v8u X[8];
    for (int k = 0; k < 4; k++) {
        __m256i pl = _mm256_permutevar8x32_epi32((__m256i)wp[k], idx_lo), ml = _mm256_permutevar8x32_epi32((__m256i)wm[k], idx_lo);
        __m256i ph = _mm256_permutevar8x32_epi32((__m256i)wp[k], idx_hi), mh = _mm256_permutevar8x32_epi32((__m256i)wm[k], idx_hi);
        X[2 * k] = _mm256_blend_epi32(pl, ml, 0xF0);
        X[2 * k + 1] = _mm256_blend_epi32(ph, mh, 0xF0);
    }
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

static QI_INL V zmask4(const uint32_t *d4) {
    V m; for (int l = 0; l < 4; l++) m[l] = QCG_ZERO(d4[l]) ? ~0ULL : 0; return m;
}
static QI_INL V negmask4(const uint32_t *d4) {
    const __m128i c = _mm_loadu_si128((const __m128i *)d4);
    return (V)_mm256_cvtepi32_epi64(_mm_srai_epi32(c, 31));
}
static QI_INL void entries4(const tentry **e, const tentry *Tj, const uint32_t *d4) {
    e[0] = Tj + (d4[0] & QCG_IDXM); e[1] = Tj + (d4[1] & QCG_IDXM); e[2] = Tj + (d4[2] & QCG_IDXM); e[3] = Tj + (d4[3] & QCG_IDXM);
}
static QI_INL void prefetch4(const tentry *Tj, const uint32_t *d4) {
    for (int l = 0; l < 4; l++) _mm_prefetch((const char *)(Tj + (d4[l] & QCG_IDXM)), _MM_HINT_T0);
}

struct bst { vfe xT, dx, dy, ik, lam, l2, t, y3, dxn; };
/* gather T_s; dx = xT - px (1 on zero-digit lanes); dy = +-yT - py; both W form */
static QI_INL void bk_pre(bst &c, const tentry *Tj, const uint32_t *d4, uint8_t zf, const vfe &px, const vfe &py, const vfe &one) {
    const tentry *e[4]; entries4(e, Tj, d4);
    V wx[4], wy[4];
    tr4(wx, e[0], e[1], e[2], e[3], 0); tr4(wy, e[0], e[1], e[2], e[3], 1);
    from_w(&c.xT, wx[0], wx[1], wx[2], wx[3]);
    fsub(&c.dx, &c.xT, &px); fwk(&c.dx);
    if (__builtin_expect(zf, 0)) fsel(&c.dx, zmask4(d4), &one, &c.dx);
    vfe yT, ny; from_w(&yT, wy[0], wy[1], wy[2], wy[3]);
    const V sm = negmask4(d4);
    fneg(&ny, &yT);
    fsel(&c.dy, sm, &ny, &yT);
    fsub(&c.dy, &c.dy, &py); fwk(&c.dy);
}
/* x3 = l2 - px - xT -> px (old point kept on zero-digit lanes); t = px_old - x3 */
static QI_INL void bk_x3(bst &c, vfe &px, const uint32_t *d4, uint8_t zf) {
    vfe x3;
    fsub(&x3, &c.l2, &px); fsub(&x3, &x3, &c.xT); fwk(&x3);
    fsub(&c.t, &px, &x3); fwk(&c.t);
    if (__builtin_expect(zf, 0)) fsel(&x3, zmask4(d4), &px, &x3);
    px = x3;
}
/* y3 = lam t - py -> py, keeping the old y on zero-digit lanes */
static QI_INL void bk_y3(bst &c, vfe &py, const uint32_t *d4, uint8_t zf) {
    vfe y3;
    fsub(&y3, &c.y3, &py); fwk(&y3);
    if (__builtin_expect(zf, 0)) fsel(&y3, zmask4(d4), &py, &y3);
    py = y3;
}
/* forward part of the next step: dxn = xN - x (1 on zero-digit lanes) */
static QI_INL void fw_next(bst &c, const tentry *Tn, const uint32_t *dn4, uint8_t zfn, const vfe &x, const vfe &one) {
    const tentry *e[4]; entries4(e, Tn, dn4);
    vfe xN; gather_x(&xN, e);
    fsub(&c.dxn, &xN, &x); fwk(&c.dxn);
    if (__builtin_expect(zfn, 0)) { const V m = zmask4(dn4); fsel(&c.dxn, m, &one, &c.dxn); }
}

struct vstate {
    vfe px[QSB_CG_BMAX / 4 + 4], py[QSB_CG_BMAX / 4 + 4], c[QSB_CG_BMAX / 4 + 4];
    bst tmp;
};

/* One batch: Q+ = z B + A, Q- = Q+ - 2A for the w->n candidates, hash, publish. */
static QI_FN void ec_batch(worker_t *w, vstate *vs) {
    shared_t *S = g_cg;
    const int use_ni = S->sha_mode.load(std::memory_order_relaxed) == 2;
    const layout_t &L = S->lay;
    const int n = w->n;
    const int nb = (n + 3) >> 2;
    const tentry *T = S->table;
    vfe *px = vs->px, *py = vs->py, *cc = vs->c;
    vfe one; fone(&one);
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
            fsub(&dx, &xT, &px[b]); fwk(&dx);
            if (zf[b]) fsel(&dx, zmask4(dg + 4 * b), &one, &dx);
            fmul(&cc[b], b ? &cc[b - 1] : &one, &dx);
        }
        acc = cc[nb - 1];
    }
    vfe xD, yD; set_w(&xD, S->dx_w); set_w(&yD, S->dy_w);
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
            if (it + 1 < nb) { fmul(&c.ik, &ubuf[ui], &cc[bp]); fmul(&ubuf[ui ^ 1], &ubuf[ui], &c.dx); ui ^= 1; ikp = &c.ik; }
            else ikp = &ubuf[ui];
            fmul(&c.lam, &c.dy, ikp);
            fsqr(&c.l2, &c.lam);
            bk_x3(c, px[b], dg + 4 * b, zf[b]);
            fmul(&c.y3, &c.lam, &c.t);
            bk_y3(c, py[b], dg + 4 * b, zf[b]);
            if (!last) fw_next(c, Tn, dgn + 4 * b, zfn[b], px[b], one);
            else { fsub(&c.dxn, &xD, &px[b]); fwk(&c.dxn); }
            fmul(&cc[b], accp, &c.dxn);
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
            fsub(&dx, &xD, &px[b]); fwk(&dx);
            const vfe *ikp;
            if (it + 1 < nb) { fmul(&ik, &ubuf[ui], &cc[bp]); fmul(&ubuf[ui ^ 1], &ubuf[ui], &dx); ui ^= 1; ikp = &ik; }
            else ikp = &ubuf[ui];
            fsub(&dy, &yD, &py[b]); fwk(&dy);
            fmul(&lam, &dy, ikp);
            fsqr(&l2, &lam);
            fsub(&xm, &l2, &px[b]); fsub(&xm, &xm, &xD); fwk(&xm);
            fsub(&t, &px[b], &xm); fwk(&t);
            fmul(&ym, &lam, &t);
            fsub(&ym, &ym, &py[b]);
            vfe xp = px[b], yp = py[b];
#if QCG_Y_PARITY_PASS && !defined(QCG_EC_HOOK)
            fnorm_weak(&xp); fnorm_weak(&xm);
            yp.n[0] = fparity_weak(&py[b]); ym.n[0] = fparity(&ym);
#else
            fnorm(&xp); fnorm(&yp); fnorm(&xm); fnorm(&ym);
#endif
#ifdef QCG_EC_HOOK
            { V a4[4], b4[4], c4[4], d4[4]; to_w(a4, &xp); to_w(b4, &yp); to_w(c4, &xm); to_w(d4, &ym);
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
#undef QI_LO
#undef QI_HI
#undef QI_INL
#undef QI_FN
#undef QI_TGT
} /* namespace v4i */
