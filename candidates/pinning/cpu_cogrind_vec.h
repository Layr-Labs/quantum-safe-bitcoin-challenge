/* cpu_cogrind_vec.h -- SIMD elliptic-curve stage of the host co-grinder (included once per ISA).
 *
 * One candidate per 64-bit lane (QCG_VW lanes: 4 with AVX2, 8 with AVX-512F), field elements in
 * libsecp256k1's 10x26 radix (limbs in the low 32 bits of each lane, so every limb product is one
 * VPMULUDQ). vfe_mul_inner / vfe_sqr_inner below are libsecp256k1's secp256k1_fe_mul_inner /
 * secp256k1_fe_sqr_inner (field_10x26_impl.h, MIT, Pieter Wuille; notice in COPYING-secp256k1)
 * with each 32x32->64 product replaced by a lane-wise VPMULUDQ; the reduction schedule and every
 * bound are unchanged (only the column sums are re-associated). Each instantiation is compiled
 * with its own target attribute and is only called when the CPU reports that ISA at runtime.
 *
 * Expects: QCG_NS (namespace), QCG_VW (4 or 8), QCG_TARGET (target attribute string).
 */
namespace QCG_NS {
#define QCG_AVX2 __attribute__((target(QCG_TARGET), always_inline))
/* GCC temporary-expression substitution can inflate the register lifetimes
 * of the wide field expressions. Keep that pass off in these two host-only
 * functions; inline helpers remain within their callers' ISA target. */
#if defined(__GNUC__) && !defined(__clang__)
#define QCG_AVX2F __attribute__((target(QCG_TARGET), noinline, optimize("no-tree-ter")))
#else
#define QCG_AVX2F __attribute__((target(QCG_TARGET), noinline))
#endif
typedef uint64_t V __attribute__((vector_size(8 * QCG_VW)));
typedef int64_t VI __attribute__((vector_size(8 * QCG_VW)));
static inline QCG_AVX2 V vset1(uint64_t x) { return (V){} + x; }
#if QCG_VW == 4
#define MUL(a, b) ((V)_mm256_mul_epu32((__m256i)(a), (__m256i)(b)))
#else
#define MUL(a, b) ((V)_mm512_mul_epu32((__m512i)(a), (__m512i)(b)))
#endif
static inline QCG_AVX2 V MULK(V c, uint64_t k) { const V kk = vset1(k); return MUL(c, kk) + (MUL(c >> 32, kk) << 32); }
struct vfe { V n[10]; };

static inline QCG_AVX2 void vfe_mul_inner(V *r, const V *a, const V *b) {
    V c, d;
    V u0, u1, u2, u3, u4, u5, u6, u7, u8;
    V t9, t1, t0, t2, t3, t4, t5, t6, t7;
    const V M = vset1(0x3FFFFFFULL), M4 = vset1(0x3FFFFFULL), R0 = vset1(0x3D10ULL);
    d  = ((((MUL(a[0], b[9]) + MUL(a[1], b[8])) + (MUL(a[2], b[7]) + MUL(a[3], b[6]))) + ((MUL(a[4], b[5]) + MUL(a[5], b[4])) + (MUL(a[6], b[3]) + MUL(a[7], b[2])))) + (MUL(a[8], b[1]) + MUL(a[9], b[0])));
    t9 = d & M; d >>= 26;
    c  = MUL(a[0], b[0]);
    d += ((((MUL(a[1], b[9]) + MUL(a[2], b[8])) + (MUL(a[3], b[7]) + MUL(a[4], b[6]))) + ((MUL(a[5], b[5]) + MUL(a[6], b[4])) + (MUL(a[7], b[3]) + MUL(a[8], b[2])))) + MUL(a[9], b[1]));
    u0 = d & M; d >>= 26; c += MUL(u0, R0);
    t0 = c & M; c >>= 26; c += (u0 << 10);
    c += MUL(a[0], b[1])
       + MUL(a[1], b[0]);
    d += (((MUL(a[2], b[9]) + MUL(a[3], b[8])) + (MUL(a[4], b[7]) + MUL(a[5], b[6]))) + ((MUL(a[6], b[5]) + MUL(a[7], b[4])) + (MUL(a[8], b[3]) + MUL(a[9], b[2]))));
    u1 = d & M; d >>= 26; c += MUL(u1, R0);
    t1 = c & M; c >>= 26; c += (u1 << 10);
    c += ((MUL(a[0], b[2]) + MUL(a[1], b[1])) + MUL(a[2], b[0]));
    d += (((MUL(a[3], b[9]) + MUL(a[4], b[8])) + (MUL(a[5], b[7]) + MUL(a[6], b[6]))) + ((MUL(a[7], b[5]) + MUL(a[8], b[4])) + MUL(a[9], b[3])));
    u2 = d & M; d >>= 26; c += MUL(u2, R0);
    t2 = c & M; c >>= 26; c += (u2 << 10);
    c += ((MUL(a[0], b[3]) + MUL(a[1], b[2])) + (MUL(a[2], b[1]) + MUL(a[3], b[0])));
    d += (((MUL(a[4], b[9]) + MUL(a[5], b[8])) + (MUL(a[6], b[7]) + MUL(a[7], b[6]))) + (MUL(a[8], b[5]) + MUL(a[9], b[4])));
    u3 = d & M; d >>= 26; c += MUL(u3, R0);
    t3 = c & M; c >>= 26; c += (u3 << 10);
    c += (((MUL(a[0], b[4]) + MUL(a[1], b[3])) + (MUL(a[2], b[2]) + MUL(a[3], b[1]))) + MUL(a[4], b[0]));
    d += (((MUL(a[5], b[9]) + MUL(a[6], b[8])) + (MUL(a[7], b[7]) + MUL(a[8], b[6]))) + MUL(a[9], b[5]));
    u4 = d & M; d >>= 26; c += MUL(u4, R0);
    t4 = c & M; c >>= 26; c += (u4 << 10);
    c += (((MUL(a[0], b[5]) + MUL(a[1], b[4])) + (MUL(a[2], b[3]) + MUL(a[3], b[2]))) + (MUL(a[4], b[1]) + MUL(a[5], b[0])));
    d += ((MUL(a[6], b[9]) + MUL(a[7], b[8])) + (MUL(a[8], b[7]) + MUL(a[9], b[6])));
    u5 = d & M; d >>= 26; c += MUL(u5, R0);
    t5 = c & M; c >>= 26; c += (u5 << 10);
    c += (((MUL(a[0], b[6]) + MUL(a[1], b[5])) + (MUL(a[2], b[4]) + MUL(a[3], b[3]))) + ((MUL(a[4], b[2]) + MUL(a[5], b[1])) + MUL(a[6], b[0])));
    d += ((MUL(a[7], b[9]) + MUL(a[8], b[8])) + MUL(a[9], b[7]));
    u6 = d & M; d >>= 26; c += MUL(u6, R0);
    t6 = c & M; c >>= 26; c += (u6 << 10);
    c += (((MUL(a[0], b[7]) + MUL(a[1], b[6])) + (MUL(a[2], b[5]) + MUL(a[3], b[4]))) + ((MUL(a[4], b[3]) + MUL(a[5], b[2])) + (MUL(a[6], b[1]) + MUL(a[7], b[0]))));
    d += MUL(a[8], b[9])
       + MUL(a[9], b[8]);
    u7 = d & M; d >>= 26; c += MUL(u7, R0);
    t7 = c & M; c >>= 26; c += (u7 << 10);
    c += ((((MUL(a[0], b[8]) + MUL(a[1], b[7])) + (MUL(a[2], b[6]) + MUL(a[3], b[5]))) + ((MUL(a[4], b[4]) + MUL(a[5], b[3])) + (MUL(a[6], b[2]) + MUL(a[7], b[1])))) + MUL(a[8], b[0]));
    d += MUL(a[9], b[9]);
    u8 = d & M; d >>= 26; c += MUL(u8, R0);
    r[3] = t3;
    r[4] = t4;
    r[5] = t5;
    r[6] = t6;
    r[7] = t7;
    r[8] = c & M; c >>= 26; c += (u8 << 10);
    c   += MUL(d, R0) + t9;
    r[9] = c & M4; c >>= 22; c += (d << 14);
    d    = MULK(c, 0x3D1ULL) + t0;
    r[0] = d & M; d >>= 26;
    d   += (c << 6) + t1;
    r[1] = d & M; d >>= 26;
    d   += t2;
    r[2] = d;
}

static inline QCG_AVX2 void vfe_sqr_inner(V *r, const V *a) {
    V c, d;
    V u0, u1, u2, u3, u4, u5, u6, u7, u8;
    V t9, t0, t1, t2, t3, t4, t5, t6, t7;
    const V M = vset1(0x3FFFFFFULL), M4 = vset1(0x3FFFFFULL), R0 = vset1(0x3D10ULL);
    d  = (((MUL(a[0] << 1, a[9]) + MUL(a[1] << 1, a[8])) + (MUL(a[2] << 1, a[7]) + MUL(a[3] << 1, a[6]))) + MUL(a[4] << 1, a[5]));
    t9 = d & M; d >>= 26;
    c  = MUL(a[0], a[0]);
    d += (((MUL(a[1] << 1, a[9]) + MUL(a[2] << 1, a[8])) + (MUL(a[3] << 1, a[7]) + MUL(a[4] << 1, a[6]))) + MUL(a[5], a[5]));
    u0 = d & M; d >>= 26; c += MUL(u0, R0);
    t0 = c & M; c >>= 26; c += (u0 << 10);
    c += MUL(a[0] << 1, a[1]);
    d += ((MUL(a[2] << 1, a[9]) + MUL(a[3] << 1, a[8])) + (MUL(a[4] << 1, a[7]) + MUL(a[5] << 1, a[6])));
    u1 = d & M; d >>= 26; c += MUL(u1, R0);
    t1 = c & M; c >>= 26; c += (u1 << 10);
    c += MUL(a[0] << 1, a[2])
       + MUL(a[1], a[1]);
    d += ((MUL(a[3] << 1, a[9]) + MUL(a[4] << 1, a[8])) + (MUL(a[5] << 1, a[7]) + MUL(a[6], a[6])));
    u2 = d & M; d >>= 26; c += MUL(u2, R0);
    t2 = c & M; c >>= 26; c += (u2 << 10);
    c += MUL(a[0] << 1, a[3])
       + MUL(a[1] << 1, a[2]);
    d += ((MUL(a[4] << 1, a[9]) + MUL(a[5] << 1, a[8])) + MUL(a[6] << 1, a[7]));
    u3 = d & M; d >>= 26; c += MUL(u3, R0);
    t3 = c & M; c >>= 26; c += (u3 << 10);
    c += ((MUL(a[0] << 1, a[4]) + MUL(a[1] << 1, a[3])) + MUL(a[2], a[2]));
    d += ((MUL(a[5] << 1, a[9]) + MUL(a[6] << 1, a[8])) + MUL(a[7], a[7]));
    u4 = d & M; d >>= 26; c += MUL(u4, R0);
    t4 = c & M; c >>= 26; c += (u4 << 10);
    c += ((MUL(a[0] << 1, a[5]) + MUL(a[1] << 1, a[4])) + MUL(a[2] << 1, a[3]));
    d += MUL(a[6] << 1, a[9])
       + MUL(a[7] << 1, a[8]);
    u5 = d & M; d >>= 26; c += MUL(u5, R0);
    t5 = c & M; c >>= 26; c += (u5 << 10);
    c += ((MUL(a[0] << 1, a[6]) + MUL(a[1] << 1, a[5])) + (MUL(a[2] << 1, a[4]) + MUL(a[3], a[3])));
    d += MUL(a[7] << 1, a[9])
       + MUL(a[8], a[8]);
    u6 = d & M; d >>= 26; c += MUL(u6, R0);
    t6 = c & M; c >>= 26; c += (u6 << 10);
    c += ((MUL(a[0] << 1, a[7]) + MUL(a[1] << 1, a[6])) + (MUL(a[2] << 1, a[5]) + MUL(a[3] << 1, a[4])));
    d += MUL(a[8] << 1, a[9]);
    u7 = d & M; d >>= 26; c += MUL(u7, R0);
    t7 = c & M; c >>= 26; c += (u7 << 10);
    c += (((MUL(a[0] << 1, a[8]) + MUL(a[1] << 1, a[7])) + (MUL(a[2] << 1, a[6]) + MUL(a[3] << 1, a[5]))) + MUL(a[4], a[4]));
    d += MUL(a[9], a[9]);
    u8 = d & M; d >>= 26; c += MUL(u8, R0);
    r[3] = t3;
    r[4] = t4;
    r[5] = t5;
    r[6] = t6;
    r[7] = t7;
    r[8] = c & M; c >>= 26; c += (u8 << 10);
    c   += MUL(d, R0) + t9;
    r[9] = c & M4; c >>= 22; c += (d << 14);
    d    = MULK(c, 0x3D1ULL) + t0;
    r[0] = d & M; d >>= 26;
    d   += (c << 6) + t1;
    r[1] = d & M; d >>= 26;
    d   += t2;
    r[2] = d;
}

/* r may alias a or b: every product is formed before the first store to r. */
static inline QCG_AVX2 void vfe_mul(vfe *r, const vfe *a, const vfe *b) { vfe_mul_inner(r->n, a->n, b->n); }
static inline QCG_AVX2 void vfe_sqr(vfe *r, const vfe *a) { vfe_sqr_inner(r->n, a->n); }
static inline QCG_AVX2 void vfe_add(vfe *r, const vfe *a) { for (int k = 0; k < 10; k++) r->n[k] += a->n[k]; }
static inline QCG_AVX2 void vfe_neg(vfe *r, const vfe *a, int m) {
    const uint64_t f = 2 * (uint64_t)(m + 1);
    r->n[0] = vset1(0x3FFFC2FULL * f) - a->n[0];
    r->n[1] = vset1(0x3FFFFBFULL * f) - a->n[1];
    for (int k = 2; k < 9; k++) r->n[k] = vset1(0x3FFFFFFULL * f) - a->n[k];
    r->n[9] = vset1(0x03FFFFFULL * f) - a->n[9];
}
static inline QCG_AVX2 void vfe_normalize_weak(vfe *r) {
    const V M = vset1(0x3FFFFFFULL);
    V t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4], t5 = r->n[5], t6 = r->n[6], t7 = r->n[7], t8 = r->n[8], t9 = r->n[9];
    V x = t9 >> 22; t9 &= vset1(0x03FFFFFULL);
    t0 += MUL(x, vset1(0x3D1ULL)); t1 += (x << 6);
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
static inline QCG_AVX2 void vfe_normalize(vfe *r) {
    const V M = vset1(0x3FFFFFFULL), one = vset1(1);
    V t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4], t5 = r->n[5], t6 = r->n[6], t7 = r->n[7], t8 = r->n[8], t9 = r->n[9];
    V m;
    V x = t9 >> 22; t9 &= vset1(0x03FFFFFULL);
    t0 += MUL(x, vset1(0x3D1ULL)); t1 += (x << 6);
    t1 += (t0 >> 26); t0 &= M;
    t2 += (t1 >> 26); t1 &= M;
    t3 += (t2 >> 26); t2 &= M; m = t2;
    t4 += (t3 >> 26); t3 &= M; m &= t3;
    t5 += (t4 >> 26); t4 &= M; m &= t4;
    t6 += (t5 >> 26); t5 &= M; m &= t5;
    t7 += (t6 >> 26); t6 &= M; m &= t6;
    t8 += (t7 >> 26); t7 &= M; m &= t7;
    t9 += (t8 >> 26); t8 &= M; m &= t8;
    x = (t9 >> 22) | ((V)(t9 == vset1(0x03FFFFFULL)) & (V)(m == M)
        & (V)((t1 + vset1(0x40ULL) + ((t0 + vset1(0x3D1ULL)) >> 26)) > M) & one);
    t0 += MUL(x, vset1(0x3D1ULL)); t1 += (x << 6);
    t1 += (t0 >> 26); t0 &= M;
    t2 += (t1 >> 26); t1 &= M;
    t3 += (t2 >> 26); t2 &= M;
    t4 += (t3 >> 26); t3 &= M;
    t5 += (t4 >> 26); t4 &= M;
    t6 += (t5 >> 26); t5 &= M;
    t7 += (t6 >> 26); t6 &= M;
    t8 += (t7 >> 26); t7 &= M;
    t9 += (t8 >> 26); t8 &= M;
    t9 &= vset1(0x03FFFFFULL);
    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4; r->n[5] = t5; r->n[6] = t6; r->n[7] = t7; r->n[8] = t8; r->n[9] = t9;
}
/* r = mask ? a : b (mask lanes all-ones or zero) */
static inline QCG_AVX2 void vfe_sel(vfe *r, V mask, const vfe *a, const vfe *b) {
    for (int k = 0; k < 10; k++) r->n[k] = (a->n[k] & mask) | (b->n[k] & ~mask);
}
static inline QCG_AVX2 void vfe_set1(vfe *r, const uint64_t t[10]) { for (int k = 0; k < 10; k++) r->n[k] = vset1(t[k]); }
static inline QCG_AVX2 void vfe_one(vfe *r) { r->n[0] = vset1(1); for (int k = 1; k < 10; k++) r->n[k] = vset1(0); }
/* r lane l = a lane (l ^ k) */
static inline QCG_AVX2 void vfe_perm_xor(vfe *r, const vfe *a, int k) {
    VI m; for (int l = 0; l < QCG_VW; l++) m[l] = l ^ k;
    for (int j = 0; j < 10; j++) r->n[j] = __builtin_shuffle(a->n[j], m);
}
/* 4x64 little-endian words (one lane each) -> 10x26 */
static inline QCG_AVX2 void vfe_from_w(vfe *r, V w0, V w1, V w2, V w3) {
    const V M = vset1(0x3FFFFFFULL);
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
/* normalized 10x26 -> 4x64 words */
static inline QCG_AVX2 void vfe_to_w(V w[4], const vfe *a) {
    const V *n = a->n;
    w[0] = n[0] | (n[1] << 26) | (n[2] << 52);
    w[1] = (n[2] >> 12) | (n[3] << 14) | (n[4] << 40);
    w[2] = (n[4] >> 24) | (n[5] << 2) | (n[6] << 28) | (n[7] << 54);
    w[3] = (n[7] >> 10) | (n[8] << 16) | (n[9] << 42);
}
/* 4x4 transpose of the x (half 0) or y (half 1) words of 4 table entries */
static inline QCG_AVX2 void tr4(__m256i w[4], const tentry *const *e, int half) {
    __m256i r0 = _mm256_load_si256((const __m256i *)((const uint8_t *)e[0] + 32 * half));
    __m256i r1 = _mm256_load_si256((const __m256i *)((const uint8_t *)e[1] + 32 * half));
    __m256i r2 = _mm256_load_si256((const __m256i *)((const uint8_t *)e[2] + 32 * half));
    __m256i r3 = _mm256_load_si256((const __m256i *)((const uint8_t *)e[3] + 32 * half));
    __m256i t0 = _mm256_unpacklo_epi64(r0, r1), t1 = _mm256_unpackhi_epi64(r0, r1);
    __m256i t2 = _mm256_unpacklo_epi64(r2, r3), t3 = _mm256_unpackhi_epi64(r2, r3);
    w[0] = _mm256_permute2x128_si256(t0, t2, 0x20); w[2] = _mm256_permute2x128_si256(t0, t2, 0x31);
    w[1] = _mm256_permute2x128_si256(t1, t3, 0x20); w[3] = _mm256_permute2x128_si256(t1, t3, 0x31);
}
/* gather QCG_VW table entries (64 B each: x words, y words) into x and y */
static inline QCG_AVX2 void vfe_gather(vfe *x, vfe *y, const tentry *const *e, V neg) {
    for (int half = 0; half < 2; half++) {
        V w[4];
#if QCG_VW == 4
        __m256i a[4]; tr4(a, e, half);
        for (int k = 0; k < 4; k++) w[k] = (V)a[k];
#else
        __m256i a[4], b[4]; tr4(a, e, half); tr4(b, e + 4, half);
        for (int k = 0; k < 4; k++) w[k] = (V)_mm512_inserti64x4(_mm512_castsi256_si512(a[k]), b[k], 1);
#endif
        if (half) for (int k = 0; k < 4; k++) w[k] ^= neg;
        vfe_from_w(half ? y : x, w[0], w[1], w[2], w[3]);
        if (half) {
            // 2p-c in radix 26. Bounds: table ordinate <3p, dy <8p.
            y->n[0] += vset1(2*0x3FFFC2FULL - (0x800001e8ULL & 0x3FFFFFFULL));
            y->n[1] += vset1(2*0x3FFFFBFULL - (0x800001e8ULL >> 26));
            for (int k = 2; k < 9; k++) y->n[k] += vset1(2*0x3FFFFFFULL);
            y->n[9] += vset1(2*0x03FFFFFULL);
        }
    }
}
static inline QCG_AVX2 uint64_t lane(V v, int l) { return v[l]; }

/* per-worker vector state */
struct vstate {
    vfe x[QSB_CG_MAXB / QCG_VW], y[QSB_CG_MAXB / QCG_VW], c[QSB_CG_MAXB / QCG_VW], dx[QSB_CG_MAXB / QCG_VW], ex[QSB_CG_MAXB / QCG_VW], ey[QSB_CG_MAXB / QCG_VW];
    V act[QSB_CG_MAXB / QCG_VW], ld[QSB_CG_MAXB / QCG_VW];
    uint8_t anyld[QSB_CG_MAXB / QCG_VW];
};

/* inv[g] = 1 / acc[g] (lane-wise) for the two block chains: one scalar Fermat inversion. */
static QCG_AVX2F void vchain_invert(vfe inv[2], const vfe acc[2]) {
    vfe m, q, T, oth, im;
    vfe_mul(&m, &acc[0], &acc[1]);
    /* butterfly: after the step with stride k, q lane l = product of m over the aligned group of
     * 2k lanes containing l; oth accumulates the partner groups, i.e. every m except m_l */
    q = m;
    for (int k = 1; k < QCG_VW; k <<= 1) {
        vfe s; vfe_perm_xor(&s, &q, k);
        if (k == 1) oth = s; else vfe_mul(&oth, &oth, &s);
        vfe_mul(&q, &q, &s);
    }
    T = q;                              /* product of all 2*QCG_VW chain values, in every lane */
    fe t5; for (int k = 0; k < 5; k++) t5.n[k] = lane(T.n[2 * k], 0) + (lane(T.n[2 * k + 1], 0) << 26);
    fe_normalize(&t5);
    fe it; fe_inv(&it, &t5);
    uint64_t t10[10]; for (int k = 0; k < 5; k++) { t10[2 * k] = it.n[k] & 0x3FFFFFFULL; t10[2 * k + 1] = it.n[k] >> 26; }
    vfe iT; vfe_set1(&iT, t10);
    vfe_mul(&im, &iT, &oth);
    vfe_mul(&inv[0], &im, &acc[1]);
    vfe_mul(&inv[1], &im, &acc[0]);
}

static QCG_AVX2F void ec_batch_vec(worker_t *w, vstate *vs) {
    shared_t *S = g_cg;
    const int n = w->n;
    const int nb = (n + QCG_VW - 1) / QCG_VW;
    recode(w);
    const tentry *T = S->table;
    vfe acc[2], inv[2], one; vfe_one(&one);
    /* window 0: load */
    for (int b = 0; b < nb; b++) {
        const tentry *e[QCG_VW];
        for (int l = 0; l < QCG_VW; l++) {
            const int i = QCG_VW * b + l;
            unsigned d = i < n ? digit_abs(w->sd[0][i]) : 0;
            e[l] = &T[d]; w->inf[i] = !d;
            if (i < n) __builtin_prefetch(&T[(size_t)QSB_CG_TSIZE + digit_abs(w->sd[1][i])]);
        }
        V neg;
        for (int l = 0; l < QCG_VW; l++) { int i = QCG_VW*b+l; neg[l] = i < n && w->sd[0][i] < 0 ? ~0ull : 0; }
        vfe_gather(&vs->x[b], &vs->y[b], e, neg);
    }
    for (int j = 1; j < QSB_CG_NWIN; j++) {
        const tentry *Tj = T + (size_t)j * QSB_CG_TSIZE;
        vfe_one(&acc[0]); vfe_one(&acc[1]);
        for (int b = 0; b < nb; b++) {
            const tentry *e[QCG_VW]; V am, lm; int anyl = 0;
            for (int l = 0; l < QCG_VW; l++) {
                const int i = QCG_VW * b + l;
                unsigned d = 0;
                if (i < n) {
                    d = digit_abs(w->sd[j][i]);
                    if (j + 1 < QSB_CG_NWIN) __builtin_prefetch(&T[(size_t)(j + 1) * QSB_CG_TSIZE + digit_abs(w->sd[j + 1][i])]);
                }
                e[l] = &Tj[d];
                const int live = i < n && d;
                am[l] = (live && !w->inf[i]) ? ~0ull : 0;
                lm[l] = (live && w->inf[i]) ? ~0ull : 0;
                anyl |= live && w->inf[i];
                if (live) w->inf[i] = 0;
            }
            vs->act[b] = am; vs->ld[b] = lm; vs->anyld[b] = (uint8_t)anyl;
            V neg;
            for (int l = 0; l < QCG_VW; l++) { int i = QCG_VW*b+l; neg[l] = i < n && w->sd[j][i] < 0 ? ~0ull : 0; }
            vfe_gather(&vs->ex[b], &vs->ey[b], e, neg);
            vfe t; vfe_neg(&t, &vs->x[b], 1);
            vfe d3 = vs->ex[b]; vfe_add(&d3, &t);                      /* mag 3 */
            vfe_sel(&vs->dx[b], vs->act[b], &d3, &one);
            vfe_mul(&acc[b & 1], &acc[b & 1], &vs->dx[b]);
            vs->c[b] = acc[b & 1];
        }
        vchain_invert(inv, acc);
        for (int b = nb - 1; b >= 0; b--) {
            const int g = b & 1;
            vfe ik, t, dy, l, l2, x3, y3, t2;
            if (b >= 2) vfe_mul(&ik, &inv[g], &vs->c[b - 2]); else ik = inv[g];
            vfe_mul(&inv[g], &inv[g], &vs->dx[b]);
            vfe_neg(&t, &vs->y[b], 1); dy = vs->ey[b]; vfe_add(&dy, &t);
            vfe_mul(&l, &dy, &ik);
            vfe_sqr(&l2, &l);
            vfe_neg(&t, &vs->x[b], 1); x3 = l2; vfe_add(&x3, &t);
            vfe_neg(&t, &vs->ex[b], 1); vfe_add(&x3, &t); vfe_normalize_weak(&x3);
            vfe_neg(&t, &x3, 1); t2 = vs->x[b]; vfe_add(&t2, &t);
            vfe_mul(&y3, &l, &t2);
            vfe_neg(&t, &vs->y[b], 1); vfe_add(&y3, &t); vfe_normalize_weak(&y3);
            const V a = vs->act[b], ldv = vs->ld[b];
            if (!vs->anyld[b]) {
                vfe_sel(&vs->x[b], a, &x3, &vs->x[b]);
                vfe_sel(&vs->y[b], a, &y3, &vs->y[b]);
            } else {
                vfe xs, ys;
                vfe_sel(&xs, ldv, &vs->ex[b], &vs->x[b]); vfe_sel(&ys, ldv, &vs->ey[b], &vs->y[b]);
                vfe_sel(&vs->x[b], a, &x3, &xs); vfe_sel(&vs->y[b], a, &y3, &ys);
            }
        }
    }
    /* final: Q0 = P + A, Q1 = P - A */
    vfe ax, ay, nax, nay;
    { uint64_t t10[10]; fe a5 = S->ax; fe_normalize(&a5);
      for (int k = 0; k < 5; k++) { t10[2 * k] = a5.n[k] & 0x3FFFFFFULL; t10[2 * k + 1] = a5.n[k] >> 26; }
      vfe_set1(&ax, t10);
      a5 = S->ay; fe_normalize(&a5);
      for (int k = 0; k < 5; k++) { t10[2 * k] = a5.n[k] & 0x3FFFFFFULL; t10[2 * k + 1] = a5.n[k] >> 26; }
      vfe_set1(&ay, t10); }
    vfe_neg(&nax, &ax, 1); vfe_neg(&nay, &ay, 1);
    vfe_one(&acc[0]); vfe_one(&acc[1]);
    for (int b = 0; b < nb; b++) {
        V am;
        for (int l = 0; l < QCG_VW; l++) { const int i = QCG_VW * b + l; am[l] = (i < n && !w->inf[i]) ? ~0ull : 0; }
        vs->act[b] = am;
        vfe t, d3; vfe_neg(&t, &vs->x[b], 1); d3 = ax; vfe_add(&d3, &t);
        vfe_sel(&vs->dx[b], vs->act[b], &d3, &one);
        vfe_mul(&acc[b & 1], &acc[b & 1], &vs->dx[b]);
        vs->c[b] = acc[b & 1];
    }
    vchain_invert(inv, acc);
    for (int b = nb - 1; b >= 0; b--) {
        const int g = b & 1;
        vfe ik, ny, sx, t;
        if (b >= 2) vfe_mul(&ik, &inv[g], &vs->c[b - 2]); else ik = inv[g];
        vfe_mul(&inv[g], &inv[g], &vs->dx[b]);
        vfe_neg(&ny, &vs->y[b], 0);
        vfe_neg(&sx, &vs->x[b], 1); vfe_add(&sx, &nax);                 /* -(px + ax), mag 4 */
        for (int recid = 0; recid < 2; recid++) {
            vfe dy, l, l2, qx, qy, t2;
            dy = recid ? nay : ay; vfe_add(&dy, &ny);
            vfe_mul(&l, &dy, &ik);
            vfe_sqr(&l2, &l);
            qx = l2; vfe_add(&qx, &sx); vfe_normalize(&qx);
            vfe_neg(&t, &qx, 1); t2 = vs->x[b]; vfe_add(&t2, &t);
            vfe_mul(&qy, &l, &t2); vfe_add(&qy, &ny); vfe_normalize(&qy);
            V xw[4]; vfe_to_w(xw, &qx);
            /* EC already produced four/eight independent pubkeys. Interleave
             * their final SHA-NI compression instead of serial OpenSSL calls.
             * Inactive lanes are hashed but never nominated. */
            alignas(16) uint8_t blk[QCG_VW][64];
            uint32_t h[QCG_VW][8];
            const uint8_t *bp[QCG_VW];
            for (int ln = 0; ln < QCG_VW; ln++) {
                blk[ln][0] = (uint8_t)(0x02 | (lane(qy.n[0], ln) & 1));
                for (int k = 0; k < 4; k++) { uint64_t v = lane(xw[3 - k], ln); for (int bb = 0; bb < 8; bb++) blk[ln][1 + 8 * k + bb] = (uint8_t)(v >> (56 - 8 * bb)); }
                blk[ln][33] = 0x80; memset(blk[ln] + 34, 0, 30); blk[ln][62] = 0x01; blk[ln][63] = 0x08;
                memcpy(h[ln], SHA_IV, 32); bp[ln] = blk[ln];
            }
            if (S->shani) {
                for (int ln = 0; ln < QCG_VW; ln += 4) qsha_x4(h + ln, bp + ln);
            } else {
                for (int ln = 0; ln < QCG_VW; ln++) sha_blocks(h[ln], blk[ln], 1);
            }
            for (int ln = 0; ln < QCG_VW; ln++) {
                const int i = QCG_VW * b + ln;
                if (i < n && !w->inf[i] && lz_ok(h[ln])) publish(w, i, recid);
            }
        }
    }
}
#undef QCG_AVX2
#undef QCG_AVX2F
#undef MUL
} /* namespace QCG_NS */
