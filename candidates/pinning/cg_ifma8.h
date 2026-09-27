/* cg_ifma8.h -- 8-lane AVX-512 IFMA elliptic-curve stage for the pinning co-grinder.
 *
 * Secp256k1 field arithmetic in 5x52-bit radix representation using native AVX-512 IFMA
 * (_mm512_madd52lo_epu64 / _mm512_madd52hi_epu64). Processes 8 candidate product chains in
 * parallel across the 512-bit vector registers.
 *
 * Included by cpu_cogrind.h inside namespace qcg.
 */
#ifndef QSB_CG_IFMA8_H
#define QSB_CG_IFMA8_H

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

namespace v8 {

#define Q8_INL __attribute__((target("avx512f,avx512ifma"), always_inline)) inline
#define Q8_FN  __attribute__((target("avx512f,avx512ifma"), noinline))

struct fe8 { __m512i l[5]; };
#define F8_M52 _mm512_set1_epi64(0xFFFFFFFFFFFFFULL)

static Q8_INL void fe8_set1(fe8 &r) {
    r.l[0] = _mm512_set1_epi64(1);
    for (int i = 1; i < 5; i++) r.l[i] = _mm512_setzero_si512();
}

static Q8_INL void fe8_zero(fe8 &r) {
    for (int i = 0; i < 5; i++) r.l[i] = _mm512_setzero_si512();
}

static Q8_INL void fe8_bcast_64(fe8 &r, const uint64_t *a) {
    const uint64_t M = 0xFFFFFFFFFFFFFULL;
    r.l[0] = _mm512_set1_epi64((long long)(a[0] & M));
    r.l[1] = _mm512_set1_epi64((long long)(((a[0] >> 52) | (a[1] << 12)) & M));
    r.l[2] = _mm512_set1_epi64((long long)(((a[1] >> 40) | (a[2] << 24)) & M));
    r.l[3] = _mm512_set1_epi64((long long)(((a[2] >> 28) | (a[3] << 36)) & M));
    r.l[4] = _mm512_set1_epi64((long long)(a[3] >> 16));
}

static Q8_INL void fe8_from64(fe8 &r, __m512i a0, __m512i a1, __m512i a2, __m512i a3) {
    const __m512i M = F8_M52;
    r.l[0] = _mm512_and_si512(a0, M);
    r.l[1] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(a0, 52), _mm512_slli_epi64(a1, 12)), M);
    r.l[2] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(a1, 40), _mm512_slli_epi64(a2, 24)), M);
    r.l[3] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(a2, 28), _mm512_slli_epi64(a3, 36)), M);
    r.l[4] = _mm512_srli_epi64(a3, 16);
}

static Q8_INL __mmask8 get_neg_mask(__m256i d) {
    return (__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(d));
}

static Q8_INL __mmask8 get_zero_mask(__m256i d) {
    return (__mmask8)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32(d, 1)));
}

static Q8_INL void pt8_load(fe8 &x, fe8 &y, const tentry *const *rows) {
    const __m512i r0 = _mm512_loadu_si512(rows[0]), r1 = _mm512_loadu_si512(rows[1]),
                  r2 = _mm512_loadu_si512(rows[2]), r3 = _mm512_loadu_si512(rows[3]),
                  r4 = _mm512_loadu_si512(rows[4]), r5 = _mm512_loadu_si512(rows[5]),
                  r6 = _mm512_loadu_si512(rows[6]), r7 = _mm512_loadu_si512(rows[7]);
    const __m512i a0 = _mm512_unpacklo_epi64(r0, r1), a1 = _mm512_unpackhi_epi64(r0, r1),
                  a2 = _mm512_unpacklo_epi64(r2, r3), a3 = _mm512_unpackhi_epi64(r2, r3),
                  a4 = _mm512_unpacklo_epi64(r4, r5), a5 = _mm512_unpackhi_epi64(r4, r5),
                  a6 = _mm512_unpacklo_epi64(r6, r7), a7 = _mm512_unpackhi_epi64(r6, r7);
    const __m512i iL = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0), iH = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
    const __m512i b0 = _mm512_permutex2var_epi64(a0, iL, a2), b2 = _mm512_permutex2var_epi64(a0, iH, a2),
                  b1 = _mm512_permutex2var_epi64(a1, iL, a3), b3 = _mm512_permutex2var_epi64(a1, iH, a3),
                  b4 = _mm512_permutex2var_epi64(a4, iL, a6), b6 = _mm512_permutex2var_epi64(a4, iH, a6),
                  b5 = _mm512_permutex2var_epi64(a5, iL, a7), b7 = _mm512_permutex2var_epi64(a5, iH, a7);
    const __m512i c0 = _mm512_shuffle_i64x2(b0, b4, 0x44), c4 = _mm512_shuffle_i64x2(b0, b4, 0xEE),
                  c1 = _mm512_shuffle_i64x2(b1, b5, 0x44), c5 = _mm512_shuffle_i64x2(b1, b5, 0xEE),
                  c2 = _mm512_shuffle_i64x2(b2, b6, 0x44), c6 = _mm512_shuffle_i64x2(b2, b6, 0xEE),
                  c3 = _mm512_shuffle_i64x2(b3, b7, 0x44), c7 = _mm512_shuffle_i64x2(b3, b7, 0xEE);
    fe8_from64(x, c0, c1, c2, c3);
    fe8_from64(y, c4, c5, c6, c7);
}

static Q8_INL void fe8_carry(fe8 &r) {
    const __m512i M = F8_M52, M48 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL), K = _mm512_set1_epi64(0x1000003D1ULL);
    __m512i c;
    c = _mm512_srli_epi64(r.l[4], 48); r.l[4] = _mm512_and_si512(r.l[4], M48);
    r.l[0] = _mm512_madd52lo_epu64(r.l[0], c, K);
    c = _mm512_srli_epi64(r.l[0], 52); r.l[0] = _mm512_and_si512(r.l[0], M); r.l[1] = _mm512_add_epi64(r.l[1], c);
    c = _mm512_srli_epi64(r.l[1], 52); r.l[1] = _mm512_and_si512(r.l[1], M); r.l[2] = _mm512_add_epi64(r.l[2], c);
    c = _mm512_srli_epi64(r.l[2], 52); r.l[2] = _mm512_and_si512(r.l[2], M); r.l[3] = _mm512_add_epi64(r.l[3], c);
    c = _mm512_srli_epi64(r.l[3], 52); r.l[3] = _mm512_and_si512(r.l[3], M); r.l[4] = _mm512_add_epi64(r.l[4], c);
}

static Q8_INL void fe8_red(fe8 &r, __m512i c0, __m512i c1, __m512i c2, __m512i c3, __m512i c4,
                           __m512i c5, __m512i c6, __m512i c7, __m512i c8, __m512i c9) {
    const __m512i Z = _mm512_setzero_si512();
#define LO(acc, x, y) acc = _mm512_madd52lo_epu64(acc, x, y)
#define HI(acc, x, y) acc = _mm512_madd52hi_epu64(acc, x, y)
    const __m512i M = F8_M52;
    __m512i t;
    t = _mm512_srli_epi64(c4, 52); c4 = _mm512_and_si512(c4, M); c5 = _mm512_add_epi64(c5, t);
    t = _mm512_srli_epi64(c5, 52); c5 = _mm512_and_si512(c5, M); c6 = _mm512_add_epi64(c6, t);
    t = _mm512_srli_epi64(c6, 52); c6 = _mm512_and_si512(c6, M); c7 = _mm512_add_epi64(c7, t);
    t = _mm512_srli_epi64(c7, 52); c7 = _mm512_and_si512(c7, M); c8 = _mm512_add_epi64(c8, t);
    t = _mm512_srli_epi64(c8, 52); c8 = _mm512_and_si512(c8, M); c9 = _mm512_add_epi64(c9, t);
    __m512i c10 = _mm512_srli_epi64(c9, 52); c9 = _mm512_and_si512(c9, M);
    const __m512i R = _mm512_set1_epi64(0x1000003D10ULL);
    LO(c0,c5,R); HI(c1,c5,R);
    LO(c1,c6,R); HI(c2,c6,R);
    LO(c2,c7,R); HI(c3,c7,R);
    LO(c3,c8,R); HI(c4,c8,R);
    LO(c4,c9,R); __m512i c5b = Z; HI(c5b,c9,R);
    c5b = _mm512_madd52lo_epu64(c5b, c10, R);
    fe8 o; o.l[0] = c0; o.l[1] = c1; o.l[2] = c2; o.l[3] = c3; o.l[4] = c4;
    t = _mm512_srli_epi64(o.l[0], 52); o.l[0] = _mm512_and_si512(o.l[0], M); o.l[1] = _mm512_add_epi64(o.l[1], t);
    t = _mm512_srli_epi64(o.l[1], 52); o.l[1] = _mm512_and_si512(o.l[1], M); o.l[2] = _mm512_add_epi64(o.l[2], t);
    t = _mm512_srli_epi64(o.l[2], 52); o.l[2] = _mm512_and_si512(o.l[2], M); o.l[3] = _mm512_add_epi64(o.l[3], t);
    t = _mm512_srli_epi64(o.l[3], 52); o.l[3] = _mm512_and_si512(o.l[3], M); o.l[4] = _mm512_add_epi64(o.l[4], t);
    t = _mm512_srli_epi64(o.l[4], 52); o.l[4] = _mm512_and_si512(o.l[4], M); c5b = _mm512_add_epi64(c5b, t);
    o.l[0] = _mm512_madd52lo_epu64(o.l[0], c5b, R);
    o.l[1] = _mm512_madd52hi_epu64(o.l[1], c5b, R);
    fe8_carry(o);
    r = o;
}

static Q8_INL void fe8_mul(fe8 &r, const fe8 &a, const fe8 &b) {
    const __m512i Z = _mm512_setzero_si512();
    __m512i c0 = Z, c1 = Z, c2 = Z, c3 = Z, c4 = Z, c5 = Z, c6 = Z, c7 = Z, c8 = Z, c9 = Z;
    const __m512i a0 = a.l[0], a1 = a.l[1], a2 = a.l[2], a3 = a.l[3], a4 = a.l[4];
    const __m512i b0 = b.l[0], b1 = b.l[1], b2 = b.l[2], b3 = b.l[3], b4 = b.l[4];
    LO(c0,a0,b0); HI(c1,a0,b0);
    LO(c1,a0,b1); LO(c1,a1,b0); HI(c2,a0,b1); HI(c2,a1,b0);
    LO(c2,a0,b2); LO(c2,a1,b1); LO(c2,a2,b0); HI(c3,a0,b2); HI(c3,a1,b1); HI(c3,a2,b0);
    LO(c3,a0,b3); LO(c3,a1,b2); LO(c3,a2,b1); LO(c3,a3,b0); HI(c4,a0,b3); HI(c4,a1,b2); HI(c4,a2,b1); HI(c4,a3,b0);
    LO(c4,a0,b4); LO(c4,a1,b3); LO(c4,a2,b2); LO(c4,a3,b1); LO(c4,a4,b0);
    HI(c5,a0,b4); HI(c5,a1,b3); HI(c5,a2,b2); HI(c5,a3,b1); HI(c5,a4,b0);
    LO(c5,a1,b4); LO(c5,a2,b3); LO(c5,a3,b2); LO(c5,a4,b1); HI(c6,a1,b4); HI(c6,a2,b3); HI(c6,a3,b2); HI(c6,a4,b1);
    LO(c6,a2,b4); LO(c6,a3,b3); LO(c6,a4,b2); HI(c7,a2,b4); HI(c7,a3,b3); HI(c7,a4,b2);
    LO(c7,a3,b4); LO(c7,a4,b3); HI(c8,a3,b4); HI(c8,a4,b3);
    LO(c8,a4,b4); HI(c9,a4,b4);
    fe8_red(r, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);
}

static Q8_INL void fe8_sqr(fe8 &r, const fe8 &a) {
    const __m512i Z = _mm512_setzero_si512();
    __m512i x1 = Z, x2 = Z, x3 = Z, x4 = Z, x5 = Z, x6 = Z, x7 = Z, x8 = Z;
    const __m512i a0 = a.l[0], a1 = a.l[1], a2 = a.l[2], a3 = a.l[3], a4 = a.l[4];
    LO(x1,a0,a1); HI(x2,a0,a1);
    LO(x2,a0,a2); HI(x3,a0,a2);
    LO(x3,a0,a3); LO(x3,a1,a2); HI(x4,a0,a3); HI(x4,a1,a2);
    LO(x4,a0,a4); LO(x4,a1,a3); HI(x5,a0,a4); HI(x5,a1,a3);
    LO(x5,a1,a4); LO(x5,a2,a3); HI(x6,a1,a4); HI(x6,a2,a3);
    LO(x6,a2,a4); HI(x7,a2,a4);
    LO(x7,a3,a4); HI(x8,a3,a4);
    __m512i c0 = Z, c1 = _mm512_slli_epi64(x1, 1), c2 = _mm512_slli_epi64(x2, 1), c3 = _mm512_slli_epi64(x3, 1),
            c4 = _mm512_slli_epi64(x4, 1), c5 = _mm512_slli_epi64(x5, 1), c6 = _mm512_slli_epi64(x6, 1),
            c7 = _mm512_slli_epi64(x7, 1), c8 = _mm512_slli_epi64(x8, 1), c9 = Z;
    LO(c0,a0,a0); HI(c1,a0,a0);
    LO(c2,a1,a1); HI(c3,a1,a1);
    LO(c4,a2,a2); HI(c5,a2,a2);
    LO(c6,a3,a3); HI(c7,a3,a3);
    LO(c8,a4,a4); HI(c9,a4,a4);
    fe8_red(r, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);
}
#undef LO
#undef HI

static Q8_INL void fe8_add(fe8 &r, const fe8 &a, const fe8 &b) {
    for (int i = 0; i < 5; i++) r.l[i] = _mm512_add_epi64(a.l[i], b.l[i]);
    fe8_carry(r);
}

static Q8_INL void fe8_sub(fe8 &r, const fe8 &a, const fe8 &b) {
    const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
                  P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4);
    r.l[0] = _mm512_sub_epi64(_mm512_add_epi64(a.l[0], P0), b.l[0]);
    r.l[1] = _mm512_sub_epi64(_mm512_add_epi64(a.l[1], P1), b.l[1]);
    r.l[2] = _mm512_sub_epi64(_mm512_add_epi64(a.l[2], P1), b.l[2]);
    r.l[3] = _mm512_sub_epi64(_mm512_add_epi64(a.l[3], P1), b.l[3]);
    r.l[4] = _mm512_sub_epi64(_mm512_add_epi64(a.l[4], P4), b.l[4]);
    fe8_carry(r);
}

static Q8_INL void fe8_sub2(fe8 &r, const fe8 &a, const fe8 &b, const fe8 &c) {
    const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 8), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 8),
                  P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 8);
    r.l[0] = _mm512_sub_epi64(_mm512_add_epi64(a.l[0], P0), _mm512_add_epi64(b.l[0], c.l[0]));
    r.l[1] = _mm512_sub_epi64(_mm512_add_epi64(a.l[1], P1), _mm512_add_epi64(b.l[1], c.l[1]));
    r.l[2] = _mm512_sub_epi64(_mm512_add_epi64(a.l[2], P1), _mm512_add_epi64(b.l[2], c.l[2]));
    r.l[3] = _mm512_sub_epi64(_mm512_add_epi64(a.l[3], P1), _mm512_add_epi64(b.l[3], c.l[3]));
    r.l[4] = _mm512_sub_epi64(_mm512_add_epi64(a.l[4], P4), _mm512_add_epi64(b.l[4], c.l[4]));
    fe8_carry(r);
}

static Q8_INL void fe8_neg(fe8 &r, const fe8 &a) {
    const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
                  P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4);
    r.l[0] = _mm512_sub_epi64(P0, a.l[0]);
    r.l[1] = _mm512_sub_epi64(P1, a.l[1]);
    r.l[2] = _mm512_sub_epi64(P1, a.l[2]);
    r.l[3] = _mm512_sub_epi64(P1, a.l[3]);
    r.l[4] = _mm512_sub_epi64(P4, a.l[4]);
    fe8_carry(r);
}

static Q8_INL void fe8_inv1(fe8 &x) {
    fe8 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;
#define SQN(dst, src, n) do { dst = src; for (int i = 0; i < (n); i++) fe8_sqr(dst, dst); } while (0)
    SQN(x2, x, 1); fe8_mul(x2, x2, x);
    SQN(x3, x2, 1); fe8_mul(x3, x3, x);
    SQN(x6, x3, 3); fe8_mul(x6, x6, x3);
    SQN(x9, x6, 3); fe8_mul(x9, x9, x3);
    SQN(x11, x9, 2); fe8_mul(x11, x11, x2);
    SQN(x22, x11, 11); fe8_mul(x22, x22, x11);
    SQN(x44, x22, 22); fe8_mul(x44, x44, x22);
    SQN(x88, x44, 44); fe8_mul(x88, x88, x44);
    SQN(x176, x88, 88); fe8_mul(x176, x176, x88);
    SQN(x220, x176, 44); fe8_mul(x220, x220, x44);
    SQN(x223, x220, 3); fe8_mul(x223, x223, x3);
    SQN(t, x223, 23); fe8_mul(t, t, x22);
    SQN(t, t, 5); fe8_mul(t, t, x);
    SQN(t, t, 3); fe8_mul(t, t, x2);
    SQN(t, t, 2); fe8_mul(x, t, x);
#undef SQN
}

static Q8_INL void fe8_canon_scatter_arr(uint64_t out[8][4], uint8_t par[8], const fe8 &a) {
    const __m512i M = F8_M52, M48 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL), K = _mm512_set1_epi64(0x1000003D1ULL);
    fe8 r = a;
    fe8_carry(r); fe8_carry(r);
    __m512i t0 = _mm512_add_epi64(r.l[0], K), c;
    c = _mm512_srli_epi64(t0, 52); t0 = _mm512_and_si512(t0, M); __m512i t1 = _mm512_add_epi64(r.l[1], c);
    c = _mm512_srli_epi64(t1, 52); t1 = _mm512_and_si512(t1, M); __m512i t2 = _mm512_add_epi64(r.l[2], c);
    c = _mm512_srli_epi64(t2, 52); t2 = _mm512_and_si512(t2, M); __m512i t3 = _mm512_add_epi64(r.l[3], c);
    c = _mm512_srli_epi64(t3, 52); t3 = _mm512_and_si512(t3, M); __m512i t4 = _mm512_add_epi64(r.l[4], c);
    const __mmask8 ge = _mm512_test_epi64_mask(t4, _mm512_set1_epi64(1LL << 48));
    t4 = _mm512_and_si512(t4, M48);
    const __m512i l0 = _mm512_mask_blend_epi64(ge, r.l[0], t0), l1 = _mm512_mask_blend_epi64(ge, r.l[1], t1),
                  l2 = _mm512_mask_blend_epi64(ge, r.l[2], t2), l3 = _mm512_mask_blend_epi64(ge, r.l[3], t3),
                  l4 = _mm512_mask_blend_epi64(ge, r.l[4], t4);
    alignas(64) uint64_t w[4][8];
    _mm512_store_si512(w[0], _mm512_or_si512(l0, _mm512_slli_epi64(l1, 52)));
    _mm512_store_si512(w[1], _mm512_or_si512(_mm512_srli_epi64(l1, 12), _mm512_slli_epi64(l2, 40)));
    _mm512_store_si512(w[2], _mm512_or_si512(_mm512_srli_epi64(l2, 24), _mm512_slli_epi64(l3, 28)));
    _mm512_store_si512(w[3], _mm512_or_si512(_mm512_srli_epi64(l3, 36), _mm512_slli_epi64(l4, 16)));
    for (int j = 0; j < 8; j++) {
        if (out) {
            out[j][0] = w[0][j]; out[j][1] = w[1][j]; out[j][2] = w[2][j]; out[j][3] = w[3][j];
        }
        if (par) par[j] = (uint8_t)(w[0][j] & 1);
    }
}

struct vstate {
    fe8 px[QSB_CG_BMAX / 8 + 4], py[QSB_CG_BMAX / 8 + 4], c[QSB_CG_BMAX / 8 + 4];
};

static Q8_FN void ec_batch(worker_t *w, vstate *vs) {
    shared_t *S = g_cg;
    const layout_t &L = S->lay;
    const int n = w->n;
    const int nb = (n + 7) >> 3;
    const tentry *T = S->table;
    fe8 *px = vs->px, *py = vs->py, *cc = vs->c;
    fe8 one; fe8_set1(one);
    const int nw = L.nwin;

    /* Window 0 (unsigned, A folded in: T0[d0]) */
    {
        const uint32_t *dg = w->dig[0];
        const tentry *T0 = T + L.off[0];
        for (int b = 0; b < nb; b++) {
            if (b + QSB_CG_PF < nb) {
                for (int l = 0; l < 8; l++)
                    _mm_prefetch((const char *)(T0 + dg[8 * (b + QSB_CG_PF) + l]), _MM_HINT_T0);
            }
            const tentry *e[8];
            for (int l = 0; l < 8; l++) e[l] = T0 + dg[8 * b + l];
            pt8_load(px[b], py[b], e);
        }
    }

    fe8 acc;
    /* Window 1: forward pass, ascending */
    {
        const uint32_t *dg = w->dig[1];
        const tentry *Tj = T + L.off[1];
        fe8_set1(acc);
        for (int b = 0; b < nb; b++) {
            if (b + QSB_CG_PF < nb) {
                for (int l = 0; l < 8; l++)
                    _mm_prefetch((const char *)(Tj + (dg[8 * (b + QSB_CG_PF) + l] & QCG_IDXM)), _MM_HINT_T0);
            }
            const tentry *e[8];
            for (int l = 0; l < 8; l++) e[l] = Tj + (dg[8 * b + l] & QCG_IDXM);
            fe8 xT, yT, dx;
            pt8_load(xT, yT, e);
            fe8_sub(dx, xT, px[b]);
            const __m256i d_vec = _mm256_loadu_si256((const __m256i *)(dg + 8 * b));
            const __mmask8 zm = get_zero_mask(d_vec);
            if (__builtin_expect(zm != 0, 0)) {
                for (int k = 0; k < 5; k++)
                    dx.l[k] = _mm512_mask_blend_epi64(zm, dx.l[k], one.l[k]);
            }
            fe8_mul(acc, acc, dx);
            cc[b] = acc;
        }
    }

    fe8 xD, yD;
    fe8_bcast_64(xD, S->dx_w);
    fe8_bcast_64(yD, S->dy_w);
    int asc = 1;

    for (int s = 1; s < nw; s++) {
        fe8_inv1(acc);
        fe8 u = acc;
        const uint32_t *dg = w->dig[s];
        const tentry *Tj = T + L.off[s];
        const int last = (s + 1 == nw);
        const uint32_t *dgn = last ? NULL : w->dig[s + 1];
        const tentry *Tn = last ? NULL : T + L.off[s + 1];
        const int step = asc ? -1 : 1;
        int b = asc ? nb - 1 : 0;
        for (int it = 0; it < nb; it++, b += step) {
            const int bp = b + step;
            const int bf = b + step * QSB_CG_PF;
            if (bf >= 0 && bf < nb) {
                for (int l = 0; l < 8; l++) {
                    _mm_prefetch((const char *)(Tj + (dg[8 * bf + l] & QCG_IDXM)), _MM_HINT_T0);
                    if (!last) _mm_prefetch((const char *)(Tn + (dgn[8 * bf + l] & QCG_IDXM)), _MM_HINT_T0);
                }
            }
            const tentry *e[8];
            for (int l = 0; l < 8; l++) e[l] = Tj + (dg[8 * b + l] & QCG_IDXM);
            fe8 xT, yT;
            pt8_load(xT, yT, e);
            const __m256i d_vec = _mm256_loadu_si256((const __m256i *)(dg + 8 * b));
            const __mmask8 zm = get_zero_mask(d_vec);
            const __mmask8 nm = get_neg_mask(d_vec);

            fe8 dx;
            fe8_sub(dx, xT, px[b]);
            if (__builtin_expect(zm != 0, 0)) {
                for (int k = 0; k < 5; k++) dx.l[k] = _mm512_mask_blend_epi64(zm, dx.l[k], one.l[k]);
            }

            fe8 ik;
            if (it + 1 < nb) {
                fe8_mul(ik, u, cc[bp]);
                fe8_mul(u, u, dx);
            } else {
                ik = u;
            }

            fe8 nyT;
            fe8_neg(nyT, yT);
            for (int k = 0; k < 5; k++) yT.l[k] = _mm512_mask_blend_epi64(nm, yT.l[k], nyT.l[k]);

            fe8 dy;
            fe8_sub(dy, yT, py[b]);

            fe8 lam, l2, x3, t, y3;
            fe8_mul(lam, dy, ik);
            fe8_sqr(l2, lam);
            fe8_sub2(x3, l2, px[b], xT);
            fe8_sub(t, px[b], x3);
            fe8_mul(y3, lam, t);
            fe8_sub(y3, y3, py[b]);

            if (__builtin_expect(zm != 0, 0)) {
                for (int k = 0; k < 5; k++) {
                    px[b].l[k] = _mm512_mask_blend_epi64(zm, x3.l[k], px[b].l[k]);
                    py[b].l[k] = _mm512_mask_blend_epi64(zm, y3.l[k], py[b].l[k]);
                }
            } else {
                px[b] = x3;
                py[b] = y3;
            }

            fe8 dxn;
            if (!last) {
                const tentry *en[8];
                for (int l = 0; l < 8; l++) en[l] = Tn + (dgn[8 * b + l] & QCG_IDXM);
                fe8 xN, yN;
                pt8_load(xN, yN, en);
                fe8_sub(dxn, xN, px[b]);
                const __m256i dn_vec = _mm256_loadu_si256((const __m256i *)(dgn + 8 * b));
                const __mmask8 zmn = get_zero_mask(dn_vec);
                if (__builtin_expect(zmn != 0, 0)) {
                    for (int k = 0; k < 5; k++) dxn.l[k] = _mm512_mask_blend_epi64(zmn, dxn.l[k], one.l[k]);
                }
            } else {
                fe8_sub(dxn, xD, px[b]);
            }

            if (it == 0) fe8_mul(cc[b], one, dxn);
            else fe8_mul(cc[b], cc[b - step], dxn);
        }
        acc = cc[b - step];
        asc = !asc;
    }

    /* Final step: Q- = Q+ + D */
    fe8_inv1(acc);
    fe8 u = acc;
    {
        const int step = asc ? -1 : 1;
        int b = asc ? nb - 1 : 0;
        for (int it = 0; it < nb; it++, b += step) {
            const int bp = b + step;
            fe8 dx;
            fe8_sub(dx, xD, px[b]);
            fe8 ik;
            if (it + 1 < nb) {
                fe8_mul(ik, u, cc[bp]);
                fe8_mul(u, u, dx);
            } else {
                ik = u;
            }
            fe8 dy;
            fe8_sub(dy, yD, py[b]);
            fe8 lam, l2, xm, t, ym;
            fe8_mul(lam, dy, ik);
            fe8_sqr(l2, lam);
            fe8_sub2(xm, l2, px[b], xD);
            fe8_sub(t, px[b], xm);
            fe8_mul(ym, lam, t);
            fe8_sub(ym, ym, py[b]);

            uint64_t xp[8][4], xm_can[8][4];
            uint8_t par_p[8], par_m[8];
            fe8_canon_scatter_arr(xp, nullptr, px[b]);
            fe8_canon_scatter_arr(nullptr, par_p, py[b]);
            fe8_canon_scatter_arr(xm_can, nullptr, xm);
            fe8_canon_scatter_arr(nullptr, par_m, ym);

#ifdef QCG_EC_HOOK
            for (int l = 0; l < 8; l++) {
                const int ci = 8 * b + l;
                if (ci < n) QCG_EC_HOOK(w, ci, xp[l], (uint64_t *)py[b].l, xm_can[l], (uint64_t *)ym.l);
            }
#endif
            uint32_t W9_p[8][9], W9_m[8][9];
            for (int l = 0; l < 8; l++) {
                pub_words(W9_p[l], xp[l], (int)par_p[l]);
                pub_words(W9_m[l], xm_can[l], (int)par_m[l]);
            }
            uint32_t h0_p[8], h0_m[8];
            pub_hash8_words(W9_p, h0_p);
            pub_hash8_words(W9_m, h0_m);

            for (int l = 0; l < 8; l++) {
                const int ci = 8 * b + l;
                if (ci >= n) break;
#if QSB_ZEROS_N >= 32
                if (h0_p[l] == 0) publish(w, ci, 0);
                if (h0_m[l] == 0) publish(w, ci, 1);
#else
                if ((h0_p[l] >> (32 - QSB_ZEROS_N)) == 0) publish(w, ci, 0);
                if ((h0_m[l] >> (32 - QSB_ZEROS_N)) == 0) publish(w, ci, 1);
#endif
            }
        }
    }
}

#undef Q8_INL
#undef Q8_FN
#undef F8_M52

} /* namespace v8 */

#endif /* QSB_CG_IFMA8_H */
