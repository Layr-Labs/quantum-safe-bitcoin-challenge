/* Host-only 5x52 IFMA field from promoted Subset a137e28.
 * ercumentyildirim / terrapinelf / Meganpark980320; libsecp256k1 lineage,
 * MIT notice retained in COPYING-secp256k1. See SUBMISSION.md for provenance. */
#pragma once
struct fe8 { __m512i l[5]; };
#define F8_M52 _mm512_set1_epi64(0xFFFFFFFFFFFFFULL)
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_carry(fe8 &r) {
    /* fold bits >= 256 first (limb 4 bit 48 and up; 2^256 = 0x1000003D1 mod p), then one carry chain:
       limbs 0..3 end < 2^52 and limb 4 < 2^48 + (small carry), so the value is < 2^257 */
    const __m512i M = F8_M52, M48 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL), K = _mm512_set1_epi64(0x1000003D1ULL);
    __m512i c;
    c = _mm512_srli_epi64(r.l[4], 48); r.l[4] = _mm512_and_si512(r.l[4], M48);
    r.l[0] = _mm512_madd52lo_epu64(r.l[0], c, K);             /* c*K < 2^52 */
    c = _mm512_srli_epi64(r.l[0], 52); r.l[0] = _mm512_and_si512(r.l[0], M); r.l[1] = _mm512_add_epi64(r.l[1], c);
    c = _mm512_srli_epi64(r.l[1], 52); r.l[1] = _mm512_and_si512(r.l[1], M); r.l[2] = _mm512_add_epi64(r.l[2], c);
    c = _mm512_srli_epi64(r.l[2], 52); r.l[2] = _mm512_and_si512(r.l[2], M); r.l[3] = _mm512_add_epi64(r.l[3], c);
    c = _mm512_srli_epi64(r.l[3], 52); r.l[3] = _mm512_and_si512(r.l[3], M); r.l[4] = _mm512_add_epi64(r.l[4], c);
}
/* Reduce a 10-column product (columns < 2^57, value < 2^514) to a normalized element.
 * Shorter reduction (after Meganpark980320's bb2a3eb7): the high columns c5..c9 are not normalized by
 * a serial carry chain before the fold. Each is split into its low 52 bits and the rest (< 2^5); both
 * parts are folded with 2^260 = R = 0x1000003D10 (mod p): lo*R as a lo/hi IFMA pair, rest*R (< 2^42)
 * with one lo IFMA. The part landing at 2^260 again (from c9) is folded once more, the bits of
 * column 4 at and above 2^256 are folded with 0x1000003D1, and one carry chain finishes.
 * 18 IFMA and 24 shift/and/add instead of 14 IFMA and ~47, with no serial chain before the fold. */
static inline __attribute__((always_inline, target("avx512f,avx512ifma")))
void fe8_red(fe8 &r, __m512i c0, __m512i c1, __m512i c2, __m512i c3, __m512i c4,
             __m512i c5, __m512i c6, __m512i c7, __m512i c8, __m512i c9) {
#define LO(acc, x, y) acc = _mm512_madd52lo_epu64(acc, x, y)
#define HI(acc, x, y) acc = _mm512_madd52hi_epu64(acc, x, y)
    const __m512i M = F8_M52, M48 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL);
    const __m512i R = _mm512_set1_epi64(0x1000003D10ULL), K = _mm512_set1_epi64(0x1000003D1ULL);
    const __m512i l5 = _mm512_and_si512(c5, M), h5 = _mm512_srli_epi64(c5, 52);
    const __m512i l6 = _mm512_and_si512(c6, M), h6 = _mm512_srli_epi64(c6, 52);
    const __m512i l7 = _mm512_and_si512(c7, M), h7 = _mm512_srli_epi64(c7, 52);
    const __m512i l8 = _mm512_and_si512(c8, M), h8 = _mm512_srli_epi64(c8, 52);
    const __m512i l9 = _mm512_and_si512(c9, M), h9 = _mm512_srli_epi64(c9, 52);
    LO(c0, l5, R); HI(c1, l5, R); LO(c1, h5, R);
    LO(c1, l6, R); HI(c2, l6, R); LO(c2, h6, R);
    LO(c2, l7, R); HI(c3, l7, R); LO(c3, h7, R);
    LO(c3, l8, R); HI(c4, l8, R); LO(c4, h8, R);
    LO(c4, l9, R);
    __m512i t5 = _mm512_setzero_si512(); HI(t5, l9, R); LO(t5, h9, R);   /* weight 2^260, < 2^42 */
    LO(c0, t5, R); HI(c1, t5, R);
    const __m512i x = _mm512_srli_epi64(c4, 48); c4 = _mm512_and_si512(c4, M48);   /* bits >= 2^256 */
    LO(c0, x, K);
    __m512i t;
    t = _mm512_srli_epi64(c0, 52); c0 = _mm512_and_si512(c0, M); c1 = _mm512_add_epi64(c1, t);
    t = _mm512_srli_epi64(c1, 52); c1 = _mm512_and_si512(c1, M); c2 = _mm512_add_epi64(c2, t);
    t = _mm512_srli_epi64(c2, 52); c2 = _mm512_and_si512(c2, M); c3 = _mm512_add_epi64(c3, t);
    t = _mm512_srli_epi64(c3, 52); c3 = _mm512_and_si512(c3, M); c4 = _mm512_add_epi64(c4, t);
    r.l[0] = c0; r.l[1] = c1; r.l[2] = c2; r.l[3] = c3; r.l[4] = c4;
}
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_mul(fe8 &r, const fe8 &a, const fe8 &b) {
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
/* r = a^2: the 10 cross products once (column sums x_k), doubled, plus the 5 squares:
 * 30 IFMA instead of 50. Column bounds match fe8_mul's (< 9 * 2^52). */
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_sqr(fe8 &r, const fe8 &a) {
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
/* r = a + b (inputs normalized) */
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_add(fe8 &r, const fe8 &a, const fe8 &b) {
    for (int i = 0; i < 5; i++) r.l[i] = _mm512_add_epi64(a.l[i], b.l[i]);
    fe8_carry(r);
}
/* r = a - b (inputs normalized, value < 2^257): a + 4p - b, all limbs stay non-negative */
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_sub(fe8 &r, const fe8 &a, const fe8 &b) {
    const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
                  P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4);
    r.l[0] = _mm512_sub_epi64(_mm512_add_epi64(a.l[0], P0), b.l[0]);
    r.l[1] = _mm512_sub_epi64(_mm512_add_epi64(a.l[1], P1), b.l[1]);
    r.l[2] = _mm512_sub_epi64(_mm512_add_epi64(a.l[2], P1), b.l[2]);
    r.l[3] = _mm512_sub_epi64(_mm512_add_epi64(a.l[3], P1), b.l[3]);
    r.l[4] = _mm512_sub_epi64(_mm512_add_epi64(a.l[4], P4), b.l[4]);
    fe8_carry(r);
}
/* r = a - b - c (inputs normalized): a + 8p - b - c with one carry pass; every limb stays non-negative
 * (8p's limbs exceed the sum of two normalized limbs) and below 2^56 */
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_sub2(fe8 &r, const fe8 &a, const fe8 &b, const fe8 &c) {
    const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 8), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 8),
                  P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 8);
    r.l[0] = _mm512_sub_epi64(_mm512_add_epi64(a.l[0], P0), _mm512_add_epi64(b.l[0], c.l[0]));
    r.l[1] = _mm512_sub_epi64(_mm512_add_epi64(a.l[1], P1), _mm512_add_epi64(b.l[1], c.l[1]));
    r.l[2] = _mm512_sub_epi64(_mm512_add_epi64(a.l[2], P1), _mm512_add_epi64(b.l[2], c.l[2]));
    r.l[3] = _mm512_sub_epi64(_mm512_add_epi64(a.l[3], P1), _mm512_add_epi64(b.l[3], c.l[3]));
    r.l[4] = _mm512_sub_epi64(_mm512_add_epi64(a.l[4], P4), _mm512_add_epi64(b.l[4], c.l[4]));
    fe8_carry(r);
}
/* r = a - b - 2c (inputs normalized): a + 12p - b - 2c, one carry pass (limbs stay in [0, 2^57)) */
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_sub3(fe8 &r, const fe8 &a, const fe8 &b, const fe8 &c) {
    const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 12), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 12),
                  P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 12);
    const __m512i Pl[5] = {P0, P1, P1, P1, P4};
    for (int i = 0; i < 5; i++)
        r.l[i] = _mm512_sub_epi64(_mm512_add_epi64(a.l[i], Pl[i]), _mm512_add_epi64(b.l[i], _mm512_add_epi64(c.l[i], c.l[i])));
    fe8_carry(r);
}
/* Carry chain without the fold of limb 4's bits >= 48: limbs 0..3 end < 2^52 and limb 4 stays below 2^52
 * (it enters below 2^51.3 here). Enough for values that only feed multiplications (IFMA reads 52 bits per
 * limb, and fe8_red's column bounds depend only on the limbs being < 2^52), not for canonicalization. */
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_carry_m(fe8 &r) {
    const __m512i M = F8_M52;
    __m512i c;
    c = _mm512_srli_epi64(r.l[0], 52); r.l[0] = _mm512_and_si512(r.l[0], M); r.l[1] = _mm512_add_epi64(r.l[1], c);
    c = _mm512_srli_epi64(r.l[1], 52); r.l[1] = _mm512_and_si512(r.l[1], M); r.l[2] = _mm512_add_epi64(r.l[2], c);
    c = _mm512_srli_epi64(r.l[2], 52); r.l[2] = _mm512_and_si512(r.l[2], M); r.l[3] = _mm512_add_epi64(r.l[3], c);
    c = _mm512_srli_epi64(r.l[3], 52); r.l[3] = _mm512_and_si512(r.l[3], M); r.l[4] = _mm512_add_epi64(r.l[4], c);
}
/* r = a - b (inputs normalized) for multiplication inputs only: a + 4p - b, fe8_carry_m (limb 4 < 2^50.2) */
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_sub_m(fe8 &r, const fe8 &a, const fe8 &b) {
    const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
                  P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4);
    r.l[0] = _mm512_sub_epi64(_mm512_add_epi64(a.l[0], P0), b.l[0]);
    r.l[1] = _mm512_sub_epi64(_mm512_add_epi64(a.l[1], P1), b.l[1]);
    r.l[2] = _mm512_sub_epi64(_mm512_add_epi64(a.l[2], P1), b.l[2]);
    r.l[3] = _mm512_sub_epi64(_mm512_add_epi64(a.l[3], P1), b.l[3]);
    r.l[4] = _mm512_sub_epi64(_mm512_add_epi64(a.l[4], P4), b.l[4]);
    fe8_carry_m(r);
}
/* r = a*b - s (a, b, s normalized) with one reduction: the product columns 0..4 start at 4p - s instead
 * of zero (every limb of 4p exceeds a normalized limb, so they stay non-negative and below 2^54; column 4
 * ends below 13 * 2^52 < 2^57), which replaces a separate subtraction and its carry pass. */
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_mul_sub(fe8 &r, const fe8 &a, const fe8 &b, const fe8 &s) {
#define LO(acc, x, y) acc = _mm512_madd52lo_epu64(acc, x, y)
#define HI(acc, x, y) acc = _mm512_madd52hi_epu64(acc, x, y)
    const __m512i Z = _mm512_setzero_si512();
    const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
                  P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4);
    __m512i c0 = _mm512_sub_epi64(P0, s.l[0]), c1 = _mm512_sub_epi64(P1, s.l[1]), c2 = _mm512_sub_epi64(P1, s.l[2]),
            c3 = _mm512_sub_epi64(P1, s.l[3]), c4 = _mm512_sub_epi64(P4, s.l[4]), c5 = Z, c6 = Z, c7 = Z, c8 = Z, c9 = Z;
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
/* r = a^2 - d - 2x (a, x normalized; d normalized or from fe8_sub_m, limb 4 < 2^50.2) with one reduction:
 * 12p - d - 2x (limbs non-negative: d4 + 2x4 < 2^51.1 < 12p4; all below 2^55.6)
 * is added to the square's columns 0..4 (which then stay below 2^56.4 < 2^57), replacing fe8_sub3's
 * separate subtraction and carry pass. */
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_sqr_sub3(fe8 &r, const fe8 &a, const fe8 &d, const fe8 &x) {
    const __m512i Z = _mm512_setzero_si512();
    const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 12), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 12),
                  P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 12);
    __m512i x1 = Z, x2 = Z, x3 = Z, x4 = Z, x5 = Z, x6 = Z, x7 = Z, x8 = Z;
    const __m512i a0 = a.l[0], a1 = a.l[1], a2 = a.l[2], a3 = a.l[3], a4 = a.l[4];
    LO(x1,a0,a1); HI(x2,a0,a1);
    LO(x2,a0,a2); HI(x3,a0,a2);
    LO(x3,a0,a3); LO(x3,a1,a2); HI(x4,a0,a3); HI(x4,a1,a2);
    LO(x4,a0,a4); LO(x4,a1,a3); HI(x5,a0,a4); HI(x5,a1,a3);
    LO(x5,a1,a4); LO(x5,a2,a3); HI(x6,a1,a4); HI(x6,a2,a3);
    LO(x6,a2,a4); HI(x7,a2,a4);
    LO(x7,a3,a4); HI(x8,a3,a4);
    __m512i s0 = _mm512_sub_epi64(P0, _mm512_add_epi64(d.l[0], _mm512_add_epi64(x.l[0], x.l[0]))),
            s1 = _mm512_sub_epi64(P1, _mm512_add_epi64(d.l[1], _mm512_add_epi64(x.l[1], x.l[1]))),
            s2 = _mm512_sub_epi64(P1, _mm512_add_epi64(d.l[2], _mm512_add_epi64(x.l[2], x.l[2]))),
            s3 = _mm512_sub_epi64(P1, _mm512_add_epi64(d.l[3], _mm512_add_epi64(x.l[3], x.l[3]))),
            s4 = _mm512_sub_epi64(P4, _mm512_add_epi64(d.l[4], _mm512_add_epi64(x.l[4], x.l[4])));
    __m512i c0 = s0, c1 = _mm512_add_epi64(_mm512_slli_epi64(x1, 1), s1), c2 = _mm512_add_epi64(_mm512_slli_epi64(x2, 1), s2),
            c3 = _mm512_add_epi64(_mm512_slli_epi64(x3, 1), s3), c4 = _mm512_add_epi64(_mm512_slli_epi64(x4, 1), s4),
            c5 = _mm512_slli_epi64(x5, 1), c6 = _mm512_slli_epi64(x6, 1), c7 = _mm512_slli_epi64(x7, 1), c8 = _mm512_slli_epi64(x8, 1), c9 = Z;
    LO(c0,a0,a0); HI(c1,a0,a0);
    LO(c2,a1,a1); HI(c3,a1,a1);
    LO(c4,a2,a2); HI(c5,a2,a2);
    LO(c6,a3,a3); HI(c7,a3,a3);
    LO(c8,a4,a4); HI(c9,a4,a4);
    fe8_red(r, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);
#undef LO
#undef HI
}
/* r = (m ? -t : t) - y (t, y normalized) with one carry pass: 8p + t - y, or 8p - t - y in the lanes of m
 * (8p's limbs exceed the sum of two normalized limbs, so every limb stays non-negative and below 2^55.2).
 * Replaces fe8_cneg followed by fe8_sub for the signed table rows. Output: limbs 0..3 < 2^52, limb 4 < 2^51.3
 * (multiplication input only). */
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_sub_sgn(fe8 &r, const fe8 &t, const fe8 &y, __mmask8 m) {
    const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 8), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 8),
                  P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 8);
    const __m512i Pl[5] = {P0, P1, P1, P1, P4};
    for (int i = 0; i < 5; i++)
        r.l[i] = _mm512_sub_epi64(_mm512_mask_sub_epi64(_mm512_add_epi64(Pl[i], t.l[i]), m, Pl[i], t.l[i]), y.l[i]);
    fe8_carry_m(r);                                    /* the result only feeds a multiplication (limb 4 < 2^51.3) */
}
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_mov(fe8 &r, const fe8 &a) {                 /* explicit vector copy (a struct copy became rep movsq) */
    for (int i = 0; i < 5; i++) _mm512_store_si512((void *)&r.l[i], _mm512_load_si512((const void *)&a.l[i]));
}
/* Fully reduce a normalized element (value < 2^257) to [0, p) and return its 4x64 little-endian
 * words per lane: fold the bits at and above 2^256 twice (value then < 2^256), then add
 * 2^256 - p = 0x1000003D1 and keep the sum when it carries past 2^256 (value >= p). */
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_canon_words(__m512i w[4], const fe8 &x) {
    const __m512i M = F8_M52, M48 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL), K = _mm512_set1_epi64(0x1000003D1ULL);
    __m512i l0 = x.l[0], l1 = x.l[1], l2 = x.l[2], l3 = x.l[3], l4 = x.l[4], c;
    for (int it = 0; it < 2; it++) {
        c = _mm512_srli_epi64(l4, 48); l4 = _mm512_and_si512(l4, M48);
        l0 = _mm512_madd52lo_epu64(l0, c, K);
        c = _mm512_srli_epi64(l0, 52); l0 = _mm512_and_si512(l0, M); l1 = _mm512_add_epi64(l1, c);
        c = _mm512_srli_epi64(l1, 52); l1 = _mm512_and_si512(l1, M); l2 = _mm512_add_epi64(l2, c);
        c = _mm512_srli_epi64(l2, 52); l2 = _mm512_and_si512(l2, M); l3 = _mm512_add_epi64(l3, c);
        c = _mm512_srli_epi64(l3, 52); l3 = _mm512_and_si512(l3, M); l4 = _mm512_add_epi64(l4, c);
    }
    __m512i u0 = _mm512_add_epi64(l0, K), u1, u2, u3, u4;
    c = _mm512_srli_epi64(u0, 52); u0 = _mm512_and_si512(u0, M); u1 = _mm512_add_epi64(l1, c);
    c = _mm512_srli_epi64(u1, 52); u1 = _mm512_and_si512(u1, M); u2 = _mm512_add_epi64(l2, c);
    c = _mm512_srli_epi64(u2, 52); u2 = _mm512_and_si512(u2, M); u3 = _mm512_add_epi64(l3, c);
    c = _mm512_srli_epi64(u3, 52); u3 = _mm512_and_si512(u3, M); u4 = _mm512_add_epi64(l4, c);
    const __mmask8 ge = _mm512_test_epi64_mask(u4, _mm512_set1_epi64(1ULL << 48));
    l0 = _mm512_mask_mov_epi64(l0, ge, u0); l1 = _mm512_mask_mov_epi64(l1, ge, u1); l2 = _mm512_mask_mov_epi64(l2, ge, u2);
    l3 = _mm512_mask_mov_epi64(l3, ge, u3); l4 = _mm512_mask_mov_epi64(l4, ge, _mm512_and_si512(u4, M48));
    w[0] = _mm512_or_si512(l0, _mm512_slli_epi64(l1, 52));
    w[1] = _mm512_or_si512(_mm512_srli_epi64(l1, 12), _mm512_slli_epi64(l2, 40));
    w[2] = _mm512_or_si512(_mm512_srli_epi64(l2, 24), _mm512_slli_epi64(l3, 28));
    w[3] = _mm512_or_si512(_mm512_srli_epi64(l3, 36), _mm512_slli_epi64(l4, 16));
}
/* y <- -y in the lanes of m (signed table digits) */
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_cneg(fe8 &y, __mmask8 m) {
    if (!m) return;
    fe8 z, n; for (int i = 0; i < 5; i++) z.l[i] = _mm512_setzero_si512();
    fe8_sub(n, z, y);
    for (int i = 0; i < 5; i++) y.l[i] = _mm512_mask_mov_epi64(y.l[i], m, n.l[i]);
}

#define Q8T __attribute__((target("avx512f,avx512ifma")))
Q8T static inline void fe8_from64(fe8 &r, __m512i a0, __m512i a1, __m512i a2, __m512i a3) {
    const __m512i M = F8_M52;
    r.l[0] = _mm512_and_si512(a0, M);
    r.l[1] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(a0, 52), _mm512_slli_epi64(a1, 12)), M);
    r.l[2] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(a1, 40), _mm512_slli_epi64(a2, 24)), M);
    r.l[3] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(a2, 28), _mm512_slli_epi64(a3, 36)), M);
    r.l[4] = _mm512_srli_epi64(a3, 16);
}
/* Load 8 table points (one 64 B row each: x0..x3 y0..y3) and transpose to SoA. */
Q8T static inline void pt8_load(fe8 &x, fe8 &y, const tentry *const *rows) {
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

