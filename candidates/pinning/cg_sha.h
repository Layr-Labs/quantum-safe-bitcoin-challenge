/* cg_sha.h -- SHA-256 kernels for the pinning co-grinder (cpu_cogrind.h). Original code.
 *
 *   s8_*     : 8-lane AVX2 multi-buffer SHA-256 (one candidate per 32-bit lane).
 *   s16_*    : 16-lane AVX-512 multi-buffer SHA-256, selected by the host tuner
 *              only when F/BW are available and this path wins on live batches.
 *   shani_*  : SHA-NI (x86 SHA extensions) single-block compression, two independent blocks
 *              interleaved.
 * Every function is compiled with its own target attribute and is called only after the
 * matching runtime CPU check, so the file builds with a plain `gcc -O3` / nvcc host pass.
 *
 * Block shapes of the pinning candidate (suffix_len = 75, sequence in suffix block 0,
 * locktime in suffix block 1 -- the rival's `cache_first` layout):
 *   tail   : suffix block 1 from the per-sequence midstate. W0/W1 carry the locktime bytes,
 *            W2..W15 are problem constants.
 *   digest : SHA256 of the 32-byte first digest: W0..W7 variable, W8 = 0x80000000,
 *            W9..W14 = 0, W15 = 256.
 *   pubkey : SHA256 of the 33-byte compressed key: W0..W8 variable, W9..W14 = 0, W15 = 264.
 *            Only H0 is returned (the leading-zero gate needs <= 32 bits for N <= 32).
 */
#ifndef QSB_CG_SHA_H
#define QSB_CG_SHA_H
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

namespace qcg_sha {

static const uint32_t K256[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
static const uint32_t IV256[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};

/* ---------------- portable scalar reference (also the no-AVX2 fallback) ---------------- */
static inline uint32_t ror32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static inline void sha_compress_ref(uint32_t st[8], const uint32_t w_in[16]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = w_in[i];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ror32(w[i - 15], 7) ^ ror32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = ror32(w[i - 2], 17) ^ ror32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + (ror32(e, 6) ^ ror32(e, 11) ^ ror32(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + w[i];
        uint32_t t2 = (ror32(a, 2) ^ ror32(a, 13) ^ ror32(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

/* ---------------- 8-lane AVX2 ---------------- */
#define QSB_SHA_AVX2 __attribute__((target("avx2"), always_inline)) inline
#define QSB_SHA_AVX2F __attribute__((target("avx2"), noinline))
typedef __m256i v8u;

static QSB_SHA_AVX2 v8u s8_add(v8u a, v8u b) { return _mm256_add_epi32(a, b); }
static QSB_SHA_AVX2 v8u s8_xor(v8u a, v8u b) { return _mm256_xor_si256(a, b); }
static QSB_SHA_AVX2 v8u s8_and(v8u a, v8u b) { return _mm256_and_si256(a, b); }
static QSB_SHA_AVX2 v8u s8_set1(uint32_t x) { return _mm256_set1_epi32((int)x); }
#define S8_ROR(x, n) _mm256_or_si256(_mm256_srli_epi32((x), (n)), _mm256_slli_epi32((x), 32 - (n)))
static QSB_SHA_AVX2 v8u s8_S0(v8u a) {   /* ror2 ^ ror13 ^ ror22 */
    v8u r = _mm256_xor_si256(_mm256_srli_epi32(a, 2), _mm256_slli_epi32(a, 30));
    r = _mm256_xor_si256(r, _mm256_srli_epi32(a, 13)); r = _mm256_xor_si256(r, _mm256_slli_epi32(a, 19));
    r = _mm256_xor_si256(r, _mm256_srli_epi32(a, 22)); r = _mm256_xor_si256(r, _mm256_slli_epi32(a, 10));
    return r;
}
static QSB_SHA_AVX2 v8u s8_S1(v8u e) {   /* ror6 ^ ror11 ^ ror25 */
    v8u r = _mm256_xor_si256(_mm256_srli_epi32(e, 6), _mm256_slli_epi32(e, 26));
    r = _mm256_xor_si256(r, _mm256_srli_epi32(e, 11)); r = _mm256_xor_si256(r, _mm256_slli_epi32(e, 21));
    r = _mm256_xor_si256(r, _mm256_srli_epi32(e, 25)); r = _mm256_xor_si256(r, _mm256_slli_epi32(e, 7));
    return r;
}
static QSB_SHA_AVX2 v8u s8_s0(v8u w) {   /* ror7 ^ ror18 ^ shr3 */
    v8u r = _mm256_xor_si256(_mm256_srli_epi32(w, 7), _mm256_slli_epi32(w, 25));
    r = _mm256_xor_si256(r, _mm256_srli_epi32(w, 18)); r = _mm256_xor_si256(r, _mm256_slli_epi32(w, 14));
    r = _mm256_xor_si256(r, _mm256_srli_epi32(w, 3));
    return r;
}
static QSB_SHA_AVX2 v8u s8_s1(v8u w) {   /* ror17 ^ ror19 ^ shr10 */
    v8u r = _mm256_xor_si256(_mm256_srli_epi32(w, 17), _mm256_slli_epi32(w, 15));
    r = _mm256_xor_si256(r, _mm256_srli_epi32(w, 19)); r = _mm256_xor_si256(r, _mm256_slli_epi32(w, 13));
    r = _mm256_xor_si256(r, _mm256_srli_epi32(w, 10));
    return r;
}
/* one round; kw = K[i] + W[i] (already summed). Maj via (a^b)&(b^c)^b with bc = b^c carried. */
#define S8_ROUND(a, b, c, d, e, f, g, h, kw, bc) do {                                        \
        v8u t1_ = s8_add(s8_add(h, s8_S1(e)), s8_add(s8_xor(g, s8_and(e, s8_xor(f, g))), (kw))); \
        v8u ab_ = s8_xor(a, b);                                                                \
        v8u t2_ = s8_add(s8_S0(a), s8_xor(s8_and(ab_, bc), b));                                \
        d = s8_add(d, t1_); h = s8_add(t1_, t2_); bc = ab_;                                    \
    } while (0)

/* Generic 8-lane compression. st[8] and w[16] word-major (v8u per word). st updated in place.
 * The 64 rounds use a rotating register naming via an 8-step unrolled macro. */
static QSB_SHA_AVX2 void s8_compress_full(v8u st[8], const v8u w_in[16]) {
    v8u w[16];
    for (int i = 0; i < 16; i++) w[i] = w_in[i];
    v8u a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
    v8u bc = s8_xor(b, c);
#define S8_R8(base, WEXPR) \
    S8_ROUND(a, b, c, d, e, f, g, h, s8_add(s8_set1(K256[base + 0]), WEXPR(0)), bc); \
    S8_ROUND(h, a, b, c, d, e, f, g, s8_add(s8_set1(K256[base + 1]), WEXPR(1)), bc); \
    S8_ROUND(g, h, a, b, c, d, e, f, s8_add(s8_set1(K256[base + 2]), WEXPR(2)), bc); \
    S8_ROUND(f, g, h, a, b, c, d, e, s8_add(s8_set1(K256[base + 3]), WEXPR(3)), bc); \
    S8_ROUND(e, f, g, h, a, b, c, d, s8_add(s8_set1(K256[base + 4]), WEXPR(4)), bc); \
    S8_ROUND(d, e, f, g, h, a, b, c, s8_add(s8_set1(K256[base + 5]), WEXPR(5)), bc); \
    S8_ROUND(c, d, e, f, g, h, a, b, s8_add(s8_set1(K256[base + 6]), WEXPR(6)), bc); \
    S8_ROUND(b, c, d, e, f, g, h, a, s8_add(s8_set1(K256[base + 7]), WEXPR(7)), bc);
#define WLOAD0(k) w[(k)]
#define WLOAD8(k) w[8 + (k)]
    S8_R8(0, WLOAD0)
    S8_R8(8, WLOAD8)
#define WEXP(k) (w[(k)] = s8_add(s8_add(w[(k)], s8_s0(w[((k) + 1) & 15])), s8_add(w[((k) + 9) & 15], s8_s1(w[((k) + 14) & 15]))))
#define WEXP0(k) WEXP(k)
#define WEXP8(k) WEXP(8 + (k))
    for (int r = 16; r < 64; r += 16) {
        S8_R8(r, WEXP0)
        S8_R8(r + 8, WEXP8)
    }
#undef WLOAD0
#undef WLOAD8
#undef WEXP0
#undef WEXP8
    st[0] = s8_add(st[0], a); st[1] = s8_add(st[1], b); st[2] = s8_add(st[2], c); st[3] = s8_add(st[3], d);
    st[4] = s8_add(st[4], e); st[5] = s8_add(st[5], f); st[6] = s8_add(st[6], g); st[7] = s8_add(st[7], h);
}

/* byte-swap of each 32-bit lane */
static QSB_SHA_AVX2 v8u s8_bswap(v8u x) {
    const v8u m = _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
                                   3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    return _mm256_shuffle_epi8(x, m);
}

/* ---------------- 16-lane AVX-512 SHA-256d ----------------
 * The four message vectors of each 16-word schedule ring and eight chaining
 * words remain vector-valued. AVX-512F provides native 32-bit rotates, and
 * 32 architectural ZMM registers give the 16-message/8-state layout room to
 * execute sixteen independent candidates before materializing output. */
#define QSB_SHA_AVX512 __attribute__((target("avx512f,avx512bw"), always_inline)) inline
typedef __m512i v16u;
static QSB_SHA_AVX512 v16u s16_add(v16u a, v16u b) { return _mm512_add_epi32(a, b); }
static QSB_SHA_AVX512 v16u s16_xor(v16u a, v16u b) { return _mm512_xor_si512(a, b); }
static QSB_SHA_AVX512 v16u s16_and(v16u a, v16u b) { return _mm512_and_si512(a, b); }
static QSB_SHA_AVX512 v16u s16_set1(uint32_t x) { return _mm512_set1_epi32((int)x); }
static QSB_SHA_AVX512 v16u s16_S0(v16u a) {
    return s16_xor(s16_xor(_mm512_ror_epi32(a, 2), _mm512_ror_epi32(a, 13)), _mm512_ror_epi32(a, 22));
}
static QSB_SHA_AVX512 v16u s16_S1(v16u e) {
    return s16_xor(s16_xor(_mm512_ror_epi32(e, 6), _mm512_ror_epi32(e, 11)), _mm512_ror_epi32(e, 25));
}
static QSB_SHA_AVX512 v16u s16_s0(v16u w) {
    return s16_xor(s16_xor(_mm512_ror_epi32(w, 7), _mm512_ror_epi32(w, 18)), _mm512_srli_epi32(w, 3));
}
static QSB_SHA_AVX512 v16u s16_s1(v16u w) {
    return s16_xor(s16_xor(_mm512_ror_epi32(w, 17), _mm512_ror_epi32(w, 19)), _mm512_srli_epi32(w, 10));
}
#define S16_ROUND(a, b, c, d, e, f, g, h, kw, bc) do {                              \
        v16u t1_ = s16_add(s16_add(h, s16_S1(e)), s16_add(s16_xor(g, s16_and(e, s16_xor(f, g))), (kw))); \
        v16u ab_ = s16_xor(a, b);                                                      \
        v16u t2_ = s16_add(s16_S0(a), s16_xor(s16_and(ab_, bc), b));                    \
        d = s16_add(d, t1_); h = s16_add(t1_, t2_); bc = ab_;                          \
    } while (0)
static QSB_SHA_AVX512 void s16_compress_full(v16u st[8], const v16u w_in[16]) {
    v16u w[16];
    for (int i = 0; i < 16; i++) w[i] = w_in[i];
    v16u a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
    v16u bc = s16_xor(b, c);
#define S16_R8(base, WEXPR) \
    S16_ROUND(a, b, c, d, e, f, g, h, s16_add(s16_set1(K256[base + 0]), WEXPR(0)), bc); \
    S16_ROUND(h, a, b, c, d, e, f, g, s16_add(s16_set1(K256[base + 1]), WEXPR(1)), bc); \
    S16_ROUND(g, h, a, b, c, d, e, f, s16_add(s16_set1(K256[base + 2]), WEXPR(2)), bc); \
    S16_ROUND(f, g, h, a, b, c, d, e, s16_add(s16_set1(K256[base + 3]), WEXPR(3)), bc); \
    S16_ROUND(e, f, g, h, a, b, c, d, s16_add(s16_set1(K256[base + 4]), WEXPR(4)), bc); \
    S16_ROUND(d, e, f, g, h, a, b, c, s16_add(s16_set1(K256[base + 5]), WEXPR(5)), bc); \
    S16_ROUND(c, d, e, f, g, h, a, b, s16_add(s16_set1(K256[base + 6]), WEXPR(6)), bc); \
    S16_ROUND(b, c, d, e, f, g, h, a, s16_add(s16_set1(K256[base + 7]), WEXPR(7)), bc);
#define S16_WLOAD0(k) w[(k)]
#define S16_WLOAD8(k) w[8 + (k)]
    S16_R8(0, S16_WLOAD0)
    S16_R8(8, S16_WLOAD8)
#define S16_WEXP(k) (w[(k)] = s16_add(s16_add(w[(k)], s16_s0(w[((k) + 1) & 15])), s16_add(w[((k) + 9) & 15], s16_s1(w[((k) + 14) & 15]))))
#define S16_WEXP0(k) S16_WEXP(k)
#define S16_WEXP8(k) S16_WEXP(8 + (k))
    for (int r = 16; r < 64; r += 16) {
        S16_R8(r, S16_WEXP0)
        S16_R8(r + 8, S16_WEXP8)
    }
#undef S16_WLOAD0
#undef S16_WLOAD8
#undef S16_WEXP
#undef S16_WEXP0
#undef S16_WEXP8
#undef S16_R8
    st[0] = s16_add(st[0], a); st[1] = s16_add(st[1], b);
    st[2] = s16_add(st[2], c); st[3] = s16_add(st[3], d);
    st[4] = s16_add(st[4], e); st[5] = s16_add(st[5], f);
    st[6] = s16_add(st[6], g); st[7] = s16_add(st[7], h);
}
#undef S16_ROUND
static QSB_SHA_AVX512 v16u s16_bswap(v16u x) {
    const v16u m = _mm512_setr_epi32(0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f,
                                     0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f,
                                     0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f,
                                     0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f);
    return _mm512_shuffle_epi8(x, m);
}


/* ---------------- SHA-NI ---------------- */
#define QSB_SHA_NI __attribute__((target("sha,sse4.1"), always_inline)) inline
#define QSB_SHA_NIF __attribute__((target("sha,sse4.1"), noinline))
/* Two independent compressions, interleaved. st: 8 words (a..h) each; w: 16 message words each
 * (host-order 32-bit words, W0 first). If out0_only, only st[0] (= H0) is meaningful afterwards. */
static QSB_SHA_NI void shani_load_state(const uint32_t *st, __m128i &S0, __m128i &S1) {
    __m128i t = _mm_loadu_si128((const __m128i *)&st[0]);
    __m128i s1 = _mm_loadu_si128((const __m128i *)&st[4]);
    t = _mm_shuffle_epi32(t, 0xB1);            /* CDAB */
    s1 = _mm_shuffle_epi32(s1, 0x1B);          /* EFGH */
    S0 = _mm_alignr_epi8(t, s1, 8);            /* ABEF */
    S1 = _mm_blend_epi16(s1, t, 0xF0);         /* CDGH */
}
static QSB_SHA_NI void shani_store_state(uint32_t *st, __m128i S0, __m128i S1) {
    __m128i t = _mm_shuffle_epi32(S0, 0x1B);   /* FEBA */
    S1 = _mm_shuffle_epi32(S1, 0xB1);          /* DCHG */
    S0 = _mm_blend_epi16(t, S1, 0xF0);         /* DCBA */
    S1 = _mm_alignr_epi8(S1, t, 8);            /* HGFE */
    _mm_storeu_si128((__m128i *)&st[0], S0);
    _mm_storeu_si128((__m128i *)&st[4], S1);
}
static QSB_SHA_NI void shani_compress2_core(
    __m128i &A0, __m128i &A1, __m128i &B0, __m128i &B1,
    __m128i MA0, __m128i MA1, __m128i MA2, __m128i MA3,
    __m128i MB0, __m128i MB1, __m128i MB2, __m128i MB3) {
    const __m128i A0s = A0, A1s = A1, B0s = B0, B1s = B1;
    __m128i mA, mB, K;
    /* group g: rounds 4g..4g+3 on message vector Mc; Mn = next (msg2 target), Mp = previous
       (alignr source), Mq = the vector msg1 updates */
#define SHANI2_ROUNDS(g, McA, McB)                                                     \
    K = _mm_loadu_si128((const __m128i *)(K256 + 4 * (g)));                            \
    mA = _mm_add_epi32(McA, K); mB = _mm_add_epi32(McB, K);                            \
    A1 = _mm_sha256rnds2_epu32(A1, A0, mA); B1 = _mm_sha256rnds2_epu32(B1, B0, mB);
#define SHANI2_TAIL()                                                                  \
    mA = _mm_shuffle_epi32(mA, 0x0E); mB = _mm_shuffle_epi32(mB, 0x0E);                \
    A0 = _mm_sha256rnds2_epu32(A0, A1, mA); B0 = _mm_sha256rnds2_epu32(B0, B1, mB);
#define SHANI2_MSG2(McA, McB, MpA, MpB, MnA, MnB)                                      \
    MnA = _mm_sha256msg2_epu32(_mm_add_epi32(MnA, _mm_alignr_epi8(McA, MpA, 4)), McA);  \
    MnB = _mm_sha256msg2_epu32(_mm_add_epi32(MnB, _mm_alignr_epi8(McB, MpB, 4)), McB);
#define SHANI2_MSG1(MqA, MqB, McA, McB)                                                \
    MqA = _mm_sha256msg1_epu32(MqA, McA); MqB = _mm_sha256msg1_epu32(MqB, McB);
    /* g = 0 */  SHANI2_ROUNDS(0, MA0, MB0) SHANI2_TAIL()
    /* g = 1 */  SHANI2_ROUNDS(1, MA1, MB1) SHANI2_TAIL() SHANI2_MSG1(MA0, MB0, MA1, MB1)
    /* g = 2 */  SHANI2_ROUNDS(2, MA2, MB2) SHANI2_TAIL() SHANI2_MSG1(MA1, MB1, MA2, MB2)
    /* g = 3 */  SHANI2_ROUNDS(3, MA3, MB3) SHANI2_MSG2(MA3, MB3, MA2, MB2, MA0, MB0) SHANI2_TAIL() SHANI2_MSG1(MA2, MB2, MA3, MB3)
    /* g = 4 */  SHANI2_ROUNDS(4, MA0, MB0) SHANI2_MSG2(MA0, MB0, MA3, MB3, MA1, MB1) SHANI2_TAIL() SHANI2_MSG1(MA3, MB3, MA0, MB0)
    /* g = 5 */  SHANI2_ROUNDS(5, MA1, MB1) SHANI2_MSG2(MA1, MB1, MA0, MB0, MA2, MB2) SHANI2_TAIL() SHANI2_MSG1(MA0, MB0, MA1, MB1)
    /* g = 6 */  SHANI2_ROUNDS(6, MA2, MB2) SHANI2_MSG2(MA2, MB2, MA1, MB1, MA3, MB3) SHANI2_TAIL() SHANI2_MSG1(MA1, MB1, MA2, MB2)
    /* g = 7 */  SHANI2_ROUNDS(7, MA3, MB3) SHANI2_MSG2(MA3, MB3, MA2, MB2, MA0, MB0) SHANI2_TAIL() SHANI2_MSG1(MA2, MB2, MA3, MB3)
    /* g = 8 */  SHANI2_ROUNDS(8, MA0, MB0) SHANI2_MSG2(MA0, MB0, MA3, MB3, MA1, MB1) SHANI2_TAIL() SHANI2_MSG1(MA3, MB3, MA0, MB0)
    /* g = 9 */  SHANI2_ROUNDS(9, MA1, MB1) SHANI2_MSG2(MA1, MB1, MA0, MB0, MA2, MB2) SHANI2_TAIL() SHANI2_MSG1(MA0, MB0, MA1, MB1)
    /* g = 10 */ SHANI2_ROUNDS(10, MA2, MB2) SHANI2_MSG2(MA2, MB2, MA1, MB1, MA3, MB3) SHANI2_TAIL() SHANI2_MSG1(MA1, MB1, MA2, MB2)
    /* g = 11 */ SHANI2_ROUNDS(11, MA3, MB3) SHANI2_MSG2(MA3, MB3, MA2, MB2, MA0, MB0) SHANI2_TAIL() SHANI2_MSG1(MA2, MB2, MA3, MB3)
    /* g = 12 */ SHANI2_ROUNDS(12, MA0, MB0) SHANI2_MSG2(MA0, MB0, MA3, MB3, MA1, MB1) SHANI2_TAIL() SHANI2_MSG1(MA3, MB3, MA0, MB0)
    /* g = 13 */ SHANI2_ROUNDS(13, MA1, MB1) SHANI2_MSG2(MA1, MB1, MA0, MB0, MA2, MB2) SHANI2_TAIL()
    /* g = 14 */ SHANI2_ROUNDS(14, MA2, MB2) SHANI2_MSG2(MA2, MB2, MA1, MB1, MA3, MB3) SHANI2_TAIL()
    /* g = 15 */ SHANI2_ROUNDS(15, MA3, MB3) SHANI2_TAIL()
#undef SHANI2_ROUNDS
#undef SHANI2_TAIL
#undef SHANI2_MSG2
#undef SHANI2_MSG1
    A0 = _mm_add_epi32(A0, A0s); A1 = _mm_add_epi32(A1, A1s);
    B0 = _mm_add_epi32(B0, B0s); B1 = _mm_add_epi32(B1, B1s);
}
/* Generic pair compression, retaining all eight output words. */
static QSB_SHA_NI void shani_compress2(uint32_t *stA, const uint32_t *wA, uint32_t *stB, const uint32_t *wB) {
    __m128i A0, A1, B0, B1;
    shani_load_state(stA, A0, A1); shani_load_state(stB, B0, B1);
    shani_compress2_core(A0, A1, B0, B1,
        _mm_loadu_si128((const __m128i *)(wA + 0)), _mm_loadu_si128((const __m128i *)(wA + 4)),
        _mm_loadu_si128((const __m128i *)(wA + 8)), _mm_loadu_si128((const __m128i *)(wA + 12)),
        _mm_loadu_si128((const __m128i *)(wB + 0)), _mm_loadu_si128((const __m128i *)(wB + 4)),
        _mm_loadu_si128((const __m128i *)(wB + 8)), _mm_loadu_si128((const __m128i *)(wB + 12)));
    shani_store_state(stA, A0, A1); shani_store_state(stB, B0, B1);
}
/* Two SHA-256d chains in lockstep. The first digest becomes the second block's
 * W0..7 in registers; only the final 256-bit outputs touch memory. W8..15
 * are Bitcoin's fixed 32-byte padding, with no per-candidate pad arrays. */
static QSB_SHA_NI void shani_sha256d2(uint32_t *stA, const uint32_t *wA, uint32_t *stB, const uint32_t *wB) {
    __m128i A0, A1, B0, B1;
    shani_load_state(stA, A0, A1); shani_load_state(stB, B0, B1);
    shani_compress2_core(A0, A1, B0, B1,
        _mm_loadu_si128((const __m128i *)(wA + 0)), _mm_loadu_si128((const __m128i *)(wA + 4)),
        _mm_loadu_si128((const __m128i *)(wA + 8)), _mm_loadu_si128((const __m128i *)(wA + 12)),
        _mm_loadu_si128((const __m128i *)(wB + 0)), _mm_loadu_si128((const __m128i *)(wB + 4)),
        _mm_loadu_si128((const __m128i *)(wB + 8)), _mm_loadu_si128((const __m128i *)(wB + 12)));
    /* S0 lanes are F,E,B,A and S1 lanes H,G,D,C. Reconstruct A..D and
     * E..H exactly as shani_store_state, but retain them as message vectors. */
    const __m128i at = _mm_shuffle_epi32(A0, 0x1B), au = _mm_shuffle_epi32(A1, 0xB1);
    const __m128i bt = _mm_shuffle_epi32(B0, 0x1B), bu = _mm_shuffle_epi32(B1, 0xB1);
    const __m128i aW0 = _mm_blend_epi16(at, au, 0xF0), aW1 = _mm_alignr_epi8(au, at, 8);
    const __m128i bW0 = _mm_blend_epi16(bt, bu, 0xF0), bW1 = _mm_alignr_epi8(bu, bt, 8);
    const __m128i pad8 = _mm_setr_epi32((int)0x80000000u, 0, 0, 0);
    const __m128i pad12 = _mm_setr_epi32(0, 0, 0, 256);
    shani_load_state(IV256, A0, A1); shani_load_state(IV256, B0, B1);
    shani_compress2_core(A0, A1, B0, B1, aW0, aW1, pad8, pad12,
                                            bW0, bW1, pad8, pad12);
    shani_store_state(stA, A0, A1); shani_store_state(stB, B0, B1);
}
/* single compression (tests only) */
static QSB_SHA_NI void shani_compress1(uint32_t *st, const uint32_t *w) {
    uint32_t st2[8]; memcpy(st2, st, 32);
    shani_compress2(st, w, st2, w);
}

} /* namespace qcg_sha */
#endif
