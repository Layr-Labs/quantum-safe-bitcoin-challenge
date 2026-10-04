/* cg_sha.h -- SHA-256 kernels for the pinning co-grinder (cpu_cogrind.h). Original code.
 *
 *   sha8_*   : 8-lane AVX2 multi-buffer SHA-256 (one candidate per 32-bit lane). Used when the
 *              CPU has no SHA extensions (the expected ranked host).
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

static constexpr uint32_t K256[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
static constexpr uint32_t IV256[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};

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

/* ---------------- structured 8-lane compressions ----------------
 * The co-grinder's three block shapes have message words that are the same in all 8 lanes of a call
 * (problem constants, SHA padding) and a scalar initial state (per-sequence midstate, or the IV).
 * s8_compress_plan<VM, H0ONLY> takes the varying words as vectors (bit j of VM: word j varies) and
 * everything else from a precomputed s8_plan:
 *   kw[t]  K_t + W_t for every message-independent W_t (no vector add in that round),
 *   kc[t]  the message-independent part of schedule word t >= 16 (sigma terms of constant words
 *          folded; terms of zero words vanish),
 *   st0    the scalar initial state, with round 0 folded (a1 = W0 + a1c, e1 = W0 + e1c).
 * Same function as s8_compress_full on the same message and state, bit for bit. */
struct s8_plan { uint32_t kw[64], kc[64], st0[8], a1c, e1c; };
static constexpr uint32_t c_ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static constexpr uint32_t c_s0(uint32_t x) { return c_ror(x, 7) ^ c_ror(x, 18) ^ (x >> 3); }
static constexpr uint32_t c_s1(uint32_t x) { return c_ror(x, 17) ^ c_ror(x, 19) ^ (x >> 10); }
static constexpr uint32_t c_S0(uint32_t x) { return c_ror(x, 2) ^ c_ror(x, 13) ^ c_ror(x, 22); }
static constexpr uint32_t c_S1(uint32_t x) { return c_ror(x, 6) ^ c_ror(x, 11) ^ c_ror(x, 25); }
/* bit t: schedule word t depends on a varying message word */
static constexpr uint64_t s8_varmask(uint32_t vm) {
    uint64_t m = vm & 0xFFFFu;
    for (int t = 16; t < 64; t++)
        if (((m >> (t - 2)) | (m >> (t - 7)) | (m >> (t - 15)) | (m >> (t - 16))) & 1) m |= 1ull << t;
    return m;
}
/* wc: the 16 message words (entries of varying words are ignored); st0: initial state */
static constexpr s8_plan s8_make_plan(const uint32_t *wc, uint32_t vm, const uint32_t *st0) {
    s8_plan p{};
    const uint64_t V = s8_varmask(vm);
    uint32_t w[64] = {};
    for (int t = 0; t < 16; t++) w[t] = ((V >> t) & 1) ? 0u : wc[t];
    for (int t = 16; t < 64; t++) {
        uint32_t k = 0;
        if (!((V >> (t - 2)) & 1)) k += c_s1(w[t - 2]);
        if (!((V >> (t - 7)) & 1)) k += w[t - 7];
        if (!((V >> (t - 15)) & 1)) k += c_s0(w[t - 15]);
        if (!((V >> (t - 16)) & 1)) k += w[t - 16];
        p.kc[t] = k;
        w[t] = ((V >> t) & 1) ? 0u : k;
    }
    for (int t = 0; t < 64; t++) p.kw[t] = K256[t] + w[t];
    for (int i = 0; i < 8; i++) p.st0[i] = st0[i];
    const uint32_t a = st0[0], b = st0[1], c = st0[2], d = st0[3], e = st0[4], f = st0[5], g = st0[6], h = st0[7];
    const uint32_t c1 = h + c_S1(e) + ((e & f) ^ (~e & g)) + K256[0];
    const uint32_t c2 = c_S0(a) + ((a & b) ^ (a & c) ^ (b & c));
    p.a1c = c1 + c2; p.e1c = d + c1;
    return p;
}
static constexpr uint32_t S8_W_DIGEST[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0x80000000u, 0, 0, 0, 0, 0, 0, 256};
static constexpr uint32_t S8_W_PUBKEY[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 264};
static constexpr s8_plan S8_PLAN_DIGEST = s8_make_plan(S8_W_DIGEST, 0xFFu, IV256);      /* SHA256 of a 32-byte message */
static constexpr s8_plan S8_PLAN_PUBKEY = s8_make_plan(S8_W_PUBKEY, 0x1FFu, IV256);     /* SHA256 of a 33-byte message */

/* varying words Wv[j] (j with bit j of VM set; word 0 must vary); out: 8 state words, or only out[0]
 * (= H0) with H0ONLY. Raw words use a 32-vector ring: fold constant terms in W16..31,
 * then expand eight words just before their eight rounds. K+W is formed on consumption,
 * never stored over a live raw word. The original round0 fold and all64 rounds remain. */
template <uint32_t VM, int H0ONLY>
static QSB_SHA_AVX2 void s8_compress_plan(v8u out[8], const v8u *Wv, const s8_plan &P) {
    static_assert(VM & 1, "round 0 folding needs a varying word 0");
    constexpr uint64_t V = s8_varmask(VM);
    static_assert((V >> 16) == (~0ull >> 16), "every schedule word must depend on the message");
    v8u W[32];
    for (int j = 0; j < 16; j++) if ((V >> j) & 1) W[j] = Wv[j];
    /* W16..W31 with the constant terms folded (compile-time structure) */
#define S8P_W(t) do {                                                                                     \
        v8u acc_; int has_ = 0;                                                                           \
        if ((V >> ((t) - 2)) & 1) { acc_ = s8_s1(W[(t) - 2]); has_ = 1; }                               \
        if ((V >> ((t) - 7)) & 1) { acc_ = has_ ? s8_add(acc_, W[(t) - 7]) : W[(t) - 7]; has_ = 1; }    \
        if ((V >> ((t) - 15)) & 1) { const v8u s_ = s8_s0(W[(t) - 15]); acc_ = has_ ? s8_add(acc_, s_) : s_; has_ = 1; } \
        if ((V >> ((t) - 16)) & 1) { acc_ = has_ ? s8_add(acc_, W[(t) - 16]) : W[(t) - 16]; has_ = 1; } \
        if (!((((V >> ((t) - 2)) & (V >> ((t) - 7)) & (V >> ((t) - 15)) & (V >> ((t) - 16))) & 1))) acc_ = s8_add(acc_, s8_set1(P.kc[t])); \
        W[t] = acc_; } while (0)
    S8P_W(16); S8P_W(17); S8P_W(18); S8P_W(19); S8P_W(20); S8P_W(21); S8P_W(22); S8P_W(23);
    S8P_W(24); S8P_W(25); S8P_W(26); S8P_W(27); S8P_W(28); S8P_W(29); S8P_W(30); S8P_W(31);
#undef S8P_W
    /* Raw schedule words occupy a 32-vector ring. Expand the next eight
     * words only after the preceding rounds have consumed their slots.
     * K+W is formed at consumption; raw W remains available to the recurrence. */
#define S8R_KW(t) (((t) < 16 && !((V >> (t)) & 1)) ? s8_set1(P.kw[t]) : s8_add(s8_set1(K256[t]), W[(t) & 31]))
#define S8R_EXPAND8(t) do { \
    for (int j_ = (t); j_ < (t) + 8; j_++) \
        W[j_ & 31] = s8_add(s8_add(s8_s1(W[(j_ - 2) & 31]), W[(j_ - 7) & 31]), \
                           s8_add(s8_s0(W[(j_ - 15) & 31]), W[(j_ - 16) & 31])); \
    } while (0)
    /* round 0 from the scalar state: the new a is kept in h, the new e in d (S8_ROUND's naming) */
    v8u a = s8_set1(P.st0[0]), b = s8_set1(P.st0[1]), c = s8_set1(P.st0[2]), d = s8_add(W[0], s8_set1(P.e1c));
    v8u e = s8_set1(P.st0[4]), f = s8_set1(P.st0[5]), g = s8_set1(P.st0[6]), h = s8_add(W[0], s8_set1(P.a1c));
    v8u bc = s8_set1(P.st0[0] ^ P.st0[1]);
    S8_ROUND(h, a, b, c, d, e, f, g, S8R_KW(1), bc); S8_ROUND(g, h, a, b, c, d, e, f, S8R_KW(2), bc);
    S8_ROUND(f, g, h, a, b, c, d, e, S8R_KW(3), bc); S8_ROUND(e, f, g, h, a, b, c, d, S8R_KW(4), bc);
    S8_ROUND(d, e, f, g, h, a, b, c, S8R_KW(5), bc); S8_ROUND(c, d, e, f, g, h, a, b, S8R_KW(6), bc);
    S8_ROUND(b, c, d, e, f, g, h, a, S8R_KW(7), bc);
    for (int t = 8; t < 64; t += 8) {
        if (t >= 32) S8R_EXPAND8(t);
        S8_ROUND(a, b, c, d, e, f, g, h, S8R_KW(t + 0), bc); S8_ROUND(h, a, b, c, d, e, f, g, S8R_KW(t + 1), bc);
        S8_ROUND(g, h, a, b, c, d, e, f, S8R_KW(t + 2), bc); S8_ROUND(f, g, h, a, b, c, d, e, S8R_KW(t + 3), bc);
        S8_ROUND(e, f, g, h, a, b, c, d, S8R_KW(t + 4), bc); S8_ROUND(d, e, f, g, h, a, b, c, S8R_KW(t + 5), bc);
        S8_ROUND(c, d, e, f, g, h, a, b, S8R_KW(t + 6), bc); S8_ROUND(b, c, d, e, f, g, h, a, S8R_KW(t + 7), bc);
    }
    out[0] = s8_add(s8_set1(P.st0[0]), a);
    if (!H0ONLY) {
        out[1] = s8_add(s8_set1(P.st0[1]), b); out[2] = s8_add(s8_set1(P.st0[2]), c); out[3] = s8_add(s8_set1(P.st0[3]), d);
        out[4] = s8_add(s8_set1(P.st0[4]), e); out[5] = s8_add(s8_set1(P.st0[5]), f); out[6] = s8_add(s8_set1(P.st0[6]), g);
        out[7] = s8_add(s8_set1(P.st0[7]), h);
    }
#undef S8R_EXPAND8
#undef S8R_KW
}

/* generic block from a scalar state (out of line: the fallback of the structured tail block) */
static QSB_SHA_AVX2F void s8_compress_mid(v8u out[8], const v8u W[16], const uint32_t st0[8]) {
    for (int k = 0; k < 8; k++) out[k] = _mm256_set1_epi32((int)st0[k]);
    s8_compress_full(out, W);
}

/* byte-swap of each 32-bit lane */
static QSB_SHA_AVX2 v8u s8_bswap(v8u x) {
    const v8u m = _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
                                   3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    return _mm256_shuffle_epi8(x, m);
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
static QSB_SHA_NI void shani_compress2_vec(uint32_t *stA, uint32_t *stB,
    __m128i MA0, __m128i MA1, __m128i MA2, __m128i MA3,
    __m128i MB0, __m128i MB1, __m128i MB2, __m128i MB3) {
    __m128i A0, A1, B0, B1;
    shani_load_state(stA, A0, A1); shani_load_state(stB, B0, B1);
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
    shani_store_state(stA, A0, A1); shani_store_state(stB, B0, B1);
}
/* Generic callers keep their original message-word interface and exact body. */
static QSB_SHA_NI void shani_compress2(uint32_t *stA, const uint32_t *wA, uint32_t *stB, const uint32_t *wB) {
    shani_compress2_vec(stA, stB,
        _mm_loadu_si128((const __m128i *)(wA+0)), _mm_loadu_si128((const __m128i *)(wA+4)),
        _mm_loadu_si128((const __m128i *)(wA+8)), _mm_loadu_si128((const __m128i *)(wA+12)),
        _mm_loadu_si128((const __m128i *)(wB+0)), _mm_loadu_si128((const __m128i *)(wB+4)),
        _mm_loadu_si128((const __m128i *)(wB+8)), _mm_loadu_si128((const __m128i *)(wB+12)));
}
/* single compression (tests only) */
static QSB_SHA_NI void shani_compress1(uint32_t *st, const uint32_t *w) {
    uint32_t st2[8]; memcpy(st2, st, 32);
    shani_compress2(st, w, st2, w);
}

} /* namespace qcg_sha */
#endif
