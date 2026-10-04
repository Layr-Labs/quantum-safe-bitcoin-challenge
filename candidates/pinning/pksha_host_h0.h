// SPDX-License-Identifier: GPL-3.0-only
#pragma once
/* Fixed33-byte compressed-pubkey H0 only. Derived from the reviewed CPU29
 * structured AVX2 SHA schedule; shared co-grind headers are unchanged.
 * Active input words0..8 vary; words9..14=0,word15=264. Inactive caller lanes
 * are masked before hit publication and their hash values are immaterial.
 * All64 SHA rounds contribute to A64; round0 is exactly folded from the IV. */
#include "cg_sha.h"
namespace qsb_pksha_h0 {
using namespace qcg_sha;
static constexpr uint32_t H0_K[64]={
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
static constexpr uint32_t H0_IV[8]={0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
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
    for (int t = 0; t < 64; t++) p.kw[t] = H0_K[t] + w[t];
    for (int i = 0; i < 8; i++) p.st0[i] = st0[i];
    const uint32_t a = st0[0], b = st0[1], c = st0[2], d = st0[3], e = st0[4], f = st0[5], g = st0[6], h = st0[7];
    const uint32_t c1 = h + c_S1(e) + ((e & f) ^ (~e & g)) + H0_K[0];
    const uint32_t c2 = c_S0(a) + ((a & b) ^ (a & c) ^ (b & c));
    p.a1c = c1 + c2; p.e1c = d + c1;
    return p;
}
static constexpr uint32_t S8_W_PUBKEY[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 264};
static constexpr s8_plan S8_PLAN_PUBKEY = s8_make_plan(S8_W_PUBKEY, 0x1FFu, H0_IV);     /* SHA256 of a 33-byte message */

/* varying words Wv[j] (j with bit j of VM set; word 0 must vary); out: 8 state words, or only out[0]
 * (= H0) with H0ONLY. Compact form: the message schedule and the per-round K + W go to a stack array
 * first, then the 63 remaining rounds run as a rolled 8-round loop (small code: the co-grinder runs
 * two or three of these back to back, and the unrolled forms overflow the decoded-op cache). */
template <uint32_t VM>
static QSB_SHA_AVX2 void s8_compress_plan(v8u out[8], const v8u *Wv, const s8_plan &P) {
    static_assert(VM & 1, "round 0 folding needs a varying word 0");
    constexpr uint64_t V = s8_varmask(VM);
    static_assert((V >> 16) == (~0ull >> 16), "every schedule word must depend on the message");
    v8u W[64];
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
    for (int t = 32; t < 64; t++) W[t] = s8_add(s8_add(s8_s1(W[t - 2]), W[t - 7]), s8_add(s8_s0(W[t - 15]), W[t - 16]));
    /* Schedule is complete: reuse W[1..63] for K+W; keep raw W[0] for folded round0. */
#define S8P_KW(t) W[t] = ((V >> (t)) & 1) ? s8_add(s8_set1(H0_K[t]), W[t]) : s8_set1(P.kw[t]);
    S8P_KW(1) S8P_KW(2) S8P_KW(3) S8P_KW(4) S8P_KW(5) S8P_KW(6) S8P_KW(7) S8P_KW(8)
    S8P_KW(9) S8P_KW(10) S8P_KW(11) S8P_KW(12) S8P_KW(13) S8P_KW(14) S8P_KW(15)
#undef S8P_KW
    for (int t = 16; t < 64; t++) W[t] = s8_add(s8_set1(H0_K[t]), W[t]);
    /* round 0 from the scalar state: the new a is kept in h, the new e in d (S8_ROUND's naming) */
    v8u a = s8_set1(P.st0[0]), b = s8_set1(P.st0[1]), c = s8_set1(P.st0[2]), d = s8_add(W[0], s8_set1(P.e1c));
    v8u e = s8_set1(P.st0[4]), f = s8_set1(P.st0[5]), g = s8_set1(P.st0[6]), h = s8_add(W[0], s8_set1(P.a1c));
    v8u bc = s8_set1(P.st0[0] ^ P.st0[1]);
    S8_ROUND(h, a, b, c, d, e, f, g, W[1], bc); S8_ROUND(g, h, a, b, c, d, e, f, W[2], bc);
    S8_ROUND(f, g, h, a, b, c, d, e, W[3], bc); S8_ROUND(e, f, g, h, a, b, c, d, W[4], bc);
    S8_ROUND(d, e, f, g, h, a, b, c, W[5], bc); S8_ROUND(c, d, e, f, g, h, a, b, W[6], bc);
    S8_ROUND(b, c, d, e, f, g, h, a, W[7], bc);
    for (int t = 8; t < 56; t += 8) {
        S8_ROUND(a, b, c, d, e, f, g, h, W[t + 0], bc); S8_ROUND(h, a, b, c, d, e, f, g, W[t + 1], bc);
        S8_ROUND(g, h, a, b, c, d, e, f, W[t + 2], bc); S8_ROUND(f, g, h, a, b, c, d, e, W[t + 3], bc);
        S8_ROUND(e, f, g, h, a, b, c, d, W[t + 4], bc); S8_ROUND(d, e, f, g, h, a, b, c, W[t + 5], bc);
        S8_ROUND(c, d, e, f, g, h, a, b, W[t + 6], bc); S8_ROUND(b, c, d, e, f, g, h, a, W[t + 7], bc);
    }
    S8_ROUND(a, b, c, d, e, f, g, h, W[56], bc); S8_ROUND(h, a, b, c, d, e, f, g, W[57], bc);
    S8_ROUND(g, h, a, b, c, d, e, f, W[58], bc); S8_ROUND(f, g, h, a, b, c, d, e, W[59], bc);
    S8_ROUND(e, f, g, h, a, b, c, d, W[60], bc); S8_ROUND(d, e, f, g, h, a, b, c, W[61], bc);
    S8_ROUND(c, d, e, f, g, h, a, b, W[62], bc);
    /* Round63 still computes its exact new A. Its new E and other digest words
     * are unused by the <=32-bit H0 gate, so omit only that dead assignment. */
    const v8u last_t1=s8_add(s8_add(a,s8_S1(f)),s8_add(s8_xor(h,s8_and(f,s8_xor(g,h))),W[63]));
    const v8u last_t2=s8_add(s8_S0(b),s8_xor(s8_and(s8_xor(b,c),bc),c));
    a=s8_add(last_t1,last_t2);
    out[0] = s8_add(s8_set1(P.st0[0]), a);
}


static QSB_SHA_AVX2 v8u pubkey_h0(const uint32_t words[16][8]) {
    v8u w[9],out[1];
    for(int k=0;k<9;k++)w[k]=_mm256_loadu_si256((const __m256i*)words[k]);
    s8_compress_plan<0x1FFu>(out,w,S8_PLAN_PUBKEY);
    return out[0];
}
} // namespace qsb_pksha_h0
