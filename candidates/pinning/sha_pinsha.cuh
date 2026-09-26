/* SPDX-License-Identifier: GPL-3.0-only
 * Constant-folded SHA-256 transforms for the pinning candidate (QSB_SHA_OPT).
 * Uses the VanitySearch S0/S1/s0/s1/Ch/Maj macros of GPUHash.h (notices and
 * GPLv3 terms there and in COPYING). Every function here is an exact algebraic
 * rearrangement of _SHA256Transform on the same padded block:
 *  - the round constants are compile-time literals (qsb_klit), so K+W and the
 *    IV-derived round-0/1 terms fold into single immediates;
 *  - round 1 from a constant (a,b) pair: Maj(A1,a,b) = (A1 & (a^b)) + (a&b)
 *    (the two masks are disjoint, so OR = +); a&b rides in t1 and is removed
 *    again from the d word (c' = c - (a&b));
 *  - round 63: the feed-forward word H0 (and H4) is folded into t1's constant,
 *    so out[0] = t1' + S0 + Maj and out[4] = e + t1' + (H4 - H0);
 *  - _SHA256Pubkey33H0 returns only word 0 of the digest (the ranked gate reads
 *    only the top QSB_ZEROS_N <= 32 bits); the other seven words are dead.
 */
#ifndef QSB_SHA_PINSHA_CUH
#define QSB_SHA_PINSHA_CUH

__host__ __device__ __forceinline__ constexpr uint32_t qsb_klit(int i)
{
    constexpr uint32_t k[64] = {
        0x428A2F98u, 0x71374491u, 0xB5C0FBCFu, 0xE9B5DBA5u, 0x3956C25Bu, 0x59F111F1u, 0x923F82A4u, 0xAB1C5ED5u,
        0xD807AA98u, 0x12835B01u, 0x243185BEu, 0x550C7DC3u, 0x72BE5D74u, 0x80DEB1FEu, 0x9BDC06A7u, 0xC19BF174u,
        0xE49B69C1u, 0xEFBE4786u, 0x0FC19DC6u, 0x240CA1CCu, 0x2DE92C6Fu, 0x4A7484AAu, 0x5CB0A9DCu, 0x76F988DAu,
        0x983E5152u, 0xA831C66Du, 0xB00327C8u, 0xBF597FC7u, 0xC6E00BF3u, 0xD5A79147u, 0x06CA6351u, 0x14292967u,
        0x27B70A85u, 0x2E1B2138u, 0x4D2C6DFCu, 0x53380D13u, 0x650A7354u, 0x766A0ABBu, 0x81C2C92Eu, 0x92722C85u,
        0xA2BFE8A1u, 0xA81A664Bu, 0xC24B8B70u, 0xC76C51A3u, 0xD192E819u, 0xD6990624u, 0xF40E3585u, 0x106AA070u,
        0x19A4C116u, 0x1E376C08u, 0x2748774Cu, 0x34B0BCB5u, 0x391C0CB3u, 0x4ED8AA4Au, 0x5B9CCA4Fu, 0x682E6FF3u,
        0x748F82EEu, 0x78A5636Fu, 0x84C87814u, 0x8CC70208u, 0x90BEFFFAu, 0xA4506CEBu, 0xBEF9A3F7u, 0xC67178F2u};
    return k[i];
}

#define QSB_IV0 0x6a09e667u
#define QSB_IV1 0xbb67ae85u
#define QSB_IV2 0x3c6ef372u
#define QSB_IV3 0xa54ff53au
#define QSB_IV4 0x510e527fu
#define QSB_IV5 0x9b05688cu
#define QSB_IV6 0x1f83d9abu
#define QSB_IV7 0x5be0cd19u

/* QSB_SHA_FMA_ADD: pipe-balance experiment. Stage 2 is ~80% ALU-pipe instructions, so every
 * two-input add of the pubkey compression is emitted as `mad.lo.u32 d, a, one, b` (one = a value
 * the compiler cannot fold, so ptxas must use IMAD on the FMA-heavy pipe). Exact: a*1 + b = a + b
 * mod 2^32. Costs +3 instructions per round (three-input IADD3 -> two-input IMADs). */
#ifndef QSB_SHA_FMA_ADD
#define QSB_SHA_FMA_ADD 1     /* pubkey-hash adds on the FMA-heavy pipe (stage 2 is ALU-bound) */
#endif
__device__ __constant__ uint32_t pin_one_mul = 1;   /* 1; also re-uploaded by the host */
__device__ __forceinline__ uint32_t qsb_fadd(uint32_t a, uint32_t one, uint32_t b) {
    uint32_t r; asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(one), "r"(b)); return r;
}

#ifndef QSB_SHA_FMA_EARLY
#define QSB_SHA_FMA_EARLY 1 /* Exact FMA-add schedule for early pubkey rounds. */
#endif

/* QSB_SHA_FMA_ROT: the same pipe-balance trick for rotations. x * 2^k as a 64-bit product is
 * {hi, lo} = {x >> (32-k), x << k}; the two halves are disjoint, so ROR(x, 32-k) = lo + hi.
 * One IMAD.WIDE.U32 (c-bank multiplier, so ptxas cannot fold it back into shifts) plus one IMAD
 * add replaces one ALU-pipe SHF. Exact for every x and every 1 <= k <= 31. */
#ifndef QSB_SHA_FMA_ROT
#define QSB_SHA_FMA_ROT 8     /* bit 8 only: schedule-sigma logical shifts as IMAD.HI (no extra
                               * instruction); rotations stay on the ALU pipe: IMAD.WIDE rotations
                               * (bits 1/2/4) measured -0.8..-2.3% */
#endif
__device__ __constant__ uint32_t pin_pow2[32] = {
    1u,2u,4u,8u,16u,32u,64u,128u,256u,512u,1024u,2048u,4096u,8192u,16384u,32768u,
    65536u,131072u,262144u,524288u,1048576u,2097152u,4194304u,8388608u,16777216u,33554432u,
    67108864u,134217728u,268435456u,536870912u,1073741824u,2147483648u};
__device__ __forceinline__ uint32_t qsb_rorf(uint32_t x, int n, uint32_t one) {
    uint64_t t = (uint64_t)x * pin_pow2[32 - n];      /* IMAD.WIDE.U32 */
    return qsb_fadd((uint32_t)t, one, (uint32_t)(t >> 32));
}
/* x >> k = high word of x * 2^(32-k): one IMAD.HI on the FMA-heavy pipe replaces one ALU-pipe
 * SHF with no extra instruction. */
__device__ __forceinline__ uint32_t qsb_shrf(uint32_t x, int k) {
    return __umulhi(x, pin_pow2[32 - k]);
}
#define S1F(x) (qsb_rorf(x,6,one) ^ qsb_rorf(x,11,one) ^ qsb_rorf(x,25,one))
#define S0F(x) (qsb_rorf(x,2,one) ^ qsb_rorf(x,13,one) ^ qsb_rorf(x,22,one))
/* schedule sigmas: bit 4 moves the two rotations, bit 8 the logical shift */
#if !QSB_SHA_FMA_ADD
#define QSB_s0M(x) s0(x)
#define QSB_s1M(x) s1(x)
#elif (QSB_SHA_FMA_ROT & 4) && (QSB_SHA_FMA_ROT & 8)
#define QSB_s0M(x) (qsb_rorf(x,7,one) ^ qsb_rorf(x,18,one) ^ qsb_shrf(x,3))
#define QSB_s1M(x) (qsb_rorf(x,17,one) ^ qsb_rorf(x,19,one) ^ qsb_shrf(x,10))
#elif (QSB_SHA_FMA_ROT & 4)
#define QSB_s0M(x) (qsb_rorf(x,7,one) ^ qsb_rorf(x,18,one) ^ ((x) >> 3))
#define QSB_s1M(x) (qsb_rorf(x,17,one) ^ qsb_rorf(x,19,one) ^ ((x) >> 10))
#elif (QSB_SHA_FMA_ROT & 8)
#define QSB_s0M(x) (ROR(x,7) ^ ROR(x,18) ^ qsb_shrf(x,3))
#define QSB_s1M(x) (ROR(x,17) ^ ROR(x,19) ^ qsb_shrf(x,10))
#else
#define QSB_s0M(x) s0(x)
#define QSB_s1M(x) s1(x)
#endif

/* QSB_SHA_ALU_ADD: the stage-0 twin of QSB_SHA_FMA_ADD, in the other direction. Stage 0 runs
 * beside the IMAD-bound chain loop, so ptxas' habit of putting the two-input SHA adds on the
 * FMA-heavy pipe (IMAD.IADD) spends the scarce pipe there. Adding a constant-bank zero makes them
 * three-input adds, which only IADD3 (ALU pipe) can do - same instruction count, exact. */
#ifndef QSB_SHA_ALU_ADD
#define QSB_SHA_ALU_ADD 1     /* route exact SHA additions to the ALU pipe */
#endif
#if QSB_SHA_ALU_ADD
__device__ __constant__ uint32_t pin_zero_add = 0;   /* 0; also re-uploaded by the host */
#define QSB_Z (pin_zero_add)
#else
#define QSB_Z 0u
#endif
#if defined(QSB_CHAIN_ALU) && QSB_CHAIN_ALU && !QSB_SHA_ALU_ADD
#error "QSB_CHAIN_ALU reads pin_zero_add, which exists only with QSB_SHA_ALU_ADD=1"
#endif

/* One round; kw = K_i + W_i (a literal when W_i is constant). */
#define QSB_RL(a, b, c, d, e, f, g, h, kw) \
    t1 = h + S1(e) + Ch(e,f,g) + (kw); \
    t2 = S0(a) + Maj(a,b,c); \
    d += t1 + QSB_Z; \
    h = t1 + t2;

#define QSB_RND15L(k) {\
QSB_RL(a, b, c, d, e, f, g, h, qsb_klit(k) + w[0]);\
QSB_RL(h, a, b, c, d, e, f, g, qsb_klit(k + 1) + w[1]);\
QSB_RL(g, h, a, b, c, d, e, f, qsb_klit(k + 2) + w[2]);\
QSB_RL(f, g, h, a, b, c, d, e, qsb_klit(k + 3) + w[3]);\
QSB_RL(e, f, g, h, a, b, c, d, qsb_klit(k + 4) + w[4]);\
QSB_RL(d, e, f, g, h, a, b, c, qsb_klit(k + 5) + w[5]);\
QSB_RL(c, d, e, f, g, h, a, b, qsb_klit(k + 6) + w[6]);\
QSB_RL(b, c, d, e, f, g, h, a, qsb_klit(k + 7) + w[7]);\
QSB_RL(a, b, c, d, e, f, g, h, qsb_klit(k + 8) + w[8]);\
QSB_RL(h, a, b, c, d, e, f, g, qsb_klit(k + 9) + w[9]);\
QSB_RL(g, h, a, b, c, d, e, f, qsb_klit(k + 10) + w[10]);\
QSB_RL(f, g, h, a, b, c, d, e, qsb_klit(k + 11) + w[11]);\
QSB_RL(e, f, g, h, a, b, c, d, qsb_klit(k + 12) + w[12]);\
QSB_RL(d, e, f, g, h, a, b, c, qsb_klit(k + 13) + w[13]);\
QSB_RL(c, d, e, f, g, h, a, b, qsb_klit(k + 14) + w[14]);\
}
#define QSB_RND16L(k) {\
QSB_RND15L(k);\
QSB_RL(b, c, d, e, f, g, h, a, qsb_klit(k + 15) + w[15]);\
}

/* WMIX with a constant-bank zero addend (QSB_SHA_ALU_ADD): keeps every schedule add on the
 * ALU pipe, same instruction count. */
#ifndef QSB_DIGEST_WMIX_Z
#define QSB_DIGEST_WMIX_Z 0
#endif
#if QSB_DIGEST_WMIX_Z && !QSB_SHA_ALU_ADD
#error "QSB_DIGEST_WMIX_Z needs the constant-bank zero from QSB_SHA_ALU_ADD=1"
#endif
#define QSB_WMIX_Z() { \
w[0] += s1(w[14]) + w[9] + s0(w[1]) + QSB_Z;\
w[1] += s1(w[15]) + w[10] + s0(w[2]) + QSB_Z;\
w[2] += s1(w[0]) + w[11] + s0(w[3]) + QSB_Z;\
w[3] += s1(w[1]) + w[12] + s0(w[4]) + QSB_Z;\
w[4] += s1(w[2]) + w[13] + s0(w[5]) + QSB_Z;\
w[5] += s1(w[3]) + w[14] + s0(w[6]) + QSB_Z;\
w[6] += s1(w[4]) + w[15] + s0(w[7]) + QSB_Z;\
w[7] += s1(w[5]) + w[0] + s0(w[8]) + QSB_Z;\
w[8] += s1(w[6]) + w[1] + s0(w[9]) + QSB_Z;\
w[9] += s1(w[7]) + w[2] + s0(w[10]) + QSB_Z;\
w[10] += s1(w[8]) + w[3] + s0(w[11]) + QSB_Z;\
w[11] += s1(w[9]) + w[4] + s0(w[12]) + QSB_Z;\
w[12] += s1(w[10]) + w[5] + s0(w[13]) + QSB_Z;\
w[13] += s1(w[11]) + w[6] + s0(w[14]) + QSB_Z;\
w[14] += s1(w[12]) + w[7] + s0(w[15]) + QSB_Z;\
w[15] += s1(w[13]) + w[8] + s0(w[0]) + QSB_Z;\
}

/* Rolling schedule step with a literal round constant (see sha_schedule_interleaved.cuh). */
#define QSB_STEPL(j, a,b,c,d,e,f,g,h, base) do { \
    w[j] += s1(w[((j)+14)&15]) + w[((j)+9)&15] + s0(w[((j)+1)&15]); \
    QSB_RL(a,b,c,d,e,f,g,h,qsb_klit((base)+(j)) + w[j]); \
} while (0)
#define QSB_INTERLEAVED15L(base) do { \
    QSB_STEPL(0,a,b,c,d,e,f,g,h,base); \
    QSB_STEPL(1,h,a,b,c,d,e,f,g,base); \
    QSB_STEPL(2,g,h,a,b,c,d,e,f,base); \
    QSB_STEPL(3,f,g,h,a,b,c,d,e,base); \
    QSB_STEPL(4,e,f,g,h,a,b,c,d,base); \
    QSB_STEPL(5,d,e,f,g,h,a,b,c,base); \
    QSB_STEPL(6,c,d,e,f,g,h,a,b,base); \
    QSB_STEPL(7,b,c,d,e,f,g,h,a,base); \
    QSB_STEPL(8,a,b,c,d,e,f,g,h,base); \
    QSB_STEPL(9,h,a,b,c,d,e,f,g,base); \
    QSB_STEPL(10,g,h,a,b,c,d,e,f,base); \
    QSB_STEPL(11,f,g,h,a,b,c,d,e,base); \
    QSB_STEPL(12,e,f,g,h,a,b,c,d,base); \
    QSB_STEPL(13,d,e,f,g,h,a,b,c,base); \
    QSB_STEPL(14,c,d,e,f,g,h,a,b,base); \
} while (0)
#define QSB_INTERLEAVED16L(base) do { \
    QSB_INTERLEAVED15L(base); \
    QSB_STEPL(15,b,c,d,e,f,g,h,a,base); \
} while (0)

/* Rounds 0 and 1 from the SHA-256 IV with message words W0, W1 (digest and pubkey
 * transforms). Round 1's Maj has two constant inputs: Maj(A1,IV0,IV1) =
 * (A1 & (IV0^IV1)) + (IV0&IV1); the constant rides in t1 and leaves again
 * through c (= IV2 - (IV0&IV1) + t1). */
#define QSB_IV_ROUNDS01(W0, W1) \
    QSB_RL(a, b, c, d, e, f, g, h, qsb_klit(0) + (W0)); \
    t1 = g + S1(d) + Ch(d,e,f) + (qsb_klit(1) + (QSB_IV0 & QSB_IV1)) + (W1); \
    t2 = S0(h) + (h & (QSB_IV0 ^ QSB_IV1)); \
    c = (QSB_IV2 - (QSB_IV0 & QSB_IV1)) + t1; \
    g = t1 + t2;

/* Round 63 (the 16th round of the last group, argument order b,c,d,e,f,g,h,a)
 * with the feed-forward of words 0 and 4 folded in: F0/F4 are the words added
 * to a/e by the feed-forward, KW = K63 + W63 + F0. */
#define QSB_R63_FF04(KWF0, F4mF0, out0, out4) \
    t1 = a + S1(f) + Ch(f,g,h) + (KWF0); \
    out0 = t1 + S0(b) + Maj(b,c,d); \
    out4 = e + t1 + (F4mF0);

/* SHA256d second compression of a 32-byte message m[0..7] (pad W8=0x80000000,
 * W9..14=0, W15=256) from the IV. Bit-identical to _SHA256TransformDigest32. */
__device__ __forceinline__ void _SHA256TransformDigest32Q(
    uint32_t out[8], const uint32_t m[8])
{
    uint32_t t1;
    uint32_t t2;
    uint32_t a = QSB_IV0, b = QSB_IV1, c = QSB_IV2, d = QSB_IV3;
    uint32_t e = QSB_IV4, f = QSB_IV5, g = QSB_IV6, h = QSB_IV7;

    uint32_t w[16];
#pragma unroll
    for (int i = 0; i < 8; i++) w[i] = m[i];

    QSB_IV_ROUNDS01(w[0], w[1]);
    QSB_RL(g, h, a, b, c, d, e, f, qsb_klit(2) + w[2]);
    QSB_RL(f, g, h, a, b, c, d, e, qsb_klit(3) + w[3]);
    QSB_RL(e, f, g, h, a, b, c, d, qsb_klit(4) + w[4]);
    QSB_RL(d, e, f, g, h, a, b, c, qsb_klit(5) + w[5]);
    QSB_RL(c, d, e, f, g, h, a, b, qsb_klit(6) + w[6]);
    QSB_RL(b, c, d, e, f, g, h, a, qsb_klit(7) + w[7]);
    QSB_RL(a, b, c, d, e, f, g, h, qsb_klit(8) + 0x80000000u);
    QSB_RL(h, a, b, c, d, e, f, g, qsb_klit(9));
    QSB_RL(g, h, a, b, c, d, e, f, qsb_klit(10));
    QSB_RL(f, g, h, a, b, c, d, e, qsb_klit(11));
    QSB_RL(e, f, g, h, a, b, c, d, qsb_klit(12));
    QSB_RL(d, e, f, g, h, a, b, c, qsb_klit(13));
    QSB_RL(c, d, e, f, g, h, a, b, qsb_klit(14));
    QSB_RL(b, c, d, e, f, g, h, a, qsb_klit(15) + 256u);

    {
        w[0] += s0(w[1]);
        w[1] += s1(256u) + s0(w[2]);
        w[2] += s1(w[0]) + s0(w[3]);
        w[3] += s1(w[1]) + s0(w[4]);
        w[4] += s1(w[2]) + s0(w[5]);
        w[5] += s1(w[3]) + s0(w[6]);
        w[6] += s1(w[4]) + 256u + s0(w[7]);
        w[7] += s1(w[5]) + w[0] + s0(0x80000000u);
        w[8]  = 0x80000000u + s1(w[6]) + w[1] + QSB_Z;
        w[9]  = s1(w[7]) + w[2];
        w[10] = s1(w[8]) + w[3];
        w[11] = s1(w[9]) + w[4];
        w[12] = s1(w[10]) + w[5];
        w[13] = s1(w[11]) + w[6];
        w[14] = s1(w[12]) + w[7] + s0(256u);
        w[15] = 256u + s1(w[13]) + w[8] + s0(w[0]);
    }

    QSB_RND16L(16);
#if QSB_DIGEST_WMIX_Z
    /* Outer-digest schedule with the constant-bank zero addend: the four-input sums become
     * five-input, so ptxas emits two IADD3 instead of IADD3 plus a multiply-pipe IMAD.IADD. */
    QSB_WMIX_Z();
#else
    WMIX();
#endif
    QSB_RND16L(32);
#if QSB_DIGEST_WMIX_Z
    QSB_WMIX_Z();
#else
    WMIX();
#endif
    QSB_RND15L(48);
    QSB_R63_FF04(qsb_klit(63) + w[15] + QSB_IV0, QSB_IV4 - QSB_IV0, out[0], out[4]);
    out[1] = QSB_IV1 + b;
    out[2] = QSB_IV2 + c;
    out[3] = QSB_IV3 + d;
    out[5] = QSB_IV5 + f;
    out[6] = QSB_IV6 + g;
    out[7] = QSB_IV7 + h;
}

/* QSB_FIN_KW_IMAD: finish-kernel pipe balance. In the pubkey compression every add already
 * runs on the multiply pipe (qsb_fadd), except the round-constant add K_i + W_i, which ptxas
 * emits as an ALU-pipe IADD3 with an immediate. With the switch on it becomes
 * qsb_fadd(W_i, one, K_i): ptxas keeps `one` in a register (it already does, for the constant
 * adds of round 63) and emits IMAD R, R_w, R_one, K_i on the multiply pipe. Exact: W*1 + K is
 * W + K mod 2^32 for every W, with the same instruction count. Rounds 0 and 1 (IV state) get
 * the same treatment below at two extra instructions per hash. With the switch at 0 every macro
 * expands to the original expression. */
#ifndef QSB_FIN_KW_IMAD
#define QSB_FIN_KW_IMAD 1
#endif
#if QSB_FIN_KW_IMAD
#define QSB_KWF(K, X) qsb_fadd((X), one, (K))
/* Rounds 0 and 1 of the pubkey hash (QSB_IV_ROUNDS01 with a..h = IV) written as two-input
 * multiply-pipe adds: every constant term is folded at compile time and joined to the one
 * live addend it meets, so round 0 is d = W0 + C and h = W0 + C', and round 1 adds W1 + C,
 * S1(d), Ch(d,IV4,IV5), then forks into c (+ IV2 - (IV0&IV1)) and g (+ S0(h), + h&(IV0^IV1)).
 * The same terms as QSB_IV_ROUNDS01, summed mod 2^32 in another order. */
#define QSB_IV_ROUNDS01_F(W0, W1) { \
    const uint32_t r0t1 = QSB_IV7 + S1(QSB_IV4) + Ch(QSB_IV4, QSB_IV5, QSB_IV6) + qsb_klit(0); \
    const uint32_t r0t2 = S0(QSB_IV0) + Maj(QSB_IV0, QSB_IV1, QSB_IV2); \
    d = qsb_fadd((W0), one, QSB_IV3 + r0t1); \
    h = qsb_fadd((W0), one, r0t1 + r0t2); \
    t1 = qsb_fadd((W1), one, QSB_IV6 + qsb_klit(1) + (QSB_IV0 & QSB_IV1)); \
    t1 = qsb_fadd(t1, one, S1(d)); \
    t1 = qsb_fadd(t1, one, Ch(d, QSB_IV4, QSB_IV5)); \
    c = qsb_fadd(t1, one, QSB_IV2 - (QSB_IV0 & QSB_IV1)); \
    t2 = qsb_fadd(t1, one, S0(h)); \
    g = qsb_fadd(t2, one, h & (QSB_IV0 ^ QSB_IV1)); }
#else
#define QSB_KWF(K, X) K + X
#define QSB_IV_ROUNDS01_F(W0, W1) QSB_IV_ROUNDS01(W0, W1)
#endif

#if QSB_SHA_FMA_ADD
#if QSB_SHA_FMA_ROT & 1
#define QSB_S1M(x) S1F(x)
#else
#define QSB_S1M(x) S1(x)
#endif
#if QSB_SHA_FMA_ROT & 2
#define QSB_S0M(x) S0F(x)
#else
#define QSB_S0M(x) S0(x)
#endif
#define QSB_RL_F(a, b, c, d, e, f, g, h, kw) \
    t1 = qsb_fadd(h, one, (kw)); \
    t1 = qsb_fadd(t1, one, QSB_S1M(e)); \
    t1 = qsb_fadd(t1, one, Ch(e,f,g)); \
    d  = qsb_fadd(d, one, t1); \
    t2 = qsb_fadd(t1, one, QSB_S0M(a)); \
    h  = qsb_fadd(t2, one, Maj(a,b,c));
#define QSB_RND15L_F(k) {\
QSB_RL_F(a, b, c, d, e, f, g, h, QSB_KWF(qsb_klit(k), w[0]));\
QSB_RL_F(h, a, b, c, d, e, f, g, QSB_KWF(qsb_klit(k + 1), w[1]));\
QSB_RL_F(g, h, a, b, c, d, e, f, QSB_KWF(qsb_klit(k + 2), w[2]));\
QSB_RL_F(f, g, h, a, b, c, d, e, QSB_KWF(qsb_klit(k + 3), w[3]));\
QSB_RL_F(e, f, g, h, a, b, c, d, QSB_KWF(qsb_klit(k + 4), w[4]));\
QSB_RL_F(d, e, f, g, h, a, b, c, QSB_KWF(qsb_klit(k + 5), w[5]));\
QSB_RL_F(c, d, e, f, g, h, a, b, QSB_KWF(qsb_klit(k + 6), w[6]));\
QSB_RL_F(b, c, d, e, f, g, h, a, QSB_KWF(qsb_klit(k + 7), w[7]));\
QSB_RL_F(a, b, c, d, e, f, g, h, QSB_KWF(qsb_klit(k + 8), w[8]));\
QSB_RL_F(h, a, b, c, d, e, f, g, QSB_KWF(qsb_klit(k + 9), w[9]));\
QSB_RL_F(g, h, a, b, c, d, e, f, QSB_KWF(qsb_klit(k + 10), w[10]));\
QSB_RL_F(f, g, h, a, b, c, d, e, QSB_KWF(qsb_klit(k + 11), w[11]));\
QSB_RL_F(e, f, g, h, a, b, c, d, QSB_KWF(qsb_klit(k + 12), w[12]));\
QSB_RL_F(d, e, f, g, h, a, b, c, QSB_KWF(qsb_klit(k + 13), w[13]));\
QSB_RL_F(c, d, e, f, g, h, a, b, QSB_KWF(qsb_klit(k + 14), w[14]));\
}
#define QSB_RND16L_F(k) {\
QSB_RND15L_F(k);\
QSB_RL_F(b, c, d, e, f, g, h, a, QSB_KWF(qsb_klit(k + 15), w[15]));\
}
#define QSB_STEPL_F(j, a,b,c,d,e,f,g,h, base) do { \
    w[j] = qsb_fadd(w[j], one, QSB_s1M(w[((j)+14)&15])); \
    w[j] = qsb_fadd(w[j], one, w[((j)+9)&15]); \
    w[j] = qsb_fadd(w[j], one, QSB_s0M(w[((j)+1)&15])); \
    QSB_RL_F(a,b,c,d,e,f,g,h,QSB_KWF(qsb_klit((base)+(j)), w[j])); \
} while (0)
#define QSB_INTERLEAVED15L_F(base) do { \
    QSB_STEPL_F(0,a,b,c,d,e,f,g,h,base); \
    QSB_STEPL_F(1,h,a,b,c,d,e,f,g,base); \
    QSB_STEPL_F(2,g,h,a,b,c,d,e,f,base); \
    QSB_STEPL_F(3,f,g,h,a,b,c,d,e,base); \
    QSB_STEPL_F(4,e,f,g,h,a,b,c,d,base); \
    QSB_STEPL_F(5,d,e,f,g,h,a,b,c,base); \
    QSB_STEPL_F(6,c,d,e,f,g,h,a,b,base); \
    QSB_STEPL_F(7,b,c,d,e,f,g,h,a,base); \
    QSB_STEPL_F(8,a,b,c,d,e,f,g,h,base); \
    QSB_STEPL_F(9,h,a,b,c,d,e,f,g,base); \
    QSB_STEPL_F(10,g,h,a,b,c,d,e,f,base); \
    QSB_STEPL_F(11,f,g,h,a,b,c,d,e,base); \
    QSB_STEPL_F(12,e,f,g,h,a,b,c,d,base); \
    QSB_STEPL_F(13,d,e,f,g,h,a,b,c,base); \
    QSB_STEPL_F(14,c,d,e,f,g,h,a,b,base); \
} while (0)
#define QSB_INTERLEAVED16L_F(base) do { \
    QSB_INTERLEAVED15L_F(base); \
    QSB_STEPL_F(15,b,c,d,e,f,g,h,a,base); \
} while (0)
#endif

/* Word 0 of SHA-256(33-byte compressed pubkey): live words m[0..8], W9..14=0,
 * W15=0x108, from the IV. Equal to out[0] of _SHA256TransformPubkey33. */
#if QSB_SHA_ALU_ADD
#pragma push_macro("QSB_Z")
#undef QSB_Z
#define QSB_Z 0u
#endif
__device__ __forceinline__ uint32_t _SHA256Pubkey33H0(const uint32_t m[9])
{
    uint32_t t1;
    uint32_t t2;
    uint32_t a = QSB_IV0, b = QSB_IV1, c = QSB_IV2, d = QSB_IV3;
    uint32_t e = QSB_IV4, f = QSB_IV5, g = QSB_IV6, h = QSB_IV7;

    uint32_t w[16];
#pragma unroll
    for (int i = 0; i < 9; i++) w[i] = m[i];

#if QSB_SHA_FMA_ADD && QSB_SHA_FMA_EARLY
    {
        const uint32_t one = pin_one_mul;
        QSB_IV_ROUNDS01_F(w[0], w[1]);
        QSB_RL_F(g, h, a, b, c, d, e, f, QSB_KWF(qsb_klit(2), w[2]));
        QSB_RL_F(f, g, h, a, b, c, d, e, QSB_KWF(qsb_klit(3), w[3]));
        QSB_RL_F(e, f, g, h, a, b, c, d, QSB_KWF(qsb_klit(4), w[4]));
        QSB_RL_F(d, e, f, g, h, a, b, c, QSB_KWF(qsb_klit(5), w[5]));
        QSB_RL_F(c, d, e, f, g, h, a, b, QSB_KWF(qsb_klit(6), w[6]));
        QSB_RL_F(b, c, d, e, f, g, h, a, QSB_KWF(qsb_klit(7), w[7]));
        QSB_RL_F(a, b, c, d, e, f, g, h, QSB_KWF(qsb_klit(8), w[8]));
        QSB_RL_F(h, a, b, c, d, e, f, g, qsb_klit(9));
        QSB_RL_F(g, h, a, b, c, d, e, f, qsb_klit(10));
        QSB_RL_F(f, g, h, a, b, c, d, e, qsb_klit(11));
        QSB_RL_F(e, f, g, h, a, b, c, d, qsb_klit(12));
        QSB_RL_F(d, e, f, g, h, a, b, c, qsb_klit(13));
        QSB_RL_F(c, d, e, f, g, h, a, b, qsb_klit(14));
        QSB_RL_F(b, c, d, e, f, g, h, a, qsb_klit(15) + 0x108u);
    }
    {
        const uint32_t one = pin_one_mul;
        w[0] = qsb_fadd(w[0], one, QSB_s0M(w[1]));
        w[1] = qsb_fadd(qsb_fadd(w[1], one, s1(0x108u)), one, QSB_s0M(w[2]));
        w[2] = qsb_fadd(qsb_fadd(w[2], one, QSB_s1M(w[0])), one, QSB_s0M(w[3]));
        w[3] = qsb_fadd(qsb_fadd(w[3], one, QSB_s1M(w[1])), one, QSB_s0M(w[4]));
        w[4] = qsb_fadd(qsb_fadd(w[4], one, QSB_s1M(w[2])), one, QSB_s0M(w[5]));
        w[5] = qsb_fadd(qsb_fadd(w[5], one, QSB_s1M(w[3])), one, QSB_s0M(w[6]));
        w[6] = qsb_fadd(qsb_fadd(qsb_fadd(w[6], one, QSB_s1M(w[4])), one, 0x108u), one, QSB_s0M(w[7]));
        w[7] = qsb_fadd(qsb_fadd(qsb_fadd(w[7], one, QSB_s1M(w[5])), one, w[0]), one, QSB_s0M(w[8]));
        w[8] = qsb_fadd(qsb_fadd(w[8], one, QSB_s1M(w[6])), one, w[1]);
        w[9] = qsb_fadd(QSB_s1M(w[7]), one, w[2]);
        w[10] = qsb_fadd(QSB_s1M(w[8]), one, w[3]);
        w[11] = qsb_fadd(QSB_s1M(w[9]), one, w[4]);
        w[12] = qsb_fadd(QSB_s1M(w[10]), one, w[5]);
        w[13] = qsb_fadd(QSB_s1M(w[11]), one, w[6]);
        w[14] = qsb_fadd(qsb_fadd(QSB_s1M(w[12]), one, w[7]), one, s0(0x108u));
        w[15] = qsb_fadd(qsb_fadd(qsb_fadd(0x108u, one, QSB_s1M(w[13])), one, w[8]), one, QSB_s0M(w[0]));
    }
#else
    QSB_IV_ROUNDS01(w[0], w[1]);
    QSB_RL(g, h, a, b, c, d, e, f, qsb_klit(2) + w[2]);
    QSB_RL(f, g, h, a, b, c, d, e, qsb_klit(3) + w[3]);
    QSB_RL(e, f, g, h, a, b, c, d, qsb_klit(4) + w[4]);
    QSB_RL(d, e, f, g, h, a, b, c, qsb_klit(5) + w[5]);
    QSB_RL(c, d, e, f, g, h, a, b, qsb_klit(6) + w[6]);
    QSB_RL(b, c, d, e, f, g, h, a, qsb_klit(7) + w[7]);
    QSB_RL(a, b, c, d, e, f, g, h, qsb_klit(8) + w[8]);
    QSB_RL(h, a, b, c, d, e, f, g, qsb_klit(9));
    QSB_RL(g, h, a, b, c, d, e, f, qsb_klit(10));
    QSB_RL(f, g, h, a, b, c, d, e, qsb_klit(11));
    QSB_RL(e, f, g, h, a, b, c, d, qsb_klit(12));
    QSB_RL(d, e, f, g, h, a, b, c, qsb_klit(13));
    QSB_RL(c, d, e, f, g, h, a, b, qsb_klit(14));
    QSB_RL(b, c, d, e, f, g, h, a, qsb_klit(15) + 0x108u);

    {
#if QSB_SHA_FMA_ADD
        const uint32_t one = pin_one_mul;
#endif
        w[0] += QSB_s0M(w[1]);
        w[1] += s1(0x108u) + QSB_s0M(w[2]);
        w[2] += QSB_s1M(w[0]) + QSB_s0M(w[3]);
        w[3] += QSB_s1M(w[1]) + QSB_s0M(w[4]);
        w[4] += QSB_s1M(w[2]) + QSB_s0M(w[5]);
        w[5] += QSB_s1M(w[3]) + QSB_s0M(w[6]);
        w[6] += QSB_s1M(w[4]) + 0x108u + QSB_s0M(w[7]);
        w[7] += QSB_s1M(w[5]) + w[0] + QSB_s0M(w[8]);
        w[8] += QSB_s1M(w[6]) + w[1];
        w[9]  = QSB_s1M(w[7]) + w[2];
        w[10] = QSB_s1M(w[8]) + w[3];
        w[11] = QSB_s1M(w[9]) + w[4];
        w[12] = QSB_s1M(w[10]) + w[5];
        w[13] = QSB_s1M(w[11]) + w[6];
        w[14] = QSB_s1M(w[12]) + w[7] + s0(0x108u);
        w[15] = 0x108u + QSB_s1M(w[13]) + w[8] + QSB_s0M(w[0]);
    }

#endif
#if QSB_SHA_FMA_ADD
    {
        const uint32_t one = pin_one_mul;
        QSB_RND16L_F(16);
        QSB_INTERLEAVED16L_F(32);
        QSB_INTERLEAVED15L_F(48);
        w[15] = qsb_fadd(w[15], one, s1(w[13]));
        w[15] = qsb_fadd(w[15], one, w[8]);
        w[15] = qsb_fadd(w[15], one, s0(w[0]));
        uint32_t r = qsb_fadd(a, one, w[15]);
        r = qsb_fadd(r, one, qsb_klit(63) + QSB_IV0);
        r = qsb_fadd(r, one, QSB_S1M(f));
        r = qsb_fadd(r, one, Ch(f,g,h));
        r = qsb_fadd(r, one, QSB_S0M(b));
        r = qsb_fadd(r, one, Maj(b,c,d));
        return r;
    }
#else
    QSB_RND16L(16);
    QSB_INTERLEAVED16L(32);
    QSB_INTERLEAVED15L(48);
    w[15] += s1(w[13]) + w[8] + s0(w[0]);
    /* round 63, a-output only, with H0 = IV0 + a folded into the constant */
    return a + S1(f) + Ch(f,g,h) + w[15] + (qsb_klit(63) + QSB_IV0) + S0(b) + Maj(b,c,d);
#endif
}
#if QSB_SHA_ALU_ADD
#pragma pop_macro("QSB_Z")
#endif

/* Ranked gate on digest word 0 (QSB_ZEROS_N <= 32). */
__device__ __forceinline__ int gpu_bench_valid_h0(uint32_t h0) {
#if QSB_ZEROS_N >= 32
    return h0 == 0u;
#else
    return (h0 >> (32 - QSB_ZEROS_N)) == 0u;
#endif
}

#endif
