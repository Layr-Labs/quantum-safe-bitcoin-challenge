/* host_producers.h -- QSB_HOST_PRODUCERS: exact host twins of the per-batch producer kernels.
 *
 * kernel_epoch_groups + kernel_build_epochs_inc + kernel_build_first_flat build, per batch of
 * 2^20 epochs, one 64-byte epoch_desc_t (SHA-256 state after the 1352-byte epoch prefix, the
 * trailing 8 bytes as two big-endian words, the six early omissions) and, per epoch, one
 * first-window-block state per first-block class (8 classes x 32 B). Nothing in them depends on
 * GPU results, so up to 3 host threads can build them ahead of time into pinned buffers; the
 * main loop then only enqueues two H2D copies (+ the tentative-count reset the producer used to
 * do) on the slot stream in front of kernel_digest. The GPU does no producer work at all.
 *
 * Host algorithm (same bytes as the kernels, less hashing): epochs are walked in lex order of
 * (o1..o6) with one SHA stream context per omission level, ctx[k] = stream over pushes
 * [0, o_k) minus o_1..o_{k-1}. The lex successor appends one push to one context (and copies
 * contexts on a carry), so each epoch hashes only its suffix P[o6+1..cut-1] (~3.6 blocks on
 * average vs ~7 for kernel_build_epochs_inc) plus its 8 first-block states. Compressions use the
 * x86 SHA extensions 4 lanes at a time (qsha_x4 from CpuGrindSubset.h, same code), else OpenSSL.
 *
 * Safety: batch 0 is always built by the GPU producers AND by the host; the GPU copy is read
 * back and every descriptor (mid, remW, early) and every used first-state word is compared.
 * Host batches are used only after that start-up self-check passes; any mismatch switches the
 * host producers off for the run. Per batch, if the host batch is not ready within
 * QSB_HP_WAIT_MS the GPU producers run for that batch (the host copy is discarded); after
 * QSB_HP_MAX_FALLBACKS consecutive fallbacks the host producers are switched off (watchdog).
 * Host-only code: the device image (and the carrier cubin) is unchanged.
 *
 * Environment (diagnostics): QSB_HP_DISABLE=1 off; QSB_HP_THREADS (1..3, default 3);
 * QSB_HP_NOSHANI=1 forces the OpenSSL path; QSB_HP_WAIT_MS (default 40);
 * QSB_HP_CORRUPT=1 flips one host word of batch 0 (self-check test). */
#pragma once
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <chrono>
#include <math.h>
#include <time.h>
#include <sched.h>
#include <pthread.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <immintrin.h>
#include <cpuid.h>
#include <openssl/sha.h>

namespace qhp {

enum { MAXK = 8, NCLS = 16, CHUNK = 16384, NSLOT = 4 };

/* ---- SHA-256 compression: 4-lane (and 1-lane) SHA-NI, else OpenSSL ---- */
#define QHP_SHA __attribute__((target("sha,sse4.1,ssse3")))
alignas(16) static const uint32_t k_[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
/* st[l] (a..h) <- compress(st[l], blk[l]), l < L; blk are 64 message bytes (big-endian words). */
template <int L>
QHP_SHA static void sha_ni(uint32_t (*st)[8], const uint8_t *const *blk) {
    const __m128i BSWAP = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
    __m128i S0[L], S1[L], I0[L], I1[L], M[L][4];
#pragma GCC unroll 4
    for (int l = 0; l < L; l++) {
        __m128i t = _mm_loadu_si128((const __m128i *)&st[l][0]);
        __m128i u = _mm_loadu_si128((const __m128i *)&st[l][4]);
        t = _mm_shuffle_epi32(t, 0xB1); u = _mm_shuffle_epi32(u, 0x1B);
        S0[l] = _mm_alignr_epi8(t, u, 8);
        S1[l] = _mm_blend_epi16(u, t, 0xF0);
        I0[l] = S0[l]; I1[l] = S1[l];
#pragma GCC unroll 4
        for (int j = 0; j < 4; j++) M[l][j] = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(blk[l] + 16 * j)), BSWAP);
    }
#pragma GCC unroll 16
    for (int r = 0; r < 16; r++) {
        const __m128i K = _mm_load_si128((const __m128i *)&k_[4 * r]);
#pragma GCC unroll 4
        for (int l = 0; l < L; l++) {
            if (r >= 4) {
                __m128i t = _mm_sha256msg1_epu32(M[l][r & 3], M[l][(r + 1) & 3]);
                t = _mm_add_epi32(t, _mm_alignr_epi8(M[l][(r + 3) & 3], M[l][(r + 2) & 3], 4));
                M[l][r & 3] = _mm_sha256msg2_epu32(t, M[l][(r + 3) & 3]);
            }
            __m128i m = _mm_add_epi32(M[l][r & 3], K);
            S1[l] = _mm_sha256rnds2_epu32(S1[l], S0[l], m);
            m = _mm_shuffle_epi32(m, 0x0E);
            S0[l] = _mm_sha256rnds2_epu32(S0[l], S1[l], m);
        }
    }
#pragma GCC unroll 4
    for (int l = 0; l < L; l++) {
        __m128i a = _mm_add_epi32(S0[l], I0[l]), b = _mm_add_epi32(S1[l], I1[l]);
        __m128i t = _mm_shuffle_epi32(a, 0x1B);
        b = _mm_shuffle_epi32(b, 0xB1);
        _mm_storeu_si128((__m128i *)&st[l][0], _mm_blend_epi16(t, b, 0xF0));
        _mm_storeu_si128((__m128i *)&st[l][4], _mm_alignr_epi8(b, t, 8));
    }
}
static bool shani_supported() {
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
    return (b >> 29) & 1;                                     /* CPUID.(7,0):EBX.SHA */
}
static bool g_shani = false;
static inline void sha_sw(uint32_t st[8], const uint8_t *blk) {
    SHA256_CTX c; memset(&c, 0, sizeof c);
    for (int i = 0; i < 8; i++) c.h[i] = st[i];
    SHA256_Transform(&c, blk);
    for (int i = 0; i < 8; i++) st[i] = c.h[i];
}
static inline void compress4(uint32_t (*st)[8], const uint8_t *const *blk) {
    if (g_shani) sha_ni<4>(st, blk);
    else for (int l = 0; l < 4; l++) sha_sw(st[l], blk[l]);
}
static inline void compress1(uint32_t st[8], const uint8_t *blk) {
    if (g_shani) { uint32_t (*s)[8] = (uint32_t (*)[8])st; sha_ni<1>(s, &blk); }
    else sha_sw(st, blk);
}
static inline uint32_t be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

/* ---- SHA stream context: state + partial block ---- */
struct SCtx { uint32_t st[8]; int len; uint8_t buf[64]; };
static inline void sc_bytes(SCtx &c, const uint8_t *p, int n) {
    while (n > 0) {
        int k = 64 - c.len; if (k > n) k = n;
        memcpy(c.buf + c.len, p, k); c.len += k; p += k; n -= k;
        if (c.len == 64) { compress1(c.st, c.buf); c.len = 0; }
    }
}
static inline void sc_push(SCtx &c, const uint8_t *row) {      /* append one SIG_PUSH_SIZE-byte push */
    if (c.len <= 64 - SIG_PUSH_SIZE) {
        memcpy(c.buf + c.len, row, SIG_PUSH_SIZE); c.len += SIG_PUSH_SIZE;
        if (c.len == 64) { compress1(c.st, c.buf); c.len = 0; }
    } else sc_bytes(c, row, SIG_PUSH_SIZE);
}

/* ---- problem constants ---- */
struct ClsVec { __m128i M1, M2, M3; uint32_t w2, w3; };   /* first-block words 4..15 (vectors), 2..3 */
struct Params {
    SCtx c0;                    /* after prefix_remainder, from the problem midstate */
    const uint8_t *rows;        /* dummy sig pushes, SIG_PUSH_SIZE bytes each, contiguous */
    int cut, K, ncls;
    uint64_t n_epochs, cap;     /* epoch space size, epochs per batch */
    ClsVec cv[NCLS];            /* first-block class words (QSB_FIRST_UNIQUE) */
};

/* First-block states of 4 classes for one epoch: compress(mid, [w0 w1 | class words 2..15]).
 * Rounds 0-1 read only w0/w1, so the first sha256rnds2 is shared by all classes. */
QHP_SHA static void first4(const uint32_t mid[8], uint32_t w0, uint32_t w1, const ClsVec *cv, uint32_t (*out)[8]) {
    __m128i t = _mm_loadu_si128((const __m128i *)&mid[0]);
    __m128i u = _mm_loadu_si128((const __m128i *)&mid[4]);
    t = _mm_shuffle_epi32(t, 0xB1); u = _mm_shuffle_epi32(u, 0x1B);
    const __m128i A0 = _mm_alignr_epi8(t, u, 8), A1 = _mm_blend_epi16(u, t, 0xF0);
    const __m128i K0 = _mm_load_si128((const __m128i *)&k_[0]);
    const __m128i S1s = _mm_sha256rnds2_epu32(A1, A0, _mm_add_epi32(_mm_set_epi32(0, 0, (int)w1, (int)w0), K0));
    __m128i S0[4], S1[4], M[4][4];
#pragma GCC unroll 4
    for (int l = 0; l < 4; l++) {
        M[l][0] = _mm_set_epi32((int)cv[l].w3, (int)cv[l].w2, (int)w1, (int)w0);
        M[l][1] = cv[l].M1; M[l][2] = cv[l].M2; M[l][3] = cv[l].M3;
        S1[l] = S1s;
        S0[l] = _mm_sha256rnds2_epu32(A0, S1s, _mm_shuffle_epi32(_mm_add_epi32(M[l][0], K0), 0x0E));
    }
#pragma GCC unroll 15
    for (int r = 1; r < 16; r++) {
        const __m128i K = _mm_load_si128((const __m128i *)&k_[4 * r]);
#pragma GCC unroll 4
        for (int l = 0; l < 4; l++) {
            if (r >= 4) {
                __m128i x = _mm_sha256msg1_epu32(M[l][r & 3], M[l][(r + 1) & 3]);
                x = _mm_add_epi32(x, _mm_alignr_epi8(M[l][(r + 3) & 3], M[l][(r + 2) & 3], 4));
                M[l][r & 3] = _mm_sha256msg2_epu32(x, M[l][(r + 3) & 3]);
            }
            __m128i m = _mm_add_epi32(M[l][r & 3], K);
            S1[l] = _mm_sha256rnds2_epu32(S1[l], S0[l], m);
            m = _mm_shuffle_epi32(m, 0x0E);
            S0[l] = _mm_sha256rnds2_epu32(S0[l], S1[l], m);
        }
    }
#pragma GCC unroll 4
    for (int l = 0; l < 4; l++) {
        __m128i a = _mm_add_epi32(S0[l], A0), b = _mm_add_epi32(S1[l], A1);
        __m128i x = _mm_shuffle_epi32(a, 0x1B);
        b = _mm_shuffle_epi32(b, 0xB1);
        _mm_storeu_si128((__m128i *)&out[l][0], _mm_blend_epi16(x, b, 0xF0));
        _mm_storeu_si128((__m128i *)&out[l][4], _mm_alignr_epi8(b, x, 8));
    }
}
static void first_sw(const Params &P, const uint32_t mid[8], const uint8_t rem8[8], uint32_t *fo) {
    for (int c = 0; c < P.ncls; c++) {
        uint8_t blk[64]; uint32_t w[16];
        memcpy(blk, rem8, 8);
        w[2] = P.cv[c].w2; w[3] = P.cv[c].w3;
        _mm_storeu_si128((__m128i *)&w[4], P.cv[c].M1); _mm_storeu_si128((__m128i *)&w[8], P.cv[c].M2);
        _mm_storeu_si128((__m128i *)&w[12], P.cv[c].M3);
        for (int j = 2; j < 16; j++) { blk[4*j] = (uint8_t)(w[j] >> 24); blk[4*j+1] = (uint8_t)(w[j] >> 16); blk[4*j+2] = (uint8_t)(w[j] >> 8); blk[4*j+3] = (uint8_t)w[j]; }
        uint32_t st[8]; memcpy(st, mid, 32); sha_sw(st, blk); memcpy(fo + (size_t)c * 8, st, 32);
    }
}

/* One pending epoch for the 4-lane tail. */
struct Lane { SCtx c; int o6; uint32_t idx; uint8_t early[MAXK]; };

/* Hash the suffix P[o6+1..cut-1] of nl (<=4) lanes in lockstep, then their first-block states.
 * The suffix is contiguous in `rows`: only each lane's first block is assembled, the others are
 * read in place. */
static void flush(const Params &P, Lane *Ls, int nl, uint8_t *ep_out, uint32_t *fi_out) {
    alignas(16) uint8_t fb[4][64];
    alignas(16) static const uint8_t dummy[64] = {0};
    const uint8_t *tail[4], *rem8[4];
    uint32_t S[4][8], F[4][8];
    int nb[4], maxnb = 0;
    for (int l = 0; l < 4; l++) {
        if (l >= nl) { nb[l] = 0; memset(S[l], 0, 32); continue; }
        const Lane &L = Ls[l];
        const uint8_t *span = P.rows + (size_t)(L.o6 + 1) * SIG_PUSH_SIZE;
        const int span_len = (P.cut - 1 - L.o6) * SIG_PUSH_SIZE;
        nb[l] = (L.c.len + span_len) >> 6;                  /* the remainder is always 8 bytes */
        memcpy(S[l], L.c.st, 32);
        if (nb[l] == 0) { memcpy(F[l], S[l], 32); rem8[l] = L.c.buf; tail[l] = dummy; }
        else {
            memcpy(fb[l], L.c.buf, L.c.len); memcpy(fb[l] + L.c.len, span, 64 - L.c.len);
            tail[l] = span + (64 - L.c.len) - 64;            /* block b >= 1 at tail + 64 b */
            rem8[l] = span + span_len - 8;
        }
        if (nb[l] > maxnb) maxnb = nb[l];
    }
    for (int b = 0; b < maxnb; b++) {
        const uint8_t *bp[4];
        for (int l = 0; l < 4; l++) bp[l] = b >= nb[l] ? dummy : b == 0 ? fb[l] : tail[l] + 64 * b;
        compress4(S, bp);
        for (int l = 0; l < 4; l++) if (b == nb[l] - 1) memcpy(F[l], S[l], 32);
    }
    for (int l = 0; l < nl; l++) {
        const Lane &L = Ls[l];
        const uint32_t w0 = be32(rem8[l]), w1 = be32(rem8[l] + 4);
        uint8_t *d = ep_out + (size_t)L.idx * 64;           /* epoch_desc_t: mid[8] remW[2] early[K] pad */
        alignas(16) uint8_t rec[64];
        memcpy(rec, F[l], 32);
        memcpy(rec + 32, &w0, 4); memcpy(rec + 36, &w1, 4);
        memset(rec + 40, 0, 24);
        memcpy(rec + 40, L.early, MAXK);
        memset(rec + 40 + P.K, 0, MAXK - P.K);
        /* write-once output read by DMA: non-temporal stores (no read-for-ownership; +40% here) */
        for (int q = 0; q < 4; q++) _mm_stream_si128((__m128i *)(d + 16 * q), _mm_load_si128((const __m128i *)(rec + 16 * q)));
        uint32_t *fo = fi_out + (size_t)L.idx * P.ncls * 8;
        if (g_shani) {
            alignas(16) uint32_t T[4][8];
            for (int c0 = 0; c0 < P.ncls; c0 += 4) {
                first4(F[l], w0, w1, P.cv + c0, T);
                if (P.ncls - c0 >= 4)
                    for (int q = 0; q < 8; q++) _mm_stream_si128((__m128i *)(fo + (size_t)c0 * 8) + q, _mm_load_si128((const __m128i *)T + q));
                else memcpy(fo + (size_t)c0 * 8, T, (size_t)(P.ncls - c0) * 32);
            }
        } else first_sw(P, F[l], rem8[l], fo);
    }
}

/* Epochs [e0, e1) of the batch starting at `base`, written at index e - base. */
static void produce(const Params &P, uint64_t base, uint64_t e0, uint64_t e1, uint8_t *ep_out, uint32_t *fi_out) {
    const int K = P.K, N = P.cut;
    uint8_t o[MAXK] = {0};
    qsb_host_unrank(e0, N, K, o);
    SCtx ctx[MAXK + 1];
    ctx[0] = P.c0;
    for (int k = 1; k <= K; k++) {
        ctx[k] = ctx[k - 1];
        for (int i = (k == 1 ? 0 : o[k - 2] + 1); i < o[k - 1]; i++) sc_push(ctx[k], P.rows + (size_t)i * SIG_PUSH_SIZE);
    }
    Lane L[4]; int nl = 0;
    for (uint64_t e = e0; e < e1; e++) {
        Lane &x = L[nl++];
        memcpy(x.c.st, ctx[K].st, 32); x.c.len = ctx[K].len; memcpy(x.c.buf, ctx[K].buf, 64);
        x.o6 = o[K - 1]; x.idx = (uint32_t)(e - base);
        memcpy(x.early, o, MAXK);
        if (nl == 4) { flush(P, L, 4, ep_out, fi_out); nl = 0; }
        if (e + 1 == e1) break;
        int i = K - 1;
        while (i >= 0 && o[i] == N - K + i) i--;
        if (i < 0) break;                                    /* end of the epoch space */
        sc_push(ctx[i + 1], P.rows + (size_t)o[i] * SIG_PUSH_SIZE);
        o[i]++;
        for (int j = i + 1; j < K; j++) { o[j] = o[j - 1] + 1; ctx[j + 1] = ctx[j]; }
    }
    if (nl) flush(P, L, nl, ep_out, fi_out);
    _mm_sfence();                                            /* order the streamed stores before "chunk done" */
}


/* ---- 16-lane AVX-512 path (QSB_HP16): the same bytes as flush()/produce(), 16 epochs per SHA-256 pass ----
 * On Zen 4 each sha256rnds2 holds an FP pipe for several cycles, so 4-lane SHA-NI is the slow way to hash on that
 * host; AVX-512F rounds (vprord, vpternlogd) for 16 lanes cost far fewer pipe-cycles per block. The suffix blocks
 * of 16 epochs are hashed in lockstep (lanes whose suffix is shorter keep their state, masked), and the
 * first-block states run class-major: one pass per class for all 16 epochs, message words 2..15 broadcast. */
#define QHP_S16 __attribute__((target("avx512f,avx512bw")))
#define HP16_ROR(x, n) _mm512_ror_epi32((x), (n))
#define HP16_ROUND(a, b, c, d, e, f, g, h, WK) do { \
    const __m512i t1_ = _mm512_add_epi32(_mm512_add_epi32(h, _mm512_ternarylogic_epi32(HP16_ROR(e, 6), HP16_ROR(e, 11), HP16_ROR(e, 25), 0x96)), \
                                         _mm512_add_epi32(_mm512_ternarylogic_epi32(e, f, g, 0xCA), (WK))); \
    const __m512i t2_ = _mm512_add_epi32(_mm512_ternarylogic_epi32(HP16_ROR(a, 2), HP16_ROR(a, 13), HP16_ROR(a, 22), 0x96), \
                                         _mm512_ternarylogic_epi32(a, b, c, 0xE8)); \
    d = _mm512_add_epi32(d, t1_); h = _mm512_add_epi32(t1_, t2_); } while (0)
QHP_S16 static inline __m512i hp16_w(__m512i *W, int t) {    /* message word t (W is the 16-word ring, updated in place) */
    if (t < 16) return W[t];
    const __m512i w15 = W[(t - 15) & 15], w2 = W[(t - 2) & 15];
    const __m512i s0 = _mm512_ternarylogic_epi32(HP16_ROR(w15, 7), HP16_ROR(w15, 18), _mm512_srli_epi32(w15, 3), 0x96);
    const __m512i s1 = _mm512_ternarylogic_epi32(HP16_ROR(w2, 17), HP16_ROR(w2, 19), _mm512_srli_epi32(w2, 10), 0x96);
    W[t & 15] = _mm512_add_epi32(_mm512_add_epi32(W[t & 15], s0), _mm512_add_epi32(W[(t - 7) & 15], s1));
    return W[t & 15];
}
#define HP16_WK(t) _mm512_add_epi32(hp16_w(W, (t)), _mm512_set1_epi32((int)k_[(t)]))
/* s <- s + compress(s, W) in 16 lanes; W[0..15] = message words (values), used as the schedule ring */
QHP_S16 static void hp16_block(__m512i s[8], __m512i *W) {
    __m512i a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
#pragma GCC unroll 8
    for (int t = 0; t < 64; t += 8) {
        HP16_ROUND(a, b, c, d, e, f, g, h, HP16_WK(t + 0)); HP16_ROUND(h, a, b, c, d, e, f, g, HP16_WK(t + 1));
        HP16_ROUND(g, h, a, b, c, d, e, f, HP16_WK(t + 2)); HP16_ROUND(f, g, h, a, b, c, d, e, HP16_WK(t + 3));
        HP16_ROUND(e, f, g, h, a, b, c, d, HP16_WK(t + 4)); HP16_ROUND(d, e, f, g, h, a, b, c, HP16_WK(t + 5));
        HP16_ROUND(c, d, e, f, g, h, a, b, HP16_WK(t + 6)); HP16_ROUND(b, c, d, e, f, g, h, a, HP16_WK(t + 7));
    }
    s[0] = _mm512_add_epi32(s[0], a); s[1] = _mm512_add_epi32(s[1], b); s[2] = _mm512_add_epi32(s[2], c); s[3] = _mm512_add_epi32(s[3], d);
    s[4] = _mm512_add_epi32(s[4], e); s[5] = _mm512_add_epi32(s[5], f); s[6] = _mm512_add_epi32(s[6], g); s[7] = _mm512_add_epi32(s[7], h);
}
#undef HP16_WK
static std::atomic<bool> g_s16{false};   /* chosen path (after the batch-0 calibration) */
static bool g_s16_ok = false;    /* the 16-lane path is available on this host */
static bool s16_supported() { __builtin_cpu_init(); return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw"); }

/* flush() for up to 16 lanes. */
QHP_S16 static void flush16(const Params &P, Lane *Ls, int nl, uint8_t *ep_out, uint32_t *fi_out) {
    alignas(64) uint8_t fb[16][64];
    alignas(64) static const uint8_t dummy[64] = {0};
    const uint8_t *tail[16], *rem8[16];
    alignas(64) uint32_t S[8][16], T[16][16];
    int nb[16], maxnb = 0;
    for (int l = 0; l < 16; l++) {
        if (l >= nl) { nb[l] = 0; for (int j = 0; j < 8; j++) S[j][l] = 0; rem8[l] = dummy; tail[l] = dummy; continue; }
        const Lane &L = Ls[l];
        const uint8_t *span = P.rows + (size_t)(L.o6 + 1) * SIG_PUSH_SIZE;
        const int span_len = (P.cut - 1 - L.o6) * SIG_PUSH_SIZE;
        nb[l] = (L.c.len + span_len) >> 6;                  /* the remainder is always 8 bytes */
        for (int j = 0; j < 8; j++) S[j][l] = L.c.st[j];
        if (nb[l] == 0) { rem8[l] = L.c.buf; tail[l] = dummy; }
        else {
            memcpy(fb[l], L.c.buf, L.c.len); memcpy(fb[l] + L.c.len, span, 64 - L.c.len);
            tail[l] = span + (64 - L.c.len) - 64;            /* block b >= 1 at tail + 64 b */
            rem8[l] = span + span_len - 8;
        }
        if (nb[l] > maxnb) maxnb = nb[l];
    }
    __m512i s[8];
    for (int j = 0; j < 8; j++) s[j] = _mm512_load_si512((const void *)S[j]);
    for (int b = 0; b < maxnb; b++) {
        __mmask16 act = 0;
        for (int l = 0; l < 16; l++) {
            const uint8_t *bp = b >= nb[l] ? dummy : b == 0 ? fb[l] : tail[l] + 64 * b;
            if (b < nb[l]) act |= (__mmask16)(1u << l);
            for (int i = 0; i < 16; i++) T[i][l] = be32(bp + 4 * i);
        }
        __m512i W[16], ns[8];
        for (int i = 0; i < 16; i++) W[i] = _mm512_load_si512((const void *)T[i]);
        for (int j = 0; j < 8; j++) ns[j] = s[j];
        hp16_block(ns, W);
        for (int j = 0; j < 8; j++) s[j] = _mm512_mask_mov_epi32(s[j], act, ns[j]);   /* finished lanes keep their state */
    }
    for (int j = 0; j < 8; j++) _mm512_store_si512((void *)S[j], s[j]);
    alignas(64) uint32_t w0a[16], w1a[16];
    for (int l = 0; l < 16; l++) { w0a[l] = be32(rem8[l]); w1a[l] = be32(rem8[l] + 4); }
    for (int l = 0; l < nl; l++) {                           /* descriptors, exactly as flush() */
        const Lane &L = Ls[l];
        const uint32_t w0 = w0a[l], w1 = w1a[l];
        uint8_t *d = ep_out + (size_t)L.idx * 64;
        alignas(16) uint8_t rec[64];
        uint32_t F[8]; for (int j = 0; j < 8; j++) F[j] = S[j][l];
        memcpy(rec, F, 32);
        memcpy(rec + 32, &w0, 4); memcpy(rec + 36, &w1, 4);
        memset(rec + 40, 0, 24);
        memcpy(rec + 40, L.early, MAXK);
        memset(rec + 40 + P.K, 0, MAXK - P.K);
        for (int q = 0; q < 4; q++) _mm_stream_si128((__m128i *)(d + 16 * q), _mm_load_si128((const __m128i *)(rec + 16 * q)));
    }
    const __m512i W0 = _mm512_load_si512((const void *)w0a), W1 = _mm512_load_si512((const void *)w1a);
    for (int c = 0; c < P.ncls; c++) {                       /* first-block states, class-major */
        alignas(16) uint32_t cw[16];
        cw[2] = P.cv[c].w2; cw[3] = P.cv[c].w3;
        _mm_storeu_si128((__m128i *)&cw[4], P.cv[c].M1); _mm_storeu_si128((__m128i *)&cw[8], P.cv[c].M2);
        _mm_storeu_si128((__m128i *)&cw[12], P.cv[c].M3);
        __m512i W[16], st[8];
        W[0] = W0; W[1] = W1;
        for (int i = 2; i < 16; i++) W[i] = _mm512_set1_epi32((int)cw[i]);
        for (int j = 0; j < 8; j++) st[j] = s[j];
        hp16_block(st, W);
        alignas(64) uint32_t O[8][16];
        for (int j = 0; j < 8; j++) _mm512_store_si512((void *)O[j], st[j]);
        for (int l = 0; l < nl; l++) {
            uint32_t *fo = fi_out + ((size_t)Ls[l].idx * P.ncls + c) * 8;
            alignas(16) uint32_t R[8]; for (int j = 0; j < 8; j++) R[j] = O[j][l];
            _mm_stream_si128((__m128i *)fo, _mm_load_si128((const __m128i *)R));
            _mm_stream_si128((__m128i *)(fo + 4), _mm_load_si128((const __m128i *)(R + 4)));
        }
    }
}
/* produce() with 16 lanes per flush. */
static void produce16(const Params &P, uint64_t base, uint64_t e0, uint64_t e1, uint8_t *ep_out, uint32_t *fi_out) {
    const int K = P.K, N = P.cut;
    uint8_t o[MAXK] = {0};
    qsb_host_unrank(e0, N, K, o);
    SCtx ctx[MAXK + 1];
    ctx[0] = P.c0;
    for (int k = 1; k <= K; k++) {
        ctx[k] = ctx[k - 1];
        for (int i = (k == 1 ? 0 : o[k - 2] + 1); i < o[k - 1]; i++) sc_push(ctx[k], P.rows + (size_t)i * SIG_PUSH_SIZE);
    }
    Lane L[16]; int nl = 0;
    for (uint64_t e = e0; e < e1; e++) {
        Lane &x = L[nl++];
        memcpy(x.c.st, ctx[K].st, 32); x.c.len = ctx[K].len; memcpy(x.c.buf, ctx[K].buf, 64);
        x.o6 = o[K - 1]; x.idx = (uint32_t)(e - base);
        memcpy(x.early, o, MAXK);
        if (nl == 16) { flush16(P, L, 16, ep_out, fi_out); nl = 0; }
        if (e + 1 == e1) break;
        int i = K - 1;
        while (i >= 0 && o[i] == N - K + i) i--;
        if (i < 0) break;                                    /* end of the epoch space */
        sc_push(ctx[i + 1], P.rows + (size_t)o[i] * SIG_PUSH_SIZE);
        o[i]++;
        for (int j = i + 1; j < K; j++) { o[j] = o[j - 1] + 1; ctx[j + 1] = ctx[j]; }
    }
    if (nl) flush16(P, L, nl, ep_out, fi_out);
    _mm_sfence();                                            /* order the streamed stores before "chunk done" */
}
static inline void produce_any(const Params &P, uint64_t base, uint64_t e0, uint64_t e1, uint8_t *ep_out, uint32_t *fi_out) {
    if (g_s16.load(std::memory_order_relaxed)) produce16(P, base, e0, e1, ep_out, fi_out); else produce(P, base, e0, e1, ep_out, fi_out);
}

/* ---- batch pipeline ----
 * Ring of NSLOT pinned slots, each NPIECE pieces (40 MiB each at the ranked shape), pinned only once the
 * search loop runs and one piece per driver call, so no allocation holds the driver for long while the
 * main thread starts up or launches. Batch 0 (always GPU-built) is also built by the host into pageable
 * memory for the self-check; the GPU's copy goes to a device scratch buffer (stream-ordered D2D copy)
 * and a worker compares the two. */
enum { NPIECE = 8 };
enum { S_FREE = 0, S_PROD, S_READY, S_UPLOAD, S_INFLIGHT };   /* UPLOAD: taken by main, copy event not yet recorded */
struct Slot {
    uint8_t *ep[NPIECE] = {}; uint32_t *fi[NPIECE] = {}; int npieces_ok = 0;
    int64_t batch = -1; int state = S_FREE;
    int nchunks = 0, next_chunk = 0, done_chunks = 0; bool abandoned = false;
    cudaEvent_t copied = nullptr;
};
struct Hp {
    Params P;
    Slot slot[NSLOT];
    std::mutex m;
    std::condition_variable cv_work, cv_ready;
    std::vector<std::thread> th;
    int64_t prod_batch = 1, need_batch = 0, n_batches = 0;
    bool stop = false, loop_started = false;
    uint64_t pe = 0;            /* epochs per piece */
    /* self-check on batch 0 */
    int check = 0;              /* 0 pending, 1 passed, -1 failed */
    uint8_t *c_ep = nullptr; uint32_t *c_fi = nullptr;           /* host-built batch 0 (pageable) */
    int c_nchunks = 0, c_next = 0, c_done = 0;
    uint8_t *d_scr_ep = nullptr; uint32_t *d_scr_fi = nullptr;   /* GPU-built batch 0 (device scratch) */
    cudaEvent_t chk_evt = nullptr; bool chk_enqueued = false, chk_running = false;
    /* rate estimates for the claim lead: GPU batch interval, host chunk time (one thread) */
    double tg = 0, t_last_acq = 0, tchunk = 0;
    /* stats / watchdog */
    uint64_t n_host = 0, n_fb = 0, ahead_sum = 0, ahead_n = 0; int consec_fb = 0, max_fb = 16, wait_ms = 40, ahead_min = 99;
    /* hashing-path calibration on batch 0: its chunks alternate SHA-NI x4 / AVX-512 x16, both are self-checked */
    double t_path[2] = {0, 0}; int n_path[2] = {0, 0};
    bool active = false, dead = false;
    int nthreads = 3, dev = 0, corrupt = 0;
};
static Hp *g_hp = nullptr;

static double now_s() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t batch_len(const Hp *h, int64_t b) {
    const uint64_t base = (uint64_t)b * h->P.cap;
    return base >= h->P.n_epochs ? 0 : (h->P.n_epochs - base < h->P.cap ? h->P.n_epochs - base : h->P.cap);
}
static void kill_locked(Hp *h, const char *why) {
    if (h->dead) return;
    h->dead = true; h->active = false; h->stop = true;
    h->cv_work.notify_all(); h->cv_ready.notify_all();
    printf("  Host producers: off (%s) -> GPU producers for the rest of the run\n", why);
    fflush(stdout);
}
/* Stop handing out chunks of a batch the main thread no longer wants. */
static void abandon_locked(Hp *h, Slot *s) {
    if (s->state == S_READY) { s->state = S_FREE; s->batch = -1; h->cv_work.notify_all(); return; }
    if (s->state != S_PROD) return;
    s->abandoned = true; s->nchunks = s->next_chunk;
    if (s->done_chunks == s->nchunks) { s->state = S_FREE; s->batch = -1; h->cv_work.notify_all(); }
}

/* Worker: compare the host-built batch 0 with the GPU's copy (device scratch -> pageable, 16 MiB at a time). */
static void run_check(Hp *h) {
    const uint64_t n = batch_len(h, 0);
    const size_t ep_n = (size_t)n * 64, fi_n = (size_t)n * h->P.ncls * 8;
    if (h->corrupt) h->c_fi[(size_t)12345 % fi_n] ^= 1u;
    const size_t BUF = 16u << 20;
    std::vector<uint8_t> buf(BUF);
    uint64_t bad_ep = 0, bad_fi = 0, first_bad = ~0ull; bool io_ok = true;
    for (size_t off = 0; off < ep_n && io_ok; off += BUF) {
        const size_t len = ep_n - off < BUF ? ep_n - off : BUF;
        io_ok = cudaMemcpy(buf.data(), h->d_scr_ep + off, len, cudaMemcpyDeviceToHost) == cudaSuccess;
        for (size_t r = 0; io_ok && r < len; r += 64)      /* mid, remW, early; the GPU never writes the pad */
            if (memcmp(buf.data() + r, h->c_ep + off + r, 40 + h->P.K)) { bad_ep++; if ((off + r) / 64 < first_bad) first_bad = (off + r) / 64; }
    }
    for (size_t off = 0; off < fi_n * 4 && io_ok; off += BUF) {
        const size_t len = fi_n * 4 - off < BUF ? fi_n * 4 - off : BUF;
        io_ok = cudaMemcpy(buf.data(), (const uint8_t *)h->d_scr_fi + off, len, cudaMemcpyDeviceToHost) == cudaSuccess;
        if (io_ok && memcmp(buf.data(), (const uint8_t *)h->c_fi + off, len))
            for (size_t w = 0; w < len / 4; w++)
                if (((const uint32_t *)buf.data())[w] != h->c_fi[off / 4 + w]) {
                    bad_fi++; const uint64_t e = (off / 4 + w) / (h->P.ncls * 8); if (e < first_bad) first_bad = e;
                }
    }
    free(h->c_ep); free(h->c_fi); h->c_ep = nullptr; h->c_fi = nullptr;
    std::lock_guard<std::mutex> g(h->m);
    h->chk_running = false;
    if (!io_ok) { (void)cudaGetLastError(); h->check = -1; kill_locked(h, "self-check readback failed"); return; }
    if (bad_ep || bad_fi) {
        h->check = -1;
        printf("  Host producers: SELF-CHECK FAILED on batch 0 (%llu descriptors, %llu first-state words differ; first epoch %llu)\n",
               (unsigned long long)bad_ep, (unsigned long long)bad_fi, (unsigned long long)first_bad);
        kill_locked(h, "self-check mismatch");
        return;
    }
    h->check = 1; h->active = true;
    printf("  Host producers: self-check passed (batch 0: %llu descriptors + %llu first-block states bit-identical)\n",
           (unsigned long long)n, (unsigned long long)n * h->P.ncls);
    fflush(stdout);
    h->cv_work.notify_all();
}

/* Release slots whose upload has completed (caller holds h->m). Workers call it too, so a slot
 * returns to the ring ~one copy time after its launch instead of at the next launch. */
static bool release_locked(Hp *h) {
    bool any = false, inflight = false;
    for (int s = 0; s < NSLOT; s++) {
        Slot &x = h->slot[s];
        if (x.state != S_INFLIGHT) continue;
        const cudaError_t q = cudaEventQuery(x.copied);
        if (q == cudaSuccess) { x.state = S_FREE; x.batch = -1; any = true; }
        else { if (q != cudaErrorNotReady) (void)cudaGetLastError(); inflight = true; }
    }
    if (any) h->cv_work.notify_all();
    return inflight;
}

static void worker(Hp *h, int id, cpu_set_t mask, bool use_mask) {
    if (use_mask) pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
    cudaSetDevice(h->dev);
    std::unique_lock<std::mutex> lk(h->m);
    while (!h->stop) {
        /* 1. self-check: build batch 0 on the host, then compare once the GPU copy is complete */
        if (h->check == 0 && h->c_next < h->c_nchunks) {
            const int ch = h->c_next++;
            lk.unlock();
            const uint64_t n = batch_len(h, 0), e0 = (uint64_t)ch * CHUNK, e1 = e0 + CHUNK < n ? e0 + CHUNK : n;
            const bool use16 = g_s16_ok && (ch & 1);
            const double t0 = now_s();
            if (use16) produce16(h->P, 0, e0, e1, h->c_ep, h->c_fi); else produce(h->P, 0, e0, e1, h->c_ep, h->c_fi);
            const double dt = now_s() - t0;
            lk.lock();
            h->tchunk = h->tchunk > 0 ? 0.8 * h->tchunk + 0.2 * dt : dt;
            if (e1 - e0 == CHUNK) { h->t_path[use16] += dt; h->n_path[use16]++; }
            h->c_done++;
            if (h->c_done == h->c_nchunks && g_s16_ok && h->n_path[0] && h->n_path[1]) {
                const double a4 = h->t_path[0] / h->n_path[0], a16 = h->t_path[1] / h->n_path[1];
                g_s16.store(a16 < 0.98 * a4, std::memory_order_relaxed);   /* adopted only if clearly faster */
                printf("  Host producers: calibration %.2f ms per chunk with SHA-NI x4, %.2f ms with AVX-512 x16: using %s\n",
                       a4 * 1e3, a16 * 1e3, g_s16.load() ? "AVX-512 x16" : "SHA-NI x4");
                fflush(stdout);
            }
            continue;
        }
        if (h->check == 0 && h->c_done == h->c_nchunks && h->chk_enqueued && !h->chk_running) {
            const cudaError_t q = cudaEventQuery(h->chk_evt);
            if (q == cudaSuccess) { h->chk_running = true; lk.unlock(); run_check(h); lk.lock(); continue; }
            if (q != cudaErrorNotReady) { (void)cudaGetLastError(); h->check = -1; kill_locked(h, "self-check event failed"); break; }
        }
        /* 2. worker 0 pins the ring, one piece per pass, once the search loop runs */
        if (id == 0 && h->loop_started) {
            Slot *a = nullptr;
            for (int s = 0; s < NSLOT && !a; s++) if (h->slot[s].npieces_ok < NPIECE) a = &h->slot[s];
            if (a) {
                const int p = a->npieces_ok;
                lk.unlock();
                bool ok = cudaHostAlloc((void **)&a->ep[p], (size_t)h->pe * 64, cudaHostAllocPortable) == cudaSuccess &&
                          cudaHostAlloc((void **)&a->fi[p], (size_t)h->pe * h->P.ncls * 32, cudaHostAllocPortable) == cudaSuccess;
                if (ok && p == 0) ok = cudaEventCreateWithFlags(&a->copied, cudaEventDisableTiming) == cudaSuccess;
                lk.lock();
                if (!ok) { (void)cudaGetLastError(); kill_locked(h, "pinned allocation failed"); break; }
                a->npieces_ok++;
                continue;
            }
        }
        /* 3. ring production (only once the self-check has passed: until then batches go to the GPU) */
        const bool inflight = release_locked(h);
        Slot *w = nullptr;
        if (h->check == 1) {
            for (int s = 0; s < NSLOT; s++) {
                Slot &x = h->slot[s];
                if (x.state == S_PROD && x.next_chunk < x.nchunks && (!w || x.batch < w->batch)) w = &x;
            }
            if (!w) {
                Slot *f = nullptr;
                for (int s = 0; s < NSLOT && !f; s++) if (h->slot[s].npieces_ok == NPIECE && h->slot[s].state == S_FREE) f = &h->slot[s];
                if (f) {
                    int64_t b = h->prod_batch > h->need_batch ? h->prod_batch : h->need_batch;
                    /* lead: a batch the host cannot finish before the main thread asks for it is wasted work;
                     * skip far enough ahead given the queued chunks, the host chunk time and the GPU batch time */
                    if (h->tg > 0 && h->tchunk > 0) {
                        int queued = 0;
                        for (int s = 0; s < NSLOT; s++) if (h->slot[s].state == S_PROD) queued += h->slot[s].nchunks - h->slot[s].done_chunks;
                        const int nch = (int)((h->P.cap + CHUNK - 1) / CHUNK);
                        const double th = (queued + nch) * h->tchunk / h->nthreads, slack = h->wait_ms * 1e-3;
                        int64_t lead = th > slack ? (int64_t)ceil((th - slack) / h->tg) : 0;
                        if (lead > 16) lead = 16;
                        if (b < h->need_batch + lead) b = h->need_batch + lead;
                    }
                    if (b < h->n_batches) {
                        w = f; w->batch = b; h->prod_batch = b + 1;
                        w->state = S_PROD; w->abandoned = false;
                        w->nchunks = (int)((batch_len(h, b) + CHUNK - 1) / CHUNK);
                        w->next_chunk = 0; w->done_chunks = 0;
                    }
                }
            }
        }
        if (!w) {
            if ((h->check == 0 && h->chk_enqueued && h->c_done == h->c_nchunks) || inflight)
                h->cv_work.wait_for(lk, std::chrono::milliseconds(2));   /* poll a pending event */
            else h->cv_work.wait(lk);
            continue;
        }
        const int ch = w->next_chunk++;
        const int64_t b = w->batch;
        lk.unlock();
        const uint64_t base = (uint64_t)b * h->P.cap, n = batch_len(h, b);
        const uint64_t e0 = base + (uint64_t)ch * CHUNK;
        const uint64_t e1 = (uint64_t)(ch + 1) * CHUNK < n ? base + (uint64_t)(ch + 1) * CHUNK : base + n;
        const int p = (int)(((uint64_t)ch * CHUNK) / h->pe);           /* chunks never straddle pieces */
        const double t0 = now_s();
        produce_any(h->P, base + (uint64_t)p * h->pe, e0, e1, w->ep[p], w->fi[p]);
        const double dt = now_s() - t0;
        lk.lock();
        h->tchunk = 0.8 * h->tchunk + 0.2 * dt;
        if (++w->done_chunks == w->nchunks) {
            if (w->abandoned) { w->state = S_FREE; w->batch = -1; h->cv_work.notify_all(); }
            else { w->state = S_READY; h->cv_ready.notify_all(); }
        }
    }
}

static void cpu_core_siblings(int cpu, cpu_set_t *out) {
    CPU_ZERO(out); CPU_SET(cpu, out);
    char path[128]; snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
    FILE *f = fopen(path, "r");
    if (!f) return;
    char buf[256];
    if (fgets(buf, sizeof buf, f)) {
        for (char *p = buf; *p;) {
            char *q; int a = (int)strtol(p, &q, 10), b = a;
            if (q == p) break;
            p = q;
            if (*p == '-') b = (int)strtol(p + 1, &p, 10);
            for (int c = a; c <= b && c < CPU_SETSIZE; c++) if (c >= 0) CPU_SET(c, out);
            if (*p == ',') p++; else break;
        }
    }
    fclose(f);
}

/* Start the host producers (main thread, once the problem, window schedule and first classes are known).
 * Workers begin at once on the self-check copy of batch 0; the pinned ring waits for the search loop. */
static void start(const digest_params_t *dp, int cut, int K, int ncls, uint64_t n_epochs, uint64_t cap) {
    if (getenv("QSB_HP_DISABLE") && atoi(getenv("QSB_HP_DISABLE"))) { printf("  Host producers: off (QSB_HP_DISABLE)\n"); return; }
    Hp *h = new Hp;
    Params &P = h->P;
    P.cut = cut; P.K = K; P.ncls = ncls; P.n_epochs = n_epochs; P.cap = cap; P.rows = dp->dummy_sigs;
    const int stream_len = (int)dp->prefix_remainder_len + SIG_PUSH_SIZE * (cut - K);
    h->pe = cap / NPIECE;
    if (K > MAXK || K < 2 || ncls < 1 || ncls > NCLS || (stream_len & 63) != 8 || cut > 150 || cap % (NPIECE * (uint64_t)CHUNK)) {
        printf("  Host producers: off (unsupported shape)\n"); delete h; return;
    }
    g_shani = shani_supported() && !(getenv("QSB_HP_NOSHANI") && atoi(getenv("QSB_HP_NOSHANI")));
    g_s16_ok = s16_supported() && !(getenv("QSB_HP_NO16") && atoi(getenv("QSB_HP_NO16")));
    g_s16.store(false);                                       /* batch 0 decides (calibration below); SHA-NI x4 until then */
    memcpy(P.c0.st, dp->midstate, 32); P.c0.len = 0;
    sc_bytes(P.c0, dp->prefix_remainder, (int)dp->prefix_remainder_len);
    for (int c = 0; c < ncls; c++) {
        const uint32_t *w = qsb_first_unique_host[c];         /* words 2..15 of the class block */
        P.cv[c].w2 = w[0]; P.cv[c].w3 = w[1];
        P.cv[c].M1 = _mm_set_epi32((int)w[5], (int)w[4], (int)w[3], (int)w[2]);
        P.cv[c].M2 = _mm_set_epi32((int)w[9], (int)w[8], (int)w[7], (int)w[6]);
        P.cv[c].M3 = _mm_set_epi32((int)w[13], (int)w[12], (int)w[11], (int)w[10]);
    }
    h->n_batches = (int64_t)((n_epochs + cap - 1) / cap);
    if (getenv("QSB_HP_THREADS")) { int t = atoi(getenv("QSB_HP_THREADS")); if (t >= 1 && t <= 3) h->nthreads = t; }
    if (getenv("QSB_HP_WAIT_MS")) { int t = atoi(getenv("QSB_HP_WAIT_MS")); if (t >= 0 && t <= 1000) h->wait_ms = t; }
    h->corrupt = getenv("QSB_HP_CORRUPT") ? atoi(getenv("QSB_HP_CORRUPT")) : 0;
    cudaGetDevice(&h->dev);
    const uint64_t n0 = batch_len(h, 0);
    h->c_ep = (uint8_t *)malloc((size_t)n0 * 64);
    h->c_fi = (uint32_t *)malloc((size_t)n0 * ncls * 32);
    h->c_nchunks = (int)((n0 + CHUNK - 1) / CHUNK);
    bool ok = h->c_ep && h->c_fi && n0 > 0 &&
              cudaMalloc((void **)&h->d_scr_ep, (size_t)n0 * 64) == cudaSuccess &&
              cudaMalloc((void **)&h->d_scr_fi, (size_t)n0 * ncls * 32) == cudaSuccess &&
              cudaEventCreateWithFlags(&h->chk_evt, cudaEventDisableTiming) == cudaSuccess;
    if (!ok) { (void)cudaGetLastError(); printf("  Host producers: off (self-check buffers)\n"); return; }
    /* Main thread stays on its current core (both SMT siblings); workers get every other allowed CPU. */
    cpu_set_t allowed, mine, wmask; bool use_mask = false;
    CPU_ZERO(&wmask);
    const int cpu = sched_getcpu();
    if (cpu >= 0 && sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
        cpu_core_siblings(cpu, &mine);
        CPU_AND(&mine, &mine, &allowed);
        for (int c = 0; c < CPU_SETSIZE; c++) if (CPU_ISSET(c, &allowed) && !CPU_ISSET(c, &mine)) CPU_SET(c, &wmask);
        if (CPU_COUNT(&wmask) >= 1 && CPU_COUNT(&mine) >= 1) {
            use_mask = true;
            pthread_setaffinity_np(pthread_self(), sizeof(mine), &mine);
        }
    }
    g_hp = h;
    for (int t = 0; t < h->nthreads; t++) h->th.emplace_back(worker, h, t, wmask, use_mask);
    printf("  Host producers: %d threads (%s, %s), %d pinned slots of %.0f MiB, self-check on batch 0\n",
           h->nthreads, g_s16_ok ? (g_shani ? "SHA-NI x4 or AVX-512 x16, calibrated on batch 0" : "AVX-512 x16 or OpenSSL, calibrated on batch 0") : g_shani ? "SHA-NI x4" : "OpenSSL", use_mask ? "off the main core" : "unpinned",
           NSLOT, (double)cap * (64 + ncls * 32) / 1048576.0);
    fflush(stdout);
}

/* Main thread, batch 0 (GPU producers), right after first_flat on slot 0's stream: stream-ordered
 * device copy of the GPU-built batch 0 for the self-check. Never waits. */
static void enqueue_check_copy(cudaStream_t st, const void *d_ep, const uint32_t *d_fi, size_t fi_pitch) {
    Hp *h = g_hp; if (!h) return;
    const size_t n = (size_t)batch_len(h, 0), w = (size_t)h->P.ncls * 32;
    cudaError_t e = cudaMemcpyAsync(h->d_scr_ep, d_ep, n * 64, cudaMemcpyDeviceToDevice, st);
    if (e == cudaSuccess) e = cudaMemcpy2DAsync(h->d_scr_fi, w, d_fi, fi_pitch, w, n, cudaMemcpyDeviceToDevice, st);
    if (e == cudaSuccess) e = cudaEventRecord(h->chk_evt, st);
    std::lock_guard<std::mutex> g(h->m);
    if (e != cudaSuccess) { (void)cudaGetLastError(); h->check = -1; kill_locked(h, "self-check copy failed"); return; }
    h->chk_enqueued = true;
    h->cv_work.notify_all();
}

/* Main thread, once per loop iteration: release slots whose upload finished. */
static void poll() {
    Hp *h = g_hp; if (!h) return;
    std::lock_guard<std::mutex> g(h->m);
    release_locked(h);
}

/* Main thread, before launching batch k: a ready host batch (slot now in flight) or null (run the GPU
 * producers for batch k). */
static Slot *acquire(int64_t k) {
    Hp *h = g_hp; if (!h) return nullptr;
    std::unique_lock<std::mutex> lk(h->m);
    const double t = now_s();
    if (k >= 3 && h->t_last_acq > 0) { const double dt = t - h->t_last_acq; h->tg = h->tg > 0 ? 0.8 * h->tg + 0.2 * dt : dt; }
    h->t_last_acq = t;
    if (k >= 1 && !h->loop_started) { h->loop_started = true; h->cv_work.notify_all(); }
    if (h->need_batch < k + 1) h->need_batch = k + 1;
    Slot *s = nullptr;
    for (int i = 0; i < NSLOT; i++)
        if (h->slot[i].batch == k && (h->slot[i].state == S_PROD || h->slot[i].state == S_READY)) s = &h->slot[i];
    if (!h->active) { if (s) abandon_locked(h, s); return nullptr; }
    if (s && s->state == S_PROD && h->wait_ms > 0)
        h->cv_ready.wait_for(lk, std::chrono::milliseconds(h->wait_ms), [&] { return s->state == S_READY || h->dead; });
    if (s && s->state == S_READY && !h->dead) {
        s->state = S_UPLOAD; h->n_host++; h->consec_fb = 0;
        if (h->n_host > 16) {       /* after the catch-up: host batches queued beyond this one */
            int ahead = 0;
            for (int i = 0; i < NSLOT; i++) if (h->slot[i].state == S_READY && h->slot[i].batch > k) ahead++;
            h->ahead_sum += ahead; h->ahead_n++; if (ahead < h->ahead_min) h->ahead_min = ahead;
        }
        return s;
    }
    if (s) abandon_locked(h, s);
    h->n_fb++;
    if (++h->consec_fb >= h->max_fb) kill_locked(h, "host cannot keep up (watchdog)");
    return nullptr;
}

/* Main thread: enqueue a host batch's upload on the slot stream (then the digest follows on it). */
static cudaError_t upload(Slot *s, cudaStream_t st, void *d_ep, uint32_t *d_fi, size_t fi_pitch, int n) {
    Hp *h = g_hp;
    const size_t w = (size_t)h->P.ncls * 32;
    cudaError_t e = cudaSuccess;
    for (int p = 0; p < NPIECE && e == cudaSuccess; p++) {
        const int64_t lo = (int64_t)p * (int64_t)h->pe;
        if (lo >= n) break;
        const size_t np = (size_t)((int64_t)n - lo < (int64_t)h->pe ? (int64_t)n - lo : (int64_t)h->pe);
        e = cudaMemcpyAsync((uint8_t *)d_ep + (size_t)lo * 64, s->ep[p], np * 64, cudaMemcpyHostToDevice, st);
        if (e == cudaSuccess)
            e = cudaMemcpy2DAsync((uint8_t *)d_fi + (size_t)lo * fi_pitch, fi_pitch, s->fi[p], w, w, np, cudaMemcpyHostToDevice, st);
    }
    if (e == cudaSuccess) e = cudaEventRecord(s->copied, st);
    /* Only now may the slot be released on its event (a stale event would read as complete). On an
     * error the run stops; the slot stays taken. */
    if (e == cudaSuccess) { std::lock_guard<std::mutex> g(h->m); s->state = S_INFLIGHT; h->cv_work.notify_one(); }
    return e;
}

static void stats(uint64_t *host, uint64_t *fb, int *state, double *ahead_avg = nullptr, int *ahead_min = nullptr) {
    Hp *h = g_hp; if (!h) { *host = *fb = 0; *state = -2; return; }
    std::lock_guard<std::mutex> g(h->m);
    *host = h->n_host; *fb = h->n_fb; *state = h->dead ? -1 : h->active ? 1 : 0;
    if (ahead_avg) *ahead_avg = h->ahead_n ? (double)h->ahead_sum / h->ahead_n : 0.0;
    if (ahead_min) *ahead_min = h->ahead_n ? h->ahead_min : 0;
}

/* Stop and join the workers (a chunk or one piece allocation takes a few ms). */
static void shutdown() {
    Hp *h = g_hp; if (!h) return;
    { std::lock_guard<std::mutex> g(h->m); h->stop = true; h->cv_work.notify_all(); h->cv_ready.notify_all(); }
    for (auto &t : h->th) if (t.joinable()) t.join();
    h->th.clear();
}

}  // namespace qhp
