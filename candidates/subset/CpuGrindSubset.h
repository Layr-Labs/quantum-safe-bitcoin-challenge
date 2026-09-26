#pragma once
/* Host-CPU co-grinder for the subset track. The field arithmetic, the 16-bit windowed host table
 * and the batch-affine additions are Ryun1's pinning CpuGrind.h (public submission 7a75fa50,
 * GPL-3), unchanged; the candidate enumeration, preimage hashing and hit publication are subset's.
 *
 * Candidates are disjoint from the GPU's: the GPU grinds every epoch (6 early omissions below
 * the cut) with its 128 window-omission patterns (h_win3); the CPU grinds epochs t, t+T, t+2T, ...
 * (T threads) with the other 158 of the C(13,3)=286 window patterns. Every CPU hit passes the same
 * exact OpenSSL gate as the GPU's tentatives (qsb_hv_check) before it is appended to
 * results/digest_hit_cpu.txt, which the harness collects with the GPU's hit file. Workers run at
 * SCHED_IDLE, so they never delay the GPU host thread; QSB_CPU_GRIND=0 compiles it out. */
#include <openssl/sha.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <sched.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <new>
#ifndef QSB_CPU_VEC
#define QSB_CPU_VEC 1              /* 8-lane AVX-512 IFMA field arithmetic when the host CPU has it */
#endif
#if QSB_CPU_VEC && defined(__x86_64__) && !defined(__CUDA_ARCH__)
#define QCPU_VEC 1
#include <immintrin.h>
#else
#define QCPU_VEC 0
#endif
#ifndef QSB_CPU_SHANI
#define QSB_CPU_SHANI 1            /* 4-lane SHA-256 with the x86 SHA extensions when the host CPU has them */
#endif
#if QSB_CPU_SHANI && defined(__x86_64__) && !defined(__CUDA_ARCH__)
#define QCPU_SHANI 1
#include <immintrin.h>
#include <cpuid.h>
#else
#define QCPU_SHANI 0
#endif

#ifndef QSB_CPU_RESERVE
#define QSB_CPU_RESERVE 2          /* logical CPUs left for the GPU host thread and driver */
#endif
// Schedule reuse adapted from newjordan 889742 / acdf549d; CPU geometry unchanged.
#ifndef QSB_CPU_FIRSTHASH_SCHEDULE
#define QSB_CPU_FIRSTHASH_SCHEDULE 1
#endif
#if QSB_CPU_FIRSTHASH_SCHEDULE != 0 && QSB_CPU_FIRSTHASH_SCHEDULE != 1
#error "QSB_CPU_FIRSTHASH_SCHEDULE must be 0 or 1"
#endif
#ifndef QSB_CPU_BATCH
#define QSB_CPU_BATCH 4096
#endif
/* Host table geometry, inspired by newjordan e55d31ef / 30cc6915.
 * This selector controls CPU digit and table representation. */
#ifndef QSB_CPU_W20
#define QSB_CPU_W20 1
#endif
#if QSB_CPU_W20 != 0 && QSB_CPU_W20 != 1
#error "QSB_CPU_W20 must be 0 or 1"
#endif
#ifndef QSB_CPU_W
#if QSB_CPU_W20
#define QSB_CPU_W 20
#else
#define QSB_CPU_W 16
#endif
#endif
static_assert(QSB_CPU_W == (QSB_CPU_W20 ? 20 : 16), "CPU width must match selector");

namespace qcpu {
static const int W = QSB_CPU_W, NW = (256 + QSB_CPU_W - 1) / QSB_CPU_W, NE = (1 << QSB_CPU_W) - 1;
#if QSB_CPU_W20
typedef uint32_t digit_t;
#else
typedef uint16_t digit_t;
#endif
/* Only the final bank is partial: W20 has 12 full banks plus 16 top bits. */
static inline unsigned window_entries(int i) {
    const int left = 256 - W * i;
    return (1u << (left < W ? left : W)) - 1u;
}
static inline size_t table_entries() {
    return (size_t)(NW - 1) * NE + window_entries(NW - 1);
}
typedef unsigned __int128 u128;
/* 64 B-aligned storage: one table point = one cache line. */
template <class T> struct qalloc64 {
    typedef T value_type;
    qalloc64() = default;
    template <class U> qalloc64(const qalloc64<U> &) {}
    T *allocate(size_t n) { void *q = nullptr; if (posix_memalign(&q, 64, n * sizeof(T) + 64)) throw std::bad_alloc(); return (T *)q; }
    void deallocate(T *q, size_t) { free(q); }
    template <class U> bool operator==(const qalloc64<U> &) const { return true; }
    template <class U> bool operator!=(const qalloc64<U> &) const { return false; }
};
struct fe { uint64_t v[4]; };      /* canonical (< p) little-endian limbs */
static const uint64_t P0 = 0xFFFFFFFEFFFFFC2FULL, PK = 0x1000003D1ULL;   /* p = 2^256 - PK */

static inline bool fe_is_zero(const fe &a) { return !(a.v[0] | a.v[1] | a.v[2] | a.v[3]); }
static inline bool fe_eq(const fe &a, const fe &b) {
    return !((a.v[0] ^ b.v[0]) | (a.v[1] ^ b.v[1]) | (a.v[2] ^ b.v[2]) | (a.v[3] ^ b.v[3]));
}
static inline bool fe_ge_p(const uint64_t v[4]) {
    return v[3] == ~0ULL && v[2] == ~0ULL && v[1] == ~0ULL && v[0] >= P0;
}
static inline void fe_sub_p(uint64_t v[4]) {           /* v -= p  ==  v += PK mod 2^256 */
    u128 c = (u128)v[0] + PK; v[0] = (uint64_t)c; c >>= 64;
    for (int i = 1; i < 4; i++) { c += v[i]; v[i] = (uint64_t)c; c >>= 64; }
}
static inline void fe_add(fe &r, const fe &a, const fe &b) {
    u128 c = 0; uint64_t t[4];
    for (int i = 0; i < 4; i++) { c += (u128)a.v[i] + b.v[i]; t[i] = (uint64_t)c; c >>= 64; }
    if (c || fe_ge_p(t)) fe_sub_p(t);
    memcpy(r.v, t, 32);
}
static inline void fe_sub(fe &r, const fe &a, const fe &b) {
    uint64_t t[4]; unsigned borrow = 0;
    for (int i = 0; i < 4; i++) {
        u128 d = (u128)a.v[i] - b.v[i] - borrow;
        t[i] = (uint64_t)d; borrow = (unsigned)((d >> 64) & 1);
    }
    if (borrow) {                                      /* t += p  ==  t -= PK mod 2^256 */
        u128 d = (u128)t[0] - PK; t[0] = (uint64_t)d; unsigned b = (unsigned)((d >> 64) & 1);
        for (int i = 1; i < 4; i++) { d = (u128)t[i] - b; t[i] = (uint64_t)d; b = (unsigned)((d >> 64) & 1); }
    }
    memcpy(r.v, t, 32);
}
static inline void fe_mul(fe &r, const fe &a, const fe &b) {
    uint64_t l[8] = {0};
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            u128 acc = (u128)a.v[i] * b.v[j] + l[i + j] + carry;
            l[i + j] = (uint64_t)acc; carry = (uint64_t)(acc >> 64);
        }
        l[i + 4] = carry;
    }
    /* fold: L + H*PK, then the <2^34 top again */
    uint64_t m[4]; u128 c = 0;
    for (int i = 0; i < 4; i++) { c += (u128)l[i] + (u128)l[i + 4] * PK; m[i] = (uint64_t)c; c >>= 64; }
    uint64_t top = (uint64_t)c;
    c = (u128)m[0] + (u128)top * PK; m[0] = (uint64_t)c; c >>= 64;
    for (int i = 1; i < 4; i++) { c += m[i]; m[i] = (uint64_t)c; c >>= 64; }
    if (c) fe_sub_p(m);                                /* wrapped past 2^256: add PK once more */
    if (fe_ge_p(m)) fe_sub_p(m);
    memcpy(r.v, m, 32);
}
static inline void fe_sqr(fe &r, const fe &a) { fe_mul(r, a, a); }
static void fe_inv(fe &r, const fe &a) {               /* a^(p-2) */
    static const uint64_t e[4] = {0xFFFFFFFEFFFFFC2DULL, ~0ULL, ~0ULL, ~0ULL};
    fe x = a, acc = {{1, 0, 0, 0}};
    for (int i = 0; i < 256; i++) {
        if ((e[i >> 6] >> (i & 63)) & 1) fe_mul(acc, acc, x);
        fe_sqr(x, x);
    }
    r = acc;
}
static void fe_from_le32(fe &r, const uint8_t b[32]) {
    for (int i = 0; i < 4; i++) { uint64_t w = 0; for (int k = 7; k >= 0; k--) w = (w << 8) | b[i * 8 + k]; r.v[i] = w; }
}
static void fe_from_bn(fe &r, const BIGNUM *bn) {
    uint8_t be[32] = {0}; int n = BN_num_bytes(bn); BN_bn2bin(bn, be + 32 - n);
    for (int i = 0; i < 4; i++) { uint64_t w = 0; for (int k = 0; k < 8; k++) w = (w << 8) | be[(3 - i) * 8 + k]; r.v[i] = w; }
}
struct pt { fe x, y; };
static pt pt_double(const pt &q) {                     /* affine doubling: lam = 3x^2 / 2y */
    fe x2, num, den, inv, lam, x3, y3, t;
    fe_sqr(x2, q.x); fe_add(num, x2, x2); fe_add(num, num, x2);
    fe_add(den, q.y, q.y); fe_inv(inv, den); fe_mul(lam, num, inv);
    fe_sqr(x3, lam); fe_sub(x3, x3, q.x); fe_sub(x3, x3, q.x);
    fe_sub(t, q.x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, q.y);
    return {x3, y3};
}

/* Batch-affine add: out[k] = in[k] + t[k] for the active k (neither is infinity and
 * x differs). inf[k] marks in[k] = O (then out = t). bad[k] set on x collisions
 * (probability ~2^-240; the candidate is dropped, never published). */
static void batch_add(pt *acc, const pt *const *tp, uint8_t *inf, uint8_t *bad, int n,
                      fe *d, fe *pre) {
    fe run = {{1, 0, 0, 0}};
    for (int k = 0; k < n; k++) {
        if (k + 16 < n && tp[k + 16]) __builtin_prefetch(tp[k + 16]);
        if (bad[k] || !tp[k]) { pre[k] = run; continue; }
        if (inf[k]) { pre[k] = run; continue; }
        fe_sub(d[k], tp[k]->x, acc[k].x);
        if (fe_is_zero(d[k])) { bad[k] = 1; pre[k] = run; continue; }
        pre[k] = run; fe_mul(run, run, d[k]);
    }
    fe inv; fe_inv(inv, run);
    for (int k = n - 1; k >= 0; k--) {
        if (bad[k] || !tp[k]) continue;
        if (inf[k]) { acc[k] = *tp[k]; inf[k] = 0; continue; }
        fe dinv; fe_mul(dinv, inv, pre[k]); fe_mul(inv, inv, d[k]);
        fe lam, t, x3, y3;
        fe_sub(t, tp[k]->y, acc[k].y); fe_mul(lam, t, dinv);
        fe_sqr(x3, lam); fe_sub(x3, x3, acc[k].x); fe_sub(x3, x3, tp[k]->x);
        fe_sub(t, acc[k].x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, acc[k].y);
        acc[k].x = x3; acc[k].y = y3;
    }
}


#if QCPU_VEC
/* ---- 8-lane path (AVX-512 IFMA, radix 2^52), selected at run time when the host CPU has it ----
 * Eight candidates per vector lane group; the same batch-affine formulas as batch_add, with the
 * batch inversion split into four interleaved Montgomery chains per window. Candidates with a zero
 * window digit (16/65536 of them) are dropped instead of taking the point-at-infinity branch. */
/* 8-lane secp256k1 field arithmetic with AVX-512 IFMA (vpmadd52luq/huq), radix 2^52.
 * Each lane holds one field element as 5 limbs; "normalized" means limbs 0..3 < 2^52 and
 * limb 4 < 2^49 (value < 2^257). IFMA multiplies only the low 52 bits of its operands, so
 * every multiplication input must be normalized. Outputs of fe8_mul/fe8_sub/fe8_add are normalized. */
struct fe8 { __m512i l[5]; };
#ifndef QSB_CPU_VECTOR_COPY
#define QSB_CPU_VECTOR_COPY 1
#endif
static inline __attribute__((always_inline, target("avx512f,avx512ifma")))
void fe8_mov(fe8 &r, const fe8 &a) {
#if QSB_CPU_VECTOR_COPY
    r.l[0] = a.l[0]; r.l[1] = a.l[1]; r.l[2] = a.l[2]; r.l[3] = a.l[3]; r.l[4] = a.l[4];
#else
    r = a;
#endif
}

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
/* Reduce a 10-column product (columns < 2^57, value < 2^514) to a normalized element. */
static inline __attribute__((always_inline, target("avx512f,avx512ifma")))
void fe8_red(fe8 &r, __m512i c0, __m512i c1, __m512i c2, __m512i c3, __m512i c4,
             __m512i c5, __m512i c6, __m512i c7, __m512i c8, __m512i c9) {
    const __m512i Z = _mm512_setzero_si512();
#define LO(acc, x, y) acc = _mm512_madd52lo_epu64(acc, x, y)
#define HI(acc, x, y) acc = _mm512_madd52hi_epu64(acc, x, y)
    const __m512i M = F8_M52;
    /* normalize the high columns c5..c9 to 52-bit limbs (carry into c10) */
    __m512i t;
    t = _mm512_srli_epi64(c4, 52); c4 = _mm512_and_si512(c4, M); c5 = _mm512_add_epi64(c5, t);
    t = _mm512_srli_epi64(c5, 52); c5 = _mm512_and_si512(c5, M); c6 = _mm512_add_epi64(c6, t);
    t = _mm512_srli_epi64(c6, 52); c6 = _mm512_and_si512(c6, M); c7 = _mm512_add_epi64(c7, t);
    t = _mm512_srli_epi64(c7, 52); c7 = _mm512_and_si512(c7, M); c8 = _mm512_add_epi64(c8, t);
    t = _mm512_srli_epi64(c8, 52); c8 = _mm512_and_si512(c8, M); c9 = _mm512_add_epi64(c9, t);
    __m512i c10 = _mm512_srli_epi64(c9, 52); c9 = _mm512_and_si512(c9, M);
    /* fold: value = L + H*2^260, 2^260 = R = 0x1000003D10 mod p; H limbs < 2^52 */
    const __m512i R = _mm512_set1_epi64(0x1000003D10ULL);
    LO(c0,c5,R); HI(c1,c5,R);
    LO(c1,c6,R); HI(c2,c6,R);
    LO(c2,c7,R); HI(c3,c7,R);
    LO(c3,c8,R); HI(c4,c8,R);
    LO(c4,c9,R); __m512i c5b = Z; HI(c5b,c9,R);
    /* c10 * 2^520 = c10 * R * 2^260 ... c10 is tiny (< 2^5): fold as c10*R into limb 5 */
    c5b = _mm512_madd52lo_epu64(c5b, c10, R);
    fe8 o; o.l[0] = c0; o.l[1] = c1; o.l[2] = c2; o.l[3] = c3; o.l[4] = c4;
    /* carry-propagate, then fold c5b (weight 2^260 -> R) */
    t = _mm512_srli_epi64(o.l[0], 52); o.l[0] = _mm512_and_si512(o.l[0], M); o.l[1] = _mm512_add_epi64(o.l[1], t);
    t = _mm512_srli_epi64(o.l[1], 52); o.l[1] = _mm512_and_si512(o.l[1], M); o.l[2] = _mm512_add_epi64(o.l[2], t);
    t = _mm512_srli_epi64(o.l[2], 52); o.l[2] = _mm512_and_si512(o.l[2], M); o.l[3] = _mm512_add_epi64(o.l[3], t);
    t = _mm512_srli_epi64(o.l[3], 52); o.l[3] = _mm512_and_si512(o.l[3], M); o.l[4] = _mm512_add_epi64(o.l[4], t);
    t = _mm512_srli_epi64(o.l[4], 52); o.l[4] = _mm512_and_si512(o.l[4], M); c5b = _mm512_add_epi64(c5b, t);
    /* c5b < 2^40: c5b * R < 2^77 -> add lo into limb 0, hi into limb 1 */
    o.l[0] = _mm512_madd52lo_epu64(o.l[0], c5b, R);
    o.l[1] = _mm512_madd52hi_epu64(o.l[1], c5b, R);
    fe8_carry(o);
    r = o;
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
/* 8-lane batch-affine EC pipeline for the CPU co-grinder (AVX-512 IFMA). Requires fe8.h, the
 * scalar fe/pt types and a table of canonical affine points (4x64 limbs, 64 B per point). */
#define Q8T __attribute__((target("avx512f,avx512ifma")))

Q8T static inline void fe8_set1(fe8 &r) {
    r.l[0] = _mm512_set1_epi64(1);
    for (int i = 1; i < 5; i++) r.l[i] = _mm512_setzero_si512();
}
Q8T static inline void fe8_bcast(fe8 &r, const fe &a) {           /* canonical scalar -> all lanes */
    const uint64_t M = 0xFFFFFFFFFFFFFULL;
    r.l[0] = _mm512_set1_epi64((long long)(a.v[0] & M));
    r.l[1] = _mm512_set1_epi64((long long)((a.v[0] >> 52 | a.v[1] << 12) & M));
    r.l[2] = _mm512_set1_epi64((long long)((a.v[1] >> 40 | a.v[2] << 24) & M));
    r.l[3] = _mm512_set1_epi64((long long)((a.v[2] >> 28 | a.v[3] << 36) & M));
    r.l[4] = _mm512_set1_epi64((long long)(a.v[3] >> 16));
}
Q8T static inline void fe8_from64(fe8 &r, __m512i a0, __m512i a1, __m512i a2, __m512i a3) {
    const __m512i M = F8_M52;
    r.l[0] = _mm512_and_si512(a0, M);
    r.l[1] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(a0, 52), _mm512_slli_epi64(a1, 12)), M);
    r.l[2] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(a1, 40), _mm512_slli_epi64(a2, 24)), M);
    r.l[3] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(a2, 28), _mm512_slli_epi64(a3, 36)), M);
    r.l[4] = _mm512_srli_epi64(a3, 16);
}
/* Load 8 table points (one 64 B row each: x0..x3 y0..y3) and transpose to SoA. */
Q8T static inline void pt8_load(fe8 &x, fe8 &y, const pt *const *rows) {
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
/* lane -> canonical 4x64 */
static inline void fe8_lane_canon(fe &r, const uint64_t l[5]) {
    typedef unsigned __int128 q128;
    q128 c = (q128)l[0] + ((q128)l[1] << 52);
    uint64_t w0 = (uint64_t)c; c >>= 64;
    c += (q128)l[2] << 40; uint64_t w1 = (uint64_t)c; c >>= 64;
    c += (q128)l[3] << 28; uint64_t w2 = (uint64_t)c; c >>= 64;
    c += (q128)l[4] << 16; uint64_t w3 = (uint64_t)c; c >>= 64;
    uint64_t top = (uint64_t)c;
    while (top) {                                      /* v = top*2^256 + w, 2^256 = PK mod p */
        q128 d = (q128)w0 + (q128)top * 0x1000003D1ULL; w0 = (uint64_t)d; d >>= 64;
        d += w1; w1 = (uint64_t)d; d >>= 64; d += w2; w2 = (uint64_t)d; d >>= 64;
        d += w3; w3 = (uint64_t)d; d >>= 64; top = (uint64_t)d;
    }
    if (w3 == ~0ULL && w2 == ~0ULL && w1 == ~0ULL && w0 >= 0xFFFFFFFEFFFFFC2FULL) {
        q128 d = (q128)w0 + 0x1000003D1ULL; w0 = (uint64_t)d; d >>= 64;
        d += w1; w1 = (uint64_t)d; d >>= 64; d += w2; w2 = (uint64_t)d; d >>= 64; w3 += (uint64_t)d;
    }
    r.v[0] = w0; r.v[1] = w1; r.v[2] = w2; r.v[3] = w3;
}
Q8T static inline void fe8_store_canon(fe out[8], const fe8 &a) {
    alignas(64) uint64_t L[5][8];
    for (int i = 0; i < 5; i++) _mm512_store_si512(L[i], a.l[i]);
    for (int j = 0; j < 8; j++) { uint64_t l[5] = {L[0][j], L[1][j], L[2][j], L[3][j], L[4][j]}; fe8_lane_canon(out[j], l); }
}
#ifndef QSB_CPU_Y_PARITY_ONLY
#define QSB_CPU_Y_PARITY_ONLY 1
#endif
#if QSB_CPU_Y_PARITY_ONLY != 0 && QSB_CPU_Y_PARITY_ONLY != 1
#error "QSB_CPU_Y_PARITY_ONLY must be 0 or 1"
#endif
/* Normalized radix52 input: l0..3 < 2^52, l4 < 2^49. Return canonical y parity. */
Q8T static inline uint8_t fe8_parity_mask(const fe8 &a) {
    const __m512i M = F8_M52;
    const __mmask8 middle = _mm512_cmpeq_epi64_mask(a.l[1], M)
                         & _mm512_cmpeq_epi64_mask(a.l[2], M)
                         & _mm512_cmpeq_epi64_mask(a.l[3], M);
    const __m512i P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL);
    const __m512i PP4 = _mm512_set1_epi64(0x1FFFFFFFFFFFFULL);
    const __mmask8 ge_p = _mm512_cmp_epu64_mask(a.l[4], P4, _MM_CMPINT_GT)
        | (_mm512_cmpeq_epi64_mask(a.l[4], P4) & middle
           & _mm512_cmp_epu64_mask(a.l[0], _mm512_set1_epi64(0xFFFFEFFFFFC2FULL), _MM_CMPINT_GE));
    const __mmask8 ge_2p = _mm512_cmpeq_epi64_mask(a.l[4], PP4) & middle
        & _mm512_cmp_epu64_mask(a.l[0], _mm512_set1_epi64(0xFFFFDFFFFF85EULL), _MM_CMPINT_GE);
    const __mmask8 odd = _mm512_test_epi64_mask(a.l[0], _mm512_set1_epi64(1));
    return (uint8_t)(odd ^ ge_p ^ ge_2p);
}
/* x[c] = x[c]^(p-2) for c < 4, interleaved (libsecp256k1's addition chain: 255 sqr + 15 mul). */
/* x^(p-2) for one 8-lane element: libsecp256k1's secp256k1_fe_inv addition chain
 * (255 squarings, 15 multiplications). */
Q8T static void fe8_inv1(fe8 &x) {
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
/* Invert the 4 interleaved chain products with ONE exponentiation (Montgomery's trick across the
 * chains: 3 + 6 extra multiplications). The chain is latency-bound either way, so this replaces
 * 4 x 270 multiplications of issue slots by 279. */
Q8T static void fe8_inv4(fe8 *x) {
    fe8 a01, a23, a, i01, i23;
    fe8_mul(a01, x[0], x[1]); fe8_mul(a23, x[2], x[3]); fe8_mul(a, a01, a23);
    fe8_inv1(a);
    fe8_mul(i01, a, a23); fe8_mul(i23, a, a01);
    const fe8 x0 = x[0], x2 = x[2];
    fe8_mul(x[0], i01, x[1]); fe8_mul(x[1], i01, x0);
    fe8_mul(x[2], i23, x[3]); fe8_mul(x[3], i23, x2);
}
/* One batch-affine window step over G groups of 8 (G % 4 == 0): acc[g] += T[dig-1].
 * rows(g, j) must give the table row for lane j of group g (never a digit-0 row). */
template <class RowFn>
Q8T static void ec8_window(fe8 *X, fe8 *Y, fe8 *D, fe8 *PRE, int G, RowFn rowfn) {
    fe8 run[4]; for (int c = 0; c < 4; c++) fe8_set1(run[c]);
    const pt *rows[8];
    for (int g = 0; g < G; g += 4) {
        for (int c = 0; c < 4; c++) {
            const int h = g + c;
            if (h + 8 < G) { const pt *pr[8]; rowfn(h + 8, pr); for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pr[j], _MM_HINT_T0); }
            rowfn(h, rows);
            fe8 tx, ty; pt8_load(tx, ty, rows);
            fe8_sub(D[h], tx, X[h]);
            fe8_mov(PRE[h], run[c]);
            fe8_mul(run[c], run[c], D[h]);
        }
    }
    fe8_inv4(run);
    for (int g = G - 4; g >= 0; g -= 4) {
        fe8 dinv[4], lam[4], tx[4], ty[4], t[4], x3[4];
        for (int c = 3; c >= 0; c--) {
            const int h = g + c;
            fe8_mul(dinv[c], run[c], PRE[h]); fe8_mul(run[c], run[c], D[h]);
            rowfn(h, rows); pt8_load(tx[c], ty[c], rows);
        }
        for (int c = 0; c < 4; c++) { fe8_sub(t[c], ty[c], Y[g + c]); fe8_mul(lam[c], t[c], dinv[c]); }
        for (int c = 0; c < 4; c++) { fe8_sqr(x3[c], lam[c]); fe8_sub(x3[c], x3[c], X[g + c]); fe8_sub(x3[c], x3[c], tx[c]); }
        for (int c = 0; c < 4; c++) { fe8_sub(t[c], X[g + c], x3[c]); fe8_mul(t[c], lam[c], t[c]); fe8_sub(Y[g + c], t[c], Y[g + c]); fe8_mov(X[g + c], x3[c]); }
    }
}
/* Final step for both recovery ids: Q_ri = C_ri + acc, C_1 = -C_0. Writes the canonical x and the
 * parity of the canonical y of each (candidate, recid). */
Q8T static void ec8_final(const fe8 *X, const fe8 *Y, fe8 *D, fe8 *PRE, int G, const fe &cx, const fe &cy,
                          fe *qx /* [G*8][2] */, uint8_t *qp) {
    fe8 CX, CY0, CY1, Z; fe8_bcast(CX, cx); fe8_bcast(CY0, cy);
    for (int i = 0; i < 5; i++) Z.l[i] = _mm512_setzero_si512();
    fe8_sub(CY1, Z, CY0);
    fe8 run[4]; for (int c = 0; c < 4; c++) fe8_set1(run[c]);
    for (int g = 0; g < G; g += 4)
        for (int c = 0; c < 4; c++) { const int h = g + c; fe8_sub(D[h], CX, X[h]); fe8_mov(PRE[h], run[c]); fe8_mul(run[c], run[c], D[h]); }
    fe8_inv4(run);
    for (int g = G - 4; g >= 0; g -= 4) {
        for (int c = 3; c >= 0; c--) {
            const int h = g + c;
            fe8 dinv; fe8_mul(dinv, run[c], PRE[h]); fe8_mul(run[c], run[c], D[h]);
            for (int ri = 0; ri < 2; ri++) {
                fe8 t, lam, x3, y3;
                fe8_sub(t, ri ? CY1 : CY0, Y[h]); fe8_mul(lam, t, dinv);
                fe8_sqr(x3, lam); fe8_sub(x3, x3, X[h]); fe8_sub(x3, x3, CX);
                fe8_sub(t, X[h], x3); fe8_mul(y3, lam, t); fe8_sub(y3, y3, Y[h]);
                fe ox[8]; fe8_store_canon(ox, x3);
#if QSB_CPU_Y_PARITY_ONLY
                const uint8_t parity = fe8_parity_mask(y3);
                for (int j = 0; j < 8; j++) { qx[(size_t)(h * 8 + j) * 2 + ri] = ox[j]; qp[(size_t)(h * 8 + j) * 2 + ri] = (uint8_t)((parity >> j) & 1); }
#else
                fe oy[8]; fe8_store_canon(oy, y3);
                for (int j = 0; j < 8; j++) { qx[(size_t)(h * 8 + j) * 2 + ri] = ox[j]; qp[(size_t)(h * 8 + j) * 2 + ri] = (uint8_t)(oy[j].v[0] & 1); }
#endif
            }
        }
    }
}
/* Window 0: acc = T0[d0 - 1] (digit-0 lanes load row 0 and are dropped by the caller). */
template <class RowFn>
Q8T static void ec8_first(fe8 *X, fe8 *Y, int G, RowFn rowfn) {
    const pt *rows[8];
    for (int h = 0; h < G; h++) {
        if (h + 8 < G) { const pt *pr[8]; rowfn(h + 8, pr); for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pr[j], _MM_HINT_T0); }
        rowfn(h, rows); pt8_load(X[h], Y[h], rows);
    }
}
#endif  /* QCPU_VEC */

#if QCPU_SHANI
/* SHA-256 compression of 4 independent (state, block) pairs with the x86 SHA extensions,
 * instruction streams interleaved so the sha256rnds2 latency of one lane hides behind the others. */
#define QSHA __attribute__((target("sha,sse4.1,ssse3")))
alignas(16) static const uint32_t qsha_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
/* st[l] (8 words, a..h) <- compress(st[l], blk[l]) for l = 0..3. */
QSHA static void qsha_x4(uint32_t (*st)[8], const uint8_t *const *blk) {
    const __m128i BSWAP = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
    __m128i S0[4], S1[4], I0[4], I1[4], M[4][4];
#pragma GCC unroll 4
    for (int l = 0; l < 4; l++) {
        __m128i t = _mm_loadu_si128((const __m128i *)&st[l][0]);           /* a b c d */
        __m128i u = _mm_loadu_si128((const __m128i *)&st[l][4]);           /* e f g h */
        t = _mm_shuffle_epi32(t, 0xB1); u = _mm_shuffle_epi32(u, 0x1B);
        S0[l] = _mm_alignr_epi8(t, u, 8);                                   /* ABEF */
        S1[l] = _mm_blend_epi16(u, t, 0xF0);                                /* CDGH */
        I0[l] = S0[l]; I1[l] = S1[l];
#pragma GCC unroll 4
        for (int j = 0; j < 4; j++) M[l][j] = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(blk[l] + 16 * j)), BSWAP);
    }
#pragma GCC unroll 16
    for (int r = 0; r < 16; r++) {
        const __m128i K = _mm_load_si128((const __m128i *)&qsha_k[4 * r]);
#pragma GCC unroll 4
        for (int l = 0; l < 4; l++) {
            if (r >= 4) {                                                   /* W[4r..4r+3] */
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
    for (int l = 0; l < 4; l++) {
        __m128i a = _mm_add_epi32(S0[l], I0[l]), b = _mm_add_epi32(S1[l], I1[l]);
        __m128i t = _mm_shuffle_epi32(a, 0x1B);                             /* FEBA */
        b = _mm_shuffle_epi32(b, 0xB1);                                     /* DCHG */
        _mm_storeu_si128((__m128i *)&st[l][0], _mm_blend_epi16(t, b, 0xF0)); /* DCBA -> a b c d */
        _mm_storeu_si128((__m128i *)&st[l][4], _mm_alignr_epi8(b, t, 8));    /* HGFE -> e f g h */
    }
}
static const uint32_t qsha_iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
#if QSB_CPU_FIRSTHASH_SCHEDULE
/* wk[0..63] = W[t] + K[t] of one block, from its 16 message words (big-endian word values). */
QSHA static void qsha_sched(uint32_t *wk, const uint32_t *w) {
    __m128i M[4];
#pragma GCC unroll 4
    for (int j = 0; j < 4; j++) {
        M[j] = _mm_loadu_si128((const __m128i *)(w + 4 * j));
        _mm_store_si128((__m128i *)(wk + 4 * j), _mm_add_epi32(M[j], _mm_load_si128((const __m128i *)&qsha_k[4 * j])));
    }
#pragma GCC unroll 12
    for (int r = 4; r < 16; r++) {
        __m128i t = _mm_sha256msg1_epu32(M[r & 3], M[(r + 1) & 3]);
        t = _mm_add_epi32(t, _mm_alignr_epi8(M[(r + 3) & 3], M[(r + 2) & 3], 4));
        M[r & 3] = _mm_sha256msg2_epu32(t, M[(r + 3) & 3]);
        _mm_store_si128((__m128i *)(wk + 4 * r), _mm_add_epi32(M[r & 3], _mm_load_si128((const __m128i *)&qsha_k[4 * r])));
    }
}
/* Rounds only: st[l] <- compress(st[l], block b) for b = 0..nblk-1, where block b of lane l is given
 * by its precomputed W+K (wk[4 * b + l], 64 words, 16 B aligned). Only the chaining state lives in
 * registers, so four lanes interleave without spilling the message schedule. */
QSHA static void qsha_x4_run(uint32_t (*st)[8], const uint32_t *const *wk, int nblk) {
    __m128i S0[4], S1[4];
#pragma GCC unroll 4
    for (int l = 0; l < 4; l++) {
        __m128i t = _mm_loadu_si128((const __m128i *)&st[l][0]);           /* a b c d */
        __m128i u = _mm_loadu_si128((const __m128i *)&st[l][4]);           /* e f g h */
        t = _mm_shuffle_epi32(t, 0xB1); u = _mm_shuffle_epi32(u, 0x1B);
        S0[l] = _mm_alignr_epi8(t, u, 8);                                   /* ABEF */
        S1[l] = _mm_blend_epi16(u, t, 0xF0);                                /* CDGH */
    }
    for (int b = 0; b < nblk; b++) {
        alignas(16) __m128i I0[4], I1[4];
        const uint32_t *const *W = wk + 4 * b;
#pragma GCC unroll 4
        for (int l = 0; l < 4; l++) { I0[l] = S0[l]; I1[l] = S1[l]; }
#pragma GCC unroll 16
        for (int r = 0; r < 16; r++) {
#pragma GCC unroll 4
            for (int l = 0; l < 4; l++) {
                __m128i m = _mm_load_si128((const __m128i *)(W[l] + 4 * r));
                S1[l] = _mm_sha256rnds2_epu32(S1[l], S0[l], m);
                m = _mm_shuffle_epi32(m, 0x0E);
                S0[l] = _mm_sha256rnds2_epu32(S0[l], S1[l], m);
            }
        }
#pragma GCC unroll 4
        for (int l = 0; l < 4; l++) { S0[l] = _mm_add_epi32(S0[l], I0[l]); S1[l] = _mm_add_epi32(S1[l], I1[l]); }
    }
#pragma GCC unroll 4
    for (int l = 0; l < 4; l++) {
        __m128i t = _mm_shuffle_epi32(S0[l], 0x1B);                         /* FEBA */
        __m128i b = _mm_shuffle_epi32(S1[l], 0xB1);                         /* DCHG */
        _mm_storeu_si128((__m128i *)&st[l][0], _mm_blend_epi16(t, b, 0xF0)); /* DCBA -> a b c d */
        _mm_storeu_si128((__m128i *)&st[l][4], _mm_alignr_epi8(b, t, 8));    /* HGFE -> e f g h */
    }
}
#endif
static bool qsha_supported() {
    unsigned a, b, cc, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &cc, &d)) return false;
    return (b >> 29) & 1;                                   /* CPUID.(7,0):EBX.SHA */
}
#endif
struct Ctx {
    const digest_params_t *dp;
    std::vector<pt, qalloc64<pt> > table;   /* full-bank stride NE; final bank bounded by remaining bits, 64 B aligned */
    fe cx, cy;                      /* C = u2*R */
    uint8_t cwin[286][3];           /* CPU window patterns: the complement of the GPU's */
    int ncwin = 0;
    int cut = 137, early = 6;
    uint64_t mid_bytes = 0;         /* preimage bytes covered by dp->midstate */
    uint64_t n_epochs = 0;
    std::atomic<uint64_t> cand{0};
    std::atomic<uint32_t> hits{0};
    std::mutex io;
    qsb_hv_t hv;                    /* exact gate, used under io */
    FILE *out = nullptr;
    int nthreads = 0;
    bool vec = false;               /* 8-lane IFMA path */
    bool shani = false;             /* 4-lane SHA-NI hashing */
#if QCPU_SHANI && QSB_CPU_FIRSTHASH_SCHEDULE
    /* SHA-NI preimage schedule, precomputed once: every epoch leaves the same number of bytes (remlen)
     * after its midstate, so block 0 = those bytes ++ the pattern's first pushes (one of nvar variants)
     * and blocks 1..nblk depend on the pattern alone (their W+K is precomputed and deduplicated). */
    bool fast = false;
    int remlen = 0, nblk = 0, nvar = 0;
    std::vector<uint16_t> pvar;             /* pattern -> block-0 variant */
    std::vector<uint8_t> vblk;              /* variant block-0 bytes, 64 each (bytes 0..remlen-1 are per epoch) */
    std::vector<const uint32_t *> pwk;      /* W+K of block b of pattern p at [p * nblk + b - 1] */
    std::vector<uint32_t, qalloc64<uint32_t> > wkpool;
#endif
};

/* Build T[i][j] = (j+1) * 2^(W i) * A, threads split by window. */
static void build_table(Ctx &c, const fe &ax, const fe &ay, int nth) {
    c.table.resize(table_entries());
    std::vector<pt> base(NW);
    base[0] = {ax, ay};
    for (int i = 1; i < NW; i++) {                     /* base[i] = 2^W * base[i-1] by W doublings */
        pt q = base[i - 1];
        for (int s = 0; s < W; s++) q = pt_double(q);
        base[i] = q;
    }
    auto work = [&](int w0) {
        for (int i = w0; i < NW; i += nth) {
            pt *T = &c.table[(size_t)i * NE];
            const int entries = (int)window_entries(i);
            T[0] = base[i];
            /* 2B by doubling, then T[j] = T[j-1] + B for the rest, in rounds of doubling width */
            T[1] = pt_double(base[i]);
            int have = 2;                                /* T[0..have-1] = 1..have multiples */
            std::vector<fe> d(entries + 1), pre(entries + 1);
            std::vector<uint8_t> inf(entries + 1, 0), bad(entries + 1, 0);
            std::vector<const pt *> tp(entries + 1);
            while (have < entries) {
                int n = have; if (have + n > entries) n = entries - have;
                /* T[have+k] = T[k] + T[have-1]  ((k+1) + have = have+k+1) */
                for (int k = 0; k < n; k++) { T[have + k] = T[k]; tp[k] = &T[have - 1]; inf[k] = 0; bad[k] = 0; }
                batch_add(&T[have], tp.data(), inf.data(), bad.data(), n, d.data(), pre.data());
                /* k = have-1 adds T[have-1] to itself: equal x, so batch_add flags it; double it. */
                for (int k = 0; k < n; k++) if (bad[k]) T[have + k] = pt_double(T[have - 1]);
                have += n;
            }
        }
    };
    std::vector<std::thread> ts;
    for (int t = 0; t < nth; t++) ts.emplace_back(work, t);
    for (auto &t : ts) t.join();
}

/* Gate one recovered key (x, parity of y); true when it is an exact hit, which is then published. */
static inline void pk_block(uint8_t *pk, const fe &x3, unsigned ypar) {    /* compressed key, bytes 0..32 */
    pk[0] = (uint8_t)(0x02 | (ypar & 1));
    for (int b = 0; b < 32; b++) pk[1 + b] = (uint8_t)(x3.v[3 - (b >> 3)] >> (8 * (7 - (b & 7))));
}
static inline bool pk_prefilter(uint32_t h0) { return (h0 >> (32 - (QSB_ZEROS_N < 32 ? QSB_ZEROS_N : 32))) == 0; }
static bool gate_publish_exact(Ctx *c, const uint8_t *sk, int ri);
static bool gate_publish(Ctx *c, uint8_t *pk, const uint8_t *sk, int ri, const fe &x3, unsigned ypar) {
    pk_block(pk, x3, ypar);
    SHA256_CTX s3; SHA256_Init(&s3); SHA256_Transform(&s3, pk);
    if (!pk_prefilter(s3.h[0])) return false;
    return gate_publish_exact(c, sk, ri);
}
/* The exact OpenSSL gate and the publication, for a candidate whose key hash passed the prefilter. */
static bool gate_publish_exact(Ctx *c, const uint8_t *sk, int ri) {
    std::lock_guard<std::mutex> g(c->io);
    if (!qsb_hv_check(&c->hv, sk, ri)) return false;
    if (!c->out) { mkdir("results", 0755); c->out = fopen("results/digest_hit_cpu.txt", "a"); }
    if (c->out) {
        fprintf(c->out, "indices=%d,%d,%d,%d,%d,%d,%d,%d,%d recid=%d\n",
                sk[0], sk[1], sk[2], sk[3], sk[4], sk[5], sk[6], sk[7], sk[8], ri);
        fflush(c->out);
    }
    c->hits++;
    return true;
}

#if QCPU_VEC
struct VecBuf { fe8 *X = nullptr, *Y = nullptr, *D = nullptr, *P = nullptr; fe *qx = nullptr; uint8_t *qp = nullptr, *bad = nullptr; };
static bool vecbuf_alloc(VecBuf &v, int B) {
    const int G = B / 8; void *q[7] = {nullptr};
    const size_t sz[7] = {sizeof(fe8) * G, sizeof(fe8) * G, sizeof(fe8) * G, sizeof(fe8) * G, sizeof(fe) * 2 * (size_t)B, 2 * (size_t)B, (size_t)B};
    for (int i = 0; i < 7; i++) if (posix_memalign(&q[i], 64, sz[i])) { for (int j = 0; j < i; j++) free(q[j]); return false; }
    v.X = (fe8 *)q[0]; v.Y = (fe8 *)q[1]; v.D = (fe8 *)q[2]; v.P = (fe8 *)q[3]; v.qx = (fe *)q[4]; v.qp = (uint8_t *)q[5]; v.bad = (uint8_t *)q[6];
    return true;
}
/* The EC part of one batch on the 8-lane path: z*A for every candidate (16 table windows), then
 * both recovery ids against C. dig is candidate-major (B x NW). */
Q8T static void vec_batch(const Ctx *c, const digit_t *dig, int B, VecBuf &v) {
    const int G = B / 8;
    for (int k = 0; k < B; k++) { unsigned z = 0; for (int i = 0; i < NW; i++) z |= (dig[(size_t)k * NW + i] == 0); v.bad[k] = (uint8_t)z; }
    const pt *T0 = c->table.data();
    ec8_first(v.X, v.Y, G, [&](int h, const pt **rows) {
        for (int j = 0; j < 8; j++) { const unsigned d0 = dig[(size_t)(h * 8 + j) * NW]; rows[j] = &T0[(d0 ? d0 : 1) - 1]; } });
    for (int i = 1; i < NW; i++) {
        const pt *T = c->table.data() + (size_t)i * NE;
        ec8_window(v.X, v.Y, v.D, v.P, G, [&](int h, const pt **rows) {
            for (int j = 0; j < 8; j++) { const unsigned di = dig[(size_t)(h * 8 + j) * NW + i]; rows[j] = &T[(di ? di : 1) - 1]; } });
    }
    ec8_final(v.X, v.Y, v.D, v.P, G, c->cx, c->cy, v.qx, v.qp);
}
#endif

#if QCPU_SHANI && QSB_CPU_FIRSTHASH_SCHEDULE
static inline uint32_t be32(const uint8_t *b) { return (uint32_t)b[0] << 24 | (uint32_t)b[1] << 16 | (uint32_t)b[2] << 8 | b[3]; }
static void prep_fast(Ctx *c) {
    const digest_params_t *dp = c->dp;
    const size_t prl = dp->prefix_remainder_len, pl = (size_t)(c->cut - c->early) * SIG_PUSH_SIZE;
    const size_t wlen = (size_t)(dp->n - c->cut - 3) * SIG_PUSH_SIZE, tl = dp->tail_section_len, sl = dp->tx_suffix_len;
    const size_t remlen = (prl + pl) % 64;
    const uint64_t tbits = (c->mid_bytes + prl + pl + wlen + tl + sl) * 8;
    const int nb = (int)((remlen + wlen + tl + sl + 9 + 63) / 64);
    if (nb < 2 || nb > 16 || c->ncwin < 1) return;
    const size_t mb = (size_t)nb * 64;
    std::vector<uint8_t> msg((size_t)c->ncwin * mb, 0);
    for (int p = 0; p < c->ncwin; p++) {                 /* the same bytes the per-candidate path assembles */
        uint8_t *m = &msg[(size_t)p * mb]; size_t o = remlen;
        const uint8_t *w3 = c->cwin[p];
        for (int i = c->cut; i < (int)dp->n; i++) {
            if (i == w3[0] || i == w3[1] || i == w3[2]) continue;
            memcpy(m + o, dp->dummy_sigs + (size_t)i * SIG_PUSH_SIZE, SIG_PUSH_SIZE); o += SIG_PUSH_SIZE;
        }
        memcpy(m + o, dp->tail_section, tl); o += tl;
        memcpy(m + o, dp->tx_suffix, sl); o += sl;
        m[o++] = 0x80;
        for (int b = 0; b < 8; b++) m[mb - 1 - b] = (uint8_t)(tbits >> (8 * b));
    }
    c->pvar.assign(c->ncwin, 0); c->vblk.clear(); int nvar = 0;
    for (int p = 0; p < c->ncwin; p++) {
        const uint8_t *m = &msg[(size_t)p * mb]; int v = 0;
        for (; v < nvar; v++) if (!memcmp(&c->vblk[(size_t)v * 64 + remlen], m + remlen, 64 - remlen)) break;
        if (v == nvar) { c->vblk.insert(c->vblk.end(), m, m + 64); nvar++; }
        c->pvar[p] = (uint16_t)v;
    }
    const int nblk = nb - 1;
    std::vector<uint32_t> uniq; std::vector<int> idx((size_t)c->ncwin * nblk);
    for (int p = 0; p < c->ncwin; p++)
        for (int b = 1; b < nb; b++) {
            alignas(16) uint32_t w[16], wk[64];
            for (int i = 0; i < 16; i++) w[i] = be32(&msg[(size_t)p * mb + (size_t)b * 64 + 4 * i]);
            qsha_sched(wk, w);
            size_t u = 0, nu = uniq.size() / 64;
            for (; u < nu; u++) if (!memcmp(&uniq[u * 64], wk, 256)) break;
            if (u == nu) uniq.insert(uniq.end(), wk, wk + 64);
            idx[(size_t)p * nblk + b - 1] = (int)u;
        }
    c->wkpool.assign(uniq.begin(), uniq.end());
    c->pwk.resize(idx.size());
    for (size_t i = 0; i < idx.size(); i++) c->pwk[i] = c->wkpool.data() + (size_t)idx[i] * 64;
    c->remlen = (int)remlen; c->nblk = nblk; c->nvar = nvar; c->fast = true;
}
#endif
static void worker(Ctx *c, int tid) {
#ifdef SCHED_IDLE
    struct sched_param sp; sp.sched_priority = 0; sched_setscheduler(0, SCHED_IDLE, &sp);
#endif
    const digest_params_t *dp = c->dp;
    const int B = QSB_CPU_BATCH;
    std::vector<pt> acc(B); std::vector<fe> d(2 * B), pre(2 * B);
    std::vector<uint8_t> inf(B), bad(B); std::vector<const pt *> tp(B);
    std::vector<digit_t> dig((size_t)B * NW);
    std::vector<uint8_t> skips((size_t)B * 9);
    uint8_t pk[64]; memset(pk, 0, 64); pk[33] = 0x80; pk[62] = 0x01; pk[63] = 0x08;   /* 264 bits */
    uint8_t blk2[64]; memset(blk2, 0, 64); blk2[32] = 0x80; blk2[62] = 0x01;          /* 256 bits */
    uint64_t epoch = (uint64_t)tid;
    int wi = c->ncwin;
    SHA256_CTX ectx; uint8_t early[16];
    std::vector<uint8_t> pbuf((size_t)dp->n * SIG_PUSH_SIZE + 64);
    auto put_digits = [&](int kk, const uint32_t *h) {    /* W-bit windows of z = h[0] (MSW) .. h[7] */
        uint64_t zl[4];
        for (int i = 0; i < 4; i++) zl[i] = ((uint64_t)h[6 - 2 * i] << 32) | h[7 - 2 * i];
        for (int i = 0; i < NW; i++) {
            int bit = i * W, li = bit >> 6, sh = bit & 63;
            uint64_t v = zl[li] >> sh;
            if (sh + W > 64 && li < 3) v |= zl[li + 1] << (64 - sh);
            dig[(size_t)kk * NW + i] = (digit_t)(v & NE);
        }
    };
#if QCPU_SHANI
    /* 4-lane path: each lane holds one candidate's chaining state (its epoch's) and the padded rest of
     * its message: the epoch's buffered bytes, the kept window pushes, tail section, suffix. */
    const size_t tl = dp->tail_section_len, sl = dp->tx_suffix_len;
    alignas(16) uint32_t lst[4][8]; alignas(16) uint8_t lmsg[4][16 * 64]; const uint8_t *lp[4];
    uint32_t est[8]; uint8_t erem[64]; size_t remlen = 0; int nb = 0; uint64_t tbits = 0;
    bool shani = c->shani;
#if QSB_CPU_FIRSTHASH_SCHEDULE
    bool fast = shani && c->fast;
    const int nblk = c->nblk;
    std::vector<uint32_t> vst((size_t)(c->nvar > 0 ? c->nvar : 1) * 8);
    alignas(16) uint32_t wkv[4][64];
    const uint32_t *lwk[4 * 16];
#endif
#endif
#if QCPU_VEC
    VecBuf vb;
    if (c->vec && !vecbuf_alloc(vb, B)) vb = VecBuf();
#endif
    for (;;) {
        int k = 0;
        while (k < B) {
            if (wi == c->ncwin) {                       /* next epoch: hash its fixed prefix once */
                if (epoch >= c->n_epochs) return;
                qsb_host_unrank(epoch, c->cut, c->early, early);
                SHA256_Init(&ectx);
                for (int i = 0; i < 8; i++) ectx.h[i] = dp->midstate[i];
                const uint64_t bits = c->mid_bytes * 8;
                ectx.Nl = (SHA_LONG)bits; ectx.Nh = (SHA_LONG)(bits >> 32); ectx.num = 0;
                if (dp->prefix_remainder_len) SHA256_Update(&ectx, dp->prefix_remainder, dp->prefix_remainder_len);
                size_t pl = 0; int e = 0;
                for (int i = 0; i < c->cut; i++) {
                    if (e < c->early && early[e] == i) { e++; continue; }
                    memcpy(pbuf.data() + pl, dp->dummy_sigs + (size_t)i * SIG_PUSH_SIZE, SIG_PUSH_SIZE); pl += SIG_PUSH_SIZE;
                }
                SHA256_Update(&ectx, pbuf.data(), pl);
#if QCPU_SHANI
                if (shani) {
                    const size_t prl = dp->prefix_remainder_len, wlen = (size_t)(dp->n - c->cut - 3) * SIG_PUSH_SIZE;
                    remlen = (prl + pl) % 64;
                    for (size_t q = 0; q < remlen; q++) { const size_t pos = prl + pl - remlen + q; erem[q] = pos < prl ? dp->prefix_remainder[pos] : pbuf[pos - prl]; }
                    for (int i = 0; i < 8; i++) est[i] = (uint32_t)ectx.h[i];
                    tbits = (c->mid_bytes + prl + pl + wlen + tl + sl) * 8;
                    nb = (int)((remlen + wlen + tl + sl + 9 + 63) / 64);
                    if (ectx.num != remlen || nb > 16) shani = false;   /* unexpected shape: stay on OpenSSL (decided at the first epoch, k = 0) */
#if QSB_CPU_FIRSTHASH_SCHEDULE
                    if (!shani || remlen != (size_t)c->remlen || nb != nblk + 1) fast = false;
                    if (fast) {                             /* the epoch's block-0 states, four variants at a time */
                        for (int v0 = 0; v0 < c->nvar; v0 += 4) {
                            alignas(16) uint32_t vs[4][8]; const uint32_t *vp[4];
                            for (int l = 0; l < 4; l++) {
                                const int v = v0 + l < c->nvar ? v0 + l : c->nvar - 1;
                                uint8_t blk[64]; memcpy(blk, &c->vblk[(size_t)v * 64], 64); memcpy(blk, erem, remlen);
                                alignas(16) uint32_t w[16]; for (int i = 0; i < 16; i++) w[i] = be32(blk + 4 * i);
                                qsha_sched(wkv[l], w); vp[l] = wkv[l]; memcpy(vs[l], est, 32);
                            }
                            qsha_x4_run(vs, vp, 1);
                            for (int l = 0; l < 4 && v0 + l < c->nvar; l++) memcpy(&vst[(size_t)(v0 + l) * 8], vs[l], 32);
                        }
                    }
#endif
                }
#endif
                epoch += (uint64_t)c->nthreads; wi = 0;
            }
#if QCPU_SHANI && QSB_CPU_FIRSTHASH_SCHEDULE
            const int pat = wi;
#endif
            const uint8_t *w3 = c->cwin[wi++];
#if QCPU_SHANI
#if QSB_CPU_FIRSTHASH_SCHEDULE
            if (fast) {                                 /* precomputed schedule: rounds only */
                const int j = k & 3;
                memcpy(lst[j], &vst[(size_t)c->pvar[pat] * 8], 32);
                for (int b = 0; b < nblk; b++) lwk[4 * b + j] = c->pwk[(size_t)pat * nblk + b];
                uint8_t *sk = &skips[(size_t)k * 9];
                for (int q = 0; q < 6; q++) sk[q] = early[q];
                sk[6] = w3[0]; sk[7] = w3[1]; sk[8] = w3[2];
                inf[k] = 1; bad[k] = 0;
                if (j == 3) {
                    qsha_x4_run(lst, lwk, nblk);
                    alignas(16) uint8_t b2[4][64]; alignas(16) uint32_t s2[4][8];
                    for (int l = 0; l < 4; l++) {
                        memcpy(b2[l], blk2, 64);
                        for (int w = 0; w < 8; w++) { b2[l][4 * w] = (uint8_t)(lst[l][w] >> 24); b2[l][4 * w + 1] = (uint8_t)(lst[l][w] >> 16); b2[l][4 * w + 2] = (uint8_t)(lst[l][w] >> 8); b2[l][4 * w + 3] = (uint8_t)lst[l][w]; }
                        memcpy(s2[l], qsha_iv, 32); lp[l] = b2[l];
                    }
                    qsha_x4(s2, lp);
                    for (int l = 0; l < 4; l++) put_digits(k - 3 + l, s2[l]);
                }
                k++;
                continue;
            }
#endif
            if (shani) {
                const int j = k & 3;
                uint8_t *m = lmsg[j]; size_t o = remlen;
                memcpy(m, erem, remlen);
                for (int i = c->cut; i < (int)dp->n; i++) {
                    if (i == w3[0] || i == w3[1] || i == w3[2]) continue;
                    memcpy(m + o, dp->dummy_sigs + (size_t)i * SIG_PUSH_SIZE, SIG_PUSH_SIZE); o += SIG_PUSH_SIZE;
                }
                memcpy(m + o, dp->tail_section, tl); o += tl;
                memcpy(m + o, dp->tx_suffix, sl); o += sl;
                m[o++] = 0x80; memset(m + o, 0, (size_t)nb * 64 - o);
                for (int b = 0; b < 8; b++) m[(size_t)nb * 64 - 1 - b] = (uint8_t)(tbits >> (8 * b));
                memcpy(lst[j], est, 32);
                uint8_t *sk = &skips[(size_t)k * 9];
                for (int q = 0; q < 6; q++) sk[q] = early[q];
                sk[6] = w3[0]; sk[7] = w3[1]; sk[8] = w3[2];
                inf[k] = 1; bad[k] = 0;
                if (j == 3) {                               /* four candidates ready: both SHA-256 passes */
                    for (int bl = 0; bl < nb; bl++) { for (int l = 0; l < 4; l++) lp[l] = lmsg[l] + (size_t)bl * 64; qsha_x4(lst, lp); }
                    alignas(16) uint8_t b2[4][64]; alignas(16) uint32_t s2[4][8];
                    for (int l = 0; l < 4; l++) {
                        memcpy(b2[l], blk2, 64);
                        for (int w = 0; w < 8; w++) { b2[l][4 * w] = (uint8_t)(lst[l][w] >> 24); b2[l][4 * w + 1] = (uint8_t)(lst[l][w] >> 16); b2[l][4 * w + 2] = (uint8_t)(lst[l][w] >> 8); b2[l][4 * w + 3] = (uint8_t)lst[l][w]; }
                        memcpy(s2[l], qsha_iv, 32); lp[l] = b2[l];
                    }
                    qsha_x4(s2, lp);
                    for (int l = 0; l < 4; l++) put_digits(k - 3 + l, s2[l]);
                }
                k++;
                continue;
            }
#endif
            SHA256_CTX s = ectx;
            uint8_t wbuf[16 * SIG_PUSH_SIZE]; size_t wl = 0;
            for (int i = c->cut; i < (int)dp->n; i++) {
                if (i == w3[0] || i == w3[1] || i == w3[2]) continue;
                memcpy(wbuf + wl, dp->dummy_sigs + (size_t)i * SIG_PUSH_SIZE, SIG_PUSH_SIZE); wl += SIG_PUSH_SIZE;
            }
            SHA256_Update(&s, wbuf, wl);
            SHA256_Update(&s, dp->tail_section, dp->tail_section_len);
            SHA256_Update(&s, dp->tx_suffix, dp->tx_suffix_len);
            SHA256_Final(blk2, &s);                     /* first digest into the second block */
            SHA256_CTX s2; SHA256_Init(&s2); SHA256_Transform(&s2, blk2);
            { uint32_t hw[8]; for (int i = 0; i < 8; i++) hw[i] = (uint32_t)s2.h[i]; put_digits(k, hw); }
            uint8_t *sk = &skips[(size_t)k * 9];
            for (int j = 0; j < 6; j++) sk[j] = early[j];
            sk[6] = w3[0]; sk[7] = w3[1]; sk[8] = w3[2];
            inf[k] = 1; bad[k] = 0;
            k++;
        }
#if QCPU_VEC
        if (vb.X) {
            vec_batch(c, dig.data(), B, vb);
#if QCPU_SHANI
            if (c->shani) {                             /* key hashes 4 at a time: 2 candidates x 2 recids */
                for (int kk = 0; kk < B; kk += 2) {
                    alignas(16) uint8_t pkb[4][64]; alignas(16) uint32_t hs[4][8]; const uint8_t *pp[4];
                    for (int l = 0; l < 4; l++) {
                        const size_t q = (size_t)(kk + (l >> 1)) * 2 + (l & 1);
                        memcpy(pkb[l], pk, 64); pk_block(pkb[l], vb.qx[q], vb.qp[q]);
                        memcpy(hs[l], qsha_iv, 32); pp[l] = pkb[l];
                    }
                    qsha_x4(hs, pp);
                    for (int q2 = 0; q2 < 2; q2++) {
                        const int q = kk + q2;
                        if (vb.bad[q]) continue;
                        for (int ri = 0; ri < 2; ri++)
                            if (pk_prefilter(hs[2 * q2 + ri][0]) && gate_publish_exact(c, &skips[(size_t)q * 9], ri)) break;   /* one recid per candidate */
                    }
                }
            } else
#endif
            for (int kk = 0; kk < B; kk++) {
                if (vb.bad[kk]) continue;
                const uint8_t *sk = &skips[(size_t)kk * 9];
                for (int ri = 0; ri < 2; ri++)
                    if (gate_publish(c, pk, sk, ri, vb.qx[(size_t)kk * 2 + ri], vb.qp[(size_t)kk * 2 + ri])) break;   /* one recid per candidate */
            }
            c->cand += B;
            continue;
        }
#endif
        for (int i = 0; i < NW; i++) {
            const pt *T = &c->table[(size_t)i * NE];
            for (int kk = 0; kk < B; kk++) { digit_t v = dig[(size_t)kk * NW + i]; tp[kk] = v ? &T[v - 1] : nullptr; }
            batch_add(acc.data(), tp.data(), inf.data(), bad.data(), B, d.data(), pre.data());
        }
        /* Both recids share the denominator x_C - x_P. */
        fe run = {{1, 0, 0, 0}};
        for (int kk = 0; kk < B; kk++) {
            if (bad[kk] || inf[kk]) { pre[kk] = run; continue; }
            fe_sub(d[kk], c->cx, acc[kk].x);
            if (fe_is_zero(d[kk])) { bad[kk] = 1; pre[kk] = run; continue; }
            pre[kk] = run; fe_mul(run, run, d[kk]);
        }
        fe inv; fe_inv(inv, run);
        for (int kk = B - 1; kk >= 0; kk--) {
            if (bad[kk] || inf[kk]) continue;
            fe dinv; fe_mul(dinv, inv, pre[kk]); fe_mul(inv, inv, d[kk]);
            for (int ri = 0; ri < 2; ri++) {
                fe cy = c->cy; if (ri) { fe z0 = {{0, 0, 0, 0}}; fe_sub(cy, z0, cy); }   /* recid 1: -C */
                fe lam, t, x3, y3;
                fe_sub(t, cy, acc[kk].y); fe_mul(lam, t, dinv);
                fe_sqr(x3, lam); fe_sub(x3, x3, acc[kk].x); fe_sub(x3, x3, c->cx);
                fe_sub(t, acc[kk].x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, acc[kk].y);
                if (gate_publish(c, pk, &skips[(size_t)kk * 9], ri, x3, (unsigned)(y3.v[0] & 1))) break;   /* one recid per candidate, like the GPU gate */
            }
        }
        c->cand += B;
    }
}

static Ctx *g_ctx = nullptr;
/* win3: the GPU's 128 window patterns (actual push indices, ascending). */
static void start(const digest_params_t *dp, const uint8_t win3[][3], int nwin, int cut, int early) {
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
#ifdef CPU_COUNT
    { cpu_set_t cs; CPU_ZERO(&cs); if (sched_getaffinity(0, sizeof cs, &cs) == 0) ncpu = CPU_COUNT(&cs); }
#endif
    if (FILE *q = fopen("/sys/fs/cgroup/cpu.max", "r")) {
        char quota[32] = {0}; long period = 0;
        if (fscanf(q, "%31s %ld", quota, &period) == 2 && strcmp(quota, "max") != 0 && period > 0) {
            long lim = (atol(quota) + period - 1) / period; if (lim > 0 && lim < ncpu) ncpu = lim;
        }
        fclose(q);
    }
    if (FILE *q = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {      /* cgroup v1 */
        long quota = -1, period = 0; if (fscanf(q, "%ld", &quota) != 1) quota = -1; fclose(q);
        if (FILE *pf = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) { if (fscanf(pf, "%ld", &period) != 1) period = 0; fclose(pf); }
        if (quota > 0 && period > 0) { long lim = (quota + period - 1) / period; if (lim > 0 && lim < ncpu) ncpu = lim; }
    }
#ifdef QSB_CPU_THREADS
    int nth = QSB_CPU_THREADS;
#else
    int nth = (int)ncpu - QSB_CPU_RESERVE;
#endif
    if (const char *e = getenv("QSB_CPU_THREADS_ENV")) nth = atoi(e);   /* dev override */
    if (nth < 1 || dp->n != 150 || cut != 137 || early != 6) { printf("  CPU co-grind: off (%d threads)\n", nth); return; }
    Ctx *c = new Ctx(); c->dp = dp; c->nthreads = nth; c->cut = cut; c->early = early;
#if QCPU_VEC
    __builtin_cpu_init();
    c->vec = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512ifma") && !getenv("QSB_CPU_NOVEC");
#endif
#if QCPU_SHANI
    c->shani = qsha_supported() && !getenv("QSB_CPU_NOSHANI");
#endif
    /* CPU window patterns: every 3-subset of {cut..n-1} not used by the GPU. */
    for (int a = cut; a < (int)dp->n; a++) for (int b = a + 1; b < (int)dp->n; b++) for (int d3 = b + 1; d3 < (int)dp->n; d3++) {
        int used = 0;
        for (int i = 0; i < nwin; i++) if (win3[i][0] == a && win3[i][1] == b && win3[i][2] == d3) { used = 1; break; }
        if (!used) { c->cwin[c->ncwin][0] = (uint8_t)a; c->cwin[c->ncwin][1] = (uint8_t)b; c->cwin[c->ncwin][2] = (uint8_t)d3; c->ncwin++; }
    }
    const uint64_t unpadded = (uint64_t)dp->prefix_remainder_len + (uint64_t)(dp->n - dp->t) * SIG_PUSH_SIZE +
                              dp->tail_section_len + dp->tx_suffix_len;
    if (dp->t != 9 || dp->total_preimage_len < unpadded || ((dp->total_preimage_len - unpadded) % 64) != 0 || c->ncwin < 1) {
        printf("  CPU co-grind: off (unexpected problem shape)\n"); delete c; return;
    }
    c->mid_bytes = dp->total_preimage_len - unpadded;
    c->n_epochs = binom_u64(cut, early);
#if QCPU_SHANI && QSB_CPU_FIRSTHASH_SCHEDULE
    if (c->shani) { try { prep_fast(c); } catch (...) { c->fast = false; } }
#endif
    if (!qsb_hv_init(&c->hv, dp, (const uint8_t (*)[QSB_SE_TWIN])win3, cut, early)) { printf("  CPU co-grind: off (gate)\n"); delete c; return; }
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *bctx = BN_CTX_new(); BIGNUM *nri = BN_new(), *ax = BN_new(), *ay = BN_new();
    EC_POINT *A = EC_POINT_new(grp);
    BN_lebin2bn(dp->neg_r_inv, 32, nri);
    if (!EC_POINT_mul(grp, A, nri, NULL, NULL, bctx) ||
        !EC_POINT_get_affine_coordinates_GFp(grp, A, ax, ay, bctx)) { printf("  CPU co-grind: off (A)\n"); delete c; return; }
    fe fax, fay; fe_from_bn(fax, ax); fe_from_bn(fay, ay);
    fe_from_le32(c->cx, dp->u2r_x); fe_from_le32(c->cy, dp->u2r_y);
    EC_POINT_free(A); BN_free(nri); BN_free(ax); BN_free(ay); BN_CTX_free(bctx); EC_GROUP_free(grp);
    std::thread([c, fax, fay, nth]() {
#ifdef SCHED_IDLE
        struct sched_param sp; sp.sched_priority = 0; sched_setscheduler(0, SCHED_IDLE, &sp);
#endif
        build_table(*c, fax, fay, nth);
        for (int t = 0; t < nth; t++) std::thread(worker, c, t).detach();
    }).detach();
    g_ctx = c;
    printf("  CPU co-grind: %d threads (of %ld CPUs), %s, %s, %d window patterns per epoch disjoint from the GPU's %d\n",
           nth, ncpu, c->vec ? "8-lane IFMA" : "scalar", c->shani ? "4-lane SHA-NI" : "OpenSSL SHA-256", c->ncwin, nwin);
    fflush(stdout);
}
static uint64_t candidates() { return g_ctx ? g_ctx->cand.load() : 0; }
static uint32_t hits() { return g_ctx ? g_ctx->hits.load() : 0; }
}  // namespace qcpu
