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
#include <pthread.h>
#include <sys/syscall.h>
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
#ifndef QSB_CPU_BATCH
#define QSB_CPU_BATCH 1024
#endif
/* Wide host table: z*A takes L lookups of signed ~257/L-bit digits (L-1 batch-affine additions).
 * The fewest lookups whose table fits the memory budget are chosen at start-up; 16 windows (34 MiB)
 * is the floor. The budget is a third of the smaller of MemAvailable and the cgroup headroom,
 * capped at QSB_CPU_TABLE_CAP_MB. */
#ifndef QSB_CPU_TABLE_CAP_MB
#define QSB_CPU_TABLE_CAP_MB 6144         /* 11 lookups (4.3 GiB with the folded copies) at most */
#endif
#ifndef QSB_CPU_LMIN
#define QSB_CPU_LMIN 10
#endif

#include <sys/mman.h>
#include <spawn.h>
#include <sys/wait.h>
#include <signal.h>
#include <string>
#include <time.h>
namespace qcpu {
static const int NWMAX = 16;
/* Memory probe (see mem_probe below). The co-grinder re-executes this binary with QSB_CPU_PROBE_MB set; this static
 * initializer then runs before main(), touches no CUDA state, takes the highest OOM score, faults in that many MiB and
 * exits. If a memory limit the process cannot see is below that, only this child is killed. */
static const int g_probe_child = [] {
    const char *e = getenv("QSB_CPU_PROBE_MB");
    if (!e) return 0;
    if (FILE *f = fopen("/proc/self/oom_score_adj", "w")) { fputs("1000", f); fclose(f); }
    const size_t n = (size_t)atoll(e) << 20;
    void *m = n ? mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0) : MAP_FAILED;
    if (m == MAP_FAILED) _exit(3);
#ifdef MADV_HUGEPAGE
    madvise(m, n, MADV_HUGEPAGE);
#endif
    const char *kf = getenv("QSB_CPU_PROBE_SELFKILL");                /* dev: behave like an OOM kill halfway */
    for (size_t o = 0; o < n; o += 4096) { ((volatile char *)m)[o] = 1; if (kf && o == (n / 2 & ~(size_t)4095)) raise(SIGKILL); }
    _exit(0);
}();
#ifdef CPU_SET
/* The process's CPU set as it was before main(): a GPU host path may pin the main thread to one core
 * later, and the co-grinder must size and place itself from the whole set, not from that core. */
#ifndef QSB_CPU_PF
#define QSB_CPU_PF 3   /* table rows are prefetched this many 8-lane groups ahead: 3 measured fastest on Zen 4 SMT
                        * once the backward pass stopped re-reading rows (newjordan, 2a1f43c5: 2: 1.610, 3: 1.615,
                        * 4: 1.586, 6: 1.589, 8: 1.575 M/cpu-s) */
#endif
static cpu_set_t g_mask0;
static const bool g_mask0_ok = [] { CPU_ZERO(&g_mask0); return sched_getaffinity(0, sizeof g_mask0, &g_mask0) == 0; }();
#endif
static_assert(QSB_CPU_BATCH % 32 == 0, "the 8-lane path needs groups of 8 in fours; keyhash16 needs 2B % 16 == 0");
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
/* Host-only: scalar inversion from the eligible completed 97f347a8 route.
 * Pinning cc75e3b's cross-lane batch tree is the structural reference. */
#ifndef QSB_CPU_SCALAR_INV
#define QSB_CPU_SCALAR_INV 1
#endif
#if QSB_CPU_SCALAR_INV
#include "CpuSafegcd.h"
#endif
#ifndef QSB_CPU_SCA_SINGLE_READ
#define QSB_CPU_SCA_SINGLE_READ 1
#endif
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

/* ---- Scalar EC path: libsecp256k1's 5x52 field arithmetic (field_5x52_int128_impl.h, MIT; notice in
 * COPYING-secp256k1), batch-affine additions over four interleaved Montgomery chains. It runs where the
 * host has no AVX-512 IFMA, and on the second thread of each core in the SMT hybrid (integer pipes). */
struct fe52 { uint64_t n[5]; };   /* magnitude as in libsecp256k1; every stored value has magnitude 1 */
static inline __attribute__((always_inline)) void fe52_mul_inner(uint64_t *r, const uint64_t *a, const uint64_t * __restrict__ b) {
    u128 c, d;
    uint64_t t3, t4, tx, u0;
    uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    const uint64_t M = 0xFFFFFFFFFFFFFULL, R = 0x1000003D10ULL;
    d = (u128)(a0) * (b[3]);
    d += (u128)(a1) * (b[2]);
    d += (u128)(a2) * (b[1]);
    d += (u128)(a3) * (b[0]);
    c = (u128)(a4) * (b[4]);
    d += (u128)(R) * ((uint64_t)c); c >>= 64;
    t3 = (uint64_t)d & M; d >>= 52;
    d += (u128)(a0) * (b[4]);
    d += (u128)(a1) * (b[3]);
    d += (u128)(a2) * (b[2]);
    d += (u128)(a3) * (b[1]);
    d += (u128)(a4) * (b[0]);
    d += (u128)(R << 12) * ((uint64_t)c);
    t4 = (uint64_t)d & M; d >>= 52;
    tx = (t4 >> 48); t4 &= (M >> 4);
    c = (u128)(a0) * (b[0]);
    d += (u128)(a1) * (b[4]);
    d += (u128)(a2) * (b[3]);
    d += (u128)(a3) * (b[2]);
    d += (u128)(a4) * (b[1]);
    u0 = (uint64_t)d & M; d >>= 52;
    u0 = (u0 << 4) | tx;
    c += (u128)(u0) * (R >> 4);
    r[0] = (uint64_t)c & M; c >>= 52;
    c += (u128)(a0) * (b[1]);
    c += (u128)(a1) * (b[0]);
    d += (u128)(a2) * (b[4]);
    d += (u128)(a3) * (b[3]);
    d += (u128)(a4) * (b[2]);
    c += (u128)((uint64_t)d & M) * (R); d >>= 52;
    r[1] = (uint64_t)c & M; c >>= 52;
    c += (u128)(a0) * (b[2]);
    c += (u128)(a1) * (b[1]);
    c += (u128)(a2) * (b[0]);
    d += (u128)(a3) * (b[4]);
    d += (u128)(a4) * (b[3]);
    c += (u128)(R) * ((uint64_t)d); d >>= 64;
    r[2] = (uint64_t)c & M; c >>= 52;
    c += (u128)(R << 12) * ((uint64_t)d);
    c += (t3);
    r[3] = (uint64_t)c & M; c >>= 52;
    r[4] = (uint64_t)c + t4;
}

static inline __attribute__((always_inline)) void fe52_sqr_inner(uint64_t *r, const uint64_t *a) {
    u128 c, d;
    uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    uint64_t t3, t4, tx, u0;
    const uint64_t M = 0xFFFFFFFFFFFFFULL, R = 0x1000003D10ULL;
    d = (u128)(a0*2) * (a3);
    d += (u128)(a1*2) * (a2);
    c = (u128)(a4) * (a4);
    d += (u128)(R) * ((uint64_t)c); c >>= 64;
    t3 = (uint64_t)d & M; d >>= 52;
    a4 *= 2;
    d += (u128)(a0) * (a4);
    d += (u128)(a1*2) * (a3);
    d += (u128)(a2) * (a2);
    d += (u128)(R << 12) * ((uint64_t)c);
    t4 = (uint64_t)d & M; d >>= 52;
    tx = (t4 >> 48); t4 &= (M >> 4);
    c = (u128)(a0) * (a0);
    d += (u128)(a1) * (a4);
    d += (u128)(a2*2) * (a3);
    u0 = (uint64_t)d & M; d >>= 52;
    u0 = (u0 << 4) | tx;
    c += (u128)(u0) * (R >> 4);
    r[0] = (uint64_t)c & M; c >>= 52;
    a0 *= 2;
    c += (u128)(a0) * (a1);
    d += (u128)(a2) * (a4);
    d += (u128)(a3) * (a3);
    c += (u128)((uint64_t)d & M) * (R); d >>= 52;
    r[1] = (uint64_t)c & M; c >>= 52;
    c += (u128)(a0) * (a2);
    c += (u128)(a1) * (a1);
    d += (u128)(a3) * (a4);
    c += (u128)(R) * ((uint64_t)d); d >>= 64;
    r[2] = (uint64_t)c & M; c >>= 52;
    c += (u128)(R << 12) * ((uint64_t)d);
    c += (t3);
    r[3] = (uint64_t)c & M; c >>= 52;
    r[4] = (uint64_t)c + t4;
}

static inline void f52_mul(fe52 &r, const fe52 &a, const fe52 &b) { uint64_t t[5]; fe52_mul_inner(t, a.n, b.n); memcpy(r.n, t, 40); }
static inline void f52_sqr(fe52 &r, const fe52 &a) { uint64_t t[5]; fe52_sqr_inner(t, a.n); memcpy(r.n, t, 40); }
static inline void f52_from(fe52 &r, const fe &a) {         /* canonical 4x64 -> 5x52 */
    const uint64_t M = 0xFFFFFFFFFFFFFULL;
    r.n[0] = a.v[0] & M; r.n[1] = (a.v[0] >> 52 | a.v[1] << 12) & M; r.n[2] = (a.v[1] >> 40 | a.v[2] << 24) & M;
    r.n[3] = (a.v[2] >> 28 | a.v[3] << 36) & M; r.n[4] = a.v[3] >> 16;
}
/* r = a - b for b of magnitude <= 1: a + 4p - b (magnitude of a plus 2) */
static inline void f52_sub(fe52 &r, const fe52 &a, const fe52 &b) {
    r.n[0] = a.n[0] + 0xFFFFEFFFFFC2FULL * 4 - b.n[0]; r.n[1] = a.n[1] + 0xFFFFFFFFFFFFFULL * 4 - b.n[1];
    r.n[2] = a.n[2] + 0xFFFFFFFFFFFFFULL * 4 - b.n[2]; r.n[3] = a.n[3] + 0xFFFFFFFFFFFFFULL * 4 - b.n[3];
    r.n[4] = a.n[4] + 0x0FFFFFFFFFFFFULL * 4 - b.n[4];
}
static inline void f52_nweak(fe52 &r) {                     /* libsecp256k1 normalize_weak: magnitude 1 */
    uint64_t t0 = r.n[0], t1 = r.n[1], t2 = r.n[2], t3 = r.n[3], t4 = r.n[4];
    uint64_t x = t4 >> 48; t4 &= 0x0FFFFFFFFFFFFULL;
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52); t0 &= 0xFFFFFFFFFFFFFULL;
    t2 += (t1 >> 52); t1 &= 0xFFFFFFFFFFFFFULL;
    t3 += (t2 >> 52); t2 &= 0xFFFFFFFFFFFFFULL;
    t4 += (t3 >> 52); t3 &= 0xFFFFFFFFFFFFFULL;
    r.n[0] = t0; r.n[1] = t1; r.n[2] = t2; r.n[3] = t3; r.n[4] = t4;
}
static inline void f52_words(uint64_t w[4], const fe52 &a) { /* libsecp256k1 normalize, then 4x64 */
    uint64_t t0 = a.n[0], t1 = a.n[1], t2 = a.n[2], t3 = a.n[3], t4 = a.n[4], m;
    uint64_t x = t4 >> 48; t4 &= 0x0FFFFFFFFFFFFULL;
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52); t0 &= 0xFFFFFFFFFFFFFULL;
    t2 += (t1 >> 52); t1 &= 0xFFFFFFFFFFFFFULL; m = t1;
    t3 += (t2 >> 52); t2 &= 0xFFFFFFFFFFFFFULL; m &= t2;
    t4 += (t3 >> 52); t3 &= 0xFFFFFFFFFFFFFULL; m &= t3;
    x = (t4 >> 48) | ((t4 == 0x0FFFFFFFFFFFFULL) & (m == 0xFFFFFFFFFFFFFULL) & (t0 >= 0xFFFFEFFFFFC2FULL));
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52); t0 &= 0xFFFFFFFFFFFFFULL;
    t2 += (t1 >> 52); t1 &= 0xFFFFFFFFFFFFFULL;
    t3 += (t2 >> 52); t2 &= 0xFFFFFFFFFFFFFULL;
    t4 += (t3 >> 52); t3 &= 0xFFFFFFFFFFFFFULL;
    t4 &= 0x0FFFFFFFFFFFFULL;
    w[0] = t0 | t1 << 52; w[1] = t1 >> 12 | t2 << 40; w[2] = t2 >> 24 | t3 << 28; w[3] = t3 >> 36 | t4 << 16;
}
static void f52_inv(fe52 &x) {                              /* x^(p-2), libsecp256k1's addition chain */
    fe52 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;
#define SQ52(dst, src, n) do { dst = src; for (int i = 0; i < (n); i++) f52_sqr(dst, dst); } while (0)
    SQ52(x2, x, 1); f52_mul(x2, x2, x);
    SQ52(x3, x2, 1); f52_mul(x3, x3, x);
    SQ52(x6, x3, 3); f52_mul(x6, x6, x3);
    SQ52(x9, x6, 3); f52_mul(x9, x9, x3);
    SQ52(x11, x9, 2); f52_mul(x11, x11, x2);
    SQ52(x22, x11, 11); f52_mul(x22, x22, x11);
    SQ52(x44, x22, 22); f52_mul(x44, x44, x22);
    SQ52(x88, x44, 44); f52_mul(x88, x88, x44);
    SQ52(x176, x88, 88); f52_mul(x176, x176, x88);
    SQ52(x220, x176, 44); f52_mul(x220, x220, x44);
    SQ52(x223, x220, 3); f52_mul(x223, x223, x3);
    SQ52(t, x223, 23); f52_mul(t, t, x22);
    SQ52(t, t, 5); f52_mul(t, t, x);
    SQ52(t, t, 3); f52_mul(t, t, x2);
    SQ52(t, t, 2); f52_mul(x, t, x);
#undef SQ52
}
static void f52_inv4(fe52 *x) {                             /* four chain products, one exponentiation */
    fe52 a01, a23, a, i01, i23;
    f52_mul(a01, x[0], x[1]); f52_mul(a23, x[2], x[3]); f52_mul(a, a01, a23);
#if QSB_CPU_SCALAR_INV
    fe ca, ia; f52_words(ca.v, a); fe_inv_var(ia, ca); f52_from(a, ia);
#else
    f52_inv(a);
#endif
    f52_mul(i01, a, a23); f52_mul(i23, a, a01);
    const fe52 x0 = x[0], x2 = x[2];
    f52_mul(x[0], i01, x[1]); f52_mul(x[1], i01, x0);
    f52_mul(x[2], i23, x[3]); f52_mul(x[3], i23, x2);
}
struct ScaBuf { fe52 *X = nullptr, *Y = nullptr, *D = nullptr, *P = nullptr, *TY = nullptr; fe *qx = nullptr; uint8_t *qp = nullptr, *bad = nullptr; };
static bool scabuf_alloc(ScaBuf &v, int B) {
    void *q[8] = {nullptr};
    const size_t sz[8] = {sizeof(fe52) * B, sizeof(fe52) * B, sizeof(fe52) * 2 * B, sizeof(fe52) * 2 * B, sizeof(fe) * 2 * (size_t)B, 2 * (size_t)B, (size_t)B, QSB_CPU_SCA_SINGLE_READ ? sizeof(fe52) * 2 * B : 64};
    for (int i = 0; i < 8; i++) if (posix_memalign(&q[i], 64, sz[i])) { for (int j = 0; j < i; j++) free(q[j]); return false; }
    v.X = (fe52 *)q[0]; v.Y = (fe52 *)q[1]; v.D = (fe52 *)q[2]; v.P = (fe52 *)q[3]; v.qx = (fe *)q[4]; v.qp = (uint8_t *)q[5]; v.bad = (uint8_t *)q[6]; v.TY = (fe52 *)q[7];
    return true;
}
/* Batch-affine additions Q_e = X[k(e)] + R_e over the elements e < E whose candidate is not bad;
 * rowfn(e, &neg) gives R_e's table row (y negated when neg). Four chains (e & 3) share one inversion.
 * out(e, lam, x3, y3...) is handled by the caller through the two modes below. */
template <class RowFn, class Out>
static void sca_add_reread(const fe52 *X, const fe52 *Y, fe52 *D, fe52 *PRE, int E, int kshift, const uint8_t *bad, RowFn rowfn, Out out) {
    /* E % 4 == 0 (the batch size is a multiple of 8) */
    fe52 run[4]; for (int c = 0; c < 4; c++) { memset(&run[c], 0, sizeof(fe52)); run[c].n[0] = 1; }
    for (int e = 0; e < E; e++) {
        const int k = e >> kshift; if (bad[k]) continue;
        if (e + 16 < E) { bool ng; __builtin_prefetch(rowfn(e + 16, ng)); }
        bool ng; const pt *r = rowfn(e, ng);
        fe52 rx; f52_from(rx, r->x);
        f52_sub(D[e], rx, X[k]); f52_nweak(D[e]);
        PRE[e] = run[e & 3]; f52_mul(run[e & 3], run[e & 3], D[e]);
    }
    f52_inv4(run);
    for (int e0 = E - 4; e0 >= 0; e0 -= 4) {           /* four elements (one per chain) stage by stage, for ILP */
        fe52 dinv[4], rx[4], ry[4], lam[4], x3[4], t[4]; int kk[4]; bool ok[4];
        for (int c = 3; c >= 0; c--) {
            const int e = e0 + c; kk[c] = e >> kshift; ok[c] = !bad[kk[c]];
            if (!ok[c]) continue;
            if (e >= 16) { bool ng; __builtin_prefetch(rowfn(e - 16, ng)); }
            f52_mul(dinv[c], run[c], PRE[e]); f52_mul(run[c], run[c], D[e]);
            bool ng; const pt *r = rowfn(e, ng);
            f52_from(rx[c], r->x); f52_from(ry[c], r->y);
            if (ng) { const fe52 z = {{0, 0, 0, 0, 0}}; f52_sub(ry[c], z, ry[c]); f52_nweak(ry[c]); }
        }
        for (int c = 0; c < 4; c++) if (ok[c]) { f52_sub(t[c], ry[c], Y[kk[c]]); f52_mul(lam[c], t[c], dinv[c]); }
        for (int c = 0; c < 4; c++) if (ok[c]) { f52_sqr(x3[c], lam[c]); f52_sub(x3[c], x3[c], X[kk[c]]); f52_sub(x3[c], x3[c], rx[c]); f52_nweak(x3[c]); }
        for (int c = 0; c < 4; c++) if (ok[c]) { f52_sub(t[c], X[kk[c]], x3[c]); f52_mul(t[c], lam[c], t[c]); f52_sub(t[c], t[c], Y[kk[c]]); f52_nweak(t[c]); }
        for (int c = 0; c < 4; c++) if (ok[c]) out(e0 + c, kk[c], x3[c], t[c]);
    }
}

/* Scalar twin of the vector single-read addition: D=rx-X and TY=sign*ry-Y
 * survive the forward pass, so the backward pass never asks rowfn for a point.
 * x3=lambda^2-D-2X combines its three subtractions in one limb pass. */
template <class RowFn, class Out>
static void sca_add(const fe52 *X, const fe52 *Y, fe52 *D, fe52 *PRE, fe52 *TY,
                    int E, int kshift, const uint8_t *bad, RowFn rowfn, Out out) {
#if QSB_CPU_SCA_SINGLE_READ
    const uint64_t pp[5] = {0xFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFULL,
                            0xFFFFFFFFFFFFFULL, 0x0FFFFFFFFFFFFULL};
    fe52 run[4]; for (int c = 0; c < 4; c++) { memset(&run[c], 0, sizeof(fe52)); run[c].n[0] = 1; }
    for (int e = 0; e < E; e++) {
        const int k = e >> kshift; if (bad[k]) continue;
        if (e + 16 < E) { bool ng; __builtin_prefetch(rowfn(e + 16, ng)); }
        bool ng; const pt *r = rowfn(e, ng); fe52 rx, ry;
        f52_from(rx, r->x); f52_from(ry, r->y);
        f52_sub(D[e], rx, X[k]); f52_nweak(D[e]);
        for (int j = 0; j < 5; j++)
            TY[e].n[j] = 8 * pp[j] - Y[k].n[j] + (ng ? 0 : ry.n[j]) - (ng ? ry.n[j] : 0);
        f52_nweak(TY[e]);
        PRE[e] = run[e & 3]; f52_mul(run[e & 3], run[e & 3], D[e]);
    }
    f52_inv4(run);
    for (int e0 = E - 4; e0 >= 0; e0 -= 4) {
        fe52 dinv[4], lam[4], x3[4], t[4]; int kk[4]; bool ok[4];
        for (int c = 3; c >= 0; c--) {
            const int e = e0 + c; kk[c] = e >> kshift; ok[c] = !bad[kk[c]];
            if (!ok[c]) continue;
            f52_mul(dinv[c], run[c], PRE[e]); f52_mul(run[c], run[c], D[e]);
        }
        for (int c = 0; c < 4; c++) if (ok[c]) f52_mul(lam[c], TY[e0 + c], dinv[c]);
        for (int c = 0; c < 4; c++) if (ok[c]) {
            f52_sqr(x3[c], lam[c]);
            for (int j = 0; j < 5; j++)
                x3[c].n[j] += 12 * pp[j] - D[e0 + c].n[j] - 2 * X[kk[c]].n[j];
            f52_nweak(x3[c]);
        }
        for (int c = 0; c < 4; c++) if (ok[c]) {
            f52_sub(t[c], X[kk[c]], x3[c]); f52_mul(t[c], lam[c], t[c]);
            f52_sub(t[c], t[c], Y[kk[c]]); f52_nweak(t[c]);
        }
        for (int c = 0; c < 4; c++) if (ok[c]) out(e0 + c, kk[c], x3[c], t[c]);
    }
#else
    sca_add_reread(X, Y, D, PRE, E, kshift, bad, rowfn, out);
#endif
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
#if QSB_CPU_SCALAR_INV
Q8T static inline void fe8_swap1(fe8 &r, const fe8 &a) { for (int i = 0; i < 5; i++) r.l[i] = _mm512_permutex_epi64(a.l[i], 0xB1); }   /* lanes 2k <-> 2k+1 */
Q8T static inline void fe8_swap2(fe8 &r, const fe8 &a) { for (int i = 0; i < 5; i++) r.l[i] = _mm512_permutex_epi64(a.l[i], 0x4E); }   /* pairs 4k <-> 4k+2 */
Q8T static inline void fe8_swap4(fe8 &r, const fe8 &a) { for (int i = 0; i < 5; i++) r.l[i] = _mm512_shuffle_i64x2(a.l[i], a.l[i], 0x4E); }  /* halves */
Q8T static void fe8_inv_lanes(fe8 &x) {
    /* Preserve the old lane-wise zero behaviour: a zero chain must not
     * poison any of the other seven lanes of the new horizontal tree. */
    __m512i cw[4]; fe8_canon_words(cw, x);
    const __m512i nz = _mm512_or_si512(_mm512_or_si512(cw[0], cw[1]),
                                      _mm512_or_si512(cw[2], cw[3]));
    const __mmask8 zero = _mm512_cmpeq_epi64_mask(nz, _mm512_setzero_si512());
    for (int i = 0; i < 5; i++) x.l[i] = _mm512_mask_mov_epi64(x.l[i], zero, _mm512_set1_epi64(i == 0 ? 1 : 0));
    fe8 a1, p1, p2, t, i2;
    fe8_swap1(a1, x); fe8_mul(p1, x, a1);                 /* lanes 2k, 2k+1: a_2k * a_2k+1 */
    fe8_swap2(t, p1); fe8_mul(p2, p1, t);                 /* lanes 4k..4k+3: the product of the four */
    fe8_swap4(t, p2); fe8_mul(t, p2, t);                  /* every lane: the product of all eight */
    alignas(64) uint64_t w[4][8]; fe8_canon_words(cw, t);
    for (int i = 0; i < 4; i++) _mm512_store_si512((void *)w[i], cw[i]);
    fe pr = {{w[0][0], w[1][0], w[2][0], w[3][0]}}, pi; fe_inv_var(pi, pr);
    fe8 I; fe8_bcast(I, pi);
    fe8_swap4(t, p2); fe8_mul(i2, I, t);                  /* lanes 0..3: 1/(a0 a1 a2 a3), lanes 4..7: 1/(a4..a7) */
    fe8_swap2(t, p1); fe8_mul(i2, i2, t);                 /* 1/(a_2k a_2k+1) */
    fe8_mul(x, i2, a1);                                   /* 1/a_k */
    for (int i = 0; i < 5; i++) x.l[i] = _mm512_mask_mov_epi64(x.l[i], zero, _mm512_setzero_si512());
}
#endif
Q8T static void fe8_inv4(fe8 *x) {
    fe8 a01, a23, a, i01, i23;
    fe8_mul(a01, x[0], x[1]); fe8_mul(a23, x[2], x[3]); fe8_mul(a, a01, a23);
#if QSB_CPU_SCALAR_INV
    fe8_inv_lanes(a);
#else
    fe8_inv1(a);
#endif
    fe8_mul(i01, a, a23); fe8_mul(i23, a, a01);
    const fe8 x0 = x[0], x2 = x[2];
    fe8_mul(x[0], i01, x[1]); fe8_mul(x[1], i01, x0);
    fe8_mul(x[2], i23, x[3]); fe8_mul(x[3], i23, x2);
}
/* One batch-affine window step over G groups of 8 (G % 4 == 0): acc[g] += sign * T[|dig| - 1].
 * rp/ng hold this window's table rows (8 per group) and lane sign masks, filled and prefetched by the previous pass.
 * nxt(hn, &rows) fills the NEXT window's rows of group hn and returns how many to prefetch (8, 16 for the C-folded
 * last window, or 0): the backward pass calls it for group G-1-h, so the next forward pass finds its first groups
 * first, and prefetches half before and half after the group's two chain multiplications. The table misses then
 * spread over this long backward pass instead of bunching in the short forward pass (after terrapinelf's 97f347a8,
 * which measured ~11% of SMT-loaded time waiting on rows with the in-pass prefetch alone). The forward pass keeps a
 * short prefetch pf groups ahead as a backstop. The forward pass keeps D = tx - X and the signed TY = +-ty - Y,
 * so the backward pass never re-reads the table: x3 = lam^2 - D - 2X (after Meganpark980320's bb2a3eb7). */
template <class NxtFn>
Q8T static void ec8_window(fe8 *X, fe8 *Y, fe8 *D, fe8 *PRE, fe8 *TY, int G, int pf, const pt *const *rp, const __mmask8 *ng, NxtFn &nxt) {
    fe8 run[4]; for (int c = 0; c < 4; c++) fe8_set1(run[c]);
    for (int g = 0; g < G; g += 4) {
        for (int c = 0; c < 4; c++) {
            const int h = g + c;
            if (h + pf < G) { const pt *const *pr = rp + (size_t)(h + pf) * 8; for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pr[j], _MM_HINT_T0); }
            fe8 tx, ty; pt8_load(tx, ty, rp + (size_t)h * 8);
            fe8_sub_m(D[h], tx, X[h]);
            fe8_sub_sgn(TY[h], ty, Y[h], ng[h]);
            fe8_mov(PRE[h], run[c]);
            fe8_mul(run[c], run[c], D[h]);
        }
    }
    fe8_inv4(run);
    for (int g = G - 4; g >= 0; g -= 4) {
        fe8 dinv[4], lam[4], t[4], x3[4];
        for (int c = 3; c >= 0; c--) {
            const int h = g + c;
            const pt *const *npr = nullptr; const int nn = nxt(G - 1 - h, &npr);
            for (int j = 0; j < nn / 2; j++) _mm_prefetch((const char *)npr[j], _MM_HINT_T0);
            fe8_mul(dinv[c], run[c], PRE[h]); fe8_mul(run[c], run[c], D[h]);
            for (int j = nn / 2; j < nn; j++) _mm_prefetch((const char *)npr[j], _MM_HINT_T0);
        }
        for (int c = 0; c < 4; c++) fe8_mul(lam[c], TY[g + c], dinv[c]);
        for (int c = 0; c < 4; c++) fe8_sqr_sub3(x3[c], lam[c], D[g + c], X[g + c]);
        for (int c = 0; c < 4; c++) { fe8_sub_m(t[c], X[g + c], x3[c]); fe8_mul_sub(Y[g + c], lam[c], t[c], Y[g + c]); fe8_mov(X[g + c], x3[c]); }
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
                fe8_sqr(x3, lam); fe8_sub2(x3, x3, X[h], CX);
                fe8_sub(t, X[h], x3); fe8_mul(y3, lam, t); fe8_sub(y3, y3, Y[h]);
                __m512i xw[4], yw[4]; fe8_canon_words(xw, x3); fe8_canon_words(yw, y3);
                alignas(64) uint64_t XW[4][8], YP[8];
                for (int i = 0; i < 4; i++) _mm512_store_si512(XW[i], xw[i]);
                _mm512_store_si512(YP, yw[0]);
                for (int j = 0; j < 8; j++) {
                    fe &o = qx[(size_t)(h * 8 + j) * 2 + ri];
                    o.v[0] = XW[0][j]; o.v[1] = XW[1][j]; o.v[2] = XW[2][j]; o.v[3] = XW[3][j];
                    qp[(size_t)(h * 8 + j) * 2 + ri] = (uint8_t)(YP[j] & 1);
                }
            }
        }
    }
}
/* Last window with C folded in: Q_ri = S + R_ri, where R_0 = s*T[a] + C and R_1 = s*T[a] - C come
 * from the precomputed tables T+C and T-C (for s < 0: R_0 = -(T-C)[a], R_1 = -(T+C)[a]). Both
 * recids go through one batch inversion (element e = 2h + ri). rowfn(h, ri, rows) gives the rows
 * and returns the lanes whose y is negated. Writes the canonical x and the y parity. */
Q8T static void ec8_final_fold(const fe8 *X, const fe8 *Y, fe8 *D, fe8 *PRE, fe8 *TY, int G, int pf, const pt *const *rp, const __mmask8 *ngv,
                               fe *qx /* [G*8][2] */, uint8_t *qp) {
    fe8 run[4]; for (int c = 0; c < 4; c++) fe8_set1(run[c]);
    const int E = 2 * G;                                 /* element e = 2 h + ri: rows rp[8 e ..], filled by the previous pass */
    for (int e0 = 0; e0 < E; e0 += 4) {
        for (int c = 0; c < 4; c++) {
            const int e = e0 + c, h = e >> 1;
            if (e + 2 * pf < E) { const pt *const *pr = rp + (size_t)(e + 2 * pf) * 8; for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pr[j], _MM_HINT_T0); }
            fe8 rx, ry; pt8_load(rx, ry, rp + (size_t)e * 8);
            fe8_sub_m(D[e], rx, X[h]);
            fe8_sub_sgn(TY[e], ry, Y[h], ngv[e]);
            fe8_mov(PRE[e], run[c]);
            fe8_mul(run[c], run[c], D[e]);
        }
    }
    fe8_inv4(run);
    for (int e0 = E - 4; e0 >= 0; e0 -= 4) {
        for (int c = 3; c >= 0; c--) {
            const int e = e0 + c, h = e >> 1, ri = e & 1;
            fe8 dinv; fe8_mul(dinv, run[c], PRE[e]); fe8_mul(run[c], run[c], D[e]);
            fe8 t, lam, x3, y3;
            fe8_mul(lam, TY[e], dinv);
            fe8_sqr_sub3(x3, lam, D[e], X[h]);
            fe8_sub_m(t, X[h], x3); fe8_mul_sub(y3, lam, t, Y[h]);
            __m512i xw[4], yw[4]; fe8_canon_words(xw, x3); fe8_canon_words(yw, y3);
            alignas(64) uint64_t XW[4][8], YP[8];
            for (int i = 0; i < 4; i++) _mm512_store_si512(XW[i], xw[i]);
            _mm512_store_si512(YP, yw[0]);
            for (int j = 0; j < 8; j++) {
                fe &o = qx[(size_t)(h * 8 + j) * 2 + ri];
                o.v[0] = XW[0][j]; o.v[1] = XW[1][j]; o.v[2] = XW[2][j]; o.v[3] = XW[3][j];
                qp[(size_t)(h * 8 + j) * 2 + ri] = (uint8_t)(YP[j] & 1);
            }
        }
    }
}
/* Window 0: acc = T0[d0 - 1] (digit-0 lanes load row 0 and are dropped by the caller). */
template <class RowFn, class NxtFn>
Q8T static void ec8_first(fe8 *X, fe8 *Y, int G, int pf, RowFn rowfn, NxtFn &nxt) {
    const pt *rows[8];
    for (int h = 0; h < G; h++) {
        if (h + pf < G) { const pt *pr[8]; rowfn(h + pf, pr); for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pr[j], _MM_HINT_T0); }
        const __mmask8 ng = rowfn(h, rows); pt8_load(X[h], Y[h], rows); fe8_cneg(Y[h], ng);
        const pt *const *npr = nullptr; const int nn = nxt(h, &npr);   /* window 1's rows, in its forward order */
        for (int j = 0; j < nn; j++) _mm_prefetch((const char *)npr[j], _MM_HINT_T0);
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
#if QCPU_VEC && QCPU_SHANI
/* ---- 16-lane SHA-256 with AVX-512F (one candidate per 32-bit lane), selected at run time ---- */
#define S16T __attribute__((target("avx512f")))
#define ROR16(x, n) _mm512_ror_epi32((x), (n))
#define SHA16_ROUND(a, b, c, d, e, f, g, h, WK) do { \
    const __m512i t1 = _mm512_add_epi32(_mm512_add_epi32(h, _mm512_ternarylogic_epi32(ROR16(e, 6), ROR16(e, 11), ROR16(e, 25), 0x96)), \
                                        _mm512_add_epi32(_mm512_ternarylogic_epi32(e, f, g, 0xCA), (WK))); \
    const __m512i t2 = _mm512_add_epi32(_mm512_ternarylogic_epi32(ROR16(a, 2), ROR16(a, 13), ROR16(a, 22), 0x96), \
                                        _mm512_ternarylogic_epi32(a, b, c, 0xE8)); \
    d = _mm512_add_epi32(d, t1); h = _mm512_add_epi32(t1, t2); } while (0)
/* s <- compress(s, block) where WK(t) yields W[t] + K[t] for all lanes */
#define SHA16_BODY(WKEXPR) do { \
    __m512i a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7]; \
    for (int t = 0; t < 64; t += 8) { \
        SHA16_ROUND(a, b, c, d, e, f, g, h, WKEXPR(t + 0)); SHA16_ROUND(h, a, b, c, d, e, f, g, WKEXPR(t + 1)); \
        SHA16_ROUND(g, h, a, b, c, d, e, f, WKEXPR(t + 2)); SHA16_ROUND(f, g, h, a, b, c, d, e, WKEXPR(t + 3)); \
        SHA16_ROUND(e, f, g, h, a, b, c, d, WKEXPR(t + 4)); SHA16_ROUND(d, e, f, g, h, a, b, c, WKEXPR(t + 5)); \
        SHA16_ROUND(c, d, e, f, g, h, a, b, WKEXPR(t + 6)); SHA16_ROUND(b, c, d, e, f, g, h, a, WKEXPR(t + 7)); \
    } \
    s[0] = _mm512_add_epi32(s[0], a); s[1] = _mm512_add_epi32(s[1], b); s[2] = _mm512_add_epi32(s[2], c); s[3] = _mm512_add_epi32(s[3], d); \
    s[4] = _mm512_add_epi32(s[4], e); s[5] = _mm512_add_epi32(s[5], f); s[6] = _mm512_add_epi32(s[6], g); s[7] = _mm512_add_epi32(s[7], h); } while (0)
S16T static void sha16_bcast(__m512i s[8], const uint32_t *wk) {        /* the same W+K for every lane */
#define WKB(t) _mm512_set1_epi32((int)wk[t])
    SHA16_BODY(WKB);
#undef WKB
}
S16T static void sha16_soa(__m512i s[8], const __m512i *wk) {           /* per-lane W+K, lane-transposed */
#define WKS(t) _mm512_load_si512((const void *)&wk[t])
    SHA16_BODY(WKS);
#undef WKS
}
/* wk[t] = W[t] + K[t] (lane-transposed) from the 16 message words w[0..15] */
S16T static void sha16_sched(__m512i *wk, const __m512i *w) {
    __m512i W[16];
#pragma GCC unroll 16
    for (int i = 0; i < 16; i++) { W[i] = w[i]; _mm512_store_si512((void *)&wk[i], _mm512_add_epi32(W[i], _mm512_set1_epi32((int)qsha_k[i]))); }
    /* fully unrolled: the ring indices become constants, so W stays in registers (the rolled loop kept W on the
     * stack and spent 34 instructions per word instead of 13) */
#pragma GCC unroll 48
    for (int t = 16; t < 64; t++) {
        const __m512i w15 = W[(t - 15) & 15], w2 = W[(t - 2) & 15];
        const __m512i s0 = _mm512_ternarylogic_epi32(ROR16(w15, 7), ROR16(w15, 18), _mm512_srli_epi32(w15, 3), 0x96);
        const __m512i s1 = _mm512_ternarylogic_epi32(ROR16(w2, 17), ROR16(w2, 19), _mm512_srli_epi32(w2, 10), 0x96);
        W[t & 15] = _mm512_add_epi32(_mm512_add_epi32(W[t & 15], s0), _mm512_add_epi32(W[(t - 7) & 15], s1));
        _mm512_store_si512((void *)&wk[t], _mm512_add_epi32(W[t & 15], _mm512_set1_epi32((int)qsha_k[t])));
    }
}
S16T static void sha16_iv(__m512i s[8]) { for (int i = 0; i < 8; i++) s[i] = _mm512_set1_epi32((int)qsha_iv[i]); }
#endif
static bool qsha_supported() {
    unsigned a, b, cc, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &cc, &d)) return false;
    return (b >> 29) & 1;                                   /* CPUID.(7,0):EBX.SHA */
}
#endif
struct Ctx {
    const digest_params_t *dp;
    pt *table = nullptr;            /* window i: entries (j+1) * 2^wsh[i] * A at table[woff[i] + j] */
    pt *cfold = nullptr;            /* last window + C, then last window - C (same entry order), or null */
    size_t table_bytes = 0;
    int nw = 0;                     /* lookups per candidate */
    int wbits[NWMAX], wsh[NWMAX];
    size_t woff[NWMAX], went[NWMAX];
    volatile double t_ready = 0;
    /* SMT hybrid: workers are pinned by core; in mode 1 the second thread of each core runs the scalar
     * (integer-multiplier) EC path next to the first thread's 8-lane IFMA path. Chosen at run time. */
    std::atomic<int> mode{0};
    std::atomic<int> pf{QSB_CPU_PF};  /* table-row prefetch distance in 8-lane groups (calibrated: QSB_CPU_PF or 8) */
    int wcpu[512];                  /* logical CPU of worker t, or -1 (unpinned) */
    uint8_t wscalar[512];           /* 1: worker t runs the scalar path in mode 1 */
    bool hybrid_ok = false;
    std::atomic<bool> unpin{false};  /* set when the calibration keeps all-IFMA: workers return to the full mask */
#ifdef CPU_SET
    cpu_set_t mask0;                /* where unpinned workers may run: the process set minus the reserved core */
    cpu_set_t host_mask;            /* the GPU host thread's affinity, read when the workers are placed */
    bool host_mask_ok = false;
    pid_t host_tid = 0;             /* the GPU host thread (start() runs on it) */
    int host_cpu = -1;              /* its CPU at start() when the host producers pinned it to one core (then the core's
                                     * other CPU gets one more worker), else -1 */
#endif
    size_t budget = 0;              /* table memory budget chosen at start-up */
    fe cx, cy;                      /* C = u2*R */
    uint8_t cwin[286][3];           /* CPU window patterns: the complement of the GPU's */
    int ncwin = 0;
    int cut = 137, early = 6;
    uint64_t mid_bytes = 0;         /* preimage bytes covered by dp->midstate */
    uint64_t n_epochs = 0;
    std::atomic<uint64_t> cand{0};
    std::atomic<uint32_t> hits{0};
#ifdef QSB_CPU_DEVBENCH
    std::atomic<uint64_t> tsc[3] = {{0}, {0}, {0}};  /* dev: ns in hashing, window additions, final + gate */
    uint64_t dev_limit = 0;                          /* dev: stop each worker once this many candidates are done */
    std::atomic<int> dev_live{0};
#endif
    std::mutex io;
    qsb_hv_t hv;                    /* exact gate, used under io */
    FILE *out = nullptr;
    int nthreads = 0;
    bool vec = false;               /* 8-lane IFMA path */
    bool shani = false;             /* 4-lane SHA-NI hashing */
    /* SHA-NI preimage schedule, precomputed once: every epoch leaves the same number of bytes (remlen)
     * after its midstate, so block 0 = those bytes ++ the pattern's first pushes (one of nvar variants)
     * and blocks 1..nblk depend on the pattern alone (their W+K is precomputed and deduplicated). */
    bool fast = false;
    int remlen = 0, nblk = 0, nvar = 0;
    std::vector<uint16_t> pvar;             /* pattern -> block-0 variant */
    std::vector<uint8_t> vblk;              /* variant block-0 bytes, 64 each (bytes 0..remlen-1 are per epoch) */
    std::vector<const uint32_t *> pwk;      /* W+K of block b of pattern p at [p * nblk + b - 1] */
    uint32_t *wkpool = nullptr;
    /* 16-lane AVX-512 hashing (mode & 2): lane-transposed block-0 variant words and pattern-block
     * schedules, 16 variants / patterns per chunk */
    bool s16 = false;
    int nvc = 0, npc = 0;                   /* variant chunks, pattern chunks */
    uint32_t *v16 = nullptr;                /* [nvc][16 words][16 lanes], epoch bytes zeroed */
    uint32_t e16mask[16];                   /* per word: bits that come from the epoch bytes */
    uint32_t *p16 = nullptr;                /* [npc][nblk][64][16]: W+K of pattern-specific blocks (lane-transposed) */
    std::vector<uint8_t> blk_const;         /* per block after block 0: 1 when every pattern has the same W+K */
};

static double now_s() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
/* Window layout for L lookups of signed digits: widths floor/ceil(257/L) (the wider ones first),
 * covering 257 bits so the top window absorbs the recoding carry; window i holds the 2^(w-1)
 * multiples 1..2^(w-1) of 2^sh[i] * A. */
/* The window layout as compile-time functions of L (the same widths and offsets as layout() below), so the
 * per-candidate digit recoding can be specialized for each table size with constant shifts and masks. */
static constexpr int win_w(int L, int i) { return 257 / L + (i < 257 % L ? 1 : 0); }
static constexpr int win_s(int L, int i) { int s = 0; for (int j = 0; j < i; j++) s += win_w(L, j); return s; }
/* Signed recoding of z (zl[0..3] little-endian, zl[4] = 0) into L digits, as the generic loop in worker_body:
 * d in (-2^(w-1), 2^(w-1)], stored as |d| | sign << 31; returns 1 when some digit is 0. */
template <int L> static inline unsigned digits_L(uint32_t *dg, const uint64_t zl[5]) {
    unsigned z = 0; uint32_t carry = 0;
#pragma GCC unroll 16
    for (int i = 0; i < L; i++) {
        const int w = win_w(L, i), bit = win_s(L, i), li = bit >> 6, sh = bit & 63;
        uint64_t v = zl[li] >> sh;
        if (sh + w > 64) v |= zl[li + 1] << (64 - sh);
        const uint32_t half = (uint32_t)1 << (w - 1), mask = ((uint32_t)1 << w) - 1;
        const uint32_t d = ((uint32_t)v & mask) + carry;
        if (i + 1 < L) {
            const uint32_t m = 0u - (uint32_t)(d > half);
            const uint32_t e = (d & ~m) | ((((half << 1) - d) | 0x80000000u) & m);
            carry = m & 1;
            dg[i] = e; z |= ((e & 0x7FFFFFFFu) == 0);
        } else { dg[i] = d; z |= (d == 0); }                /* the top window's value never exceeds 2^(w-1) */
    }
    return z;
}
static size_t layout(int L, int *bits, int *sh, size_t *off, size_t *ent) {
    size_t tot = 0; int s = 0;
    for (int i = 0; i < L; i++) {
        const int w = 257 / L + (i < 257 % L ? 1 : 0);
        bits[i] = w; sh[i] = s; off[i] = tot; ent[i] = (size_t)1 << (w - 1);
        s += w; tot += ent[i];
    }
    return tot;
}
static long long read_ll(const char *path) {
    long long v = -1; if (FILE *f = fopen(path, "r")) { char b[64] = {0}; if (fscanf(f, "%63s", b) == 1 && strcmp(b, "max")) v = atoll(b); fclose(f); }
    return v;
}
/* Bytes the table may take: a third of the smaller of MemAvailable and the cgroup headroom. */
static long long g_avail = -1;     /* MemAvailable clamped by every visible cgroup limit (set by mem_budget) */
static bool g_oom_group = false;   /* some visible cgroup kills its whole group on OOM (memory.oom.group = 1) */
static bool g_thp_never = false;   /* transparent huge pages disabled */
static size_t mem_budget() {
    long long avail = -1;
    if (FILE *f = fopen("/proc/meminfo", "r")) {
        char k[64]; long long v; char u[16];
        while (fscanf(f, "%63s %lld %15s", k, &v, u) >= 2) if (!strcmp(k, "MemAvailable:")) { avail = v * 1024; break; }
        fclose(f);
    }
    if (avail < 0) return 0;
    /* cgroup v2 / v1 limits of this process's cgroup and every ancestor that is visible */
    auto clamp = [&](const char *dir, const char *maxf, const char *curf) {
        char a[640], b[640];
        snprintf(a, sizeof a, "%s/%s", dir, maxf); snprintf(b, sizeof b, "%s/%s", dir, curf);
        const long long lim = read_ll(a), cur = read_ll(b);
        if (lim > 0 && lim < (1LL << 50) && cur >= 0 && lim - cur < avail) avail = lim - cur;
    };
    if (FILE *f = fopen("/proc/self/cgroup", "r")) {
        char line[768];
        while (fgets(line, sizeof line, f)) {
            char *q = strrchr(line, ':'); if (!q) continue;
            const bool v2 = !strncmp(line, "0::", 3), mem = strstr(line, "memory") != nullptr;
            if (!v2 && !mem) continue;
            char rel[512]; snprintf(rel, sizeof rel, "%s", q + 1); rel[strcspn(rel, "\n")] = 0;
            const char *base = v2 ? "/sys/fs/cgroup" : "/sys/fs/cgroup/memory";
            for (;;) {                                   /* walk up to the mount root */
                char dir[640]; snprintf(dir, sizeof dir, "%s%s", base, rel);
                if (v2) { clamp(dir, "memory.max", "memory.current"); clamp(dir, "memory.high", "memory.current");
                          char g[700]; snprintf(g, sizeof g, "%s/memory.oom.group", dir); if (read_ll(g) == 1) g_oom_group = true; }
                else clamp(dir, "memory.limit_in_bytes", "memory.usage_in_bytes");
                char *sl = strrchr(rel, '/'); if (!sl || sl == rel) { if (rel[0] && strcmp(rel, "/")) { rel[0] = 0; continue; } break; }
                *sl = 0;
            }
        }
        fclose(f);
    }
    clamp("/sys/fs/cgroup", "memory.max", "memory.current");
    clamp("/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes");
    { char g[64] = "/sys/fs/cgroup/memory.oom.group"; if (read_ll(g) == 1) g_oom_group = true; }
    g_avail = avail;
    if (avail <= 0) return 0;
    size_t b = (size_t)avail / 3, cap = (size_t)QSB_CPU_TABLE_CAP_MB << 20;
    if (FILE *f = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r")) {   /* no huge pages: 4 KiB pages make a huge table */
        char t[128] = {0}; if (fgets(t, sizeof t, f) && strstr(t, "[never]")) { g_thp_never = true; if (cap > ((size_t)8192 << 20)) cap = (size_t)8192 << 20; }   /* slow to fault in and to free */
        fclose(f);
    }
    if (const char *e = getenv("QSB_CPU_TABLE_MB")) cap = (size_t)atoll(e) << 20;   /* dev override */
    if (const char *e = getenv("QSB_CPU_BUDGET_MB")) return (size_t)atoll(e) << 20;  /* dev override: exact budget */
    return b < cap ? b : cap;
}
/* Can this process fault in `bytes` more without being OOM-killed? Spawns this binary as a sacrificial child (see
 * g_probe_child) and waits for it, at most 40 s. Any failure (spawn, kill, timeout) answers no. */
static bool mem_probe(size_t bytes) {
    char exe[4096]; const ssize_t k = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (k <= 0) return false;
    exe[k] = 0;
    std::vector<std::string> env;
    for (char **e = environ; e && *e; e++) if (strncmp(*e, "QSB_CPU_PROBE_MB=", 17)) env.push_back(*e);
    env.push_back("QSB_CPU_PROBE_MB=" + std::to_string((unsigned long long)((bytes >> 20) + 256)));
    std::vector<char *> envp; for (auto &x : env) envp.push_back((char *)x.c_str()); envp.push_back(nullptr);
    char *argv[] = {exe, nullptr};
    pid_t pid = 0;   /* the child inherits this (SCHED_IDLE) thread's policy; no worker runs yet, so the CPUs are free */
    const int rc = posix_spawn(&pid, exe, nullptr, nullptr, argv, envp.data());
    const bool dbg = getenv("QSB_CPU_PROBE_DEBUG") != nullptr;
    if (dbg) fprintf(stderr, "probe: spawn rc %d pid %d exe %s\n", rc, (int)pid, exe);
    if (rc != 0 || pid <= 0) return false;
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    int st = 0;
    for (;;) {
        const pid_t r = waitpid(pid, &st, WNOHANG);
        if (r == pid) break;
        if (r < 0) { if (dbg) fprintf(stderr, "probe: waitpid errno %d\n", errno); return false; }
        struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
        if ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9 > 40) { kill(pid, SIGKILL); waitpid(pid, &st, 0); return false; }
        usleep(20000);
    }
    if (dbg) fprintf(stderr, "probe: status 0x%x exited %d code %d signaled %d sig %d\n", st, WIFEXITED(st), WEXITSTATUS(st), WIFSIGNALED(st), WTERMSIG(st));
    return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}
static pt *table_alloc(size_t bytes) {
    const size_t HP = (size_t)2 << 20, len = (bytes + HP - 1) / HP * HP + HP;
    void *m = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (m == MAP_FAILED) return nullptr;
    uintptr_t a = ((uintptr_t)m + HP - 1) / HP * HP;                 /* 2 MiB aligned for huge pages */
#ifdef MADV_HUGEPAGE
    madvise((void *)a, len - (a - (uintptr_t)m), MADV_HUGEPAGE);
#endif
    return (pt *)a;
}
/* Choose the layout, allocate, and build T_i[j] = (j+1) * 2^sh[i] * A. All windows advance
 * together in doubling rounds (T[have+k] = T[k] + T[have-1]); each round's additions are split
 * into batches of <= 4096 that the threads take from a shared counter. */
static bool build_table(Ctx &c, const fe &ax, const fe &ay, int nth) {
    size_t budget = mem_budget();
    /* The 10-lookup table (19 GiB with the folded copies) only when memory is plentiful: at most a quarter of what is
     * available under every visible limit, no visible cgroup that kills its whole group on OOM, and a sacrificial
     * child has just faulted that much in without being killed. Otherwise the capped budget above (11 lookups). */
    if (!getenv("QSB_CPU_TABLE_MB") && !getenv("QSB_CPU_BUDGET_MB") && !getenv("QSB_CPU_NOPROBE") && QSB_CPU_LMIN <= 10) {
        int b[NWMAX], s[NWMAX]; size_t o[NWMAX], e[NWMAX];
        const size_t t10 = layout(10, b, s, o, e), need = (t10 + 2 * e[9]) * sizeof(pt);
        const bool test = getenv("QSB_CPU_PROBE_TEST") != nullptr;     /* dev: skip the 4x rule */
        if (budget < need && g_avail > 0 && !g_oom_group && !g_thp_never && need <= ((size_t)24 << 30) &&
            (test || (size_t)g_avail >= 4 * need)) {
            const bool ok = mem_probe(need);
            printf("  CPU co-grind: memory probe for %zu MiB (available %lld MiB): %s\n", need >> 20, g_avail >> 20, ok ? "passed" : "failed");
            fflush(stdout);
            if (ok) budget = need;
        }
    }
    c.budget = budget;
    int L = 16; size_t tot = 0;
    for (int l = QSB_CPU_LMIN; l <= 16; l++) {             /* the table plus the two C-folded copies of its last window */
        int b[NWMAX], s[NWMAX]; size_t o[NWMAX], e[NWMAX];
        const size_t t = layout(l, b, s, o, e);
        if (l == 16 || (t + 2 * e[l - 1]) * sizeof(pt) <= budget) { L = l; tot = t; break; }
    }
    for (;;) {
        c.nw = L; tot = layout(L, c.wbits, c.wsh, c.woff, c.went);
        c.table = table_alloc(tot * sizeof(pt));
        if (c.table || L == 16) break;
        L++;
    }
    if (!c.table) return false;
    c.table_bytes = tot * sizeof(pt);
    const size_t nl = c.went[L - 1];
    if (!getenv("QSB_CPU_NOFOLD")) c.cfold = table_alloc(2 * nl * sizeof(pt));
    std::vector<pt> base(L);
    {
        pt q = {ax, ay};
        for (int i = 0; i < L; i++) {
            base[i] = q;
            if (i + 1 < L) for (int s = 0; s < c.wbits[i]; s++) q = pt_double(q);
        }
    }
    for (int i = 0; i < L; i++) { pt *T = c.table + c.woff[i]; T[0] = base[i]; T[1] = pt_double(base[i]); }
    const size_t CH = 4096;
    std::atomic<bool> failed{false};
    for (size_t have = 2;; have *= 2) {
        struct Task { int i; size_t k0, n; };
        std::vector<Task> tasks;
        for (int i = 0; i < L; i++) {
            if (have >= c.went[i]) continue;
            size_t n = have; if (have + n > c.went[i]) n = c.went[i] - have;
            for (size_t k0 = 0; k0 < n; k0 += CH) tasks.push_back({i, k0, n - k0 < CH ? n - k0 : CH});
        }
        if (tasks.empty()) break;
        std::atomic<size_t> next{0};
        auto work = [&]() { try {
            std::vector<fe> d(CH), pre(CH);
            std::vector<uint8_t> inf(CH), bad(CH);
            std::vector<const pt *> tp(CH);
            for (size_t q; (q = next.fetch_add(1)) < tasks.size();) {
                const Task &t = tasks[q];
                pt *T = c.table + c.woff[t.i];
                for (size_t k = 0; k < t.n; k++) { T[have + t.k0 + k] = T[t.k0 + k]; tp[k] = &T[have - 1]; inf[k] = 0; bad[k] = 0; }
                batch_add(&T[have + t.k0], tp.data(), inf.data(), bad.data(), (int)t.n, d.data(), pre.data());
                /* k = have-1 adds T[have-1] to itself: equal x, so batch_add flags it; double it. */
                for (size_t k = 0; k < t.n; k++) if (bad[k]) T[have + t.k0 + k] = pt_double(T[have - 1]);
            }
        } catch (...) { failed = true; } };
        std::vector<std::thread> ts;
        for (int t = 1; t < nth; t++) { try { ts.emplace_back(work); } catch (...) { break; } }
        work();                                         /* this thread takes batches too (and finishes them alone if no thread could start) */
        for (auto &t : ts) t.join();
        if (failed) return false;
    }
    if (c.cfold) {                                      /* TP[j] = T_last[j] + C, TM[j] = T_last[j] - C */
        const pt *TL = c.table + c.woff[L - 1];
        pt cp = {c.cx, c.cy}, cm = cp; const fe z0 = {{0, 0, 0, 0}}; fe_sub(cm.y, z0, c.cy);
        std::atomic<size_t> next{0}; const size_t nt = (nl + CH - 1) / CH;
        auto work = [&]() { try {
            std::vector<fe> d(CH), pre(CH); std::vector<uint8_t> inf(CH), bad(CH); std::vector<const pt *> tp(CH);
            for (size_t q; (q = next.fetch_add(1)) < 2 * nt;) {
                const size_t k0 = (q % nt) * CH, n = nl - k0 < CH ? nl - k0 : CH; const int m = (int)(q / nt);
                pt *O = c.cfold + (size_t)m * nl + k0;
                for (size_t k = 0; k < n; k++) { O[k] = TL[k0 + k]; tp[k] = m ? &cm : &cp; inf[k] = 0; bad[k] = 0; }
                batch_add(O, tp.data(), inf.data(), bad.data(), (int)n, d.data(), pre.data());
            }                                           /* an entry equal to -+C (probability ~2^-230) stays wrong: its hits are lost, never published */
        } catch (...) { failed = true; } };
        std::vector<std::thread> ts;
        for (int t = 1; t < nth; t++) { try { ts.emplace_back(work); } catch (...) { break; } }
        work();
        for (auto &t : ts) t.join();
        if (failed) c.cfold = nullptr;
        else c.table_bytes += 2 * nl * sizeof(pt);
    }
    return true;
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

/* The EC part of one batch on the scalar path; same inputs and outputs as vec_batch. */
static void sca_batch(const Ctx *c, const uint32_t *dig, const uint8_t *zdig, int B, ScaBuf &v) {
    const int nw = c->nw;
    memcpy(v.bad, zdig, (size_t)B);
    auto entry = [&](int k, int i, const pt *T, bool &ng) -> const pt * {
        const uint32_t di = dig[(size_t)k * NWMAX + i], a = di & 0x7FFFFFFFu; ng = (di >> 31) != 0;
        return &T[(a ? a : 1) - 1];
    };
    {   /* window 0: X, Y = the first digit's entry */
        const pt *T = c->table + c->woff[0];
        for (int k = 0; k < B; k++) {
            if (v.bad[k]) continue;
            bool ng; const pt *r = entry(k, 0, T, ng);
            f52_from(v.X[k], r->x); f52_from(v.Y[k], r->y);
            if (ng) { const fe52 z = {{0, 0, 0, 0, 0}}; f52_sub(v.Y[k], z, v.Y[k]); f52_nweak(v.Y[k]); }
        }
    }
    const bool fold = c->cfold != nullptr;
    for (int i = 1; i < (fold ? nw - 1 : nw); i++) {
        const pt *T = c->table + c->woff[i];
        sca_add(v.X, v.Y, v.D, v.P, v.TY, B, 0, v.bad, [&](int e, bool &ng) { return entry(e, i, T, ng); },
                [&](int, int k, const fe52 &x3, const fe52 &y3) { v.X[k] = x3; v.Y[k] = y3; });
    }
    auto emit = [&](int k, int ri, const fe52 &x3, const fe52 &y3) {
        uint64_t w[4], yw[4]; f52_words(w, x3); f52_words(yw, y3);
        fe &o = v.qx[(size_t)k * 2 + ri]; o.v[0] = w[0]; o.v[1] = w[1]; o.v[2] = w[2]; o.v[3] = w[3];
        v.qp[(size_t)k * 2 + ri] = (uint8_t)(yw[0] & 1);
    };
    if (fold) {                                          /* last window with C folded in: element e = 2k + ri */
        const int i = nw - 1; const pt *TP = c->cfold, *TM = c->cfold + c->went[i];
        sca_add(v.X, v.Y, v.D, v.P, v.TY, 2 * B, 1, v.bad, [&](int e, bool &ng) {
                    const int k = e >> 1, ri = e & 1; const pt *r = entry(k, i, TP, ng);
                    return (ng ^ (ri != 0)) ? TM + (r - TP) : r; },
                [&](int e, int k, const fe52 &x3, const fe52 &y3) { emit(k, e & 1, x3, y3); });
    } else {                                              /* Q_ri = X + (+-C): element e = 2k + ri */
        pt cps[2]; cps[0].x = c->cx; cps[0].y = c->cy; cps[1] = cps[0];
        const fe z0 = {{0, 0, 0, 0}}; fe_sub(cps[1].y, z0, c->cy);
        sca_add(v.X, v.Y, v.D, v.P, v.TY, 2 * B, 1, v.bad, [&](int e, bool &ng) { ng = false; return &cps[e & 1]; },
                [&](int e, int k, const fe52 &x3, const fe52 &y3) { emit(k, e & 1, x3, y3); });
    }
}

#if QCPU_VEC
struct VecBuf { fe8 *X = nullptr, *Y = nullptr, *D = nullptr, *P = nullptr, *TY = nullptr; fe *qx = nullptr; uint8_t *qp = nullptr, *bad = nullptr;
                const pt **rb[2] = {nullptr, nullptr}; __mmask8 *nb[2] = {nullptr, nullptr}; };   /* row buffers: 2G groups of 8 each */
static bool vecbuf_alloc(VecBuf &v, int B) {
    const int G = B / 8; void *q[12] = {nullptr};
    const size_t sz[12] = {sizeof(fe8) * G, sizeof(fe8) * G, sizeof(fe8) * 2 * G, sizeof(fe8) * 2 * G, sizeof(fe8) * 2 * G,
                           sizeof(fe) * 2 * (size_t)B, 2 * (size_t)B, (size_t)B,
                           sizeof(pt *) * 16 * (size_t)G, sizeof(pt *) * 16 * (size_t)G, 2 * (size_t)G, 2 * (size_t)G};
    for (int i = 0; i < 12; i++) if (posix_memalign(&q[i], 64, sz[i])) { for (int j = 0; j < i; j++) free(q[j]); return false; }
    v.X = (fe8 *)q[0]; v.Y = (fe8 *)q[1]; v.D = (fe8 *)q[2]; v.P = (fe8 *)q[3]; v.TY = (fe8 *)q[4];
    v.qx = (fe *)q[5]; v.qp = (uint8_t *)q[6]; v.bad = (uint8_t *)q[7];
    v.rb[0] = (const pt **)q[8]; v.rb[1] = (const pt **)q[9]; v.nb[0] = (__mmask8 *)q[10]; v.nb[1] = (__mmask8 *)q[11];
    return true;
}
/* The EC part of one batch on the 8-lane path: z*A for every candidate (c->nw table windows), then
 * both recovery ids against C. dig is candidate-major (B x NWMAX). */
Q8T static void vec_batch(const Ctx *c, const uint32_t *dig, const uint8_t *zdig, int B, VecBuf &v) {
    const int G = B / 8, nw = c->nw, pf = c->pf.load(std::memory_order_relaxed);
    memcpy(v.bad, zdig, (size_t)B);
    const bool fold = c->cfold != nullptr;
    /* digit = |d| | sign << 31; |d| = 0 only in dropped lanes (they load entry 0). Window i's rows go to buffer i & 1. */
    auto fill = [&](int i, int hn, const pt *const **out) -> int {      /* rows of window i, group hn */
        if (i >= nw) return 0;
        const int b = i & 1;
        if (fold && i == nw - 1) {                                       /* C-folded last window: elements 2 hn + ri */
            const pt *TP = c->cfold, *TM = c->cfold + c->went[nw - 1];
            for (int ri = 0; ri < 2; ri++) {
                const pt **rows = v.rb[b] + (size_t)(2 * hn + ri) * 8; unsigned ng = 0;
                for (int j = 0; j < 8; j++) {
                    const uint32_t di = dig[(size_t)(hn * 8 + j) * NWMAX + i], a = di & 0x7FFFFFFFu, sg = di >> 31;
                    rows[j] = &((sg ^ (unsigned)ri) ? TM : TP)[(a ? a : 1) - 1]; ng |= sg << j;
                }
                v.nb[b][2 * hn + ri] = (__mmask8)ng;
            }
            *out = v.rb[b] + (size_t)(2 * hn) * 8; return 16;
        }
        const pt *T = c->table + c->woff[i]; const pt **rows = v.rb[b] + (size_t)hn * 8; unsigned ng = 0;
        for (int j = 0; j < 8; j++) {
            const uint32_t di = dig[(size_t)(hn * 8 + j) * NWMAX + i], a = di & 0x7FFFFFFFu;
            rows[j] = &T[(a ? a : 1) - 1]; ng |= (di >> 31) << j;
        }
        v.nb[b][hn] = (__mmask8)ng;
        *out = rows; return 8;
    };
    {
        const pt *T0 = c->table + c->woff[0];
        auto rowfn0 = [&](int h, const pt **rows) -> __mmask8 {
            unsigned ng = 0;
            for (int j = 0; j < 8; j++) {
                const uint32_t di = dig[(size_t)(h * 8 + j) * NWMAX], a = di & 0x7FFFFFFFu;
                rows[j] = &T0[(a ? a : 1) - 1]; ng |= (di >> 31) << j;
            }
            return (__mmask8)ng;
        };
        auto nxt1 = [&](int hn, const pt *const **out) -> int { return fill(1, hn, out); };
        ec8_first(v.X, v.Y, G, pf, rowfn0, nxt1);
    }
    for (int i = 1; i < nw; i++) {
        const int b = i & 1;
        if (fold && i == nw - 1) {                                       /* last window with C folded in: both recids at once */
            ec8_final_fold(v.X, v.Y, v.D, v.P, v.TY, G, pf, v.rb[b], v.nb[b], v.qx, v.qp);
            return;
        }
        auto nxt = [&](int hn, const pt *const **out) -> int { return fill(i + 1, hn, out); };
        ec8_window(v.X, v.Y, v.D, v.P, v.TY, G, pf, v.rb[b], v.nb[b], nxt);
    }
    ec8_final(v.X, v.Y, v.D, v.P, G, c->cx, c->cy, v.qx, v.qp);
}
#endif

#if QCPU_SHANI
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
    void *q = nullptr;
    if (posix_memalign(&q, 64, uniq.size() * 4)) return;
    c->wkpool = (uint32_t *)q; memcpy(c->wkpool, uniq.data(), uniq.size() * 4);
    c->pwk.resize(idx.size());
    for (size_t i = 0; i < idx.size(); i++) c->pwk[i] = c->wkpool + (size_t)idx[i] * 64;
    c->remlen = (int)remlen; c->nblk = nblk; c->nvar = nvar; c->fast = true;
#if QCPU_VEC
    if (c->vec) {                                       /* lane-transposed tables for the 16-lane hashing */
        c->blk_const.assign(nblk, 1);
        for (int b = 0; b < nblk; b++) for (int p = 1; p < c->ncwin; p++) if (c->pwk[(size_t)p * nblk + b] != c->pwk[b]) c->blk_const[b] = 0;
        c->nvc = (nvar + 15) / 16; c->npc = (c->ncwin + 15) / 16;
        void *a = nullptr, *b2 = nullptr;
        if (posix_memalign(&a, 64, (size_t)c->nvc * 256 * 4) || posix_memalign(&b2, 64, (size_t)c->npc * nblk * 64 * 16 * 4)) { free(a); free(b2); return; }
        c->v16 = (uint32_t *)a; c->p16 = (uint32_t *)b2;
        for (int i = 0; i < 16; i++) {                  /* bits of message word i that are epoch bytes */
            uint32_t m = 0; for (int q = 0; q < 4; q++) if ((size_t)(4 * i + q) < remlen) m |= 0xFFu << (24 - 8 * q);
            c->e16mask[i] = m;
        }
        for (int ch = 0; ch < c->nvc; ch++)
            for (int l = 0; l < 16; l++) {
                const int v = ch * 16 + l < nvar ? ch * 16 + l : nvar - 1;
                for (int i = 0; i < 16; i++) c->v16[((size_t)ch * 16 + i) * 16 + l] = be32(&c->vblk[(size_t)v * 64 + 4 * i]) & ~c->e16mask[i];
            }
        for (int ch = 0; ch < c->npc; ch++)
            for (int b = 0; b < nblk; b++)
                for (int l = 0; l < 16; l++) {
                    const int p = ch * 16 + l < c->ncwin ? ch * 16 + l : c->ncwin - 1;
                    const uint32_t *w = c->pwk[(size_t)p * nblk + b];
                    for (int t = 0; t < 64; t++) c->p16[(((size_t)ch * nblk + b) * 64 + t) * 16 + l] = w[t];
                }
        c->s16 = true;
    }
#endif
}
/* Message words of the compressed key's SHA-256 block (33 bytes, 264-bit length). */
static inline void pk_words(uint32_t *w, const fe &x, unsigned ypar) {
    const uint64_t v3 = x.v[3], v2 = x.v[2], v1 = x.v[1], v0 = x.v[0];
    w[0] = ((0x02u | (ypar & 1)) << 24) | (uint32_t)(v3 >> 40);
    w[1] = (uint32_t)(v3 >> 8);
    w[2] = (uint32_t)(v3 << 24) | (uint32_t)(v2 >> 40);
    w[3] = (uint32_t)(v2 >> 8);
    w[4] = (uint32_t)(v2 << 24) | (uint32_t)(v1 >> 40);
    w[5] = (uint32_t)(v1 >> 8);
    w[6] = (uint32_t)(v1 << 24) | (uint32_t)(v0 >> 40);
    w[7] = (uint32_t)(v0 >> 8);
    w[8] = (uint32_t)(v0 << 24) | 0x800000u;
    for (int i = 9; i < 15; i++) w[i] = 0;
    w[15] = 264;
}
#endif
#if QCPU_VEC && QCPU_SHANI
/* Every candidate of one epoch at once (16-lane hashing): block-0 states of the nvar variants, then
 * blocks 1..nblk and the SHA-256d outer block of the ncwin patterns, 16 per chunk. dg[p] = digest words. */
S16T static void epoch16(const Ctx *c, const uint32_t est[8], const uint8_t *erem, uint32_t (*dg)[8], uint32_t *vs /* [nvc][8][16] */) {
    const int nblk = c->nblk;
    alignas(64) __m512i wk[64];
    uint8_t eb[64] = {0}; memcpy(eb, erem, (size_t)c->remlen);
    for (int ch = 0; ch < c->nvc; ch++) {
        __m512i w[16], st[8];
        for (int i = 0; i < 16; i++) {
            const __m512i vw = _mm512_load_si512((const void *)&c->v16[((size_t)ch * 16 + i) * 16]);
            w[i] = c->e16mask[i] ? _mm512_or_si512(vw, _mm512_set1_epi32((int)(be32(eb + 4 * i) & c->e16mask[i]))) : vw;
        }
        sha16_sched(wk, w);
        for (int i = 0; i < 8; i++) st[i] = _mm512_set1_epi32((int)est[i]);
        sha16_soa(st, wk);
        for (int i = 0; i < 8; i++) _mm512_store_si512((void *)&vs[((size_t)ch * 8 + i) * 16], st[i]);
    }
    for (int ch = 0; ch < c->npc; ch++) {
        alignas(64) uint32_t tmp[8][16];
        for (int l = 0; l < 16; l++) {
            const int p = ch * 16 + l < c->ncwin ? ch * 16 + l : c->ncwin - 1, v = c->pvar[p];
            for (int i = 0; i < 8; i++) tmp[i][l] = vs[((size_t)(v >> 4) * 8 + i) * 16 + (v & 15)];
        }
        __m512i st[8]; for (int i = 0; i < 8; i++) st[i] = _mm512_load_si512((const void *)tmp[i]);
        for (int b = 0; b < nblk; b++) {
            if (c->blk_const[b]) sha16_bcast(st, c->pwk[b]);
            else sha16_soa(st, (const __m512i *)&c->p16[((size_t)ch * nblk + b) * 64 * 16]);
        }
        __m512i w[16];
        for (int i = 0; i < 8; i++) w[i] = st[i];
        w[8] = _mm512_set1_epi32((int)0x80000000u); for (int i = 9; i < 15; i++) w[i] = _mm512_setzero_si512(); w[15] = _mm512_set1_epi32(256);
        sha16_sched(wk, w); sha16_iv(st); sha16_soa(st, wk);
        for (int i = 0; i < 8; i++) _mm512_store_si512((void *)tmp[i], st[i]);
        for (int l = 0; l < 16 && ch * 16 + l < c->ncwin; l++) for (int i = 0; i < 8; i++) dg[ch * 16 + l][i] = tmp[i][l];
    }
}
/* First digest word of the compressed-key hash of entries 0..n-1 (n % 16 == 0), 16 at a time. */
S16T static void keyhash16(const fe *qx, const uint8_t *qp, int n, uint32_t *h0) {
    alignas(64) __m512i wk[64]; alignas(64) uint32_t mw[16][16];
    for (int e0 = 0; e0 < n; e0 += 16) {
        for (int l = 0; l < 16; l++) { uint32_t w[16]; pk_words(w, qx[e0 + l], qp[e0 + l]); for (int i = 0; i < 16; i++) mw[i][l] = w[i]; }
        __m512i w[16]; for (int i = 0; i < 16; i++) w[i] = _mm512_load_si512((const void *)mw[i]);
        sha16_sched(wk, w); __m512i st[8]; sha16_iv(st); sha16_soa(st, wk);
        _mm512_storeu_si512((void *)&h0[e0], st[0]);
    }
}
#endif
static void worker_body(Ctx *c, int tid) {
#ifdef SCHED_IDLE
    struct sched_param sp; sp.sched_priority = 0; sched_setscheduler(0, SCHED_IDLE, &sp);
#endif
    bool pinned = false;
#ifdef CPU_SET
    if (tid < 512 && c->wcpu[tid] >= 0) { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(c->wcpu[tid], &cs); pinned = pthread_setaffinity_np(pthread_self(), sizeof cs, &cs) == 0; }
#endif
    const digest_params_t *dp = c->dp;
    const int B = QSB_CPU_BATCH;
    std::vector<pt> acc(B); std::vector<fe> d(2 * B), pre(2 * B);
    std::vector<uint8_t> inf(B), bad(B); std::vector<const pt *> tp(B); std::vector<pt> ntp(B);
    std::vector<uint32_t> dig((size_t)B * NWMAX);
    const int nw = c->nw;
    std::vector<uint8_t> skips((size_t)B * 9);
    uint8_t pk[64]; memset(pk, 0, 64); pk[33] = 0x80; pk[62] = 0x01; pk[63] = 0x08;   /* 264 bits */
    uint8_t blk2[64]; memset(blk2, 0, 64); blk2[32] = 0x80; blk2[62] = 0x01;          /* 256 bits */
    uint64_t epoch = (uint64_t)tid;
    int wi = c->ncwin;
    SHA256_CTX ectx; uint8_t early[16];
    std::vector<uint8_t> pbuf((size_t)dp->n * SIG_PUSH_SIZE + 64);
    std::vector<uint8_t> zdig(B);                      /* 1 when some table digit of the candidate is 0 */
    auto put_digits = [&](int kk, const uint32_t *h) {    /* table digits of z = h[0] (MSW) .. h[7] */
        uint64_t zl[5];
        for (int i = 0; i < 4; i++) zl[i] = ((uint64_t)h[6 - 2 * i] << 32) | h[7 - 2 * i];
        zl[4] = 0;
        uint32_t *dg = &dig[(size_t)kk * NWMAX];
        switch (nw) {                                   /* constant shifts and masks for the usual table sizes */
            case 10: zdig[kk] = (uint8_t)digits_L<10>(dg, zl); return;
            case 11: zdig[kk] = (uint8_t)digits_L<11>(dg, zl); return;
            case 12: zdig[kk] = (uint8_t)digits_L<12>(dg, zl); return;
            case 13: zdig[kk] = (uint8_t)digits_L<13>(dg, zl); return;
            case 14: zdig[kk] = (uint8_t)digits_L<14>(dg, zl); return;
            case 15: zdig[kk] = (uint8_t)digits_L<15>(dg, zl); return;
            case 16: zdig[kk] = (uint8_t)digits_L<16>(dg, zl); return;
            default: break;
        }
        unsigned z = 0; uint32_t carry = 0;
        for (int i = 0; i < nw; i++) {                  /* signed recoding: d in (-2^(w-1), 2^(w-1)], z = sum d_i 2^sh_i */
            const int bit = c->wsh[i], w = c->wbits[i], li = bit >> 6, sh = bit & 63;
            uint64_t v = li < 4 ? zl[li] >> sh : 0;
            if (sh + w > 64 && li < 3) v |= zl[li + 1] << (64 - sh);
            const uint32_t half = (uint32_t)1 << (w - 1);
            const uint32_t d = (uint32_t)(v & (((uint64_t)1 << w) - 1)) + carry;
            /* branch-free: d > 2^(w-1) becomes -(2^w - d) with a carry (never in the top window, whose value is <= 2^(w-1)) */
            const uint32_t m = 0u - (uint32_t)((d > half) & (i + 1 < nw));
            const uint32_t e = (d & ~m) | ((((half << 1) - d) | 0x80000000u) & m);
            carry = m & 1;
            dig[(size_t)kk * NWMAX + i] = e; z |= ((e & 0x7FFFFFFFu) == 0);
        }
        zdig[kk] = (uint8_t)z;
    };
#if QCPU_SHANI
    /* 4-lane path: each lane holds one candidate's chaining state (its epoch's) and the padded rest of
     * its message: the epoch's buffered bytes, the kept window pushes, tail section, suffix. */
    const size_t tl = dp->tail_section_len, sl = dp->tx_suffix_len;
    alignas(16) uint32_t lst[4][8]; alignas(16) uint8_t lmsg[4][16 * 64]; const uint8_t *lp[4];
    uint32_t est[8]; uint8_t erem[64]; size_t remlen = 0; int nb = 0; uint64_t tbits = 0;
    bool shani = c->shani, fast = c->shani && c->fast;
    const int nblk = c->nblk;
    std::vector<uint32_t> vst((size_t)(c->nvar > 0 ? c->nvar : 1) * 8);   /* per-epoch block-0 states */
    bool vst_ok = false, dg_ok = false;                  /* which per-epoch data the current epoch has */
    std::vector<uint32_t> dg16((size_t)c->ncwin * 8 + 8), h016((size_t)2 * B);
    std::vector<uint32_t, qalloc64<uint32_t> > vs16((size_t)(c->nvc > 0 ? c->nvc : 1) * 128);   /* aligned: zmm stores */
    alignas(16) uint32_t wkv[4][64], wk2[4][64];
    const uint32_t *lwk[4 * 16], *lw2[4] = {wk2[0], wk2[1], wk2[2], wk2[3]};
#endif
#if QCPU_VEC
    VecBuf vb;
    if (c->vec && !vecbuf_alloc(vb, B)) vb = VecBuf();
#endif
    ScaBuf sb;                                           /* scalar path (always available; the hybrid switches per batch) */
    if (!scabuf_alloc(sb, B)) sb = ScaBuf();
#ifdef QSB_CPU_DEVBENCH
#define QDEV_MARK(i) do { const uint64_t _t = (uint64_t)(now_s() * 1e9); c->tsc[i] += _t - qdev_t; qdev_t = _t; } while (0)
    uint64_t qdev_t = (uint64_t)(now_s() * 1e9);
#else
#define QDEV_MARK(i) do {} while (0)
#endif
    for (;;) {
#ifdef QSB_CPU_DEVBENCH
        if (c->dev_limit && c->cand.load() >= c->dev_limit) { c->dev_live--; return; }
#endif
        int k = 0;
        const int md = c->mode.load(std::memory_order_relaxed);
#ifdef CPU_SET
        if (pinned && c->unpin.load(std::memory_order_relaxed)) { pthread_setaffinity_np(pthread_self(), sizeof c->mask0, &c->mask0); pinned = false; }
#endif
#if QCPU_VEC && QCPU_SHANI
        const bool use16 = fast && c->s16 && (md & 2);
#else
        const bool use16 = false;
#endif
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
                    if (!shani || remlen != (size_t)c->remlen || nb != nblk + 1) fast = false;
                    vst_ok = dg_ok = false;
                    if (fast && !use16) {                   /* the epoch's block-0 states, four variants at a time */
                        vst_ok = true;
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
                }
#endif
                epoch += (uint64_t)c->nthreads; wi = 0;
            }
            const int pat = wi;
            const uint8_t *w3 = c->cwin[wi++];
#if QCPU_SHANI
#if QCPU_VEC && QCPU_SHANI
            if (use16) {                                /* 16-lane hashing: the whole epoch at once */
                if (!dg_ok) { epoch16(c, est, erem, (uint32_t (*)[8])dg16.data(), vs16.data()); dg_ok = true; }
                put_digits(k, dg16.data() + (size_t)pat * 8);
                uint8_t *sk = &skips[(size_t)k * 9];
                for (int q = 0; q < 6; q++) sk[q] = early[q];
                sk[6] = w3[0]; sk[7] = w3[1]; sk[8] = w3[2];
                inf[k] = 1; bad[k] = 0;
                k++;
                continue;
            }
#endif
            if (fast) {                                 /* precomputed schedule: rounds only */
                if (!vst_ok) {                          /* the epoch began under the other hashing mode */
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
                    vst_ok = true;
                }
                const int j = k & 3;
                memcpy(lst[j], &vst[(size_t)c->pvar[pat] * 8], 32);
                for (int b = 0; b < nblk; b++) lwk[4 * b + j] = c->pwk[(size_t)pat * nblk + b];
                uint8_t *sk = &skips[(size_t)k * 9];
                for (int q = 0; q < 6; q++) sk[q] = early[q];
                sk[6] = w3[0]; sk[7] = w3[1]; sk[8] = w3[2];
                inf[k] = 1; bad[k] = 0;
                if (j == 3) {
                    qsha_x4_run(lst, lwk, nblk);
                    alignas(16) uint32_t s2[4][8];
                    for (int l = 0; l < 4; l++) {               /* SHA-256d: the digest words are the message */
                        alignas(16) uint32_t w[16] = {lst[l][0], lst[l][1], lst[l][2], lst[l][3], lst[l][4], lst[l][5], lst[l][6], lst[l][7],
                                                      0x80000000u, 0, 0, 0, 0, 0, 0, 256};
                        qsha_sched(wk2[l], w); memcpy(s2[l], qsha_iv, 32);
                    }
                    qsha_x4_run(s2, lw2, 1);
                    for (int l = 0; l < 4; l++) put_digits(k - 3 + l, s2[l]);
                }
                k++;
                continue;
            }
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
        QDEV_MARK(0);
        const fe *QX = nullptr; const uint8_t *QP = nullptr, *QB = nullptr;
#if QCPU_VEC
        if (vb.X && !((md & 1) && tid < 512 && c->wscalar[tid])) {
            vec_batch(c, dig.data(), zdig.data(), B, vb); QX = vb.qx; QP = vb.qp; QB = vb.bad;
        } else
#endif
        if (sb.X) { sca_batch(c, dig.data(), zdig.data(), B, sb); QX = sb.qx; QP = sb.qp; QB = sb.bad; }
        if (QX) {
            QDEV_MARK(1);
#if QCPU_SHANI
#if QCPU_VEC && QCPU_SHANI
            if (use16) {                                /* key hashes 16 at a time */
                keyhash16(QX, QP, 2 * B, h016.data());
                for (int q = 0; q < B; q++) {
                    if (QB[q]) continue;
                    for (int ri = 0; ri < 2; ri++)
                        if (pk_prefilter(h016[(size_t)q * 2 + ri]) && gate_publish_exact(c, &skips[(size_t)q * 9], ri)) break;   /* one recid per candidate */
                }
            } else
#endif
            if (c->shani) {                             /* key hashes 4 at a time: 2 candidates x 2 recids */
                for (int kk = 0; kk < B; kk += 2) {
                    alignas(16) uint32_t hs[4][8];
                    for (int l = 0; l < 4; l++) {
                        const size_t q = (size_t)(kk + (l >> 1)) * 2 + (l & 1);
                        alignas(16) uint32_t w[16]; pk_words(w, QX[q], QP[q]);
                        qsha_sched(wk2[l], w); memcpy(hs[l], qsha_iv, 32);
                    }
                    qsha_x4_run(hs, lw2, 1);
                    for (int q2 = 0; q2 < 2; q2++) {
                        const int q = kk + q2;
                        if (QB[q]) continue;
                        for (int ri = 0; ri < 2; ri++)
                            if (pk_prefilter(hs[2 * q2 + ri][0]) && gate_publish_exact(c, &skips[(size_t)q * 9], ri)) break;   /* one recid per candidate */
                    }
                }
            } else
#endif
            for (int kk = 0; kk < B; kk++) {
                if (QB[kk]) continue;
                const uint8_t *sk = &skips[(size_t)kk * 9];
                for (int ri = 0; ri < 2; ri++)
                    if (gate_publish(c, pk, sk, ri, QX[(size_t)kk * 2 + ri], QP[(size_t)kk * 2 + ri])) break;   /* one recid per candidate */
            }
            QDEV_MARK(2);
            c->cand += B;
            continue;
        }
        for (int i = 0; i < nw; i++) {
            const pt *T = c->table + c->woff[i];
            for (int kk = 0; kk < B; kk++) {
                const uint32_t v = dig[(size_t)kk * NWMAX + i], a = v & 0x7FFFFFFFu;
                if (!a) { tp[kk] = nullptr; continue; }
                if (v >> 31) { const fe z0 = {{0, 0, 0, 0}}; ntp[kk].x = T[a - 1].x; fe_sub(ntp[kk].y, z0, T[a - 1].y); tp[kk] = &ntp[kk]; }
                else tp[kk] = &T[a - 1];
            }
            batch_add(acc.data(), tp.data(), inf.data(), bad.data(), B, d.data(), pre.data());
        }
        QDEV_MARK(1);
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
        QDEV_MARK(2);
        c->cand += B;
    }
}

/* Pin the workers core by core (thread_siblings_list, within this process's affinity mask), leaving
 * one core free for the GPU host thread; the second worker of each core is the hybrid's scalar one. */
static void smt_plan(Ctx *c, int nth) {
    for (int t = 0; t < 512; t++) { c->wcpu[t] = -1; c->wscalar[t] = 0; }
    c->hybrid_ok = false;
#ifdef CPU_SET
    cpu_set_t cs; CPU_ZERO(&cs);
    if (nth > 512) return;
    if (g_mask0_ok) cs = g_mask0; else if (sched_getaffinity(0, sizeof cs, &cs) != 0) return;
    cpu_set_t host; CPU_ZERO(&host);                    /* the GPU host thread's CPUs if it is pinned (read now: it may pin itself after start()) */
    CPU_ZERO(&c->host_mask);
    c->host_mask_ok = c->host_tid > 0 && sched_getaffinity(c->host_tid, sizeof c->host_mask, &c->host_mask) == 0;
    const bool host_pinned = c->host_mask_ok && CPU_COUNT(&c->host_mask) < CPU_COUNT(&cs);
    if (host_pinned) host = c->host_mask;
    std::vector<std::vector<int> > cores; std::vector<uint8_t> seen(CPU_SETSIZE, 0);
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (!CPU_ISSET(cpu, &cs) || seen[cpu]) continue;
        char path[128]; snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        FILE *f = fopen(path, "r"); if (!f) return;
        char buf[256] = {0}; const bool ok = fgets(buf, sizeof buf, f) != nullptr; fclose(f); if (!ok) return;
        std::vector<int> core;
        for (char *q = buf; *q;) {                      /* "a-b,c,..." */
            char *e; long a = strtol(q, &e, 10); if (e == q) break; long b = a;
            if (*e == '-') { q = e + 1; b = strtol(q, &e, 10); }
            for (long x = a; x <= b && x < CPU_SETSIZE; x++) if (x >= 0 && CPU_ISSET(x, &cs) && !seen[x]) { core.push_back((int)x); seen[x] = 1; }
            q = e; if (*q == ',') q++; else break;
        }
        if (core.empty()) { core.push_back(cpu); seen[cpu] = 1; }
        cores.push_back(core);
    }
    int rsv = -1;                                       /* the GPU host thread's core stays free (else the last 2-thread core) */
    if (host_pinned)
        for (int i = 0; i < (int)cores.size() && rsv < 0; i++) for (int x : cores[i]) if (CPU_ISSET(x, &host)) { rsv = i; break; }
    for (int i = (int)cores.size() - 1; i >= 0 && rsv < 0; i--) if (cores[i].size() >= 2) rsv = i;
    if (rsv < 0 || cores.size() < 2) return;
    int extra = -1;                                     /* the host core's other CPU, when start() planned a worker for it */
    if (host_pinned && c->host_cpu >= 0 && CPU_ISSET(c->host_cpu, &host))
        for (int x : cores[rsv]) if (x != c->host_cpu && CPU_ISSET(x, &host)) { extra = x; break; }
    c->mask0 = cs;                                      /* unpinned workers: every CPU but the reserved core (or but the host CPU) */
    if (extra >= 0) CPU_CLR(c->host_cpu, &c->mask0);
    else for (int x : cores[rsv]) CPU_CLR(x, &c->mask0);
    int t = 0; bool pair = false;
    for (int i = 0; i < (int)cores.size() && t < nth; i++) {
        if (i == rsv) continue;
        for (size_t k = 0; k < cores[i].size() && t < nth; k++) { c->wcpu[t] = cores[i][k]; c->wscalar[t] = k == 1; pair |= k == 1; t++; }
    }
    if (extra >= 0 && t < nth) { c->wcpu[t] = extra; c->wscalar[t] = 0; t++; }   /* IFMA: its sibling is the spinning host thread */
    c->hybrid_ok = pair;
#endif
}
static void worker(Ctx *c, int tid) {
    try { worker_body(c, tid); }                       /* a failed allocation ends this worker, never the process */
    catch (const std::exception &e) { printf("  CPU co-grind: worker %d stopped (%s)\n", tid, e.what()); fflush(stdout); }
    catch (...) { printf("  CPU co-grind: worker %d stopped\n", tid); fflush(stdout); }
}
static Ctx *g_ctx = nullptr;
/* win3: the GPU's 128 window patterns (actual push indices, ascending). */
static void start(const digest_params_t *dp, const uint8_t win3[][3], int nwin, int cut, int early) {
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
#ifdef CPU_COUNT
    { cpu_set_t cs; CPU_ZERO(&cs); if (g_mask0_ok) ncpu = CPU_COUNT(&g_mask0); else if (sched_getaffinity(0, sizeof cs, &cs) == 0) ncpu = CPU_COUNT(&cs); }
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
    int host_cpu = -1;
#ifdef CPU_SET
    {   /* The host producers pin this (GPU host) thread to one core before start(). It spins on one CPU of that core;
         * the other CPU gets one more worker (placed there by smt_plan), instead of staying idle. */
        cpu_set_t cur; CPU_ZERO(&cur); __builtin_cpu_init();
        if (g_mask0_ok && sched_getaffinity(0, sizeof cur, &cur) == 0 && CPU_COUNT(&cur) == 2 && CPU_COUNT(&g_mask0) > 2 &&
            nth == (int)CPU_COUNT(&g_mask0) - QSB_CPU_RESERVE && !getenv("QSB_CPU_NOEXTRA") && !getenv("QSB_CPU_NOPIN") &&
            __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512ifma") && !getenv("QSB_CPU_NOVEC")) {   /* only smt_plan places it */
            const int cpu = sched_getcpu();
            if (cpu >= 0 && CPU_ISSET(cpu, &cur)) { host_cpu = cpu; nth += 1; }
        }
    }
#endif
    if (const char *e = getenv("QSB_CPU_THREADS_ENV")) { nth = atoi(e); host_cpu = -1; }   /* dev override */
    if (nth < 1 || dp->n != 150 || cut != 137 || early != 6) { printf("  CPU co-grind: off (%d threads)\n", nth); return; }
    Ctx *c = new Ctx(); c->dp = dp; c->nthreads = nth; c->cut = cut; c->early = early; c->host_cpu = host_cpu;
#ifdef CPU_SET
    c->host_tid = (pid_t)syscall(SYS_gettid);           /* start() runs on the GPU host thread */
#endif
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
#if QCPU_SHANI
    if (c->shani && !getenv("QSB_CPU_NOFAST")) { try { prep_fast(c); } catch (...) { c->fast = false; } }
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
    try { std::thread([c, fax, fay, nth]() {
      try {
  #ifdef SCHED_IDLE
          struct sched_param sp; sp.sched_priority = 0; sched_setscheduler(0, SCHED_IDLE, &sp);
  #endif
  #ifdef CPU_SET
          /* This thread inherits the GPU host thread's CPU mask, which a host path may have pinned to one
           * core: widen it to the pre-main() set minus that core, so the table build and the workers
           * (which inherit it) use every other CPU. */
          if (g_mask0_ok) {
              cpu_set_t m = g_mask0, h; CPU_ZERO(&h);
              if (c->host_tid > 0 && sched_getaffinity(c->host_tid, sizeof h, &h) == 0 && CPU_COUNT(&h) < CPU_COUNT(&m)) {
                  cpu_set_t t = m;
                  for (int x = 0; x < CPU_SETSIZE; x++) if (CPU_ISSET(x, &h)) CPU_CLR(x, &t);
                  if (CPU_COUNT(&t) > 0) m = t;
              }
              pthread_setaffinity_np(pthread_self(), sizeof m, &m);
          }
  #endif
          const double t0 = now_s();
          if (!build_table(*c, fax, fay, nth)) { printf("  CPU co-grind: off (table allocation)\n"); fflush(stdout); return; }
  #ifdef QSB_CPU_DEVBENCH
          if (const char *e = getenv("QSB_CPU_DEVCAND")) c->dev_limit = strtoull(e, nullptr, 10);
          c->dev_live = nth;
  #endif
          c->t_ready = now_s();
          printf("  CPU co-grind: table %d lookups (%d..%d-bit signed digits%s), %zu MiB (budget %zu MiB), built in %.2fs\n", c->nw,
                 c->wbits[c->nw - 1], c->wbits[0], c->cfold ? ", C folded into the last window" : "", c->table_bytes >> 20, c->budget >> 20,
                 c->t_ready - t0);
          fflush(stdout);
          if (c->vec && !getenv("QSB_CPU_NOPIN")) smt_plan(c, nth);
          else for (int t = 0; t < 512; t++) { c->wcpu[t] = -1; c->wscalar[t] = 0; }
          for (int t = 0; t < nth; t++) { try { std::thread(worker, c, t).detach(); } catch (...) { break; } }
          if (c->vec && (c->s16 || c->hybrid_ok)) {       /* choose the hashing and the hybrid on this host (ABBA, 3 s each) */
              if (const char *fm = getenv("QSB_CPU_MODE")) { c->mode = atoi(fm) & 3; if (!(c->mode & 1) && c->hybrid_ok) c->unpin = true; return; }
              auto ab = [&](int m0, int m1, double r[2]) {
                  r[0] = r[1] = 0;
                  for (int round = 0; round < 2; round++)
                      for (int k = 0; k < 2; k++) {
                          const int w = round ? 1 - k : k;
                          c->mode = w ? m1 : m0; usleep(300000);
                          const uint64_t c0 = c->cand.load(); const double s0 = now_s();
                          usleep(3000000);
                          r[w] += (double)(c->cand.load() - c0) / (now_s() - s0) / 2;
                      }
              };
              usleep(1500000);
              int best = 0; double r[2];
              if (c->s16) {
                  ab(0, 2, r); if (r[1] > 1.02 * r[0]) best = 2;
                  printf("  CPU co-grind: calibration 4-lane SHA-NI %.2f M/s, 16-lane AVX-512 SHA %.2f M/s\n", r[0] / 1e6, r[1] / 1e6);
              }
              if (c->hybrid_ok) {
                  ab(best, best | 1, r); if (r[1] > 1.02 * r[0]) best |= 1;
                  printf("  CPU co-grind: calibration all-IFMA %.2f M/s, SMT hybrid %.2f M/s\n", r[0] / 1e6, r[1] / 1e6);
              }
              c->mode = best;
              if (!(best & 1) && c->hybrid_ok) c->unpin = true;   /* no hybrid: back to the unpinned scheduling */
              {   /* then the table-row prefetch distance in the chosen mode: memory latency differs between hosts */
                  const int p0 = QSB_CPU_PF, p1 = 8; double rp[2] = {0, 0};
                  for (int round = 0; round < 2; round++)
                      for (int k = 0; k < 2; k++) {
                          const int w = round ? 1 - k : k;
                          c->pf = w ? p1 : p0; usleep(300000);
                          const uint64_t c0 = c->cand.load(); const double s0 = now_s();
                          usleep(3000000);
                          rp[w] += (double)(c->cand.load() - c0) / (now_s() - s0) / 2;
                      }
                  c->pf = rp[1] > 1.02 * rp[0] ? p1 : p0;
                  printf("  CPU co-grind: calibration prefetch %d groups %.2f M/s, %d groups %.2f M/s\n", p0, rp[0] / 1e6, p1, rp[1] / 1e6);
              }
              printf("  CPU co-grind: using %s hashing, %s, prefetch %d groups\n", best & 2 ? "16-lane AVX-512" : "4-lane SHA-NI", best & 1 ? "SMT hybrid" : "all-IFMA", c->pf.load());
              fflush(stdout);
          }
      } catch (...) { printf("  CPU co-grind: off (builder)\n"); fflush(stdout); }
    }).detach(); } catch (...) { printf("  CPU co-grind: off (thread)\n"); fflush(stdout); return; }
    g_ctx = c;
    printf("  CPU co-grind: %d threads (of %ld CPUs), %s, %s, %d window patterns per epoch disjoint from the GPU's %d\n",
           nth, ncpu, c->vec ? "8-lane IFMA" : "scalar", c->shani ? (c->fast ? "4-lane SHA-NI, precomputed schedule" : "4-lane SHA-NI") : "OpenSSL SHA-256",
           c->ncwin, nwin);
    if (c->fast) printf("  CPU co-grind: %d block-0 variants, %d pattern blocks after block 0\n", c->nvar, c->nblk);
    fflush(stdout);
#ifdef QSB_CPU_DEVBENCH
    /* Dev only (never in a ranked build): grind on the CPU alone for QSB_CPU_DEVBENCH seconds after
     * the table is ready, report the rate and exit. */
    while (c->t_ready == 0) usleep(10000);
    if (getenv("QSB_CPU_DEVCAND")) {                 /* deterministic run: wait for the workers to stop */
        while (c->dev_live.load() > 0) usleep(10000);
        if (c->out) fflush(c->out);
        printf("DEVCAND %llu candidates, %u hits\n", (unsigned long long)c->cand.load(), c->hits.load());
        fflush(stdout); _exit(0);
    }
    const uint64_t c0 = c->cand.load(); sleep(QSB_CPU_DEVBENCH);
    const double el = now_s() - c->t_ready; const uint64_t c1 = c->cand.load();
    { const double t = (double)(c->tsc[0] + c->tsc[1] + c->tsc[2]) / 100.0;
      printf("DEVBENCH phases: hash+digits %.1f%%  windows %.1f%%  final+gate %.1f%%  (%.0f ns/cand/thread)\n", c->tsc[0] / t, c->tsc[1] / t, c->tsc[2] / t,
             100.0 * t / (double)c->cand.load()); }
    printf("DEVBENCH cpu %.3f M/s (%llu cand in %.1fs, %u hits)\n", (c1 - c0) / (double)QSB_CPU_DEVBENCH / 1e6,
           (unsigned long long)c1, el, c->hits.load());
    fflush(stdout); _exit(0);
#endif
    fflush(stdout);
}
static uint64_t candidates() { return g_ctx ? g_ctx->cand.load() : 0; }
static uint32_t hits() { return g_ctx ? g_ctx->hits.load() : 0; }
}  // namespace qcpu
