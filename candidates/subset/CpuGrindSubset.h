#pragma once
/* Host-CPU co-grinder for the subset track. The field arithmetic, the windowed host table and the
 * batch-affine additions derive from Ryun1's pinning CpuGrind.h (public submission 7a75fa50, GPL-3);
 * the candidate enumeration, preimage hashing and hit publication are subset's. The table is sized at
 * run time: signed digits in mixed-width windows (12 windows of 20-22 bits, 1.06 GiB, 11 additions per
 * candidate where memory allows; at worst 15 windows, 68 MiB), on 2 MiB pages, each window's rows
 * prefetched during the previous window's backward pass.
 *
 * Candidates are disjoint from the GPU's: the GPU grinds every epoch (6 early omissions below
 * the cut) with its 128 window-omission patterns (h_win3); the CPU grinds epochs t, t+T, t+2T, ...
 * (T threads) with the other 158 of the C(13,3)=286 window patterns. Every CPU hit passes the same
 * exact OpenSSL gate as the GPU's tentatives (qsb_hv_check) before it is appended to
 * results/digest_hit_cpu.txt, which the harness collects with the GPU's hit file. Workers run at
 * SCHED_IDLE, so they never delay the GPU host thread; QSB_CPU_GRIND=0 compiles it out.
 *
 * 2026-09-26 (GLV12xl_x_wt_nx): the batch inversions invert one scalar (Bernstein-Yang safegcd on the
 * integer pipes, the 8 lanes combined and split with 6 permuted multiplications) instead of 255 vector
 * squarings; the window's backward pass runs one group at a time in registers; x3 and y3 subtract from
 * the folded product columns and carry once; multiplication-only elements skip the limb masks (IFMA reads
 * bits 51:0); products accumulate low and high partial products separately (shorter chains); the
 * word-message SHA-256 is VEX-encoded and starts from the IV; ec8_final stores its outputs with vector
 * transposes; the table conversion uses VBMI2 funnel shifts. Same candidates, gate and records. */
#include <openssl/sha.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <utility>
#include <sched.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
#define QSB_CPU_BATCH 2048         /* candidates per batch: the per-worker EC state (~0.5 MB) stays in L2 with both SMT threads busy */
#endif
/* Fixed-base table geometry, chosen at run time (Geo): signed digits, the fewest windows whose table fits in
 * QSB_CPU_TAB_FRAC of the memory this process may still use (MemAvailable and the cgroup limit), capped at
 * QSB_CPU_TAB_CAP_MB and at no fewer than QSB_CPU_NW_MIN windows; never more than 15 windows (68 MiB). */
#ifndef QSB_CPU_NW_MIN
#define QSB_CPU_NW_MIN 12
#endif
#ifndef QSB_CPU_TAB_CAP_MB
#define QSB_CPU_TAB_CAP_MB 4096
#endif
#ifndef QSB_CPU_TAB_FRAC
#define QSB_CPU_TAB_FRAC 0.25
#endif
#ifndef QSB_CPU_PFD
#define QSB_CPU_PFD 8              /* table-row prefetch distance, in groups of 8 candidates */
#endif

namespace qcpu {
static const int NWMAX = 16;
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
 * (probability ~2^-240; the candidate is dropped, never published). neg[k] (optional): add -t[k]. */
static void batch_add(pt *acc, const pt *const *tp, uint8_t *inf, uint8_t *bad, int n,
                      fe *d, fe *pre, const uint8_t *neg = nullptr) {
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
        const bool ng = neg && neg[k];
        if (inf[k]) { acc[k] = *tp[k]; if (ng) { const fe z0 = {{0, 0, 0, 0}}; fe_sub(acc[k].y, z0, acc[k].y); } inf[k] = 0; continue; }
        fe dinv; fe_mul(dinv, inv, pre[k]); fe_mul(inv, inv, d[k]);
        fe lam, t, x3, y3;
        if (ng) { fe_add(t, tp[k]->y, acc[k].y); const fe z0 = {{0, 0, 0, 0}}; fe_sub(t, z0, t); }   /* -ty - y */
        else fe_sub(t, tp[k]->y, acc[k].y);
        fe_mul(lam, t, dinv);
        fe_sqr(x3, lam); fe_sub(x3, x3, acc[k].x); fe_sub(x3, x3, tp[k]->x);
        fe_sub(t, acc[k].x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, acc[k].y);
        acc[k].x = x3; acc[k].y = y3;
    }
}

/* Table geometry. Window i covers bits [off[i], off[i] + wid[i]) of z. Signed (sgn): windows 0..nw-2 take
 * digits in (-2^(wid-1), 2^(wid-1)] and pass a carry up, the top window takes its bits plus the carry
 * (0..2^wid); unsigned: digits 0..2^wid - 1. Window i holds ent[i] points (j + 1) * 2^off[i] * A,
 * j < ent[i] = the largest |digit|, at table[base[i] + j]. A digit is stored as |d| | (d < 0) << 31;
 * a zero digit skips the window (scalar path) or drops the candidate (8-lane path). */
struct Geo { int nw = 0; bool sgn = false; int off[NWMAX] = {0}, wid[NWMAX] = {0}; uint32_t ent[NWMAX] = {0}; size_t base[NWMAX] = {0}; size_t total = 0; };
static Geo geo_make(int nw, bool sgn) {
    Geo g; if (nw < 9) nw = 9; if (nw > NWMAX) nw = NWMAX;
    g.nw = nw; g.sgn = sgn;
    if (!sgn) {                                        /* unsigned: 256 bits in nw near-equal windows */
        const int w = 256 / nw, r = 256 - w * nw;
        for (int i = 0; i < nw; i++) g.wid[i] = w + (i < r);
    } else {                                           /* signed: nw - 1 windows of w or w + 1 bits + a b-bit top, least entries */
        double best = -1; int bb = 1;
        for (int b = 1; b <= 30; b++) {
            const int k = nw - 1, rem = 256 - b, w = rem / k, r = rem - w * k;
            if (w + (r > 0) > 30) continue;
            const double cost = r * ldexp(1.0, w) + (k - r) * ldexp(1.0, w - 1) + ldexp(1.0, b);
            if (best < 0 || cost < best) { best = cost; bb = b; }
        }
        const int k = nw - 1, rem = 256 - bb, w = rem / k, r = rem - w * k;
        for (int i = 0; i < k; i++) g.wid[i] = w + (i < r);
        g.wid[k] = bb;
    }
    size_t o = 0; int off = 0;
    for (int i = 0; i < nw; i++) {
        g.off[i] = off; off += g.wid[i];
        g.ent[i] = !sgn ? (uint32_t)((1u << g.wid[i]) - 1) : (i < nw - 1 ? 1u << (g.wid[i] - 1) : 1u << g.wid[i]);
        g.base[i] = o; o += g.ent[i];
    }
    g.total = o;
    return g;
}
/* Digits of z (zb: 8 SHA state words per candidate, h0 = most significant) into ds[i * B + k]. */
static void recode_scalar(const Geo &g, const uint32_t *zb, uint32_t *ds, int B) {
    for (int k = 0; k < B; k++) {
        const uint32_t *h = zb + (size_t)k * 8;
        uint64_t zl[5];
        for (int i = 0; i < 4; i++) zl[i] = ((uint64_t)h[6 - 2 * i] << 32) | h[7 - 2 * i];
        zl[4] = 0;
        uint32_t cy = 0;
        for (int i = 0; i < g.nw; i++) {
            const int li = g.off[i] >> 6, sh = g.off[i] & 63, w = g.wid[i];
            uint64_t v = zl[li] >> sh;
            if (sh + w > 64) v |= zl[li + 1] << (64 - sh);
            uint32_t u = (uint32_t)(v & ((1ULL << w) - 1)), ng = 0;
            if (g.sgn) {
                u += cy;
                if (i < g.nw - 1) { ng = u > (1u << (w - 1)); if (ng) u = (1u << w) - u; cy = ng; }
            }
            ds[(size_t)i * B + k] = u | (ng << 31);
        }
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
/* Lazy normalization for elements that only ever feed IFMA multiplications ("mul-only": the chain
 * products, PRE, dinv, lam, the y differences and D): the carries are propagated exactly as in
 * fe8_carry, but limbs 0..3 keep the bits they carried out (bits 52 and up). IFMA reads only bits 51:0
 * of a multiplicand, so such an element multiplies exactly like its normalized twin; it must never be an
 * operand of fe8_sub/fe8_add/fe8_sub_sgn, a subtrahend of the fused ops, or be canonicalized. Limb 4
 * is masked as before (its bits 48..51 are inside the multiplicand window). 11 ops instead of 15. */
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_carry_lz(fe8 &r) {
    const __m512i M48 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL), K = _mm512_set1_epi64(0x1000003D1ULL);
    __m512i c;
    c = _mm512_srli_epi64(r.l[4], 48); r.l[4] = _mm512_and_si512(r.l[4], M48);
    r.l[0] = _mm512_madd52lo_epu64(r.l[0], c, K);
    c = _mm512_srli_epi64(r.l[0], 52); r.l[1] = _mm512_add_epi64(r.l[1], c);
    c = _mm512_srli_epi64(r.l[1], 52); r.l[2] = _mm512_add_epi64(r.l[2], c);
    c = _mm512_srli_epi64(r.l[2], 52); r.l[3] = _mm512_add_epi64(r.l[3], c);
    c = _mm512_srli_epi64(r.l[3], 52); r.l[4] = _mm512_add_epi64(r.l[4], c);
}
/* Fold a 10-column product (columns < 9 * 2^52, value < 2^514) to five columns o[0..4] of the same residue,
 * each a sum of at most 12 terms below 2^52 (so < 2^56): value = L + H*2^260, 2^260 = R = 0x1000003D10 mod p.
 * Only the high columns c5..c9 are normalized (they become IFMA multiplicands of R); the low columns take
 * the folded products unnormalized. fe8_red = fold + fe8_carry (fe8_carry's single pass: fold at 2^256 of
 * limb 4's bits >= 48, c = l4 >> 48 < 2^9 so c*K < 2^42; then limbs 0..3 -> 52 bits) normalizes the sum.
 * The fused fe8_sqr_sub2 / fe8_mul_sub subtract from the folded columns and carry once. */
static inline __attribute__((always_inline, target("avx512f,avx512ifma")))
void fe8_fold(__m512i *o, __m512i c0, __m512i c1, __m512i c2, __m512i c3, __m512i c4,
              __m512i c5, __m512i c6, __m512i c7, __m512i c8, __m512i c9) {
    const __m512i Z = _mm512_setzero_si512();
#define LO(acc, x, y) acc = _mm512_madd52lo_epu64(acc, x, y)
#define HI(acc, x, y) acc = _mm512_madd52hi_epu64(acc, x, y)
    const __m512i M = F8_M52;
    /* normalize the high columns c5..c9 to 52-bit limbs (carry into c10); c4 keeps its high bits */
    __m512i t;
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
    /* c5b < 2^43 (weight 2^260 -> R): lo into limb 0, hi into limb 1 */
    o[0] = _mm512_madd52lo_epu64(c0, c5b, R);
    o[1] = _mm512_madd52hi_epu64(c1, c5b, R);
    o[2] = c2; o[3] = c3; o[4] = c4;
}
static inline __attribute__((always_inline, target("avx512f,avx512ifma")))
void fe8_red(fe8 &r, __m512i c0, __m512i c1, __m512i c2, __m512i c3, __m512i c4,
             __m512i c5, __m512i c6, __m512i c7, __m512i c8, __m512i c9) {
    fe8 o; fe8_fold(o.l, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);
    fe8_carry(o);
    r = o;
}
/* The 10 product columns of a*b (25 low + 25 high partial products) and of a^2 (the ten cross products once,
 * doubled with one shift per column, plus the five squares: 30 IFMA instead of 50; the same column bounds). */
static inline __attribute__((always_inline, target("avx512f,avx512ifma")))
void fe8_mul_cols(__m512i *c, const fe8 &a, const fe8 &b) {
    const __m512i Z = _mm512_setzero_si512();
    /* low and high partial products in separate accumulators (chains of at most 5 IFMA instead of 9), summed per column */
    __m512i c0 = Z, c1 = Z, c2 = Z, c3 = Z, c4 = Z, c5 = Z, c6 = Z, c7 = Z, c8 = Z, d1 = Z, d2 = Z, d3 = Z, d4 = Z, d5 = Z, d6 = Z, d7 = Z, d8 = Z, d9 = Z;
    const __m512i a0 = a.l[0], a1 = a.l[1], a2 = a.l[2], a3 = a.l[3], a4 = a.l[4];
    const __m512i b0 = b.l[0], b1 = b.l[1], b2 = b.l[2], b3 = b.l[3], b4 = b.l[4];
    LO(c0,a0,b0); HI(d1,a0,b0);
    LO(c1,a0,b1); LO(c1,a1,b0); HI(d2,a0,b1); HI(d2,a1,b0);
    LO(c2,a0,b2); LO(c2,a1,b1); LO(c2,a2,b0); HI(d3,a0,b2); HI(d3,a1,b1); HI(d3,a2,b0);
    LO(c3,a0,b3); LO(c3,a1,b2); LO(c3,a2,b1); LO(c3,a3,b0); HI(d4,a0,b3); HI(d4,a1,b2); HI(d4,a2,b1); HI(d4,a3,b0);
    LO(c4,a0,b4); LO(c4,a1,b3); LO(c4,a2,b2); LO(c4,a3,b1); LO(c4,a4,b0);
    HI(d5,a0,b4); HI(d5,a1,b3); HI(d5,a2,b2); HI(d5,a3,b1); HI(d5,a4,b0);
    LO(c5,a1,b4); LO(c5,a2,b3); LO(c5,a3,b2); LO(c5,a4,b1); HI(d6,a1,b4); HI(d6,a2,b3); HI(d6,a3,b2); HI(d6,a4,b1);
    LO(c6,a2,b4); LO(c6,a3,b3); LO(c6,a4,b2); HI(d7,a2,b4); HI(d7,a3,b3); HI(d7,a4,b2);
    LO(c7,a3,b4); LO(c7,a4,b3); HI(d8,a3,b4); HI(d8,a4,b3);
    LO(c8,a4,b4); HI(d9,a4,b4);
    c[0] = c0; c[1] = _mm512_add_epi64(c1, d1); c[2] = _mm512_add_epi64(c2, d2); c[3] = _mm512_add_epi64(c3, d3); c[4] = _mm512_add_epi64(c4, d4);
    c[5] = _mm512_add_epi64(c5, d5); c[6] = _mm512_add_epi64(c6, d6); c[7] = _mm512_add_epi64(c7, d7); c[8] = _mm512_add_epi64(c8, d8); c[9] = d9;
}
static inline __attribute__((always_inline, target("avx512f,avx512ifma")))
void fe8_sqr_cols(__m512i *c, const fe8 &a) {
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
    c[0] = c0; c[1] = c1; c[2] = c2; c[3] = c3; c[4] = c4; c[5] = c5; c[6] = c6; c[7] = c7; c[8] = c8; c[9] = c9;
}
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_mul(fe8 &r, const fe8 &a, const fe8 &b) {
    __m512i c[10]; fe8_mul_cols(c, a, b);
    fe8_red(r, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]);
}
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_mul_lz(fe8 &r, const fe8 &a, const fe8 &b) {
    __m512i c[10]; fe8_mul_cols(c, a, b);
    fe8 o; fe8_fold(o.l, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]);
    fe8_carry_lz(o); r = o;
}
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_sqr(fe8 &r, const fe8 &a) {
    __m512i c[10]; fe8_sqr_cols(c, a);
    fe8_red(r, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]);
}
/* Fused r = a^2 - b - c and r = a*b - c with ONE carry pass: the folded product columns (< 2^56) plus 4p
 * minus the normalized operands (limbs < 2^52, limb 4 < 2^48 + 2^7). Every limb stays non-negative
 * (4p_i - 2 * 2^52 > 2^53 for limbs 0..3, 4p_4 - 2 * (2^48 + 2^7) > 2^49 - 2^8 for limb 4) and below
 * 2^56 + 2^54 < 2^57, which fe8_carry normalizes (limb 4 ends < 2^48 + 2^5). The residue is the same as
 * the unfused sequence's; only the (non-canonical) representative may differ, which no output sees:
 * records hold canonical x and the parity of canonical y. */
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_sqr_sub2(fe8 &r, const fe8 &a, const fe8 &b, const fe8 &c) {
    const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
                  P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4);
    __m512i k[10]; fe8_sqr_cols(k, a);
    fe8 o; fe8_fold(o.l, k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8], k[9]);
    o.l[0] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(o.l[0], P0), b.l[0]), c.l[0]);
    o.l[1] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(o.l[1], P1), b.l[1]), c.l[1]);
    o.l[2] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(o.l[2], P1), b.l[2]), c.l[2]);
    o.l[3] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(o.l[3], P1), b.l[3]), c.l[3]);
    o.l[4] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(o.l[4], P4), b.l[4]), c.l[4]);
    fe8_carry(o);
    r = o;
}
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_mul_sub(fe8 &r, const fe8 &a, const fe8 &b, const fe8 &c) {
    const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
                  P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4);
    __m512i k[10]; fe8_mul_cols(k, a, b);
    fe8 o; fe8_fold(o.l, k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8], k[9]);
    o.l[0] = _mm512_sub_epi64(_mm512_add_epi64(o.l[0], P0), c.l[0]);
    o.l[1] = _mm512_sub_epi64(_mm512_add_epi64(o.l[1], P1), c.l[1]);
    o.l[2] = _mm512_sub_epi64(_mm512_add_epi64(o.l[2], P1), c.l[2]);
    o.l[3] = _mm512_sub_epi64(_mm512_add_epi64(o.l[3], P1), c.l[3]);
    o.l[4] = _mm512_sub_epi64(_mm512_add_epi64(o.l[4], P4), c.l[4]);
    fe8_carry(o);
    r = o;
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
/* r = (a or -a in the lanes of m) - b (inputs normalized): per limb a + 4p - b, or 4p - a - b in the lanes of m.
 * Both stay non-negative: a_i + b_i < 2^53 < 4p_i for limbs 0..3, and a normalized limb 4 is < 2^48 + 2^6
 * (fe8_carry: masked to 48 bits, then a carry < 2^6), so a_4 + b_4 < 2^49 + 2^7 < 4p_4 = 2^50 - 4. */
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_sub_sgn(fe8 &r, const fe8 &a, const fe8 &b, __mmask8 m) {
    const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
                  P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4), Z = _mm512_setzero_si512();
    r.l[0] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[0], m, Z, a.l[0]), P0), b.l[0]);
    r.l[1] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[1], m, Z, a.l[1]), P1), b.l[1]);
    r.l[2] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[2], m, Z, a.l[2]), P1), b.l[2]);
    r.l[3] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[3], m, Z, a.l[3]), P1), b.l[3]);
    r.l[4] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[4], m, Z, a.l[4]), P4), b.l[4]);
    fe8_carry(r);
}
/* Mul-only variants of fe8_sub and fe8_sub_sgn (normalized inputs; the output only feeds IFMA). */
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_sub_lz(fe8 &r, const fe8 &a, const fe8 &b) {
    const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
                  P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4);
    fe8 o;
    o.l[0] = _mm512_sub_epi64(_mm512_add_epi64(a.l[0], P0), b.l[0]);
    o.l[1] = _mm512_sub_epi64(_mm512_add_epi64(a.l[1], P1), b.l[1]);
    o.l[2] = _mm512_sub_epi64(_mm512_add_epi64(a.l[2], P1), b.l[2]);
    o.l[3] = _mm512_sub_epi64(_mm512_add_epi64(a.l[3], P1), b.l[3]);
    o.l[4] = _mm512_sub_epi64(_mm512_add_epi64(a.l[4], P4), b.l[4]);
    fe8_carry_lz(o); r = o;
}
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_sub_sgn_lz(fe8 &r, const fe8 &a, const fe8 &b, __mmask8 m) {
    const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
                  P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4), Z = _mm512_setzero_si512();
    fe8 o;
    o.l[0] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[0], m, Z, a.l[0]), P0), b.l[0]);
    o.l[1] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[1], m, Z, a.l[1]), P1), b.l[1]);
    o.l[2] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[2], m, Z, a.l[2]), P1), b.l[2]);
    o.l[3] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[3], m, Z, a.l[3]), P1), b.l[3]);
    o.l[4] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[4], m, Z, a.l[4]), P4), b.l[4]);
    fe8_carry_lz(o); r = o;
}
/* r = a - b - c (inputs normalized) with one carry pass: a + 8p - b - c, every limb non-negative
 * (b_i + c_i < 2^53 <= 8p_i for limbs 0..3, < 2^50 < 8p_4) and < 2^56, which fe8_carry normalizes. */
static inline __attribute__((target("avx512f,avx512ifma")))
void fe8_sub2(fe8 &r, const fe8 &a, const fe8 &b, const fe8 &c) {
    const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 8), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 8),
                  P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 8);
    r.l[0] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(a.l[0], P0), b.l[0]), c.l[0]);
    r.l[1] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(a.l[1], P1), b.l[1]), c.l[1]);
    r.l[2] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(a.l[2], P1), b.l[2]), c.l[2]);
    r.l[3] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(a.l[3], P1), b.l[3]), c.l[3]);
    r.l[4] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(a.l[4], P4), b.l[4]), c.l[4]);
    fe8_carry(r);
}
/* 8-lane batch-affine EC pipeline for the CPU co-grinder (AVX-512 IFMA). Requires fe8.h, the
 * scalar fe/pt types and a table of canonical affine points (4x64 limbs, 64 B per point). */
#define Q8T __attribute__((target("avx512f,avx512ifma")))

/* fe8 copy as five vector moves (a plain struct assignment compiles to rep movsq here) */
Q8T static inline void fe8_cp(fe8 &r, const fe8 &a) {
    for (int i = 0; i < 5; i++) _mm512_store_si512(&r.l[i], _mm512_load_si512(&a.l[i]));
}
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
#ifndef QSB_CPU_VBMI2
#define QSB_CPU_VBMI2 1            /* funnel shifts (vpshrdq) in the 4x64 -> 5x52 conversion; every IFMA CPU in service has VBMI2 */
#endif
#if QSB_CPU_VBMI2
#define Q8TX __attribute__((target("avx512f,avx512ifma,avx512vbmi2")))
Q8TX static inline void fe8_from64(fe8 &r, __m512i a0, __m512i a1, __m512i a2, __m512i a3) {
    const __m512i M = F8_M52;
    r.l[0] = _mm512_and_si512(a0, M);
    r.l[1] = _mm512_and_si512(_mm512_shrdi_epi64(a0, a1, 52), M);   /* (a1:a0) >> 52 */
    r.l[2] = _mm512_and_si512(_mm512_shrdi_epi64(a1, a2, 40), M);
    r.l[3] = _mm512_and_si512(_mm512_shrdi_epi64(a2, a3, 28), M);
    r.l[4] = _mm512_srli_epi64(a3, 16);
}
#else
#define Q8TX Q8T
Q8T static inline void fe8_from64(fe8 &r, __m512i a0, __m512i a1, __m512i a2, __m512i a3) {
    const __m512i M = F8_M52;
    r.l[0] = _mm512_and_si512(a0, M);
    r.l[1] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(a0, 52), _mm512_slli_epi64(a1, 12)), M);
    r.l[2] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(a1, 40), _mm512_slli_epi64(a2, 24)), M);
    r.l[3] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(a2, 28), _mm512_slli_epi64(a3, 36)), M);
    r.l[4] = _mm512_srli_epi64(a3, 16);
}
#endif
/* Load 8 table points (one 64 B row each: x0..x3 y0..y3) and transpose to SoA. */
Q8TX static inline void pt8_load(fe8 &x, fe8 &y, const pt *const *rows) {
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
/* Canonical form of 8 normalized elements (value v < 2^256 + 2^215 < 2p): v - p when v + (2^256 - p)
 * reaches 2^256, else v. fe8_ge_p returns that mask and, with u != nullptr, v + 2^256 - p mod 2^256. */
Q8T static inline __mmask8 fe8_ge_p(const fe8 &a, fe8 *u) {
    const __m512i M = F8_M52, K = _mm512_set1_epi64(0x1000003D1ULL);
    __m512i u0 = _mm512_add_epi64(a.l[0], K), c;
    c = _mm512_srli_epi64(u0, 52); u0 = _mm512_and_si512(u0, M); __m512i u1 = _mm512_add_epi64(a.l[1], c);
    c = _mm512_srli_epi64(u1, 52); u1 = _mm512_and_si512(u1, M); __m512i u2 = _mm512_add_epi64(a.l[2], c);
    c = _mm512_srli_epi64(u2, 52); u2 = _mm512_and_si512(u2, M); __m512i u3 = _mm512_add_epi64(a.l[3], c);
    c = _mm512_srli_epi64(u3, 52); u3 = _mm512_and_si512(u3, M); __m512i u4 = _mm512_add_epi64(a.l[4], c);
    const __mmask8 ge = _mm512_test_epi64_mask(u4, _mm512_set1_epi64((long long)~0x0FFFFFFFFFFFFULL));   /* u >= 2^256 */
    if (u) { u->l[0] = u0; u->l[1] = u1; u->l[2] = u2; u->l[3] = u3; u->l[4] = _mm512_and_si512(u4, _mm512_set1_epi64(0x0FFFFFFFFFFFFULL)); }
    return ge;
}
/* canonical x of 8 lanes as 4x64 limbs (w[k][lane]) */
Q8T static inline void fe8_canon64(uint64_t w[4][8], const fe8 &a) {
    fe8 u; const __mmask8 ge = fe8_ge_p(a, &u);
    __m512i r[5];
    for (int i = 0; i < 5; i++) r[i] = _mm512_mask_blend_epi64(ge, a.l[i], u.l[i]);
    _mm512_store_si512(w[0], _mm512_or_si512(r[0], _mm512_slli_epi64(r[1], 52)));
    _mm512_store_si512(w[1], _mm512_or_si512(_mm512_srli_epi64(r[1], 12), _mm512_slli_epi64(r[2], 40)));
    _mm512_store_si512(w[2], _mm512_or_si512(_mm512_srli_epi64(r[2], 24), _mm512_slli_epi64(r[3], 28)));
    _mm512_store_si512(w[3], _mm512_or_si512(_mm512_srli_epi64(r[3], 36), _mm512_slli_epi64(r[4], 16)));
}
/* parity of the canonical y of 8 lanes (p is odd: v - p flips v's parity) */
Q8T static inline __mmask8 fe8_parity(const fe8 &a) {
    return (__mmask8)(_mm512_test_epi64_mask(a.l[0], _mm512_set1_epi64(1)) ^ fe8_ge_p(a, nullptr));
}
/* x^(p-2) for one 8-lane element: libsecp256k1's secp256k1_fe_inv addition chain
 * (255 squarings, 15 multiplications). Kept for QSB_CPU_SCALAR_INV=0 and as the reference. */
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
#ifndef QSB_CPU_SCALAR_INV
#define QSB_CPU_SCALAR_INV 1       /* batch inversions: one scalar safegcd inverse instead of 255 vector squarings */
#endif
#if QSB_CPU_SCALAR_INV
/* ---- Variable-time scalar inverse mod p for the batch inversions: Bernstein-Yang "safegcd" divsteps in
 * 62-bit batches, after libsecp256k1's secp256k1_modinv64_var (MIT; notice in COPYING-secp256k1). It runs on
 * the integer pipes, so the inversion no longer occupies the vector pipes with 255 dependent squarings. */
struct s62 { int64_t v[5]; };                         /* signed, 62-bit limbs (the top one carries the sign) */
struct trans2x2 { int64_t u, v, q, r; };
static const s62 S62_P = {{-0x1000003D1LL, 0, 0, 0, 256}};   /* p = 2^256 - 0x1000003D1 */
static const uint64_t S62_PINV = 0x27C7F6E22DDACACFULL;      /* p^-1 mod 2^62 */
/* 62 divsteps on the low limbs (f0, g0) of (f, g); the transition matrix t and the new eta. */
static inline int64_t divsteps_62_var(int64_t eta, uint64_t f0, uint64_t g0, trans2x2 *t) {
    uint64_t u = 1, v = 0, q = 0, r = 1, f = f0, g = g0, m; uint32_t w; int i = 62, limit, zeros;
    for (;;) {
        zeros = __builtin_ctzll(g | (~0ULL << i));    /* a sentinel bit counts zeros only up to i */
        g >>= zeros; u <<= zeros; v <<= zeros; eta -= zeros; i -= zeros;
        if (i == 0) break;
        if (eta < 0) {
            uint64_t tmp;
            eta = -eta;
            tmp = f; f = g; g = -tmp;
            tmp = u; u = q; q = -tmp;
            tmp = v; v = r; r = -tmp;
            limit = ((int)eta + 1) > i ? i : ((int)eta + 1);
            m = (~0ULL >> (64 - limit)) & 63U;
            w = (uint32_t)((f * g * (f * f - 2)) & m);   /* -g/f mod 2^6 (f odd: f*(2-f^2) = f^-1 mod 64) */
        } else {
            limit = ((int)eta + 1) > i ? i : ((int)eta + 1);
            m = (~0ULL >> (64 - limit)) & 15U;
            w = (uint32_t)(f + (((f + 1) & 4) << 1));      /* f^-1 mod 16 */
            w = (uint32_t)((-(uint64_t)w * g) & m);
        }
        g += f * w; q += u * w; r += v * w;
    }
    t->u = (int64_t)u; t->v = (int64_t)v; t->q = (int64_t)q; t->r = (int64_t)r;
    return eta;
}
/* (d, e) = t * (d, e) / 2^62 mod p, keeping them in (-2p, p) */
static inline void update_de_62(s62 *d, s62 *e, const trans2x2 *t) {
    const uint64_t M62 = ~0ULL >> 2;
    const int64_t d0 = d->v[0], d1 = d->v[1], d2 = d->v[2], d3 = d->v[3], d4 = d->v[4];
    const int64_t e0 = e->v[0], e1 = e->v[1], e2 = e->v[2], e3 = e->v[3], e4 = e->v[4];
    const int64_t u = t->u, v = t->v, q = t->q, r = t->r;
    int64_t md, me, sd, se; __int128 cd, ce;
    sd = d4 >> 63; se = e4 >> 63;
    md = (u & sd) + (v & se);
    me = (q & sd) + (r & se);
    cd = (__int128)u * d0 + (__int128)v * e0;
    ce = (__int128)q * d0 + (__int128)r * e0;
    md -= (int64_t)((S62_PINV * (uint64_t)cd + (uint64_t)md) & M62);
    me -= (int64_t)((S62_PINV * (uint64_t)ce + (uint64_t)me) & M62);
    cd += (__int128)S62_P.v[0] * md;
    ce += (__int128)S62_P.v[0] * me;
    cd >>= 62; ce >>= 62;                             /* the low 62 bits are zero by construction */
    cd += (__int128)u * d1 + (__int128)v * e1;
    ce += (__int128)q * d1 + (__int128)r * e1;
    d->v[0] = (int64_t)((uint64_t)(int64_t)cd & M62); cd >>= 62;
    e->v[0] = (int64_t)((uint64_t)(int64_t)ce & M62); ce >>= 62;
    cd += (__int128)u * d2 + (__int128)v * e2;
    ce += (__int128)q * d2 + (__int128)r * e2;
    d->v[1] = (int64_t)((uint64_t)(int64_t)cd & M62); cd >>= 62;
    e->v[1] = (int64_t)((uint64_t)(int64_t)ce & M62); ce >>= 62;
    cd += (__int128)u * d3 + (__int128)v * e3;
    ce += (__int128)q * d3 + (__int128)r * e3;
    d->v[2] = (int64_t)((uint64_t)(int64_t)cd & M62); cd >>= 62;
    e->v[2] = (int64_t)((uint64_t)(int64_t)ce & M62); ce >>= 62;
    cd += (__int128)u * d4 + (__int128)v * e4;
    ce += (__int128)q * d4 + (__int128)r * e4;
    cd += (__int128)S62_P.v[4] * md;                  /* p's limbs 1..3 are zero */
    ce += (__int128)S62_P.v[4] * me;
    d->v[3] = (int64_t)((uint64_t)(int64_t)cd & M62); cd >>= 62;
    e->v[3] = (int64_t)((uint64_t)(int64_t)ce & M62); ce >>= 62;
    d->v[4] = (int64_t)cd;
    e->v[4] = (int64_t)ce;
}
/* (f, g) = t * (f, g) / 2^62 over the low len limbs */
static inline void update_fg_62_var(int len, s62 *f, s62 *g, const trans2x2 *t) {
    const uint64_t M62 = ~0ULL >> 2;
    const int64_t u = t->u, v = t->v, q = t->q, r = t->r;
    int64_t fi = f->v[0], gi = g->v[0];
    __int128 cf = (__int128)u * fi + (__int128)v * gi;
    __int128 cg = (__int128)q * fi + (__int128)r * gi;
    cf >>= 62; cg >>= 62;
    for (int i = 1; i < len; ++i) {
        fi = f->v[i]; gi = g->v[i];
        cf += (__int128)u * fi + (__int128)v * gi;
        cg += (__int128)q * fi + (__int128)r * gi;
        f->v[i - 1] = (int64_t)((uint64_t)(int64_t)cf & M62); cf >>= 62;
        g->v[i - 1] = (int64_t)((uint64_t)(int64_t)cg & M62); cg >>= 62;
    }
    f->v[len - 1] = (int64_t)cf;
    g->v[len - 1] = (int64_t)cg;
}
/* r in (-2p, p) -> [0, p), negated first when sign < 0 */
static inline void normalize_62(s62 *r, int64_t sign) {
    const int64_t M62 = (int64_t)(~0ULL >> 2);
    int64_t r0 = r->v[0], r1 = r->v[1], r2 = r->v[2], r3 = r->v[3], r4 = r->v[4];
    int64_t cond_add = r4 >> 63;
    r0 += S62_P.v[0] & cond_add; r4 += S62_P.v[4] & cond_add;
    const int64_t cond_negate = sign >> 63;
    r0 = (r0 ^ cond_negate) - cond_negate; r1 = (r1 ^ cond_negate) - cond_negate; r2 = (r2 ^ cond_negate) - cond_negate;
    r3 = (r3 ^ cond_negate) - cond_negate; r4 = (r4 ^ cond_negate) - cond_negate;
    r1 += r0 >> 62; r0 &= M62; r2 += r1 >> 62; r1 &= M62; r3 += r2 >> 62; r2 &= M62; r4 += r3 >> 62; r3 &= M62;
    cond_add = r4 >> 63;
    r0 += S62_P.v[0] & cond_add; r4 += S62_P.v[4] & cond_add;
    r1 += r0 >> 62; r0 &= M62; r2 += r1 >> 62; r1 &= M62; r3 += r2 >> 62; r2 &= M62; r4 += r3 >> 62; r3 &= M62;
    r->v[0] = r0; r->v[1] = r1; r->v[2] = r2; r->v[3] = r3; r->v[4] = r4;
}
/* x = x^-1 mod p for x in [0, p) (0 -> 0); false if the divstep loop did not settle within 24 batches. */
static bool modinv_var(s62 *x) {
    s62 d = {{0, 0, 0, 0, 0}}, e = {{1, 0, 0, 0, 0}}, f = S62_P, g = *x;
    int len = 5; int64_t eta = -1;
    for (int it = 0; it < 24; it++) {
        trans2x2 t;
        eta = divsteps_62_var(eta, (uint64_t)f.v[0], (uint64_t)g.v[0], &t);
        update_de_62(&d, &e, &t);
        update_fg_62_var(len, &f, &g, &t);
        if (g.v[0] == 0) {
            int64_t cond = 0; for (int j = 1; j < len; ++j) cond |= g.v[j];
            if (cond == 0) { normalize_62(&d, f.v[len - 1]); *x = d; return true; }
        }
        const int64_t fn = f.v[len - 1], gn = g.v[len - 1];
        int64_t cond = ((int64_t)len - 2) >> 63; cond |= fn ^ (fn >> 63); cond |= gn ^ (gn >> 63);
        if (cond == 0) { f.v[len - 2] |= (int64_t)((uint64_t)fn << 62); g.v[len - 2] |= (int64_t)((uint64_t)gn << 62); --len; }
    }
    return false;
}
/* r = a^-1 (canonical a; 0 -> 0), canonical. Falls back to the Fermat inversion if the loop did not settle. */
static void fe_inv_var(fe &r, const fe &a) {
    const uint64_t M = ~0ULL >> 2; s62 x;
    x.v[0] = (int64_t)(a.v[0] & M); x.v[1] = (int64_t)((a.v[0] >> 62 | a.v[1] << 2) & M);
    x.v[2] = (int64_t)((a.v[1] >> 60 | a.v[2] << 4) & M); x.v[3] = (int64_t)((a.v[2] >> 58 | a.v[3] << 6) & M);
    x.v[4] = (int64_t)(a.v[3] >> 56);
    if (!modinv_var(&x)) { fe_inv(r, a); return; }
    r.v[0] = (uint64_t)x.v[0] | (uint64_t)x.v[1] << 62; r.v[1] = (uint64_t)x.v[1] >> 2 | (uint64_t)x.v[2] << 60;
    r.v[2] = (uint64_t)x.v[2] >> 4 | (uint64_t)x.v[3] << 58; r.v[3] = (uint64_t)x.v[3] >> 6 | (uint64_t)x.v[4] << 56;
}
/* x = x^-1 lane-wise: Montgomery's trick across the 8 lanes (a 3-level tree of permuted multiplications:
 * 3 to combine, 3 to split), one scalar inversion of the lanes' product. A zero lane zeroes every lane's
 * result, as the Fermat inversion did (x-collisions, probability 2^-240; the exact gate absorbs them). */
Q8T static inline void fe8_swap1(fe8 &r, const fe8 &a) { for (int i = 0; i < 5; i++) r.l[i] = _mm512_permutex_epi64(a.l[i], 0xB1); }   /* lanes 2k <-> 2k+1 */
Q8T static inline void fe8_swap2(fe8 &r, const fe8 &a) { for (int i = 0; i < 5; i++) r.l[i] = _mm512_permutex_epi64(a.l[i], 0x4E); }   /* pairs 4k <-> 4k+2 */
Q8T static inline void fe8_swap4(fe8 &r, const fe8 &a) { for (int i = 0; i < 5; i++) r.l[i] = _mm512_shuffle_i64x2(a.l[i], a.l[i], 0x4E); }  /* halves */
Q8T static void fe8_inv_lanes(fe8 &x) {
    fe8 a1, p1, p2, t, i2;
    fe8_swap1(a1, x); fe8_mul(p1, x, a1);                 /* lanes 2k, 2k+1: a_2k * a_2k+1 */
    fe8_swap2(t, p1); fe8_mul(p2, p1, t);                 /* lanes 4k..4k+3: the product of the four */
    fe8_swap4(t, p2); fe8_mul(t, p2, t);                  /* every lane: the product of all eight */
    alignas(64) uint64_t w[4][8]; fe8_canon64(w, t);
    fe pr = {{w[0][0], w[1][0], w[2][0], w[3][0]}}, pi; fe_inv_var(pi, pr);
    fe8 I; fe8_bcast(I, pi);
    fe8_swap4(t, p2); fe8_mul(i2, I, t);                  /* lanes 0..3: 1/(a0 a1 a2 a3), lanes 4..7: 1/(a4..a7) */
    fe8_swap2(t, p1); fe8_mul(i2, i2, t);                 /* 1/(a_2k a_2k+1) */
    fe8_mul(x, i2, a1);                                   /* 1/a_k */
}
#endif
/* Invert the 4 interleaved chain products with ONE inversion (Montgomery's trick across the
 * chains: 3 + 6 extra multiplications). */
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
/* One batch-affine window step over G groups of 8 (G % 4 == 0): acc[g] += sign * T[|digit| - 1].
 * rp/ng hold this window's table rows (8 per group, never a digit-0 row) and lane sign masks, computed and
 * prefetched by the previous pass. With nxt, the backward pass computes the next window's rows into rpn/ngn
 * and prefetches them (next group 0 first), so the table misses spread over the long backward pass instead
 * of bunching in the short forward pass; the forward pass only pulls its rows QSB_CPU_PFD groups ahead. */
template <class RowFn>
Q8TX static void ec8_window(fe8 *X, fe8 *Y, fe8 *D, fe8 *PRE, fe8 *TX, fe8 *TY, int G, const pt *const *rp, const __mmask8 *ng,
                           const pt **rpn, __mmask8 *ngn, const RowFn *nxt) {
    const int PF = QSB_CPU_PFD;
    fe8 run[4]; for (int c = 0; c < 4; c++) fe8_set1(run[c]);
    for (int g = 0; g < G; g += 4) {
        for (int c = 0; c < 4; c++) {
            const int h = g + c;
            if (h + PF < G) { const pt *const *pr = rp + (size_t)(h + PF) * 8; for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pr[j], _MM_HINT_T0); }
            pt8_load(TX[h], TY[h], rp + (size_t)h * 8);   /* kept (transposed) for the backward pass */
            fe8_sub_lz(D[h], TX[h], X[h]);
            fe8_cp(PRE[h], run[c]);
            fe8_mul_lz(run[c], run[c], D[h]);
        }
    }
    fe8_inv4(run);
    for (int g = G - 4; g >= 0; g -= 4) {
        if (nxt)
            for (int c = 3; c >= 0; c--) {
                const int hn = G - 1 - (g + c); const pt **pr = rpn + (size_t)hn * 8;
                ngn[hn] = (*nxt)(hn, pr);
                for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pr[j], _MM_HINT_T0);
            }
        for (int c = 3; c >= 0; c--) {              /* one group at a time, its temporaries in registers */
            const int h = g + c;
            fe8 dinv, t, lam, x3;
            fe8_mul_lz(dinv, run[c], PRE[h]); fe8_mul_lz(run[c], run[c], D[h]);   /* chain c steps back one group */
            fe8_sub_sgn_lz(t, TY[h], Y[h], ng[h]); fe8_mul_lz(lam, t, dinv);
            fe8_sqr_sub2(x3, lam, X[h], TX[h]);                 /* X, Y stay normalized: they are subtrahends */
            fe8_sub_lz(t, X[h], x3); fe8_mul_sub(Y[h], lam, t, Y[h]); fe8_cp(X[h], x3);
        }
    }
}
/* Canonical x of 8 lanes into the (candidate, recid) array: lane j -> o[2 * j] (o = &qx[h * 16 + ri]).
 * The four 64-bit limb rows are transposed to one 4-limb row per lane (unpack + permute) and stored as
 * eight 256-bit rows, instead of 32 scalar extractions and stores. */
Q8T static inline void fe8_store_x(fe *o, const fe8 &a) {
    fe8 u; const __mmask8 ge = fe8_ge_p(a, &u);
    __m512i r[5];
    for (int i = 0; i < 5; i++) r[i] = _mm512_mask_blend_epi64(ge, a.l[i], u.l[i]);
    const __m512i w0 = _mm512_or_si512(r[0], _mm512_slli_epi64(r[1], 52)),
                  w1 = _mm512_or_si512(_mm512_srli_epi64(r[1], 12), _mm512_slli_epi64(r[2], 40)),
                  w2 = _mm512_or_si512(_mm512_srli_epi64(r[2], 24), _mm512_slli_epi64(r[3], 28)),
                  w3 = _mm512_or_si512(_mm512_srli_epi64(r[3], 36), _mm512_slli_epi64(r[4], 16));
    const __m512i a0 = _mm512_unpacklo_epi64(w0, w1), a1 = _mm512_unpackhi_epi64(w0, w1),
                  a2 = _mm512_unpacklo_epi64(w2, w3), a3 = _mm512_unpackhi_epi64(w2, w3);
    const __m512i iL = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0), iH = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
    const __m512i b0 = _mm512_permutex2var_epi64(a0, iL, a2),   /* lanes 0, 2 */
                  b1 = _mm512_permutex2var_epi64(a1, iL, a3),   /* lanes 1, 3 */
                  b2 = _mm512_permutex2var_epi64(a0, iH, a2),   /* lanes 4, 6 */
                  b3 = _mm512_permutex2var_epi64(a1, iH, a3);   /* lanes 5, 7 */
    _mm256_storeu_si256((__m256i *)(o + 0), _mm512_castsi512_si256(b0)); _mm256_storeu_si256((__m256i *)(o + 4), _mm512_extracti64x4_epi64(b0, 1));
    _mm256_storeu_si256((__m256i *)(o + 2), _mm512_castsi512_si256(b1)); _mm256_storeu_si256((__m256i *)(o + 6), _mm512_extracti64x4_epi64(b1, 1));
    _mm256_storeu_si256((__m256i *)(o + 8), _mm512_castsi512_si256(b2)); _mm256_storeu_si256((__m256i *)(o + 12), _mm512_extracti64x4_epi64(b2, 1));
    _mm256_storeu_si256((__m256i *)(o + 10), _mm512_castsi512_si256(b3)); _mm256_storeu_si256((__m256i *)(o + 14), _mm512_extracti64x4_epi64(b3, 1));
}
/* bit j of x -> byte j (0 or 1) */
static inline uint64_t spread8(unsigned x) {
    const uint64_t v = ((uint64_t)x * 0x0101010101010101ULL) & 0x8040201008040201ULL;
    return ((v + 0x7F7F7F7F7F7F7F7FULL) & 0x8080808080808080ULL) >> 7;
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
        for (int c = 0; c < 4; c++) { const int h = g + c; fe8_sub_lz(D[h], CX, X[h]); fe8_cp(PRE[h], run[c]); fe8_mul_lz(run[c], run[c], D[h]); }
    fe8_inv4(run);
    for (int g = G - 4; g >= 0; g -= 4) {
        for (int c = 3; c >= 0; c--) {
            const int h = g + c;
            fe8 dinv; fe8_mul_lz(dinv, run[c], PRE[h]); fe8_mul_lz(run[c], run[c], D[h]);
            unsigned par[2];
            for (int ri = 0; ri < 2; ri++) {
                fe8 t, lam, x3, y3;
                fe8_sub_lz(t, ri ? CY1 : CY0, Y[h]); fe8_mul_lz(lam, t, dinv);
                fe8_sqr_sub2(x3, lam, X[h], CX);
                fe8_sub_lz(t, X[h], x3); fe8_mul_sub(y3, lam, t, Y[h]);
                fe8_store_x(qx + (size_t)h * 16 + ri, x3);
                par[ri] = fe8_parity(y3);
            }
            /* qp[(h * 8 + j) * 2 + ri] = bit j of par[ri]: 16 bytes, the two recids interleaved */
            _mm_storeu_si128((__m128i *)(qp + (size_t)h * 16),
                             _mm_unpacklo_epi8(_mm_cvtsi64_si128((long long)spread8(par[0])), _mm_cvtsi64_si128((long long)spread8(par[1]))));
        }
    }
}
/* Window 0: acc = sign * T0[|d0| - 1] (digit-0 lanes load row 0 and are dropped by the caller); computes
 * and prefetches window 1's rows (nxt) into rpn/ngn. */
template <class RowFn>
Q8TX static void ec8_first(fe8 *X, fe8 *Y, int G, RowFn rowfn, RowFn nxt, const pt **rpn, __mmask8 *ngn) {
    alignas(64) const pt *rows[8];
    fe8 Z; for (int i = 0; i < 5; i++) Z.l[i] = _mm512_setzero_si512();
    const int PF = QSB_CPU_PFD;
    for (int h = 0; h < PF && h < G; h++) { alignas(64) const pt *pr[8]; rowfn(h, pr); for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pr[j], _MM_HINT_T0); }
    for (int h = 0; h < G; h++) {
        if (h + PF < G) { alignas(64) const pt *pr[8]; rowfn(h + PF, pr); for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pr[j], _MM_HINT_T0); }
        { const pt **pn = rpn + (size_t)h * 8; ngn[h] = nxt(h, pn); for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pn[j], _MM_HINT_T0); }
        const __mmask8 ng = rowfn(h, rows);
        fe8 ty; pt8_load(X[h], ty, rows);
        if (ng) { fe8 n; fe8_sub(n, Z, ty); for (int i = 0; i < 5; i++) Y[h].l[i] = _mm512_mask_blend_epi64(ng, ty.l[i], n.l[i]); }
        else fe8_cp(Y[h], ty);
    }
}
/* Digits of 16 candidates at a time (zb: 8 SHA state words per candidate, h0 most significant) into
 * ds[i * B + k], with the per-window shifts uniform across the lanes; bad[k] = some digit is zero. */
Q8T static void recode16(const Geo &g, const uint32_t *zb, uint32_t *ds, uint8_t *bad, int B) {
    const __m512i vidx = _mm512_set_epi32(120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0);
    const __m512i Z = _mm512_setzero_si512(), SB = _mm512_set1_epi32((int)0x80000000u), ONE = _mm512_set1_epi32(1);
    const int nw = g.nw, ns = g.sgn ? nw - 1 : 0;        /* windows 0..ns-1 are signed */
    __m128i shr[NWMAX], shl[NWMAX]; __m512i msk[NWMAX], half[NWMAX], full[NWMAX]; int wq[NWMAX]; bool two[NWMAX];
    for (int i = 0; i < nw; i++) {
        const int s = g.off[i] & 31, w = g.wid[i];
        wq[i] = g.off[i] >> 5; two[i] = s + w > 32;
        shr[i] = _mm_cvtsi32_si128(s); shl[i] = _mm_cvtsi32_si128(32 - s);
        msk[i] = _mm512_set1_epi32((int)((1u << w) - 1)); half[i] = _mm512_set1_epi32((int)(1u << (w - 1))); full[i] = _mm512_set1_epi32((int)(1u << w));
    }
    for (int k0 = 0; k0 < B; k0 += 16) {
        __m512i zw[9];                                   /* zw[j] = bits 32j..32j+31 of z = word h[7 - j] */
        for (int j = 0; j < 8; j++) zw[j] = _mm512_i32gather_epi32(vidx, (const void *)(zb + (size_t)k0 * 8 + (7 - j)), 4);
        zw[8] = Z;
        __m512i cy = Z; __mmask16 zero = 0;
        for (int i = 0; i < nw; i++) {
            __m512i v = _mm512_srl_epi32(zw[wq[i]], shr[i]);
            if (two[i]) v = _mm512_or_si512(v, _mm512_sll_epi32(zw[wq[i] + 1], shl[i]));
            v = _mm512_add_epi32(_mm512_and_si512(v, msk[i]), cy);   /* cy = 0 unless signed */
            if (i < ns) {
                const __mmask16 m = _mm512_cmpgt_epu32_mask(v, half[i]);
                v = _mm512_mask_sub_epi32(v, m, full[i], v);
                cy = _mm512_maskz_mov_epi32(m, ONE);
                zero |= _mm512_testn_epi32_mask(v, v);
                v = _mm512_mask_or_epi32(v, m, v, SB);
            } else zero |= _mm512_testn_epi32_mask(v, v);
            _mm512_storeu_si512((void *)(ds + (size_t)i * B + k0), v);
        }
        _mm_storeu_si128((__m128i *)(bad + k0), _mm512_cvtepi32_epi8(_mm512_maskz_mov_epi32(zero, ONE)));
    }
}
#endif  /* QCPU_VEC */

#if QCPU_SHANI
/* SHA-256 compression of 4 independent (state, block) pairs with the x86 SHA extensions,
 * instruction streams interleaved so the sha256rnds2 latency of one lane hides behind the others. */
#define QSHA __attribute__((target("sha,sse4.1,ssse3,avx")))   /* VEX: 3-operand adds/palignr feed sha256rnds2 without copies */
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
static bool qsha_supported() {
    unsigned a, b, cc, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &cc, &d)) return false;
    __builtin_cpu_init();                                   /* the 4-lane code is VEX-encoded: needs AVX too */
    return ((b >> 29) & 1) && __builtin_cpu_supports("avx");   /* CPUID.(7,0):EBX.SHA */
}
/* Precomputed-schedule compression: rows[l][b] points to the 64 words W[i] + K[i] of lane l's
 * block b (b < nblk), so the rounds need no message expansion (no sha256msg1/msg2, byte swaps or
 * K additions) and the chaining state stays in the ABEF/CDGH layout across the nblk blocks. Used
 * for the tail blocks that do not depend on the epoch: every one of them is one of a few fixed
 * contents per problem, scheduled once in start(). */
QSHA static void qsha_x4p(uint32_t (*st)[8], const uint32_t *const *const *rows, int nblk) {
    __m128i S0[4], S1[4];
#pragma GCC unroll 4
    for (int l = 0; l < 4; l++) {
        __m128i t = _mm_loadu_si128((const __m128i *)&st[l][0]);
        __m128i u = _mm_loadu_si128((const __m128i *)&st[l][4]);
        t = _mm_shuffle_epi32(t, 0xB1); u = _mm_shuffle_epi32(u, 0x1B);
        S0[l] = _mm_alignr_epi8(t, u, 8); S1[l] = _mm_blend_epi16(u, t, 0xF0);
    }
    for (int b = 0; b < nblk; b++) {
        const uint32_t *const w[4] = {rows[0][b], rows[1][b], rows[2][b], rows[3][b]};
        __m128i I0[4], I1[4];
#pragma GCC unroll 4
        for (int l = 0; l < 4; l++) { I0[l] = S0[l]; I1[l] = S1[l]; }
#pragma GCC unroll 16
        for (int r = 0; r < 16; r++) {
#pragma GCC unroll 4
            for (int l = 0; l < 4; l++) {
                S1[l] = _mm_sha256rnds2_epu32(S1[l], S0[l], _mm_loadl_epi64((const __m128i *)(w[l] + 4 * r)));
                S0[l] = _mm_sha256rnds2_epu32(S0[l], S1[l], _mm_loadl_epi64((const __m128i *)(w[l] + 4 * r + 2)));
            }
        }
#pragma GCC unroll 4
        for (int l = 0; l < 4; l++) { S0[l] = _mm_add_epi32(S0[l], I0[l]); S1[l] = _mm_add_epi32(S1[l], I1[l]); }
    }
#pragma GCC unroll 4
    for (int l = 0; l < 4; l++) {
        __m128i t = _mm_shuffle_epi32(S0[l], 0x1B), b = _mm_shuffle_epi32(S1[l], 0xB1);
        _mm_storeu_si128((__m128i *)&st[l][0], _mm_blend_epi16(t, b, 0xF0));
        _mm_storeu_si128((__m128i *)&st[l][4], _mm_alignr_epi8(b, t, 8));
    }
}
/* SHA-256 compression with the message given as 16 native-order words per lane (no byte round trip),
 * L lanes interleaved: the second SHA-256 (message = the first digest's state words + fixed padding)
 * and the key hashes. L = 2 keeps state, message and schedule in the 16 legacy-SSE registers that
 * sha256rnds2 can address (4 lanes spill ~60 moves per lane and are not faster). */
template <int L>
QSHA static inline void qsha_xw(uint32_t (*st)[8], const uint32_t (*wd)[16]) {
    __m128i S0[L], S1[L], I0[L], I1[L], M[L][4];
#pragma GCC unroll 4
    for (int l = 0; l < L; l++) {
        __m128i t = _mm_loadu_si128((const __m128i *)&st[l][0]);
        __m128i u = _mm_loadu_si128((const __m128i *)&st[l][4]);
        t = _mm_shuffle_epi32(t, 0xB1); u = _mm_shuffle_epi32(u, 0x1B);
        S0[l] = _mm_alignr_epi8(t, u, 8); S1[l] = _mm_blend_epi16(u, t, 0xF0);
        I0[l] = S0[l]; I1[l] = S1[l];
#pragma GCC unroll 4
        for (int j = 0; j < 4; j++) M[l][j] = _mm_loadu_si128((const __m128i *)(wd[l] + 4 * j));
    }
#pragma GCC unroll 16
    for (int r = 0; r < 16; r++) {
        const __m128i K = _mm_load_si128((const __m128i *)&qsha_k[4 * r]);
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
QSHA static void qsha_x4w(uint32_t (*st)[8], const uint32_t (*wd)[16]) {
    qsha_xw<2>(st, wd); qsha_xw<2>(st + 2, wd + 2);
}
/* qsha_xw<L> from the SHA-256 IV: the initial ABEF/CDGH pairs are constants (no state load or shuffle),
 * so the second SHA-256 and the key hashes need no per-call state initialization. Same rounds as qsha_xw. */
template <int L>
QSHA static inline void qsha_xw_iv(uint32_t (*st)[8], const uint32_t (*wd)[16]) {
    const __m128i IV0 = _mm_set_epi32((int)0x6a09e667, (int)0xbb67ae85, (int)0x510e527f, (int)0x9b05688c);   /* ABEF: lanes f e b a */
    const __m128i IV1 = _mm_set_epi32((int)0x3c6ef372, (int)0xa54ff53a, (int)0x1f83d9ab, (int)0x5be0cd19);   /* CDGH: lanes h g d c */
    __m128i S0[L], S1[L], M[L][4];
#pragma GCC unroll 4
    for (int l = 0; l < L; l++) {
        S0[l] = IV0; S1[l] = IV1;
#pragma GCC unroll 4
        for (int j = 0; j < 4; j++) M[l][j] = _mm_loadu_si128((const __m128i *)(wd[l] + 4 * j));
    }
#pragma GCC unroll 16
    for (int r = 0; r < 16; r++) {
        const __m128i K = _mm_load_si128((const __m128i *)&qsha_k[4 * r]);
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
        __m128i a = _mm_add_epi32(S0[l], IV0), b = _mm_add_epi32(S1[l], IV1);
        __m128i t = _mm_shuffle_epi32(a, 0x1B);
        b = _mm_shuffle_epi32(b, 0xB1);
        _mm_storeu_si128((__m128i *)&st[l][0], _mm_blend_epi16(t, b, 0xF0));
        _mm_storeu_si128((__m128i *)&st[l][4], _mm_alignr_epi8(b, t, 8));
    }
}
QSHA static void qsha_x4w_iv(uint32_t (*st)[8], const uint32_t (*wd)[16]) {
    qsha_xw_iv<2>(st, wd); qsha_xw_iv<2>(st + 2, wd + 2);
}
/* W[i] + K[i] for i < 64 of one 64-byte block (big-endian words), with qsha_x4's schedule steps. */
QSHA static void qsha_schedule(uint32_t wk[64], const uint8_t *blk) {
    const __m128i BSWAP = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
    __m128i M[4];
    for (int j = 0; j < 4; j++) M[j] = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(blk + 16 * j)), BSWAP);
#pragma GCC unroll 16
    for (int r = 0; r < 16; r++) {
        if (r >= 4) {
            __m128i t = _mm_sha256msg1_epu32(M[r & 3], M[(r + 1) & 3]);
            t = _mm_add_epi32(t, _mm_alignr_epi8(M[(r + 3) & 3], M[(r + 2) & 3], 4));
            M[r & 3] = _mm_sha256msg2_epu32(t, M[(r + 3) & 3]);
        }
        _mm_storeu_si128((__m128i *)(wk + 4 * r), _mm_add_epi32(M[r & 3], _mm_load_si128((const __m128i *)&qsha_k[4 * r])));
    }
}
#endif
struct Ctx {
    const digest_params_t *dp;
    Geo g;                          /* table geometry */
    pt *table = nullptr;            /* g.total points of 64 B, 2 MiB aligned (transparent huge pages) */
    void *table_map = nullptr; size_t table_map_bytes = 0;
    const pt *tw[NWMAX] = {nullptr};   /* first entry of window i */
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
#if QCPU_SHANI
    /* Hashing plan (hash_plan): the message after an epoch's state is block 0 (the epoch's buffered
     * bytes + the first window pushes) and blocks 1..nb-1, whose contents do not depend on the epoch.
     * The CPU patterns share few block-0 contents (groups) and few distinct later blocks. */
    bool hplan = false;
    int h_nb = 0, h_ng = 0;
    uint8_t h_g0[286];              /* block-0 group of each CPU pattern */
    const uint32_t *h_wkp[286][16]; /* schedule of fixed block b = 1..nb-1 of each CPU pattern (index b-1) */
    uint64_t h_binom[256][8];       /* binom_u64(n, k) for the epoch unrank */
    std::vector<uint8_t> h_gblk;    /* ng x 64: block 0 of each group, epoch bytes left zero */
    std::vector<uint32_t, qalloc64<uint32_t> > h_wk;   /* distinct fixed blocks x 64 words W[i]+K[i] */
#endif
};

/* The table as a 2 MiB-aligned anonymous mapping with transparent huge pages (it is read at random);
 * pages are touched first by the builder threads. */
static bool table_alloc(Ctx &c) {
    const size_t H = (size_t)2 << 20, bytes = (c.g.total * sizeof(pt) + H - 1) / H * H;
    void *m = mmap(nullptr, bytes + H, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return false;
    const uintptr_t a = ((uintptr_t)m + H - 1) & ~(uintptr_t)(H - 1);
#ifdef MADV_HUGEPAGE
    if (!getenv("QSB_CPU_NOTHP")) madvise((void *)a, bytes, MADV_HUGEPAGE);   /* dev override: 4 KiB pages */
#endif
    c.table_map = m; c.table_map_bytes = bytes + H; c.table = (pt *)a;
    for (int i = 0; i < c.g.nw; i++) c.tw[i] = c.table + c.g.base[i];
    return true;
}
static void table_free(Ctx &c) {
    if (c.table_map) munmap(c.table_map, c.table_map_bytes);
    c.table_map = nullptr; c.table = nullptr;
}
static pt pt_add_aff(const pt &a, const pt &b) {       /* a + b for distinct x */
    fe d, inv, t, lam, x3, y3;
    fe_sub(d, b.x, a.x); fe_inv(inv, d); fe_sub(t, b.y, a.y); fe_mul(lam, t, inv);
    fe_sqr(x3, lam); fe_sub(x3, x3, a.x); fe_sub(x3, x3, b.x);
    fe_sub(t, a.x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, a.y);
    return {x3, y3};
}
static pt pt_mul_small(const pt &q, uint64_t k) {      /* k * q, 1 <= k << n (no intermediate meets +-q) */
    pt r = q;
    for (int b = 62 - __builtin_clzll(k | 1); b >= 0; b--) { r = pt_double(r); if ((k >> b) & 1) r = pt_add_aff(r, q); }
    return r;
}
/* Window i: T[j] = (j + 1) * B_i, B_i = 2^off[i] * A. The first S entries by doubling rounds; entries
 * cS..cS+S-1 (c >= 1) as T[k] + cS * B_i. The (window, chunk) tasks are shared by nth threads. */
static void build_table(Ctx &c, const fe &ax, const fe &ay, int nth) {
    const Geo &g = c.g;
    const uint32_t S = 1u << 16;
    std::vector<pt> base(g.nw);
    { pt q = {ax, ay}; int at = 0; for (int i = 0; i < g.nw; i++) { while (at < g.off[i]) { q = pt_double(q); at++; } base[i] = q; } }
    auto chunk0 = [&](int i) {
        pt *T = c.table + g.base[i];
        const uint32_t n0 = g.ent[i] < S ? g.ent[i] : S;
        T[0] = base[i];
        if (n0 < 2) return;
        T[1] = pt_double(base[i]);
        uint32_t have = 2;                               /* T[0..have-1] = 1..have multiples */
        std::vector<fe> d(n0), pre(n0); std::vector<uint8_t> inf(n0), bad(n0); std::vector<const pt *> tp(n0);
        while (have < n0) {
            uint32_t n = have; if (have + n > n0) n = n0 - have;
            /* T[have+k] = T[k] + T[have-1]  ((k+1) + have = have+k+1) */
            for (uint32_t k = 0; k < n; k++) { T[have + k] = T[k]; tp[k] = &T[have - 1]; inf[k] = 0; bad[k] = 0; }
            batch_add(&T[have], tp.data(), inf.data(), bad.data(), (int)n, d.data(), pre.data());
            /* k = have-1 adds T[have-1] to itself: equal x, so batch_add flags it; double it. */
            for (uint32_t k = 0; k < n; k++) if (bad[k]) T[have + k] = pt_double(T[have - 1]);
            have += n;
        }
    };
    auto chunkc = [&](int i, uint32_t cc) {
        pt *T = c.table + g.base[i];
        const uint32_t o = cc * S, n = g.ent[i] - o < S ? g.ent[i] - o : S;
        const pt Q = pt_mul_small(T[S - 1], cc);         /* T[S-1] = S * B_i */
        std::vector<fe> d(n), pre(n); std::vector<uint8_t> inf(n, 0), bad(n, 0); std::vector<const pt *> tp(n, &Q);
        for (uint32_t k = 0; k < n; k++) T[o + k] = T[k];
        batch_add(&T[o], tp.data(), inf.data(), bad.data(), (int)n, d.data(), pre.data());
        for (uint32_t k = 0; k < n; k++) if (bad[k]) T[o + k] = pt_double(Q);   /* only cc = 1, k = S-1: T[k] = Q */
    };
    std::vector<std::pair<int, uint32_t> > tasks;
    for (int i = 0; i < g.nw; i++) for (uint32_t cc = 1; (uint64_t)cc * S < g.ent[i]; cc++) tasks.push_back({i, cc});
    for (int phase = 0; phase < 2; phase++) {
        std::atomic<size_t> next{0};
        const size_t nt = phase ? tasks.size() : (size_t)g.nw;
        auto work = [&]() {
            for (size_t t; (t = next.fetch_add(1)) < nt;) { if (phase) chunkc(tasks[t].first, tasks[t].second); else chunk0((int)t); }
        };
        std::vector<std::thread> ts;
        const int m = (size_t)nth < nt ? nth : (int)nt;
        for (int t = 0; t < m; t++) ts.emplace_back(work);
        for (auto &t : ts) t.join();
    }
}
/* Spot-check the table against OpenSSL: in every window the first, the last and a middle entry and both
 * sides of the first chunk boundary. */
static bool table_check(const Ctx &c) {
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *bx = BN_CTX_new(); BIGNUM *nri = BN_new(), *k = BN_new(), *ord = BN_new(), *xx = BN_new(), *yy = BN_new();
    EC_POINT *P = grp ? EC_POINT_new(grp) : nullptr;
    bool ok = grp && bx && nri && k && ord && xx && yy && P && EC_GROUP_get_order(grp, ord, bx) &&
              BN_lebin2bn(c.dp->neg_r_inv, 32, nri);
    for (int i = 0; ok && i < c.g.nw; i++) {
        const uint32_t e = c.g.ent[i];
        const uint32_t js[5] = {0, e / 2, e - 1, e > 65536 ? 65535u : 0u, e > 65536 ? 65536u : 0u};
        for (int q = 0; ok && q < 5; q++) {
            const uint32_t j = js[q];
            ok = BN_set_word(k, (BN_ULONG)j + 1) && BN_lshift(k, k, c.g.off[i]) && BN_mod_mul(k, k, nri, ord, bx) &&
                 EC_POINT_mul(grp, P, k, NULL, NULL, bx) && EC_POINT_get_affine_coordinates_GFp(grp, P, xx, yy, bx);
            if (ok) { fe fx, fy; fe_from_bn(fx, xx); fe_from_bn(fy, yy); const pt &t = c.table[c.g.base[i] + j]; ok = fe_eq(fx, t.x) && fe_eq(fy, t.y); }
        }
    }
    if (P) EC_POINT_free(P);
    BN_free(nri); BN_free(k); BN_free(ord); BN_free(xx); BN_free(yy); if (bx) BN_CTX_free(bx); if (grp) EC_GROUP_free(grp);
    return ok;
}
/* Bytes this process may still take: MemAvailable, capped by the cgroup (v2 or v1) limit minus usage. */
static double mem_avail() {
    double avail = -1;
    if (FILE *f = fopen("/proc/meminfo", "r")) {
        char line[256]; long long kb;
        while (fgets(line, sizeof line, f)) if (sscanf(line, "MemAvailable: %lld kB", &kb) == 1) { avail = (double)kb * 1024.0; break; }
        fclose(f);
    }
    auto rd = [](const char *path, double &v) -> bool {
        FILE *f = fopen(path, "r"); if (!f) return false;
        char b[64] = {0}; const bool ok = fscanf(f, "%63s", b) == 1; fclose(f);
        if (!ok || !strcmp(b, "max")) return false;
        v = atof(b); return v > 0 && v < 1e18;
    };
    double lim, cur;
    if (rd("/sys/fs/cgroup/memory.max", lim)) { if (!rd("/sys/fs/cgroup/memory.current", cur)) cur = 0; if (avail < 0 || lim - cur < avail) avail = lim - cur; }
    if (rd("/sys/fs/cgroup/memory/memory.limit_in_bytes", lim)) { if (!rd("/sys/fs/cgroup/memory/memory.usage_in_bytes", cur)) cur = 0; if (avail < 0 || lim - cur < avail) avail = lim - cur; }
    return avail;
}
/* Geometry: QSB_CPU_NW / QSB_CPU_UNSIGNED (dev overrides), else the fewest windows (>= QSB_CPU_NW_MIN) whose
 * table fits the budget, else 15 (68 MiB, the size of the old 16 x 16-bit table). */
static Geo geo_choose() {
    double budget = mem_avail() * QSB_CPU_TAB_FRAC;
    if (budget > (double)QSB_CPU_TAB_CAP_MB * 1048576.0) budget = (double)QSB_CPU_TAB_CAP_MB * 1048576.0;
    int nw = 15;
    for (int n = QSB_CPU_NW_MIN; n < 15; n++) if ((double)geo_make(n, true).total * sizeof(pt) <= budget) { nw = n; break; }
    if (const char *e = getenv("QSB_CPU_NW")) nw = atoi(e);
    return geo_make(nw, !getenv("QSB_CPU_UNSIGNED"));
}

#if QCPU_SHANI
/* Plan the 4-lane hashing (after ncwin, mid_bytes): group the CPU patterns by block 0 and schedule
 * every distinct later block once. Leaves hplan false for an unexpected shape (old path then). */
static void hash_plan(Ctx &c) {
    const digest_params_t *dp = c.dp;
    const size_t prl = dp->prefix_remainder_len, pl = (size_t)(c.cut - c.early) * SIG_PUSH_SIZE,
                 wlen = (size_t)(dp->n - c.cut - 3) * SIG_PUSH_SIZE, tl = dp->tail_section_len, sl = dp->tx_suffix_len;
    const size_t remlen = (prl + pl) % 64;
    const int nb = (int)((remlen + wlen + tl + sl + 9 + 63) / 64);
    if (nb < 2 || nb > 16 || c.ncwin < 1 || c.ncwin > 286) return;
    const uint64_t tbits = (c.mid_bytes + prl + pl + wlen + tl + sl) * 8;
    std::vector<uint8_t> blocks, m((size_t)nb * 64);
    std::vector<uint16_t> sidx((size_t)c.ncwin * 16);
    int ng = 0; c.h_gblk.clear();
    for (int wi = 0; wi < c.ncwin; wi++) {
        const uint8_t *w3 = c.cwin[wi];
        size_t o = remlen; memset(m.data(), 0, m.size());
        for (int i = c.cut; i < (int)dp->n; i++) {
            if (i == w3[0] || i == w3[1] || i == w3[2]) continue;
            memcpy(&m[o], dp->dummy_sigs + (size_t)i * SIG_PUSH_SIZE, SIG_PUSH_SIZE); o += SIG_PUSH_SIZE;
        }
        memcpy(&m[o], dp->tail_section, tl); o += tl;
        memcpy(&m[o], dp->tx_suffix, sl); o += sl;
        m[o] = 0x80;
        for (int b = 0; b < 8; b++) m[(size_t)nb * 64 - 1 - b] = (uint8_t)(tbits >> (8 * b));
        int g = 0; while (g < ng && memcmp(&c.h_gblk[(size_t)g * 64], m.data(), 64)) g++;
        if (g == ng) { c.h_gblk.insert(c.h_gblk.end(), m.begin(), m.begin() + 64); ng++; }
        c.h_g0[wi] = (uint8_t)g;
        for (int b = 1; b < nb; b++) {
            const size_t nbk = blocks.size() / 64; size_t q = 0;
            while (q < nbk && memcmp(&blocks[q * 64], &m[(size_t)b * 64], 64)) q++;
            if (q == nbk) blocks.insert(blocks.end(), m.begin() + (size_t)b * 64, m.begin() + (size_t)b * 64 + 64);
            sidx[(size_t)wi * 16 + b] = (uint16_t)q;
        }
    }
    c.h_wk.assign(blocks.size(), 0);                    /* (blocks/64) x 64 words */
    for (size_t q = 0; q < blocks.size() / 64; q++) qsha_schedule(&c.h_wk[q * 64], &blocks[q * 64]);
    for (int wi = 0; wi < c.ncwin; wi++) for (int b = 1; b < nb; b++) c.h_wkp[wi][b - 1] = &c.h_wk[(size_t)sidx[(size_t)wi * 16 + b] * 64];
    for (int n = 0; n < 256; n++) for (int k = 0; k < 8; k++) c.h_binom[n][k] = binom_u64(n, k);
    if (c.cut > 255 || c.early > 7) return;
    c.h_nb = nb; c.h_ng = ng; c.hplan = true;
}
#endif
/* Gate one recovered key (x, parity of y); true when it is an exact hit, which is then published. */
static inline void pk_block(uint8_t *pk, const fe &x3, unsigned ypar) {    /* compressed key, bytes 0..32 */
    pk[0] = (uint8_t)(0x02 | (ypar & 1));
    for (int b = 0; b < 32; b++) pk[1 + b] = (uint8_t)(x3.v[3 - (b >> 3)] >> (8 * (7 - (b & 7))));
}
/* The key-hash block as native words (pk_block's 33 bytes + padding for 264 bits); w[9..14] are left
 * as they are (zero) and w[15] = 264 is set by the caller once. */
#if QCPU_SHANI
/* word k (LE memory) = key bytes 4k+3..4k; key byte m >= 1 is x's LE byte 32 - m, key byte 0 the prefix */
QSHA static inline void pk_words(uint32_t *w, const fe &x, unsigned ypar) {
    const __m128i lo = _mm_loadu_si128((const __m128i *)&x.v[0]), hi = _mm_loadu_si128((const __m128i *)&x.v[2]);
    const __m128i I03 = _mm_set_epi8(4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, -128, 15, 14, 13);
    const __m128i I47 = _mm_set_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    _mm_storeu_si128((__m128i *)&w[0], _mm_or_si128(_mm_shuffle_epi8(hi, I03), _mm_cvtsi32_si128((int)((0x02u | (ypar & 1)) << 24))));
    _mm_storeu_si128((__m128i *)&w[4], _mm_shuffle_epi8(_mm_alignr_epi8(hi, lo, 1), I47));
    w[8] = ((uint32_t)(x.v[0] & 0xff) << 24) | 0x00800000u;
}
#endif
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
struct VecBuf { fe8 *X = nullptr, *Y = nullptr, *D = nullptr, *P = nullptr, *TX = nullptr, *TY = nullptr; fe *qx = nullptr; uint8_t *qp = nullptr, *bad = nullptr; };
static bool vecbuf_alloc(VecBuf &v, int B) {
    const int G = B / 8; void *q[9] = {nullptr};
    const size_t sz[9] = {sizeof(fe8) * G, sizeof(fe8) * G, sizeof(fe8) * G, sizeof(fe8) * G, sizeof(fe8) * G, sizeof(fe8) * G,
                          sizeof(fe) * 2 * (size_t)B, 2 * (size_t)B, (size_t)B};
    for (int i = 0; i < 9; i++) if (posix_memalign(&q[i], 64, sz[i])) { for (int j = 0; j < i; j++) free(q[j]); return false; }
    v.X = (fe8 *)q[0]; v.Y = (fe8 *)q[1]; v.D = (fe8 *)q[2]; v.P = (fe8 *)q[3]; v.TX = (fe8 *)q[4]; v.TY = (fe8 *)q[5];
    v.qx = (fe *)q[6]; v.qp = (uint8_t *)q[7]; v.bad = (uint8_t *)q[8];
    return true;
}
/* Row function of one window: the 8 lanes' rows T[|d| - 1] (row 0 for a zero digit) and their sign mask. */
struct RowSgn {
    const pt *T; const uint32_t *di;
    Q8T __mmask8 operator()(int h, const pt **rows) const {
        const __m512i one = _mm512_set1_epi64(1);
        const __m512i e = _mm512_cvtepu32_epi64(_mm256_loadu_si256((const __m256i *)(di + (size_t)h * 8)));
        __m512i ix = _mm512_and_si512(e, _mm512_set1_epi64(0x7FFFFFFF));
        ix = _mm512_sub_epi64(_mm512_max_epu64(ix, one), one);
        _mm512_storeu_si512((void *)rows, _mm512_add_epi64(_mm512_set1_epi64((long long)(uintptr_t)T), _mm512_slli_epi64(ix, 6)));
        return _mm512_test_epi64_mask(e, _mm512_set1_epi64(0x80000000LL));
    }
};
/* The EC part of one batch on the 8-lane path: z*A for every candidate (g.nw table windows), then
 * both recovery ids against C. zb holds the candidates' z words, ds receives the digits (window-major). */
Q8TX static void vec_batch(const Ctx *c, const uint32_t *zb, uint32_t *ds, int B, VecBuf &v) {
    const int G = B / 8;
    const Geo &g = c->g;
    recode16(g, zb, ds, v.bad, B);
    /* two row buffers: this window's rows and the next window's, filled (and prefetched) one pass ahead */
    static thread_local std::vector<const pt *> rpa, rpb;
    static thread_local std::vector<__mmask8> nga, ngb;
    if ((int)rpa.size() < G * 8) { rpa.resize((size_t)G * 8); rpb.resize((size_t)G * 8); nga.resize((size_t)G); ngb.resize((size_t)G); }
    const pt **rp = rpa.data(), **rpn = rpb.data(); __mmask8 *ng = nga.data(), *ngn = ngb.data();
    ec8_first(v.X, v.Y, G, RowSgn{c->tw[0], ds}, RowSgn{c->tw[1], ds + B}, rp, ng);
    for (int i = 1; i < g.nw; i++) {
        if (i + 1 < g.nw) { const RowSgn rn{c->tw[i + 1], ds + (size_t)(i + 1) * B}; ec8_window(v.X, v.Y, v.D, v.P, v.TX, v.TY, G, rp, ng, rpn, ngn, &rn); }
        else ec8_window<RowSgn>(v.X, v.Y, v.D, v.P, v.TX, v.TY, G, rp, ng, nullptr, nullptr, nullptr);
        std::swap(rp, rpn); std::swap(ng, ngn);
    }
    ec8_final(v.X, v.Y, v.D, v.P, G, c->cx, c->cy, v.qx, v.qp);
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
    std::vector<uint32_t, qalloc64<uint32_t> > zb((size_t)B * 8), ds((size_t)NWMAX * B);   /* z words; digits */
    std::vector<uint8_t> ngs(B);
    std::vector<uint8_t> skips((size_t)B * 9 + 8);   /* +8: the wide skip stores below overrun by 3 bytes */
    uint8_t pk[64]; memset(pk, 0, 64); pk[33] = 0x80; pk[62] = 0x01; pk[63] = 0x08;   /* 264 bits */
    uint8_t blk2[64]; memset(blk2, 0, 64); blk2[32] = 0x80; blk2[62] = 0x01;          /* 256 bits */
    uint64_t epoch = (uint64_t)tid;
    int wi = c->ncwin;
    SHA256_CTX ectx; uint8_t early[16];
    std::vector<uint8_t> pbuf((size_t)dp->n * SIG_PUSH_SIZE + 64);
    auto put_digits = [&](int kk, const uint32_t *h) { memcpy(&zb[(size_t)kk * 8], h, 32); };   /* z = h[0] (MSW) .. h[7] */
#if QCPU_SHANI
    /* 4-lane path: each lane holds one candidate's chaining state (its epoch's) and the padded rest of
     * its message: the epoch's buffered bytes, the kept window pushes, tail section, suffix. */
    const size_t tl = dp->tail_section_len, sl = dp->tx_suffix_len;
    alignas(16) uint32_t lst[4][8]; alignas(16) uint8_t lmsg[4][16 * 64]; const uint8_t *lp[4];
    uint32_t est[8]; uint8_t erem[64]; size_t remlen = 0; int nb = 0; uint64_t tbits = 0;
    bool shani = c->shani;
    /* planned path: per-epoch block-0 states of the ng groups, fixed later blocks by schedule */
    const bool hplan = shani && c->hplan;
    std::vector<uint8_t> lgblk(c->h_gblk);                 /* this worker's group blocks (epoch bytes patched) */
    /* block-0 schedules of the groups; they change only with the epoch's buffered bytes (erem), which
     * are the same for most consecutive epochs of a worker */
    std::vector<uint32_t, qalloc64<uint32_t> > gwk((size_t)(c->h_ng + 3) * 64);
    uint8_t gwk_erem[64]; bool gwk_ok = false;
    /* the epoch prefix (prefix remainder + kept early pushes) and its chaining state after every full
     * block: consecutive epochs of a worker differ only from their first differing early omission on */
    const size_t pfx_max = dp->prefix_remainder_len + (size_t)c->cut * SIG_PUSH_SIZE;
    std::vector<uint8_t> pfx(pfx_max + 64);
    if (dp->prefix_remainder_len) memcpy(pfx.data(), dp->prefix_remainder, dp->prefix_remainder_len);
    std::vector<uint32_t> pst((pfx_max / 64 + 2) * 8);
    memcpy(pst.data(), dp->midstate, 32);
    uint8_t pv_early[16]; bool pv_ok = false;
    alignas(16) uint32_t gst[286 + 3][8];
    const uint32_t *const *lrow[4];
    alignas(16) uint32_t w2[4][16]; memset(w2, 0, sizeof w2);
    for (int l = 0; l < 4; l++) { w2[l][8] = 0x80000000u; w2[l][15] = 256; }   /* second SHA: 32-byte message */
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
#if QCPU_SHANI
                if (shani && hplan) {                       /* re-hash the prefix from the first block this epoch changes */
                    {                                       /* qsb_host_unrank with a binomial table */
                        uint64_t rank = epoch; int lo = 0;
                        for (int i = 0; i < c->early; i++) {
                            int cc = lo;
                            for (;;) { const uint64_t cnt = c->h_binom[c->cut - cc - 1][c->early - i - 1]; if (rank < cnt) break; rank -= cnt; cc++; }
                            early[i] = (uint8_t)cc; lo = cc + 1;
                        }
                    }
                    const size_t prl = dp->prefix_remainder_len;
                    size_t from = 0;
                    if (pv_ok) {
                        int e = 0; while (e < c->early && early[e] == pv_early[e]) e++;
                        const int lo = e < c->early ? (early[e] < pv_early[e] ? early[e] : pv_early[e]) : c->cut;
                        from = prl + (size_t)(lo - e) * SIG_PUSH_SIZE;   /* the kept pushes below push lo are unchanged */
                    }
                    size_t pl = 0; int e2 = 0;
                    for (int i = 0; i < c->cut; i++) {
                        if (e2 < c->early && early[e2] == i) { e2++; continue; }
                        memcpy(&pfx[prl + pl], dp->dummy_sigs + (size_t)i * SIG_PUSH_SIZE, SIG_PUSH_SIZE); pl += SIG_PUSH_SIZE;
                    }
                    const size_t lp = prl + pl, nfull = lp / 64;
                    SHA256_CTX pc;
                    for (size_t b = from / 64; b < nfull; b++) {
                        for (int i = 0; i < 8; i++) pc.h[i] = pst[b * 8 + i];
                        SHA256_Transform(&pc, &pfx[b * 64]);
                        for (int i = 0; i < 8; i++) pst[(b + 1) * 8 + i] = (uint32_t)pc.h[i];
                    }
                    memcpy(est, &pst[nfull * 8], 32);
                    remlen = lp % 64; memcpy(erem, &pfx[nfull * 64], remlen);
                    nb = c->h_nb;
                    memcpy(pv_early, early, (size_t)c->early); pv_ok = true;
                    const int ng = c->h_ng;
                    if (!gwk_ok || memcmp(gwk_erem, erem, remlen)) {
                        for (int q = 0; q < ng; q++) { memcpy(&lgblk[(size_t)q * 64], erem, remlen); qsha_schedule(&gwk[(size_t)q * 64], &lgblk[(size_t)q * 64]); }
                        for (int q = ng; q < ng + 3; q++) memcpy(&gwk[(size_t)q * 64], &gwk[0], 64 * sizeof(uint32_t));   /* lane padding */
                        memcpy(gwk_erem, erem, remlen); gwk_ok = true;
                    }
                    for (int g = 0; g < ng; g += 4) {      /* block 0 of each group: once per epoch, not per candidate */
                        const uint32_t *gp[4]; const uint32_t *const *gr[4] = {&gp[0], &gp[1], &gp[2], &gp[3]};
                        for (int l = 0; l < 4; l++) { memcpy(gst[g + l], est, 32); gp[l] = &gwk[(size_t)(g + l) * 64]; }
                        qsha_x4p(&gst[g], gr, 1);
                    }
                    epoch += (uint64_t)c->nthreads; wi = 0;
                    goto have_epoch;
                }
#endif
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
                }
#endif
                epoch += (uint64_t)c->nthreads; wi = 0;
            }
#if QCPU_SHANI
            have_epoch:
#endif
            const int pi = wi++;
            const uint8_t *w3 = c->cwin[pi];
#if QCPU_SHANI
            if (shani && hplan) {
                const int j = k & 3;
                memcpy(lst[j], gst[c->h_g0[pi]], 32);
                lrow[j] = c->h_wkp[pi];
                uint8_t *sk = &skips[(size_t)k * 9];
                { uint64_t e8; memcpy(&e8, early, 8); memcpy(sk, &e8, 8);                   /* bytes 0..5 = early, 6..7 overwritten next */
                  const uint32_t w4 = (uint32_t)w3[0] | (uint32_t)w3[1] << 8 | (uint32_t)w3[2] << 16; memcpy(sk + 6, &w4, 4); }
                inf[k] = 1; bad[k] = 0;
                if (j == 3) {                               /* four candidates ready: blocks 1..nb-1, then the second SHA-256 */
                    qsha_x4p(lst, lrow, nb - 1);
                    alignas(16) uint32_t s2[4][8];
                    for (int l = 0; l < 4; l++) memcpy(w2[l], lst[l], 32);
                    qsha_x4w_iv(s2, w2);
                    for (int l = 0; l < 4; l++) {           /* z = the state words h0 (MSW) .. h7 */
                        uint32_t *zr = &zb[(size_t)(k - 3 + l) * 8];
                        _mm_storeu_si128((__m128i *)zr, _mm_load_si128((const __m128i *)&s2[l][0]));
                        _mm_storeu_si128((__m128i *)(zr + 4), _mm_load_si128((const __m128i *)&s2[l][4]));
                    }
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
#if QCPU_VEC
        if (vb.X) {
            vec_batch(c, zb.data(), ds.data(), B, vb);
#if QCPU_SHANI
            if (c->shani) {                             /* key hashes 4 at a time: 2 candidates x 2 recids */
                alignas(16) uint32_t pw[4][16]; memset(pw, 0, sizeof pw);
                for (int l = 0; l < 4; l++) pw[l][15] = 264;
                for (int kk = 0; kk < B; kk += 2) {
                    alignas(16) uint32_t hs[4][8];
                    for (int l = 0; l < 4; l++) {
                        const size_t q = (size_t)(kk + (l >> 1)) * 2 + (l & 1);
                        pk_words(pw[l], vb.qx[q], vb.qp[q]);
                    }
                    qsha_x4w_iv(hs, pw);
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
        recode_scalar(c->g, zb.data(), ds.data(), B);
        for (int i = 0; i < c->g.nw; i++) {
            const pt *T = c->tw[i]; const uint32_t *di = &ds[(size_t)i * B];
            for (int kk = 0; kk < B; kk++) { const uint32_t e = di[kk], m = e & 0x7FFFFFFFu; tp[kk] = m ? &T[m - 1] : nullptr; ngs[kk] = (uint8_t)(e >> 31); }
            batch_add(acc.data(), tp.data(), inf.data(), bad.data(), B, d.data(), pre.data(), ngs.data());
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
/* The process's CPU set as it was before main(): other start-up code (the host producers) may later pin
 * the main thread to its own core, and the co-grinder must still see, and run on, the rest of the CPUs. */
#ifdef CPU_COUNT
static cpu_set_t g_initial_cpus;
static int g_initial_ok = [] { CPU_ZERO(&g_initial_cpus); return sched_getaffinity(0, sizeof g_initial_cpus, &g_initial_cpus) == 0 ? 1 : 0; }();
#endif
static void start(const digest_params_t *dp, const uint8_t win3[][3], int nwin, int cut, int early) {
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
#ifdef CPU_COUNT
    cpu_set_t work_cpus; CPU_ZERO(&work_cpus); bool work_mask = false;
    {
        cpu_set_t cs; CPU_ZERO(&cs);
        if (g_initial_ok) cs = g_initial_cpus; else if (sched_getaffinity(0, sizeof cs, &cs) != 0) CPU_ZERO(&cs);
        if (CPU_COUNT(&cs) > 0) ncpu = CPU_COUNT(&cs);
        cpu_set_t now; CPU_ZERO(&now);                    /* the main thread's current set (maybe narrowed) */
        if (g_initial_ok && sched_getaffinity(0, sizeof now, &now) == 0 && CPU_COUNT(&now) < CPU_COUNT(&cs)) {
            for (int c = 0; c < CPU_SETSIZE; c++) if (CPU_ISSET(c, &cs) && !CPU_ISSET(c, &now)) CPU_SET(c, &work_cpus);
            work_mask = CPU_COUNT(&work_cpus) >= 1;       /* workers: every CPU the main thread no longer uses */
        }
    }
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
#if QSB_CPU_VBMI2
    c->vec = c->vec && __builtin_cpu_supports("avx512vbmi2");
#endif
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
    if (c->shani) hash_plan(*c);
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
    c->g = geo_choose();
    if (!table_alloc(*c)) { c->g = geo_make(15, true); if (!table_alloc(*c)) { printf("  CPU co-grind: off (table memory)\n"); delete c; return; } }
#ifdef CPU_COUNT
    std::thread([c, fax, fay, nth, work_mask, work_cpus]() {
        if (work_mask) sched_setaffinity(0, sizeof work_cpus, &work_cpus);   /* table build + workers inherit */
#else
    std::thread([c, fax, fay, nth]() {
#endif
#ifdef SCHED_IDLE
        struct sched_param sp; sp.sched_priority = 0; sched_setscheduler(0, SCHED_IDLE, &sp);
#endif
        build_table(*c, fax, fay, nth);
        if (!table_check(*c)) { printf("  CPU co-grind: off (table check failed)\n"); fflush(stdout); table_free(*c); return; }
        for (int t = 0; t < nth; t++) std::thread(worker, c, t).detach();
    }).detach();
    g_ctx = c;
    printf("  CPU co-grind: %d threads (of %ld CPUs), %s, %s, %d window patterns per epoch disjoint from the GPU's %d; "
           "table %d %s windows of %d..%d bits, %.0f MiB\n",
           nth, ncpu, c->vec ? "8-lane IFMA" : "scalar", c->shani ? "4-lane SHA-NI" : "OpenSSL SHA-256", c->ncwin, nwin,
           c->g.nw, c->g.sgn ? "signed" : "unsigned", c->g.wid[c->g.nw - 1], c->g.wid[0], c->g.total * sizeof(pt) / 1048576.0);
    fflush(stdout);
}
static uint64_t candidates() { return g_ctx ? g_ctx->cand.load() : 0; }
static uint32_t hits() { return g_ctx ? g_ctx->hits.load() : 0; }
}  // namespace qcpu
