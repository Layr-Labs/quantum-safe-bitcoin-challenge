#pragma once
/* Host-CPU co-grinder (Q520): idle runner cores search a sequence range disjoint from
 * the GPU's (sequences count down from 0xFFFFFFFE; the GPU counts up from 0x80000000),
 * and every candidate hit passes the same exact OpenSSL gate (qsb_host_exact_hit)
 * before it is written to results/pinning_hit_cpu.txt. Worker threads run at
 * SCHED_IDLE so they never delay the GPU host thread.
 *
 * Math: Q(recid) = z*A +/- C with A = neg_r_inv*G and C = u2*R, z = SHA256d(preimage).
 * z*A uses 16 unsigned 16-bit windows over a host table T[i][d-1] = d*2^(16i)*A and
 * batch-affine additions (Montgomery's trick across a batch of candidates). */
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
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#ifndef QSB_CPU_GRIND
#define QSB_CPU_GRIND 1
#endif
#ifndef QSB_CPU_RESERVE
#define QSB_CPU_RESERVE 2          /* logical CPUs left for the GPU host thread and driver */
#endif
#ifndef QSB_CPU_BATCH
#define QSB_CPU_BATCH 4096
#endif
#ifndef QSB_CPU_W
#define QSB_CPU_W 16             /* window bits: 16 -> 16 adds, 64 MiB table; 12 -> 22 adds, 5.6 MiB */
#endif

namespace qcpu {
static const int W = QSB_CPU_W, NW = (256 + QSB_CPU_W - 1) / QSB_CPU_W, NE = (1 << QSB_CPU_W) - 1;
typedef unsigned __int128 u128;
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

struct Ctx {
    const pinning2_params_t *pp;
    std::vector<pt> table;          /* NW windows x NE entries */
    fe cx, cy;                      /* C = u2*R */
    std::atomic<uint64_t> cand{0};
    std::atomic<uint32_t> hits{0};
    std::mutex io;
    int nthreads = 0;
};

static inline void be_words_to_digest(const uint32_t h[8], uint8_t out[32]) {
    for (int i = 0; i < 8; i++) { out[4*i] = h[i] >> 24; out[4*i+1] = h[i] >> 16; out[4*i+2] = h[i] >> 8; out[4*i+3] = h[i]; }
}

/* Build T[i][j] = (j+1) * 2^(W i) * A, threads split by window. */
static void build_table(Ctx &c, const fe &ax, const fe &ay, int nth) {
    c.table.resize((size_t)NW * NE);
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
            T[0] = base[i];
            /* 2B by doubling, then T[j] = T[j-1] + B for the rest, in rounds of doubling width */
            T[1] = pt_double(base[i]);
            int have = 2;                                /* T[0..have-1] = 1..have multiples */
            std::vector<fe> d(NE + 1), pre(NE + 1);
            std::vector<uint8_t> inf(NE + 1, 0), bad(NE + 1, 0);
            std::vector<const pt *> tp(NE + 1);
            while (have < NE) {
                int n = have; if (have + n > NE) n = NE - have;
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

static void worker(Ctx *c, int tid) {
#ifdef SCHED_IDLE
    struct sched_param sp; sp.sched_priority = 0; sched_setscheduler(0, SCHED_IDLE, &sp);
#endif
    const pinning2_params_t *pp = c->pp;
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *bctx = BN_CTX_new();
    BIGNUM *order = BN_new(), *nri = BN_new(), *rx = BN_new(), *ry = BN_new();
    EC_POINT *R = EC_POINT_new(grp);
    EC_GROUP_get_order(grp, order, bctx);
    BN_lebin2bn(pp->neg_r_inv, 32, nri); BN_lebin2bn(pp->u2r_x, 32, rx); BN_lebin2bn(pp->u2r_y, 32, ry);
    EC_POINT_set_affine_coordinates_GFp(grp, R, rx, ry, bctx);

    const int B = QSB_CPU_BATCH;
    std::vector<pt> acc(B); std::vector<fe> d(2 * B), pre(2 * B);
    std::vector<uint8_t> inf(B), bad(B); std::vector<const pt *> tp(B);
    std::vector<uint16_t> dig((size_t)B * NW);
    std::vector<uint32_t> lts(B);
    const uint32_t seq = 0xFFFFFFFEu - (uint32_t)tid;
    const uint32_t sl = pp->suffix_len, so = pp->seq_offset, lo = pp->lt_offset;
    uint8_t buf[128]; memset(buf, 0, sizeof buf); memcpy(buf, pp->suffix, sl);
    buf[so] = seq; buf[so + 1] = seq >> 8; buf[so + 2] = seq >> 16; buf[so + 3] = seq >> 24;
    buf[sl] = 0x80;
    const int nblk = (sl < 56) ? 1 : 2;
    { uint64_t bits = (uint64_t)pp->total_preimage_len * 8; int lenoff = nblk * 64 - 8;
      for (int i = 0; i < 8; i++) buf[lenoff + 7 - i] = (uint8_t)(bits >> (8 * i)); }
    SHA256_CTX mid; SHA256_Init(&mid); for (int i = 0; i < 8; i++) mid.h[i] = pp->midstate[i];
    if (nblk == 2 && lo >= 64) SHA256_Transform(&mid, buf);   /* lt lives in the last block */
    const bool lt_in_last = (nblk == 1) || lo >= 64;
    FILE *out = nullptr;
    uint32_t lt = 0;
    uint8_t blk2[64];                                  /* second-SHA block: 32-byte digest + pad */
    memset(blk2, 0, 64); blk2[32] = 0x80; blk2[62] = 0x01;   /* 256 bits */
    uint8_t pk[64]; memset(pk, 0, 64); pk[33] = 0x80; pk[62] = 0x01; pk[63] = 0x08;  /* 264 bits */
    for (;;) {
        /* SHA256d for B consecutive locktimes, then the 16 digits of each z */
        for (int k = 0; k < B; k++, lt++) {
            lts[k] = lt;
            buf[lo] = lt; buf[lo + 1] = lt >> 8; buf[lo + 2] = lt >> 16; buf[lo + 3] = lt >> 24;
            SHA256_CTX s = mid;
            if (lt_in_last) SHA256_Transform(&s, buf + (nblk == 2 ? 64 : 0));
            else { SHA256_Transform(&s, buf); SHA256_Transform(&s, buf + 64); }
            be_words_to_digest(s.h, blk2);
            SHA256_CTX s2; SHA256_Init(&s2); SHA256_Transform(&s2, blk2);
            /* z big-endian = s2.h[0] (most significant) .. s2.h[7]; window i = bits 16i..16i+15 */
            uint64_t zl[4];                                /* little-endian limbs of z */
            for (int i = 0; i < 4; i++) zl[i] = ((uint64_t)s2.h[6 - 2 * i] << 32) | s2.h[7 - 2 * i];
            for (int i = 0; i < NW; i++) {
                int bit = i * W, li = bit >> 6, sh = bit & 63;
                uint64_t v = zl[li] >> sh;
                if (sh + W > 64 && li < 3) v |= zl[li + 1] << (64 - sh);
                dig[(size_t)k * NW + i] = (uint16_t)(v & NE);
            }
            inf[k] = 1; bad[k] = 0;
        }
        for (int i = 0; i < NW; i++) {
            const pt *T = &c->table[(size_t)i * NE];
            for (int k = 0; k < B; k++) { uint16_t v = dig[(size_t)k * NW + i]; tp[k] = v ? &T[v - 1] : nullptr; }
            batch_add(acc.data(), tp.data(), inf.data(), bad.data(), B, d.data(), pre.data());
        }
        /* Both recids share the denominator x_C - x_P. */
        fe run = {{1, 0, 0, 0}};
        for (int k = 0; k < B; k++) {
            if (bad[k] || inf[k]) { pre[k] = run; continue; }
            fe_sub(d[k], c->cx, acc[k].x);
            if (fe_is_zero(d[k])) { bad[k] = 1; pre[k] = run; continue; }
            pre[k] = run; fe_mul(run, run, d[k]);
        }
        fe inv; fe_inv(inv, run);
        for (int k = B - 1; k >= 0; k--) {
            if (bad[k] || inf[k]) continue;
            fe dinv; fe_mul(dinv, inv, pre[k]); fe_mul(inv, inv, d[k]);
            for (int ri = 0; ri < 2; ri++) {
                fe cy = c->cy; if (ri) { fe z0 = {{0, 0, 0, 0}}; fe_sub(cy, z0, cy); }   /* recid 1: -C */
                fe lam, t, x3, y3;
                fe_sub(t, cy, acc[k].y); fe_mul(lam, t, dinv);
                fe_sqr(x3, lam); fe_sub(x3, x3, acc[k].x); fe_sub(x3, x3, c->cx);
                fe_sub(t, acc[k].x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, acc[k].y);
                pk[0] = (uint8_t)(0x02 | (y3.v[0] & 1));
                for (int b = 0; b < 32; b++) pk[1 + b] = (uint8_t)(x3.v[3 - (b >> 3)] >> (8 * (7 - (b & 7))));
                SHA256_CTX s3; SHA256_Init(&s3); SHA256_Transform(&s3, pk);
                if ((s3.h[0] >> (32 - (QSB_ZEROS_N < 32 ? QSB_ZEROS_N : 32))) != 0) continue;
                if (!qsb_host_exact_hit(pp, seq, lts[k], ri, grp, bctx, order, nri, R)) continue;
                std::lock_guard<std::mutex> g(c->io);
                if (!out) { mkdir("results", 0755); out = fopen("results/pinning_hit_cpu.txt", "a"); }
                if (out) { fprintf(out, "sequence=%u locktime=%u recid=%d\n", seq, lts[k], ri); fflush(out); }
                c->hits++;
                break;                                  /* at most one recid per candidate, like the GPU */
            }
        }
        c->cand += B;
    }
}

static Ctx *g_ctx = nullptr;
static void start(const pinning2_params_t *pp) {
#if QSB_CPU_GRIND
    /* Usable CPUs: the affinity mask, capped by a cgroup v2 cpu.max quota when present
     * (containers often expose every host CPU but grant only a share). */
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
    int nth = QSB_CPU_THREADS;                    /* dev override: emulate a given runner */
#else
    int nth = (int)ncpu - QSB_CPU_RESERVE;
#endif
    if (nth < 1) { printf("  CPU co-grind: off (%ld CPUs)\n", ncpu); return; }
    Ctx *c = new Ctx(); c->pp = pp; c->nthreads = nth;
    /* A = neg_r_inv * G via OpenSSL, C = u2*R from the problem. */
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *bctx = BN_CTX_new(); BIGNUM *nri = BN_new(), *ax = BN_new(), *ay = BN_new();
    EC_POINT *A = EC_POINT_new(grp);
    BN_lebin2bn(pp->neg_r_inv, 32, nri);
    if (!EC_POINT_mul(grp, A, nri, NULL, NULL, bctx) ||
        !EC_POINT_get_affine_coordinates_GFp(grp, A, ax, ay, bctx)) { printf("  CPU co-grind: off (A)\n"); return; }
    fe fax, fay; fe_from_bn(fax, ax); fe_from_bn(fay, ay);
    fe_from_le32(c->cx, pp->u2r_x); fe_from_le32(c->cy, pp->u2r_y);
    EC_POINT_free(A); BN_free(nri); BN_free(ax); BN_free(ay); BN_CTX_free(bctx); EC_GROUP_free(grp);
    std::thread([c, fax, fay, nth]() {
#ifdef SCHED_IDLE
        struct sched_param sp; sp.sched_priority = 0; sched_setscheduler(0, SCHED_IDLE, &sp);
#endif
        build_table(*c, fax, fay, nth);
        for (int t = 0; t < nth; t++) std::thread(worker, c, t).detach();
    }).detach();
    g_ctx = c;
    printf("  CPU co-grind: %d threads (of %ld CPUs), sequences from 0x%08X downward\n", nth, ncpu, 0xFFFFFFFEu);
#else
    (void)pp;
#endif
}
static uint64_t candidates() { return g_ctx ? g_ctx->cand.load() : 0; }
static uint32_t hits() { return g_ctx ? g_ctx->hits.load() : 0; }
}  // namespace qcpu
