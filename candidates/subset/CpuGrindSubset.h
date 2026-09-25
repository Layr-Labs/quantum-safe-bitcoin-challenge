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
 * SCHED_IDLE when permitted; CPU quota/cache/bandwidth contention can still delay GPU work.
 * Subset adaptation: terrapinelf, public PR1593 / 7d8df21525bdd7162798e645543487cf13091a4c.
 * Local port adds owned cancellation, capped budgets and checked output. GPL-3 retained.
 * QSB_CPU_GRIND=0 compiles the adjunct out. */
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
#include <errno.h>
#include <limits.h>
#include <fcntl.h>
#include <memory>
#include <string>
#include <exception>

#ifndef QSB_CPU_RESERVE
#define QSB_CPU_RESERVE 2          /* logical CPUs left for the GPU host thread and driver */
#endif
#ifndef QSB_CPU_BATCH
#define QSB_CPU_BATCH 4096
#endif
#ifndef QSB_CPU_W
#define QSB_CPU_W 16
#endif

static_assert(QSB_CPU_BATCH == 4096 && QSB_CPU_W == 16, "CPU port freezes donor batch/window");
static_assert(QSB_CPU_RESERVE >= 1, "CPU port reserves at least one logical CPU");

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
 * (exceptional candidates are dropped; exact publication is not a recall proof). */
static bool batch_add(pt *acc, const pt *const *tp, uint8_t *inf, uint8_t *bad, int n,
                      fe *d, fe *pre, const std::atomic<bool> *stop = nullptr) {
    fe run = {{1, 0, 0, 0}};
    for (int k = 0; k < n; k++) {
        if ((k & 63) == 0 && stop && stop->load(std::memory_order_relaxed)) return false;
        if (k + 16 < n && tp[k + 16]) __builtin_prefetch(tp[k + 16]);
        if (bad[k] || !tp[k]) { pre[k] = run; continue; }
        if (inf[k]) { pre[k] = run; continue; }
        fe_sub(d[k], tp[k]->x, acc[k].x);
        if (fe_is_zero(d[k])) { bad[k] = 1; pre[k] = run; continue; }
        pre[k] = run; fe_mul(run, run, d[k]);
    }
    fe inv; fe_inv(inv, run);
    for (int k = n - 1; k >= 0; k--) {
        if ((k & 63) == 0 && stop && stop->load(std::memory_order_relaxed)) return false;
        if (bad[k] || !tp[k]) continue;
        if (inf[k]) { acc[k] = *tp[k]; inf[k] = 0; continue; }
        fe dinv; fe_mul(dinv, inv, pre[k]); fe_mul(inv, inv, d[k]);
        fe lam, t, x3, y3;
        fe_sub(t, tp[k]->y, acc[k].y); fe_mul(lam, t, dinv);
        fe_sqr(x3, lam); fe_sub(x3, x3, acc[k].x); fe_sub(x3, x3, tp[k]->x);
        fe_sub(t, acc[k].x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, acc[k].y);
        acc[k].x = x3; acc[k].y = y3;
    }
    return true;
}

struct Ctx {
    const digest_params_t *dp = nullptr; // owner joins before borrowed problem storage dies
    std::vector<pt> table;
    fe cx, cy;
    uint8_t cwin[286][3];
    int ncwin = 0, cut = 137, early = 6, nthreads = 0;
    uint64_t mid_bytes = 0, n_epochs = 0;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> cand{0}, hits{0};
    std::mutex io;
    qsb_hv_t hv{}; // independent CPU gate, used only under io once workers begin
    int outfd = -1;
    off_t output_bytes = 0;
    ~Ctx() {
        if (outfd >= 0) {
            if (fsync(outfd) != 0) fprintf(stderr, "CPU co-grind: final flush failed errno=%d\n", errno);
            if (close(outfd) != 0) fprintf(stderr, "CPU co-grind: close failed errno=%d\n", errno);
        }
        EC_POINT_free(hv.Ru2); BN_free(hv.order); BN_free(hv.nri);
        BN_CTX_free(hv.ctx); EC_GROUP_free(hv.grp);
    }
};
static bool stopped(const Ctx &c) { return c.stop.load(std::memory_order_relaxed); }
static void fail(Ctx &c, const char *what) {
    const int saved = errno;
    c.stop.store(true, std::memory_order_relaxed);
    fprintf(stderr, "CPU co-grind disabled: %s (errno=%d)\n", what, saved);
}
// Destructor joins a partially created set during allocation/thread-start failure.
struct Threads {
    Ctx &c;
    std::vector<std::thread> v;
    explicit Threads(Ctx &ctx): c(ctx) {}
    void join() { for (auto &t : v) if (t.joinable()) t.join(); }
    ~Threads() { for (auto &t : v) if (t.joinable()) c.stop.store(true); join(); }
};
template<class F> static void guarded(Ctx &c, F fn) noexcept {
    try { fn(); }
    catch (const std::exception &e) { fail(c, e.what()); }
    catch (...) { fail(c, "unknown worker exception"); }
}
// Called by the coordinator before spawning either phase. Children inherit its policy.
static bool idle_or_disable(Ctx &c) {
#ifdef SCHED_IDLE
    struct sched_param sp{};
    if (sched_setscheduler(0, SCHED_IDLE, &sp) == 0) return true;
    fail(c, "SCHED_IDLE failed; zero-worker fallback");
#else
    errno = ENOTSUP; fail(c, "SCHED_IDLE unavailable; zero-worker fallback");
#endif
    return false;
}
// Exact gate and output are serialized. Only a fully written record increments hits.
static bool publish(Ctx &c, const uint8_t *sk, int ri) {
    std::lock_guard<std::mutex> lock(c.io);
    if (stopped(c) || !qsb_hv_check(&c.hv, sk, ri)) return false;
    if (c.outfd < 0) {
        if (mkdir("results", 0755) != 0 && errno != EEXIST) { fail(c, "mkdir"); return false; }
        // A second start/process must not append duplicate CPU candidates to an old file.
        c.outfd = open("results/digest_hit_cpu.txt", O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC, 0644);
        if (c.outfd < 0) { fail(c, "exclusive hit-file open"); return false; }
    }
    char line[128];
    int n = snprintf(line, sizeof line, "indices=%d,%d,%d,%d,%d,%d,%d,%d,%d recid=%d\n",
        sk[0],sk[1],sk[2],sk[3],sk[4],sk[5],sk[6],sk[7],sk[8],ri);
    if (n < 0 || (size_t)n >= sizeof line) { errno=EOVERFLOW; fail(c,"hit format"); return false; }
    int done=0;
    while (done<n) {
        ssize_t wrote=write(c.outfd,line+done,(size_t)(n-done));
        if (wrote<0 && errno==EINTR) continue;
        if (wrote<=0) {
            int saved=errno;
            if (ftruncate(c.outfd,c.output_bytes)!=0) fprintf(stderr,"CPU co-grind: partial-line rollback failed errno=%d\n",errno);
            errno=saved; fail(c,"hit write"); return false;
        }
        done+=(int)wrote;
    }
    c.output_bytes+=n; c.hits.fetch_add(1,std::memory_order_relaxed);
    return true;
}

/* Build T[i][j] = (j+1) * 2^(W i) * A, threads split by window. */
static bool build_table(Ctx &c, const fe &ax, const fe &ay, int nth) {
    if (stopped(c)) return false;
    c.table.resize((size_t)NW * NE);
    std::vector<pt> base(NW);
    base[0] = {ax, ay};
    for (int i = 1; i < NW; i++) {                     /* base[i] = 2^W * base[i-1] by W doublings */
        if (stopped(c)) return false;
        pt q = base[i - 1];
        for (int s = 0; s < W; s++) { if (stopped(c)) return false; q = pt_double(q); }
        base[i] = q;
    }
    auto work = [&](int w0) {
        for (int i = w0; i < NW; i += nth) {
            if (stopped(c)) return;
            pt *T = &c.table[(size_t)i * NE];
            T[0] = base[i];
            /* 2B by doubling, then T[j] = T[j-1] + B for the rest, in rounds of doubling width */
            T[1] = pt_double(base[i]);
            int have = 2;                                /* T[0..have-1] = 1..have multiples */
            std::vector<fe> d(NE + 1), pre(NE + 1);
            std::vector<uint8_t> inf(NE + 1, 0), bad(NE + 1, 0);
            std::vector<const pt *> tp(NE + 1);
            while (have < NE) {
                if (stopped(c)) return;
                int n = have; if (have + n > NE) n = NE - have;
                /* T[have+k] = T[k] + T[have-1]  ((k+1) + have = have+k+1) */
                for (int k = 0; k < n; k++) { T[have + k] = T[k]; tp[k] = &T[have - 1]; inf[k] = 0; bad[k] = 0; }
                if (!batch_add(&T[have], tp.data(), inf.data(), bad.data(), n, d.data(), pre.data(), &c.stop)) return;
                /* k = have-1 adds T[have-1] to itself: equal x, so batch_add flags it; double it. */
                for (int k = 0; k < n; k++) if (bad[k]) T[have + k] = pt_double(T[have - 1]);
                have += n;
            }
        }
    };
    Threads ts(c); ts.v.reserve(nth);
    for (int t=0;t<nth && !stopped(c);t++) ts.v.emplace_back([&,t]() { guarded(c,[&]() { work(t); }); });
    ts.join();
    return !stopped(c);
}

static void worker(Ctx *c, int tid) {
    uint64_t completed=0;
    struct CountOnExit {
        std::atomic<uint64_t> &out; uint64_t &local;
        ~CountOnExit() { out.fetch_add(local,std::memory_order_relaxed); }
    } count_on_exit{c->cand,completed}; // count only finished candidates, including explicit exception drops
    const digest_params_t *dp = c->dp;
    const int B = QSB_CPU_BATCH;
    std::vector<pt> acc(B); std::vector<fe> d(2 * B), pre(2 * B);
    std::vector<uint8_t> inf(B), bad(B); std::vector<const pt *> tp(B);
    std::vector<uint16_t> dig((size_t)B * NW);
    std::vector<uint8_t> skips((size_t)B * 9);
    uint8_t pk[64]; memset(pk, 0, 64); pk[33] = 0x80; pk[62] = 0x01; pk[63] = 0x08;   /* 264 bits */
    uint8_t blk2[64]; memset(blk2, 0, 64); blk2[32] = 0x80; blk2[62] = 0x01;          /* 256 bits */
    uint64_t epoch = (uint64_t)tid;
    int wi = c->ncwin;
    SHA256_CTX ectx; uint8_t early[16];
    std::vector<uint8_t> pbuf((size_t)dp->n * SIG_PUSH_SIZE + 64);
    for (;;) {
        if (stopped(*c)) return;
        int k = 0;
        while (k < B) {
            if (stopped(*c)) return;
            if (wi == c->ncwin) {                       /* next epoch: hash its fixed prefix once */
                if (epoch >= c->n_epochs) break;
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
                epoch += (uint64_t)c->nthreads; wi = 0;
            }
            const uint8_t *w3 = c->cwin[wi++];
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
            uint64_t zl[4];                             /* little-endian limbs of z = s2.h[0] (MSW) .. h[7] */
            for (int i = 0; i < 4; i++) zl[i] = ((uint64_t)s2.h[6 - 2 * i] << 32) | s2.h[7 - 2 * i];
            for (int i = 0; i < NW; i++) {
                int bit = i * W, li = bit >> 6, sh = bit & 63;
                uint64_t v = zl[li] >> sh;
                if (sh + W > 64 && li < 3) v |= zl[li + 1] << (64 - sh);
                dig[(size_t)k * NW + i] = (uint16_t)(v & NE);
            }
            uint8_t *sk = &skips[(size_t)k * 9];
            for (int j = 0; j < 6; j++) sk[j] = early[j];
            sk[6] = w3[0]; sk[7] = w3[1]; sk[8] = w3[2];
            inf[k] = 1; bad[k] = 0;
            k++;
        }
        if (!k) return;
        const int count=k; // process a final partial batch without counting uncomputed candidates
        for (int i = 0; i < NW; i++) {
            if (stopped(*c)) return;
            const pt *T = &c->table[(size_t)i * NE];
            for (int kk = 0; kk < count; kk++) { uint16_t v = dig[(size_t)kk * NW + i]; tp[kk] = v ? &T[v - 1] : nullptr; }
            if (!batch_add(acc.data(), tp.data(), inf.data(), bad.data(), count, d.data(), pre.data(), &c->stop)) return;
        }
        /* Both recids share the denominator x_C - x_P. */
        fe run = {{1, 0, 0, 0}};
        for (int kk = 0; kk < count; kk++) {
            if ((kk & 63)==0 && stopped(*c)) return;
            if (bad[kk] || inf[kk]) { pre[kk] = run; continue; }
            fe_sub(d[kk], c->cx, acc[kk].x);
            if (fe_is_zero(d[kk])) { bad[kk] = 1; pre[kk] = run; continue; }
            pre[kk] = run; fe_mul(run, run, d[kk]);
        }
        fe inv; fe_inv(inv, run);
        for (int kk = count - 1; kk >= 0; kk--) {
            if (stopped(*c)) return;
            if (bad[kk] || inf[kk]) { ++completed; continue; }
            fe dinv; fe_mul(dinv, inv, pre[kk]); fe_mul(inv, inv, d[kk]);
            for (int ri = 0; ri < 2; ri++) {
                if (stopped(*c)) return;
                fe cy = c->cy; if (ri) { fe z0 = {{0, 0, 0, 0}}; fe_sub(cy, z0, cy); }   /* recid 1: -C */
                fe lam, t, x3, y3;
                fe_sub(t, cy, acc[kk].y); fe_mul(lam, t, dinv);
                fe_sqr(x3, lam); fe_sub(x3, x3, acc[kk].x); fe_sub(x3, x3, c->cx);
                fe_sub(t, acc[kk].x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, acc[kk].y);
                pk[0] = (uint8_t)(0x02 | (y3.v[0] & 1));
                for (int b = 0; b < 32; b++) pk[1 + b] = (uint8_t)(x3.v[3 - (b >> 3)] >> (8 * (7 - (b & 7))));
                SHA256_CTX s3; SHA256_Init(&s3); SHA256_Transform(&s3, pk);
                if ((s3.h[0] >> (32 - (QSB_ZEROS_N < 32 ? QSB_ZEROS_N : 32))) != 0) continue;
                const uint8_t *sk = &skips[(size_t)kk * 9];
                if (publish(*c, sk, ri)) break; // one verified recid per unique candidate
                if (stopped(*c)) return;
            }
            ++completed;
        }
    }
}

// Conservative integer CPU budget: affinity, then every visible cgroup ancestor.
static void quota_at(const std::string &dir, bool v2, long &budget) {
    long long quota=-1, period=0;
    if (v2) {
        FILE *f=fopen((dir+"/cpu.max").c_str(),"r");
        if (!f) return;
        char q[64]{};
        int fields=fscanf(f,"%63s %lld",q,&period);
        if (fields!=2 || period<=0) { budget=0; fclose(f); return; }
        if (strcmp(q,"max")!=0) {
            char *end=nullptr; errno=0; quota=strtoll(q,&end,10);
            if (errno || !end || *end) quota=0;
        }
        fclose(f);
    } else {
        FILE *f=fopen((dir+"/cpu.cfs_quota_us").c_str(),"r");
        if (f) { if (fscanf(f,"%lld",&quota)!=1) quota=0; fclose(f); }
        f=fopen((dir+"/cpu.cfs_period_us").c_str(),"r");
        if (f) { if (fscanf(f,"%lld",&period)!=1) period=0; fclose(f); }
    }
    // Round DOWN, unlike donor. Fractional capacity is not a whole worker slot.
    if (quota>=0 && period<=0) budget=0;
    else if (quota>=0 && quota/period<budget) budget=(long)(quota/period);
}
static void quota_chain(const std::string &root, const std::string &relative, bool v2, long &budget) {
    if (relative.empty() || relative[0]!='/' || relative.find("..")!=std::string::npos) return;
    std::string p=root+(relative=="/" ? "" : relative);
    while (p.size()>=root.size()) {
        quota_at(p,v2,budget);
        if (p==root) break;
        size_t slash=p.rfind('/'); if (slash<root.size()) p=root; else p.resize(slash);
    }
}
static int thread_budget(long &budget) {
    budget=0;
#ifdef CPU_COUNT
    cpu_set_t affinity; CPU_ZERO(&affinity);
    if (sched_getaffinity(0,sizeof affinity,&affinity)!=0) return 0;
    budget=CPU_COUNT(&affinity);
#else
    return 0; // Unknown affinity is not permission to consume every online CPU.
#endif
    if (FILE *f=fopen("/proc/self/cgroup","r")) {
        char line[4096];
        while (fgets(line,sizeof line,f)) {
            char *first=strchr(line,':'); if (!first) continue;
            char *second=strchr(first+1,':'); if (!second) continue;
            *second=0; std::string controllers=","+std::string(first+1)+",";
            std::string relative=second+1; while (!relative.empty() && (relative.back()=='\n'||relative.back()=='\r')) relative.pop_back();
            if (first+1==second) quota_chain("/sys/fs/cgroup",relative,true,budget);
            if (controllers.find(",cpu,")!=std::string::npos) {
                quota_chain("/sys/fs/cgroup/cpu",relative,false,budget);
                quota_chain("/sys/fs/cgroup/cpu,cpuacct",relative,false,budget);
            }
        }
        fclose(f);
    }
    // Also cover a cgroup namespace mounted directly at its effective root.
    quota_at("/sys/fs/cgroup",true,budget);
    quota_at("/sys/fs/cgroup/cpu",false,budget);
    quota_at("/sys/fs/cgroup/cpu,cpuacct",false,budget);
    long available=budget>QSB_CPU_RESERVE ? budget-QSB_CPU_RESERVE : 0;
    if (available>32) available=32; // bounded optional host resource footprint
    long requested=available;
#ifdef QSB_CPU_THREADS
    requested=QSB_CPU_THREADS;
#endif
    if (const char *e=getenv("QSB_CPU_THREADS_ENV")) {
        char *end=nullptr; errno=0; long value=strtol(e,&end,10);
        if (errno || end==e || !end || *end) return 0;
        requested=value;
    }
    if (requested<0) requested=0;
    if (requested>available) requested=available;
    return (int)requested;
}
static bool prepare(Ctx &c, const digest_params_t *dp, const uint8_t win3[][3],
                    int nwin, int cut, int early, fe &fax, fe &fay) {
    if (dp->n!=150 || dp->t!=9 || cut!=137 || early!=6 || nwin!=128 || dp->prefix_remainder_len>=64) return false;
    for (int i=0;i<nwin;i++) {
        if (win3[i][0]<cut || win3[i][0]>=win3[i][1] || win3[i][1]>=win3[i][2] || win3[i][2]>=dp->n) return false;
        for (int j=0;j<i;j++) if (memcmp(win3[i],win3[j],3)==0) return false;
    }
    c.dp=dp; c.cut=cut; c.early=early;
    for (int a=cut;a<(int)dp->n;a++) for (int b=a+1;b<(int)dp->n;b++) for (int d=b+1;d<(int)dp->n;d++) {
        bool used=false;
        for (int i=0;i<nwin;i++) if (win3[i][0]==a && win3[i][1]==b && win3[i][2]==d) { used=true; break; }
        if (!used) { c.cwin[c.ncwin][0]=(uint8_t)a; c.cwin[c.ncwin][1]=(uint8_t)b; c.cwin[c.ncwin++][2]=(uint8_t)d; }
    }
    if (c.ncwin!=158) return false;
    uint64_t unpadded=(uint64_t)dp->prefix_remainder_len+(uint64_t)(dp->n-dp->t)*SIG_PUSH_SIZE+dp->tail_section_len+dp->tx_suffix_len;
    if (unpadded+72>4096 || dp->total_preimage_len<unpadded || (dp->total_preimage_len-unpadded)%64) return false;
    c.mid_bytes=dp->total_preimage_len-unpadded;
    c.n_epochs=binom_u64(cut,early);
    if (!qsb_hv_init(&c.hv,dp,(const uint8_t (*)[QSB_SE_TWIN])win3,cut,early)) return false;
    BIGNUM *x=BN_new(), *y=BN_new(); EC_POINT *a=EC_POINT_new(c.hv.grp);
    bool ok=x && y && a && !BN_is_zero(c.hv.nri) &&
        EC_POINT_mul(c.hv.grp,a,c.hv.nri,nullptr,nullptr,c.hv.ctx) &&
        EC_POINT_get_affine_coordinates_GFp(c.hv.grp,a,x,y,c.hv.ctx);
    if (ok) { fe_from_bn(fax,x); fe_from_bn(fay,y); }
    EC_POINT_free(a); BN_free(x); BN_free(y);
    fe_from_le32(c.cx,dp->u2r_x); fe_from_le32(c.cy,dp->u2r_y);
    return ok && !fe_ge_p(c.cx.v) && !fe_ge_p(c.cy.v);
}

class Session {
    std::unique_ptr<Ctx> ctx;
    std::thread coordinator;
    bool attempted=false;
public:
    Session() = default;
    Session(const Session&)=delete;
    Session& operator=(const Session&)=delete;
    ~Session() { stop_join(); }
    void start(const digest_params_t *dp, const uint8_t win3[][3], int nwin, int cut, int early,
               bool single_gpu) noexcept {
        if (attempted) return;
        attempted=true;
        try {
            if (!single_gpu) { fprintf(stderr,"CPU co-grind: off (requires sole GPU process)\n"); return; }
            long budget=0; int nth=thread_budget(budget);
            if (nth<1) { fprintf(stderr,"CPU co-grind: off (budget=%ld reserve=%d)\n",budget,QSB_CPU_RESERVE); return; }
            ctx.reset(new Ctx()); ctx->nthreads=nth;
            fe ax,ay;
            if (!prepare(*ctx,dp,win3,nwin,cut,early,ax,ay)) { fprintf(stderr,"CPU co-grind: off (shape/window/gate)\n"); ctx.reset(); return; }
            Ctx *c=ctx.get();
            coordinator=std::thread([c,ax,ay,nth]() {
                guarded(*c,[&]() {
                    if (!idle_or_disable(*c) || stopped(*c)) return;
                    const int builders=nth<NW ? nth : NW;
                    if (!build_table(*c,ax,ay,builders) || stopped(*c)) return;
                    Threads workers(*c); workers.v.reserve(nth);
                    for (int t=0;t<nth && !stopped(*c);t++) workers.v.emplace_back([c,t]() { guarded(*c,[&]() { worker(c,t); }); });
                    workers.join();
                });
            });
            fprintf(stderr,"CPU co-grind: started optional adjunct, capped_threads=%d budget=%ld reserve=%d windows=158; counts separate from GPU\n",nth,budget,QSB_CPU_RESERVE);
        } catch (const std::exception &e) {
            fprintf(stderr,"CPU co-grind: startup disabled (%s)\n",e.what()); stop_join();
        } catch (...) { fprintf(stderr,"CPU co-grind: startup disabled (unknown exception)\n"); stop_join(); }
    }
    void stop_join() noexcept {
        if (!ctx) return;
        ctx->stop.store(true,std::memory_order_relaxed);
        if (coordinator.joinable()) coordinator.join();
        fprintf(stderr,"CPU co-grind: completed_candidates=%llu published_hits=%llu (CPU only)\n",
            (unsigned long long)ctx->cand.load(),(unsigned long long)ctx->hits.load());
        ctx.reset(); // flush/close and exact-gate cleanup after every worker has joined
    }
};
} // namespace qcpu
