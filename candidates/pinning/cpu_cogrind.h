/* cpu_cogrind.h -- host-CPU co-grinding for the pinning search.
 *
 * The GPU walks sequences upward from 0x80000000 (about 900 of them in a 1200 s run), each over
 * locktimes [LT_MIN, LT_MAX). Idle host cores grind sequences counting DOWN from 0xFFFFFFFE over
 * the same locktime range, so the two candidate sets are disjoint by construction and every CPU
 * hit is a new, distinct candidate. Each CPU-nominated hit is re-derived by the tree's unchanged
 * exact OpenSSL gate (qsb_host_exact_hit) before it is appended to results/pinning_hit_cpu.txt
 * in the seed's `sequence= locktime= recid=` format.
 *
 * Per candidate the CPU compresses the locktime block of the 75-byte suffix (the sequence block
 * is compressed once per sequence), SHA256s the digest, and recovers
 *     Q = z * B +- A,   B = neg_r_inv * G,  A = u2 * R
 * with a 16-bit fixed-window table of B multiples (16 windows x 65535 affine points, 64 MiB,
 * built at startup) and affine additions whose inversions are batched over the 1024 candidates
 * of a batch (Montgomery's trick); both recids share the denominator of the last addition. The
 * elliptic-curve stage runs 8 (AVX-512F) or 4 (AVX2) candidates per vector in libsecp256k1's 10x26
 * field (cpu_cogrind_vec.h), the faster of the two as timed at startup, else libsecp256k1's
 * scalar 5x52 field (both MIT, Pieter Wuille; notice in COPYING-secp256k1).
 *
 * Contention safety. Workers run SCHED_IDLE (fallback nice 19), so the GPU host thread always
 * preempts them. The worker count is min(affinity CPUs, cgroup CPU quota) minus a reserve; a
 * runtime check lowers it if the workers receive less CPU time than they ask for (hidden quota
 * or foreign load). Once a minute the controller pauses the workers for two 1 s windows and
 * compares the GPU batch interval with and without them; it sheds a quarter of the workers if
 * two consecutive checks show a GPU loss above 1.5% that exceeds what the CPU adds.
 * QSB_CPU_GRIND=0 removes all of it.
 */
#ifndef QSB_CPU_COGRIND_H
#define QSB_CPU_COGRIND_H
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <atomic>
#include <x86intrin.h>

namespace qcg {

/* ---------------- field: libsecp256k1 5x52 (MIT) ---------------- */
typedef unsigned __int128 u128;
struct fe { uint64_t n[5]; };

static inline void fe_mul_inner(uint64_t *r, const uint64_t *a, const uint64_t * __restrict__ b) {
    u128 c, d;
    uint64_t t3, t4, tx, u0;
    uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    const uint64_t M = 0xFFFFFFFFFFFFFULL, R = 0x1000003D10ULL;
    d  = (u128)a0 * b[3] + (u128)a1 * b[2] + (u128)a2 * b[1] + (u128)a3 * b[0];
    c  = (u128)a4 * b[4];
    d += (c & M) * R; c >>= 52;
    t3 = (uint64_t)d & M; d >>= 52;
    d += (u128)a0 * b[4] + (u128)a1 * b[3] + (u128)a2 * b[2] + (u128)a3 * b[1] + (u128)a4 * b[0];
    d += c * R;
    t4 = (uint64_t)d & M; d >>= 52;
    tx = (t4 >> 48); t4 &= (M >> 4);
    c  = (u128)a0 * b[0];
    d += (u128)a1 * b[4] + (u128)a2 * b[3] + (u128)a3 * b[2] + (u128)a4 * b[1];
    u0 = (uint64_t)d & M; d >>= 52;
    u0 = (u0 << 4) | tx;
    c += (u128)u0 * (R >> 4);
    r[0] = (uint64_t)c & M; c >>= 52;
    c += (u128)a0 * b[1] + (u128)a1 * b[0];
    d += (u128)a2 * b[4] + (u128)a3 * b[3] + (u128)a4 * b[2];
    c += (d & M) * R; d >>= 52;
    r[1] = (uint64_t)c & M; c >>= 52;
    c += (u128)a0 * b[2] + (u128)a1 * b[1] + (u128)a2 * b[0];
    d += (u128)a3 * b[4] + (u128)a4 * b[3];
    c += (d & M) * R; d >>= 52;
    r[2] = (uint64_t)c & M; c >>= 52;
    c += d * R + t3;
    r[3] = (uint64_t)c & M; c >>= 52;
    c += t4;
    r[4] = (uint64_t)c;
}

static inline void fe_sqr_inner(uint64_t *r, const uint64_t *a) {
    u128 c, d;
    uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    int64_t t3, t4, tx, u0;
    const uint64_t M = 0xFFFFFFFFFFFFFULL, R = 0x1000003D10ULL;
    d  = (u128)(a0*2) * a3 + (u128)(a1*2) * a2;
    c  = (u128)a4 * a4;
    d += (c & M) * R; c >>= 52;
    t3 = (uint64_t)d & M; d >>= 52;
    a4 *= 2;
    d += (u128)a0 * a4 + (u128)(a1*2) * a3 + (u128)a2 * a2;
    d += c * R;
    t4 = (uint64_t)d & M; d >>= 52;
    tx = (t4 >> 48); t4 &= (M >> 4);
    c  = (u128)a0 * a0;
    d += (u128)a1 * a4 + (u128)(a2*2) * a3;
    u0 = (uint64_t)d & M; d >>= 52;
    u0 = (u0 << 4) | tx;
    c += (u128)u0 * (R >> 4);
    r[0] = (uint64_t)c & M; c >>= 52;
    a0 *= 2;
    c += (u128)a0 * a1;
    d += (u128)a2 * a4 + (u128)a3 * a3;
    c += (d & M) * R; d >>= 52;
    r[1] = (uint64_t)c & M; c >>= 52;
    c += (u128)a0 * a2 + (u128)a1 * a1;
    d += (u128)a3 * a4;
    c += (d & M) * R; d >>= 52;
    r[2] = (uint64_t)c & M; c >>= 52;
    c += d * R + t3;
    r[3] = (uint64_t)c & M; c >>= 52;
    c += t4;
    r[4] = (uint64_t)c;
}

/* inputs: magnitude <= 8; output magnitude 1 */
static inline void fe_mul(fe *r, const fe *a, const fe *b) {
    if (r == b) { fe t = *b; fe_mul_inner(r->n, a->n, t.n); }
    else fe_mul_inner(r->n, a->n, b->n);
}
static inline void fe_sqr(fe *r, const fe *a) { fe_sqr_inner(r->n, a->n); }
static inline void fe_add(fe *r, const fe *a) { for (int i = 0; i < 5; i++) r->n[i] += a->n[i]; }
/* r = -a, a of magnitude <= m; result magnitude m+1 */
static inline void fe_neg(fe *r, const fe *a, int m) {
    r->n[0] = 0xFFFFEFFFFFC2FULL * 2 * (m + 1) - a->n[0];
    r->n[1] = 0xFFFFFFFFFFFFFULL * 2 * (m + 1) - a->n[1];
    r->n[2] = 0xFFFFFFFFFFFFFULL * 2 * (m + 1) - a->n[2];
    r->n[3] = 0xFFFFFFFFFFFFFULL * 2 * (m + 1) - a->n[3];
    r->n[4] = 0x0FFFFFFFFFFFFULL * 2 * (m + 1) - a->n[4];
}
static inline void fe_normalize_weak(fe *r) {
    uint64_t t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4];
    uint64_t x = t4 >> 48; t4 &= 0x0FFFFFFFFFFFFULL;
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52); t0 &= 0xFFFFFFFFFFFFFULL;
    t2 += (t1 >> 52); t1 &= 0xFFFFFFFFFFFFFULL;
    t3 += (t2 >> 52); t2 &= 0xFFFFFFFFFFFFFULL;
    t4 += (t3 >> 52); t3 &= 0xFFFFFFFFFFFFFULL;
    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4;
}
static inline void fe_normalize(fe *r) {
    uint64_t t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4];
    uint64_t m;
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
    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4;
}
static inline int fe_is_zero_norm(const fe *a) { return (a->n[0] | a->n[1] | a->n[2] | a->n[3] | a->n[4]) == 0; }
/* 4x64 little-endian words -> 5x52 */
static inline void fe_from_w(fe *r, const uint64_t *w) {
    const uint64_t M = 0xFFFFFFFFFFFFFULL;
    r->n[0] = w[0] & M;
    r->n[1] = ((w[0] >> 52) | (w[1] << 12)) & M;
    r->n[2] = ((w[1] >> 40) | (w[2] << 24)) & M;
    r->n[3] = ((w[2] >> 28) | (w[3] << 36)) & M;
    r->n[4] = w[3] >> 16;
}
/* normalized 5x52 -> 4x64 little-endian words */
static inline void fe_to_w(uint64_t *w, const fe *a) {
    w[0] = a->n[0] | (a->n[1] << 52);
    w[1] = (a->n[1] >> 12) | (a->n[2] << 40);
    w[2] = (a->n[2] >> 24) | (a->n[3] << 28);
    w[3] = (a->n[3] >> 36) | (a->n[4] << 16);
}
/* a^(p-2): libsecp256k1's addition chain */
static void fe_inv(fe *r, const fe *a) {
    fe x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t1;
    int j;
    fe_sqr(&x2, a); fe_mul(&x2, &x2, a);
    fe_sqr(&x3, &x2); fe_mul(&x3, &x3, a);
    x6 = x3; for (j = 0; j < 3; j++) fe_sqr(&x6, &x6); fe_mul(&x6, &x6, &x3);
    x9 = x6; for (j = 0; j < 3; j++) fe_sqr(&x9, &x9); fe_mul(&x9, &x9, &x3);
    x11 = x9; for (j = 0; j < 2; j++) fe_sqr(&x11, &x11); fe_mul(&x11, &x11, &x2);
    x22 = x11; for (j = 0; j < 11; j++) fe_sqr(&x22, &x22); fe_mul(&x22, &x22, &x11);
    x44 = x22; for (j = 0; j < 22; j++) fe_sqr(&x44, &x44); fe_mul(&x44, &x44, &x22);
    x88 = x44; for (j = 0; j < 44; j++) fe_sqr(&x88, &x88); fe_mul(&x88, &x88, &x44);
    x176 = x88; for (j = 0; j < 88; j++) fe_sqr(&x176, &x176); fe_mul(&x176, &x176, &x88);
    x220 = x176; for (j = 0; j < 44; j++) fe_sqr(&x220, &x220); fe_mul(&x220, &x220, &x44);
    x223 = x220; for (j = 0; j < 3; j++) fe_sqr(&x223, &x223); fe_mul(&x223, &x223, &x3);
    t1 = x223; for (j = 0; j < 23; j++) fe_sqr(&t1, &t1); fe_mul(&t1, &t1, &x22);
    for (j = 0; j < 5; j++) fe_sqr(&t1, &t1); fe_mul(&t1, &t1, a);
    for (j = 0; j < 3; j++) fe_sqr(&t1, &t1); fe_mul(&t1, &t1, &x2);
    for (j = 0; j < 2; j++) fe_sqr(&t1, &t1); fe_mul(r, a, &t1);
}

/* ---------------- configuration ---------------- */
/* Fixed-window width, chosen at startup: 20 bits (13 windows, 832 MiB table) when the host has
 * plenty of free memory, else 16 bits (16 windows, 64 MiB). */
static int g_W = 16, g_nwin = 16;
static unsigned g_tsize = 1u << 16;
#define QSB_CG_W g_W
#define QSB_CG_NWIN g_nwin
#define QSB_CG_TSIZE g_tsize                 /* entries per window, index 0 unused */
#define QSB_CG_NWIN_MAX 32
#ifndef QSB_CG_BATCH_MIN
#define QSB_CG_BATCH_MIN 1024             /* candidates per inversion batch (lower bound) */
#endif
#ifndef QSB_CG_MAXB
#define QSB_CG_MAXB 2048
#endif
#define QSB_CG_MAXW 256                   /* max worker threads */

struct tentry { uint64_t x[4], y[4]; };   /* 64 B, one cache line */

/* affine P = P + Q for distinct x; used only at table build */
static void aff_add1(fe *x1, fe *y1, const fe *x2, const fe *y2) {
    fe dx, dy, t, l, l2, x3, y3;
    fe_neg(&t, x1, 1); dx = *x2; fe_add(&dx, &t);
    fe_neg(&t, y1, 1); dy = *y2; fe_add(&dy, &t);
    fe_normalize(&dx); fe_inv(&t, &dx); fe_mul(&l, &dy, &t);
    fe_sqr(&l2, &l);
    fe_neg(&t, x1, 1); x3 = l2; fe_add(&x3, &t); fe_neg(&t, x2, 1); fe_add(&x3, &t);
    fe_normalize_weak(&x3);
    fe_neg(&t, &x3, 1); fe t2 = *x1; fe_add(&t2, &t); fe_mul(&y3, &l, &t2);
    fe_neg(&t, y1, 1); fe_add(&y3, &t); fe_normalize_weak(&y3);
    *x1 = x3; *y1 = y3;
}
/* affine doubling */
static void aff_dbl1(fe *x1, fe *y1) {
    fe t, l, x2, y2, num, den;
    fe_sqr(&num, x1); t = num; fe_add(&num, &t); fe_add(&num, &t);        /* 3x^2 (mag 3) */
    den = *y1; fe_add(&den, y1); fe_normalize(&den); fe_inv(&t, &den);  /* 1/(2y) */
    fe_mul(&l, &num, &t);
    fe_sqr(&x2, &l); fe_neg(&t, x1, 1); fe_add(&x2, &t); fe_add(&x2, &t); fe_normalize_weak(&x2);
    fe_neg(&t, &x2, 1); fe t2 = *x1; fe_add(&t2, &t); fe_mul(&y2, &l, &t2);
    fe_neg(&t, y1, 1); fe_add(&y2, &t); fe_normalize_weak(&y2);
    *x1 = x2; *y1 = y2;
}
static void tentry_set(tentry *e, fe x, fe y) { fe_normalize(&x); fe_normalize(&y); fe_to_w(e->x, &x); fe_to_w(e->y, &y); }

/* Window j of the table: e[d] = d * Bj, d = 1..2^W-1, Bj = 2^(W j) * B.
 * Row 0 (d = 1..256) sequentially, then row r = row r-1 + 256*Bj with one batched
 * inversion per row. The single doubling in that chain (d = 512 = 256 + 256) is done
 * separately. */
static void build_window(tentry *e, const fe *bx, const fe *by) {
    fe rx[256], ry[256], c[256], inv, t, mx, my;
    fe px = *bx, py = *by;
    rx[0] = px; ry[0] = py;
    for (int k = 1; k < 256; k++) {
        if (k == 1) aff_dbl1(&px, &py); else aff_add1(&px, &py, bx, by);
        rx[k] = px; ry[k] = py;
    }
    for (int k = 0; k < 256; k++) if (k + 1 < (int)QSB_CG_TSIZE) tentry_set(&e[k + 1], rx[k], ry[k]);   /* d = 1..256 */
    mx = rx[255]; my = ry[255];                                            /* 256 * Bj */
    const unsigned rows = QSB_CG_TSIZE / 256;
    for (unsigned r = 1; r < rows; r++) {
        /* new d = 256 r + k + 1 for k = 0..255; skip d >= 2^W */
        int n = 256;
        fe dx[256];
        for (int k = 0; k < n; k++) {
            if (r == 1 && k == 255) { dx[k].n[0] = 1; dx[k].n[1] = dx[k].n[2] = dx[k].n[3] = dx[k].n[4] = 0; continue; }
            fe_neg(&t, &rx[k], 1); dx[k] = mx; fe_add(&dx[k], &t);
        }
        c[0] = dx[0];
        for (int k = 1; k < n; k++) fe_mul(&c[k], &c[k - 1], &dx[k]);
        fe_normalize(&c[n - 1]); fe_inv(&inv, &c[n - 1]);
        for (int k = n - 1; k >= 0; k--) {
            fe ik;
            if (k > 0) { fe_mul(&ik, &inv, &c[k - 1]); fe_mul(&inv, &inv, &dx[k]); } else ik = inv;
            if (r == 1 && k == 255) { fe x = mx, y = my; aff_dbl1(&x, &y); rx[k] = x; ry[k] = y; continue; }
            fe dy, l, l2, x3, y3;
            fe_neg(&t, &ry[k], 1); dy = my; fe_add(&dy, &t);
            fe_mul(&l, &dy, &ik); fe_sqr(&l2, &l);
            fe_neg(&t, &rx[k], 1); x3 = l2; fe_add(&x3, &t); fe_neg(&t, &mx, 1); fe_add(&x3, &t); fe_normalize_weak(&x3);
            fe_neg(&t, &x3, 1); fe t2 = rx[k]; fe_add(&t2, &t); fe_mul(&y3, &l, &t2);
            fe_neg(&t, &ry[k], 1); fe_add(&y3, &t); fe_normalize_weak(&y3);
            rx[k] = x3; ry[k] = y3;
        }
        for (int k = 0; k < n; k++) {
            unsigned d = 256 * r + (unsigned)k + 1;
            if (d < QSB_CG_TSIZE) tentry_set(&e[d], rx[k], ry[k]);
        }
    }
}

/* ---------------- shared state ---------------- */
struct shared_t {
    const pinning2_params_t *pp;
    uint32_t lt_min, lt_range, chunks_per_seq;
    int nblk, cache_first;                /* suffix blocks; block 0 holds seq only -> once per sequence */
    uint64_t n_chunks;
    tentry *table;                        /* QSB_CG_NWIN * QSB_CG_TSIZE */
    fe ax, ay;                            /* A = u2 R (recid 0) */
    uint8_t nri[32];                      /* neg_r_inv, little-endian */
    int table_started;
    int hit_fd;
    std::atomic<uint64_t> next_chunk;
    std::atomic<uint64_t> cand_done;
    std::atomic<uint64_t> hits;
    std::atomic<uint64_t> tentative;
    std::atomic<int> allowed;             /* workers with id < allowed may run */
    std::atomic<int> stop;
    std::atomic<int> running;             /* workers currently inside a batch */
    std::atomic<int> ready;               /* table built */
    std::atomic<int> failed;
    int nworkers;
    int simd_ok;                          /* bit 4: AVX2 usable, bit 8: AVX-512F usable */
    int simd_env;                         /* QSB_CPU_GRIND_SIMD override (0/4/8), else -1 */
    std::atomic<int> simd;                /* chosen path: 8, 4 or 0 (scalar); -1 until chosen */
    std::atomic<uint64_t> busy_ns[QSB_CG_MAXW];   /* per-worker thread CPU time */
    std::atomic<uint64_t> sha_cyc, ec_cyc;
};
static shared_t *g_cg = NULL;
static int g_ctl_verbose = 0;

static inline uint64_t thread_cpu_ns() {
    struct timespec ts; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static inline double mono_s() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline void be_store32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16); p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v; }

static inline int lz_ok(const uint32_t *h) {
    for (int i = 0; i < QSB_ZEROS_N / 32; i++) if (h[i]) return 0;
#if (QSB_ZEROS_N % 32) != 0
    if (h[QSB_ZEROS_N / 32] >> (32 - (QSB_ZEROS_N % 32))) return 0;
#endif
    return 1;
}

struct worker_t {
    int id;
    uint64_t next_k;                            /* this worker's next epoch / chunk index */
    /* batch state */
    int n;
    uint32_t seq, lt[QSB_CG_MAXB];
    uint64_t z[QSB_CG_MAXB][4];                 /* little-endian 64-bit words */
    fe x[QSB_CG_MAXB], y[QSB_CG_MAXB], c[QSB_CG_MAXB], dx[QSB_CG_MAXB];
    uint8_t inf[QSB_CG_MAXB];
    const tentry *te[QSB_CG_MAXB];
    int lst[QSB_CG_MAXB];
    uint8_t msg[128];
    uint64_t cur_seq_tag;                       /* sequence whose block-0 midstate is cached, +1 */
    uint32_t mid1[8];
    EC_GROUP *grp; BN_CTX *ctx; BIGNUM *order, *nri, *rx, *ry; EC_POINT *Ru2;
};

/* 0 scalar, 1 AVX2, 2 AVX-512F, 3 while the startup timing runs */
static inline int simd_code() { const int m = g_cg->simd.load(std::memory_order_relaxed); return m < 0 ? 3 : m == 8 ? 2 : m == 4 ? 1 : 0; }

static const uint32_t SHA_IV[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};

static inline void sha_blocks(uint32_t st[8], const uint8_t *p, size_t nblk) {
    SHA256_CTX c; memcpy(c.h, st, 32);
    for (size_t i = 0; i < nblk; i++) SHA256_Transform(&c, p + 64 * i);
    memcpy(st, c.h, 32);
}

/* digit j (bits W j .. W j + W - 1) of the 256-bit little-endian word array */
static inline unsigned digit(const uint64_t *z, int j) {
    const unsigned b = (unsigned)(QSB_CG_W * j);
    const unsigned w = b >> 6, s = b & 63;
    uint64_t v = z[w] >> s;
    if (s + QSB_CG_W > 64 && w < 3) v |= z[w + 1] << (64 - s);
    return (unsigned)(v & (QSB_CG_TSIZE - 1));
}

/* A tentative CPU hit: re-derive (sequence, locktime, recid) with the exact
 * OpenSSL gate, append it to the CPU hit file only if it passes. */
static void publish(worker_t *w, int i, int recid) {
    shared_t *S = g_cg;
    S->tentative.fetch_add(1, std::memory_order_relaxed);
    if (qsb_host_exact_hit(S->pp, w->seq, w->lt[i], recid, w->grp, w->ctx, w->order, w->nri, w->Ru2)) {
        char line[96];
        int wl = snprintf(line, sizeof line, "sequence=%u locktime=%u recid=%d\n", w->seq, w->lt[i], recid);
        if (write(S->hit_fd, line, (size_t)wl) == wl) S->hits.fetch_add(1, std::memory_order_relaxed);
    }
}

/* Montgomery batch inversion over QSB_CG_G interleaved chains (element i is in chain i % G),
 * so the prefix-product and back-substitution multiplies of neighbouring elements are
 * independent and overlap in the pipeline (a 5x52 multiply's latency is ~3.5x its issue
 * cost). acc[g]: product of chain g's active denominators; on return inv[g] = 1/acc[g]. */
#ifndef QSB_CG_G
#define QSB_CG_G 4
#endif
static inline void fe_one(fe *r) { r->n[0] = 1; r->n[1] = r->n[2] = r->n[3] = r->n[4] = 0; }
static void chain_invert(fe inv[QSB_CG_G], const fe acc[QSB_CG_G]) {
    fe pre[QSB_CG_G], tot, it;
    pre[0] = acc[0];
    for (int g = 1; g < QSB_CG_G; g++) fe_mul(&pre[g], &pre[g - 1], &acc[g]);
    tot = pre[QSB_CG_G - 1]; fe_normalize(&tot);
    fe_inv(&it, &tot);
    for (int g = QSB_CG_G - 1; g > 0; g--) { fe_mul(&inv[g], &it, &pre[g - 1]); fe_mul(&it, &it, &acc[g]); }
    inv[0] = it;
}

/* Batch EC: P_i = z_i * B, then Q_i = P_i +- A, then pubkey hash gate. Hits -> exact gate -> file.
 * The active elements of each step are compacted into a list; list position k belongs to chain
 * k % G, and the back-substitution runs G consecutive positions (G distinct chains) in lockstep so
 * their serial multiply chains interleave. */
static void ec_batch(worker_t *w) {
    shared_t *S = g_cg;
    const int n = w->n;
    const int G = QSB_CG_G;
    const tentry *T = S->table;
    fe t, acc[QSB_CG_G], inv[QSB_CG_G];
    int *L = w->lst;
    for (int i = 0; i < n; i++) {
        unsigned d = digit(w->z[i], 0);
        if (d) { fe_from_w(&w->x[i], T[d].x); fe_from_w(&w->y[i], T[d].y); w->inf[i] = 0; }
        else w->inf[i] = 1;
        __builtin_prefetch(&T[(size_t)QSB_CG_TSIZE + digit(w->z[i], 1)]);
    }
    for (int j = 1; j < QSB_CG_NWIN; j++) {
        const tentry *Tj = T + (size_t)j * QSB_CG_TSIZE;
        for (int g = 0; g < G; g++) fe_one(&acc[g]);
        /* pass 1: denominators + per-chain prefix products over the active list */
        int m = 0;
        for (int i = 0; i < n; i++) {
            unsigned d = digit(w->z[i], j);
            if (j + 1 < QSB_CG_NWIN) __builtin_prefetch(&T[(size_t)(j + 1) * QSB_CG_TSIZE + digit(w->z[i], j + 1)]);
            if (!d) continue;
            const tentry *e = &Tj[d];
            if (w->inf[i]) { fe_from_w(&w->x[i], e->x); fe_from_w(&w->y[i], e->y); w->inf[i] = 0; continue; }
            const int g = m % G;
            L[m] = i; w->te[m] = e;
            fe ex; fe_from_w(&ex, e->x);
            fe_neg(&t, &w->x[i], 1); w->dx[m] = ex; fe_add(&w->dx[m], &t);      /* mag 3 */
            fe_mul(&acc[g], &acc[g], &w->dx[m]);
            w->c[m] = acc[g];
            m++;
        }
        chain_invert(inv, acc);
        /* pass 2: backwards in blocks of G positions (distinct chains) */
        for (int k0 = m - 1; k0 >= 0; k0 -= G) {
            const int nb = k0 + 1 < G ? k0 + 1 : G;
            fe ik[QSB_CG_G], l[QSB_CG_G], l2[QSB_CG_G], x3[QSB_CG_G], y3[QSB_CG_G], dy[QSB_CG_G], t2[QSB_CG_G];
            for (int u = 0; u < nb; u++) {
                const int k = k0 - u, g = k % G;
                if (k >= G) fe_mul(&ik[u], &inv[g], &w->c[k - G]); else ik[u] = inv[g];
            }
            for (int u = 0; u < nb; u++) { const int k = k0 - u, g = k % G; fe_mul(&inv[g], &inv[g], &w->dx[k]); }
            for (int u = 0; u < nb; u++) {
                const int i = L[k0 - u]; fe ey; fe_from_w(&ey, w->te[k0 - u]->y);
                fe_neg(&t, &w->y[i], 1); dy[u] = ey; fe_add(&dy[u], &t);
            }
            for (int u = 0; u < nb; u++) fe_mul(&l[u], &dy[u], &ik[u]);
            for (int u = 0; u < nb; u++) fe_sqr(&l2[u], &l[u]);
            for (int u = 0; u < nb; u++) {
                const int i = L[k0 - u]; fe ex; fe_from_w(&ex, w->te[k0 - u]->x);
                fe_neg(&t, &w->x[i], 1); x3[u] = l2[u]; fe_add(&x3[u], &t); fe_neg(&t, &ex, 1); fe_add(&x3[u], &t); fe_normalize_weak(&x3[u]);
                fe_neg(&t, &x3[u], 1); t2[u] = w->x[i]; fe_add(&t2[u], &t);
            }
            for (int u = 0; u < nb; u++) fe_mul(&y3[u], &l[u], &t2[u]);
            for (int u = 0; u < nb; u++) {
                const int i = L[k0 - u];
                fe_neg(&t, &w->y[i], 1); fe_add(&y3[u], &t); fe_normalize_weak(&y3[u]);
                w->x[i] = x3[u]; w->y[i] = y3[u];
            }
        }
    }
    /* final: Q0 = P + A, Q1 = P - A; shared denominator ax - px */
    {
        for (int g = 0; g < G; g++) fe_one(&acc[g]);
        int m = 0;
        for (int i = 0; i < n; i++) {
            if (w->inf[i]) continue;
            const int g = m % G;
            L[m] = i;
            fe_neg(&t, &w->x[i], 1); w->dx[m] = S->ax; fe_add(&w->dx[m], &t);
            fe_mul(&acc[g], &acc[g], &w->dx[m]);
            w->c[m] = acc[g];
            m++;
        }
        chain_invert(inv, acc);
        fe ikv[QSB_CG_MAXB];
        for (int k0 = m - 1; k0 >= 0; k0 -= G) {
            const int nb = k0 + 1 < G ? k0 + 1 : G;
            for (int u = 0; u < nb; u++) {
                const int k = k0 - u, g = k % G;
                if (k >= G) fe_mul(&ikv[k], &inv[g], &w->c[k - G]); else ikv[k] = inv[g];
            }
            for (int u = 0; u < nb; u++) { const int k = k0 - u, g = k % G; fe_mul(&inv[g], &inv[g], &w->dx[k]); }
        }
        /* both recids of G candidates in lockstep */
        for (int k0 = 0; k0 < m; k0 += G) {
            const int nb = m - k0 < G ? m - k0 : G;
            fe l[2 * QSB_CG_G], l2[2 * QSB_CG_G], qx[2 * QSB_CG_G], qy[2 * QSB_CG_G], t2[2 * QSB_CG_G], dy[2 * QSB_CG_G], ny[QSB_CG_G], sx[QSB_CG_G];
            for (int u = 0; u < nb; u++) {
                const int i = L[k0 + u];
                fe na; fe_neg(&ny[u], &w->y[i], 1);
                fe_neg(&sx[u], &w->x[i], 1); fe_neg(&na, &S->ax, 1); fe_add(&sx[u], &na);      /* -(px + ax), mag 4 */
                dy[2 * u] = S->ay; fe_add(&dy[2 * u], &ny[u]);                                 /* ay - py */
                fe_neg(&dy[2 * u + 1], &S->ay, 1); fe_add(&dy[2 * u + 1], &ny[u]);             /* -ay - py */
            }
            for (int v = 0; v < 2 * nb; v++) fe_mul(&l[v], &dy[v], &ikv[k0 + v / 2]);
            for (int v = 0; v < 2 * nb; v++) fe_sqr(&l2[v], &l[v]);
            for (int v = 0; v < 2 * nb; v++) {
                const int i = L[k0 + v / 2];
                qx[v] = l2[v]; fe_add(&qx[v], &sx[v / 2]); fe_normalize(&qx[v]);
                fe_neg(&t, &qx[v], 1); t2[v] = w->x[i]; fe_add(&t2[v], &t);
            }
            for (int v = 0; v < 2 * nb; v++) fe_mul(&qy[v], &l[v], &t2[v]);
            for (int v = 0; v < 2 * nb; v++) {
                const int i = L[k0 + v / 2], recid = v & 1;
                fe_add(&qy[v], &ny[v / 2]); fe_normalize(&qy[v]);
                uint64_t xw[4]; fe_to_w(xw, &qx[v]);
                uint8_t blk[64];
                blk[0] = (uint8_t)(0x02 | (qy[v].n[0] & 1));
                for (int b = 0; b < 32; b++) blk[1 + b] = (uint8_t)(xw[3 - b / 8] >> (56 - 8 * (b % 8)));
                blk[33] = 0x80; memset(blk + 34, 0, 30); blk[62] = 0x01; blk[63] = 0x08;   /* 264 bits */
                uint32_t h[8]; memcpy(h, SHA_IV, 32); sha_blocks(h, blk, 1);
                if (lz_ok(h)) publish(w, i, recid);
            }
        }
    }
}

/* SIMD elliptic-curve stage: 4 lanes (AVX2) and 8 lanes (AVX-512F), chosen at runtime. */
#if defined(__x86_64__) && !defined(QSB_CG_NO_SIMD)
#include <immintrin.h>
#define QCG_NS v4
#define QCG_VW 4
#define QCG_TARGET "avx2"
#include "cpu_cogrind_vec.h"
#undef QCG_NS
#undef QCG_VW
#undef QCG_TARGET
#define QCG_NS v8
#define QCG_VW 8
#define QCG_TARGET "avx512f"
#include "cpu_cogrind_vec.h"
#undef QCG_NS
#undef QCG_VW
#undef QCG_TARGET
#define QSB_CG_HAVE_SIMD 1
#else
#define QSB_CG_HAVE_SIMD 0
#endif

/* Fill the batch with one chunk of locktimes of one CPU sequence: SHA256d per candidate. */
static int fill_batch(worker_t *w, uint8_t *scratch) {
    shared_t *S = g_cg;
    const pinning2_params_t *pp = S->pp;
    (void)scratch;
    /* Worker w walks its own sequences 0xFFFFFFFE - (code + 256 r), r = 0, 1, ..., with
     * code = 64 * path + w (path 0 scalar, 1 AVX2, 2 AVX-512F, 3 = the startup timing batches),
     * locktimes rising from LT_MIN. Every published CPU hit therefore names the worker and EC
     * path that found it, and its locktime shows how far that worker had got. */
    const uint32_t code = (uint32_t)(64 * simd_code() + (w->id & 63));
    const uint64_t k = w->next_k++;
    const uint64_t r = k / S->chunks_per_seq;
    if (r >= 0x100000ull) return 0;
    const uint32_t seq = 0xFFFFFFFEu - (code + 256u * (uint32_t)r);
    const uint32_t off = (uint32_t)(k % S->chunks_per_seq) * (uint32_t)QSB_CG_BATCH_MIN;
    int n = QSB_CG_BATCH_MIN;
    if (off + (uint32_t)n > S->lt_range) n = (int)(S->lt_range - off);
    const uint32_t lt0 = S->lt_min + off;
    uint8_t *m = w->msg;
    const uint32_t sl = pp->suffix_len, so = pp->seq_offset, lo = pp->lt_offset;
    if (w->cur_seq_tag != (uint64_t)seq + 1) {
        memset(m, 0, 128);
        memcpy(m, pp->suffix, sl);
        for (int b = 0; b < 4; b++) m[so + b] = (uint8_t)(seq >> (8 * b));
        m[sl] = 0x80;
        const uint64_t bits = (uint64_t)pp->total_preimage_len * 8;
        const int lenoff = S->nblk * 64 - 8;
        for (int b = 0; b < 8; b++) m[lenoff + 7 - b] = (uint8_t)(bits >> (8 * b));
        memcpy(w->mid1, pp->midstate, 32);
        if (S->cache_first) sha_blocks(w->mid1, m, 1);
        w->cur_seq_tag = (uint64_t)seq + 1;
    }
    w->seq = seq;
    for (int i = 0; i < n; i++) {
        const uint32_t lt = lt0 + (uint32_t)i;
        for (int b = 0; b < 4; b++) m[lo + b] = (uint8_t)(lt >> (8 * b));
        uint32_t st[8]; memcpy(st, w->mid1, 32);
        if (S->cache_first) sha_blocks(st, m + 64, 1); else sha_blocks(st, m, (size_t)S->nblk);
        uint8_t d1[64];
        for (int b = 0; b < 8; b++) be_store32(d1 + 4 * b, st[b]);
        d1[32] = 0x80; memset(d1 + 33, 0, 29); d1[62] = 0x01; d1[63] = 0x00;   /* 256 bits */
        uint32_t h2[8]; memcpy(h2, SHA_IV, 32); sha_blocks(h2, d1, 1);
        w->z[i][0] = ((uint64_t)h2[6] << 32) | h2[7];
        w->z[i][1] = ((uint64_t)h2[4] << 32) | h2[5];
        w->z[i][2] = ((uint64_t)h2[2] << 32) | h2[3];
        w->z[i][3] = ((uint64_t)h2[0] << 32) | h2[1];
        w->lt[i] = lt;
    }
    w->n = n;
    return n;
}

/* ---------------- core plan (light profile) ---------------- */
#ifndef QSB_CG_WMAX
#define QSB_CG_WMAX 64
#endif
static int g_ncores_w = -1;                 /* usable worker cores; -1 = topology unknown */
static int g_core_cpu[QSB_CG_WMAX > 0 ? QSB_CG_WMAX : 1];
static cpu_set_t g_host_set;
static int g_host_ok = 0;
static int read_int_file(const char *p, long *v) {
    FILE *f = fopen(p, "r"); if (!f) return 0;
    const int ok = fscanf(f, "%ld", v) == 1; fclose(f); return ok;
}
/* One logical CPU per physical core for the workers: cores ordered by maximum frequency
 * (performance cores first on hybrid parts), then from the highest CPU number down; the core the
 * calling (GPU host) thread runs on is never used. The host thread's own mask becomes the allowed
 * CPUs minus every hyperthread of the chosen cores (only if >= 4 CPUs remain). */
static void plan_cores() {
    cpu_set_t aff; CPU_ZERO(&aff);
    if (sched_getaffinity(0, sizeof aff, &aff) != 0) return;
    const int hcpu = sched_getcpu();
    enum { MAXC = 1024 };
    static long pkg[MAXC], core[MAXC], fmax[MAXC];
    int cpus[MAXC], n = 0;
    for (int c = 0; c < MAXC && c < CPU_SETSIZE; c++) {
        if (!CPU_ISSET(c, &aff)) continue;
        char p[160];
        snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/core_id", c);
        if (!read_int_file(p, &core[c])) return;
        snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", c);
        if (!read_int_file(p, &pkg[c])) pkg[c] = 0;
        snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c);
        if (!read_int_file(p, &fmax[c])) fmax[c] = 0;
        cpus[n++] = c;
    }
    if (n == 0 || hcpu < 0 || hcpu >= MAXC || !CPU_ISSET(hcpu, &aff)) return;
    /* candidates: highest max frequency first, then highest CPU number */
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) {
        const int a = cpus[i], b = cpus[j];
        if (fmax[b] > fmax[a] || (fmax[b] == fmax[a] && b > a)) { cpus[i] = b; cpus[j] = a; }
    }
    int k = 0;
    for (int i = 0; i < n && k < QSB_CG_WMAX; i++) {
        const int c = cpus[i];
        if (pkg[c] == pkg[hcpu] && core[c] == core[hcpu]) continue;      /* the host thread's core */
        int dup = 0;
        for (int j = 0; j < k; j++) if (pkg[g_core_cpu[j]] == pkg[c] && core[g_core_cpu[j]] == core[c]) dup = 1;
        if (!dup) g_core_cpu[k++] = c;
    }
    g_ncores_w = k;
    CPU_ZERO(&g_host_set); int left = 0;
    for (int i = 0; i < n; i++) {
        const int c = cpus[i]; int used = 0;
        for (int j = 0; j < k; j++) if (pkg[g_core_cpu[j]] == pkg[c] && core[g_core_cpu[j]] == core[c]) used = 1;
        if (!used) { CPU_SET(c, &g_host_set); left++; }
    }
    g_host_ok = left >= 4;
}
static void pin_worker(int id) {
    if (g_ncores_w <= 0) return;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(g_core_cpu[id % g_ncores_w], &cs);
    pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
}

static void set_idle_priority() {
    struct sched_param sp; memset(&sp, 0, sizeof sp);
    if (pthread_setschedparam(pthread_self(), SCHED_IDLE, &sp) != 0)
        setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 19);
}

static void *worker_main(void *arg) {
    shared_t *S = g_cg;
    const int id = (int)(intptr_t)arg;
    set_idle_priority();
    pin_worker(id);
    worker_t *w = (worker_t *)aligned_alloc(64, (sizeof(worker_t) + 63) & ~(size_t)63);
    uint8_t *scratch = NULL;
    if (!w) return NULL;
    w->id = id; w->next_k = 0; w->cur_seq_tag = 0;
    w->grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    w->ctx = BN_CTX_new(); w->order = BN_new(); w->nri = BN_new(); w->rx = BN_new(); w->ry = BN_new();
    w->Ru2 = w->grp ? EC_POINT_new(w->grp) : NULL;
    if (!w->grp || !w->ctx || !w->order || !w->nri || !w->rx || !w->ry || !w->Ru2 ||
        !EC_GROUP_get_order(w->grp, w->order, w->ctx) ||
        !BN_lebin2bn(S->pp->neg_r_inv, 32, w->nri) || !BN_lebin2bn(S->pp->u2r_x, 32, w->rx) ||
        !BN_lebin2bn(S->pp->u2r_y, 32, w->ry) ||
        !EC_POINT_set_affine_coordinates_GFp(w->grp, w->Ru2, w->rx, w->ry, w->ctx)) { S->failed.store(1); return NULL; }
#if QSB_CG_HAVE_SIMD
    v4::vstate *vs4 = NULL; v8::vstate *vs8 = NULL;
    if (S->simd_ok & 4) vs4 = (v4::vstate *)aligned_alloc(64, (sizeof(v4::vstate) + 63) & ~(size_t)63);
    if (S->simd_ok & 8) vs8 = (v8::vstate *)aligned_alloc(64, (sizeof(v8::vstate) + 63) & ~(size_t)63);
#endif
    while (!S->ready.load(std::memory_order_acquire)) { if (S->stop.load() || S->failed.load()) return NULL; usleep(2000); }
#if QSB_CG_HAVE_SIMD
    /* Worker 0 picks the fastest available EC path on this CPU (a few timed batches each; the
     * candidates are real work and are counted); the others wait for the choice. */
    if (id == 0 && S->simd < 0) {
        int best = S->simd_env >= 0 ? S->simd_env : 0; double bt = 1e30;
        const int cand[3] = {8, 4, 0};
        for (int c = 0; c < 3 && S->simd_env < 0; c++) {
            const int m = cand[c];
            if (m && !(S->simd_ok & m)) continue;
            if (!m && S->simd_ok) continue;                        /* scalar only without SIMD */
            double t = 0;
            for (int rep = 0; rep < 3; rep++) {
                const int n = fill_batch(w, scratch);
                if (!n) break;
                const uint64_t r0 = __rdtsc();
                if (m == 8) v8::ec_batch_vec(w, vs8); else if (m == 4) v4::ec_batch_vec(w, vs4); else ec_batch(w);
                const double dt = (double)(__rdtsc() - r0) / n;
                if (rep > 0) t += dt;                                 /* first batch warms caches */
                S->cand_done.fetch_add((uint64_t)n, std::memory_order_relaxed);
            }
            if (t > 0 && t < bt) { bt = t; best = m; }
        }
        S->simd = best;
        if (g_ctl_verbose) printf("  [CPU] EC path: %s\n", best == 8 ? "avx512f x8" : best == 4 ? "avx2 x4" : "scalar");
    }
    while (S->simd < 0) { if (S->stop.load()) return NULL; usleep(1000); }
#endif
    while (!S->stop.load(std::memory_order_relaxed)) {
        if (id >= S->allowed.load(std::memory_order_relaxed)) { usleep(5000); continue; }
        S->running.fetch_add(1);
        const uint64_t c0 = thread_cpu_ns();
        const uint64_t r0 = __rdtsc();
        int n = fill_batch(w, scratch);
        const uint64_t r1 = __rdtsc();
#if QSB_CG_HAVE_SIMD
        if (n && S->simd == 8) v8::ec_batch_vec(w, vs8); else
        if (n && S->simd == 4) v4::ec_batch_vec(w, vs4); else
#endif
        if (n) ec_batch(w);
        const uint64_t r2 = __rdtsc();
        S->sha_cyc.fetch_add(r1 - r0, std::memory_order_relaxed); S->ec_cyc.fetch_add(r2 - r1, std::memory_order_relaxed);
        S->busy_ns[id].fetch_add(thread_cpu_ns() - c0, std::memory_order_relaxed);
        S->running.fetch_sub(1);
        if (!n) break;
        S->cand_done.fetch_add((uint64_t)n, std::memory_order_relaxed);
    }
    return NULL;
}

/* table-builder thread: windows are split across builders */
struct build_arg { int j0, j1; fe bx[QSB_CG_NWIN_MAX], by[QSB_CG_NWIN_MAX]; };
static void *builder_main(void *arg) {
    build_arg *a = (build_arg *)arg;
    set_idle_priority();
    pin_worker(0);
    for (int j = a->j0; j < a->j1; j++) build_window(g_cg->table + (size_t)j * QSB_CG_TSIZE, &a->bx[j], &a->by[j]);
    return NULL;
}

/* ---------------- CPU budget ---------------- */
static double cgroup_quota_cpus() {
    FILE *f = fopen("/sys/fs/cgroup/cpu.max", "r");
    if (f) {
        char q[64] = {0}; long long per = 0;
        int ok = fscanf(f, "%63s %lld", q, &per); fclose(f);
        if (ok == 2 && strcmp(q, "max") != 0 && per > 0) return (double)atoll(q) / (double)per;
        if (ok >= 1) return -1.0;
    }
    f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r");
    if (!f) f = fopen("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us", "r");
    if (f) {
        long long q = -1; int ok = fscanf(f, "%lld", &q); fclose(f);
        FILE *g = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r");
        if (!g) g = fopen("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us", "r");
        long long p = 100000; if (g) { if (fscanf(g, "%lld", &p) != 1) p = 100000; fclose(g); }
        if (ok == 1 && q > 0 && p > 0) return (double)q / (double)p;
    }
    return -1.0;
}

/* Bytes of memory this process may still use: MemAvailable, capped by a cgroup memory limit. */
static double mem_headroom() {
    double avail = -1;
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) { char k[64]; long long v; char u[16];
        while (fscanf(f, "%63s %lld %15s", k, &v, u) >= 2) if (!strcmp(k, "MemAvailable:")) { avail = (double)v * 1024; break; }
        fclose(f); }
    const char *lim[][2] = {{"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"},
                            {"/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"}};
    for (int i = 0; i < 2; i++) {
        FILE *a = fopen(lim[i][0], "r"); if (!a) continue;
        char buf[64] = {0}; int ok = fscanf(a, "%63s", buf) == 1; fclose(a);
        if (!ok || !strcmp(buf, "max")) break;
        double l = atof(buf), u = 0;
        FILE *b = fopen(lim[i][1], "r"); if (b) { if (fscanf(b, "%lf", &u) != 1) u = 0; fclose(b); }
        if (l > 0 && l < 1e17) { const double h = l - u; if (avail < 0 || h < avail) avail = h; }
        break;
    }
    return avail;
}
static void choose_width() {
    int w = 16;
    const double big = 13.0 * (1 << 20) * 64;                   /* W = 20: 832 MiB */
    const double room = mem_headroom();
    /* The 832 MiB table is used only with a very wide margin: 16x the table in free memory
     * (MemAvailable, capped by a visible cgroup memory limit) and room under RLIMIT_AS and
     * RLIMIT_DATA. Otherwise the 64 MiB table: an unseen memory limit must never cost the run. */
    int ok = room > 16 * big;
    {
        double vm = 0; FILE *f = fopen("/proc/self/statm", "r");
        if (f) { double pages; if (fscanf(f, "%lf", &pages) == 1) vm = pages * (double)sysconf(_SC_PAGESIZE); fclose(f); }
        struct rlimit rl;
        if (getrlimit(RLIMIT_AS, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY && (double)rl.rlim_cur < vm + 4 * big) ok = 0;
        if (getrlimit(RLIMIT_DATA, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY && (double)rl.rlim_cur < vm + 4 * big) ok = 0;
    }
    (void)ok;                          /* light profile: always the 64 MiB, 16-bit table */
    if (getenv("QSB_CPU_GRIND_W")) w = atoi(getenv("QSB_CPU_GRIND_W"));
    if (w < 9 || w > 22) w = 16;
    g_W = w; g_nwin = (256 + w - 1) / w; g_tsize = 1u << w;
    if (g_ctl_verbose) printf("  CPU co-grind: %d-bit windows (%d windows, %.0f MiB table; %.1f GiB free)\n", g_W, g_nwin,
           (double)g_nwin * g_tsize * 64 / 1048576.0, room / 1073741824.0);
}

/* ---------------- controller (called from the GPU host loop) ---------------- */
struct ctl_t {
    int wmax, cur, ceiling;
    int phase;              /* 0 before the table is ready, 1 ramping/steady */
    double t_period, hold_until, t_full;
    uint64_t drains, starved;            /* current evaluation period */
    uint64_t base_drains, base_starved;  /* before any worker ran */
    double base_rate;
    int share_done;
    double last_done;
    /* A/B measurement */
    int ab_left;
    double ab_on_sum; int ab_on_n;
    double ab_off_sum; int ab_off_n;
    double next_ab;
    uint64_t busy0; double busy_t0;
    uint64_t cand0; double cand_t0;
    double win_t0; int win_n, win_skip, strikes, ab_level;
    int verbose;
};
static ctl_t g_ctl;

static uint64_t busy_total() { uint64_t s = 0; for (int i = 0; i < g_cg->nworkers; i++) s += g_cg->busy_ns[i].load(std::memory_order_relaxed); return s; }

/* Table construction, started by the first qcg::tick() -- i.e. only once the GPU pipeline is
 * running, so the host memory it takes and the CPU time of the builders never compete with the
 * GPU's own startup, and an allocation failure just leaves the co-grinder off. */
static build_arg g_ba[QSB_CG_NWIN_MAX];
static void *table_main(void *) {
    shared_t *S = g_cg;
    set_idle_priority();
    choose_width();
    S->table = (tentry *)aligned_alloc(2u << 20, (size_t)QSB_CG_NWIN * QSB_CG_TSIZE * sizeof(tentry));
    if (!S->table && g_W > 16) { g_W = 16; g_nwin = 16; g_tsize = 1u << 16;
        S->table = (tentry *)aligned_alloc(2u << 20, (size_t)QSB_CG_NWIN * QSB_CG_TSIZE * sizeof(tentry)); }
    if (!S->table) { S->failed.store(1); return NULL; }
    madvise(S->table, (size_t)QSB_CG_NWIN * QSB_CG_TSIZE * sizeof(tentry), MADV_HUGEPAGE);
    for (int j = 0; j < QSB_CG_NWIN; j++) memset(&S->table[(size_t)j * QSB_CG_TSIZE], 0, sizeof(tentry));
    /* window bases Bj = 2^(W j) * neg_r_inv * G, via OpenSSL */
    fe bx[QSB_CG_NWIN_MAX], by[QSB_CG_NWIN_MAX];
    {
        EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
        BN_CTX *ctx = BN_CTX_new();
        BIGNUM *order = BN_new(), *k = BN_new(), *x = BN_new(), *y = BN_new();
        EC_POINT *P = EC_POINT_new(grp);
        EC_GROUP_get_order(grp, order, ctx);
        BN_lebin2bn(S->nri, 32, k);
        for (int j = 0; j < QSB_CG_NWIN; j++) {
            EC_POINT_mul(grp, P, k, NULL, NULL, ctx);
            EC_POINT_get_affine_coordinates(grp, P, x, y, ctx);
            uint8_t xb[32], yb[32];
            BN_bn2lebinpad(x, xb, 32); BN_bn2lebinpad(y, yb, 32);
            uint64_t xw[4], yw[4]; memcpy(xw, xb, 32); memcpy(yw, yb, 32);
            fe_from_w(&bx[j], xw); fe_from_w(&by[j], yw);
            for (int s = 0; s < QSB_CG_W; s++) BN_mod_lshift1(k, k, order, ctx);
        }
        BN_free(order); BN_free(k); BN_free(x); BN_free(y); EC_POINT_free(P); BN_CTX_free(ctx); EC_GROUP_free(grp);
    }
    /* builders write the table window by window, so its pages are touched gradually */
    int nb = S->nworkers / 2 < QSB_CG_NWIN ? S->nworkers / 2 : QSB_CG_NWIN; if (nb > 8) nb = 8; if (nb < 1) nb = 1;
    pthread_t t[8];
    for (int b = 0; b < nb; b++) {
        g_ba[b].j0 = QSB_CG_NWIN * b / nb; g_ba[b].j1 = QSB_CG_NWIN * (b + 1) / nb;
        memcpy(g_ba[b].bx, bx, sizeof bx); memcpy(g_ba[b].by, by, sizeof by);
    }
    int started = 0;
    for (int b = 0; b < nb; b++) if (pthread_create(&t[b], NULL, builder_main, &g_ba[b]) == 0) started++; else { builder_main(&g_ba[b]); }
    for (int b = 0; b < nb; b++) if (b < started) pthread_join(t[b], NULL);
    S->ready.store(1, std::memory_order_release);
    return NULL;
}

/* Start: build the table in the background and spawn the (paused) workers. */
static int start(const pinning2_params_t *pp, uint32_t lt_min, uint32_t lt_max) {
#ifndef QSB_CPU_GRIND
#define QSB_CPU_GRIND 1
#endif
    if (!QSB_CPU_GRIND) return 0;
    const char *env = getenv("QSB_CPU_GRIND_THREADS");
    int ncpu = 0;
    { cpu_set_t cs; CPU_ZERO(&cs); if (sched_getaffinity(0, sizeof cs, &cs) == 0) ncpu = CPU_COUNT(&cs); }
    if (ncpu <= 0) ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    double quota = cgroup_quota_cpus();
    /* Without a CPU quota, SCHED_IDLE workers yield every CPU the GPU host thread wants, so all
     * CPUs but one (that thread's) are used: measured GPU loss within noise (<0.2%). Under a
     * cgroup quota SCHED_IDLE does not help -- throttling stops the GPU thread too -- so the
     * workers leave ~1.5 CPUs of the quota free (workers = ceil(quota) - 2): at workers = quota
     * the GPU lost ~1.6%, at ceil(quota) - 2 nothing measurable. */
    int nw;
    if (quota > 0 && quota < ncpu) nw = (int)ceil(quota) - 2;
    else nw = ncpu - 1;
    if (nw < 0) nw = 0;
    /* Light profile: at most QSB_CG_WMAX workers, one per physical core, never on the core the
     * GPU host thread starts on; the host thread is then kept off the workers' cores. */
    plan_cores();
    if (nw > QSB_CG_WMAX) nw = QSB_CG_WMAX;
    if (g_ncores_w >= 0 && nw > g_ncores_w) nw = g_ncores_w;
    if (env) nw = atoi(env);
    if (nw > 64) nw = 64;                  /* worker id is encoded in 6 bits of the search position */
    if (g_ncores_w > 0) {
        printf("  CPU co-grind: worker cores");
        for (int i = 0; i < g_ncores_w && i < nw; i++) printf(" %d", g_core_cpu[i]);
        printf(" (host thread started on %d)\n", sched_getcpu());
    }
    printf("  CPU co-grind: %d CPUs in affinity, cgroup quota %s%.2f, %d workers%s\n",
           ncpu, quota > 0 ? "" : "none ", quota > 0 ? quota : 0.0, nw > 0 ? nw : 0,
           __builtin_cpu_supports("avx512f") ? ", avx512f" : __builtin_cpu_supports("avx2") ? ", avx2" : "");
    if (nw <= 0) return 0;
    if (pp->suffix_len > 119 || pp->seq_offset + 4 > pp->suffix_len || pp->lt_offset + 4 > pp->suffix_len || lt_max <= lt_min) return 0;

    shared_t *S = new shared_t();
    g_cg = S;
    S->pp = pp; S->lt_min = lt_min; S->lt_range = lt_max - lt_min;
    S->chunks_per_seq = (S->lt_range + QSB_CG_BATCH_MIN - 1) / QSB_CG_BATCH_MIN;
    S->n_chunks = (uint64_t)S->chunks_per_seq * 0x3FFFFFFFull;      /* sequences 0xFFFFFFFE down to 0xC0000000 */
    S->nblk = pp->suffix_len < 56 ? 1 : 2;
    S->cache_first = S->nblk == 2 && pp->seq_offset + 4 <= 64 && pp->lt_offset >= 64;
    S->simd.store(0); S->simd_ok = 0; S->simd_env = -1;
#if QSB_CG_HAVE_SIMD
    S->simd_ok = (__builtin_cpu_supports("avx2") ? 4 : 0) | (__builtin_cpu_supports("avx512f") ? 8 : 0);
    if (getenv("QSB_CPU_GRIND_SIMD")) { S->simd_env = atoi(getenv("QSB_CPU_GRIND_SIMD")); if (S->simd_env != 0 && !(S->simd_ok & S->simd_env)) S->simd_env = 0; }
    S->simd.store(-1);
#endif
    mkdir("results", 0755);
    S->hit_fd = open("results/pinning_hit_cpu.txt", O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (S->hit_fd < 0) { g_cg = NULL; return 0; }
    {   /* A = u2 R (recid 0); the table itself is built later, see table_main */
        uint64_t aw[4], bw[4]; memcpy(aw, pp->u2r_x, 32); memcpy(bw, pp->u2r_y, 32);
        fe_from_w(&S->ax, aw); fe_from_w(&S->ay, bw);
        memcpy(S->nri, pp->neg_r_inv, 32);
    }
    S->allowed.store(0);
    S->nworkers = nw;
    if (g_ncores_w > 0 && nw > 0 && g_host_ok) {
        /* keep the GPU host thread off the workers' cores (both hyperthreads of each) */
        pthread_setaffinity_np(pthread_self(), sizeof g_host_set, &g_host_set);
    }
    for (int i = 0; i < nw; i++) {
        pthread_t t;
        if (pthread_create(&t, NULL, worker_main, (void *)(intptr_t)i) != 0) { S->nworkers = i; break; }
        pthread_detach(t);
    }
    memset(&g_ctl, 0, sizeof g_ctl);
    g_ctl.wmax = S->nworkers; g_ctl.cur = 0;
    g_ctl.verbose = g_ctl_verbose = getenv("QSB_CPU_GRIND_VERBOSE") != NULL;
    return S->nworkers;
}

/* Exactness self-test: recompute a few CPU candidates with the OpenSSL gate's recovery and
 * compare pubkey hashes. Runs once in the first worker-free moment (called by the host loop). */

static void set_allowed(int n) { if (g_cg) g_cg->allowed.store(n, std::memory_order_relaxed); g_ctl.cur = n; }

/* Called by the GPU host loop after every drained GPU batch of gpu_batch candidates. `starved`
 * is 1 when the host found the pipeline's other in-flight batch already finished, i.e. the GPU ran
 * out of queued work because the host was late.
 *
 * Policy. Before the table is ready no worker runs, and the starvation rate of the unloaded host
 * is recorded. Then 2 workers start; every 5 s (and >= 20 batches) the starvation rate is compared
 * with that baseline: more than 1 point above it halves the workers (2 -> 0) and caps them there
 * for 60 s; otherwise they double up to the budget. Once at the budget the CPU-time share check
 * runs (hidden quota), and once a minute an on/off comparison of the GPU batch interval sheds a
 * quarter of the workers only after three consecutive windows each lose more than 3% and more
 * than the CPU adds. */
static void tick(double now, double gpu_batch, int starved) {
    shared_t *S = g_cg;
    if (!S) return;
    if (!S->table_started) {
        S->table_started = 1;
        pthread_t tt;
        if (pthread_create(&tt, NULL, table_main, NULL) == 0) pthread_detach(tt); else S->failed.store(1);
    }
    ctl_t &C = g_ctl;
    const double dt = C.last_done > 0 ? now - C.last_done : 0;
    C.last_done = now;
    if (!S->ready.load(std::memory_order_acquire) || S->failed.load() || S->simd.load() < 0) return;
    if (C.phase == 0) { C.phase = 4; C.t_period = now; return; }
    if (C.phase == 4) {                 /* baseline: >= 3 s and >= 20 batches with no worker running */
        C.base_drains++; C.base_starved += starved != 0;
        if (now - C.t_period < 3.0 || C.base_drains < 20) return;
        C.phase = 0;                    /* falls through to the start below */
    }
    if (S->tentative.load() >= 8 && S->hits.load() == 0) {   /* CPU path disagrees with the exact gate */
        if (C.cur) set_allowed(0);
        C.wmax = 0; return;
    }
    C.drains++; C.starved += starved != 0;
    if (C.phase == 0) {             /* entered only from the baseline phase */
        C.base_rate = C.base_drains >= 20 ? (double)C.base_starved / C.base_drains : 0;
        C.phase = 1; C.t_period = now; C.drains = C.starved = 0; C.ceiling = C.wmax; C.hold_until = 0;
        C.cand0 = S->cand_done.load(); C.cand_t0 = now; C.next_ab = now + 60.0;
        set_allowed(C.base_rate > 0.05 ? 0 : (C.wmax < 2 ? C.wmax : 2));
        if (C.verbose) printf("  [CPU] base starvation %.3f (%llu drains); start with %d workers\n",
                              C.base_rate, (unsigned long long)C.base_drains, C.cur);
        return;
    }
    if (C.phase == 3) {                                  /* A/B window in progress */
        const int on = (C.ab_left & 1) == 0;
        if (C.win_skip) C.win_skip = 0;
        else { if (on) { C.ab_on_sum += dt; C.ab_on_n++; } else { C.ab_off_sum += dt; C.ab_off_n++; } C.win_n++; }
        if (C.win_n < 3 || now - C.win_t0 < 1.0) return;
        C.ab_left--;
        C.win_t0 = now; C.win_n = 0; C.win_skip = 1;
        if (C.ab_left > 0) { set_allowed(((C.ab_left & 1) == 0) ? C.ab_level : 0); return; }
        set_allowed(C.ab_level);
        const double on_t = C.ab_on_sum / (C.ab_on_n ? C.ab_on_n : 1);
        const double off_t = C.ab_off_sum / (C.ab_off_n ? C.ab_off_n : 1);
        const double loss = on_t / off_t - 1.0;
        const uint64_t cd = S->cand_done.load();
        const double cpu_rate = now > C.cand_t0 ? (double)(cd - C.cand0) / (now - C.cand_t0) : 0;
        C.cand0 = cd; C.cand_t0 = now;
        const double cpu_frac = on_t > 0 ? cpu_rate / (gpu_batch / on_t) : 0;
        if (C.verbose) printf("  [CPU] A/B: gpu batch on %.5fs off %.5fs (loss %+.3f%%), cpu %.0f cand/s, %d workers, hits %llu/%llu exact\n",
                              on_t, off_t, 100 * loss, cpu_rate, C.cur,
                              (unsigned long long)S->hits.load(), (unsigned long long)S->tentative.load());
        const int bad = loss > 0.03 && loss > cpu_frac;
        if (bad && C.strikes >= 2) {
            int nw = C.wmax - (C.wmax + 3) / 4; if (nw < 0) nw = 0;
            C.wmax = nw; C.ceiling = nw; if (C.cur > nw) set_allowed(nw); C.strikes = 0; C.next_ab = now + 5.0;
        } else if (bad) { C.strikes++; C.next_ab = now + 2.0; }
        else { C.strikes = 0; C.next_ab = now + 60.0; }
        C.phase = 1; C.t_period = now; C.drains = C.starved = 0;
        return;
    }
    /* phase 1: starvation-driven ramp */
    if (now - C.t_period >= 5.0 && C.drains >= 20) {
        const double rate = (double)C.starved / C.drains;
        if (rate > C.base_rate + 0.01 && C.cur > 0) {
            const int nw = C.cur <= 2 ? 0 : C.cur / 2;
            set_allowed(nw); C.ceiling = nw; C.hold_until = now + 60.0;
            if (C.verbose) printf("  [CPU] starvation %.3f (base %.3f): %d workers\n", rate, C.base_rate, nw);
        } else if (C.cur < C.wmax) {
            int cap = now < C.hold_until ? C.ceiling : C.wmax;
            int nw = C.cur < 1 ? 1 : C.cur * 2; if (nw > cap) nw = cap;
            if (nw > C.cur) {
                set_allowed(nw);
                if (C.verbose) printf("  [CPU] starvation %.3f: ramp to %d workers\n", rate, nw);
                if (nw == C.wmax) { C.t_full = now; C.busy0 = busy_total(); C.busy_t0 = now; }
            }
        }
        C.t_period = now; C.drains = C.starved = 0;
    }
    /* share check, once, 2 s after first reaching the budget */
    if (!C.share_done && C.cur == C.wmax && C.wmax > 0 && C.t_full > 0 && now - C.t_full >= 2.0) {
        C.share_done = 1;
        const double got = (double)(busy_total() - C.busy0) * 1e-9 / (now - C.busy_t0);
        if (C.verbose) printf("  [CPU] share check: %d workers received %.2f CPUs\n", C.cur, got);
        if (got < 0.8 * C.cur) { int nw = (int)got - 1; if (nw < 0) nw = 0; C.wmax = nw; C.ceiling = nw; set_allowed(nw); }
    }
    if (now >= C.next_ab && C.cur > 0) {
        C.phase = 3; C.ab_level = C.cur; C.ab_left = 4; C.ab_on_sum = C.ab_off_sum = 0; C.ab_on_n = C.ab_off_n = 0;
        C.win_t0 = now; C.win_n = 0; C.win_skip = 1;
    }
}

/* Stop the workers and wait (bounded) for in-flight batches so no worker is inside
 * OpenSSL or the hit file when the process exits. */
static void stop_and_report() {
    shared_t *S = g_cg;
    if (!S) return;
    S->stop.store(1);
    for (int i = 0; i < 200 && S->running.load() > 0; i++) usleep(1000);
    if (g_ctl.verbose && S->cand_done.load())
        printf("  [CPU] tsc/cand: sha %.0f ec %.0f\n",
               (double)S->sha_cyc.load() / S->cand_done.load(), (double)S->ec_cyc.load() / S->cand_done.load());
    printf("  CPU co-grind: %llu candidates, %llu tentative, %llu verified hits written\n",
           (unsigned long long)S->cand_done.load(), (unsigned long long)S->tentative.load(),
           (unsigned long long)S->hits.load());
    fflush(stdout);
}

} /* namespace qcg */
#endif /* QSB_CPU_COGRIND_H */
