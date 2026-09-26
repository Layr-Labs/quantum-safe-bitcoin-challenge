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
#ifndef QSB_CG_W
#define QSB_CG_W 16                       /* fixed-window width (bits) */
#endif
#define QSB_CG_NWIN ((256 + QSB_CG_W - 1) / QSB_CG_W)
#define QSB_CG_TSIZE (1u << QSB_CG_W)     /* entries per window, index 0 unused */
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
    const uint64_t k = S->next_chunk.fetch_add(1, std::memory_order_relaxed);
    if (k >= S->n_chunks) return 0;
    const uint32_t seq = 0xFFFFFFFEu - (uint32_t)(k / S->chunks_per_seq);
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

static void set_idle_priority() {
    struct sched_param sp; memset(&sp, 0, sizeof sp);
    if (pthread_setschedparam(pthread_self(), SCHED_IDLE, &sp) != 0)
        setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 19);
}

static void *worker_main(void *arg) {
    shared_t *S = g_cg;
    const int id = (int)(intptr_t)arg;
    set_idle_priority();
    worker_t *w = (worker_t *)aligned_alloc(64, (sizeof(worker_t) + 63) & ~(size_t)63);
    uint8_t *scratch = NULL;
    if (!w) return NULL;
    w->id = id; w->cur_seq_tag = 0;
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
    while (!S->ready.load(std::memory_order_acquire)) { if (S->stop.load()) return NULL; usleep(2000); }
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
struct build_arg { int j0, j1; fe bx[QSB_CG_NWIN], by[QSB_CG_NWIN]; };
static void *builder_main(void *arg) {
    build_arg *a = (build_arg *)arg;
    set_idle_priority();
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

/* ---------------- controller (called from the GPU host loop) ---------------- */
struct ctl_t {
    int wmax, cur;
    int phase;              /* 0 warm-up, 1 cpu-share check, 2 steady; 3 A/B off-window */
    double t_phase;
    double last_done;       /* time of the last GPU batch completion */
    /* A/B measurement */
    int ab_left;
    double ab_on_sum; int ab_on_n;
    double ab_off_sum; int ab_off_n;
    double next_ab;
    uint64_t busy0; double busy_t0;
    uint64_t cand0; double cand_t0;
    double win_t0; int win_n, win_skip, strikes;
    int verbose;
};
static ctl_t g_ctl;

static uint64_t busy_total() { uint64_t s = 0; for (int i = 0; i < g_cg->nworkers; i++) s += g_cg->busy_ns[i].load(std::memory_order_relaxed); return s; }

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
    if (env) nw = atoi(env);
    if (nw > QSB_CG_MAXW) nw = QSB_CG_MAXW;
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
    S->table = (tentry *)aligned_alloc(2u << 20, (size_t)QSB_CG_NWIN * QSB_CG_TSIZE * sizeof(tentry));
    if (!S->table) { g_cg = NULL; return 0; }
    madvise(S->table, (size_t)QSB_CG_NWIN * QSB_CG_TSIZE * sizeof(tentry), MADV_HUGEPAGE);
    for (int j = 0; j < QSB_CG_NWIN; j++) memset(&S->table[(size_t)j * QSB_CG_TSIZE], 0, sizeof(tentry));
    /* window bases Bj = 2^(W j) * neg_r_inv * G and A = u2 R, via OpenSSL */
    static build_arg ba[QSB_CG_NWIN];
    {
        EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
        BN_CTX *ctx = BN_CTX_new();
        BIGNUM *order = BN_new(), *k = BN_new(), *x = BN_new(), *y = BN_new();
        EC_POINT *P = EC_POINT_new(grp);
        EC_GROUP_get_order(grp, order, ctx);
        BN_lebin2bn(pp->neg_r_inv, 32, k);
        fe bx[QSB_CG_NWIN], by[QSB_CG_NWIN];
        for (int j = 0; j < QSB_CG_NWIN; j++) {
            EC_POINT_mul(grp, P, k, NULL, NULL, ctx);
            EC_POINT_get_affine_coordinates(grp, P, x, y, ctx);
            uint8_t xb[32], yb[32];
            BN_bn2lebinpad(x, xb, 32); BN_bn2lebinpad(y, yb, 32);
            uint64_t xw[4], yw[4]; memcpy(xw, xb, 32); memcpy(yw, yb, 32);
            fe_from_w(&bx[j], xw); fe_from_w(&by[j], yw);
            for (int s = 0; s < QSB_CG_W; s++) BN_mod_lshift1(k, k, order, ctx);
        }
        uint64_t aw[4], bw[4]; memcpy(aw, pp->u2r_x, 32); memcpy(bw, pp->u2r_y, 32);
        fe_from_w(&S->ax, aw); fe_from_w(&S->ay, bw);
        BN_free(order); BN_free(k); BN_free(x); BN_free(y); EC_POINT_free(P); BN_CTX_free(ctx); EC_GROUP_free(grp);
        int nb = nw < 8 ? nw : 8; if (nb < 1) nb = 1;
        pthread_t bt[8];
        for (int b = 0; b < nb; b++) {
            ba[b].j0 = QSB_CG_NWIN * b / nb; ba[b].j1 = QSB_CG_NWIN * (b + 1) / nb;
            memcpy(ba[b].bx, bx, sizeof bx); memcpy(ba[b].by, by, sizeof by);
        }
        /* builders run detached-by-join in a helper so start() returns at once */
        struct helper_arg { int nb; build_arg *ba; };
        static helper_arg ha; ha.nb = nb; ha.ba = ba;
        pthread_t ht;
        auto helper = [](void *p) -> void * {
            helper_arg *h = (helper_arg *)p;
            pthread_t t[8];
            for (int b = 0; b < h->nb; b++) pthread_create(&t[b], NULL, builder_main, &h->ba[b]);
            for (int b = 0; b < h->nb; b++) pthread_join(t[b], NULL);
            g_cg->ready.store(1, std::memory_order_release);
            return NULL;
        };
        (void)bt;
        if (pthread_create(&ht, NULL, helper, &ha) != 0) { g_cg = NULL; return 0; }
        pthread_detach(ht);
    }
    S->allowed.store(0);
    S->nworkers = nw;
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

/* Called by the GPU host loop after every drained GPU batch of gpu_batch candidates. */
static void tick(double now, double gpu_batch) {
    shared_t *S = g_cg;
    if (!S) return;
    ctl_t &C = g_ctl;
    const double dt = C.last_done > 0 ? now - C.last_done : 0;
    C.last_done = now;
    if (!S->ready.load(std::memory_order_acquire) || S->failed.load()) return;
    if (S->tentative.load() >= 8 && S->hits.load() == 0) {   /* CPU path disagrees with the exact gate */
        if (C.cur) printf("  CPU co-grind: off (%llu tentative hits, none exact)\n", (unsigned long long)S->tentative.load());
        C.wmax = 0; set_allowed(0); return;
    }
    if (C.phase == 0) {                        /* table ready: enable all workers */
        set_allowed(C.wmax);
        C.phase = 1; C.t_phase = now;
        C.busy0 = busy_total(); C.busy_t0 = now;
        C.cand0 = S->cand_done.load(); C.cand_t0 = now;
        C.next_ab = now + 20.0;
        return;
    }
    if (C.phase == 1 && now - C.t_phase >= 2.0) {
        /* CPU share check: do the workers get the CPU time they ask for? */
        const double got = (double)(busy_total() - C.busy0) * 1e-9 / (now - C.busy_t0);
        if (C.verbose) printf("  [CPU] share check: %d workers received %.2f CPUs\n", C.cur, got);
        if (got < 0.8 * C.cur) {
            int nw = (int)got - 2; if (nw < 0) nw = 0;
            C.wmax = nw; set_allowed(nw);
            printf("  CPU co-grind: workers received %.1f CPUs; using %d\n", got, nw);
        }
        C.phase = 2; C.t_phase = now;
        return;
    }
    if (C.phase == 2 && now >= C.next_ab && C.cur > 0) {
        /* A/B: four windows on,off,on,off; each lasts >= 1 s and >= 3 GPU batches, and the
         * first batch after each switch (it straddles both settings) is not measured */
        C.phase = 3; C.ab_left = 4; C.ab_on_sum = C.ab_off_sum = 0; C.ab_on_n = C.ab_off_n = 0;
        C.win_t0 = now; C.win_n = 0; C.win_skip = 1;
        return;
    }
    if (C.phase == 3) {
        /* windows: >= 1 s and >= 3 GPU batches each */
        const int on = (C.ab_left & 1) == 0;       /* 4: on, 3: off, 2: on, 1: off */
        if (C.win_skip) C.win_skip = 0;
        else { if (on) { C.ab_on_sum += dt; C.ab_on_n++; } else { C.ab_off_sum += dt; C.ab_off_n++; } C.win_n++; }
        if (C.win_n < 3 || now - C.win_t0 < 1.0) return;
        C.ab_left--;
        C.win_t0 = now; C.win_n = 0; C.win_skip = 1;
        if (C.ab_left > 0) { set_allowed(((C.ab_left & 1) == 0) ? C.wmax : 0); return; }
        set_allowed(C.wmax);
        const double on_t = C.ab_on_sum / (C.ab_on_n ? C.ab_on_n : 1);
        const double off_t = C.ab_off_sum / (C.ab_off_n ? C.ab_off_n : 1);
        const double loss = on_t / off_t - 1.0;     /* GPU slowdown with the workers on */
        const uint64_t cd = S->cand_done.load();
        const double cpu_rate = C.cand_t0 > 0 && now > C.cand_t0 ? (double)(cd - C.cand0) / (now - C.cand_t0) : 0;
        C.cand0 = cd; C.cand_t0 = now;
        if (C.verbose) printf("  [CPU] A/B: gpu batch on %.5fs off %.5fs (loss %+.3f%%), cpu %.0f cand/s, %d workers, hits %llu/%llu exact\n",
                              on_t, off_t, 100 * loss, cpu_rate, C.wmax,
                              (unsigned long long)S->hits.load(), (unsigned long long)S->tentative.load());
        /* The comparison guards against real contention (a hidden CPU quota, a shared core),
         * not against noise: one window's noise is up to ~1% on short GPU batches. Shed a quarter
         * of the workers only after two consecutive windows each show a GPU loss above 1.5% that
         * also exceeds what the workers add. */
        const double cpu_frac = on_t > 0 ? cpu_rate / (gpu_batch / on_t) : 0;
        const int bad = loss > 0.015 && loss > cpu_frac;
        if (bad && C.strikes >= 1) {
            int nw = C.wmax - (C.wmax + 3) / 4; if (nw < 0) nw = 0;
            C.wmax = nw; set_allowed(nw); C.strikes = 0;
            printf("  CPU co-grind: GPU batch time +%.2f%% with workers; using %d\n", 100 * loss, nw);
            C.next_ab = now + 5.0;
        } else if (bad) { C.strikes = 1; C.next_ab = now + 2.0; }
        else { C.strikes = 0; C.next_ab = now + 60.0; }
        C.phase = 2;
        return;
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
