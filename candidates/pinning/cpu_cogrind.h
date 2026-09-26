/* cpu_cogrind.h -- host-CPU co-grinding for the pinning search (v2).
 *
 * The GPU walks sequences upward from 0x80000000. Idle host cores grind sequences counting DOWN
 * from 0xFFFFFFFE over the same locktime range [LT_MIN, LT_MAX), in chunks of QSB_CG_B locktimes
 * handed out by an atomic counter, so the two candidate sets are disjoint by construction. Every
 * CPU-nominated hit is re-derived by the tree's unchanged exact OpenSSL gate (qsb_host_exact_hit)
 * before it is appended to results/pinning_hit_cpu.txt as `sequence= locktime= recid=`.
 *
 * Per candidate (sequence block compressed once per sequence):
 *   SHA   : suffix block 1 from the sequence midstate, SHA256 of the 32-byte digest -> z.
 *           8-lane AVX2 multi-buffer SHA-256, or SHA-NI (two blocks interleaved) when present.
 *   EC    : Q+ = z B + A, Q- = Q+ - 2A with B = neg_r_inv G, A = u2 R. z is recoded into one
 *           unsigned low window whose table has A folded in (entry d = d B + A) and signed
 *           windows above it (entries m 2^pos B, m = 1..2^(w-1); the sign flips y). The chain
 *           gives Q+ directly; Q- is one more addition with the constant -2A. All additions are
 *           affine, their inversions batched over the QSB_CG_B candidates of a chunk (one
 *           field inversion per step). The table layout (72 MiB .. 1.2 GiB) is chosen from
 *           MemAvailable at start and built in the background by SCHED_IDLE threads.
 *   HASH  : SHA256 of both compressed keys (8-lane AVX2), H0 leading-zero prefilter, then the
 *           exact gate.
 * EC back ends: AVX2 4-lane libsecp256k1 10x26 field (cpu_cogrind_vec.h), scalar 4x64
 * MULX/ADCX/ADOX (cg_fe4.h), portable C. Worker 0 times the available combinations on real
 * batches at start and keeps the fastest.
 *
 * Contention safety (unchanged from the v1 co-grinder): SCHED_IDLE workers (fallback nice 19);
 * worker count min(affinity CPUs, cgroup quota) minus a reserve; a share check after the start
 * and a periodic GPU on/off A/B that sheds workers on a measured GPU loss. QSB_COGRIND=0 removes
 * all of it.
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
#include <immintrin.h>
#include <cpuid.h>
#include "cg_fe4.h"
#include "cg_sha.h"
#include "cg_v26asm.h"

namespace qcg {
static cpu_set_t g_worker_set; static int g_worker_set_on = 0;

/* ---------------- configuration ---------------- */
#ifndef QSB_CG_B
#define QSB_CG_B 2048                     /* candidates (locktimes) per chunk = inversion batch */
#endif
#define QSB_CG_BMAX QSB_CG_B
#ifndef QSB_CG_PF
#define QSB_CG_PF 4                       /* table prefetch distance, in 4-candidate blocks */
#endif
#define QSB_CG_MAXWIN 16
#define QSB_CG_MAXW 256                   /* max worker threads */
#if (QSB_CG_B % 8) != 0
#error "QSB_CG_B must be a multiple of 8"
#endif

/* digit codes (windows >= 1): bits 0..29 entry index (= magnitude - 1), bit 30 zero digit, bit 31 negative */
#define QCG_IDXM 0x3FFFFFFFu
#define QCG_ZERO(c) (((c) >> 30) & 1u)
#define QCG_NEG(c) ((c) >> 31)
#define QCG_ZCODE (1u << 30)

struct tentry { uint64_t x[4], y[4]; };   /* 64 B, one cache line, canonical affine coordinates */

struct layout_t {
    int nwin;                             /* window 0 (unsigned, A folded) + signed windows */
    int w[QSB_CG_MAXWIN], pos[QSB_CG_MAXWIN];
    uint64_t off[QSB_CG_MAXWIN], cnt[QSB_CG_MAXWIN], total;
    const char *name;
};
static int layout_make(layout_t *L, const char *name, int w0, const int *ws, int k) {
    memset(L, 0, sizeof *L);
    L->name = name; L->nwin = 1 + k;
    L->w[0] = w0; L->pos[0] = 0; L->cnt[0] = 1ull << w0; L->off[0] = 0;
    int pos = w0; uint64_t off = L->cnt[0];
    for (int j = 1; j <= k; j++) {
        L->w[j] = ws[j - 1]; L->pos[j] = pos; L->cnt[j] = 1ull << (ws[j - 1] - 1); L->off[j] = off;
        pos += ws[j - 1]; off += L->cnt[j];
    }
    L->total = off;
    /* the top signed digit must absorb the recoding carry: pos_k + w_k >= 257 */
    return (k >= 1 && L->pos[k] + L->w[k] >= 257 && L->nwin <= QSB_CG_MAXWIN) ? 0 : -1;
}
/* 1216 MiB: 12 table additions + 1 */
static const int LAY_L_W[] = {21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22};
/* 4096 MiB: one fewer signed window and one fewer affine chain addition. */
static const int LAY_XL_W[] = {23, 23, 23, 23, 23, 24, 24, 24, 24, 24};
/* 400 MiB: 13 */
static const int LAY_M_W[] = {19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20};
/* 160 MiB: 14 */
static const int LAY_S_W[] = {18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19};
/* 72 MiB: 15 */
static const int LAY_T_W[] = {17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18};
static int layout_by_name(layout_t *L, const char *nm) {
    if (!strcmp(nm, "xlarge")) return layout_make(L, "xlarge", 22, LAY_XL_W, 10);
    if (!strcmp(nm, "large"))  return layout_make(L, "large", 22, LAY_L_W, 11);
    if (!strcmp(nm, "medium")) return layout_make(L, "medium", 20, LAY_M_W, 12);
    if (!strcmp(nm, "small"))  return layout_make(L, "small", 18, LAY_S_W, 13);
    if (!strcmp(nm, "tiny"))   return layout_make(L, "tiny", 17, LAY_T_W, 14);
    return -1;
}

/* ---------------- shared state ---------------- */
struct shared_t {
    const pinning2_params_t *pp;
    uint32_t lt_min, lt_range, chunks_per_seq;
    int nblk, cache_first;
    uint64_t n_chunks;
    layout_t lay;
    tentry *table; size_t table_bytes;
    uint64_t ax_w[4], ay_w[4];            /* A = u2 R */
    uint64_t dx_w[4], dy_w[4];            /* D = -2A */
    uint32_t w1_tmpl[16];                 /* suffix block 1 words with the locktime bytes cleared */
    int lt_word[4], lt_shift[4];          /* where locktime byte b lands in block 1 */
    int hit_fd;
    std::atomic<uint64_t> next_chunk;
    std::atomic<uint64_t> cand_done;
    std::atomic<uint64_t> hits;
    std::atomic<uint64_t> tentative;
    std::atomic<int> allowed;
    std::atomic<int> stop;
    std::atomic<int> running;
    std::atomic<int> ready;
    std::atomic<int> failed;
    int nworkers;
    int has_avx2, has_adx, has_sha;
    int ec_env, sha_env;                  /* overrides, -1 = auto */
    std::atomic<int> ec_mode;             /* 0 = C, 1 = scalar asm, 2 = avx2 x4; -1 until chosen */
    std::atomic<int> sha_mode;            /* 0 = ref, 1 = avx2 x8, 2 = sha-ni */
    std::atomic<uint64_t> busy_ns[QSB_CG_MAXW];
    std::atomic<uint64_t> sha_cyc, ec_cyc;
    double t_build;
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

struct worker_t {
    int id;
    int n;                                /* candidates in the batch */
    uint32_t seq, lt0;
    uint64_t cur_seq_tag;
    uint32_t mid1[8];                     /* state after suffix block 0 for cur seq */
    alignas(64) uint64_t zq[4][QSB_CG_BMAX + 8];          /* z of the batch, word-major: zq[k][i] = word k (LE) of candidate i */
    alignas(64) uint32_t dig[QSB_CG_MAXWIN][QSB_CG_BMAX + 32];
    uint8_t zf[QSB_CG_MAXWIN][QSB_CG_BMAX / 4 + 8];
    EC_GROUP *grp; BN_CTX *ctx; BIGNUM *order, *nri, *rx, *ry; EC_POINT *Ru2;
};

/* A tentative CPU hit: re-derive (sequence, locktime, recid) with the exact OpenSSL gate and
 * append it to the CPU hit file only if it passes. */
static void publish(worker_t *w, int i, int recid) {
    shared_t *S = g_cg;
    S->tentative.fetch_add(1, std::memory_order_relaxed);
    const uint32_t lt = w->lt0 + (uint32_t)i;
    if (qsb_host_exact_hit(S->pp, w->seq, lt, recid, w->grp, w->ctx, w->order, w->nri, w->Ru2)) {
        char line[96];
        int wl = snprintf(line, sizeof line, "sequence=%u locktime=%u recid=%d\n", w->seq, lt, recid);
        if (write(S->hit_fd, line, (size_t)wl) == wl) S->hits.fetch_add(1, std::memory_order_relaxed);
    }
}

/* scalar field inversion used by every EC path (Fermat, 4x64) */
static int g_fe_asm = 0;
static void scalar_inv(uint64_t *r, const uint64_t *a) {
    if (g_fe_asm) qcg_fe::fe_inv<qcg_fe::FeAsm>(r, a); else qcg_fe::fe_inv<qcg_fe::FeC>(r, a);
}

static void set_idle_priority() {
    struct sched_param sp; memset(&sp, 0, sizeof sp);
    if (pthread_setschedparam(pthread_self(), SCHED_IDLE, &sp) != 0)
        setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 19);
}

/* ---------------- digit recoding ---------------- */
/* per-window constants, filled by start(): word index, shift, mask, 2^w, 2^(w-1) */
struct rwin { uint32_t wi, sh, two_word; uint64_t mask, full, half; };
static rwin g_rw[QSB_CG_MAXWIN];
/* z: 4 little-endian 64-bit words. Writes dig[j][i] for every window. Returns a bitmask of
 * windows (>= 1) with a zero digit. Branch-free except for the (rare) zero digit. */
static inline unsigned recode(worker_t *w, int i, const uint64_t *q) {
    const layout_t &L = g_cg->lay;
    unsigned zmask = 0;
    w->dig[0][i] = (uint32_t)(q[0] & g_rw[0].mask);
    uint64_t carry = 0;
    const int k = L.nwin - 1;
    for (int j = 1; j <= k; j++) {
        const rwin &r = g_rw[j];
        uint64_t v = q[r.wi] >> r.sh;
        if (r.two_word) v |= q[r.wi + 1] << (64 - r.sh);      /* per-window constant: predictable */
        v = (v & r.mask) + carry;                              /* top window: mask covers all remaining bits */
        const uint64_t neg = (uint64_t)(v > r.half) & (uint64_t)(j < k);
        const uint64_t m = neg ? r.full - v : v;              /* magnitude, 0 .. half */
        carry = neg;
        uint32_t code = (uint32_t)(m - 1) | (uint32_t)(neg << 31);
        if (__builtin_expect(m == 0, 0)) { code = QCG_ZCODE; zmask |= 1u << j; }
        w->dig[j][i] = code;
    }
    return zmask;
}
static void recode_init(const layout_t &L) {
    g_rw[0].mask = (1ull << L.w[0]) - 1;
    for (int j = 1; j < L.nwin; j++) {
        rwin &r = g_rw[j];
        const unsigned pos = (unsigned)L.pos[j], wd = (unsigned)L.w[j];
        const int top = (j == L.nwin - 1);
        const unsigned bits = top ? 256 - pos : wd;           /* the top window takes all remaining bits */
        r.wi = pos >> 6; r.sh = pos & 63;
        r.two_word = (r.sh + bits > 64 && r.wi < 3) ? 1 : 0;
        r.mask = (bits >= 64) ? ~0ull : (1ull << bits) - 1;
        r.full = 1ull << wd; r.half = 1ull << (wd - 1);
    }
}

/* 4 candidates per AVX2 vector (the carry chains of different candidates run in parallel) */
__attribute__((target("avx2"), noinline))
static unsigned recode_avx2(worker_t *w, int np) {
    const layout_t &L = g_cg->lay;
    const int k = L.nwin - 1;
    unsigned zmask = 0;
    const __m256i idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    const __m256i one = _mm256_set1_epi64x(1), zc = _mm256_set1_epi64x(QCG_ZCODE);
    for (int i = 0; i < np; i += 4) {
        const __m256i q[4] = {_mm256_load_si256((const __m256i *)&w->zq[0][i]), _mm256_load_si256((const __m256i *)&w->zq[1][i]),
                              _mm256_load_si256((const __m256i *)&w->zq[2][i]), _mm256_load_si256((const __m256i *)&w->zq[3][i])};
        const __m256i d0 = _mm256_and_si256(q[0], _mm256_set1_epi64x((long long)g_rw[0].mask));
        _mm_storeu_si128((__m128i *)&w->dig[0][i], _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(d0, idx)));
        __m256i carry = _mm256_setzero_si256();
        for (int j = 1; j <= k; j++) {
            const rwin &r = g_rw[j];
            __m256i v = _mm256_srl_epi64(q[r.wi], _mm_cvtsi32_si128((int)r.sh));
            if (r.two_word) v = _mm256_or_si256(v, _mm256_sll_epi64(q[r.wi + 1], _mm_cvtsi32_si128((int)(64 - r.sh))));
            v = _mm256_add_epi64(_mm256_and_si256(v, _mm256_set1_epi64x((long long)r.mask)), carry);
            __m256i neg = _mm256_cmpgt_epi64(v, _mm256_set1_epi64x((long long)r.half));      /* values < 2^33: signed compare ok */
            if (j == k) neg = _mm256_setzero_si256();
            const __m256i m = _mm256_blendv_epi8(v, _mm256_sub_epi64(_mm256_set1_epi64x((long long)r.full), v), neg);
            carry = _mm256_and_si256(neg, one);
            __m256i code = _mm256_or_si256(_mm256_sub_epi64(m, one), _mm256_slli_epi64(_mm256_and_si256(neg, one), 31));
            const __m256i z = _mm256_cmpeq_epi64(m, _mm256_setzero_si256());
            if (__builtin_expect(!_mm256_testz_si256(z, z), 0)) { code = _mm256_blendv_epi8(code, zc, z); zmask |= 1u << j; }
            _mm_storeu_si128((__m128i *)&w->dig[j][i], _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(code, idx)));
        }
    }
    return zmask;
}
/* recode candidates [0, np) from w->zq */
static unsigned recode_all(worker_t *w, int np) {
    if (g_cg->has_avx2) return recode_avx2(w, np);
    unsigned zm = 0;
    for (int i = 0; i < np; i++) { const uint64_t q[4] = {w->zq[0][i], w->zq[1][i], w->zq[2][i], w->zq[3][i]}; zm |= recode(w, i, q); }
    return zm;
}

/* ---------------- SHA front end ---------------- */
static void seq_midstate(worker_t *w, uint32_t seq) {
    shared_t *S = g_cg; const pinning2_params_t *pp = S->pp;
    uint8_t m[128]; memset(m, 0, 128);
    memcpy(m, pp->suffix, pp->suffix_len);
    for (int b = 0; b < 4; b++) m[pp->seq_offset + b] = (uint8_t)(seq >> (8 * b));
    m[pp->suffix_len] = 0x80;
    const uint64_t bits = (uint64_t)pp->total_preimage_len * 8;
    const int lenoff = S->nblk * 64 - 8;
    for (int b = 0; b < 8; b++) m[lenoff + 7 - b] = (uint8_t)(bits >> (8 * b));
    uint32_t wv[16];
    for (int i = 0; i < 16; i++) wv[i] = (uint32_t)m[4 * i] << 24 | (uint32_t)m[4 * i + 1] << 16 | (uint32_t)m[4 * i + 2] << 8 | m[4 * i + 3];
    memcpy(w->mid1, pp->midstate, 32);
    qcg_sha::sha_compress_ref(w->mid1, wv);
}

/* generic (any layout) scalar z for one candidate */
static void z_generic(const worker_t *w, uint32_t seq, uint32_t lt, uint64_t *q) {
    shared_t *S = g_cg; const pinning2_params_t *pp = S->pp;
    uint8_t m[128]; memset(m, 0, 128);
    memcpy(m, pp->suffix, pp->suffix_len);
    for (int b = 0; b < 4; b++) m[pp->seq_offset + b] = (uint8_t)(seq >> (8 * b));
    for (int b = 0; b < 4; b++) m[pp->lt_offset + b] = (uint8_t)(lt >> (8 * b));
    m[pp->suffix_len] = 0x80;
    const uint64_t bits = (uint64_t)pp->total_preimage_len * 8;
    const int lenoff = S->nblk * 64 - 8;
    for (int b = 0; b < 8; b++) m[lenoff + 7 - b] = (uint8_t)(bits >> (8 * b));
    uint32_t st[8]; memcpy(st, pp->midstate, 32);
    for (int k = 0; k < S->nblk; k++) {
        uint32_t wv[16];
        for (int i = 0; i < 16; i++) wv[i] = (uint32_t)m[64 * k + 4 * i] << 24 | (uint32_t)m[64 * k + 4 * i + 1] << 16 | (uint32_t)m[64 * k + 4 * i + 2] << 8 | m[64 * k + 4 * i + 3];
        qcg_sha::sha_compress_ref(st, wv);
    }
    (void)w;
    uint32_t d[16] = {st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7], 0x80000000u, 0, 0, 0, 0, 0, 0, 256};
    uint32_t h[8]; memcpy(h, qcg_sha::IV256, 32);
    qcg_sha::sha_compress_ref(h, d);
    q[0] = (uint64_t)h[6] << 32 | h[7]; q[1] = (uint64_t)h[4] << 32 | h[5];
    q[2] = (uint64_t)h[2] << 32 | h[3]; q[3] = (uint64_t)h[0] << 32 | h[1];
}

__attribute__((target("avx2"), noinline))
static unsigned z_avx2_8(worker_t *w, int i0) {
    using namespace qcg_sha;
    shared_t *S = g_cg;
    v8u lt = _mm256_add_epi32(_mm256_set1_epi32((int)(w->lt0 + (uint32_t)i0)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    v8u W[16];
    for (int k = 0; k < 16; k++) W[k] = _mm256_set1_epi32((int)S->w1_tmpl[k]);
    for (int b = 0; b < 4; b++) {
        v8u byte = _mm256_and_si256(_mm256_srli_epi32(lt, 8 * b), _mm256_set1_epi32(0xFF));
        W[S->lt_word[b]] = _mm256_or_si256(W[S->lt_word[b]], _mm256_sllv_epi32(byte, _mm256_set1_epi32(S->lt_shift[b])));
    }
    v8u st[8];
    for (int k = 0; k < 8; k++) st[k] = _mm256_set1_epi32((int)w->mid1[k]);
    s8_compress_full(st, W);
    for (int k = 0; k < 8; k++) W[k] = st[k];
    W[8] = _mm256_set1_epi32((int)0x80000000u);
    for (int k = 9; k < 15; k++) W[k] = _mm256_setzero_si256();
    W[15] = _mm256_set1_epi32(256);
    for (int k = 0; k < 8; k++) st[k] = _mm256_set1_epi32((int)IV256[k]);
    s8_compress_full(st, W);
    /* zq[k] for lanes: word k (LE 64-bit) = Z[7-2k-1] << 32 | Z[7-2k] (Z0 most significant) */
    for (int k = 0; k < 4; k++) {
        const __m256i hi = st[6 - 2 * k], lo = st[7 - 2 * k];
        const __m256i a = _mm256_unpacklo_epi32(lo, hi), b = _mm256_unpackhi_epi32(lo, hi);   /* lanes 0,1,4,5 | 2,3,6,7 */
        _mm256_storeu_si256((__m256i *)&w->zq[k][i0], _mm256_permute2x128_si256(a, b, 0x20));
        _mm256_storeu_si256((__m256i *)&w->zq[k][i0 + 4], _mm256_permute2x128_si256(a, b, 0x31));
    }
    return 0;
}

__attribute__((target("sha,sse4.1"), noinline))
static unsigned z_shani_2(worker_t *w, int i0) {
    using namespace qcg_sha;
    shared_t *S = g_cg;
    uint32_t WA[16], WB[16];
    memcpy(WA, S->w1_tmpl, 64); memcpy(WB, S->w1_tmpl, 64);
    const uint32_t la = w->lt0 + (uint32_t)i0, lb = la + 1;
    for (int b = 0; b < 4; b++) {
        WA[S->lt_word[b]] |= ((la >> (8 * b)) & 0xFF) << S->lt_shift[b];
        WB[S->lt_word[b]] |= ((lb >> (8 * b)) & 0xFF) << S->lt_shift[b];
    }
    uint32_t sa[8], sb[8];
    memcpy(sa, w->mid1, 32); memcpy(sb, w->mid1, 32);
    shani_compress2(sa, WA, sb, WB);
    uint32_t DA[16] = {sa[0], sa[1], sa[2], sa[3], sa[4], sa[5], sa[6], sa[7], 0x80000000u, 0, 0, 0, 0, 0, 0, 256};
    uint32_t DB[16] = {sb[0], sb[1], sb[2], sb[3], sb[4], sb[5], sb[6], sb[7], 0x80000000u, 0, 0, 0, 0, 0, 0, 256};
    memcpy(sa, IV256, 32); memcpy(sb, IV256, 32);
    shani_compress2(sa, DA, sb, DB);
    for (int k = 0; k < 4; k++) {
        w->zq[k][i0] = (uint64_t)sa[6 - 2 * k] << 32 | sa[7 - 2 * k];
        w->zq[k][i0 + 1] = (uint64_t)sb[6 - 2 * k] << 32 | sb[7 - 2 * k];
    }
    return 0;
}

/* pad the digit arrays (prefetch overrun, 4-lane blocks) and set the per-block zero-digit flags;
 * np = computed candidates (multiple of 8), zmask = windows with a zero digit */
static void finish_digits(worker_t *w, int np, unsigned zmask) {
    shared_t *S = g_cg;
    for (int j = 0; j < S->lay.nwin; j++) for (int i = np; i < np + 32; i++) w->dig[j][i] = w->dig[j][np - 1];
    const int nb = np / 4;
    for (int j = 1; j < S->lay.nwin; j++) {
        if (!(zmask >> j & 1)) { memset(w->zf[j], 0, (size_t)nb + 8); continue; }
        for (int b = 0; b < nb + 8; b++) {
            uint8_t f = 0;
            if (b < nb) for (int l = 0; l < 4; l++) f |= (uint8_t)QCG_ZERO(w->dig[j][4 * b + l]);
            w->zf[j][b] = f;
        }
    }
}

/* Fill the batch with one chunk: z and digits per candidate. Returns n (0 when exhausted). */
static int fill_batch(worker_t *w) {
    shared_t *S = g_cg;
    const uint64_t k = S->next_chunk.fetch_add(1, std::memory_order_relaxed);
    if (k >= S->n_chunks) return 0;
    const uint32_t seq = 0xFFFFFFFEu - (uint32_t)(k / S->chunks_per_seq);
    const uint32_t off = (uint32_t)(k % S->chunks_per_seq) * (uint32_t)QSB_CG_B;
    int n = QSB_CG_B;
    if (off + (uint32_t)n > S->lt_range) n = (int)(S->lt_range - off);
    w->seq = seq; w->lt0 = S->lt_min + off; w->n = n;
    const int np = (n + 7) & ~7;              /* computed candidates (padding lanes are real locktimes, never published) */
    unsigned zmask = 0;
    const int mode = S->cache_first ? S->sha_mode.load(std::memory_order_relaxed) : 0;
    if (S->cache_first && w->cur_seq_tag != (uint64_t)seq + 1) { seq_midstate(w, seq); w->cur_seq_tag = (uint64_t)seq + 1; }
    if (mode == 2) { for (int i = 0; i < np; i += 2) z_shani_2(w, i); }
    else if (mode == 1) { for (int i = 0; i < np; i += 8) z_avx2_8(w, i); }
    else {
        for (int i = 0; i < np; i++) { uint64_t q[4]; z_generic(w, seq, w->lt0 + (uint32_t)i, q); for (int k = 0; k < 4; k++) w->zq[k][i] = q[k]; }
    }
    zmask = recode_all(w, np);
    finish_digits(w, np, zmask);
    return n;
}

/* ---------------- scalar EC back end (4x64, one candidate per element) ---------------- */
#include "cg_ec_scalar.h"

/* ---------------- AVX2 EC back end ---------------- */
#if defined(__x86_64__) && !defined(QSB_CG_NO_SIMD)
#include "cpu_cogrind_vec.h"
#define QSB_CG_HAVE_SIMD 1
#else
#define QSB_CG_HAVE_SIMD 0
#endif

/* ---------------- table build ---------------- */
#include "cg_table.h"


static void run_ec(worker_t *w, int mode, void *vs, void *ss) {
#if QSB_CG_HAVE_SIMD
    if (mode == 2) { v4::ec_batch(w, (v4::vstate *)vs); return; }
#else
    (void)vs;
#endif
    if (mode == 1) ec_scalar<qcg_fe::FeAsm>(w, (sstate *)ss);
    else ec_scalar<qcg_fe::FeC>(w, (sstate *)ss);
}

static void *worker_main(void *arg) {
    shared_t *S = g_cg;
    const int id = (int)(intptr_t)arg;
    if (g_worker_set_on) pthread_setaffinity_np(pthread_self(), sizeof g_worker_set, &g_worker_set);
    set_idle_priority();
    worker_t *w = (worker_t *)aligned_alloc(64, (sizeof(worker_t) + 63) & ~(size_t)63);
    if (!w) return NULL;
    w->id = id; w->cur_seq_tag = 0;
    w->grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    w->ctx = BN_CTX_new(); w->order = BN_new(); w->nri = BN_new(); w->rx = BN_new(); w->ry = BN_new();
    w->Ru2 = w->grp ? EC_POINT_new(w->grp) : NULL;
    if (!w->grp || !w->ctx || !w->order || !w->nri || !w->rx || !w->ry || !w->Ru2 ||
        !EC_GROUP_get_order(w->grp, w->order, w->ctx) ||
        !BN_lebin2bn(S->pp->neg_r_inv, 32, w->nri) || !BN_lebin2bn(S->pp->u2r_x, 32, w->rx) ||
        !BN_lebin2bn(S->pp->u2r_y, 32, w->ry) ||
        !EC_POINT_set_affine_coordinates(w->grp, w->Ru2, w->rx, w->ry, w->ctx)) { S->failed.store(1); return NULL; }
    void *vs = NULL;
#if QSB_CG_HAVE_SIMD
    if (S->has_avx2) vs = aligned_alloc(64, (sizeof(v4::vstate) + 63) & ~(size_t)63);
#endif
    sstate *ss = (sstate *)aligned_alloc(64, (sizeof(sstate) + 63) & ~(size_t)63);
    while (!S->ready.load(std::memory_order_acquire)) { if (S->stop.load() || S->failed.load()) return NULL; usleep(2000); }
    /* Worker 0 picks the SHA and EC paths on this CPU from timed real batches (their candidates
     * are real work and are counted); the others wait. Each (SHA, EC) combination runs 5 batches,
     * the first discarded; the minimum per-candidate time of each stage decides. The AVX2 EC
     * stage is the design point: the scalar MULX stage is chosen only if it is >10% faster. */
    if (id == 0 && S->ec_mode.load() < 0) {
        int ecs[3], nec = 0, shs[3], nsh = 0;
        if (S->ec_env >= 0) ecs[nec++] = S->ec_env;
        else { if (S->has_avx2 && QSB_CG_HAVE_SIMD) ecs[nec++] = 2; if (S->has_adx) ecs[nec++] = 1; if (!nec) ecs[nec++] = 0; }
        if (S->sha_env >= 0) shs[nsh++] = S->sha_env;
        else { if (S->has_sha) shs[nsh++] = 2; if (S->has_avx2) shs[nsh++] = 1; if (!nsh) shs[nsh++] = 0; }
        double sha_best[3] = {1e30, 1e30, 1e30}, ec_best[3] = {1e30, 1e30, 1e30};   /* indexed by mode */
        for (int a = 0; a < nsh; a++) {
            S->sha_mode.store(shs[a]);
            for (int b = 0; b < nec; b++) {
                for (int rep = 0; rep < 5; rep++) {
                    const uint64_t r0 = __rdtsc();
                    const int n = fill_batch(w);
                    if (!n) break;
                    const uint64_t r1 = __rdtsc();
                    run_ec(w, ecs[b], vs, ss);
                    const uint64_t r2 = __rdtsc();
                    S->cand_done.fetch_add((uint64_t)n, std::memory_order_relaxed);
                    if (rep == 0) continue;
                    const double ts = (double)(r1 - r0) / n, te = (double)(r2 - r1) / n;
                    if (ts < sha_best[shs[a]]) sha_best[shs[a]] = ts;
                    if (te < ec_best[ecs[b]]) ec_best[ecs[b]] = te;
                }
            }
        }
        int bsha = shs[0], bec = ecs[0];
        for (int a = 1; a < nsh; a++) if (sha_best[shs[a]] < sha_best[bsha]) bsha = shs[a];
        for (int b = 1; b < nec; b++) {
            const int m = ecs[b];
            const double margin = (bec == 2 && m != 2) ? 0.9 : (m == 2 && bec != 2) ? 1.0 / 0.9 : 1.0;
            if (ec_best[m] < margin * ec_best[bec]) bec = m;
        }
        S->sha_mode.store(bsha);
        S->ec_mode.store(bec);
        if (g_ctl_verbose) {
            char tb[6][16];
            const double tv[6] = {sha_best[2], sha_best[1], sha_best[0], ec_best[2], ec_best[1], ec_best[0]};
            for (int q = 0; q < 6; q++) { if (tv[q] < 1e29) snprintf(tb[q], sizeof tb[q], "%.0f", tv[q]); else snprintf(tb[q], sizeof tb[q], "-"); }
            printf("  [CPU] tsc/cand sha: sha-ni %s avx2x8 %s ref %s | ec: avx2x4 %s mulx %s c %s\n", tb[0], tb[1], tb[2], tb[3], tb[4], tb[5]);
            printf("  [CPU] chosen: sha=%s ec=%s, table %s (%.0f MiB, %d windows, built in %.2f s)\n",
                   bsha == 2 ? "sha-ni" : bsha == 1 ? "avx2x8" : "ref",
                   bec == 2 ? "avx2x4" : bec == 1 ? "mulx" : "c", S->lay.name,
                   (double)S->table_bytes / 1048576.0, S->lay.nwin, S->t_build);
        }
    }
    while (S->ec_mode.load() < 0) { if (S->stop.load()) return NULL; usleep(1000); }
    const int ecm = S->ec_mode.load();
    while (!S->stop.load(std::memory_order_relaxed)) {
        if (id >= S->allowed.load(std::memory_order_relaxed)) { usleep(5000); continue; }
        S->running.fetch_add(1);
        const uint64_t c0 = thread_cpu_ns();
        const uint64_t r0 = __rdtsc();
        int n = fill_batch(w);
        const uint64_t r1 = __rdtsc();
        if (n) run_ec(w, ecm, vs, ss);
        const uint64_t r2 = __rdtsc();
        S->sha_cyc.fetch_add(r1 - r0, std::memory_order_relaxed); S->ec_cyc.fetch_add(r2 - r1, std::memory_order_relaxed);
        S->busy_ns[id].fetch_add(thread_cpu_ns() - c0, std::memory_order_relaxed);
        S->running.fetch_sub(1);
        if (!n) break;
        S->cand_done.fetch_add((uint64_t)n, std::memory_order_relaxed);
    }
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
static double mem_available_mib() {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256]; double v = -1;
    while (fgets(line, sizeof line, f)) { long long kb; if (sscanf(line, "MemAvailable: %lld kB", &kb) == 1) { v = kb / 1024.0; break; } }
    fclose(f);
    /* a cgroup memory limit (v2 memory.max / v1 memory.limit_in_bytes) can be tighter than
     * the host's MemAvailable */
    static const char *lim_f[2] = {"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"};
    static const char *use_f[2] = {"/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"};
    for (int k = 0; k < 2; k++) {
        FILE *g = fopen(lim_f[k], "r");
        if (!g) continue;
        char q[64] = {0};
        if (fscanf(g, "%63s", q) == 1 && strcmp(q, "max") != 0) {
            const double lim = atof(q) / 1048576.0;
            double used = -1;
            FILE *u = fopen(use_f[k], "r");
            if (u) { long long cur; if (fscanf(u, "%lld", &cur) == 1) used = cur / 1048576.0; fclose(u); }
            /* v1 reports "no limit" as a huge number; ignore anything beyond 1 PiB */
            if (lim < 1073741824.0 && used >= 0 && (v < 0 || lim - used < v)) v = lim - used;
        }
        fclose(g);
    }
    return v;
}

/* ---------------- controller (called from the GPU host loop) ---------------- */
struct ctl_t {
    int wmax, cur;
    int phase;              /* 0 warm-up, 1 cpu-share check, 2 steady; 3 A/B off-window */
    double t_phase;
    double last_done;
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

/* Start: pick the table layout, build it in the background and spawn the (paused) workers. */
static int start(const pinning2_params_t *pp, uint32_t lt_min, uint32_t lt_max) {
#ifndef QSB_COGRIND
#define QSB_COGRIND 1
#endif
    if (!QSB_COGRIND) return 0;
    const char *env = getenv("QSB_COGRIND_THREADS");
    int ncpu = 0;
    { cpu_set_t cs; CPU_ZERO(&cs); if (sched_getaffinity(0, sizeof cs, &cs) == 0) ncpu = CPU_COUNT(&cs); }
    if (ncpu <= 0) ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    double quota = cgroup_quota_cpus();
    /* Without a CPU quota, SCHED_IDLE workers yield every CPU the GPU host thread wants, so all
     * CPUs but one (that thread's) are used. Under a cgroup quota SCHED_IDLE does not help --
     * throttling stops the GPU thread too -- so the workers leave ~1.5 CPUs of the quota free. */
    int nw;
    /* SMT guard: pin the GPU host thread (the caller) to the CPU it is on and keep the workers off
     * that CPU and its hyperthread siblings, so no worker shares a core with the GPU driver thread
     * (SCHED_IDLE only yields the same logical CPU, not its sibling). Fallback: previous behaviour. */
    int guard_ok = 0;
    {
        cpu_set_t all; CPU_ZERO(&all);
        const int hc = sched_getcpu();
        if (hc >= 0 && sched_getaffinity(0, sizeof all, &all) == 0 && CPU_ISSET(hc, &all)) {
            cpu_set_t ws = all; CPU_CLR(hc, &ws);
            char path[128]; snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", hc);
            FILE *sf = fopen(path, "r");
            if (sf) {
                char buf[256] = {0};
                if (fgets(buf, sizeof buf, sf)) {
                    for (char *tok = strtok(buf, ",\n"); tok; tok = strtok(NULL, ",\n")) {
                        int a = -1, b = -1;
                        if (sscanf(tok, "%d-%d", &a, &b) == 2) { for (int c = a; c <= b && c < CPU_SETSIZE; c++) if (c >= 0) CPU_CLR(c, &ws); }
                        else if (sscanf(tok, "%d", &a) == 1 && a >= 0 && a < CPU_SETSIZE) CPU_CLR(a, &ws);
                    }
                }
                fclose(sf);
            }
            if (CPU_COUNT(&ws) >= 1) {
                cpu_set_t hs; CPU_ZERO(&hs); CPU_SET(hc, &hs);
                if (pthread_setaffinity_np(pthread_self(), sizeof hs, &hs) == 0) { g_worker_set = ws; g_worker_set_on = 1; guard_ok = 1; }
            }
        }
    }
    if (quota > 0 && quota < ncpu) nw = (int)ceil(quota) - 2;
    else nw = guard_ok ? CPU_COUNT(&g_worker_set) : ncpu - 1;
    if (nw < 0) nw = 0;
    if (env) nw = atoi(env);
    if (nw > QSB_CG_MAXW) nw = QSB_CG_MAXW;
    const int has_avx2 = __builtin_cpu_supports("avx2"), has_bmi2 = __builtin_cpu_supports("bmi2");
    int has_adx = 0;
    { unsigned r[4] = {0, 0, 0, 0}; __cpuid_count(7, 0, r[0], r[1], r[2], r[3]); has_adx = (r[1] >> 19) & 1; }
    const int has_sha = __builtin_cpu_supports("sha") && __builtin_cpu_supports("sse4.1");
    /* table layout from the memory budget */
    const double avail = mem_available_mib();
    layout_t L;
    const char *lenv = getenv("QSB_COGRIND_TABLE");
    int lok = -1;
    if (lenv) lok = layout_by_name(&L, lenv);
    if (lok != 0) {
        if (avail >= 16384) lok = layout_by_name(&L, "xlarge");
        else if (avail < 0 || avail >= 6144) lok = layout_by_name(&L, "large");
        else if (avail >= 2048) lok = layout_by_name(&L, "medium");
        else if (avail >= 900) lok = layout_by_name(&L, "small");
        else if (avail >= 500) lok = layout_by_name(&L, "tiny");
    }
    printf("  CPU co-grind: %d CPUs in affinity, cgroup quota %s%.2f, %d workers%s%s%s%s, mem avail %.0f MiB\n",
           ncpu, quota > 0 ? "" : "none ", quota > 0 ? quota : 0.0, nw > 0 ? nw : 0, guard_ok ? ", smt-guard" : "",
           has_avx2 ? ", avx2" : "", has_adx && has_bmi2 ? ", adx" : "", has_sha ? ", sha-ni" : "", avail);
    if (nw <= 0 || lok != 0) return 0;
    if (pp->suffix_len > 119 || pp->seq_offset + 4 > pp->suffix_len || pp->lt_offset + 4 > pp->suffix_len || lt_max <= lt_min) return 0;

    shared_t *S = new shared_t();
    g_cg = S;
    S->pp = pp; S->lt_min = lt_min; S->lt_range = lt_max - lt_min;
    S->lay = L;
    S->chunks_per_seq = (S->lt_range + QSB_CG_B - 1) / QSB_CG_B;
    S->n_chunks = (uint64_t)S->chunks_per_seq * 0x3FFFFFFFull;      /* sequences 0xFFFFFFFE down to 0xC0000000 */
    S->nblk = pp->suffix_len < 56 ? 1 : 2;
    S->cache_first = S->nblk == 2 && pp->seq_offset + 4 <= 64 && pp->lt_offset >= 64;
    S->has_avx2 = has_avx2; S->has_adx = has_adx && has_bmi2; S->has_sha = has_sha;
    g_fe_asm = S->has_adx;
    S->ec_env = -1; S->sha_env = -1;
    if (getenv("QSB_COGRIND_EC")) {
        const char *e = getenv("QSB_COGRIND_EC");
        S->ec_env = !strcmp(e, "avx2") ? 2 : !strcmp(e, "mulx") ? 1 : 0;
        if ((S->ec_env == 2 && !(has_avx2 && QSB_CG_HAVE_SIMD)) || (S->ec_env == 1 && !S->has_adx)) S->ec_env = 0;
    }
    if (getenv("QSB_COGRIND_SHA")) {
        const char *e = getenv("QSB_COGRIND_SHA");
        S->sha_env = !strcmp(e, "sha-ni") ? 2 : !strcmp(e, "avx2") ? 1 : 0;
        if ((S->sha_env == 2 && !has_sha) || (S->sha_env == 1 && !has_avx2)) S->sha_env = 0;
    }
    if (!S->cache_first) S->sha_env = 0;   /* non-standard layout: generic scalar SHA */
    S->ec_mode.store(-1); S->sha_mode.store(0);
    /* suffix block 1 template and the locktime byte positions */
    if (S->cache_first) {
        uint8_t m[128]; memset(m, 0, 128);
        memcpy(m, pp->suffix, pp->suffix_len);
        for (int b = 0; b < 4; b++) m[pp->lt_offset + b] = 0;
        m[pp->suffix_len] = 0x80;
        const uint64_t bits = (uint64_t)pp->total_preimage_len * 8;
        for (int b = 0; b < 8; b++) m[120 + 7 - b] = (uint8_t)(bits >> (8 * b));
        for (int i = 0; i < 16; i++) S->w1_tmpl[i] = (uint32_t)m[64 + 4 * i] << 24 | (uint32_t)m[64 + 4 * i + 1] << 16 | (uint32_t)m[64 + 4 * i + 2] << 8 | m[64 + 4 * i + 3];
        for (int b = 0; b < 4; b++) { const int p = (int)pp->lt_offset + b - 64; S->lt_word[b] = p >> 2; S->lt_shift[b] = 8 * (3 - (p & 3)); }
    }
    mkdir("results", 0755);
    S->hit_fd = open("results/pinning_hit_cpu.txt", O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (S->hit_fd < 0) { g_cg = NULL; return 0; }
    if (table_start(S, nw) != 0) { g_cg = NULL; return 0; }
    recode_init(S->lay);
    S->allowed.store(0);
    S->nworkers = nw;
    for (int i = 0; i < nw; i++) {
        pthread_t t;
        if (pthread_create(&t, NULL, worker_main, (void *)(intptr_t)i) != 0) { S->nworkers = i; break; }
        pthread_detach(t);
    }
    memset(&g_ctl, 0, sizeof g_ctl);
    g_ctl.wmax = S->nworkers; g_ctl.cur = 0;
    g_ctl.verbose = g_ctl_verbose = getenv("QSB_COGRIND_VERBOSE") != NULL;
    return S->nworkers;
}

static void set_allowed(int n) { if (g_cg) g_cg->allowed.store(n, std::memory_order_relaxed); g_ctl.cur = n; }

/* Called by the GPU host loop after every drained GPU batch of gpu_batch candidates.
 * (Unchanged logic from the v1 co-grinder.) */
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
    if (C.phase == 0) {
        set_allowed(C.wmax);
        C.phase = 1; C.t_phase = now;
        C.busy0 = busy_total(); C.busy_t0 = now;
        C.cand0 = S->cand_done.load(); C.cand_t0 = now;
        C.next_ab = now + 20.0;
        return;
    }
    if (C.phase == 1 && now - C.t_phase >= 2.0) {
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
        C.phase = 3; C.ab_left = 4; C.ab_on_sum = C.ab_off_sum = 0; C.ab_on_n = C.ab_off_n = 0;
        C.win_t0 = now; C.win_n = 0; C.win_skip = 1;
        return;
    }
    if (C.phase == 3) {
        const int on = (C.ab_left & 1) == 0;
        if (C.win_skip) C.win_skip = 0;
        else { if (on) { C.ab_on_sum += dt; C.ab_on_n++; } else { C.ab_off_sum += dt; C.ab_off_n++; } C.win_n++; }
        if (C.win_n < 3 || now - C.win_t0 < 1.0) return;
        C.ab_left--;
        C.win_t0 = now; C.win_n = 0; C.win_skip = 1;
        if (C.ab_left > 0) { set_allowed(((C.ab_left & 1) == 0) ? C.wmax : 0); return; }
        set_allowed(C.wmax);
        const double on_t = C.ab_on_sum / (C.ab_on_n ? C.ab_on_n : 1);
        const double off_t = C.ab_off_sum / (C.ab_off_n ? C.ab_off_n : 1);
        const double loss = on_t / off_t - 1.0;
        const uint64_t cd = S->cand_done.load();
        const double cpu_rate = C.cand_t0 > 0 && now > C.cand_t0 ? (double)(cd - C.cand0) / (now - C.cand_t0) : 0;
        C.cand0 = cd; C.cand_t0 = now;
        if (C.verbose) printf("  [CPU] A/B: gpu batch on %.5fs off %.5fs (loss %+.3f%%), cpu %.0f cand/s, %d workers, hits %llu/%llu exact\n",
                              on_t, off_t, 100 * loss, cpu_rate, C.wmax,
                              (unsigned long long)S->hits.load(), (unsigned long long)S->tentative.load());
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
