/* pk_offload.h -- host half of QSB_PK_OFFLOAD (see pinning.cu).
 *
 * The finish kernel's leading pk_blocks blocks store, per candidate, the two recovered
 * x-coordinates (four 16-byte planes of little-endian 64-bit limbs) and a parity byte (bit r =
 * y-parity of the recid-r key). The host copies those planes into pinned memory and the
 * workers below compute word 0 of SHA-256(compressed key) for both keys, exactly as
 * _SHA256Pubkey33H0 does on the GPU. A word with QSB_ZEROS_N leading zero bits is a tentative
 * hit; it is published only after qsb_host_exact_hit re-derives the candidate from
 * (sequence, locktime, recid) with OpenSSL, and only once (process-wide dedupe set).
 *
 * Self-test: a batch submitted in verify mode also carries the GPU's own H0 words for the
 * same candidates. The workers compare every word and publish nothing (the GPU published
 * those blocks' hits itself). Any mismatch switches the offload off for the rest of the run.
 *
 * Engines (chosen at start-up by CPU support and a short timing): x86 SHA extensions with
 * four independent messages interleaved, AVX2 with eight messages in 32-bit lanes, or a
 * portable scalar compression. Host-only; SIMD code uses GCC target attributes so the
 * default nvcc host flags suffice.
 */
#pragma once
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <pthread.h>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <unordered_set>
#include <x86intrin.h>
#include <cpuid.h>

namespace pko {

static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
static const uint32_t IV256[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                  0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};

struct u128 { uint64_t lo, hi; };

/* The 16 message words of the compressed key (prefix 0x02|par, then x big-endian, then
 * padding for 33 bytes = 264 bits), the same words pb[0..8] the finish kernel builds. */
static inline void key_words(uint32_t w[16], uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3,
                             unsigned par) {
    const uint32_t x[8] = {(uint32_t)l0, (uint32_t)(l0 >> 32), (uint32_t)l1, (uint32_t)(l1 >> 32),
                           (uint32_t)l2, (uint32_t)(l2 >> 32), (uint32_t)l3, (uint32_t)(l3 >> 32)};
    w[0] = ((0x02u | (par & 1u)) << 24) | (x[7] >> 8);
    for (int k = 1; k < 8; k++) w[k] = (x[8 - k] << 24) | (x[7 - k] >> 8);
    w[8] = (x[0] << 24) | 0x00800000u;
    for (int k = 9; k < 15; k++) w[k] = 0;
    w[15] = 264u;
}

static inline uint32_t ror32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

/* Portable reference: word 0 of the digest of one block from the IV. */
static inline uint32_t h0_scalar(const uint32_t m[16]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = m[i];
    for (int i = 16; i < 64; i++) {
        const uint32_t s0 = ror32(w[i-15], 7) ^ ror32(w[i-15], 18) ^ (w[i-15] >> 3);
        const uint32_t s1 = ror32(w[i-2], 17) ^ ror32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = IV256[0], b = IV256[1], c = IV256[2], d = IV256[3];
    uint32_t e = IV256[4], f = IV256[5], g = IV256[6], h = IV256[7];
    for (int i = 0; i < 64; i++) {
        const uint32_t S1 = ror32(e, 6) ^ ror32(e, 11) ^ ror32(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = h + S1 + ch + K256[i] + w[i];
        const uint32_t S0 = ror32(a, 2) ^ ror32(a, 13) ^ ror32(a, 22);
        const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    return a + IV256[0];
}

/* ---- x86 SHA extensions, four messages interleaved ---- */
#define PKO_SHANI __attribute__((target("sha,sse4.1,ssse3")))
PKO_SHANI static void h0_shani_x4(const uint32_t (*m)[16], uint32_t out[4]) {
    __m128i S0[4], S1[4], M[4][4];
    const __m128i t0 = _mm_shuffle_epi32(_mm_loadu_si128((const __m128i *)&IV256[0]), 0xB1);
    const __m128i u0 = _mm_shuffle_epi32(_mm_loadu_si128((const __m128i *)&IV256[4]), 0x1B);
    const __m128i A0 = _mm_alignr_epi8(t0, u0, 8), B0 = _mm_blend_epi16(u0, t0, 0xF0);
    for (int l = 0; l < 4; l++) {
        S0[l] = A0; S1[l] = B0;
        for (int j = 0; j < 4; j++) M[l][j] = _mm_loadu_si128((const __m128i *)&m[l][4 * j]);
    }
    for (int r = 0; r < 16; r++) {
        const __m128i K = _mm_loadu_si128((const __m128i *)&K256[4 * r]);
        for (int l = 0; l < 4; l++) {
            if (r >= 4) {
                __m128i t = _mm_sha256msg1_epu32(M[l][r & 3], M[l][(r + 1) & 3]);
                t = _mm_add_epi32(t, _mm_alignr_epi8(M[l][(r + 3) & 3], M[l][(r + 2) & 3], 4));
                M[l][r & 3] = _mm_sha256msg2_epu32(t, M[l][(r + 3) & 3]);
            }
            __m128i k = _mm_add_epi32(M[l][r & 3], K);
            S1[l] = _mm_sha256rnds2_epu32(S1[l], S0[l], k);
            k = _mm_shuffle_epi32(k, 0x0E);
            S0[l] = _mm_sha256rnds2_epu32(S0[l], S1[l], k);
        }
    }
    /* S0 holds (F, E, B, A) in lanes 0..3: word 0 of the digest is lane 3 plus IV[0]. */
    for (int l = 0; l < 4; l++) out[l] = (uint32_t)_mm_extract_epi32(S0[l], 3) + IV256[0];
}

/* ---- AVX2, eight messages in 32-bit lanes ---- */
#define PKO_AVX2 __attribute__((target("avx2")))
PKO_AVX2 static inline __m256i v_ror(__m256i x, int n) {
    return _mm256_or_si256(_mm256_srli_epi32(x, n), _mm256_slli_epi32(x, 32 - n));
}
PKO_AVX2 static void h0_avx2_x8(const uint32_t (*m)[16], uint32_t out[8]) {
    __m256i w[16];
    for (int i = 0; i < 16; i++)
        w[i] = _mm256_setr_epi32((int)m[0][i], (int)m[1][i], (int)m[2][i], (int)m[3][i],
                                 (int)m[4][i], (int)m[5][i], (int)m[6][i], (int)m[7][i]);
    __m256i a = _mm256_set1_epi32((int)IV256[0]), b = _mm256_set1_epi32((int)IV256[1]);
    __m256i c = _mm256_set1_epi32((int)IV256[2]), d = _mm256_set1_epi32((int)IV256[3]);
    __m256i e = _mm256_set1_epi32((int)IV256[4]), f = _mm256_set1_epi32((int)IV256[5]);
    __m256i g = _mm256_set1_epi32((int)IV256[6]), h = _mm256_set1_epi32((int)IV256[7]);
    for (int i = 0; i < 64; i++) {
        __m256i wi;
        if (i < 16) wi = w[i];
        else {
            const __m256i x15 = w[(i - 15) & 15], x2 = w[(i - 2) & 15];
            const __m256i s0 = _mm256_xor_si256(_mm256_xor_si256(v_ror(x15, 7), v_ror(x15, 18)),
                                                _mm256_srli_epi32(x15, 3));
            const __m256i s1 = _mm256_xor_si256(_mm256_xor_si256(v_ror(x2, 17), v_ror(x2, 19)),
                                                _mm256_srli_epi32(x2, 10));
            wi = _mm256_add_epi32(_mm256_add_epi32(w[i & 15], s0),
                                  _mm256_add_epi32(w[(i - 7) & 15], s1));
            w[i & 15] = wi;
        }
        const __m256i S1 = _mm256_xor_si256(_mm256_xor_si256(v_ror(e, 6), v_ror(e, 11)), v_ror(e, 25));
        const __m256i ch = _mm256_xor_si256(_mm256_and_si256(e, f), _mm256_andnot_si256(e, g));
        const __m256i t1 = _mm256_add_epi32(_mm256_add_epi32(h, S1),
                           _mm256_add_epi32(ch, _mm256_add_epi32(wi, _mm256_set1_epi32((int)K256[i]))));
        const __m256i S0 = _mm256_xor_si256(_mm256_xor_si256(v_ror(a, 2), v_ror(a, 13)), v_ror(a, 22));
        const __m256i mj = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(c, _mm256_or_si256(a, b)));
        const __m256i t2 = _mm256_add_epi32(S0, mj);
        h = g; g = f; f = e; e = _mm256_add_epi32(d, t1); d = c; c = b; b = a; a = _mm256_add_epi32(t1, t2);
    }
    a = _mm256_add_epi32(a, _mm256_set1_epi32((int)IV256[0]));
    _mm256_storeu_si256((__m256i *)out, a);
}

enum Engine { ENG_SCALAR = 0, ENG_SHANI = 1, ENG_AVX2 = 2 };
static const char *engine_name(int e) { return e == ENG_SHANI ? "sha-ni x4" : e == ENG_AVX2 ? "avx2 x8" : "scalar"; }

/* Word 0 of both keys of candidates [i0, i1) into h0[2*(i-i0)+r]. */
static void h0_range(int eng, const u128 *pa, const u128 *pb, const u128 *pc, const u128 *pd,
                     const uint8_t *par, size_t i0, size_t i1, uint32_t *h0) {
    const int L = eng == ENG_SHANI ? 4 : eng == ENG_AVX2 ? 8 : 1;
    alignas(32) uint32_t m[8][16];
    alignas(32) uint32_t o[8];
    size_t n = 2 * (i1 - i0), k = 0;
    while (k < n) {
        int got = 0;
        for (; got < L && k + got < n; got++) {
            const size_t q = k + got, i = i0 + q / 2; const int r = (int)(q & 1);
            const u128 lo = r ? pc[i] : pa[i], hi = r ? pd[i] : pb[i];
            key_words(m[got], lo.lo, lo.hi, hi.lo, hi.hi, (unsigned)(par[i] >> r));
        }
        for (int z = got; z < L; z++) memcpy(m[z], m[0], sizeof m[0]);
        if (eng == ENG_SHANI) h0_shani_x4(m, o);
        else if (eng == ENG_AVX2) h0_avx2_x8(m, o);
        else o[0] = h0_scalar(m[0]);
        for (int z = 0; z < got; z++) h0[k + z] = o[z];
        k += (size_t)got;
    }
}

/* SHA extensions (CPUID.7.0:EBX bit 29) with SSE4.1 (CPUID.1:ECX bit 19); read directly so any
 * GCC version's __builtin_cpu_supports feature list does not matter. */
static bool has_shani() {
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & (1u << 19))) return false;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
    return (b >> 29) & 1u;
}

static double mono_s() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* Choose the fastest engine this CPU supports, after checking it against the scalar path. */
static int pick_engine(double *per_s) {
    const size_t N = 4096;
    std::vector<u128> a(N), b(N), c(N), d(N); std::vector<uint8_t> p(N);
    uint64_t s = 0x9e3779b97f4a7c15ULL;
    auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    for (size_t i = 0; i < N; i++) { a[i] = {rnd(), rnd()}; b[i] = {rnd(), rnd()}; c[i] = {rnd(), rnd()}; d[i] = {rnd(), rnd()}; p[i] = (uint8_t)rnd(); }
    std::vector<uint32_t> ref(2 * N), got(2 * N);
    h0_range(ENG_SCALAR, a.data(), b.data(), c.data(), d.data(), p.data(), 0, N, ref.data());
    int best = ENG_SCALAR; double best_rate = 0;
    const int cand[3] = {ENG_SCALAR, ENG_SHANI, ENG_AVX2};
    for (int e : cand) {
        if (e == ENG_SHANI && !has_shani()) continue;
        if (e == ENG_AVX2 && !__builtin_cpu_supports("avx2")) continue;
        h0_range(e, a.data(), b.data(), c.data(), d.data(), p.data(), 0, N, got.data());
        if (memcmp(ref.data(), got.data(), 4 * 2 * N) != 0) continue;   /* never use a disagreeing engine */
        const double t0 = mono_s();
        for (int rep = 0; rep < 4; rep++) h0_range(e, a.data(), b.data(), c.data(), d.data(), p.data(), 0, N, got.data());
        const double rate = 4.0 * N / (mono_s() - t0);                  /* candidates per second, one thread */
        if (rate > best_rate) { best_rate = rate; best = e; }
    }
    if (per_s) *per_s = best_rate;
    return best;
}

/* ---------------- work items and the worker pool ---------------- */
struct Item {
    const uint8_t *buf;      /* pinned host copy of the device planes (same layout, capacity cap) */
    uint32_t cap, n, seq, lt;
    int verify;
    std::atomic<uint32_t> next{0}, done{0};
    uint32_t nchunks;
    std::atomic<int> *release;   /* set to 0 when every chunk is done */
    std::atomic<uint64_t> mismatch{0};
};

typedef int (*gate_fn)(uint32_t seq, uint32_t lt, int recid, void *tls);

struct Pool {
    int eng = ENG_SCALAR, nworkers = 0, zeros = 24;
    std::mutex mu; std::condition_variable cv;
    std::deque<Item *> q;
    std::vector<std::thread> th;
    std::atomic<int> stop{0};
    std::atomic<uint64_t> cand_done{0}, busy_ns{0}, tentative{0}, published{0}, gate_rejects{0};
    std::atomic<uint64_t> verified_words{0}, mismatches{0};
    std::mutex pub_mu; std::unordered_set<uint64_t> pub_seen; int hit_fd = -1;
    gate_fn gate = nullptr;
    void *(*tls_new)() = nullptr;
    uint32_t chunk = 8192;
};

/* Publish (seq, lt, recid) once, as one write() so a line is never torn. */
static int publish(Pool *P, uint32_t seq, uint32_t lt, int recid) {
    const uint64_t key = ((uint64_t)seq << 33) | ((uint64_t)lt << 1) | (uint64_t)(recid & 1);
    std::lock_guard<std::mutex> g(P->pub_mu);
    if (!P->pub_seen.insert(key).second) return 0;
    char line[96];
    const int len = snprintf(line, sizeof line, "sequence=%u locktime=%u recid=%d\n", seq, lt, recid);
    if (P->hit_fd >= 0 && len > 0) { ssize_t w = write(P->hit_fd, line, (size_t)len); (void)w; }
    P->published.fetch_add(1);
    return 1;
}

static void run_chunk(Pool *P, Item *it, uint32_t c, void *tls, std::vector<uint32_t> &h0) {
    const uint32_t i0 = c * P->chunk, i1 = i0 + P->chunk < it->n ? i0 + P->chunk : it->n;
    const u128 *pa = (const u128 *)it->buf, *pb = pa + it->cap, *pc = pb + it->cap, *pd = pc + it->cap;
    const uint8_t *par = (const uint8_t *)(pd + it->cap);
    const uint32_t *gpu = (const uint32_t *)(par + it->cap);
    h0.resize(2 * (size_t)(i1 - i0));
    h0_range(P->eng, pa, pb, pc, pd, par, i0, i1, h0.data());
    const uint32_t zmask = P->zeros >= 32 ? 0xFFFFFFFFu : ~(0xFFFFFFFFu >> P->zeros);
    if (it->verify) {
        uint64_t bad = 0, words = 0;
        for (uint32_t i = i0; i < i1; i++) {
            const uint32_t g0 = gpu[2 * i], c0 = h0[2 * (i - i0)];
            bad += g0 != c0; words++;
            if ((g0 & zmask) != 0) {   /* the GPU skips recid 1 after a recid-0 hit */
                bad += gpu[2 * i + 1] != h0[2 * (i - i0) + 1]; words++;
            }
        }
        it->mismatch.fetch_add(bad); P->mismatches.fetch_add(bad); P->verified_words.fetch_add(words);
        return;
    }
    for (uint32_t i = i0; i < i1; i++)
        for (int r = 0; r < 2; r++)
            if ((h0[2 * (i - i0) + r] & zmask) == 0) {
                P->tentative.fetch_add(1);
                const uint32_t lt = it->lt + i;
                if (P->gate(it->seq, lt, r, tls)) publish(P, it->seq, lt, r);
                else P->gate_rejects.fetch_add(1);
            }
}

static void worker(Pool *P) {
    /* Idle class, as the co-grinder: the CUDA host thread always preempts the workers. */
    { sched_param sp; sp.sched_priority = 0;
      if (pthread_setschedparam(pthread_self(), SCHED_IDLE, &sp) != 0) setpriority(PRIO_PROCESS, 0, 19); }
    void *tls = P->tls_new ? P->tls_new() : nullptr;
    std::vector<uint32_t> h0;
    for (;;) {
        Item *it = nullptr; uint32_t c = 0, nch = 0, n = 0; std::atomic<int> *rel = nullptr;
        {
            /* Everything the completion step needs is read while this worker still owns an
             * unfinished chunk; after its done increment it never touches the item again
             * (the item may be released and resubmitted immediately). */
            std::unique_lock<std::mutex> lk(P->mu);
            P->cv.wait(lk, [&] { return P->stop.load() || !P->q.empty(); });
            if (P->stop.load()) return;
            it = P->q.front();
            nch = it->nchunks; n = it->n; rel = it->release;
            c = it->next.fetch_add(1);
            if (c + 1 >= nch) P->q.pop_front();
        }
        if (c >= nch) continue;
        const double t0 = mono_s();
        run_chunk(P, it, c, tls, h0);
        P->busy_ns.fetch_add((uint64_t)((mono_s() - t0) * 1e9));
        const uint32_t i0 = c * P->chunk;
        P->cand_done.fetch_add((n - i0) < P->chunk ? n - i0 : P->chunk);
        if (it->done.fetch_add(1, std::memory_order_acq_rel) + 1 == nch) {
            if (rel) rel->store(0, std::memory_order_release);
        }
    }
}

static void submit(Pool *P, Item *it) {
    it->nchunks = (it->n + P->chunk - 1) / P->chunk;
    it->next.store(0); it->done.store(0); it->mismatch.store(0);
    if (it->nchunks == 0) { if (it->release) it->release->store(0); return; }
    { std::lock_guard<std::mutex> g(P->mu); P->q.push_back(it); }
    P->cv.notify_all();
}

static int start(Pool *P, int nworkers) {
    P->nworkers = nworkers;
    for (int i = 0; i < nworkers; i++) {
        try { std::thread t(worker, P); t.detach(); } catch (...) { P->nworkers = i; break; }
    }
    return P->nworkers;
}

}  // namespace pko
