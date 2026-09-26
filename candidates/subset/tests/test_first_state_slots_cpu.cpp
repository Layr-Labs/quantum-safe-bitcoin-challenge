// SPDX-License-Identifier: GPL-3.0-only
// Focused host-only check for the live short-epoch schedule and producer path.
// The production headers are included below; the CUDA symbols are only stubs so
// that the actual host code can run without a GPU.  The test is compiled twice
// by run_first_state_slots_cpu.sh (QSB_SE_WINDOWS=128 and 256).
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <openssl/sha.h>

#define SIG_PUSH_SIZE 10
#define QSB_SE_PER_EPOCH QSB_SE_WINDOWS
#define QSB_SE_TWIN 3
#define QSB_SE_CUT 137
#define QSB_SE_EARLY 6
#define QSB_SHA_UNROLL_CONST 0
#define ZLAB_DUAL_EPOCH_SHA 0

/* Minimal CUDA vocabulary needed to parse the production schedule/producer headers. */
#define __device__
#define __constant__
#define __global__
#define __forceinline__ inline
#define __launch_bounds__(...)

struct CudaIndex { unsigned x = 0; };
static CudaIndex blockIdx, blockDim, threadIdx;
struct uint4 { uint32_t x, y, z, w; };

enum cudaError_t { cudaSuccess = 0, cudaErrorNotReady = 34 };
using cudaStream_t = void *;
using cudaEvent_t = void *;
enum cudaMemcpyKind {
    cudaMemcpyHostToHost,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
};
static constexpr unsigned cudaEventDisableTiming = 2;
static constexpr unsigned cudaHostAllocPortable = 1;

struct Copy2DTrace {
    bool enabled = false;
    size_t calls = 0;
    size_t expected_dpitch = 0, expected_spitch = 0, expected_width = 0, expected_height = 0;
    const uint8_t *dst_begin = nullptr, *dst_end = nullptr;
    const uint8_t *src_begin = nullptr, *src_end = nullptr;
};
static Copy2DTrace g_copy2d;

static inline cudaError_t cudaGetLastError() { return cudaSuccess; }
static inline cudaError_t cudaGetDevice(int *d) { if (d) *d = 0; return cudaSuccess; }
static inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
static inline const char *cudaGetErrorString(cudaError_t) { return "cpu test stub"; }
static inline cudaError_t cudaEventQuery(cudaEvent_t) { return cudaSuccess; }
static inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
static inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t *e, unsigned) { if (e) *e = nullptr; return cudaSuccess; }
static inline cudaError_t cudaMemcpy(void *dst, const void *src, size_t n, cudaMemcpyKind) {
    if (n) std::memcpy(dst, src, n);
    return cudaSuccess;
}
static inline cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t n, cudaMemcpyKind, cudaStream_t = nullptr) {
    return cudaMemcpy(dst, src, n, cudaMemcpyHostToHost);
}
static inline cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
                                            size_t width, size_t height, cudaMemcpyKind, cudaStream_t = nullptr) {
    if (g_copy2d.enabled) {
        assert(dpitch == g_copy2d.expected_dpitch);
        assert(spitch == g_copy2d.expected_spitch);
        assert(width == g_copy2d.expected_width);
        assert(height == g_copy2d.expected_height);
        const uint8_t *d = (const uint8_t *)dst, *s = (const uint8_t *)src;
        const size_t dst_last = height ? (height - 1) * dpitch + width : 0;
        const size_t src_last = height ? (height - 1) * spitch + width : 0;
        assert(d >= g_copy2d.dst_begin && d + dst_last <= g_copy2d.dst_end);
        assert(s >= g_copy2d.src_begin && s + src_last <= g_copy2d.src_end);
        g_copy2d.calls++;
    }
    for (size_t y = 0; y < height; y++) std::memcpy((uint8_t *)dst + y * dpitch, (const uint8_t *)src + y * spitch, width);
    return cudaSuccess;
}
static inline cudaError_t cudaMalloc(void **p, size_t n) { *p = std::malloc(n ? n : 1); return *p ? cudaSuccess : cudaErrorNotReady; }
static inline cudaError_t cudaHostAlloc(void **p, size_t n, unsigned) { *p = std::malloc(n ? n : 1); return *p ? cudaSuccess : cudaErrorNotReady; }

/* Support types and helpers used by the production headers. */
typedef struct {
    uint32_t midstate[8];
    uint8_t prefix_remainder[64];
    uint32_t prefix_remainder_len;
    uint8_t *dummy_sigs;
} digest_params_t;
typedef struct {
    uint32_t mid[8];
    uint32_t remW[2];
    uint8_t early[QSB_SE_EARLY];
    uint8_t pad[64 - 8 * 4 - 2 * 4 - QSB_SE_EARLY];
} epoch_desc_t;

static inline uint32_t qsb_host_rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static void qsb_host_unrank(uint64_t rank, int n, int t, uint8_t *out);

/* The schedule header's device functions are not called by this test. */
static uint32_t K[64];
static const uint32_t SHA256_K_CANONICAL[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};
static inline void test_from_symbol(void *dst, const void *src, size_t n) { std::memcpy(dst, src, n); }
#define QSB_FROM_SYMBOL(dst, sym, n) (test_from_symbol((dst), &(sym), (n)), cudaSuccess)
#define QSB_TO_SYMBOL(sym, src, n) (test_from_symbol(&(sym), (src), (n)), cudaSuccess)
#define _SHA256Transform(st, words) do { (st)[0] ^= (words)[0]; } while (0)
#define S2Round(a,b,c,d,e,f,g,h,unused,w) do { (a) ^= (w); } while (0)
template <int B> static inline void qsb_compress_constant(uint32_t *) {}
static inline void qsb_compress_constant_rolled(uint32_t *) {}

#include "gpu_epochs/window_schedule_shared.cuh"
#include "gpu_epochs/host_producers.h"

static uint64_t test_binom(int n, int k) {
    if (k < 0 || n < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    uint64_t r = 1;
    for (int i = 0; i < k; i++) r = r * (uint64_t)(n - i) / (uint64_t)(i + 1);
    return r;
}

/* Ranking support is a fixture dependency; the producer itself is the production qhp::produce(). */
static void qsb_host_unrank(uint64_t rank, int n, int t, uint8_t *out) {
    int lo = 0;
    for (int i = 0; i < t; i++) {
        int c = lo;
        for (;;) {
            uint64_t cnt = test_binom(n - c - 1, t - i - 1);
            if (rank < cnt) break;
            rank -= cnt;
            c++;
        }
        out[i] = (uint8_t)c;
        lo = c + 1;
    }
}

static std::vector<std::array<uint8_t, 3>> selected_windows() {
    std::vector<std::array<uint8_t, 3>> out;
    for (int a = 0; a < 13; a++) for (int b = a + 1; b < 13; b++) for (int c = b + 1; c < 13; c++) {
#if QSB_SE_WINDOWS == 256
        if (a >= 1 && c <= 7 && !(a == 1 && b == 2)) continue;
#elif QSB_SE_WINDOWS == 128
        if (!((a >= 6) || (a <= 5 && b >= 7) || (a == 0 && b == 1 && c >= 8 && c <= 10))) continue;
#else
#error "QSB_SE_WINDOWS must be 128 or 256"
#endif
        out.push_back({(uint8_t)(QSB_SE_CUT + a), (uint8_t)(QSB_SE_CUT + b), (uint8_t)(QSB_SE_CUT + c)});
    }
    auto key = [](const std::array<uint8_t, 3> &w, bool second) {
        uint32_t k = 0;
        if (second) {
            for (int i = 12, n = 0; i >= 0 && n < 5; i--)
                if (i != w[0] - QSB_SE_CUT && i != w[1] - QSB_SE_CUT && i != w[2] - QSB_SE_CUT) { k = (k << 4) | (uint32_t)i; n++; }
        } else {
            for (int i = 0, n = 0; i < 13 && n < 6; i++)
                if (i != w[0] - QSB_SE_CUT && i != w[1] - QSB_SE_CUT && i != w[2] - QSB_SE_CUT) { k = (k << 4) | (uint32_t)i; n++; }
        }
        return k;
    };
    std::stable_sort(out.begin(), out.end(), [&](const auto &x, const auto &y) {
        const uint32_t xs = key(x, true), ys = key(y, true);
        return xs != ys ? xs < ys : key(x, false) < key(y, false);
    });
    assert(out.size() == QSB_SE_WINDOWS);
    return out;
}

static void fill_rows(std::vector<uint8_t> &rows, bool equal) {
    rows.resize(150 * SIG_PUSH_SIZE);
    for (int i = 0; i < 150; i++) for (int j = 0; j < SIG_PUSH_SIZE; j++)
        rows[(size_t)i * SIG_PUSH_SIZE + j] = equal ? 0x5a : (uint8_t)((i * 29 + j * 17 + 3) & 0xff);
}

static void fill_constants(uint32_t constant[5]) {
    for (int i = 0; i < 5; i++) constant[i] = 0x10203040u + (uint32_t)i * 0x11111111u;
}

static void init_params(qhp::Params &p, const std::vector<uint8_t> &rows, int ncls) {
    assert(ncls >= 1 && ncls <= qhp::NCLS);
    std::memset(&p, 0, sizeof(p));
    p.rows = rows.data();
    p.cut = QSB_SE_CUT;
    p.K = QSB_SE_EARLY;
    p.ncls = ncls;
    p.n_epochs = 64;
    p.cap = 64;
    for (int i = 0; i < 8; i++) p.c0.st[i] = 0x6a09e667u + (uint32_t)i * 0x11111111u;
    uint8_t prefix[42];
    for (int i = 0; i < (int)sizeof(prefix); i++) prefix[i] = (uint8_t)(0xa0 + i);
    p.c0.len = 0;
    qhp::sc_bytes(p.c0, prefix, (int)sizeof(prefix));
    for (int c = 0; c < ncls; c++) {
        const uint32_t *w = qsb_first_unique_host[c];
        p.cv[c].w2 = w[0];
        p.cv[c].w3 = w[1];
        p.cv[c].M1 = _mm_set_epi32((int)w[5], (int)w[4], (int)w[3], (int)w[2]);
        p.cv[c].M2 = _mm_set_epi32((int)w[9], (int)w[8], (int)w[7], (int)w[6]);
        p.cv[c].M3 = _mm_set_epi32((int)w[13], (int)w[12], (int)w[11], (int)w[10]);
    }
}

static void store_be32(uint8_t *p, uint32_t w) {
    p[0] = (uint8_t)(w >> 24); p[1] = (uint8_t)(w >> 16);
    p[2] = (uint8_t)(w >> 8); p[3] = (uint8_t)w;
}

/* Independent oracle for the live first-block states.  The producer's first_sw()
 * also calls OpenSSL, but this path rebuilds the block from the emitted epoch
 * descriptor and the schedule's class words before invoking SHA256_Transform. */
static void sha256_first_oracle(const epoch_desc_t &epoch, int cls, uint32_t out[8]) {
    SHA256_CTX c;
    std::memset(&c, 0, sizeof(c));
    for (int i = 0; i < 8; i++) c.h[i] = epoch.mid[i];
    uint8_t block[64] = {};
    store_be32(block, epoch.remW[0]);
    store_be32(block + 4, epoch.remW[1]);
    for (int j = 0; j < 14; j++) store_be32(block + 8 + 4 * j, qsb_first_unique_host[cls][j]);
    SHA256_Transform(&c, block);
    std::memcpy(out, c.h, sizeof(uint32_t) * 8);
}

static void verify_guards(const std::vector<uint8_t> &buf, size_t guard, uint8_t value) {
    for (size_t i = 0; i < guard; i++) {
        assert(buf[i] == value);
        assert(buf[buf.size() - guard + i] == value);
    }
}

static void verify_word_guards(const std::vector<uint32_t> &buf, size_t guard, uint32_t value) {
    for (size_t i = 0; i < guard; i++) {
        assert(buf[i] == value);
        assert(buf[buf.size() - guard + i] == value);
    }
}

static void verify_producer_and_upload(const std::vector<uint8_t> &rows, int ncls) {
    const int candidate_slots = QSB_FIRST_SLOTS;
    assert(ncls > 0 && ncls <= candidate_slots && ncls <= qhp::NCLS);
    for (int lane = 0; lane < QSB_SE_PER_EPOCH; lane++) {
        assert(QSB_FIRST_CLASS[lane] >= 0 && QSB_FIRST_CLASS[lane] < ncls);
        const uint32_t rec = QSB_LANE_CLASS[lane];
        assert((rec >> 16) < (uint32_t)candidate_slots);
        assert((rec >> 16) < (uint32_t)ncls);
    }

    qhp::Params p;
    init_params(p, rows, ncls);
    qhp::g_shani = false;  // force the exact scalar/OpenSSL producer path on every CPU.
    qhp::g_s16.store(false, std::memory_order_relaxed);
    const size_t epochs = 37, words_per_epoch = (size_t)ncls * 8;
    constexpr size_t guard_bytes = 32, guard_words = guard_bytes / sizeof(uint32_t);
    const size_t ep_bytes = epochs * 64, fi_words = epochs * words_per_epoch;
    const uint8_t ep_guard = 0xa7;
    const uint32_t fi_guard = 0xd3e4f5a6u;
    std::vector<uint8_t> ep_storage(guard_bytes + ep_bytes + guard_bytes, ep_guard);
    std::vector<uint32_t> fi_storage(guard_words + fi_words + guard_words, fi_guard);
    uint8_t *ep_out = ep_storage.data() + guard_bytes;
    uint32_t *fi_out = fi_storage.data() + guard_words;
    qhp::produce(p, 0, 0, epochs, ep_out, fi_out);

    for (size_t e = 0; e < epochs; e++) {
        epoch_desc_t epoch;
        std::memcpy(&epoch, ep_out + e * sizeof(epoch_desc_t), sizeof(epoch));
        for (int c = 0; c < ncls; c++) {
            uint32_t expected[8];
            sha256_first_oracle(epoch, c, expected);
            assert(std::memcmp(expected, fi_out + e * words_per_epoch + (size_t)c * 8, sizeof(expected)) == 0);
        }
    }
    verify_guards(ep_storage, guard_bytes, ep_guard);
    verify_word_guards(fi_storage, guard_words, fi_guard);

    const size_t fi_pitch_words = (size_t)candidate_slots * 8;
    const size_t fi_pitch_bytes = fi_pitch_words * sizeof(uint32_t);
    const uint8_t dst_ep_guard = 0x5c;
    const uint32_t dst_fi_guard = 0x8b9cadbeu;
    const size_t dst_fi_words = epochs * fi_pitch_words;
    std::vector<uint8_t> dst_ep_storage(guard_bytes + ep_bytes + guard_bytes, dst_ep_guard);
    std::vector<uint32_t> dst_fi_storage(guard_words + dst_fi_words + guard_words, dst_fi_guard);
    uint8_t *dst_ep = dst_ep_storage.data() + guard_bytes;
    uint32_t *dst_fi = dst_fi_storage.data() + guard_words;

    qhp::Hp h;
    h.P.ncls = ncls;
    h.pe = epochs;
    qhp::Slot slot;
    slot.ep[0] = ep_out;
    slot.fi[0] = fi_out;
    slot.npieces_ok = 1;
    slot.batch = 0;
    slot.state = qhp::S_READY;
    qhp::g_hp = &h;
    g_copy2d = {};
    g_copy2d.enabled = true;
    g_copy2d.expected_dpitch = fi_pitch_bytes;
    g_copy2d.expected_spitch = words_per_epoch * sizeof(uint32_t);
    g_copy2d.expected_width = words_per_epoch * sizeof(uint32_t);
    g_copy2d.expected_height = epochs;
    g_copy2d.dst_begin = reinterpret_cast<const uint8_t *>(dst_fi);
    g_copy2d.dst_end = g_copy2d.dst_begin + epochs * fi_pitch_bytes;
    g_copy2d.src_begin = reinterpret_cast<const uint8_t *>(fi_out);
    g_copy2d.src_end = g_copy2d.src_begin + fi_words * sizeof(uint32_t);
    assert(qhp::upload(&slot, nullptr, dst_ep, dst_fi, fi_pitch_bytes, (int)epochs) == cudaSuccess);
    g_copy2d.enabled = false;
    qhp::g_hp = nullptr;
    assert(slot.state == qhp::S_INFLIGHT);
    assert(g_copy2d.calls == 1);

    assert(std::memcmp(dst_ep, ep_out, ep_bytes) == 0);
    for (size_t e = 0; e < epochs; e++) {
        assert(std::memcmp(dst_fi + e * fi_pitch_words, fi_out + e * words_per_epoch,
                           words_per_epoch * sizeof(uint32_t)) == 0);
        for (size_t w = words_per_epoch; w < fi_pitch_words; w++) assert(dst_fi[e * fi_pitch_words + w] == dst_fi_guard);
    }
    verify_guards(dst_ep_storage, guard_bytes, dst_ep_guard);
    verify_word_guards(dst_fi_storage, guard_words, dst_fi_guard);
    std::printf("producer oracle/upload: windows=%d classes=%d epochs=%zu pitch=%zu bytes PASS\n",
                QSB_SE_WINDOWS, ncls, epochs, fi_pitch_bytes);
}

static void verify_host_fallback(int ncls) {
    assert(ncls > qhp::NCLS && ncls <= QSB_FIRST_SLOTS);
    digest_params_t dp = {};
    dp.prefix_remainder_len = 0;
    qhp::g_hp = nullptr;
    qhp::start(&dp, QSB_SE_CUT, QSB_SE_EARLY, ncls, 37,
               (uint64_t)qhp::NPIECE * (uint64_t)qhp::CHUNK);
    uint64_t host = 0, fallback = 0;
    int state = 0;
    qhp::stats(&host, &fallback, &state);
    assert(state == -2 && host == 0 && fallback == 0);
    std::printf("host producer fallback: windows=%d classes=%d slots=%d ncls_limit=%d PASS\n",
                QSB_SE_WINDOWS, ncls, QSB_FIRST_SLOTS, qhp::NCLS);
}

static void run_case(bool equal_rows) {
    std::vector<uint8_t> rows;
    fill_rows(rows, equal_rows);
    const auto windows = selected_windows();
    uint32_t constant[5];
    fill_constants(constant);
    assert(qsb_prepare_window_schedule(rows.data(), reinterpret_cast<const uint8_t (*)[3]>(windows.data()), constant) == 0);
    const int ncls = qsb_first_class_count;
    if (equal_rows) {
        assert(ncls == 1);
        for (int lane = 0; lane < QSB_SE_PER_EPOCH; lane++) assert(QSB_FIRST_CLASS[lane] == 0);
    } else {
        const int expected = QSB_SE_WINDOWS == 128 ? 8 : 54;
        assert(ncls == expected);
    }
    for (int lane = 0; lane < QSB_SE_PER_EPOCH; lane++) {
        assert(QSB_FIRST_CLASS[lane] >= 0 && QSB_FIRST_CLASS[lane] < ncls);
        const uint32_t rec = QSB_LANE_CLASS[lane];
        assert((rec >> 16) < (uint32_t)QSB_FIRST_SLOTS);
        assert((rec >> 16) < (uint32_t)ncls);
    }
    if (ncls <= qhp::NCLS) verify_producer_and_upload(rows, ncls);
    else verify_host_fallback(ncls);
    std::printf("schedule/index bounds: windows=%d classes=%d equal_rows=%d PASS\n", QSB_SE_WINDOWS, ncls, equal_rows ? 1 : 0);
}

int main() {
    std::memcpy(K, SHA256_K_CANONICAL, sizeof(K));
    assert(std::memcmp(K, SHA256_K_CANONICAL, sizeof(K)) == 0);
    run_case(false);
    run_case(true);
    return 0;
}
