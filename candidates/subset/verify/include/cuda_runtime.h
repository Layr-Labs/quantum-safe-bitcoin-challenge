/* Host emulation of the CUDA-PTX constructs used by the subset tree.
 *
 * - Every 256-lane CTA runs as 256 ucontext fibers on one OS thread; the
 *   fibers are resumed round-robin, so __syncthreads() is an exact barrier.
 * - __shared__ memory is function-local static thread_local storage: one copy
 *   per OS thread, and each OS thread runs one CTA at a time. Blocks of the
 *   same launch run on different OS threads (OpenMP).
 * - QSB_EMU_REVERSE=1 resumes lanes in reverse order; results must not change
 *   (a same-round shared-memory race would show up as a difference).
 * - __syncwarp() (warp-local barrier) is resumed like a full barrier, so the
 *   functional result is checked here; which lanes may read what across a
 *   warp barrier is checked by verify/warp_model.py.
 * - Every lane of a CTA must execute the same sequence of barrier kinds.
 * Device memory is host memory; copies are memcpy. No timing is modeled. */
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <atomic>
#include <ucontext.h>
#include <omp.h>

#define __device__
#define __host__
#define __global__
#define __forceinline__ inline
#define __noinline__
#define __restrict__
#define __constant__
#define __shared__ static thread_local
#define __launch_bounds__(...)
#define __align__(n) alignas(n)
#define __popc(x) __builtin_popcount((unsigned)(x))

struct ulonglong2 { uint64_t x, y; };
static inline ulonglong2 make_ulonglong2(uint64_t x, uint64_t y) { ulonglong2 r = {x, y}; return r; }
struct uint4 { uint32_t x, y, z, w; };
static inline uint4 make_uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) { uint4 r = {x, y, z, w}; return r; }

struct EmuIdx { uint32_t x, y, z; };
static thread_local EmuIdx blockIdx, threadIdx, blockDim;

static inline uint32_t __byte_perm(uint32_t a, uint32_t b, unsigned select) {
    uint64_t both = (uint64_t)a | ((uint64_t)b << 32); uint32_t out = 0;
    for (int i = 0; i < 4; i++) {
        unsigned s = (select >> (4 * i)) & 15, byte = (unsigned)((both >> (8 * (s & 7))) & 255);
        if (s & 8) byte = (byte & 128) ? 255 : 0;
        out |= byte << (8 * i);
    }
    return out;
}
static inline uint32_t atomicAdd(uint32_t *p, uint32_t v) {
    return __sync_fetch_and_add(p, v);
}

/* ---- fiber CTA scheduler ---- */
enum { EMU_MAX_LANES = 256, EMU_STACK = 256 * 1024 };
struct EmuSched {
    ucontext_t sched;
    ucontext_t lane[EMU_MAX_LANES];
    char *stacks = nullptr;
    int cur = 0;
    bool done[EMU_MAX_LANES];
    int barriers[EMU_MAX_LANES];
    uint64_t sig[EMU_MAX_LANES];
    const std::function<void()> *body = nullptr;
};
static thread_local EmuSched *g_emu = nullptr;
static std::atomic<uint64_t> g_emu_ctas{0}, g_emu_barriers{0};

static inline void emu_barrier(unsigned kind) {
    EmuSched *s = g_emu;
    int c = s->cur;
    s->barriers[c]++;
    s->sig[c] = s->sig[c] * 1000003ULL + kind;
    swapcontext(&s->lane[c], &s->sched);
}
static inline void __syncthreads() { emu_barrier(1); }
static inline void __syncwarp() { emu_barrier(2); }
static void emu_lane_entry(int idx) {
    EmuSched *s = g_emu;
    (*s->body)();
    s->done[idx] = true;
}
static inline bool emu_reverse() {
    static int r = -1;
    if (r < 0) { const char *e = getenv("QSB_EMU_REVERSE"); r = (e && *e == '1') ? 1 : 0; }
    return r == 1;
}
static void emu_run_cta(uint32_t block, uint32_t nthreads, const std::function<void()> &body) {
    if (!g_emu) { g_emu = new EmuSched(); g_emu->stacks = (char *)malloc((size_t)EMU_MAX_LANES * EMU_STACK); }
    EmuSched &s = *g_emu;
    if (nthreads > EMU_MAX_LANES) { fprintf(stderr, "emu: too many lanes\n"); abort(); }
    s.body = &body;
    for (uint32_t i = 0; i < nthreads; i++) {
        getcontext(&s.lane[i]);
        s.lane[i].uc_stack.ss_sp = s.stacks + (size_t)i * EMU_STACK;
        s.lane[i].uc_stack.ss_size = EMU_STACK;
        s.lane[i].uc_link = &s.sched;
        makecontext(&s.lane[i], (void (*)())emu_lane_entry, 1, (int)i);
        s.done[i] = false; s.barriers[i] = 0; s.sig[i] = 0;
    }
    bool rev = emu_reverse();
    bool any = true;
    uint64_t rounds = 0;
    while (any) {
        any = false;
        for (uint32_t k = 0; k < nthreads; k++) {
            uint32_t i = rev ? nthreads - 1 - k : k;
            if (s.done[i]) continue;
            s.cur = (int)i;
            blockIdx.x = block; blockIdx.y = blockIdx.z = 0;
            threadIdx.x = i; threadIdx.y = threadIdx.z = 0;
            blockDim.x = nthreads; blockDim.y = blockDim.z = 1;
            swapcontext(&s.sched, &s.lane[i]);
            if (!s.done[i]) any = true;
        }
        rounds++;
    }
    for (uint32_t i = 1; i < nthreads; i++)
        if (s.barriers[i] != s.barriers[0] || s.sig[i] != s.sig[0]) {
            fprintf(stderr, "emu: barrier mismatch in block %u: lane 0 hit %d, lane %u hit %d\n",
                    block, s.barriers[0], i, s.barriers[i]);
            abort();
        }
    g_emu_ctas++;
    g_emu_barriers += (uint64_t)s.barriers[0];
}
template <class F>
static void emu_launch(uint64_t nblocks, uint64_t nthreads, F &&f) {
    const std::function<void()> body = f;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t b = 0; b < (int64_t)nblocks; b++) emu_run_cta((uint32_t)b, (uint32_t)nthreads, body);
}
#define QSB_EMU_LAUNCH(NB, NT, ...) emu_launch((uint64_t)(NB), (uint64_t)(NT), [&]() { __VA_ARGS__; })

/* ---- runtime API ---- */
typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorMemoryAllocation = 2 };
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
                      cudaMemcpyDeviceToDevice, cudaMemcpyDefault };
template <class T> static cudaError_t cudaMalloc(T **p, size_t n) {
    /* QSB_EMU_FAIL_MALLOC_SIZE=<bytes> makes allocations of exactly that size fail
     * (used to exercise the pipeline's non-fatal fallback). */
    static long long fail_size = -2;
    if (fail_size == -2) { const char *e = getenv("QSB_EMU_FAIL_MALLOC_SIZE"); fail_size = e ? atoll(e) : -1; }
    if (fail_size >= 0 && (long long)n == fail_size) { *p = nullptr; return cudaErrorMemoryAllocation; }
    *p = (T *)calloc(1, n ? n : 1);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
static cudaError_t cudaFree(void *p) { free(p); return cudaSuccess; }
static cudaError_t cudaMemcpy(void *d, const void *s, size_t n, int) { memmove(d, s, n); return cudaSuccess; }
static cudaError_t cudaMemset(void *d, int v, size_t n) { memset(d, v, n); return cudaSuccess; }
template <class T> static cudaError_t cudaMemcpyToSymbol(T &dst, const void *src, size_t n) {
    if (n > sizeof(dst)) { fprintf(stderr, "emu: symbol copy too large\n"); abort(); }
    memcpy((void *)&dst, src, n); return cudaSuccess;
}
template <class T> static cudaError_t cudaMemcpyFromSymbol(void *dst, const T &src, size_t n) {
    if (n > sizeof(src)) { fprintf(stderr, "emu: symbol read too large\n"); abort(); }
    memcpy(dst, (const void *)&src, n); return cudaSuccess;
}
static cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
/* Last-error slot: only the emulated L2-persistence refusals below set it, so
 * a test can check that the advisory calls cannot abort the launch checks. */
static cudaError_t emu_last_error = cudaSuccess;
static cudaError_t cudaGetLastError() { cudaError_t e = emu_last_error; emu_last_error = cudaSuccess; return e; }
static const char *cudaGetErrorString(cudaError_t e) { return e ? "emulated error" : "no error"; }
static cudaError_t cudaSetDevice(int) { return cudaSuccess; }
static cudaError_t cudaGetDeviceCount(int *n) { *n = 1; return cudaSuccess; }

/* ---- L2 persistence hint (advisory) ----
 * QSB_EMU_L2 = ok (default): the device offers 72 MiB / 64 MiB and accepts;
 *            = absent: the device reports 0 (hint not offered);
 *            = refuse: the device offers it but both setters fail and leave
 *              a pending last error. Results must be identical in all modes. */
enum { cudaErrorUnsupportedLimit = 215 };
enum cudaLimit { cudaLimitStackSize = 0, cudaLimitPersistingL2CacheSize = 6 };
enum cudaDeviceAttr { cudaDevAttrMaxPersistingL2CacheSize = 108, cudaDevAttrMaxAccessPolicyWindowSize = 109 };
enum cudaAccessProperty { cudaAccessPropertyNormal = 0, cudaAccessPropertyStreaming = 1, cudaAccessPropertyPersisting = 2 };
struct cudaAccessPolicyWindow {
    void *base_ptr; size_t num_bytes; float hitRatio;
    enum cudaAccessProperty hitProp; enum cudaAccessProperty missProp;
};
typedef union { char pad[64]; struct cudaAccessPolicyWindow accessPolicyWindow; } cudaStreamAttrValue;
enum { cudaStreamAttributeAccessPolicyWindow = 1 };
typedef void *cudaStream_t;
static int emu_l2_mode() {  /* 0 ok, 1 absent, 2 refuse */
    const char *e = getenv("QSB_EMU_L2");
    if (!e || !strcmp(e, "ok")) return 0;
    if (!strcmp(e, "absent")) return 1;
    if (!strcmp(e, "refuse")) return 2;
    fprintf(stderr, "emu: bad QSB_EMU_L2=%s\n", e); abort();
}
static size_t emu_l2_limit = 0;
static cudaStreamAttrValue emu_l2_window = {};
static cudaError_t cudaDeviceGetAttribute(int *v, enum cudaDeviceAttr a, int) {
    int m = emu_l2_mode();
    if (a == cudaDevAttrMaxPersistingL2CacheSize) *v = m == 1 ? 0 : 72 << 20;
    else if (a == cudaDevAttrMaxAccessPolicyWindowSize) *v = m == 1 ? 0 : 64 << 20;
    else return cudaErrorInvalidValue;
    return cudaSuccess;
}
static cudaError_t cudaDeviceSetLimit(int limit, size_t value) {
    if (limit != cudaLimitPersistingL2CacheSize) return cudaSuccess;
    if (emu_l2_mode() == 2) return emu_last_error = cudaErrorUnsupportedLimit;
    if (value == 0 || value > (size_t)(72 << 20)) { fprintf(stderr, "emu: bad persisting L2 size\n"); abort(); }
    emu_l2_limit = value; return cudaSuccess;
}
static cudaError_t cudaStreamSetAttribute(cudaStream_t s, int attr, const cudaStreamAttrValue *v) {
    if (s != nullptr || attr != cudaStreamAttributeAccessPolicyWindow || !v) return emu_last_error = cudaErrorInvalidValue;
    if (emu_l2_mode() == 2) return emu_last_error = cudaErrorInvalidValue;
    const struct cudaAccessPolicyWindow &w = v->accessPolicyWindow;
    if (!w.base_ptr || w.num_bytes == 0 || w.num_bytes > (size_t)(64 << 20) || w.num_bytes > emu_l2_limit ||
        w.hitRatio != 1.0f || w.hitProp != cudaAccessPropertyPersisting || w.missProp != cudaAccessPropertyStreaming) {
        fprintf(stderr, "emu: unexpected access policy window\n"); abort();
    }
    emu_l2_window = *v; return cudaSuccess;
}
struct cudaDeviceProp { char name[64]; int multiProcessorCount; };
static cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p, int) {
    snprintf(p->name, sizeof(p->name), "host emulation"); p->multiProcessorCount = 256; return cudaSuccess;
}
