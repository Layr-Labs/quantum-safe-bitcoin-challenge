/* hit_telemetry.h -- QSB_HIT_TELEMETRY (host only, not an image knob; default 1 in this tree, 0 = the previous host code): an
 * in-run log of the GPU's clock, power, temperature and clock-event reasons, and of the GPU and co-grinder progress, carried in
 * the ORDER of the GPU hit lines inside each batch. The public hit list of a ranked draw keeps file order and nothing else from
 * the run, so this is the only channel from the ranked runner back to us. No line is added or changed.
 *
 * Exactness: the hits themselves are untouched. Every tentative goes through the exact host gate (qsb_hv_check) and the fence's
 * first-publication set exactly as without the switch (qsb_hv_publish_line in qsb_host_verify.h is qsb_hv_publish with the
 * write deferred); the batch's verified lines are sorted into a canonical order (their nine indices, then recid) and the first
 * m = min(n, 20) of them are written in the permutation whose Lehmer code carries floor(log2(m!)) bits. A permutation is a
 * bijection, so each verified hit is written exactly once and the hit set (and the count) is the run's own. The harness
 * verifies hits one by one and never reads their order.
 *
 * Sampler: one detached thread, started when the search loop starts. It dlopens libnvidia-ml.so.1 (rootless reads) and every
 * second queues one 96-bit frame; if NVML cannot be loaded, initialised or read (no driver library in the container, no
 * permission, an old driver), those fields are zero and the order carries progress only. Nothing here can stop the run: a
 * thread that cannot be created means no telemetry (the lines then keep their canonical order), the sampler's own errors end
 * the sampler only, and the GPU host thread never waits on it: it takes whatever bits are queued (a mutex held for a few
 * hundred nanoseconds), zero bits when none are. Frames (MSB first, CRC-8 poly 0x07 over the bits after the sync byte):
 *   sample: sync 0xA7 (8) | type 0 (2) | seq (11: seconds since the sampler started) | GPU batches mod 4096 (12) |
 *           co-grinder candidates / 2^20 mod 65536 (16) | SM clock / 15 MHz (8) | power W (10) | GPU temperature C (7) |
 *           reasons (7: idle, SW power cap, HW slowdown, SW thermal, HW thermal, HW power brake, other) |
 *           memory temperature C (7, 0 = not reported) | crc (8)                                                   = 96 bits
 *   event:  sync 0xA7 | type 1 | seq (11) | code (8) | value (32) | crc (8)                                      = 69 bits
 *           code 1 enforced power limit (mW), 2 NVML status (0 ok, else the failing step), 3 sampler start (ms after exec),
 *           4 frames dropped (queue full), 5 sample period (ms)
 * A decoder takes the GPU hits of the public list in file order, groups them by batch (epoch rank >> 20), reads each batch's
 * first m lines as a Lehmer code against their canonical order, and scans the bit stream for frames whose CRC matches. */
#pragma once
#include <dlfcn.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <string>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <algorithm>

namespace qtel {

/* ---- bit queue (sampler -> GPU host thread) ---- */
struct BitQ { std::mutex m; std::vector<uint8_t> b; size_t head = 0; uint64_t dropped = 0; };
static BitQ &g_q = *new BitQ;   /* never destroyed: no static destructor runs under a live sampler at exit */
enum { QMAX_BITS = 1 << 15 };
static void q_push(const std::vector<uint8_t> &frame) {
    std::lock_guard<std::mutex> g(g_q.m);
    if (g_q.b.size() - g_q.head + frame.size() > (size_t)QMAX_BITS) { g_q.dropped++; return; }
    if (g_q.head > (1u << 16)) { g_q.b.erase(g_q.b.begin(), g_q.b.begin() + (long)g_q.head); g_q.head = 0; }
    g_q.b.insert(g_q.b.end(), frame.begin(), frame.end());
}
/* Up to n (<= 62) bits, MSB first, zero-padded at the end when fewer are queued. */
static uint64_t q_pop(int n) {
    std::lock_guard<std::mutex> g(g_q.m);
    uint64_t v = 0;
    for (int i = 0; i < n; i++) { v <<= 1; if (g_q.head < g_q.b.size()) v |= g_q.b[g_q.head++]; }
    return v;
}

/* ---- frames ---- */
static void put(std::vector<uint8_t> &f, uint64_t v, int n) { for (int i = n - 1; i >= 0; i--) f.push_back((uint8_t)((v >> i) & 1)); }
static uint8_t crc8(const uint8_t *bits, size_t n) {
    uint8_t c = 0;
    for (size_t i = 0; i < n; i++) { const uint8_t fb = (uint8_t)(((c >> 7) & 1) ^ bits[i]); c = (uint8_t)(c << 1); if (fb) c ^= 0x07; }
    return c;
}
static void finish(std::vector<uint8_t> &f) { const uint8_t c = crc8(f.data() + 8, f.size() - 8); put(f, c, 8); q_push(f); }
struct Sample { unsigned seq, gpu_batches, cpu_m, sm_mhz, power_w, temp_c, reasons, mem_c; };
static unsigned clampu(double v, unsigned mx) { return v < 0 ? 0u : v > mx ? mx : (unsigned)(v + 0.5); }
static void push_sample(const Sample &s) {
    std::vector<uint8_t> f; f.reserve(96);
    put(f, 0xA7, 8); put(f, 0, 2); put(f, s.seq & 2047, 11); put(f, s.gpu_batches & 4095, 12); put(f, s.cpu_m & 65535, 16);
    put(f, clampu(s.sm_mhz / 15.0, 255), 8); put(f, clampu(s.power_w, 1023), 10); put(f, clampu(s.temp_c, 127), 7);
    put(f, s.reasons & 127, 7); put(f, clampu(s.mem_c, 127), 7);
    finish(f);
}
static void push_event(unsigned seq, unsigned code, uint32_t value) {
    std::vector<uint8_t> f; f.reserve(69);
    put(f, 0xA7, 8); put(f, 1, 2); put(f, seq & 2047, 11); put(f, code & 255, 8); put(f, value, 32);
    finish(f);
}
/* NVML clock-event reason bits -> the frame's 7 bits */
static unsigned reason_bits(unsigned long long r) {
    return (unsigned)((r & 0x1ull) ? 1 : 0) | ((r & 0x4ull) ? 2 : 0) | ((r & 0x8ull) ? 4 : 0) | ((r & 0x20ull) ? 8 : 0) |
           ((r & 0x40ull) ? 16 : 0) | ((r & 0x80ull) ? 32 : 0) | ((r & ~0xEDull) ? 64 : 0);
}

/* ---- NVML, loaded at run time (no link dependency; rootless reads only) ---- */
typedef struct nvmlDevice_st *nvmlDevice_t;
struct nvmlFieldValue_t { unsigned int fieldId, scopeId; long long timestamp, latencyUsec; int valueType, nvmlReturn; unsigned long long value; };
struct Nvml {
    void *lib = nullptr; nvmlDevice_t dev = nullptr;
    int (*init)(void) = nullptr;
    int (*by_pci)(const char *, nvmlDevice_t *) = nullptr;
    int (*by_index)(unsigned, nvmlDevice_t *) = nullptr;
    int (*clock)(nvmlDevice_t, int, unsigned *) = nullptr;
    int (*power)(nvmlDevice_t, unsigned *) = nullptr;
    int (*temp)(nvmlDevice_t, int, unsigned *) = nullptr;
    int (*reasons)(nvmlDevice_t, unsigned long long *) = nullptr;
    int (*limit)(nvmlDevice_t, unsigned *) = nullptr;
    int (*fields)(nvmlDevice_t, int, nvmlFieldValue_t *) = nullptr;
};
enum { NVML_FI_DEV_MEMORY_TEMP = 82 };   /* nvml.h field id; a GeForce driver may answer "not supported" */
/* 0 on success, else the step that failed (1 dlopen, 2 symbols, 3 init, 4 device) */
static int nvml_open(Nvml &n, const char *pci) {
    n.lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!n.lib) return 1;
#define QTEL_SYM(f, name) *(void **)(&n.f) = dlsym(n.lib, name)
    QTEL_SYM(init, "nvmlInit_v2"); QTEL_SYM(by_pci, "nvmlDeviceGetHandleByPciBusId_v2"); QTEL_SYM(by_index, "nvmlDeviceGetHandleByIndex_v2");
    QTEL_SYM(clock, "nvmlDeviceGetClockInfo"); QTEL_SYM(power, "nvmlDeviceGetPowerUsage"); QTEL_SYM(temp, "nvmlDeviceGetTemperature");
    QTEL_SYM(reasons, "nvmlDeviceGetCurrentClocksEventReasons");
    if (!n.reasons) QTEL_SYM(reasons, "nvmlDeviceGetCurrentClocksThrottleReasons");
    QTEL_SYM(limit, "nvmlDeviceGetEnforcedPowerLimit"); QTEL_SYM(fields, "nvmlDeviceGetFieldValues");
#undef QTEL_SYM
    if (!n.init || !n.clock || !n.power || !n.temp || (!n.by_pci && !n.by_index)) return 2;
    if (n.init() != 0) return 3;
    if (!(pci && pci[0] && n.by_pci && n.by_pci(pci, &n.dev) == 0) && !(n.by_index && n.by_index(0, &n.dev) == 0)) return 4;
    return 0;
}

/* ---- thermal streak (QSB_RT_THERMAL in tree.cu) ----
 * Consecutive 1 s samples whose NVML clock-event reasons include one of QSB_RT_THERMAL_MASK (default: SW thermal slowdown
 * 0x20 and HW thermal slowdown 0x40). Written by the sampler only, read by the GPU host thread; 0 when NVML is not usable. */
#ifndef QSB_RT_THERMAL_MASK
#define QSB_RT_THERMAL_MASK 0x60ull
#endif
static std::atomic<int> g_therm_streak{0};

/* ---- sampler ---- */
typedef uint64_t (*count_fn)();
typedef int (*stop_fn)();
static double now_s() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static const double g_exec_t = now_s();   /* static initialisation: about exec */
/* gpu: GPU candidates walked so far (the host loop's g_total_searched); cpu: co-grinder candidates (nullptr if none); stop: the
 * stop signal; pci: the device's PCI bus id (cudaDeviceGetPCIBusId), may be empty. */
static void start(count_fn gpu, count_fn cpu, stop_fn stop, const char *pci) {
    std::string pcis = pci ? pci : "";
    const unsigned exec_ms = (unsigned)((now_s() - g_exec_t) * 1000.0);
    try { std::thread([gpu, cpu, stop, pcis, exec_ms]() { try {
        Nvml n; const int st = nvml_open(n, pcis.c_str());
        push_event(0, 3, exec_ms); push_event(0, 2, (uint32_t)st); push_event(0, 5, 1000);
        if (st == 0 && n.limit) { unsigned mw = 0; if (n.limit(n.dev, &mw) == 0) push_event(0, 1, mw); }
        const double t0 = now_s();
        uint64_t last_drop = 0;
        for (unsigned seq = 1; !stop(); seq++) {
            const double due = t0 + seq; double w;
            while ((w = due - now_s()) > 0 && !stop()) { struct timespec d = {0, (long)((w < 0.2 ? w : 0.2) * 1e9)}; nanosleep(&d, nullptr); }
            if (stop()) break;
            Sample s = {}; s.seq = seq;
            s.gpu_batches = (unsigned)(gpu() >> 27); s.cpu_m = cpu ? (unsigned)(cpu() >> 20) : 0;
            if (st == 0) {
                unsigned v = 0; unsigned long long r = 0;
                if (n.clock(n.dev, 1 /* NVML_CLOCK_SM */, &v) == 0) s.sm_mhz = v;
                if (n.power(n.dev, &v) == 0) s.power_w = (unsigned)((v + 500) / 1000);
                if (n.temp(n.dev, 0 /* NVML_TEMPERATURE_GPU */, &v) == 0) s.temp_c = v;
                if (n.reasons && n.reasons(n.dev, &r) == 0) s.reasons = reason_bits(r);
                if (r & (unsigned long long)(QSB_RT_THERMAL_MASK)) g_therm_streak.fetch_add(1, std::memory_order_relaxed);
                else g_therm_streak.store(0, std::memory_order_relaxed);
                if (n.fields) {
                    nvmlFieldValue_t fv; memset(&fv, 0, sizeof fv); fv.fieldId = NVML_FI_DEV_MEMORY_TEMP;
                    if (n.fields(n.dev, 1, &fv) == 0 && fv.nvmlReturn == 0) s.mem_c = (unsigned)(fv.value & 0xffffffffull);
                }
            }
            push_sample(s);
            uint64_t d; { std::lock_guard<std::mutex> g(g_q.m); d = g_q.dropped; }
            if (d != last_drop) { push_event(seq, 4, (uint32_t)d); last_drop = d; }
        }
    } catch (...) {} }).detach(); } catch (...) {}   /* no thread, or a sampler error: no telemetry, the run goes on */
}

/* ---- encoder: order one batch's verified lines ---- */
struct Line { uint8_t key[10]; int len; char text[96]; };
static int lehmer_bits(int m) { double s = 0; for (int i = 2; i <= m; i++) s += log2((double)i); return (int)floor(s + 1e-9); }
static bool key_less(const Line &a, const Line &b) { return memcmp(a.key, b.key, 10) < 0; }
/* Sort L[0..n) canonically, then permute its first m = min(n, 20) entries by the next floor(log2(m!)) queued bits (Lehmer code,
 * first digit least significant). Returns the bits carried. */
static int order(Line *L, int n) {
    std::sort(L, L + n, key_less);
    const int m = n < 20 ? n : 20;
    if (m < 2) return 0;
    const int nb = lehmer_bits(m);
    uint64_t v = q_pop(nb);
    Line tmp[20]; int rem[20]; for (int i = 0; i < m; i++) { tmp[i] = L[i]; rem[i] = i; }
    int nr = m;
    for (int i = 0; i < m; i++) {
        const uint64_t base = (uint64_t)(m - i), d = v % base; v /= base;
        L[i] = tmp[rem[d]];
        for (int k = (int)d; k + 1 < nr; k++) rem[k] = rem[k + 1];
        nr--;
    }
    return nb;
}

}  // namespace qtel
