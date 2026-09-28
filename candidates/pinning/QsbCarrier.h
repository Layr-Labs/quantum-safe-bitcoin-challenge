#pragma once
/* Native sm_89 carrier.
 *
 * The fixed build line (`nvcc -O3 ... pinning.cu`, no -arch) embeds compute_52 PTX,
 * which the driver JIT-compiles for the RTX 4090. PTX for .target sm_52 cannot use
 * any sm_75+ instruction, so the table loads cannot carry an L2 prefetch-size hint.
 *
 * This file loads a second image of the SAME source, compiled offline by
 * build_carrier.sh with `-arch=sm_89 -DQSB_CARRIER_BUILD=1`, from a base64 copy in
 * qsb_carrier_sm89.h, through the CUDA runtime library API. QSB_CARRIER_BUILD changes
 * one device line: the first 16 B load of each 64 B table record becomes
 * `ld.global.nc.L2::64B`, so the whole record (both 32 B sectors) is fetched as one
 * DRAM access instead of two independent sector misses.
 *
 * If the GPU is not sm_89, the image is missing, a required kernel or the zeros
 * fingerprint fails to resolve, or the image was built for another QSB_ZEROS_N,
 * the program runs the unchanged compute_52 kernels. Later symbol-upload and
 * launch errors are fatal rather than silently losing hits. The exact OpenSSL
 * host gate checks every published hit in both modes.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tuple>
#include <utility>
#include <type_traits>

#ifndef QSB_CARRIER
#define QSB_CARRIER 1
#endif

/* QSB_NOJIT (host only; after cefika 6d9b1000): while the carrier is on, every launch
 * site takes a carrier kernel, including the register-root inverse and its startup
 * check, so the compute_52 image is never needed. Its module is then never touched: no
 * preload, no constant uploads or reads. Under the CUDA 12 default of lazy module loading
 * its PTX is therefore never JIT-compiled. The carrier is only switched off inside
 * qsb_carrier_init, before any upload, so the compute_52 image never misses a constant
 * that a later compute_52 launch would read. */
#ifndef QSB_NOJIT
#define QSB_NOJIT 1
#endif

/* QSB_ARMS (host only, default 1): in-run multi-arm A/B probe (challenges/qsb-tools/
 * PROBE-DESIGN.md). The five-arm rectangle build loads all five images of qsb_carrier_sm89.h (the arm table of
 * build_carrier.sh) and time-slices them (qsb_arms_init, QsbProbe below). 1 compiles every probe
 * line out: the base single-image carrier and search, unchanged. QSB_PROBE is the effective
 * switch: never inside the carrier image build itself, and only with the embedded carrier. */
#ifndef QSB_ARMS
#define QSB_ARMS 1
#endif
#if QSB_ARMS != 1 && QSB_ARMS != 5
#error "rectangle probe QSB_ARMS must be 1 (base) or 5 (all probe arms)"
#endif
#if QSB_ARMS > 1 && QSB_CARRIER && !defined(QSB_CARRIER_BUILD)
#define QSB_PROBE 1
#else
#define QSB_PROBE 0
#endif

enum QsbCarrierKernel {
    QK_S0 = 0,   /* kernel_pinning_pipeline<true,0>  (prepare) */
    QK_S2,       /* kernel_pinning_pipeline<true,2>  (finish)  */
    QK_RGP,      /* qsb_root_group_prepare */
    QK_ISR,      /* qsb_invert_super_roots */
    QK_RGF,      /* qsb_root_group_finish  */
    QK_BUILD,    /* kernel_build_gtable    */
    QK_YOFF,     /* qsb_table_offset_y     */
    QK_RF,       /* qsb_root_fused<K>             (optional: empty name when absent) */
    QK_RR,       /* qsb_root_register             (optional) */
    QK_PFC,      /* qsb_prefix_field_check_kernel (optional) */
    QK_N
};

struct QsbCarrierState {
    int on;
    int nojit;   /* QSB_NOJIT and every kernel any launch site may take resolved in the carrier */
    cudaLibrary_t lib;
    cudaKernel_t k[QK_N];
};
static QsbCarrierState g_qsb_carrier = {0, 0, nullptr, {}};

#if QSB_CARRIER && !defined(QSB_CARRIER_BUILD)
#include "qsb_carrier_sm89.h"

static int qsb_b64_val(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

/* Decode the line-split base64 image. Returns a malloc'd buffer or nullptr. */
static unsigned char *qsb_carrier_decode(size_t *out_len) {
    unsigned char *buf = (unsigned char *)malloc(qsb_carrier_cubin_bytes + 4);
    if (!buf) return nullptr;
    size_t n = 0; unsigned acc = 0; int bits = 0;
    for (unsigned li = 0; li < qsb_carrier_b64_lines; li++) {
        for (const unsigned char *p = (const unsigned char *)qsb_carrier_b64[li]; *p; p++) {
            int v = qsb_b64_val(*p);
            if (v < 0) continue;              /* '=' padding */
            acc = (acc << 6) | (unsigned)v; bits += 6;
            if (bits >= 8) {
                bits -= 8;
                if (n >= qsb_carrier_cubin_bytes) { free(buf); return nullptr; }
                buf[n++] = (unsigned char)(acc >> bits);
            }
        }
    }
    if (n != qsb_carrier_cubin_bytes) { free(buf); return nullptr; }
    *out_len = n;
    return buf;
}

static void qsb_carrier_off(const char *why) {
    if (g_qsb_carrier.lib) cudaLibraryUnload(g_qsb_carrier.lib);
    g_qsb_carrier.on = 0; g_qsb_carrier.nojit = 0; g_qsb_carrier.lib = nullptr;
    memset(g_qsb_carrier.k, 0, sizeof(g_qsb_carrier.k));
    cudaGetLastError();                        /* clear any sticky-free error from the attempt */
    printf("  Native sm_89 carrier: off (%s); using the compute_52 image\n", why);
}

#if QSB_PROBE
/* ---- Multi-arm probe (challenges/qsb-tools/PROBE-DESIGN.md) ----
 * Arm a is image a of qsb_carrier_sm89.h: this same source built with arm a's extra
 * nvcc/ptxas flags (table at the top of build_carrier.sh). Arm 0 is the control image and is
 * loaded by the unchanged code below into g_qsb_carrier; every other arm is its own library
 * with its own kernel table. The search loop points g_qsb_carrier.lib/.k at the active arm
 * (qsb_arm_select) only between slices, when every slot has been drained. */
#if !defined(QSB_CARRIER_ARMS) || QSB_ARMS > QSB_CARRIER_ARMS
#error "QSB_ARMS exceeds the arm count of qsb_carrier_sm89.h (regenerate it with build_carrier.sh)"
#endif
struct QsbArm { cudaLibrary_t lib; cudaKernel_t k[QK_N]; };
static QsbArm g_qsb_arm[QSB_ARMS];
static int g_qsb_arms_on = 1;          /* arms loaded and resolved; 1 = the base single-image path */
static char g_qsb_arm_name[QK_N][192]; /* the kernel names arm 0 resolved */

static unsigned char *qsb_arm_decode(int a, size_t *out_len) {
    const size_t bytes = qsb_carrier_arm_bytes[a];
    unsigned char *buf = (unsigned char *)malloc(bytes + 4);
    if (!buf) return nullptr;
    size_t n = 0; unsigned acc = 0; int bits = 0;
    for (unsigned li = 0; li < qsb_carrier_arm_b64_lines[a]; li++) {
        for (const unsigned char *p = (const unsigned char *)qsb_carrier_arm_b64[a][li]; *p; p++) {
            int v = qsb_b64_val(*p);
            if (v < 0) continue;              /* '=' padding */
            acc = (acc << 6) | (unsigned)v; bits += 6;
            if (bits >= 8) {
                bits -= 8;
                if (n >= bytes) { free(buf); return nullptr; }
                buf[n++] = (unsigned char)(acc >> bits);
            }
        }
    }
    if (n != bytes) { free(buf); return nullptr; }
    *out_len = n;
    return buf;
}

/* Runs at the end of a successful qsb_carrier_init (arm 0 = g_qsb_carrier). Every kernel arm 0
 * resolved must resolve in every arm, and every arm must carry the same QSB_ZEROS_N. Any failure
 * unloads the extra libraries and leaves g_qsb_arms_on = 1: the base search runs unchanged. */
static void qsb_arms_init() {
    g_qsb_arm[0].lib = g_qsb_carrier.lib;
    memcpy(g_qsb_arm[0].k, g_qsb_carrier.k, sizeof(g_qsb_carrier.k));
    printf("  Probe arm 0: sha256 %.16s... [%s] loaded OK (control image)\n",
           qsb_carrier_arm_sha256[0], qsb_carrier_arm_flags[0]);
    const char *why = nullptr;
    int a = 1;
    for (; a < QSB_ARMS; a++) {
        QsbArm &A = g_qsb_arm[a];
        A.lib = nullptr; memset(A.k, 0, sizeof(A.k));
        size_t len = 0;
        unsigned char *img = qsb_arm_decode(a, &len);
        if (!img) { why = "embedded image failed to decode"; break; }
        cudaError_t e = cudaLibraryLoadData(&A.lib, img, nullptr, nullptr, 0, nullptr, nullptr, 0);
        free(img);
        if (e != cudaSuccess) { A.lib = nullptr; why = cudaGetErrorString(e); break; }
        for (int i = 0; i < QK_N && !why; i++) {
            if (!g_qsb_carrier.k[i]) continue;   /* arm 0 runs this one on compute_52: so do all arms */
            if (cudaLibraryGetKernel(&A.k[i], A.lib, g_qsb_arm_name[i]) != cudaSuccess || !A.k[i])
                why = "kernel missing from image";
        }
        if (why) break;
        void *dz = nullptr; size_t zb = 0; int zeros = -1;
        e = cudaLibraryGetGlobal(&dz, &zb, A.lib, "qsb_carrier_zeros");
        if (e == cudaSuccess && zb == sizeof(int))
            e = cudaMemcpy(&zeros, dz, sizeof(int), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess || zeros != QSB_ZEROS_N) { why = "image built for another QSB_ZEROS_N"; break; }
        /* Advisory preload: under lazy loading this loads the arm's kernels now, before the
         * batch is sized to the free VRAM, instead of at the arm's first launch. A failure only
         * leaves the lazy load in place. The prepare kernel's register count identifies the arm
         * build in the log (build_carrier.sh prints the expected value). */
        int s0_regs = -1;
        for (int i = 0; i < QK_N; i++) {
            if (!A.k[i]) continue;
            cudaFuncAttributes fa;
            if (cudaFuncGetAttributes(&fa, (const void *)A.k[i]) == cudaSuccess) {
                if (i == QK_S0) s0_regs = fa.numRegs;
            } else {
                cudaGetLastError();
            }
        }
        printf("  Probe arm %d: sha256 %.16s... [%s] loaded OK (%zu-byte image, prepare regs %d)\n",
               a, qsb_carrier_arm_sha256[a], qsb_carrier_arm_flags[a], len, s0_regs);
    }
    if (why) {
        for (int b = 1; b < QSB_ARMS; b++) {
            if (g_qsb_arm[b].lib) cudaLibraryUnload(g_qsb_arm[b].lib);
            g_qsb_arm[b].lib = nullptr; memset(g_qsb_arm[b].k, 0, sizeof(g_qsb_arm[b].k));
        }
        cudaGetLastError();                    /* clear any sticky-free error from the attempt */
        g_qsb_arms_on = 1;
        printf("  Probe arms: off (arm %d: %s); running the base single-arm search\n", a, why);
        return;
    }
    g_qsb_arms_on = QSB_ARMS;
}
#endif

static void qsb_carrier_init(const cudaDeviceProp &prop) {
    if (prop.major != 8 || prop.minor != 9) { qsb_carrier_off("device is not sm_89"); return; }
    size_t len = 0;
    unsigned char *img = qsb_carrier_decode(&len);
    if (!img) { qsb_carrier_off("embedded image failed to decode"); return; }
    cudaError_t e = cudaLibraryLoadData(&g_qsb_carrier.lib, img, nullptr, nullptr, 0,
                                        nullptr, nullptr, 0);
    free(img);
    if (e != cudaSuccess) { qsb_carrier_off(cudaGetErrorString(e)); return; }
    /* The generated name list may stop at QK_YOFF (an image generator that only knows the
     * required kernels). The optional kernels past its end are then looked up by their
     * fixed mangled names; one that does not resolve stays on the compute_52 image and
     * turns QSB_NOJIT off, so its constants are still uploaded there. */
    const int n_gen = (int)(sizeof(qsb_carrier_kernel_names) / sizeof(qsb_carrier_kernel_names[0]));
    char rf_name[64] = "";
#ifdef QSB_RF_K
    snprintf(rf_name, sizeof(rf_name), "_Z14qsb_root_fusedILi%dEEvPmi", (int)(QSB_RF_K));
#endif
    const char *fixed[QK_N] = {};
    fixed[QK_RF] = rf_name;
    fixed[QK_RR] = "_Z17qsb_root_registerPmi";
    fixed[QK_PFC] = "_Z29qsb_prefix_field_check_kernelPj";
    int all = 1;
    for (int i = 0; i < QK_N; i++) {
        g_qsb_carrier.k[i] = nullptr;
        const char *name = i < n_gen ? qsb_carrier_kernel_names[i] : fixed[i];
        if (i >= QK_RF) {
            if (!name || !name[0]) name = fixed[i];
            if (!name || !name[0]) { all = 0; continue; }
            if (cudaLibraryGetKernel(&g_qsb_carrier.k[i], g_qsb_carrier.lib, name) != cudaSuccess) {
                g_qsb_carrier.k[i] = nullptr; all = 0; cudaGetLastError();
            }
#if QSB_PROBE
            snprintf(g_qsb_arm_name[i], sizeof(g_qsb_arm_name[i]), "%s", name);
#endif
            continue;
        }
        e = cudaLibraryGetKernel(&g_qsb_carrier.k[i], g_qsb_carrier.lib, name);
        if (e != cudaSuccess) { qsb_carrier_off("kernel missing from image"); return; }
#if QSB_PROBE
        snprintf(g_qsb_arm_name[i], sizeof(g_qsb_arm_name[i]), "%s", name);
#endif
    }
    void *dz = nullptr; size_t zb = 0; int zeros = -1;
    e = cudaLibraryGetGlobal(&dz, &zb, g_qsb_carrier.lib, "qsb_carrier_zeros");
    if (e == cudaSuccess && zb == sizeof(int))
        e = cudaMemcpy(&zeros, dz, sizeof(int), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess || zeros != QSB_ZEROS_N) { qsb_carrier_off("image built for another QSB_ZEROS_N"); return; }
    g_qsb_carrier.on = 1;
    g_qsb_carrier.nojit = QSB_NOJIT && all;
    printf("  Native sm_89 carrier: on (%zu-byte image, sha256 %.16s..., L2::64B record loads, %s)\n",
           len, qsb_carrier_cubin_sha256, g_qsb_carrier.nojit ? "no compute_52 JIT" : "root kernels partly compute_52");
#if QSB_PROBE
    qsb_arms_init();
#endif
}
#else
static void qsb_carrier_init(const cudaDeviceProp &) {}
#endif

static inline bool qsb_carrier_has(int kid) { return g_qsb_carrier.on && g_qsb_carrier.k[kid]; }

/* Launch the carrier image of `kern`. The static kernel pointer only supplies the
 * parameter types: every argument is converted to its declared parameter type before
 * its address goes to cudaLaunchKernel, exactly as a <<<>>> launch would. */
template <typename... P, typename... A, size_t... I>
static cudaError_t qsb_carrier_launch_impl(int kid, dim3 g, dim3 b, cudaStream_t st,
                                           std::index_sequence<I...>, A &&...a) {
    std::tuple<typename std::decay<P>::type...> vals(std::forward<A>(a)...);
    void *argv[sizeof...(P) > 0 ? sizeof...(P) : 1] = {(void *)&std::get<I>(vals)...};
    return cudaLaunchKernel((const void *)g_qsb_carrier.k[kid], g, b, argv, 0, st);
}
template <typename... P, typename... A>
static cudaError_t qsb_carrier_launch(void (*)(P...), int kid, dim3 g, dim3 b, cudaStream_t st,
                                      A &&...a) {
    static_assert(sizeof...(P) == sizeof...(A), "carrier launch: argument count mismatch");
    cudaError_t e = qsb_carrier_launch_impl<P...>(kid, g, b, st,
                                                std::index_sequence_for<P...>{},
                                                std::forward<A>(a)...);
    if (e != cudaSuccess) {
        fprintf(stderr, "Native carrier kernel %d launch failed: %s\n",
                kid, cudaGetErrorString(e));
        exit(2);
    }
    return e;
}

/* cudaMemcpyToSymbol into the carrier image's copy of the symbol when it is on, and into
 * the compute_52 image's copy unless QSB_NOJIT keeps that image untouched (see above):
 * with the carrier on, no compute_52 kernel is ever launched. */
template <class T>
static cudaError_t qsb_to_symbol(const T &sym, const char *name, const void *src, size_t n) {
#if QSB_PROBE
    /* Probe: every upload (startup constants and the per-sequence tail table alike) goes to
     * every arm's library, so whichever arm launches next reads current values. */
    if (g_qsb_carrier.on && g_qsb_arms_on > 1) {
        for (int a = 0; a < g_qsb_arms_on; a++) {
            void *d = nullptr; size_t sz = 0;
            cudaError_t e = cudaLibraryGetGlobal(&d, &sz, g_qsb_arm[a].lib, name);
            if (e != cudaSuccess) return e;
            if (n > sz) return cudaErrorInvalidValue;
            e = cudaMemcpy(d, src, n, cudaMemcpyHostToDevice);
            if (e != cudaSuccess) return e;
        }
        if (g_qsb_carrier.nojit) return cudaSuccess;
        return cudaMemcpyToSymbol(sym, src, n);
    }
#endif
    if (g_qsb_carrier.on) {
        void *d = nullptr; size_t sz = 0;
        cudaError_t e = cudaLibraryGetGlobal(&d, &sz, g_qsb_carrier.lib, name);
        if (e != cudaSuccess) return e;
        if (n > sz) return cudaErrorInvalidValue;
        e = cudaMemcpy(d, src, n, cudaMemcpyHostToDevice);
        if (e != cudaSuccess) return e;
        if (g_qsb_carrier.nojit) return cudaSuccess;
    }
    return cudaMemcpyToSymbol(sym, src, n);
}
#define QSB_TO_SYMBOL(sym, src, n) qsb_to_symbol(sym, #sym, src, n)

/* Read back an uploaded constant from the image the kernels run from. */
template <class T>
static cudaError_t qsb_from_symbol(void *dst, const T &sym, const char *name, size_t n) {
    if (g_qsb_carrier.on) {
        void *d = nullptr; size_t sz = 0;
        cudaError_t e = cudaLibraryGetGlobal(&d, &sz, g_qsb_carrier.lib, name);
        if (e != cudaSuccess) return e;
        if (n > sz) return cudaErrorInvalidValue;
        return cudaMemcpy(dst, d, n, cudaMemcpyDeviceToHost);
    }
    return cudaMemcpyFromSymbol(dst, sym, n);
}
#define QSB_FROM_SYMBOL(dst, sym, n) qsb_from_symbol(dst, sym, #sym, n)


#if QSB_PROBE
/* ---- Probe schedule (PROBE-DESIGN.md "Encoding") ----
 * Round robin over the arms, one slice each: slice g runs arm k = g % N on the fresh sequence
 * SEQ_MIN + (k << 26) + j, j = g / N, launching batches from locktime offset 0 upward while
 * less than QSB_PROBE_SLICE_MS of host monotonic time has passed since the slice began (at
 * least one batch; the slice also ends if it reaches the end of the locktime range). The loop
 * then drains every slot -- each hit is published with its own slot_seq/slot_lt exactly as at
 * a base sequence rollover -- and only then selects the next arm. The work of arm k in slice j
 * is read back from the verified hits (highest locktime in its sequence); qsbprobe.py does the
 * analysis. Stdout gets cumulative per-arm lines every QSB_PROBE_PRINT_ROUNDS rounds and once
 * more when the harness's SIGTERM ends the run (async-signal-safe write of a pre-formatted
 * buffer, then the default action). The lines avoid the "M/s", "(nM/" and "Done:" patterns the
 * harness parses from stdout. */
#if !QSB_SLOTPIPE
#error "the multi-arm probe is written for the slotted pipeline (QSB_SLOTPIPE=1)"
#endif
#include <signal.h>
#include <unistd.h>
#include <time.h>
#ifndef QSB_PROBE_SLICE_MS
#define QSB_PROBE_SLICE_MS 900
#endif
#ifndef QSB_PROBE_PRINT_ROUNDS
#define QSB_PROBE_PRINT_ROUNDS 16
#endif
static inline void qsb_arm_select(int a) {
    g_qsb_carrier.lib = g_qsb_arm[a].lib;
    memcpy(g_qsb_carrier.k, g_qsb_arm[a].k, sizeof(g_qsb_carrier.k));
}
/* Host-side arm settings (probe #3): the persisting-L2 access-policy window of each arm, in MiB
 * (0 = leave the base's window). The search loop registers every stream that carries the window
 * (qsb_probe_add_stream) and the base window (qsb_probe_set_window); qsb_probe_apply_window(a)
 * rewrites that attribute on every registered stream between slices, when all slots are drained.
 * The persisting set-aside (cudaLimitPersistingL2CacheSize) is not touched: only the window moves. */
#ifndef QSB_PROBE_WIN_MIB
#define QSB_PROBE_WIN_MIB {0}
#endif
#ifndef QSB_PROBE_HIT_PCT
#define QSB_PROBE_HIT_PCT {0}
#endif
static const int g_qsb_arm_win_mib[QSB_ARMS] = QSB_PROBE_WIN_MIB;
static const int g_qsb_arm_hit_pct[QSB_ARMS] = QSB_PROBE_HIT_PCT;   /* 0 = the base hitRatio */
static cudaStream_t g_qsb_probe_streams[32];
static int g_qsb_probe_nstreams = 0;
static cudaStreamAttrValue g_qsb_probe_base_av;
static int g_qsb_probe_have_av = 0;
static int g_qsb_probe_cur_win = -1;
static void qsb_probe_add_stream(cudaStream_t s) {
    if (g_qsb_probe_nstreams < 32) g_qsb_probe_streams[g_qsb_probe_nstreams++] = s;
}
static void qsb_probe_set_window(const cudaStreamAttrValue &av) { g_qsb_probe_base_av = av; g_qsb_probe_have_av = 1; }
static void qsb_probe_apply_window(int a) {
    if (!g_qsb_probe_have_av) return;
    const int mib = g_qsb_arm_win_mib[a], pct = g_qsb_arm_hit_pct[a];
    const int key = mib * 1000 + pct;
    if (key == g_qsb_probe_cur_win) return;
    cudaStreamAttrValue av = g_qsb_probe_base_av;
    if (mib > 0) {
        int max_window = 0, dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&max_window, cudaDevAttrMaxAccessPolicyWindowSize, dev);
        const size_t want = (size_t)mib << 20;
        av.accessPolicyWindow.num_bytes = want < (size_t)max_window ? want : (size_t)max_window;
    }
    if (pct > 0) av.accessPolicyWindow.hitRatio = (float)pct / 100.0f;
    for (int i = 0; i < g_qsb_probe_nstreams; i++)
        cudaStreamSetAttribute(g_qsb_probe_streams[i], cudaStreamAttributeAccessPolicyWindow, &av);
    cudaGetLastError();
    g_qsb_probe_cur_win = key;
}
struct QsbProbe {
    int n; double t_slice; unsigned long long batch;
    unsigned long long g;           /* slices begun */
    int arm; unsigned j; double t0; unsigned nb; int exhausted;
    unsigned long long slices[QSB_ARMS], batches[QSB_ARMS], warm_batches[QSB_ARMS], exhausted_n[QSB_ARMS];
    double warm_secs[QSB_ARMS];     /* busy time of every slice but the arm's first */
};
static QsbProbe g_qsb_probe;
static char g_qsb_probe_sum[2][QSB_ARMS * 320];
static int g_qsb_probe_sum_len[2];
static int g_qsb_probe_sum_i = -1;   /* buffer the handler may print; release/acquire published */
static double qsb_probe_now() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void qsb_probe_on_term(int sig) {
    const int i = __atomic_load_n(&g_qsb_probe_sum_i, __ATOMIC_ACQUIRE);
    if (i >= 0) { ssize_t w = write(STDOUT_FILENO, g_qsb_probe_sum[i], (size_t)g_qsb_probe_sum_len[i]); (void)w; }
    signal(sig, SIG_DFL);
    raise(sig);
}
/* Format the cumulative per-arm lines into the buffer the signal handler is not reading. */
static void qsb_probe_format() {
    const QsbProbe &P = g_qsb_probe;
    const int i = __atomic_load_n(&g_qsb_probe_sum_i, __ATOMIC_RELAXED) == 0 ? 1 : 0;
    char *buf = g_qsb_probe_sum[i];
    const size_t cap = sizeof(g_qsb_probe_sum[i]);
    size_t len = 0;
    for (int a = 0; a < P.n; a++) {
        const double ws = P.warm_secs[a];
        const double rate = ws > 0 ? (double)P.warm_batches[a] * (double)P.batch / ws / 1e6 : 0.0;
        int w = snprintf(buf + len, cap - len,
                         "  qsbprobe arm %d: slices %llu batches %llu cands %lluM secs %.2f rate %.3fMc/s "
                         "exhausted %llu [%s]\n",
                         a, P.slices[a], P.batches[a], P.batches[a] * P.batch / 1000000ull, ws, rate,
                         P.exhausted_n[a], qsb_carrier_arm_flags[a]);
        if (w < 0 || (size_t)w >= cap - len) break;
        len += (size_t)w;
    }
    g_qsb_probe_sum_len[i] = (int)len;
    __atomic_store_n(&g_qsb_probe_sum_i, i, __ATOMIC_RELEASE);
}
/* After the startup uploads, before the search loop. False: run the base loop unchanged. */
static bool qsb_probe_begin(int effective_total, bool seq_override, uint32_t seq_min, int batch) {
    if (!g_qsb_carrier.on || g_qsb_arms_on < 2) {
        printf("  Probe arms: off (%s); running the base single-arm search\n",
               g_qsb_carrier.on ? "extra arms not loaded" : "carrier off");
        return false;
    }
    if (effective_total != 1 || seq_override) {
        qsb_arm_select(0);
        printf("  Probe arms: off (multi-GPU or seq_start override); running the base single-arm search\n");
        return false;
    }
    memset(&g_qsb_probe, 0, sizeof(g_qsb_probe));
    g_qsb_probe.n = g_qsb_arms_on;
    g_qsb_probe.t_slice = QSB_PROBE_SLICE_MS * 1e-3;
    g_qsb_probe.batch = (unsigned long long)batch;
    qsb_probe_format();
    struct sigaction sa; memset(&sa, 0, sizeof(sa));
    sa.sa_handler = qsb_probe_on_term; sigemptyset(&sa.sa_mask);
    sigaction(SIGTERM, &sa, nullptr);
    printf("  Probe v4: %d arms, round robin, %d ms slices; "
           "bounded sequence-major rectangles, LT base 500000000\n",
           g_qsb_probe.n, (int)QSB_PROBE_SLICE_MS);
    fflush(stdout);
    return true;
}
/* Top of a sequence iteration (every slot drained): select the arm, return the sequence. */
static uint32_t qsb_probe_slice_begin(uint32_t seq_min) {
    QsbProbe &P = g_qsb_probe;
    P.arm = (int)(P.g % (unsigned long long)P.n);
    P.j = (unsigned)(P.g / (unsigned long long)P.n);
    if (P.j >= 384u) { fprintf(stderr, "probe: slice counter overflow\n"); exit(2); }
    qsb_arm_select(P.arm);
    qsb_probe_apply_window(P.arm);
    P.nb = 0; P.exhausted = 1;
    P.t0 = qsb_probe_now();
    uint32_t seq = 0;
    if (!qsb_rect_slice((unsigned)P.arm, P.j, &seq)) {
        fprintf(stderr, "probe: rectangle domain exhausted\n"); exit(2);
    }
    return seq;
}
/* Top of each batch iteration: true ends the slice (never before its first batch). */
static inline bool qsb_probe_batch_stop() {
    QsbProbe &P = g_qsb_probe;
    if (P.nb && qsb_probe_now() - P.t0 >= P.t_slice) { P.exhausted = 0; return true; }
    P.nb++;
    return false;
}
/* After the slice's slots are drained. */
static void qsb_probe_slice_end() {
    QsbProbe &P = g_qsb_probe;
    const double secs = qsb_probe_now() - P.t0;
    const int a = P.arm;
    P.slices[a]++; P.batches[a] += P.nb; P.exhausted_n[a] += (unsigned long long)P.exhausted;
    if (P.j > 0) { P.warm_batches[a] += P.nb; P.warm_secs[a] += secs; }
    P.g++;
    if (P.g % (unsigned long long)P.n == 0) {
        qsb_probe_format();
        if ((P.g / (unsigned long long)P.n) % QSB_PROBE_PRINT_ROUNDS == 0) {
            const int i = __atomic_load_n(&g_qsb_probe_sum_i, __ATOMIC_RELAXED);
            fwrite(g_qsb_probe_sum[i], 1, (size_t)g_qsb_probe_sum_len[i], stdout);
            fflush(stdout);
        }
    }
}
#endif
