#pragma once
/* Native sm_89 carrier for the subset search.
 *
 * Design and most of this file are Ryun1's native sm_89 carrier from the pinning
 * track (public submission 25bd990a, candidates/pinning/QsbCarrier.h,
 * build_carrier.sh, CARRIER.md; GPL-3). Credit for the idea and the loader goes to
 * Ryun1. This port adapts it to the subset tree: other kernels, every host upload
 * mirrored into both images, a build-knob fingerprint, and a per-launch fallback.
 *
 * Why. The fixed build line (`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu ...`, no
 * -arch) embeds compute_52 PTX, which the driver JIT-compiles for the RTX 4090. PTX
 * for .target sm_52 cannot express any sm_75+ instruction, so the GLV12 table loads
 * cannot carry an L2 prefetch-size hint.
 *
 * What. build_carrier.sh compiles the SAME source offline with
 * `-arch=sm_89 -cubin -DQSB_CARRIER_BUILD=1` and embeds the cubin as base64 in
 * qsb_carrier_sm89.h. QSB_CARRIER_BUILD changes one device line (qsb_s3_load in
 * tree.cu): the first 16 B load of each cold (DRAM-segment) 64 B table record
 * becomes `ld.global.cs.nc.L2::64B`, so a miss fetches both 32 B sectors of the
 * record as one DRAM access. At startup this file loads that image with
 * cudaLibraryLoadData and resolves the four search-loop kernels (epoch groups,
 * incremental epochs, first-block states, digest); sp_launch in tree.cu launches
 * them with cudaLaunchKernel.
 *
 * Module state. The image is a separate CUDA module with its own copy of every
 * __device__ / __constant__ global. Every host upload goes through QSB_TO_SYMBOL,
 * which writes the normal (JIT) symbol first and then the carrier's global of the
 * same name, so kernels of either image read identical data. No kernel writes a
 * module global; the search kernels hand data to each other only through buffers
 * passed by pointer, which both images share. The table build (kernel_build_gtable,
 * kernel_gt_heal_scan) also runs from the image when it is on, so the ranked path
 * launches no compute_52 kernel and the driver JIT-compiles nothing at startup.
 * Neither kernel reads an uploaded constant; the OpenSSL spot check still gates the table.
 *
 * Fallback. Everything is optional. If the GPU is not sm_89, QSB_CARRIER_DISABLE is
 * set in the environment, the image fails to decode or load, a kernel is missing,
 * the image's build knobs differ from this binary's (qsb_carrier_knobs), or an
 * upload into the image fails, the program prints
 * `Native sm_89 carrier: off (...)` and runs the unchanged compute_52 kernels. A
 * failed carrier launch mid-run switches to the <<<>>> launch of the same kernel
 * (both images hold the same uploads, so mixing them is safe). The exact OpenSSL
 * host gate (QSB_HOST_VERIFY) re-derives every hit in both modes.
 *
 * Rebuild rule. Rerun build_carrier.sh after ANY edit to device code in this tree.
 * A kernel whose parameter list changed no longer resolves (mangled name) and a
 * changed build knob fails the fingerprint, both of which fall back cleanly; other
 * stale device code would run, and the host gate would still block wrong hits.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tuple>
#include <utility>
#include <type_traits>

/* QSB_CARRIER (kill switch): 1 = use the embedded native sm_89 image when it loads;
 * 0 = no carrier code at all (every launch is the compute_52 <<<>>> launch). */
#ifndef QSB_CARRIER
#define QSB_CARRIER 1
#endif

/* Build-knob fingerprint: "NAME=value;" per knob, compared byte for byte between
 * this binary (host pass) and the image (device global qsb_carrier_knobs). */
#define QSB_CARRIER_STR2(x) #x
#define QSB_CARRIER_STR(x) QSB_CARRIER_STR2(x)
#define QSB_CARRIER_KV(name) #name "=" QSB_CARRIER_STR(name) ";"

enum QsbCarrierKernel {
    QK_EG = 0,   /* kernel_epoch_groups     */
    QK_BEI,      /* kernel_build_epochs_inc */
    QK_BFF,      /* kernel_build_first_flat */
    QK_DIG,      /* kernel_digest           */
    QK_BUILD,    /* kernel_build_gtable     */
    QK_HEAL,     /* kernel_gt_heal_scan     */
    QK_N
};

struct QsbCarrierState {
    int on;
    int running;             /* set once the search loop may have work queued on the image */
    cudaLibrary_t lib;
    cudaKernel_t k[QK_N];
};
static QsbCarrierState g_qsb_carrier = {0, 0, nullptr, {}};

/* Uploads made while the carrier is on go only into the image, so the compute_52
 * module is never touched and the driver never JIT-compiles it. Each such upload is
 * recorded here; if the carrier is switched off later, qsb_carrier_off replays them
 * into the normal image before any <<<>>> fallback launch can read them. */
struct QsbCarrierUpload { const void *sym; void *data; size_t n; };
static QsbCarrierUpload g_qsb_uploads[32];
static int g_qsb_upload_count = 0;
static int g_qsb_upload_overflow = 0;

static void qsb_carrier_replay_uploads() {
    for (int i = 0; i < g_qsb_upload_count; i++) {
        cudaError_t e = cudaMemcpyToSymbol(g_qsb_uploads[i].sym, g_qsb_uploads[i].data, g_qsb_uploads[i].n);
        if (e != cudaSuccess) {
            fprintf(stderr, "carrier fallback: replaying upload %d failed: %s\n", i, cudaGetErrorString(e));
            exit(2);
        }
    }
}

static void qsb_carrier_off(const char *why) {
    /* Never unload an image whose kernels may still be in flight; just stop using it. */
    if (g_qsb_carrier.lib && !g_qsb_carrier.running) {
        cudaLibraryUnload(g_qsb_carrier.lib);
        g_qsb_carrier.lib = nullptr;
    }
    const int was_on = g_qsb_carrier.on;
    g_qsb_carrier.on = 0;
    memset(g_qsb_carrier.k, 0, sizeof(g_qsb_carrier.k));
    cudaGetLastError();      /* clear the non-sticky error of the failed attempt */
    if (was_on) {
        if (g_qsb_upload_overflow) {
            fprintf(stderr, "carrier fallback: upload log overflowed; cannot restore the normal image\n");
            exit(2);
        }
        qsb_carrier_replay_uploads();   /* the normal image has not seen these yet */
    }
    printf("  Native sm_89 carrier: off (%s); using the compute_52 image%s\n", why,
           was_on ? " from here on" : "");
    fflush(stdout);
}

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

/* `knobs` is this binary's QSB_CARRIER_KNOBS string (host pass, same macros). */
static void qsb_carrier_init(const cudaDeviceProp &prop, const char *knobs) {
    const char *dis = getenv("QSB_CARRIER_DISABLE");
    if (dis && *dis && strcmp(dis, "0") != 0) { qsb_carrier_off("disabled by QSB_CARRIER_DISABLE"); return; }
    if (prop.major != 8 || prop.minor != 9) { qsb_carrier_off("device is not sm_89"); return; }
    size_t len = 0;
    unsigned char *img = qsb_carrier_decode(&len);
    if (!img) { qsb_carrier_off("embedded image failed to decode"); return; }
    cudaError_t e = cudaLibraryLoadData(&g_qsb_carrier.lib, img, nullptr, nullptr, 0,
                                        nullptr, nullptr, 0);
    free(img);
    if (e != cudaSuccess) { g_qsb_carrier.lib = nullptr; qsb_carrier_off(cudaGetErrorString(e)); return; }
    for (int i = 0; i < QK_N; i++) {
        e = cudaLibraryGetKernel(&g_qsb_carrier.k[i], g_qsb_carrier.lib, qsb_carrier_kernel_names[i]);
        if (e != cudaSuccess) { qsb_carrier_off("kernel missing from image"); return; }
    }
    /* Build fingerprint: the image must have been compiled with this binary's knobs. */
    void *dk = nullptr; size_t kb = 0;
    const size_t want = strlen(knobs) + 1;
    e = cudaLibraryGetGlobal(&dk, &kb, g_qsb_carrier.lib, "qsb_carrier_knobs");
    if (e != cudaSuccess || kb != want) { qsb_carrier_off("image built with other knobs"); return; }
    char *img_knobs = (char *)malloc(want);
    if (!img_knobs) { qsb_carrier_off("out of host memory"); return; }
    e = cudaMemcpy(img_knobs, dk, want, cudaMemcpyDeviceToHost);
    const int same = (e == cudaSuccess) && memcmp(img_knobs, knobs, want) == 0;
    free(img_knobs);
    if (!same) { qsb_carrier_off("image built with other knobs"); return; }
    g_qsb_carrier.on = 1;
    printf("  Native sm_89 carrier: on (%zu-byte image, sha256 %.16s..., L2::64B cold-record loads)\n",
           len, qsb_carrier_cubin_sha256);
    fflush(stdout);
}
#else
static void qsb_carrier_init(const cudaDeviceProp &, const char *) {}
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
    return qsb_carrier_launch_impl<P...>(kid, g, b, st, std::index_sequence_for<P...>{},
                                         std::forward<A>(a)...);
}
/* Try the carrier launch; on failure switch the carrier off (the caller then issues
 * the <<<>>> launch of the same kernel). Returns true when the carrier launched it. */
template <typename... P, typename... A>
static bool qsb_carrier_try(void (*kern)(P...), int kid, dim3 g, dim3 b, cudaStream_t st,
                            A &&...a) {
    if (!qsb_carrier_has(kid)) return false;
    cudaError_t e = qsb_carrier_launch(kern, kid, g, b, st, std::forward<A>(a)...);
    if (e == cudaSuccess) return true;
    char why[160];
    snprintf(why, sizeof(why), "launch failed: %s", cudaGetErrorString(e));
    qsb_carrier_off(why);
    return false;
}

/* cudaMemcpyToSymbol into the normal (JIT) image, mirrored into the carrier image's
 * global of the same name when the carrier is on. A global the image does not
 * contain is referenced by none of its kernels and is skipped; any other failure
 * switches the carrier off (the JIT image already holds the upload). */
template <class T>
static cudaError_t qsb_to_symbol(const T &sym, const char *name, const void *src, size_t n) {
    if (!g_qsb_carrier.on) return cudaMemcpyToSymbol(sym, src, n);
    /* Record first, so a later fallback can restore the normal image. */
    if (g_qsb_upload_count < (int)(sizeof(g_qsb_uploads) / sizeof(g_qsb_uploads[0]))) {
        void *copy = malloc(n);
        if (!copy) g_qsb_upload_overflow = 1;
        else {
            memcpy(copy, src, n);
            g_qsb_uploads[g_qsb_upload_count++] = {(const void *)&sym, copy, n};
        }
    } else g_qsb_upload_overflow = 1;
    void *d = nullptr; size_t sz = 0;
    cudaError_t ce = cudaLibraryGetGlobal(&d, &sz, g_qsb_carrier.lib, name);
    if (ce == cudaErrorSymbolNotFound || ce == cudaErrorInvalidSymbol) { cudaGetLastError(); return cudaSuccess; }
    if (ce == cudaSuccess && n > sz) ce = cudaErrorInvalidValue;
    if (ce == cudaSuccess) ce = cudaMemcpy(d, src, n, cudaMemcpyHostToDevice);
    if (ce != cudaSuccess) {
        char why[160];
        snprintf(why, sizeof(why), "upload of %s failed: %s", name, cudaGetErrorString(ce));
        qsb_carrier_off(why);   /* replays every recorded upload, this one included */
    }
    return cudaSuccess;
}
/* cudaMemcpyFromSymbol that reads the image's copy while the carrier is on (both
 * images hold identical statically initialized data), so the normal module stays unloaded. */
template <class T>
static cudaError_t qsb_from_symbol(void *dst, const T &sym, const char *name, size_t n) {
    if (g_qsb_carrier.on) {
        void *d = nullptr; size_t sz = 0;
        cudaError_t ce = cudaLibraryGetGlobal(&d, &sz, g_qsb_carrier.lib, name);
        if (ce == cudaSuccess && sz >= n) return cudaMemcpy(dst, d, n, cudaMemcpyDeviceToHost);
        cudaGetLastError();
    }
    return cudaMemcpyFromSymbol(dst, sym, n);
}
#define QSB_FROM_SYMBOL(dst, sym, n) qsb_from_symbol(dst, sym, #sym, n)
#define QSB_TO_SYMBOL(sym, src, n) qsb_to_symbol(sym, #sym, src, n)
