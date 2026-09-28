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
/* QSB_NOJIT (host only): while the carrier is on, every launch site takes the carrier
 * kernel, so the compute_52 image is never needed. Its module is then never touched:
 * no preload, no constant uploads into it. Under the CUDA 12 default of lazy module
 * loading its PTX is therefore never JIT-compiled inside the timed window (on a cold
 * JIT cache that compile is seconds; terrapinelf measured 4.2 s for the subset module).
 * The carrier is only ever switched off during qsb_carrier_init, before any upload,
 * so the compute_52 image still receives every upload whenever it is the one in use.
 * 0 restores the double upload and the preload. */
#ifndef QSB_NOJIT
#define QSB_NOJIT 1
#endif

enum QsbCarrierKernel {
    QK_S0 = 0,   /* kernel_pinning_pipeline<true,0>  (prepare) */
    QK_S2,       /* kernel_pinning_pipeline<true,2>  (finish)  */
    QK_RGP,      /* qsb_root_group_prepare */
    QK_ISR,      /* qsb_invert_super_roots */
    QK_RGF,      /* qsb_root_group_finish  */
    QK_BUILD,    /* kernel_build_gtable    */
    QK_YOFF,     /* qsb_table_offset_y     */
    QK_RF,       /* qsb_root_fused<N> (optional: empty name when QSB_ROOT_FUSED=0) */
    QK_N
};

struct QsbCarrierState {
    int on;
    cudaLibrary_t lib;
    cudaKernel_t k[QK_N];
    cudaLibrary_t finish_lib;
    cudaKernel_t finish_k;
};
static QsbCarrierState g_qsb_carrier = {0, nullptr, {}, nullptr, nullptr};

#if QSB_CARRIER && !defined(QSB_CARRIER_BUILD)
#include "qsb_carrier_sm89.h"
#include "qsb_finish_carrier_sm89.h"

static int qsb_b64_val(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

/* Decode the line-split base64 image. Returns a malloc'd buffer or nullptr. */
static unsigned char *qsb_carrier_decode_image(size_t expected, unsigned lines,
                                               const char *const *b64, size_t *out_len) {
    unsigned char *buf = (unsigned char *)malloc(expected + 4);
    if (!buf) return nullptr;
    size_t n = 0; unsigned acc = 0; int bits = 0;
    for (unsigned li = 0; li < lines; li++) {
        for (const unsigned char *p = (const unsigned char *)b64[li]; *p; p++) {
            int v = qsb_b64_val(*p);
            if (v < 0) continue;              /* '=' padding */
            acc = (acc << 6) | (unsigned)v; bits += 6;
            if (bits >= 8) {
                bits -= 8;
                if (n >= expected) { free(buf); return nullptr; }
                buf[n++] = (unsigned char)(acc >> bits);
            }
        }
    }
    if (n != expected) { free(buf); return nullptr; }
    *out_len = n;
    return buf;
}
static unsigned char *qsb_carrier_decode(size_t *out_len) {
    return qsb_carrier_decode_image(qsb_carrier_cubin_bytes, qsb_carrier_b64_lines,
                                    qsb_carrier_b64, out_len);
}

static void qsb_carrier_off(const char *why) {
    if (g_qsb_carrier.finish_lib) cudaLibraryUnload(g_qsb_carrier.finish_lib);
    if (g_qsb_carrier.lib) cudaLibraryUnload(g_qsb_carrier.lib);
    g_qsb_carrier.on = 0; g_qsb_carrier.lib = nullptr;
    g_qsb_carrier.finish_lib = nullptr; g_qsb_carrier.finish_k = nullptr;
    memset(g_qsb_carrier.k, 0, sizeof(g_qsb_carrier.k));
    cudaGetLastError();                        /* clear any sticky-free error from the attempt */
    printf("  Native sm_89 carrier: off (%s); using the compute_52 image\n", why);
}

static void qsb_carrier_init(const cudaDeviceProp &prop) {
    if (prop.major != 8 || prop.minor != 9) { qsb_carrier_off("device is not sm_89"); return; }
    size_t len = 0;
    unsigned char *img = qsb_carrier_decode(&len);
    if (!img) { qsb_carrier_off("embedded image failed to decode"); return; }
    cudaError_t e = cudaLibraryLoadData(&g_qsb_carrier.lib, img, nullptr, nullptr, 0,
                                        nullptr, nullptr, 0);
    free(img);
    if (e != cudaSuccess) { qsb_carrier_off(cudaGetErrorString(e)); return; }
    for (int i = 0; i < QK_N; i++) {
        if (!qsb_carrier_kernel_names[i][0]) { g_qsb_carrier.k[i] = nullptr; continue; }  /* optional, absent */
        e = cudaLibraryGetKernel(&g_qsb_carrier.k[i], g_qsb_carrier.lib, qsb_carrier_kernel_names[i]);
        if (e != cudaSuccess) { qsb_carrier_off("kernel missing from image"); return; }
    }
    void *dz = nullptr; size_t zb = 0; int zeros = -1;
    e = cudaLibraryGetGlobal(&dz, &zb, g_qsb_carrier.lib, "qsb_carrier_zeros");
    if (e == cudaSuccess && zb == sizeof(int))
        e = cudaMemcpy(&zeros, dz, sizeof(int), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess || zeros != QSB_ZEROS_N) { qsb_carrier_off("image built for another QSB_ZEROS_N"); return; }
    size_t flen = 0;
    unsigned char *fimg = qsb_carrier_decode_image(qsb_finish_cubin_bytes,
                              qsb_finish_b64_lines, qsb_finish_b64, &flen);
    if (!fimg) { qsb_carrier_off("finish image failed to decode"); return; }
    e = cudaLibraryLoadData(&g_qsb_carrier.finish_lib, fimg, nullptr, nullptr, 0,
                            nullptr, nullptr, 0);
    free(fimg);
    if (e != cudaSuccess) { qsb_carrier_off("finish image failed to load"); return; }
    e = cudaLibraryGetKernel(&g_qsb_carrier.finish_k, g_qsb_carrier.finish_lib,
                             qsb_finish_kernel_names[QK_S2]);
    if (e != cudaSuccess) { qsb_carrier_off("finish kernel missing"); return; }
    void *fz = nullptr; size_t fzb = 0; int fzeros = -1;
    e = cudaLibraryGetGlobal(&fz, &fzb, g_qsb_carrier.finish_lib, "qsb_carrier_zeros");
    if (e == cudaSuccess && fzb == sizeof(int))
        e = cudaMemcpy(&fzeros, fz, sizeof(int), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess || fzeros != QSB_ZEROS_N) {
        qsb_carrier_off("finish image built for another QSB_ZEROS_N"); return;
    }
    g_qsb_carrier.on = 1;
    printf("  Native sm_89 carrier: on (%zu-byte base %.16s..., %zu-byte exact SHA finish %.16s...)\n",
           len, qsb_carrier_cubin_sha256, flen, qsb_finish_cubin_sha256);
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
    cudaKernel_t kernel = (kid == QK_S2) ? g_qsb_carrier.finish_k : g_qsb_carrier.k[kid];
    return cudaLaunchKernel((const void *)kernel, g, b, argv, 0, st);
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
    if (g_qsb_carrier.on) {
        void *d = nullptr; size_t sz = 0;
        cudaError_t e = cudaLibraryGetGlobal(&d, &sz, g_qsb_carrier.lib, name);
        if (e != cudaSuccess) return e;
        if (n > sz) return cudaErrorInvalidValue;
        e = cudaMemcpy(d, src, n, cudaMemcpyHostToDevice);
        if (e != cudaSuccess) return e;
        void *fd = nullptr; size_t fsz = 0;
        e = cudaLibraryGetGlobal(&fd, &fsz, g_qsb_carrier.finish_lib, name);
        if (e == cudaSuccess) {
            if (n > fsz) return cudaErrorInvalidValue;
            e = cudaMemcpy(fd, src, n, cudaMemcpyHostToDevice);
            if (e != cudaSuccess) return e;
        } else {
            cudaGetLastError();
        }
#if QSB_NOJIT
        return cudaSuccess;
#endif
    }
    return cudaMemcpyToSymbol(sym, src, n);
}
#define QSB_TO_SYMBOL(sym, src, n) qsb_to_symbol(sym, #sym, src, n)
