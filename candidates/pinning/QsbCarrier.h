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
 * Every step is optional. If the GPU is not sm_89, the image is missing, any kernel
 * or symbol fails to resolve, or the image was built for a different QSB_ZEROS_N,
 * the program runs the unchanged compute_52 kernels. The exact OpenSSL host gate
 * still checks every published hit in both modes.
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

enum QsbCarrierKernel {
    QK_S0 = 0,   /* kernel_pinning_pipeline<true,0>  (prepare) */
    QK_S2,       /* kernel_pinning_pipeline<true,2>  (finish)  */
    QK_RGP,      /* qsb_root_group_prepare */
    QK_ISR,      /* qsb_invert_super_roots */
    QK_RGF,      /* qsb_root_group_finish  */
    QK_BUILD,    /* kernel_build_gtable    */
    QK_YOFF,     /* qsb_table_offset_y     */
    QK_N
};

struct QsbCarrierState {
    int on;
    cudaLibrary_t lib;
    cudaKernel_t k[QK_N];
};
static QsbCarrierState g_qsb_carrier = {0, nullptr, {}};

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
    g_qsb_carrier.on = 0; g_qsb_carrier.lib = nullptr;
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
        e = cudaLibraryGetKernel(&g_qsb_carrier.k[i], g_qsb_carrier.lib, qsb_carrier_kernel_names[i]);
        if (e != cudaSuccess) { qsb_carrier_off("kernel missing from image"); return; }
    }
    void *dz = nullptr; size_t zb = 0; int zeros = -1;
    e = cudaLibraryGetGlobal(&dz, &zb, g_qsb_carrier.lib, "qsb_carrier_zeros");
    if (e == cudaSuccess && zb == sizeof(int))
        e = cudaMemcpy(&zeros, dz, sizeof(int), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess || zeros != QSB_ZEROS_N) { qsb_carrier_off("image built for another QSB_ZEROS_N"); return; }
    g_qsb_carrier.on = 1;
    printf("  Native sm_89 carrier: on (%zu-byte image, sha256 %.16s..., L2::64B record loads)\n",
           len, qsb_carrier_cubin_sha256);
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
    return qsb_carrier_launch_impl<P...>(kid, g, b, st, std::index_sequence_for<P...>{},
                                         std::forward<A>(a)...);
}

/* cudaMemcpyToSymbol that targets the carrier image's copy of the symbol when it is on. */
template <class T>
static cudaError_t qsb_to_symbol(const T &sym, const char *name, const void *src, size_t n) {
    if (g_qsb_carrier.on) {
        void *d = nullptr; size_t sz = 0;
        cudaError_t e = cudaLibraryGetGlobal(&d, &sz, g_qsb_carrier.lib, name);
        if (e != cudaSuccess) return e;
        if (n > sz) return cudaErrorInvalidValue;
        return cudaMemcpy(d, src, n, cudaMemcpyHostToDevice);
    }
    return cudaMemcpyToSymbol(sym, src, n);
}
#define QSB_TO_SYMBOL(sym, src, n) qsb_to_symbol(sym, #sym, src, n)
