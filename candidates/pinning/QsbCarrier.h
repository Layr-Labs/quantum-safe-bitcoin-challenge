#pragma once
/* Native-image loader adapted from terrapinelf, public PR #1471.
 * This experiment changes only the SASS order of four independent table loads.
 * No L2 hint, GLV rewrite, or batch-size change is imported.
 * GPLv3 as part of the existing pinning executable; see COPYING.
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
#ifndef QSB_SASS_PERMUTE
#define QSB_SASS_PERMUTE 1
#endif
#if QSB_SASS_PERMUTE != 0 && QSB_SASS_PERMUTE != 1
#error "QSB_SASS_PERMUTE must be 0 or 1"
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
    cudaLibrary_t control_lib;
    cudaKernel_t control_kernel;
};
static QsbCarrierState g_qsb_carrier = {};

#if QSB_CARRIER && !defined(QSB_CARRIER_BUILD)
#include "qsb_carrier_sm89.h"
namespace qsb_control_image {
#include "qsb_carrier_control_sm89.h"
}

static int qsb_b64_val(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

/* Decode the line-split base64 image. Returns a malloc'd buffer or nullptr. */
static unsigned char *qsb_carrier_decode(size_t *out_len, bool control = false) {
    control = control || !QSB_SASS_PERMUTE;
    const size_t bytes = control ? qsb_control_image::qsb_carrier_cubin_bytes : qsb_carrier_cubin_bytes;
    const unsigned lines = control ? qsb_control_image::qsb_carrier_b64_lines : qsb_carrier_b64_lines;
    const char *const *data = control ? qsb_control_image::qsb_carrier_b64 : qsb_carrier_b64;
    unsigned char *buf = (unsigned char *)malloc(bytes + 4);
    if (!buf) return nullptr;
    size_t n = 0; unsigned acc = 0; int bits = 0;
    for (unsigned li = 0; li < lines; li++) {
        for (const unsigned char *p = (const unsigned char *)data[li]; *p; p++) {
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
    img = qsb_carrier_decode(&len, true);
    if (!img) { fprintf(stderr,"SASS control image decode failed\n"); exit(2); }
    e = cudaLibraryLoadData(&g_qsb_carrier.control_lib, img, nullptr, nullptr, 0, nullptr, nullptr, 0);
    free(img);
    if (e != cudaSuccess || cudaLibraryGetKernel(&g_qsb_carrier.control_kernel,
         g_qsb_carrier.control_lib, qsb_carrier_kernel_names[QK_S0]) != cudaSuccess) {
        fprintf(stderr,"SASS control image load failed\n"); exit(2);
    }
    g_qsb_carrier.on = 1;
    printf("  Native sm_89 carrier: on (%zu-byte image, sha256 %.16s..., independent-load schedule)\n",
           len, QSB_SASS_PERMUTE ? qsb_carrier_cubin_sha256 : qsb_control_image::qsb_carrier_cubin_sha256);
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
    return cudaLaunchKernel((const void *)(kid == QK_N ? g_qsb_carrier.control_kernel : g_qsb_carrier.k[kid]), g, b, argv, 0, st);
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

/* cudaMemcpyToSymbol that targets the carrier image's copy of the symbol when it is on. */
template <class T>
static cudaError_t qsb_to_symbol(const T &sym, const char *name, const void *src, size_t n) {
    if (g_qsb_carrier.on) {
        void *d = nullptr; size_t sz = 0;
        cudaError_t e = cudaLibraryGetGlobal(&d, &sz, g_qsb_carrier.lib, name);
        if (e != cudaSuccess) return e;
        if (n > sz) return cudaErrorInvalidValue;
        e = cudaMemcpy(d, src, n, cudaMemcpyHostToDevice);
        if (e != cudaSuccess) return e;
        e = cudaLibraryGetGlobal(&d, &sz, g_qsb_carrier.control_lib, name);
        if (e != cudaSuccess || n > sz) return e != cudaSuccess ? e : cudaErrorInvalidValue;
        return cudaMemcpy(d, src, n, cudaMemcpyHostToDevice);
    }
    return cudaMemcpyToSymbol(sym, src, n);
}
#define QSB_TO_SYMBOL(sym, src, n) qsb_to_symbol(sym, #sym, src, n)
