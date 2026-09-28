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
 * kernel_gt_heal_scan) stays on the JIT image: it writes only the table buffer.
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
    QK_GT,       /* kernel_build_gtable     */
    QK_HEAL,     /* kernel_gt_heal_scan     */
    QK_N
};

/* No-JIT startup. With the carrier on, nothing touches the compute_52 image: every
 * kernel the ranked path launches (table build and heal scan included) comes from the
 * native image, uploads go only to the image's globals, and host reads of device
 * constants read the image. Under the CUDA 12 default of lazy module loading, the
 * compute_52 PTX is then never JIT-compiled, which removes the cold JIT (a few seconds
 * of CPU on every ranked run: each run is a fresh uid, and new source means new PTX)
 * from the timed window. Each upload is logged; if the carrier is ever switched off,
 * qsb_carrier_off replays the log into the compute_52 image (JIT-compiling it then)
 * before any <<<>>> launch, so the fallback sees exactly the same data. */
struct QsbUpload { const void *sym; const char *name; void *data; size_t n; };
static QsbUpload *g_qsb_uploads = nullptr;       /* grows as needed; uploads are startup-only */
static int g_qsb_n_uploads = 0, g_qsb_cap_uploads = 0;
static void (*g_qsb_jit_hook)(void) = nullptr;   /* JIT-image-only setup, run on fallback */
static bool qsb_upload_log(const void *sym, const char *name, const void *src, size_t n) {
    if (g_qsb_n_uploads == g_qsb_cap_uploads) {
        const int cap = g_qsb_cap_uploads ? 2 * g_qsb_cap_uploads : 32;
        QsbUpload *grown = (QsbUpload *)realloc(g_qsb_uploads, (size_t)cap * sizeof(QsbUpload));
        if (!grown) return false;
        g_qsb_uploads = grown; g_qsb_cap_uploads = cap;
    }
    void *copy = malloc(n ? n : 1);
    if (!copy) return false;
    memcpy(copy, src, n);
    g_qsb_uploads[g_qsb_n_uploads++] = {sym, name, copy, n};
    return true;
}

struct QsbCarrierState {
    int on;
    int running;             /* set once the search loop may have work queued on the image */
    cudaLibrary_t lib;
    cudaKernel_t k[QK_N];
};
static QsbCarrierState g_qsb_carrier = {0, 0, nullptr, {}};

/* ---- In-run A/B (QSB_AB; host only, the device images are untouched) ----
 * A second native image B (qsb_carrier_b_sm89.h, written by build_carrier_b.sh / mk_b_header.py) is
 * loaded next to image A. Every host upload is mirrored into both images and read back from B before
 * the search loop; the A/B scheduler in tree.cu then routes the four search-loop kernels (epoch groups,
 * incremental epochs, first-block states, digest) of variant-B batches to image B through
 * g_qsb_img. Table build and heal scan stay on image A (they only write the shared table buffer).
 * B is optional: if it is absent, fails to load or verify, or a B launch fails, B is switched off
 * (never unloaded: kernels may be in flight) and its batches run on image A; nothing ever exits
 * because of B. Image A going off switches B off first. */
#ifndef QSB_AB
#define QSB_AB 1
#endif
static QsbCarrierState g_qsb_carrier_b = {0, 0, nullptr, {}};
static int g_qsb_img = 0;                 /* image of the next search-loop launch: 0 = A, 1 = B */
static char g_qsb_ab_b_why[160] = "not loaded";
static int g_qsb_ab_b_was_on = 0;
static void qsb_ab_b_off(const char *why) {
    const int was_on = g_qsb_carrier_b.on;
    g_qsb_carrier_b.on = 0;
    g_qsb_img = 0;
    snprintf(g_qsb_ab_b_why, sizeof(g_qsb_ab_b_why), "%s", why);
    cudaGetLastError();
    if (was_on) { printf("  A/B: image B off (%s); variant-B batches run on image A from here on\n", why); fflush(stdout); }
}
static inline bool qsb_ab_b_has(int kid) {
    return g_qsb_carrier_b.on && kid != QK_GT && kid != QK_HEAL && g_qsb_carrier_b.k[kid];
}

static void qsb_carrier_off(const char *why) {
    qsb_ab_b_off("image A off");
    /* Never unload an image whose kernels may still be in flight; just stop using it. */
    if (g_qsb_carrier.lib && !g_qsb_carrier.running) {
        cudaLibraryUnload(g_qsb_carrier.lib);
        g_qsb_carrier.lib = nullptr;
    }
    const int was_on = g_qsb_carrier.on;
    g_qsb_carrier.on = 0;
    memset(g_qsb_carrier.k, 0, sizeof(g_qsb_carrier.k));
    cudaGetLastError();      /* clear the non-sticky error of the failed attempt */
    printf("  Native sm_89 carrier: off (%s); using the compute_52 image%s\n", why,
           was_on ? " from here on" : "");
    /* Replay every upload that went only to the image into the compute_52 image. */
    int bad = 0;
    for (int i = 0; i < g_qsb_n_uploads; i++) {
        if (cudaMemcpyToSymbol(g_qsb_uploads[i].sym, g_qsb_uploads[i].data,
                               g_qsb_uploads[i].n) != cudaSuccess) bad++;
        free(g_qsb_uploads[i].data);
    }
    const int replayed = g_qsb_n_uploads;
    g_qsb_n_uploads = 0;
    if (replayed && cudaDeviceSynchronize() != cudaSuccess) bad++;
    if (was_on && g_qsb_jit_hook) g_qsb_jit_hook();
    if (replayed)
        printf("  compute_52 image: %d upload(s) replayed%s\n", replayed, bad ? ", SOME FAILED" : "");
    fflush(stdout);
}

#if QSB_CARRIER && !defined(QSB_CARRIER_BUILD)
#include "qsb_carrier_sm89.h"
#if QSB_AB && __has_include("qsb_carrier_b_sm89.h")
#define QSB_AB_B_IMAGE 1
namespace qsb_ab_b {                        /* same names as image A's header, own namespace */
#include "qsb_carrier_b_sm89.h"
}
#include <openssl/sha.h>
#else
#define QSB_AB_B_IMAGE 0
#endif

static int qsb_b64_val(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

/* Decode the line-split base64 image. Returns a malloc'd buffer or nullptr. */
static unsigned char *qsb_carrier_decode_img(size_t *out_len, size_t bytes, unsigned lines, const char *const *b64) {
    unsigned char *buf = (unsigned char *)malloc(bytes + 4);
    if (!buf) return nullptr;
    size_t n = 0; unsigned acc = 0; int bits = 0;
    for (unsigned li = 0; li < lines; li++) {
        for (const unsigned char *p = (const unsigned char *)b64[li]; *p; p++) {
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
static unsigned char *qsb_carrier_decode(size_t *out_len) {
    return qsb_carrier_decode_img(out_len, qsb_carrier_cubin_bytes, qsb_carrier_b64_lines, qsb_carrier_b64);
}

#if QSB_AB_B_IMAGE
/* Print the knob-by-knob difference of two "NAME=value;" fingerprints (at most `cap` bytes). */
static void qsb_ab_knob_delta(const char *a, const char *b, char *out, size_t cap) {
    size_t w = 0; out[0] = 0;
    auto val = [](const char *s, const char *name, size_t nl, char *v, size_t vc) -> bool {
        for (const char *p = s; *p;) {
            const char *eq = strchr(p, '='), *sc = strchr(p, ';');
            if (!eq || !sc) break;
            if ((size_t)(eq - p) == nl && !strncmp(p, name, nl)) {
                size_t l = (size_t)(sc - eq - 1); if (l >= vc) l = vc - 1;
                memcpy(v, eq + 1, l); v[l] = 0; return true;
            }
            p = sc + 1;
        }
        return false;
    };
    for (int pass = 0; pass < 2; pass++) {
        const char *s = pass ? b : a, *o = pass ? a : b;
        for (const char *p = s; *p;) {
            const char *eq = strchr(p, '='), *sc = strchr(p, ';');
            if (!eq || !sc) break;
            const size_t nl = (size_t)(eq - p);
            char name[96], va[96], vb[96];
            if (nl < sizeof(name)) {
                memcpy(name, p, nl); name[nl] = 0;
                const bool ha = val(a, name, nl, va, sizeof va), hb = val(b, name, nl, vb, sizeof vb);
                const bool diff = pass == 0 ? (!hb || strcmp(va, vb) != 0) : !val(o, name, nl, vb, sizeof vb);
                if (diff && w + 1 < cap)
                    w += (size_t)snprintf(out + w, cap - w, "%s%s=%s->%s", w ? " " : "", name,
                                          ha ? va : "(none)", hb ? vb : "(none)");
            }
            p = sc + 1;
        }
    }
    if (!w) snprintf(out, cap, "none (identical fingerprints)");
}

/* Load image B next to image A (A must be on). Called before the first upload. */
static void qsb_ab_b_init(const char *knobs) {
    const char *env = getenv("QSB_AB");
    if (env && *env && atoi(env) == 0) { qsb_ab_b_off("disabled by QSB_AB=0"); return; }
    namespace B = qsb_ab_b;
    size_t len = 0;
    unsigned char *img = qsb_carrier_decode_img(&len, B::qsb_carrier_cubin_bytes, B::qsb_carrier_b64_lines, B::qsb_carrier_b64);
    if (!img) { qsb_ab_b_off("embedded image B failed to decode"); return; }
    unsigned char md[32]; char hex[65];
    SHA256(img, len, md);
    for (int i = 0; i < 32; i++) snprintf(hex + 2 * i, 3, "%02x", md[i]);
    if (strcmp(hex, B::qsb_carrier_cubin_sha256) != 0) { free(img); qsb_ab_b_off("image B sha256 mismatch"); return; }
    cudaError_t e = cudaLibraryLoadData(&g_qsb_carrier_b.lib, img, nullptr, nullptr, 0, nullptr, nullptr, 0);
    free(img);
    if (e != cudaSuccess) { g_qsb_carrier_b.lib = nullptr; char w[160]; snprintf(w, sizeof w, "image B load: %s", cudaGetErrorString(e)); qsb_ab_b_off(w); return; }
    const int used[4] = {QK_EG, QK_BEI, QK_BFF, QK_DIG};
    for (int i = 0; i < 4; i++) {
        const int k = used[i];
        if (strcmp(qsb_carrier_kernel_names[k], B::qsb_carrier_kernel_names[k]) != 0 ||
            cudaLibraryGetKernel(&g_qsb_carrier_b.k[k], g_qsb_carrier_b.lib, B::qsb_carrier_kernel_names[k]) != cudaSuccess) {
            qsb_ab_b_off("image B kernel ABI differs from image A"); return;
        }
    }
    /* integrity: the image's own fingerprint is the one recorded when its header was generated */
    void *dk = nullptr; size_t kb = 0;
    const size_t want = strlen(B::qsb_carrier_b_knobs) + 1;
    e = cudaLibraryGetGlobal(&dk, &kb, g_qsb_carrier_b.lib, "qsb_carrier_knobs");
    char *ik = (char *)malloc(want);
    const bool ok = e == cudaSuccess && kb == want && ik && cudaMemcpy(ik, dk, want, cudaMemcpyDeviceToHost) == cudaSuccess &&
                    memcmp(ik, B::qsb_carrier_b_knobs, want) == 0;
    free(ik);
    if (!ok) { qsb_ab_b_off("image B fingerprint differs from its header"); return; }
    g_qsb_carrier_b.on = 1; g_qsb_ab_b_was_on = 1;
    char delta[512];
    qsb_ab_knob_delta(knobs, B::qsb_carrier_b_knobs, delta, sizeof delta);
    printf("  A/B: image B loaded (%zu-byte image, sha256 %.16s..., build flags '%s'; knob delta vs A: %s)\n",
           len, B::qsb_carrier_cubin_sha256, B::qsb_carrier_b_flags, delta);
    fflush(stdout);
}

/* After every upload, before the search loop: each logged upload must read back byte-identical from
 * image B (where B has that global, with image A's size); image-initialized globals that differ from
 * image A's are listed (a code variant may change a device table). Any mismatch switches B off. */
static void qsb_ab_b_verify() {
    if (!g_qsb_carrier_b.on || !g_qsb_carrier.on) return;
    namespace B = qsb_ab_b;
    int mirrored = 0;
    for (int i = 0; i < g_qsb_n_uploads; i++) {
        const QsbUpload &u = g_qsb_uploads[i];
        void *db = nullptr, *da = nullptr; size_t sb = 0, sa = 0;
        const cudaError_t eb = cudaLibraryGetGlobal(&db, &sb, g_qsb_carrier_b.lib, u.name);
        if (eb != cudaSuccess) { cudaGetLastError(); continue; }        /* B does not reference it */
        const cudaError_t ea = cudaLibraryGetGlobal(&da, &sa, g_qsb_carrier.lib, u.name);
        if (ea != cudaSuccess || sa != sb) { cudaGetLastError(); qsb_ab_b_off("uploaded global size differs between A and B"); return; }
        void *h = malloc(u.n ? u.n : 1);
        const bool same = h && cudaMemcpy(h, db, u.n, cudaMemcpyDeviceToHost) == cudaSuccess && memcmp(h, u.data, u.n) == 0;
        free(h);
        if (!same) { char w[160]; snprintf(w, sizeof w, "upload %s did not read back from image B", u.name); qsb_ab_b_off(w); return; }
        mirrored++;
    }
    char diff[512] = ""; size_t w = 0; int ndiff = 0, nsame = 0;
    for (unsigned i = 0; i < B::qsb_carrier_b_n_globals; i++) {
        const char *nm = B::qsb_carrier_b_globals[i].name;
        if (!strcmp(nm, "qsb_carrier_knobs")) continue;
        bool uploaded = false;
        for (int j = 0; j < g_qsb_n_uploads; j++) if (!strcmp(g_qsb_uploads[j].name, nm)) uploaded = true;
        if (uploaded) continue;
        void *db = nullptr, *da = nullptr; size_t sb = 0, sa = 0;
        bool same = false;
        if (cudaLibraryGetGlobal(&db, &sb, g_qsb_carrier_b.lib, nm) == cudaSuccess &&
            cudaLibraryGetGlobal(&da, &sa, g_qsb_carrier.lib, nm) == cudaSuccess && sa == sb) {
            void *ha = malloc(sa ? sa : 1), *hb = malloc(sb ? sb : 1);
            same = ha && hb && cudaMemcpy(ha, da, sa, cudaMemcpyDeviceToHost) == cudaSuccess &&
                   cudaMemcpy(hb, db, sb, cudaMemcpyDeviceToHost) == cudaSuccess && memcmp(ha, hb, sa) == 0;
            free(ha); free(hb);
        }
        cudaGetLastError();
        if (same) { nsame++; continue; }
        ndiff++;
        if (w + 1 < sizeof diff) w += (size_t)snprintf(diff + w, sizeof diff - w, "%s%s", w ? "," : "", nm);
    }
    printf("  A/B: image B verified: %d upload(s) read back identical from B; image-initialized globals: %d identical to A, %d differ%s%s\n",
           mirrored, nsame, ndiff, ndiff ? ": " : "", diff);
    fflush(stdout);
}
#endif

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
#if QSB_AB_B_IMAGE
    qsb_ab_b_init(knobs);
#else
    qsb_ab_b_off(QSB_AB ? "no image B in this build (qsb_carrier_b_sm89.h absent)" : "QSB_AB=0 at compile time");
#endif
}
#else
static void qsb_carrier_init(const cudaDeviceProp &, const char *) {}
#endif

static inline bool qsb_carrier_has(int kid) { return g_qsb_carrier.on && g_qsb_carrier.k[kid]; }

/* Launch the carrier image of `kern`. The static kernel pointer only supplies the
 * parameter types: every argument is converted to its declared parameter type before
 * its address goes to cudaLaunchKernel, exactly as a <<<>>> launch would. */
template <typename... P, typename... A, size_t... I>
static cudaError_t qsb_carrier_launch_impl(cudaKernel_t kernel, dim3 g, dim3 b, cudaStream_t st,
                                           std::index_sequence<I...>, A &&...a) {
    std::tuple<typename std::decay<P>::type...> vals(std::forward<A>(a)...);
    void *argv[sizeof...(P) > 0 ? sizeof...(P) : 1] = {(void *)&std::get<I>(vals)...};
    return cudaLaunchKernel((const void *)kernel, g, b, argv, 0, st);
}
template <typename... P, typename... A>
static cudaError_t qsb_carrier_launch(void (*)(P...), int kid, dim3 g, dim3 b, cudaStream_t st,
                                      A &&...a) {
    static_assert(sizeof...(P) == sizeof...(A), "carrier launch: argument count mismatch");
    return qsb_carrier_launch_impl<P...>(g_qsb_carrier.k[kid], g, b, st, std::index_sequence_for<P...>{},
                                         std::forward<A>(a)...);
}
/* Try the carrier launch; on failure switch the carrier off (the caller then issues
 * the <<<>>> launch of the same kernel). Returns true when the carrier launched it. */
template <typename... P, typename... A>
static bool qsb_carrier_try(void (*kern)(P...), int kid, dim3 g, dim3 b, cudaStream_t st,
                            A &&...a) {
    if (!qsb_carrier_has(kid)) return false;
    if (g_qsb_img == 1 && qsb_ab_b_has(kid)) {   /* A/B: variant-B batch on image B (arguments copied) */
        static_assert(sizeof...(P) == sizeof...(A), "carrier launch: argument count mismatch");
        const cudaError_t eb = qsb_carrier_launch_impl<P...>(g_qsb_carrier_b.k[kid], g, b, st,
                                                             std::index_sequence_for<P...>{}, a...);
        if (eb == cudaSuccess) return true;
        char whyb[160];
        snprintf(whyb, sizeof(whyb), "launch failed: %s", cudaGetErrorString(eb));
        qsb_ab_b_off(whyb);                        /* then the same launch on image A below */
    }
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
    /* Carrier on: write the image only and log the upload for a possible fallback. */
    if (!qsb_upload_log((const void *)&sym, name, src, n)) {
        qsb_carrier_off("out of host memory for the upload log");   /* replays the earlier uploads */
        return cudaMemcpyToSymbol(sym, src, n);
    }
    if (g_qsb_carrier_b.on) {                         /* A/B: the same bytes into image B's global */
        void *db = nullptr; size_t szb = 0;
        cudaError_t cb = cudaLibraryGetGlobal(&db, &szb, g_qsb_carrier_b.lib, name);
        if (cb == cudaErrorSymbolNotFound || cb == cudaErrorInvalidSymbol) cudaGetLastError();   /* B does not use it */
        else {
            if (cb == cudaSuccess && n > szb) cb = cudaErrorInvalidValue;
            if (cb == cudaSuccess) cb = cudaMemcpy(db, src, n, cudaMemcpyHostToDevice);
            if (cb != cudaSuccess) {
                char whyb[160];
                snprintf(whyb, sizeof(whyb), "upload of %s into image B failed: %s", name, cudaGetErrorString(cb));
                qsb_ab_b_off(whyb);
            }
        }
    }
    void *d = nullptr; size_t sz = 0;
    cudaError_t ce = cudaLibraryGetGlobal(&d, &sz, g_qsb_carrier.lib, name);
    if (ce == cudaErrorSymbolNotFound || ce == cudaErrorInvalidSymbol) { cudaGetLastError(); return cudaSuccess; }
    if (ce == cudaSuccess && n > sz) ce = cudaErrorInvalidValue;
    if (ce == cudaSuccess) ce = cudaMemcpy(d, src, n, cudaMemcpyHostToDevice);
    if (ce != cudaSuccess) {
        char why[160];
        snprintf(why, sizeof(why), "upload of %s failed: %s", name, cudaGetErrorString(ce));
        qsb_carrier_off(why);                         /* replays this upload too */
    }
    return cudaSuccess;
}
#define QSB_TO_SYMBOL(sym, src, n) qsb_to_symbol(sym, #sym, src, n)

/* cudaMemcpyFromSymbol that reads the image's copy while the carrier is on (the host
 * only reads constants both images define identically, e.g. the SHA-256 K table). */
template <class T>
static cudaError_t qsb_from_symbol(void *dst, const T &sym, const char *name, size_t n) {
    if (g_qsb_carrier.on) {
        void *d = nullptr; size_t sz = 0;
        cudaError_t ce = cudaLibraryGetGlobal(&d, &sz, g_qsb_carrier.lib, name);
        if (ce == cudaSuccess && n <= sz) ce = cudaMemcpy(dst, d, n, cudaMemcpyDeviceToHost);
        else if (ce == cudaSuccess) ce = cudaErrorInvalidValue;
        if (ce == cudaSuccess) return cudaSuccess;
        cudaGetLastError();
    }
    return cudaMemcpyFromSymbol(dst, sym, n);
}
#define QSB_FROM_SYMBOL(dst, sym, n) qsb_from_symbol(dst, sym, #sym, n)
