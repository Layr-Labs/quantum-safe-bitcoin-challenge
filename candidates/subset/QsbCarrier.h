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

/* QSB_ARMS (host only, default 1): in-run multi-arm A/B probe (see the "Multi-arm probe" section
 * below and challenges' PROBE notes). N > 1 loads the first N images of qsb_carrier_sm89.h (the
 * ARMS table of build_carrier.sh: this same source built with each arm's extra device -D flags)
 * and time-slices their digest kernels over disjoint epoch ranges. 1 compiles every probe line
 * out: the base single-image carrier and search, unchanged. QSB_PROBE is the effective switch:
 * never inside the carrier image build itself, and only with the embedded carrier. */
#ifndef QSB_ARMS
#define QSB_ARMS 1
#endif
#if QSB_ARMS < 1 || QSB_ARMS > 8
#error "QSB_ARMS must be 1..8"
#endif
#if QSB_ARMS > 1 && QSB_CARRIER && !defined(QSB_CARRIER_BUILD)
#define QSB_PROBE 1
#else
#define QSB_PROBE 0
#endif
/* Device-only knobs an arm image may set differently from this binary's host pass (everything
 * else in the knob fingerprint must match byte for byte). Each one changes only kernel_digest's
 * code: no host path, table layout, upload or producer depends on it. */
#ifndef QSB_PROBE_FREE_KNOBS
#define QSB_PROBE_FREE_KNOBS "ZLAB_DUAL_EPOCH_SHA", "QSB_Q_MIX", "QSB_GATE_PAIR", \
    "QSB_PAIR_SHA_UNROLL_CONST", "QSB_PAIR_SHA_UNROLL_CONST_INNER", "QSB_PAIR_SHA_UNROLL_WINDOW", \
    "QSB_SM_SKEW_NS", "QSB_DIGEST_MINB", "QSB_ROOT_LUT_SMEM", "QSB_SHA_CONST_PEEL", "QSB_PSI_HOIST", \
    "QSB_Q_SPREAD", "QSB_TREE_UNROLL", "QSB_GATHER_L1_POLICY", "QSB_CODE_ROLL", "QSB_WSEC_L1LAST"
#endif

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
struct QsbUpload { const void *sym; void *data; size_t n; };
static QsbUpload *g_qsb_uploads = nullptr;       /* grows as needed; uploads are startup-only */
static int g_qsb_n_uploads = 0, g_qsb_cap_uploads = 0;
static void (*g_qsb_jit_hook)(void) = nullptr;   /* JIT-image-only setup, run on fallback */
static bool qsb_upload_log(const void *sym, const void *src, size_t n) {
    if (g_qsb_n_uploads == g_qsb_cap_uploads) {
        const int cap = g_qsb_cap_uploads ? 2 * g_qsb_cap_uploads : 32;
        QsbUpload *grown = (QsbUpload *)realloc(g_qsb_uploads, (size_t)cap * sizeof(QsbUpload));
        if (!grown) return false;
        g_qsb_uploads = grown; g_qsb_cap_uploads = cap;
    }
    void *copy = malloc(n ? n : 1);
    if (!copy) return false;
    memcpy(copy, src, n);
    g_qsb_uploads[g_qsb_n_uploads++] = {sym, copy, n};
    return true;
}

struct QsbCarrierState {
    int on;
    int running;             /* set once the search loop may have work queued on the image */
    cudaLibrary_t lib;
    cudaKernel_t k[QK_N];
};
static QsbCarrierState g_qsb_carrier = {0, 0, nullptr, {}};

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

#if QSB_PROBE
#include <stdint.h>
#include <time.h>
/* ---- Multi-arm probe: loader ----
 * Arm a is image a of qsb_carrier_sm89.h. Arm 0 is the control image and is loaded by the
 * unchanged code below into g_qsb_carrier; every other arm is its own library. Only the DIGEST
 * kernel differs between arms: the producer kernels (epoch groups, epochs_inc, first-block
 * states) and the start-up table kernels always run from arm 0, so every arm consumes
 * bit-identical epoch descriptors and the arm effect is kernel_digest's alone. The search loop
 * switches g_qsb_carrier.k[QK_DIG] (qsb_arm_select) only between slices, with every slot
 * drained. */
#if !defined(QSB_CARRIER_ARMS) || QSB_ARMS > QSB_CARRIER_ARMS
#error "QSB_ARMS exceeds the arm count of qsb_carrier_sm89.h (regenerate it with build_carrier.sh)"
#endif
struct QsbArm { cudaLibrary_t lib; cudaKernel_t dig; };
static QsbArm g_qsb_arm[QSB_ARMS];
static int g_qsb_arms_on = 1;          /* arms loaded and resolved; 1 = the base single-arm path */

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

/* The arm image's knob string must equal this binary's except for the values of the device-only
 * knobs in QSB_PROBE_FREE_KNOBS (same knobs, same order). */
static bool qsb_knob_is_free(const char *name, size_t len) {
    static const char *const free_knobs[] = {QSB_PROBE_FREE_KNOBS};
    for (size_t i = 0; i < sizeof(free_knobs) / sizeof(free_knobs[0]); i++)
        if (strlen(free_knobs[i]) == len && memcmp(free_knobs[i], name, len) == 0) return true;
    return false;
}
static bool qsb_knobs_compatible(const char *mine, const char *arm) {
    for (;;) {
        const char *em = strchr(mine, ';'), *ea = strchr(arm, ';');
        if (!em || !ea) return !em && !ea && !*mine && !*arm;
        const char *qm = (const char *)memchr(mine, '=', (size_t)(em - mine));
        const char *qa = (const char *)memchr(arm, '=', (size_t)(ea - arm));
        if (!qm || !qa || qm - mine != qa - arm || memcmp(mine, arm, (size_t)(qm - mine)) != 0) return false;
        const bool same = (em - mine) == (ea - arm) && memcmp(mine, arm, (size_t)(em - mine)) == 0;
        if (!same && !qsb_knob_is_free(mine, (size_t)(qm - mine))) return false;
        mine = em + 1; arm = ea + 1;
    }
}

/* Runs at the end of a successful qsb_carrier_init (arm 0 = g_qsb_carrier). Every arm must load,
 * resolve the digest kernel under arm 0's name, and carry a compatible knob string. Any failure
 * unloads the extra libraries and leaves g_qsb_arms_on = 1: the base search runs unchanged. */
static void qsb_arms_init(const char *knobs) {
    g_qsb_arm[0].lib = g_qsb_carrier.lib;
    g_qsb_arm[0].dig = g_qsb_carrier.k[QK_DIG];
    printf("  Probe arm 0: sha256 %.16s... [%s] loaded OK (control image)\n",
           qsb_carrier_arm_sha256[0], qsb_carrier_arm_flags[0]);
    const char *why = nullptr;
    int a = 1;
    for (; a < QSB_ARMS; a++) {
        QsbArm &A = g_qsb_arm[a];
        A.lib = nullptr; A.dig = nullptr;
        size_t len = 0;
        unsigned char *img = qsb_arm_decode(a, &len);
        if (!img) { why = "embedded image failed to decode"; break; }
        cudaError_t e = cudaLibraryLoadData(&A.lib, img, nullptr, nullptr, 0, nullptr, nullptr, 0);
        free(img);
        if (e != cudaSuccess) { A.lib = nullptr; why = cudaGetErrorString(e); break; }
        if (cudaLibraryGetKernel(&A.dig, A.lib, qsb_carrier_kernel_names[QK_DIG]) != cudaSuccess || !A.dig) {
            why = "digest kernel missing from image"; break;
        }
        void *dk = nullptr; size_t kb = 0;
        e = cudaLibraryGetGlobal(&dk, &kb, A.lib, "qsb_carrier_knobs");
        char *ak = (e == cudaSuccess && kb > 0 && kb < 65536) ? (char *)malloc(kb + 1) : nullptr;
        if (!ak) { why = "image has no readable knob string"; break; }
        e = cudaMemcpy(ak, dk, kb, cudaMemcpyDeviceToHost);
        ak[kb] = 0;
        const bool ok = e == cudaSuccess && qsb_knobs_compatible(knobs, ak);
        free(ak);
        if (!ok) { why = "image built with incompatible knobs"; break; }
        /* The base's shared-memory carveout hint, on this arm's digest kernel too (advisory). */
        if (cudaFuncSetAttribute((const void *)A.dig, cudaFuncAttributePreferredSharedMemoryCarveout,
                                 cudaSharedmemCarveoutMaxShared) != cudaSuccess) cudaGetLastError();
        /* Advisory preload: under lazy loading this loads the kernel now instead of at the arm's
         * first launch (the first round of slices is excluded from the analysis either way). The
         * register count identifies the arm build in the log (build_carrier.sh prints it too). */
        int regs = -1;
        cudaFuncAttributes fa;
        if (cudaFuncGetAttributes(&fa, (const void *)A.dig) == cudaSuccess) regs = fa.numRegs;
        else cudaGetLastError();
        printf("  Probe arm %d: sha256 %.16s... [%s] loaded OK (%zu-byte image, digest regs %d)\n",
               a, qsb_carrier_arm_sha256[a], qsb_carrier_arm_flags[a], len, regs);
    }
    if (why) {
        for (int b = 1; b < QSB_ARMS; b++) {
            if (g_qsb_arm[b].lib) cudaLibraryUnload(g_qsb_arm[b].lib);
            g_qsb_arm[b].lib = nullptr; g_qsb_arm[b].dig = nullptr;
        }
        cudaGetLastError();                    /* clear the non-sticky error of the attempt */
        g_qsb_arms_on = 1;
        printf("  Probe arms: off (arm %d: %s); running the base single-arm search\n", a, why);
        fflush(stdout);
        return;
    }
    g_qsb_arms_on = QSB_ARMS;
    fflush(stdout);
}
/* Upload into every extra arm's global of that name (arm 0 = g_qsb_carrier is written by the
 * caller). A failure switches the probe off: the search has not started, so every launch then
 * comes from arm 0 again. */
static void qsb_arms_upload(const char *name, const void *src, size_t n) {
    for (int a = 1; a < g_qsb_arms_on; a++) {
        void *d = nullptr; size_t sz = 0;
        cudaError_t ce = cudaLibraryGetGlobal(&d, &sz, g_qsb_arm[a].lib, name);
        if (ce == cudaErrorSymbolNotFound || ce == cudaErrorInvalidSymbol) { cudaGetLastError(); continue; }
        if (ce == cudaSuccess && n > sz) ce = cudaErrorInvalidValue;
        if (ce == cudaSuccess) ce = cudaMemcpy(d, src, n, cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) {
            cudaGetLastError();
            printf("  Probe arms: off (upload of %s into arm %d failed: %s); running the base single-arm search\n",
                   name, a, cudaGetErrorString(ce));
            fflush(stdout);
            g_qsb_arms_on = 1;
            return;
        }
    }
}
/* Only between slices, with every slot drained: later digest launches come from arm a. */
static void qsb_arm_select(int a) { g_qsb_carrier.k[QK_DIG] = g_qsb_arm[a].dig; }

/* ---- Multi-arm probe: slice schedule (host) ----
 * Encoding: the C(137,6) epoch ranks are split into QSB_ARMS equal arm spans A = floor(R/N);
 * each span into QSB_PROBE_SLOTS equal slice slots S = floor(A/QSB_PROBE_SLOTS). Arm k's slice j
 * searches epochs [k*A + j*S, k*A + j*S + w) with its own w, from the slot start upward, in
 * batches of QSB_PROBE_BATCH epochs (x the GPU's 128 window triples), for QSB_PROBE_SLICE_MS
 * of wall time or until the slot is full (w = S, "exhausted"). Round robin: slice g = j*N + k.
 * A slice starts with both slots idle and ends with both slots drained, so every candidate of
 * a slice is searched by exactly one arm's digest kernel, and no epoch rank is searched twice.
 * Work of (k, j) = highest verified GPU-hit epoch rank in its slot - slot start + 1
 * (challenges/qsb-tools/qsbprobe_subset.py). */
#ifndef QSB_PROBE_SLOTS
#define QSB_PROBE_SLOTS 256       /* slices per arm (7 arms x 256 x 0.715 s = 1281 s > a 1200 s run) */
#endif
#ifndef QSB_PROBE_SLICE_MS
#define QSB_PROBE_SLICE_MS 700    /* ~3.5M epochs at the base rate; a slot holds 4.59M (7 arms): +30% headroom */
#endif
#ifndef QSB_PROBE_BATCH
#define QSB_PROBE_BATCH 32768     /* epochs per launch (~6.5 ms): ~115 batches per slice */
#endif
#ifndef QSB_PROBE_DITHER_MS
#define QSB_PROBE_DITHER_MS 8     /* > one batch */
#endif
#ifndef QSB_PROBE_PRINT_ROUNDS
#define QSB_PROBE_PRINT_ROUNDS 40
#endif
struct QsbProbe {
    int n, in_slice, exhausted, arm;
    unsigned j;
    double t_slice, t0, t_end;
    uint64_t A, S, pos, per_epoch, nb, g;
    unsigned long long slices[QSB_ARMS], batches[QSB_ARMS], epochs[QSB_ARMS], warm_epochs[QSB_ARMS],
        exhausted_n[QSB_ARMS];
    double warm_secs[QSB_ARMS];     /* wall time of every slice but the arm's first */
};
static QsbProbe g_qsb_probe;
static double qsb_probe_now() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
/* Cumulative per-arm lines. The unit Mc/s keeps them out of the harness's M/s rate parser. */
static void qsb_probe_print(const char *tag) {
    const QsbProbe &P = g_qsb_probe;
    for (int a = 0; a < P.n; a++) {
        const double ws = P.warm_secs[a];
        const double rate = ws > 0 ? (double)P.warm_epochs[a] * (double)P.per_epoch / ws / 1e6 : 0.0;
        printf("  qsbprobe%s arm %d: slices %llu batches %llu epochs %llu secs %.2f rate %.3fMc/s "
               "exhausted %llu [%s]\n", tag, a, P.slices[a], P.batches[a], P.epochs[a], ws, rate,
               P.exhausted_n[a], qsb_carrier_arm_flags[a]);
    }
    fflush(stdout);
}
/* After the startup uploads, before the search loop. False: run the base loop unchanged. */
static bool qsb_probe_begin(int effective_total, int se_mode, uint64_t n_epochs, uint64_t per_epoch) {
    if (!g_qsb_carrier.on || g_qsb_arms_on < 2) {
        printf("  Probe arms: off (%s); running the base single-arm search\n",
               g_qsb_carrier.on ? "extra arms not loaded" : "carrier off");
        return false;
    }
    if (effective_total != 1 || !se_mode || n_epochs < (uint64_t)g_qsb_arms_on * QSB_PROBE_SLOTS * QSB_PROBE_BATCH) {
        qsb_arm_select(0);
        printf("  Probe arms: off (multi-GPU or not the short-epoch shape); running the base single-arm search\n");
        return false;
    }
    memset(&g_qsb_probe, 0, sizeof(g_qsb_probe));
    QsbProbe &P = g_qsb_probe;
    P.n = g_qsb_arms_on;
    P.t_slice = QSB_PROBE_SLICE_MS * 1e-3;
    P.A = n_epochs / (uint64_t)P.n;
    P.S = P.A / QSB_PROBE_SLOTS;
    P.per_epoch = per_epoch;
    printf("  Probe: %d arms, round robin, %d ms slices of %d-epoch batches; arm k slice j searches epoch ranks "
           "[k*%llu + j*%llu, +%llu)\n", P.n, (int)QSB_PROBE_SLICE_MS, (int)QSB_PROBE_BATCH,
           (unsigned long long)P.A, (unsigned long long)P.S, (unsigned long long)P.S);
    fflush(stdout);
    return true;
}
/* Before each launch, with the slot about to be used already collected.
 * 0: launch *n epochs from rank *base. 1: the slice is over (drain both slots, then
 * qsb_probe_slice_end). 2: every slot of the schedule is used (end the search). */
static int qsb_probe_next(uint64_t *base, int *n) {
    QsbProbe &P = g_qsb_probe;
    if (!P.in_slice) {
        P.arm = (int)(P.g % (uint64_t)P.n);
        P.j = (unsigned)(P.g / (uint64_t)P.n);
        if (P.j >= QSB_PROBE_SLOTS) return 2;
        qsb_arm_select(P.arm);
        P.in_slice = 1; P.pos = 0; P.nb = 0; P.exhausted = 0;
        P.t0 = qsb_probe_now();
        /* Per-round dither of +-QSB_PROBE_DITHER_MS (same for every arm of round j, so the
         * paired analysis cancels it): with ~115 batches per slice the batch count would
         * otherwise be a step function of the arm's rate. */
        uint32_t h = P.j * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        P.t_end = P.t_slice + QSB_PROBE_DITHER_MS * 1e-3 * ((double)(h & 0xffffu) / 32768.0 - 1.0);
    } else if (qsb_probe_now() - P.t0 >= P.t_end) {
        return 1;                       /* never before the slice's first batch */
    }
    if (P.pos >= P.S) { P.exhausted = 1; return 1; }
    const uint64_t left = P.S - P.pos;
    *n = (int)(left < (uint64_t)QSB_PROBE_BATCH ? left : (uint64_t)QSB_PROBE_BATCH);
    *base = (uint64_t)P.arm * P.A + (uint64_t)P.j * P.S + P.pos;
    P.pos += (uint64_t)*n;
    P.nb++;
    return 0;
}
/* After the slice's slots are drained. */
static void qsb_probe_slice_end() {
    QsbProbe &P = g_qsb_probe;
    if (!P.in_slice) return;
    const double secs = qsb_probe_now() - P.t0;
    const int a = P.arm;
    P.slices[a]++; P.batches[a] += P.nb; P.epochs[a] += P.pos; P.exhausted_n[a] += (unsigned long long)P.exhausted;
    if (P.j > 0) { P.warm_epochs[a] += P.pos; P.warm_secs[a] += secs; }
    P.in_slice = 0;
    P.g++;
    if (P.g % ((uint64_t)P.n * QSB_PROBE_PRINT_ROUNDS) == 0) qsb_probe_print("");
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
#if QSB_PROBE
    qsb_arms_init(knobs);
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
    /* Carrier on: write the image only and log the upload for a possible fallback. */
    if (!qsb_upload_log((const void *)&sym, src, n)) {
        qsb_carrier_off("out of host memory for the upload log");   /* replays the earlier uploads */
        return cudaMemcpyToSymbol(sym, src, n);
    }
    void *d = nullptr; size_t sz = 0;
    cudaError_t ce = cudaLibraryGetGlobal(&d, &sz, g_qsb_carrier.lib, name);
    if (ce == cudaErrorSymbolNotFound || ce == cudaErrorInvalidSymbol) {
        cudaGetLastError();
#if QSB_PROBE
        if (g_qsb_arms_on > 1) qsb_arms_upload(name, src, n);   /* an arm image may still hold it */
#endif
        return cudaSuccess;
    }
    if (ce == cudaSuccess && n > sz) ce = cudaErrorInvalidValue;
    if (ce == cudaSuccess) ce = cudaMemcpy(d, src, n, cudaMemcpyHostToDevice);
    if (ce != cudaSuccess) {
        char why[160];
        snprintf(why, sizeof(why), "upload of %s failed: %s", name, cudaGetErrorString(ce));
        qsb_carrier_off(why);                         /* replays this upload too */
        return cudaSuccess;
    }
#if QSB_PROBE
    if (g_qsb_arms_on > 1) qsb_arms_upload(name, src, n);   /* every arm reads the same data */
#endif
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
