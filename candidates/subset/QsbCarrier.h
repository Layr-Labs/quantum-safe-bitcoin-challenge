#pragma once
/* Native sm_89 carrier for the subset search.
 *
 * Design and most of this file are Ryun1's native sm_89 carrier from the pinning
 * track (public submission 25bd990a, candidates/pinning/QsbCarrier.h,
 * build_carrier.sh, CARRIER.md; GPL-3). Credit for the idea and the loader goes to
 * Ryun1. This port adapts it to the subset tree: six kernels, carrier-only startup
 * uploads with replay on fallback, a build-knob fingerprint, and per-launch fallback.
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
 * cudaLibraryLoadData and resolves six kernels (epoch groups, incremental epochs,
 * first-block states, digest, table build, table heal scan); sp_launch in tree.cu
 * launches them with cudaLaunchKernel.
 *
 * Module state. The image is a separate CUDA module with its own copy of every
 * __device__ / __constant__ global. With QSB_TO_SYMBOL, an active carrier receives
 * the startup upload and a host replay copy is logged; the JIT symbol is populated
 * only after fallback. Replay and the JIT setup hook must finish before using the
 * JIT image. No kernel writes a module global; kernels hand data to each other
 * only through shared pointer arguments. Table build and heal use the carrier
 * when available, avoiding the cold JIT path.
 *
 * Fallback. Everything is optional. If the GPU is not sm_89, QSB_CARRIER_DISABLE is
 * set in the environment, the image fails to decode or load, a kernel is missing,
 * the image's build knobs differ from this binary's (qsb_carrier_knobs), or an
 * upload into the image fails, the program prints
 * `Native sm_89 carrier: off (...)` and runs the unchanged compute_52 kernels. A
 * failed carrier launch mid-run switches to the <<<>>> launch of the same kernel
 * after restoring the JIT uploads. This is the intended fallback contract, not
 * proof that every asynchronous error/replay failure path has been exercised.
 * The exact OpenSSL host gate (QSB_HOST_VERIFY) re-derives every hit in both modes.
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

 

#ifndef QSB_CARRIER
#define QSB_CARRIER 1
#endif

 

#define QSB_CARRIER_STR2(x) #x
#define QSB_CARRIER_STR(x) QSB_CARRIER_STR2(x)
#define QSB_CARRIER_KV(name) #name "=" QSB_CARRIER_STR(name) ";"

enum QsbCarrierKernel {
QK_EG = 0,  
QK_BEI,  
QK_BFF,  
QK_DIG,  
QK_GT,  
QK_HEAL,  
#if QSB_YOFF_S
QK_YOFF,  
#endif
QK_N
};

 








struct QsbUpload { const void *sym; void *data; size_t n; };
static QsbUpload *g_qsb_uploads = nullptr;  
static int g_qsb_n_uploads = 0, g_qsb_cap_uploads = 0;
static void (*g_qsb_jit_hook)(void) = nullptr;  
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
int running;  
cudaLibrary_t lib;
cudaKernel_t k[QK_N];
};
static QsbCarrierState g_qsb_carrier = {0, 0, nullptr, {}};

static void qsb_carrier_off(const char *why) {
 
if (g_qsb_carrier.lib && !g_qsb_carrier.running) {
cudaLibraryUnload(g_qsb_carrier.lib);
g_qsb_carrier.lib = nullptr;
}
const int was_on = g_qsb_carrier.on;
g_qsb_carrier.on = 0;
memset(g_qsb_carrier.k, 0, sizeof(g_qsb_carrier.k));
cudaGetLastError();  
printf("  Native sm_89 carrier: off (%s); using the compute_52 image%s\n", why,
was_on ? " from here on" : "");
 
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

 
#ifndef QSB_CARRIER_LZ4
#define QSB_CARRIER_LZ4 0
#endif
#if QSB_CARRIER_LZ4
#include "qsb_lz4_block.h"
#endif
static unsigned char *qsb_carrier_decode(size_t *out_len) {
#if QSB_CARRIER_LZ4
const size_t payload=qsb_carrier_payload_bytes;
#else
const size_t payload=qsb_carrier_cubin_bytes;
#endif
unsigned char *buf = (unsigned char *)malloc(payload + 4);
if (!buf) return nullptr;
size_t n = 0; unsigned acc = 0; int bits = 0;
for (unsigned li = 0; li < qsb_carrier_b64_lines; li++) {
for (const unsigned char *p = (const unsigned char *)qsb_carrier_b64[li]; *p; p++) {
int v = qsb_b64_val(*p);
if (v < 0) continue;  
acc = (acc << 6) | (unsigned)v; bits += 6;
if (bits >= 8) {
bits -= 8;
if (n >= payload) { free(buf); return nullptr; }
buf[n++] = (unsigned char)(acc >> bits);
}
}
}
if (n != payload) { free(buf); return nullptr; }
#if QSB_CARRIER_LZ4
unsigned char *image=(unsigned char *)malloc(qsb_carrier_cubin_bytes);
if(!image){free(buf);return nullptr;}
bool ok=qsb_lz4_block(buf,n,image,qsb_carrier_cubin_bytes);free(buf);
if(!ok){free(image);return nullptr;}
*out_len=qsb_carrier_cubin_bytes;return image;
#else
*out_len = n;
return buf;
#endif
}

 
static void qsb_carrier_init(const cudaDeviceProp &prop, const char *knobs) {
const char *dis = getenv("QSB_CARRIER_DISABLE");
if (dis && *dis && strcmp(dis, "0") != 0) { qsb_carrier_off("disabled by QSB_CARRIER_DISABLE"); return; }
if (prop.major != 8 || prop.minor != 9) { qsb_carrier_off("device is not sm_89"); return; }
 

if (sizeof(qsb_carrier_kernel_names)/sizeof(qsb_carrier_kernel_names[0]) != QK_N) {
qsb_carrier_off("image kernel count differs from this build"); return;
}
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

 



template <class T>
static cudaError_t qsb_to_symbol(const T &sym, const char *name, const void *src, size_t n) {
if (!g_qsb_carrier.on) return cudaMemcpyToSymbol(sym, src, n);
 
if (!qsb_upload_log((const void *)&sym, src, n)) {
qsb_carrier_off("out of host memory for the upload log");  
return cudaMemcpyToSymbol(sym, src, n);
}
void *d = nullptr; size_t sz = 0;
cudaError_t ce = cudaLibraryGetGlobal(&d, &sz, g_qsb_carrier.lib, name);
if (ce == cudaErrorSymbolNotFound || ce == cudaErrorInvalidSymbol) { cudaGetLastError(); return cudaSuccess; }
if (ce == cudaSuccess && n > sz) ce = cudaErrorInvalidValue;
if (ce == cudaSuccess) ce = cudaMemcpy(d, src, n, cudaMemcpyHostToDevice);
if (ce != cudaSuccess) {
char why[160];
snprintf(why, sizeof(why), "upload of %s failed: %s", name, cudaGetErrorString(ce));
qsb_carrier_off(why);  
}
return cudaSuccess;
}
#define QSB_TO_SYMBOL(sym, src, n) qsb_to_symbol(sym, #sym, src, n)

 

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
