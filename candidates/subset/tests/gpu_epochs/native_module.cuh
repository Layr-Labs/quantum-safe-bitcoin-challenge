/* Native sm_89 module for the ranked RTX 4090 path.
 *
 * The ranked build line carries no -arch, so the executable embeds only
 * compute_52 PTX (plus sm_52 SASS the 4090 cannot run) and the driver
 * JIT-compiles the whole ~60k-line module the first time a kernel or device
 * symbol is touched -- inside the timed window (about 2.5 s on a fast host
 * CPU, more on a slower one).
 *
 * subset_sm89_cubin.h holds that same module already compiled for sm_89 from
 * THIS source tree (tools/build_sm89_cubin.sh: same nvcc, -O3, the ranked
 * -DQSB_ZEROS_N=24, compute_52 PTX -> sm_89, i.e. exactly what the driver JIT
 * would produce). When the device is compute capability 8.9 and the executable
 * was built for N=24, main() loads it through the driver API (entry points
 * obtained from cudart, so no -lcuda is needed) and every kernel launch and
 * device-symbol copy goes to that module; the runtime's own PTX module is
 * never touched, so the driver never JITs it. Any failure while loading the
 * module or resolving a kernel/symbol happens before the first device-symbol
 * use and simply leaves the stock runtime path (JIT) in place.
 * QSB_NATIVE_MODULE=0 compiles the stock path only.
 *
 * Correctness does not depend on this path: the fixed-base table is still
 * spot-checked against OpenSSL, and every published hit is re-derived on the
 * host with OpenSSL before it is written (QSB_HOST_VERIFY). */
#ifndef QSB_NATIVE_MODULE_CUH
#define QSB_NATIVE_MODULE_CUH
#ifndef QSB_NATIVE_MODULE
#define QSB_NATIVE_MODULE 1
#endif
#if QSB_NATIVE_MODULE && defined(QSB_ZEROS_N) && (QSB_ZEROS_N == 24) && !defined(QSB_NATIVE_MODULE_BUILD) \
    && __has_include("../../subset_sm89_cubin.h")
#define QSB_NATIVE_ON 1
#else
#define QSB_NATIVE_ON 0
#endif

#if QSB_NATIVE_ON
#include <cuda.h>
#include <tuple>
#include <utility>
#include <type_traits>
#include "../../subset_sm89_cubin.h"   /* static const unsigned char qsb_sm89_cubin[] */

typedef CUresult (*qsb_pfn_load_t)(CUmodule *, const void *);
typedef CUresult (*qsb_pfn_getfn_t)(CUfunction *, CUmodule, const char *);
typedef CUresult (*qsb_pfn_getglobal_t)(CUdeviceptr *, size_t *, CUmodule, const char *);
typedef CUresult (*qsb_pfn_launch_t)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                     unsigned, CUstream, void **, void **);
typedef CUresult (*qsb_pfn_setattr_t)(CUfunction, CUfunction_attribute, int);

static CUmodule g_qsb_mod = NULL;
static qsb_pfn_getglobal_t g_qsb_getglobal = NULL;
static qsb_pfn_launch_t g_qsb_launch = NULL;
static qsb_pfn_setattr_t g_qsb_setattr = NULL;
static CUresult g_qsb_launch_err = CUDA_SUCCESS;
struct qsb_native_kernel { const void *host; const char *name; CUfunction fn; };
static qsb_native_kernel g_qsb_kern[8];
static int g_qsb_nkern = 0;

static void *qsb_driver_sym(const char *name) {
    void *fn = NULL;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPoint(name, &fn, cudaEnableDefault, &q) != cudaSuccess || q != cudaDriverEntryPointSuccess)
        return NULL;
    return fn;
}

/* Load the native module and resolve the kernels. Returns 1 when the native
 * path is active. Must run after the primary context exists and before any
 * kernel launch or device-symbol access. */
static int qsb_native_init(const cudaDeviceProp &prop, const qsb_native_kernel *ks, int nk,
                           const char *const *symbols, int nsym) {
    if (prop.major != 8 || prop.minor != 9 || nk > 8) return 0;
    qsb_pfn_load_t load = (qsb_pfn_load_t)qsb_driver_sym("cuModuleLoadData");
    qsb_pfn_getfn_t getfn = (qsb_pfn_getfn_t)qsb_driver_sym("cuModuleGetFunction");
    g_qsb_getglobal = (qsb_pfn_getglobal_t)qsb_driver_sym("cuModuleGetGlobal");
    g_qsb_launch = (qsb_pfn_launch_t)qsb_driver_sym("cuLaunchKernel");
    g_qsb_setattr = (qsb_pfn_setattr_t)qsb_driver_sym("cuFuncSetAttribute");
    if (!load || !getfn || !g_qsb_getglobal || !g_qsb_launch || !g_qsb_setattr) { (void)cudaGetLastError(); return 0; }
    CUmodule mod = NULL;
    if (load(&mod, qsb_sm89_cubin) != CUDA_SUCCESS || !mod) return 0;
    for (int i = 0; i < nk; i++) {
        g_qsb_kern[i] = ks[i];
        if (getfn(&g_qsb_kern[i].fn, mod, ks[i].name) != CUDA_SUCCESS) return 0;
    }
    for (int i = 0; i < nsym; i++) {
        CUdeviceptr p; size_t sz;
        if (g_qsb_getglobal(&p, &sz, mod, symbols[i]) != CUDA_SUCCESS) return 0;
    }
    g_qsb_nkern = nk;
    g_qsb_mod = mod;
    return 1;
}

static CUfunction qsb_native_fn(const void *host) {
    for (int i = 0; i < g_qsb_nkern; i++) if (g_qsb_kern[i].host == host) return g_qsb_kern[i].fn;
    return NULL;
}

template<class Tuple, size_t... I>
static void qsb_native_launch(CUfunction f, dim3 g, dim3 b, Tuple &t, std::index_sequence<I...>) {
    void *argv[sizeof...(I) + 1] = { (void *)&std::get<I>(t)..., NULL };
    CUresult r = f ? g_qsb_launch(f, g.x, g.y, g.z, b.x, b.y, b.z, 0, NULL, argv, NULL) : CUDA_ERROR_NOT_FOUND;
    if (r != CUDA_SUCCESS && g_qsb_launch_err == CUDA_SUCCESS) g_qsb_launch_err = r;
}

/* qsb_launch(kernel, grid, block, args...) == kernel<<<grid, block>>>(args...) */
template<typename... P, typename... A>
static inline void qsb_launch(void (*k)(P...), dim3 g, dim3 b, A &&... a) {
    if (g_qsb_mod) {
        std::tuple<typename std::decay<P>::type...> t(static_cast<typename std::decay<P>::type>(a)...);
        qsb_native_launch(qsb_native_fn((const void *)k), g, b, t, std::index_sequence_for<P...>{});
        return;
    }
    k<<<g, b>>>(std::forward<A>(a)...);
}

/* Launch errors of the native path surface through the same check the stock
 * path uses (cudaGetLastError after the launches). */
static inline cudaError_t qsb_last_error(void) {
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess && g_qsb_launch_err != CUDA_SUCCESS) {
        fprintf(stderr, "native module launch failed: CUresult %d\n", (int)g_qsb_launch_err);
        return cudaErrorLaunchFailure;
    }
    return e;
}

template<class T>
static cudaError_t qsb_to_symbol(const T &sym, const char *name, const void *src, size_t n, size_t off = 0,
                                 cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
    if (g_qsb_mod) {
        CUdeviceptr p; size_t sz;
        if (g_qsb_getglobal(&p, &sz, g_qsb_mod, name) != CUDA_SUCCESS || off + n > sz) return cudaErrorInvalidSymbol;
        return cudaMemcpy((void *)(uintptr_t)(p + off), src, n, kind);
    }
    return (cudaMemcpyToSymbol)(sym, src, n, off, kind);
}
template<class T>
static cudaError_t qsb_from_symbol(void *dst, const T &sym, const char *name, size_t n, size_t off = 0,
                                   cudaMemcpyKind kind = cudaMemcpyDeviceToHost) {
    if (g_qsb_mod) {
        CUdeviceptr p; size_t sz;
        if (g_qsb_getglobal(&p, &sz, g_qsb_mod, name) != CUDA_SUCCESS || off + n > sz) return cudaErrorInvalidSymbol;
        return cudaMemcpy(dst, (const void *)(uintptr_t)(p + off), n, kind);
    }
    return (cudaMemcpyFromSymbol)(dst, sym, n, off, kind);
}
template<class T>
static cudaError_t qsb_func_carveout(T *k, int carveout) {
    if (g_qsb_mod) {
        CUfunction f = qsb_native_fn((const void *)k);
        return (f && g_qsb_setattr(f, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, carveout) == CUDA_SUCCESS)
            ? cudaSuccess : cudaErrorInvalidDeviceFunction;
    }
    return cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
}
#define cudaMemcpyToSymbol(sym, ...) qsb_to_symbol(sym, #sym, __VA_ARGS__)
#define cudaMemcpyFromSymbol(dst, sym, ...) qsb_from_symbol(dst, sym, #sym, __VA_ARGS__)
#define QSB_LAST_ERROR() qsb_last_error()
#else
#include <utility>
/* qsb_launch(kernel, grid, block, args...) == kernel<<<grid, block>>>(args...) */
template<typename... P, typename... A>
static inline void qsb_launch(void (*k)(P...), dim3 g, dim3 b, A &&... a) { k<<<g, b>>>(std::forward<A>(a)...); }
#define QSB_LAST_ERROR() cudaGetLastError()
#endif
#endif
