#pragma once
// Host-only module routing. Device definitions and arithmetic stay in tree.cu.
#include <cuda_runtime.h>
#include <openssl/sha.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include "native_role.cuh"
#include "native_types.cuh"
#if QSB_HOST_CARRIER
#if CUDART_VERSION < 12080 || (defined(QSB_ZEROS_N) && QSB_ZEROS_N != 24)
#error "Native carrier requires CUDA 12.8+ headers and N=24"
#endif
#include "native_abi.cuh"
#define QSB_NATIVE_ENABLED 1
#else
#define QSB_NATIVE_ENABLED 0
#endif

namespace qsb_native {
static void check(cudaError_t err, const char *operation) {
    if (err != cudaSuccess) {
        fprintf(stderr, "QSB CUDA %s: %s\n", operation, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#if QSB_NATIVE_ENABLED
struct Kernel {
    const char *name, *symbol;
    size_t count;
    size_t sizes[40];
    cudaKernel_t handle;
    bool abi_checked;
};
struct Global {
    const char *name;
    size_t expected;
    void *pointer;
};
#include "native_manifest.cuh"
static cudaLibrary_t library;
static bool frozen = false;
static void fail(const char *message) {
    fprintf(stderr, "QSB native: %s\n", message);
    exit(EXIT_FAILURE);
}
static void require_frozen() {
    if (!frozen) fail("module selection must precede symbols and launches");
}
static void environment() {
    // Override EAGER before the first CUDA call, including cudaSetDevice.
    if (setenv("CUDA_MODULE_LOADING", "LAZY", 1) ||
        setenv("CUDA_MODULE_DATA_LOADING", "LAZY", 1)) fail("cannot set lazy loading");
}
static void select(const cudaDeviceProp &prop) {
    if (frozen) fail("module selection repeated");
    if (prop.major != 8 || prop.minor != 9) fail("carrier requires sm89 device");
    char path[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (n < 0 || n >= (ssize_t)sizeof(path) - 1) fail("cannot locate executable");
    path[n] = 0;
    char *slash = strrchr(path, '/');
    if (!slash || (size_t)(slash + 1 - path) + sizeof(QSB_NATIVE_FILE) > sizeof(path))
        fail("native payload path too long");
    strcpy(slash + 1, QSB_NATIVE_FILE);
    FILE *f = fopen(path, "rb");
    if (!f) fail("native payload unavailable");
    unsigned char *data = (unsigned char *)malloc(QSB_NATIVE_BYTES);
    if (!data) fail("payload allocation failed");
    size_t got = fread(data, 1, QSB_NATIVE_BYTES, f);
    int extra = fgetc(f), read_error = ferror(f);
    fclose(f);
    if (got != QSB_NATIVE_BYTES || extra != EOF || read_error) fail("payload length mismatch");
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(data, QSB_NATIVE_BYTES, digest);
    if (memcmp(digest, payload_sha256, sizeof(digest))) fail("payload SHA256 mismatch");
    check(cudaLibraryLoadData(&library, data, NULL, NULL, 0, NULL, NULL, 0), "native library load");
    // Retain the immutable payload for the lifetime of the library.
    for (auto &k : kernels)
        check(cudaLibraryGetKernel(&k.handle, library, k.symbol), k.name);
    for (auto &g : globals) {
        size_t bytes = 0;
        check(cudaLibraryGetGlobal(&g.pointer, &bytes, library, g.name), g.name);
        if (!g.pointer || bytes != g.expected) fail("native global size mismatch");
    }
    frozen = true;
    fprintf(stderr, "QSB module: native sm89 N=24; seven kernels, thirteen host globals checked\n");
}
static Global &global(const char *name, size_t declared, size_t count) {
    for (auto &g : globals) {
        if (!strcmp(g.name, name)) {
            if (g.expected != declared || count > g.expected) fail("symbol copy bounds mismatch");
            return g;
        }
    }
    fail("symbol absent from native manifest");
    return globals[0];
}
template<class T> static cudaError_t to_symbol(SymbolTag<T> tag,
        const void *source, size_t count) {
    require_frozen();
    cudaError_t err = cudaMemcpy(global(tag.name, sizeof(T), count).pointer, source, count, cudaMemcpyHostToDevice);
    check(err, tag.name);
    return err;
}
template<class T> static cudaError_t from_symbol(SymbolTag<T> tag,
        void *dest, size_t count) {
    require_frozen();
    cudaError_t err = cudaMemcpy(dest, global(tag.name, sizeof(T), count).pointer, count, cudaMemcpyDeviceToHost);
    check(err, tag.name);
    return err;
}
template<class T> struct identity { typedef T type; };
// Only the typed tag deduces P. NULL/0 and integral expressions convert to
// the declared kernel types BEFORE their addresses are packed.
template<class... P> static void launch(KernelTag<P...> tag,
        dim3 grid, dim3 block, typename identity<P>::type... params) {
    require_frozen();
    Kernel *found = NULL;
    for (auto &k : kernels) if (!strcmp(k.name, tag.name)) { found = &k; break; }
    if (!found) fail("kernel absent from native manifest");
    if (!found->abi_checked) {
        const size_t sizes[] = {sizeof(P)...};
        if (found->count != sizeof...(P)) fail("kernel parameter count mismatch");
        for (size_t i = 0; i < sizeof...(P); ++i)
            if (sizes[i] != found->sizes[i]) fail("kernel parameter width mismatch");
        found->abi_checked = true;
    }
    void *args[] = {(void *)&params...};
    check(cudaLaunchKernel((const void *)found->handle, grid, block, args, 0, NULL), tag.name);
}
#else
static void environment() {}
static void select(const cudaDeviceProp &) {
    fprintf(stderr, "QSB module: source (native disabled for this build)\n");
}
#endif
} // namespace qsb_native

#if QSB_NATIVE_ENABLED
#define QSB_TO_SYMBOL(symbol, source, bytes) qsb_native::to_symbol(qsb_native::abi::symbol, source, bytes)
#define QSB_FROM_SYMBOL(dest, symbol, bytes) qsb_native::from_symbol(qsb_native::abi::symbol, dest, bytes)
#define QSB_LAUNCH(kernel, grid, block, ...) qsb_native::launch(qsb_native::abi::kernel, grid, block, __VA_ARGS__)
#else
#define QSB_TO_SYMBOL(symbol, source, bytes) cudaMemcpyToSymbol(symbol, source, bytes)
#define QSB_FROM_SYMBOL(dest, symbol, bytes) cudaMemcpyFromSymbol(dest, symbol, bytes)
#define QSB_LAUNCH(kernel, grid, block, ...) do { kernel<<<grid, block>>>(__VA_ARGS__); qsb_native::check(cudaGetLastError(), #kernel); } while (0)
#endif
