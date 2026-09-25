#pragma once
// Host-only module routing. Device definitions and arithmetic stay in tree.cu.
#include <cuda_runtime.h>
#include <openssl/sha.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#ifndef QSB_NATIVE_ADAPT
#define QSB_NATIVE_ADAPT 1
#endif
#if QSB_NATIVE_ADAPT != 0 && QSB_NATIVE_ADAPT != 1
#error QSB_NATIVE_ADAPT must be 0 or 1
#endif
#include "native_adapt.cuh"
#ifndef QSB_NATIVE_MODULE
#define QSB_NATIVE_MODULE 0
#endif
#if QSB_NATIVE_MODULE && CUDART_VERSION >= 12080 && (!defined(QSB_ZEROS_N) || QSB_ZEROS_N == 24)
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
#include "native_contract.cuh"
#include "native_manifest.cuh"
#include "native_image.cuh"
static cudaLibrary_t library;
static bool frozen = false, active = false;
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
static void select(const cudaDeviceProp &prop, const void *source_desc, size_t descriptor_bytes,
                   const uint32_t *source_banks, size_t bank_bytes) {
    if (frozen) fail("module selection repeated");
    if (prop.major != 8 || prop.minor != 9) {
        frozen = true;
        fprintf(stderr, "QSB module: source (device is not sm89)\n");
        return;
    }
    if (descriptor_bytes != sizeof(native_contract_desc) ||
        bank_bytes != sizeof(native_contract_banks) ||
        memcmp(source_desc, native_contract_desc, descriptor_bytes) ||
        memcmp(source_banks, native_contract_banks, bank_bytes))
        fail("source geometry differs from native contract");
    unsigned char *data = (unsigned char *)malloc(QSB_NATIVE_BYTES);
    if (!data) fail("payload allocation failed");
    size_t decoded = 0; unsigned acc = 0; int bits = 0;
    for (const char *line : native_b64) {
        for (const unsigned char *p = (const unsigned char *)line; *p; ++p) {
            const unsigned char c = *p;
            if (c == '=') continue;
            const int v = c >= 'A' && c <= 'Z' ? c-'A' :
                          c >= 'a' && c <= 'z' ? c-'a'+26 :
                          c >= '0' && c <= '9' ? c-'0'+52 : c=='+' ? 62 : c=='/' ? 63 : -1;
            if (v < 0) fail("invalid embedded payload encoding");
            acc = (acc << 6) | (unsigned)v; bits += 6;
            if (bits >= 8) {
                bits -= 8;
                if (decoded >= QSB_NATIVE_BYTES) fail("payload length mismatch");
                data[decoded++] = (unsigned char)(acc >> bits);
            }
        }
    }
    if (decoded != QSB_NATIVE_BYTES) fail("payload length mismatch");
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
    // Entry ABIs alone cannot reject an old twelve-term module.
    void *descriptor = NULL; size_t descriptor_size = 0;
    check(cudaLibraryGetGlobal(&descriptor, &descriptor_size, library, "QSB_S3_DESC"),
          "native descriptor lookup");
    if (!descriptor || descriptor_size != descriptor_bytes)
        fail("native descriptor size mismatch");
    unsigned char actual[sizeof(native_contract_desc)];
    check(cudaMemcpy(actual, descriptor, sizeof(actual), cudaMemcpyDeviceToHost),
          "native descriptor readback");
    if (memcmp(actual, source_desc, sizeof(actual)))
        fail("native descriptor contents mismatch");
    active = true;
    frozen = true;
    fprintf(stderr, "QSB module: native sm89 N=24; frozen device kernels and host globals checked\n");
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
template<class T> static cudaError_t to_symbol(const char *name, const T &symbol,
        const void *source, size_t count) {
    require_frozen();
    cudaError_t err;
    if (active) {
#if QSB_NATIVE_ADAPT
        check(cudaMemcpyToSymbol(symbol, source, count), name);
#endif
        err = cudaMemcpy(global(name, sizeof(T), count).pointer, source, count, cudaMemcpyHostToDevice);
    }
    else err = cudaMemcpyToSymbol(symbol, source, count);
    check(err, name);
    return err;
}
template<class T> static cudaError_t from_symbol(const char *name, void *dest,
        const T &symbol, size_t count) {
    require_frozen();
    cudaError_t err;
    if (active) err = cudaMemcpy(dest, global(name, sizeof(T), count).pointer, count, cudaMemcpyDeviceToHost);
    else err = cudaMemcpyFromSymbol(dest, symbol, count);
    check(err, name);
    return err;
}
// Apply launch-resource preferences to the selected module, never the dormant source image.
template<class... P> static cudaError_t attribute(const char *name, void (*source)(P...),
        cudaFuncAttribute attr, int value, int device) {
    require_frozen();
    if (!active) return cudaFuncSetAttribute((const void *)source, attr, value);
#if QSB_NATIVE_ADAPT
    check(cudaFuncSetAttribute((const void *)source, attr, value), name);
    cudaFuncAttributes source_attributes;
    check(cudaFuncGetAttributes(&source_attributes, (const void *)source), "source digest materialization");
#endif
    for (auto &k : kernels) if (!strcmp(k.name, name)) {
        cudaError_t err = cudaKernelSetAttributeForDevice(k.handle, attr, value, device);
        check(err, name);
        return err;
    }
    fail("attribute kernel absent from native manifest");
    return cudaErrorInvalidValue;
}
template<class T> struct identity { typedef T type; };
// Only the declared function pointer deduces P. NULL/0 and integral expressions
// convert to the kernel's declared types BEFORE their addresses are packed.
template<class... P> static void launch_on(bool native, const char *name, void (*source)(P...),
        dim3 grid, dim3 block, size_t shared, cudaStream_t stream, typename identity<P>::type... params) {
    require_frozen();
    if (native && !active) fail("native launch without a loaded module");
    const void *function = (const void *)source;
    if (native) {
        Kernel *found = NULL;
        for (auto &k : kernels) if (!strcmp(k.name, name)) { found = &k; break; }
        if (!found) fail("kernel absent from native manifest");
        if (!found->abi_checked) {
            const size_t sizes[] = {sizeof(P)...};
            if (found->count != sizeof...(P)) fail("kernel parameter count mismatch");
            for (size_t i = 0; i < sizeof...(P); ++i)
                if (sizes[i] != found->sizes[i]) fail("kernel parameter width mismatch");
            found->abi_checked = true;
        }
        function = (const void *)found->handle;
    }
    void *args[] = {(void *)&params...};
    check(cudaLaunchKernel(function, grid, block, args, shared, stream), name);
}
template<class... P> static void launch(const char *name, void (*source)(P...),
        dim3 grid, dim3 block, size_t shared, cudaStream_t stream, typename identity<P>::type... params) {
    launch_on(active,name,source,grid,block,shared,stream,params...);
}
#else
static void environment() {}
static void select(const cudaDeviceProp &) {
    fprintf(stderr, "QSB module: source (native disabled for this build)\n");
}
#endif
} // namespace qsb_native

#if QSB_NATIVE_ENABLED
#define QSB_TO_SYMBOL(symbol, source, bytes) qsb_native::to_symbol(#symbol, symbol, source, bytes)
#define QSB_FROM_SYMBOL(dest, symbol, bytes) qsb_native::from_symbol(#symbol, dest, symbol, bytes)
#define QSB_DIGEST_LAUNCH(native, kernel, grid, block, shared, stream, ...) qsb_native::launch_on(native, #kernel, kernel, grid, block, shared, stream, __VA_ARGS__)
#define QSB_LAUNCH(kernel, grid, block, shared, stream, ...) qsb_native::launch(#kernel, kernel, grid, block, shared, stream, __VA_ARGS__)
#define QSB_ATTRIBUTE(kernel, attr, value, device) qsb_native::attribute(#kernel, kernel, attr, value, device)
#else
#define QSB_TO_SYMBOL(symbol, source, bytes) cudaMemcpyToSymbol(symbol, source, bytes)
#define QSB_FROM_SYMBOL(dest, symbol, bytes) cudaMemcpyFromSymbol(dest, symbol, bytes)
#define QSB_DIGEST_LAUNCH(native, kernel, grid, block, shared, stream, ...) kernel<<<grid, block, shared, stream>>>(__VA_ARGS__)
#define QSB_LAUNCH(kernel, grid, block, shared, stream, ...) kernel<<<grid, block, shared, stream>>>(__VA_ARGS__)
#define QSB_ATTRIBUTE(kernel, attr, value, device) cudaFuncSetAttribute(kernel, attr, value)
#endif
