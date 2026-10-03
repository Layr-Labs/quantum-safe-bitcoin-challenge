#pragma once
// Secondary root library owner. Native execution and performance await official validation.
// The caller must authenticate the immutable donor image before passing these bytes,
// preserve primary b59 prepare/finish, select geometry BEFORE allocations, and check
// every return. Native library initialization and CUDA graph binding remain untested.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
namespace qsb_root_secondary_research {
struct State {
    cudaLibrary_t library = nullptr;
    cudaKernel_t register512 = nullptr;
    cudaKernel_t fused4 = nullptr;
    bool constants_ready = false;
    bool launch_failed = false;
};
static cudaError_t init(State &s, const void *image, size_t image_bytes,
                        const cudaDeviceProp &device) {
    // Initialization only on a fresh owner, before problem uploads or any launches.
    if (s.library || !image || image_bytes != 547040 ||
        device.major != 8 || device.minor != 9) return cudaErrorInvalidValue;
    cudaLibrary_t library = nullptr;
    cudaError_t e = cudaLibraryLoadData(&library, image, nullptr, nullptr, 0,
                                       nullptr, nullptr, 0);
    if (e != cudaSuccess) return e;
    cudaKernel_t rr = nullptr, rf = nullptr;
    e = cudaLibraryGetKernel(&rr, library, "_Z20qsb_root_register512Pmi");
    if (e == cudaSuccess)
        e = cudaLibraryGetKernel(&rf, library, "_Z14qsb_root_fusedILi4EEvPmi");
    void *fingerprint_address = nullptr; size_t bytes = 0; int fingerprint = 0;
    if (e == cudaSuccess)
        e = cudaLibraryGetGlobal(&fingerprint_address, &bytes, library,
                                 "qsb_carrier_zeros");
    if (e == cudaSuccess && bytes != sizeof(fingerprint)) e = cudaErrorInvalidValue;
    if (e == cudaSuccess)
        e = cudaMemcpy(&fingerprint, fingerprint_address, bytes, cudaMemcpyDeviceToHost);
    if (e == cudaSuccess && fingerprint != (24 | 0x20000000)) e = cudaErrorInvalidValue;
    if (e != cudaSuccess) {
        cudaError_t cleanup = cudaLibraryUnload(library);
        return cleanup == cudaSuccess ? e : cleanup;
    }
    s.library = library; s.register512 = rr; s.fused4 = rf;
    return cudaSuccess;
}
static cudaError_t write_constant(State &s, const char *name, const void *value, size_t n) {
    void *address = nullptr; size_t bytes = 0;
    cudaError_t e = cudaLibraryGetGlobal(&address, &bytes, s.library, name);
    if (e != cudaSuccess) return e;
    if (bytes != n) return cudaErrorInvalidValue;
    return cudaMemcpy(address, value, bytes, cudaMemcpyHostToDevice);
}
static cudaError_t set_problem(State &s, const uint64_t y[4], const uint64_t iso_y[4],
                               const uint64_t inverse_u[4]) {
    // Must be called between drained jobs, never while prior kernels use these constants.
    s.constants_ready = false;
    if (!s.library || !y || !iso_y || !inverse_u || s.launch_failed) return cudaErrorInvalidValue;
    const uint32_t zero = 0, one = 1;
    cudaError_t e = write_constant(s, "pin_u2ry_words", y, 32);
    if (e == cudaSuccess) e = write_constant(s, "pin_iso_u2ry_words", iso_y, 32);
    // qwr_inverse_scaled and the fused fallback both read inverse_u.
    if (e == cudaSuccess) e = write_constant(s, "pin_iso_invu_words", inverse_u, 32);
    if (e == cudaSuccess) e = write_constant(s, "pin_zero_add", &zero, sizeof(zero));
    if (e == cudaSuccess) e = write_constant(s, "pin_one_mul", &one, sizeof(one));
    if (e == cudaSuccess) s.constants_ready = true;
    return e; // partial upload cannot enable launch; caller must handle failure.
}
static cudaError_t launch(State &s, bool use_register, uint64_t *roots,
                          int count, cudaStream_t stream) {
    if (!s.library || !s.constants_ready || s.launch_failed || !roots ||
        count < 1 || count > 512) return cudaErrorInvalidValue;
    void *arguments[] = {&roots, &count};
    const cudaKernel_t kernel = use_register ? s.register512 : s.fused4;
    cudaError_t e = cudaLaunchKernel((const void *)kernel, dim3(1), dim3(128),
                                    arguments, 0, stream);
    if (e != cudaSuccess) s.launch_failed = true;
    return e; // caller must terminate failed work, not silently use stale outputs.
}
} // namespace qsb_root_secondary_research
