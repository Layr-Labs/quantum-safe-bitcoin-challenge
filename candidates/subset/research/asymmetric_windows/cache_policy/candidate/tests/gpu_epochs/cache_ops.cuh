// Research cache hints. Ordinary host branches support address/value audits only.
// PTX .cs allocates with evict-first; it does not remove memory traffic.
#pragma once
__device__ __forceinline__ uint64_t qsb_cs_load64(const uint64_t *p){
#if defined(__CUDA_ARCH__)
    uint64_t r;asm volatile("ld.global.cs.u64 %0, [%1];" : "=l"(r) : "l"(p) : "memory");return r;
#else
    return *p;
#endif
}
__device__ __forceinline__ void qsb_cs_store64(uint64_t *p,uint64_t v){
#if defined(__CUDA_ARCH__)
    asm volatile("st.global.cs.u64 [%0], %1;" :: "l"(p),"l"(v) : "memory");
#else
    *p=v;
#endif
}
__device__ __forceinline__ ulonglong2 qsb_cs_load128(const ulonglong2 *p){
#if defined(__CUDA_ARCH__)
    ulonglong2 r;asm volatile("ld.global.cs.v2.u64 {%0,%1}, [%2];" : "=l"(r.x),"=l"(r.y) : "l"(p) : "memory");return r;
#else
    return *p;
#endif
}
__device__ __forceinline__ void qsb_cs_store128(ulonglong2 *p,ulonglong2 v){
#if defined(__CUDA_ARCH__)
    asm volatile("st.global.cs.v2.u64 [%0], {%1,%2};" :: "l"(p),"l"(v.x),"l"(v.y) : "memory");
#else
    *p=v;
#endif
}
