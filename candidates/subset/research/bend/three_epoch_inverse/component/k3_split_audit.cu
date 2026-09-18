#define main qsb_grinder_main
#include "tree.cu"
#undef main

__device__ __forceinline__ void k3_load(uint64_t x[5], const uint64_t *src, int lane) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) x[k] = src[(size_t)lane * 4 + k];
    x[4] = 0;
}

__device__ __forceinline__ void k3_store(uint64_t *dst, int lane, const uint64_t x[5]) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) dst[(size_t)lane * 4 + k] = x[k];
}

/* Component screen only: invabc is supplied by an independent caller. The
 * kernel exercises the exact six-multiply K3 product/split schedule using the
 * promoted qsb_field_mul_raw primitive. Zero substitution and output masking
 * are intentionally outside this arithmetic-only codegen screen. */
__global__ void k3_split_component(const uint64_t *as, const uint64_t *bs,
                                   const uint64_t *cs, const uint64_t *invabcs,
                                   uint64_t *leaves, uint64_t *ias,
                                   uint64_t *ibs, uint64_t *ics) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t a[5], b[5], c[5], invabc[5];
    uint64_t ab[5], abc[5], invab[5], ia[5], ib[5], ic[5];
    k3_load(a, as, lane); k3_load(b, bs, lane); k3_load(c, cs, lane);
    k3_load(invabc, invabcs, lane);
    qsb_field_mul_raw(ab, a, b);          /* 1: ab */
    qsb_field_mul_raw(abc, ab, c);        /* 2: leaf product (kept observable) */
    qsb_field_mul_raw(invab, invabc, c);  /* 3: 1/ab */
    qsb_field_mul_raw(ia, invab, b);      /* 4: 1/a */
    qsb_field_mul_raw(ib, invab, a);      /* 5: 1/b */
    qsb_field_mul_raw(ic, invabc, ab);    /* 6: 1/c */
    k3_store(leaves, lane, abc);
    k3_store(ias, lane, ia); k3_store(ibs, lane, ib); k3_store(ics, lane, ic);
}

int main() { return 0; }
