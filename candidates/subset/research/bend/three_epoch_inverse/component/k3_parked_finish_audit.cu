#define main qsb_grinder_main
#include "tree.cu"
#undef main

__device__ __forceinline__ void k3f_load(uint64_t x[5], const uint64_t *src, int lane) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) x[k] = src[(size_t)lane * 4 + k];
    x[4] = 0;
}

__device__ __forceinline__ void k3f_store(uint64_t *dst, int lane, const uint64_t x[5]) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) dst[(size_t)lane * 4 + k] = x[k];
}

/* Post-inverse component screen for an earlier K3 epoch. p1 and p2 are
 * (yR*ZZZ-Y)*ZZ and (yR*ZZZ+Y)*ZZ, parked before W=ZZZ*d is inverted.
 * The supplied inv is 1/W. This isolates the exact six field multiplies used
 * after reload; preparation, tree inversion, zero masking and hit hashing are
 * deliberately outside this static code-generation screen. */
__global__ void k3_parked_finish_component(
    const uint64_t *p1s, const uint64_t *p2s, const uint64_t *invs,
    const uint64_t *xrs, const uint64_t *yrs,
    uint64_t *x1s, uint64_t *y1s, uint64_t *x2s, uint64_t *y2s
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t p1[5], p2[5], inv[5], xr[5], yr[5];
    uint64_t m1[5], m2[5], s[5], t[5], x1[5], y1[5], x2[5], y2[5];
    uint64_t cc[5] = {QSB_U2R_C[0], QSB_U2R_C[1], QSB_U2R_C[2], QSB_U2R_C[3], 0};
    k3f_load(p1, p1s, lane); k3f_load(p2, p2s, lane); k3f_load(inv, invs, lane);
    k3f_load(xr, xrs, lane); k3f_load(yr, yrs, lane);
    qsb_field_mul_raw(m1, p1, inv);        /* 1: lambda1 */
    qsb_field_mul_raw(m2, p2, inv);        /* 2: -lambda2 */
    _ModAdd256(s, m1, m2);
    _ModSub256(t, m1, cc);
    qsb_field_mul_raw(x1, s, t);           /* 3: x1-xR */
    _ModAdd256(x1, x1, xr);
    _ModSub256(t, xr, x1);
    qsb_field_mul_raw(y1, m1, t);          /* 4: y1+yR */
    _ModSub256(y1, yr);
    _ModSub256(t, m2, cc);
    qsb_field_mul_raw(x2, s, t);           /* 5: x2-xR */
    _ModAdd256(x2, x2, xr);
    _ModSub256(t, xr, x2);
    qsb_field_mul_raw(y2, m2, t);          /* 6: -(y2)+yR */
    _ModSub256(y2, yr);
    _ModNeg256(y2);
    k3f_store(x1s, lane, x1); k3f_store(y1s, lane, y1);
    k3f_store(x2s, lane, x2); k3f_store(y2s, lane, y2);
}

int main() { return 0; }
