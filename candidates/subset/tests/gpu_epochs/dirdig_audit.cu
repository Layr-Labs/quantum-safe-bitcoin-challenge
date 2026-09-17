/* Audit: direct regular-digit extraction must reproduce the
 * peel recurrence's signed odd digits -- table index and negation -- for every
 * chunk of the active geometry. Build:
 *   nvcc -O3 -DQSB_ZEROS_N=24 -o dirdig_audit dirdig_audit.cu -lcrypto -lm */
#define main qsb_grinder_main
#include "tree.cu"
#undef main
#include <vector>

__global__ void audit_digits(const uint64_t *ks, uint32_t *bad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t k[4]; for (int j = 0; j < 4; j++) k[j] = ks[i*4+j];
    /* reference: the peel recurrence */
    int32_t e[GT_CHUNKS];
    gt_recode_signed(k, e);
    /* candidate: direct field extraction from the setup value */
    uint64_t M[4]; int sign; gt_recode_setup(k, M, &sign);
    uint64_t sflag = (uint64_t)(sign < 0);
    unsigned pos = 1u;
    for (int c = 0; c < GT_CHUNKS; c++) {
        uint32_t idx_r, idx_d; uint64_t neg_r, neg_d;
        gt_digit_idx(e[c], &idx_r, &neg_r);
        gt_direct_digit(M, sflag, pos, gt_width(c), c == GT_CHUNKS-1, &idx_d, &neg_d);
        pos += gt_width(c);
        if (idx_r != idx_d || neg_r != neg_d || idx_r >= gt_entries(c)) atomicAdd(bad, 1u);
    }
}

int main() {
    const int n = 1 << 20;
    std::vector<uint64_t> ks(n * 4);
    /* deterministic pseudo-random scalars plus boundary values */
    uint64_t x = 0x243F6A8885A308D3ULL;
    for (int i = 0; i < n; i++) for (int j = 0; j < 4; j++) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17; ks[i*4+j] = x;
    }
    const uint64_t nlim[4] = {0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
    for (int j = 0; j < 4; j++) { ks[j] = 0; ks[4+j] = (j==0); ks[8+j] = ~0ULL; ks[12+j] = nlim[j]; ks[16+j] = nlim[j]; }
    ks[16] -= 1; ks[20] = 2; for (int j = 1; j < 4; j++) ks[20+j] = 0;
    uint64_t *dk; uint32_t *dbad, bad = 0;
    cudaMalloc(&dk, ks.size()*8); cudaMalloc(&dbad, 4);
    cudaMemcpy(dk, ks.data(), ks.size()*8, cudaMemcpyHostToDevice);
    cudaMemcpy(dbad, &bad, 4, cudaMemcpyHostToDevice);
    audit_digits<<<(n+255)/256, 256>>>(dk, dbad, n);
    cudaDeviceSynchronize();
    cudaMemcpy(&bad, dbad, 4, cudaMemcpyDeviceToHost);
    printf("direct digits: %d scalars x %d chunks, mismatches=%u (%s)\n",
           n, GT_CHUNKS, bad, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
