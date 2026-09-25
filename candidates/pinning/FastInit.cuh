// SPDX-License-Identifier: GPL-3.0-only
// Fast table initialisation for the pinning candidate (QSB_FAST_INIT).
//
// The promoted tree builds the fixed-base table in three serial host/device
// steps: six OpenSSL ladders of up to 16384 points each on the host, one
// mixed addition per table entry on the GPU, and one _ModInv per entry.
// This file replaces the first and last step, keeping every table byte
// identical:
//   1. the host supplies, per chunk, the 2R affine points 2^j * base_c
//      (j = 0..2R-1, R = QSB_GT_RADIX_BITS, 14 for the ranked geometry) plus the chunk-0 biased start K*A, all iso-scaled like the
//      ladder points (gt_point_to_limbs);
//   2. kernel_build_ladders_bits derives every L[lo] = lo * base_c (chunk 0:
//      K*A + lo*A) and H[hi] = hi * 2^14 * base_c from the set bits of its
//      index, at most 14 projective mixed additions per slot, then converts
//      to affine with one block-wide batched inversion;
//   3. kernel_build_gtable_batched is the promoted kernel_build_gtable with
//      the per-entry _ModInv replaced by the same block-wide batched inversion
//      (qsb_block_inverse<false>, 256 leaves, one _ModInv per block).
// Affine coordinates are unique and every multiply here is canonical for
// canonical inputs, so the resulting bytes equal the promoted builder's.
// A -DQSB_TABLE_CKSUM=1 build checksums the finished table and exits, which
// is how the two builders were compared bit for bit.
#pragma once

static_assert(QSB_BIGTBL, "fast init targets the GLV12 dense-table geometry");
#define QSB_LADDER_LBITS QSB_GT_RADIX_BITS        /* L uses bits 0..R-1 of lo, H bits R..2R-1 of hi */
#define QSB_LADDER_BITS (2 * QSB_GT_RADIX_BITS)   /* 2^0 .. 2^(2R-1) times base_c per chunk */
static_assert(GT_LO == (1u << QSB_LADDER_LBITS), "ladder radix must be 2^QSB_GT_RADIX_BITS");
static_assert(GT_HI == GT_LO, "H ladders are sized like L ladders");

__device__ __forceinline__ void qsb_load_affine(const uint64_t *__restrict__ src,
                                                uint64_t x[4], uint64_t y[4]) {
    #pragma unroll
    for (int limb = 0; limb < 4; limb++) { x[limb] = src[limb]; y[limb] = src[4 + limb]; }
}

/* One thread per ladder slot. Slot s encodes (chunk, kind, index) as
 * s = chunk * 2 * GT_LO + kind * GT_LO + index, kind 0 = L, kind 1 = H.
 * Unused slots (L[0] of chunks >= 1, every H[0]) stay zero like the promoted
 * host buffers. Every lane joins the block-wide inversion (identity when it
 * has no point). */
__global__ void __launch_bounds__(256) kernel_build_ladders_bits(
    const uint64_t *__restrict__ d_bits,   /* [GT_CHUNKS][QSB_LADDER_BITS][8] */
    const uint64_t *__restrict__ d_first,  /* [8]: chunk-0 start K*A */
    uint64_t *__restrict__ d_L,            /* [GT_CHUNKS][GT_LO][8] */
    uint64_t *__restrict__ d_H)            /* [GT_CHUNKS][GT_HI][8] */
{
    const unsigned slot = blockIdx.x * 256u + threadIdx.x;
    const unsigned slot_count = (unsigned)GT_CHUNKS * 2u * GT_LO;
    const bool active = slot < slot_count;
    unsigned chunk = 0, kind = 0, index = 0;
    if (active) {
        chunk = slot / (2u * GT_LO);
        kind = (slot / GT_LO) & 1u;
        index = slot % GT_LO;
    }
    uint64_t px[4], py[4], pz[5] = {1, 0, 0, 0, 0};
    bool have_point = false;
    if (active) {
        const uint64_t *bits = d_bits +
            ((size_t)chunk * QSB_LADDER_BITS + (kind ? QSB_LADDER_LBITS : 0u)) * 8u;
        if (kind == 0u && chunk == 0u) { qsb_load_affine(d_first, px, py); have_point = true; }
        #pragma unroll 1
        for (int bit = 0; bit < QSB_LADDER_LBITS; bit++) {
            if (((index >> bit) & 1u) == 0u) continue;
            const uint64_t *bp = bits + (size_t)bit * 8u;
            if (!have_point) { qsb_load_affine(bp, px, py); have_point = true; continue; }
            uint64_t qx[4], qy[4];
            qsb_load_affine(bp, qx, qy);
            _PointAddSecp256k1(px, py, pz, qx, qy);
        }
    }
    uint64_t inv[5] = {1, 0, 0, 0, 0};
    if (have_point) { Load256(inv, pz); }
    qsb_block_inverse<false>(inv);
    if (!active) return;
    uint64_t *dst = kind ? d_H + ((size_t)chunk * GT_HI + index) * 8u
                         : d_L + ((size_t)chunk * GT_LO + index) * 8u;
    if (have_point) {
        _ModMult(px, inv);
        _ModMult(py, inv);
        #pragma unroll
        for (int limb = 0; limb < 4; limb++) { dst[limb] = px[limb]; dst[4 + limb] = py[limb]; }
    } else {
        #pragma unroll
        for (int limb = 0; limb < 8; limb++) dst[limb] = 0ULL;
    }
}

/* kernel_build_gtable with the per-entry _ModInv replaced by one block-wide
 * batched inversion. Same index algebra, same loads, same multiplies. */
__global__ void __launch_bounds__(256) kernel_build_gtable_batched(
    const uint64_t *__restrict__ d_L,
    const uint64_t *__restrict__ d_H,
    uint8_t *__restrict__ gTable)
{
    const uint64_t entry = (uint64_t)blockIdx.x * 256u + threadIdx.x;
    int ch = -1;
    if (entry < GT_TOTAL_ENTRIES) {
        #pragma unroll
        for (int chunk = 0; chunk < GT_CHUNKS; chunk++)
            if (entry >= gt_offset(chunk) && entry < (uint64_t)gt_offset(chunk) + gt_entries(chunk)) ch = chunk;
    }
    uint64_t rx[4] = {0, 0, 0, 0}, ry[4] = {0, 0, 0, 0}, pz[5] = {1, 0, 0, 0, 0};
    bool need_inverse = false;
    int record = 0;
    if (ch >= 0) {
        record = (int)(entry - gt_offset(ch));
        const int multiple = ch == 0 ? record : 2 * record + 1;
        const int hi = multiple >> QSB_GT_RADIX_BITS, lo = multiple & (GT_LO - 1);
        const uint64_t *Hp = d_H + ((size_t)ch * GT_HI + hi) * 8;
        const uint64_t *Lp = d_L + ((size_t)ch * GT_LO + lo) * 8;
        if (hi == 0) {
            qsb_load_affine(Lp, rx, ry);
        } else {
            uint64_t qx[4], qy[4];
            qsb_load_affine(Hp, rx, ry);
            qsb_load_affine(Lp, qx, qy);
            _PointAddSecp256k1(rx, ry, pz, qx, qy);
            need_inverse = true;
        }
    }
    uint64_t inv[5] = {1, 0, 0, 0, 0};
    if (need_inverse) { Load256(inv, pz); }
    qsb_block_inverse<false>(inv);
    if (ch < 0) return;
    if (need_inverse) {
        _ModMult(rx, inv);
        _ModMult(ry, inv);
    }
    {   uint64_t a0=ry[0],a1=ry[1],a2=ry[2],a3=ry[3];
        asm("add.cc.u64 %0, %0, 0x800001E8; addc.cc.u64 %1, %1, 0; addc.cc.u64 %2, %2, 0; addc.u64 %3, %3, 0;"
            : "+l"(a0), "+l"(a1), "+l"(a2), "+l"(a3));
        ry[0]=a0; ry[1]=a1; ry[2]=a2; ry[3]=a3; }
    size_t off = ((size_t)gt_offset(ch) + record) * 64;
    memcpy(gTable + off, rx, 32);
    memcpy(gTable + off + 32, ry, 32);
}

#if QSB_TABLE_CKSUM
/* Order-independent 64-bit digest of the whole table (sum and xor of a
 * per-entry mix). Diagnostic build only. */
__global__ void kernel_table_checksum(const uint8_t *__restrict__ gTable,
                                      unsigned long long *__restrict__ acc)
{
    const uint64_t entry = (uint64_t)blockIdx.x * 256u + threadIdx.x;
    if (entry >= GT_TOTAL_ENTRIES) return;
    const uint64_t *words = (const uint64_t *)(gTable + entry * 64);
    uint64_t digest = entry * 0x9E3779B97F4A7C15ULL;
    #pragma unroll
    for (int limb = 0; limb < 8; limb++) {
        digest ^= words[limb];
        digest *= 0xBF58476D1CE4E5B9ULL;
        digest ^= digest >> 31;
    }
    atomicAdd(acc, (unsigned long long)digest);
    atomicXor(acc + 1, (unsigned long long)digest);
}
#endif

/* Host: the 2R doubling points per chunk plus the chunk-0 start, iso-scaled.
 * Chunk 0: base = A = neg_r_inv*G, start = bias*A. Chunk c >= 1:
 * base = 2^(shift_c - 1) * A. Bit j holds 2^j * base. Returns 1 on success. */
static int gt_build_bit_points(uint64_t *hBits, uint64_t *hFirst,
                               const uint8_t neg_r_inv[32],
                               const uint64_t alpha_le[4], const uint64_t beta_le[4]) {
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *x = BN_new(), *y = BN_new(), *factor = BN_new(), *order = BN_new(),
           *nri = BN_new(), *bscal = BN_new(), *field_p = BN_new(),
           *alpha = BN_new(), *beta = BN_new(), *bias = BN_new();
    EC_POINT *base = EC_POINT_new(grp), *pt = EC_POINT_new(grp), *first = EC_POINT_new(grp);
    int ok = grp && ctx && x && y && factor && order && nri && bscal && field_p &&
             alpha && beta && bias && base && pt && first;
    if (ok) ok = EC_GROUP_get_order(grp, order, ctx) &&
                 EC_GROUP_get_curve_GFp(grp, field_p, NULL, NULL, ctx) &&
                 BN_lebin2bn((const uint8_t *)alpha_le, 32, alpha) != NULL &&
                 BN_lebin2bn((const uint8_t *)beta_le, 32, beta) != NULL &&
                 BN_lebin2bn(neg_r_inv, 32, nri) != NULL;
#if QSB_BIGTBL
    if (ok) ok = BN_set_word(bias, QSB_GT_TOP_CENTER + 1u) && BN_lshift(bias, bias, QSB_GT_TOP_SHIFT - 1u) &&
                 BN_sub_word(bias, 1u << 17);
#else
    if (ok) ok = BN_set_word(bias, 333126) && BN_lshift(bias, bias, 108) && BN_sub_word(bias, 1u << 17);
#endif
    for (int ch = 0; ok && ch < GT_CHUNKS; ch++) {
        if (ch == 0) {
            ok = EC_POINT_mul(grp, base, nri, NULL, NULL, ctx) &&
                 BN_mod_mul(bscal, bias, nri, order, ctx) &&
                 EC_POINT_mul(grp, first, bscal, NULL, NULL, ctx);
            if (ok) gt_point_to_limbs(grp, first, x, y, alpha, beta, field_p, ctx, hFirst);
        } else {
            ok = BN_one(factor) && BN_lshift(factor, factor, gt_shift(ch) - 1) &&
                 BN_mod_mul(bscal, factor, nri, order, ctx) &&
                 EC_POINT_mul(grp, base, bscal, NULL, NULL, ctx);
        }
        if (ok) ok = EC_POINT_copy(pt, base) != 0;
        for (int bit = 0; ok && bit < QSB_LADDER_BITS; bit++) {
            gt_point_to_limbs(grp, pt, x, y, alpha, beta, field_p, ctx,
                              hBits + ((size_t)ch * QSB_LADDER_BITS + bit) * 8);
            if (bit + 1 < QSB_LADDER_BITS) ok = EC_POINT_dbl(grp, pt, pt, ctx);
        }
    }
    BN_free(x); BN_free(y); BN_free(factor); BN_free(order); BN_free(nri);
    BN_free(bscal); BN_free(field_p); BN_free(alpha); BN_free(beta); BN_free(bias);
    EC_POINT_free(base); EC_POINT_free(pt); EC_POINT_free(first);
    EC_GROUP_free(grp); BN_CTX_free(ctx);
    return ok;
}

/* Host driver: bit points -> device -> ladder kernel. dL/dH must already be
 * allocated (GT_CHUNKS*GT_LO*8 and GT_CHUNKS*GT_HI*8 limbs). Returns 1 on success. */
static int qsb_fast_ladders(uint64_t *dL, uint64_t *dH, const uint8_t neg_r_inv[32],
                            const uint64_t alpha_le[4], const uint64_t beta_le[4]) {
    const size_t bits_bytes = (size_t)GT_CHUNKS * QSB_LADDER_BITS * 8 * sizeof(uint64_t);
    uint64_t *hBits = (uint64_t *)calloc(1, bits_bytes), hFirst[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (!hBits) { fprintf(stderr, "[qsb_fast_ladders] OOM: bit points\n"); return 0; }
    if (!gt_build_bit_points(hBits, hFirst, neg_r_inv, alpha_le, beta_le)) {
        fprintf(stderr, "[qsb_fast_ladders] OpenSSL bit-point construction failed\n");
        free(hBits); return 0;
    }
    uint64_t *dBits = NULL, *dFirst = NULL;
    cudaError_t err = cudaMalloc(&dBits, bits_bytes);
    if (err == cudaSuccess) err = cudaMalloc(&dFirst, sizeof(hFirst));
    if (err == cudaSuccess) err = cudaMemcpy(dBits, hBits, bits_bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dFirst, hFirst, sizeof(hFirst), cudaMemcpyHostToDevice);
    free(hBits);
    if (err == cudaSuccess) {
        const unsigned slots = (unsigned)GT_CHUNKS * 2u * GT_LO;
        kernel_build_ladders_bits<<<(slots + 255u) / 256u, 256>>>(dBits, dFirst, dL, dH);
        err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaDeviceSynchronize();
    }
    cudaFree(dBits); cudaFree(dFirst);
    if (err != cudaSuccess) {
        fprintf(stderr, "[qsb_fast_ladders] ladder kernel failed: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}
