// SPDX-License-Identifier: GPL-3.0-only
// The validated 18 MiB GLV builder, with separate names for coexistence.
#pragma once
enum { GLV18_GT_CHUNKS=8, GLV18_GT_TOTAL_ENTRIES=(9u<<15), GLV18_GT_LO=256, GLV18_GT_HI=512 };
__host__ __device__ __forceinline__ unsigned glv18_gt_entries(int c){return c==7?(1u<<16):(1u<<15);}
__host__ __device__ __forceinline__ unsigned glv18_gt_offset(int c){return (unsigned)c<<15;}
__host__ __device__ __forceinline__ int glv18_gt_shift(int c){return 16*c;}
__global__ void glv18_kernel_build_gtable(
    const uint64_t * __restrict__ d_L,   /* [GLV18_GT_CHUNKS][GLV18_GT_LO][8] : x[4] then y[4] */
    const uint64_t * __restrict__ d_H,   /* [GLV18_GT_CHUNKS][GLV18_GT_HI][8] */
    uint8_t * __restrict__ gTable)
{
    uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= GLV18_GT_TOTAL_ENTRIES) return;
    int ch=(int)(t>>15);if(ch>7)ch=7;
    int d=(int)(t-glv18_gt_offset(ch));
    int m  = 2*d + 1;                        /* odd multiple below 2^17 */
    int hi = m >> 8, lo = m & 255;           /* lo odd; hi < 512 */

    const uint64_t *Hp = d_H + ((size_t)ch * GLV18_GT_HI + hi) * 8;
    const uint64_t *Lp = d_L + ((size_t)ch * GLV18_GT_LO + lo) * 8;

    uint64_t rx[4], ry[4];
    if (hi == 0) {
        for (int k = 0; k < 4; k++) { rx[k] = Lp[k]; ry[k] = Lp[4 + k]; }
    } else {
        uint64_t px[4], py[4], pz[5] = {1, 0, 0, 0, 0}, qx[4], qy[4];
        for (int k = 0; k < 4; k++) {
            px[k] = Hp[k]; py[k] = Hp[4 + k];
            qx[k] = Lp[k]; qy[k] = Lp[4 + k];
        }
        _PointAddSecp256k1(px, py, pz, qx, qy);
        _ModInv(pz);
        _ModMult(px, pz); _ModMult(py, pz);
        for (int k = 0; k < 4; k++) { rx[k] = px[k]; ry[k] = py[k]; }
    }
    /* Limbs are little-endian in memory, which is exactly the table's byte
     * order, so the store is a straight copy. */
    size_t off = ((size_t)glv18_gt_offset(ch) + d) * 64;
    memcpy(gTable + off,      rx, 32);
    memcpy(gTable + off + 32, ry, 32);
}

/* Affine (x,y) of a point, as the 4+4 little-endian limbs the table uses. */
static void glv18_gt_point_to_limbs(EC_GROUP *grp, EC_POINT *pt, BIGNUM *x, BIGNUM *y,
                              BN_CTX *ctx, uint64_t out[8]) {
    uint8_t xb[32], yb[32];
    memset(xb, 0, 32); memset(yb, 0, 32);
    EC_POINT_get_affine_coordinates_GFp(grp, pt, x, y, ctx);
    BN_bn2bin(x, xb + (32 - BN_num_bytes(x)));
    BN_bn2bin(y, yb + (32 - BN_num_bytes(y)));
    for (int j = 0; j < 16; j++) { uint8_t t = xb[j]; xb[j] = xb[31-j]; xb[31-j] = t; }
    for (int j = 0; j < 16; j++) { uint8_t t = yb[j]; yb[j] = yb[31-j]; yb[31-j] = t; }
    memcpy(out,     xb, 32);
    memcpy(out + 4, yb, 32);
}

/* L[ch][lo] = lo * 2^(16ch) * (A/2), and
 * H[ch][hi] = hi * 256 * 2^(16ch) * (A/2). Index 0 is left zeroed and
 * the kernel handles hi=0 directly. The ladders contain 4,336 real points;
 * the full table has 294,912 entries. */
/* Build the ladders for base A/2 where A = neg_r_inv * G (problem-dependent).
 * With the table on base A, recoding z directly gives z*A = z*neg_r_inv*G =
 * (neg_r_inv*z mod n)*G = u1*G, so the kernel skips gpu_scalar_mulmod. neg_r_inv
 * comes from the runtime problem (little-endian 32 bytes), so the ladders are
 * rebuilt per instance and NOT cached across problems (anti-replay). */
static void glv18_gt_build_ladders(uint64_t *hL, uint64_t *hH, const uint8_t neg_r_inv[32]) {
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *x = BN_new(), *y = BN_new(), *shift = BN_new(), *inv2 = BN_new(),
           *order = BN_new(), *nri = BN_new(), *bscal = BN_new();
    EC_POINT *base = EC_POINT_new(grp), *step = EC_POINT_new(grp), *acc = EC_POINT_new(grp);
    /* base = A/2 = (2^-1 * neg_r_inv mod n) * G */
    EC_GROUP_get_order(grp, order, ctx);
    BN_set_word(shift, 2); BN_mod_inverse(inv2, shift, order, ctx);
    BN_lebin2bn(neg_r_inv, 32, nri);                     /* neg_r_inv is LE, like d_nri */
    BN_mod_mul(bscal, inv2, nri, order, ctx);            /* (2^-1 * neg_r_inv) mod n */
    EC_POINT_mul(grp, base, bscal, NULL, NULL, ctx);     /* base = bscal * G = A/2 */
    memset(hL, 0, (size_t)GLV18_GT_CHUNKS * GLV18_GT_LO * 8 * sizeof(uint64_t));
    memset(hH, 0, (size_t)GLV18_GT_CHUNKS * GLV18_GT_HI * 8 * sizeof(uint64_t));
    for (int ch = 0; ch < GLV18_GT_CHUNKS; ch++) {
        if (ch > 0) { BN_set_word(shift, 1ul << (glv18_gt_shift(ch) - glv18_gt_shift(ch-1))); EC_POINT_mul(grp, base, NULL, base, shift, ctx); }
        EC_POINT_copy(acc, base);
        for (int lo = 1; lo < GLV18_GT_LO; lo++) {                 /* L[lo] = lo * B */
            glv18_gt_point_to_limbs(grp, acc, x, y, ctx, hL + ((size_t)ch * GLV18_GT_LO + lo) * 8);
            EC_POINT_add(grp, acc, acc, base, ctx);
        }
        BN_set_word(shift, 256);                             /* step = 256 * B */
        EC_POINT_mul(grp, step, NULL, base, shift, ctx);
        EC_POINT_copy(acc, step);
        for (int hi = 1; hi < (int)(glv18_gt_entries(ch) >> 7); hi++) {   /* m=2d+1 < 2*entries */                 /* H[hi] = hi * 256 * B */
            glv18_gt_point_to_limbs(grp, acc, x, y, ctx, hH + ((size_t)ch * GLV18_GT_HI + hi) * 8);
            EC_POINT_add(grp, acc, acc, step, ctx);
        }
    }
    BN_free(x); BN_free(y); BN_free(shift); BN_free(inv2); BN_free(order);
    BN_free(nri); BN_free(bscal);
    EC_POINT_free(base); EC_POINT_free(step); EC_POINT_free(acc);
    EC_GROUP_free(grp); BN_CTX_free(ctx);
}

/* Spot-check the built table against OpenSSL. The builder runs on hardware this
 * code has never executed on, so a silent wrong table -- which would simply
 * produce zero verifiable hits and burn the whole run -- must be caught here
 * and fall back, not discovered from the scorecard. */
static int glv18_gt_spot_check(const uint8_t *gTable, int samples,
                         const uint8_t neg_r_inv[32]) {
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *x = BN_new(), *y = BN_new(), *k = BN_new(), *inv2 = BN_new(), *order = BN_new(),
           *nri = BN_new(), *half_nri = BN_new();
    EC_POINT *pt = EC_POINT_new(grp);
    uint64_t want[8];
    int ok = 1;
    unsigned seed = 0x9e3779b9u;
    EC_GROUP_get_order(grp, order, ctx);
    BN_set_word(k, 2); BN_mod_inverse(inv2, k, order, ctx);   /* inv2 = 2^-1 mod n */
    BN_lebin2bn(neg_r_inv, 32, nri);
    BN_mod_mul(half_nri, inv2, nri, order, ctx);              /* (2^-1 * neg_r_inv) mod n = A/2 scalar */
    for (int t = 0; t < samples && ok; t++) {
        /* always include the corners of each chunk, then pseudo-random entries */
        int ch, i;
        if (t < GLV18_GT_CHUNKS * 4) {
            ch = t / 4;
            const int corner[4] = {0, 1, 2, (int)glv18_gt_entries(ch) - 1};
            i = corner[t % 4];
        } else {
            seed = seed * 1664525u + 1013904223u;
            ch = (int)(seed >> 28) % GLV18_GT_CHUNKS;
            i  = (int)((seed >> 4) & (glv18_gt_entries(ch) - 1));
        }
        /* want = (2i+1) * 2^glv18_gt_shift(ch) * (A/2). */
        BN_one(k);
        BN_lshift(k, k, glv18_gt_shift(ch));
        BN_mul_word(k, (BN_ULONG)(2*i + 1));
        BN_mod_mul(k, k, half_nri, order, ctx);
        EC_POINT_mul(grp, pt, k, NULL, NULL, ctx);
        glv18_gt_point_to_limbs(grp, pt, x, y, ctx, want);
        size_t off = ((size_t)glv18_gt_offset(ch) + i) * 64;
        if (memcmp(gTable + off,      want,     32) != 0 ||
            memcmp(gTable + off + 32, want + 4, 32) != 0) {
            fprintf(stderr, "  GTable spot check FAILED at chunk %d entry %d\n", ch, i);
            ok = 0;
        }
    }
    BN_free(x); BN_free(y); BN_free(k); BN_free(inv2); BN_free(order); BN_free(nri); BN_free(half_nri);
    EC_POINT_free(pt); EC_GROUP_free(grp); BN_CTX_free(ctx);
    return ok;
}

/* OpenSSL fallback builder (only if the GPU builder's spot check fails). Emits
 * the signed table: entry (ch,d) = (2d+1) * 2^(16ch) * (G/2). Walks odd
 * multiples by stepping 2*base_c per entry (acc = base_c, 3base_c, ...). */
static void glv18_compute_gtable(uint8_t *gTable, const uint8_t neg_r_inv[32]) {
    /* No cache: base A/2 is problem-dependent (neg_r_inv fresh per instance). */
    printf("  Computing GTable (OpenSSL fallback)...\n");
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *x = BN_new(), *y = BN_new(), *shift = BN_new(), *inv2 = BN_new(), *order = BN_new(),
           *nri = BN_new(), *bscal = BN_new();
    EC_POINT *base = EC_POINT_new(grp), *pt = EC_POINT_new(grp), *two_base = EC_POINT_new(grp);
    /* base = A/2 = (2^-1 * neg_r_inv mod n) * G */
    EC_GROUP_get_order(grp, order, ctx);
    BN_set_word(shift, 2); BN_mod_inverse(inv2, shift, order, ctx);
    BN_lebin2bn(neg_r_inv, 32, nri);
    BN_mod_mul(bscal, inv2, nri, order, ctx);
    EC_POINT_mul(grp, base, bscal, NULL, NULL, ctx);
    for (int ch = 0; ch < GLV18_GT_CHUNKS; ch++) {
        if (ch > 0) { BN_set_word(shift, 1ul << (glv18_gt_shift(ch) - glv18_gt_shift(ch-1))); EC_POINT_mul(grp, base, NULL, base, shift, ctx); }
        BN_set_word(shift, 2); EC_POINT_mul(grp, two_base, NULL, base, shift, ctx);  /* 2*base_c */
        EC_POINT_copy(pt, base);                                                     /* (2*0+1)*base_c */
        for (unsigned d = 0; d < glv18_gt_entries(ch); d++) {
            EC_POINT_get_affine_coordinates_GFp(grp, pt, x, y, ctx);
            uint8_t xb[32], yb[32]; memset(xb,0,32); memset(yb,0,32);
            BN_bn2bin(x, xb+(32-BN_num_bytes(x)));
            BN_bn2bin(y, yb+(32-BN_num_bytes(y)));
            for(int j=0;j<16;j++){uint8_t t=xb[j];xb[j]=xb[31-j];xb[31-j]=t;}
            for(int j=0;j<16;j++){uint8_t t=yb[j];yb[j]=yb[31-j];yb[31-j]=t;}
            size_t off = ((size_t)glv18_gt_offset(ch) + d) * 64;
            memcpy(gTable + off,      xb, 32);
            memcpy(gTable + off + 32, yb, 32);
            if (d < glv18_gt_entries(ch) - 1) EC_POINT_add(grp, pt, pt, two_base, ctx);
        }
    }
    BN_free(x);BN_free(y);BN_free(shift);BN_free(inv2);BN_free(order);BN_free(nri);BN_free(bscal);
    EC_POINT_free(base);EC_POINT_free(pt);EC_POINT_free(two_base);
    EC_GROUP_free(grp);BN_CTX_free(ctx);
}
