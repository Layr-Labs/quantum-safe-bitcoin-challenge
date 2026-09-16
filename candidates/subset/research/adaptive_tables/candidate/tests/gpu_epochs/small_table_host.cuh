#pragma once
static void small_point_to_limbs(EC_GROUP *grp, EC_POINT *pt, BIGNUM *x, BIGNUM *y,
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
static void small_build_ladders(uint64_t *hL, uint64_t *hH, const uint8_t neg_r_inv[32]) {
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
    memset(hL, 0, (size_t)SMALL_CHUNKS * SMALL_LO * 8 * sizeof(uint64_t));
    memset(hH, 0, (size_t)SMALL_CHUNKS * SMALL_HI * 8 * sizeof(uint64_t));
    for (int ch = 0; ch < SMALL_CHUNKS; ch++) {
        if (ch > 0) { BN_set_word(shift, 65536); EC_POINT_mul(grp, base, NULL, base, shift, ctx); }
        EC_POINT_copy(acc, base);
        for (int lo = 1; lo < SMALL_LO; lo++) {                 /* L[lo] = lo * B */
            small_point_to_limbs(grp, acc, x, y, ctx, hL + ((size_t)ch * SMALL_LO + lo) * 8);
            EC_POINT_add(grp, acc, acc, base, ctx);
        }
        BN_set_word(shift, 256);                             /* step = 256 * B */
        EC_POINT_mul(grp, step, NULL, base, shift, ctx);
        EC_POINT_copy(acc, step);
        for (int hi = 1; hi < SMALL_HI; hi++) {                 /* H[hi] = hi * 256 * B */
            small_point_to_limbs(grp, acc, x, y, ctx, hH + ((size_t)ch * SMALL_HI + hi) * 8);
            EC_POINT_add(grp, acc, acc, step, ctx);
        }
    }
    BN_free(x); BN_free(y); BN_free(shift); BN_free(inv2); BN_free(order);
    BN_free(nri); BN_free(bscal);
    EC_POINT_free(base); EC_POINT_free(step); EC_POINT_free(acc);
    EC_GROUP_free(grp); BN_CTX_free(ctx);
}
static int small_spot_check(const uint8_t *gTableX, const uint8_t *gTableY, int samples,
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
        if (t < SMALL_CHUNKS * 4) {
            ch = t / 4;
            static const int corner[4] = {0, 1, 2, SMALL_ENTRIES - 1};
            i = corner[t % 4];
        } else {
            seed = seed * 1664525u + 1013904223u;
            ch = (int)(seed >> 28) % SMALL_CHUNKS;
            i  = (int)((seed >> 4) & (SMALL_ENTRIES - 1));
        }
        /* want = (2i+1) * 2^(16*ch) * (A/2) = (2i+1) * 2^(16*ch) * (inv2*neg_r_inv) * G (mod n) */
        BN_one(k);
        BN_lshift(k, k, 16 * ch);
        BN_mul_word(k, (BN_ULONG)(2*i + 1));
        BN_mod_mul(k, k, half_nri, order, ctx);
        EC_POINT_mul(grp, pt, k, NULL, NULL, ctx);
        small_point_to_limbs(grp, pt, x, y, ctx, want);
        size_t off = ((size_t)ch * SMALL_ENTRIES + i) * 32;
        if (memcmp(gTableX + off, want,     32) != 0 ||
            memcmp(gTableY + off, want + 4, 32) != 0) {
            fprintf(stderr, "  GTable spot check FAILED at chunk %d entry %d\n", ch, i);
            ok = 0;
        }
    }
    BN_free(x); BN_free(y); BN_free(k); BN_free(inv2); BN_free(order); BN_free(nri); BN_free(half_nri);
    EC_POINT_free(pt); EC_GROUP_free(grp); BN_CTX_free(ctx);
    return ok;
}
static void small_compute_gtable(uint8_t *gTableX, uint8_t *gTableY, const uint8_t neg_r_inv[32]) {
    size_t gt_bytes = (size_t)SMALL_CHUNKS * SMALL_ENTRIES * 32;
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
    for (int ch = 0; ch < SMALL_CHUNKS; ch++) {
        if (ch > 0) { BN_set_word(shift, 65536); EC_POINT_mul(grp, base, NULL, base, shift, ctx); }
        BN_set_word(shift, 2); EC_POINT_mul(grp, two_base, NULL, base, shift, ctx);  /* 2*base_c */
        EC_POINT_copy(pt, base);                                                     /* (2*0+1)*base_c */
        for (unsigned d = 0; d < SMALL_ENTRIES; d++) {
            EC_POINT_get_affine_coordinates_GFp(grp, pt, x, y, ctx);
            uint8_t xb[32], yb[32]; memset(xb,0,32); memset(yb,0,32);
            BN_bn2bin(x, xb+(32-BN_num_bytes(x)));
            BN_bn2bin(y, yb+(32-BN_num_bytes(y)));
            for(int j=0;j<16;j++){uint8_t t=xb[j];xb[j]=xb[31-j];xb[31-j]=t;}
            for(int j=0;j<16;j++){uint8_t t=yb[j];yb[j]=yb[31-j];yb[31-j]=t;}
            size_t off = ((size_t)ch * SMALL_ENTRIES + d) * 32;
            memcpy(gTableX + off, xb, 32);
            memcpy(gTableY + off, yb, 32);
            if (d < SMALL_ENTRIES - 1) EC_POINT_add(grp, pt, pt, two_base, ctx);
        }
    }
    BN_free(x);BN_free(y);BN_free(shift);BN_free(inv2);BN_free(order);BN_free(nri);BN_free(bscal);
    EC_POINT_free(base);EC_POINT_free(pt);EC_POINT_free(two_base);
    EC_GROUP_free(grp);BN_CTX_free(ctx);
    (void)gt_bytes;
}
static void small_build_table(uint8_t **tableX,uint8_t **tableY,const uint8_t nri[32]){
    const size_t bytes=(size_t)SMALL_CHUNKS*SMALL_ENTRIES*32;
    wide_cuda_require(cudaMalloc(tableX,bytes),"allocate small table X");
    wide_cuda_require(cudaMalloc(tableY,bytes),"allocate small table Y");
    const size_t lb=(size_t)SMALL_CHUNKS*SMALL_LO*8*sizeof(uint64_t);
    const size_t hb=(size_t)SMALL_CHUNKS*SMALL_HI*8*sizeof(uint64_t);
    std::vector<uint64_t>L(lb/8),H(hb/8);
    small_build_ladders(L.data(),H.data(),nri);
    uint64_t *dL=nullptr,*dH=nullptr;
    wide_cuda_require(cudaMalloc(&dL,lb),"allocate small L");
    wide_cuda_require(cudaMalloc(&dH,hb),"allocate small H");
    wide_cuda_require(cudaMemcpy(dL,L.data(),lb,cudaMemcpyHostToDevice),"copy small L");
    wide_cuda_require(cudaMemcpy(dH,H.data(),hb,cudaMemcpyHostToDevice),"copy small H");
    small_build_kernel<<<(SMALL_CHUNKS*SMALL_ENTRIES+255)/256,256>>>(dL,dH,*tableX,*tableY);
    wide_cuda_require(cudaGetLastError(),"small table launch");
    wide_cuda_require(cudaDeviceSynchronize(),"small table execution");
    wide_cuda_require(cudaFree(dL),"free small L");wide_cuda_require(cudaFree(dH),"free small H");
    std::vector<uint8_t>x(bytes),y(bytes);
    wide_cuda_require(cudaMemcpy(x.data(),*tableX,bytes,cudaMemcpyDeviceToHost),"check small X");
    wide_cuda_require(cudaMemcpy(y.data(),*tableY,bytes,cudaMemcpyDeviceToHost),"check small Y");
    if(!small_spot_check(x.data(),y.data(),SMALL_CHUNKS*4+192,nri)){
        small_compute_gtable(x.data(),y.data(),nri);
        if(!small_spot_check(x.data(),y.data(),SMALL_CHUNKS*4+192,nri)){
            fprintf(stderr,"Small table OpenSSL fallback failed\n");exit(2);
        }
        wide_cuda_require(cudaMemcpy(*tableX,x.data(),bytes,cudaMemcpyHostToDevice),"copy fallback X");
        wide_cuda_require(cudaMemcpy(*tableY,y.data(),bytes,cudaMemcpyHostToDevice),"copy fallback Y");
    }
}
