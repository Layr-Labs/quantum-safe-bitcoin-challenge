// GPL-3.0-only. One-wave affine pairing of the three L2-resident GLV12 pairs.
// Included after tree_inverse.cuh. QSB_OW_L2PAIR is defined by tree.cu.
#pragma once
#if QSB_OW_L2PAIR && QSB_S3 && QSB_S3_HALF_WALK
/* (Q2+Q3), (P0+P1), (P2+P3) are affine sums of pinned-window points. The three
 * denominators x_b-x_a go through one block inverse (iso_scale=0: these are raw
 * table x's, not the recovery's isomorphic W). DRAM terms stay mixed adds.
 * A zero denominator substitutes 1 so the block product stays invertible, then
 * the candidate is dropped (ZZ=0). Every thread executes this body, including
 * the tree barrier, so the recovery tree later in kernel_digest still lines up. */
__device__ __forceinline__ void qsb_ow_affine(uint64_t *x, uint64_t *y,
    uint64_t *a, uint64_t *ay, uint64_t *b, uint64_t *by, const uint64_t *inv) {
    uint64_t m[4], t[4], vi[4];
    vi[0]=inv[0]; vi[1]=inv[1]; vi[2]=inv[2]; vi[3]=inv[3];
    _ModSub256(m, by, ay);
    _ModMult(m, vi);
    _ModSqr(t, m);
    _ModSub256(t, t, a);
    _ModSub256(x, t, b);
    _ModSub256(t, a, x);
    _ModMult(t, m);
    _ModSub256(y, t, ay);
}
__device__ __forceinline__ void qsb_ow_l2pair(uint64_t *X, uint64_t *Y, uint64_t *ZZ, uint64_t *ZZZ,
    const uint64_t k[4], const uint8_t *gTable, uint32_t &bad) {
    uint64_t mag[2][2]; unsigned sgn[2];
    q9_glv_split(k, mag[0], mag[1], &sgn[0], &sgn[1]);
    qsb_s3_walker w;
    qsb_s3_begin(w, mag[0], sgn[0], mag[1], sgn[1]);
    uint32_t code[12];
    #pragma unroll
    for (int t = 0; t < GT_GLV_TERMS; t++) {
        if (t == QSB_S3_PSI_TERM) qsb_s3_psi_swap(w);
        code[t] = qsb_s3_code_half(w, t, QSB_S3_DESC[t]);
    }
    const int pa[3] = {2, 6, 8};
    const int pb[3] = {3, 7, 9};
    uint64_t dx[3][4];
    int badpair = 0;
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        uint64_t ax[4], ay[4], bx[4], by[4];
        qsb_s3_load(gTable, code[pa[i]], QSB_S3_DESC[pa[i]].off >= GT_DENSE_ENTRIES, ax, ay);
        qsb_s3_load(gTable, code[pb[i]], QSB_S3_DESC[pb[i]].off >= GT_DENSE_ENTRIES, bx, by);
        _ModSub256(dx[i], bx, ax);
        if (!(dx[i][0] | dx[i][1] | dx[i][2] | dx[i][3])) {
            badpair = 1;
            dx[i][0] = 1; dx[i][1] = dx[i][2] = dx[i][3] = 0;
        }
    }
    uint64_t p[5], u[5], v[5], p01[5];
    #pragma unroll
    for (int s = 0; s < 4; s++) { u[s] = dx[0][s]; v[s] = dx[1][s]; }
    u[4] = v[4] = 0;
    QSB_TREE_MUL(p01, u, v);
    #pragma unroll
    for (int s = 0; s < 4; s++) v[s] = dx[2][s];
    v[4] = 0;
    QSB_TREE_MUL(p, p01, v);
    qsb_block_inverse_tree_at(p, 0);
    uint64_t inv0[5], inv1[5], inv2[5], inv01[5], d1[5];
    QSB_TREE_MUL(inv2, p, p01);
    QSB_TREE_MUL(inv01, p, v);
    #pragma unroll
    for (int s = 0; s < 4; s++) d1[s] = dx[1][s];
    d1[4] = 0;
    QSB_TREE_MUL(inv0, inv01, d1);
    QSB_TREE_MUL(inv1, inv01, u);
    uint64_t invs0[4], invs1[4], invs2[4];
    #pragma unroll
    for (int s = 0; s < 4; s++) { invs0[s] = inv0[s]; invs1[s] = inv1[s]; invs2[s] = inv2[s]; }

    uint64_t x0[4], y0[4], x1[4], y1[4], cx[4], cy[4], sx[4], sy[4];
    qsb_s3_load(gTable, code[0], false, x0, y0);
    qsb_s3_load(gTable, code[1], false, x1, y1);
    qsb_filter_point_seed(X, Y, ZZ, ZZZ, x0, y0, x1, y1, bad);
    {
        uint64_t ax[4], ay[4], bx[4], by[4];
        qsb_s3_load(gTable, code[2], false, ax, ay);
        qsb_s3_load(gTable, code[3], false, bx, by);
        qsb_ow_affine(sx, sy, ax, ay, bx, by, invs0);
    }
    qsb_filter_point_add<true>(X, Y, ZZ, ZZZ, sx, sy, y0, bad);
    qsb_s3_load(gTable, code[4], QSB_S3_DESC[4].off >= GT_DENSE_ENTRIES, cx, cy);
    qsb_filter_point_add<true>(X, Y, ZZ, ZZZ, cx, cy, y0, bad);
    qsb_s3_load(gTable, code[5], QSB_S3_DESC[5].off >= GT_DENSE_ENTRIES, cx, cy);
    qsb_filter_point_add<true>(X, Y, ZZ, ZZZ, cx, cy, y0, bad);
    {
        const uint64_t beta[4] = {0xC1396C28719501EEULL, 0x9CF0497512F58995ULL,
                                  0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL};
        qsb_filter_mul(X, X, beta, bad);
    }
    {
        uint64_t ax[4], ay[4], bx[4], by[4];
        qsb_s3_load(gTable, code[6], false, ax, ay);
        qsb_s3_load(gTable, code[7], false, bx, by);
        qsb_ow_affine(sx, sy, ax, ay, bx, by, invs1);
    }
    qsb_filter_point_add<true>(X, Y, ZZ, ZZZ, sx, sy, y0, bad);
    {
        uint64_t ax[4], ay[4], bx[4], by[4];
        qsb_s3_load(gTable, code[8], false, ax, ay);
        qsb_s3_load(gTable, code[9], false, bx, by);
        qsb_ow_affine(sx, sy, ax, ay, bx, by, invs2);
    }
    qsb_filter_point_add<true>(X, Y, ZZ, ZZZ, sx, sy, y0, bad);
    qsb_s3_load(gTable, code[10], QSB_S3_DESC[10].off >= GT_DENSE_ENTRIES, cx, cy);
    qsb_filter_point_add<true>(X, Y, ZZ, ZZZ, cx, cy, y0, bad);
    qsb_s3_load(gTable, code[11], QSB_S3_DESC[11].off >= GT_DENSE_ENTRIES, cx, cy);
    qsb_filter_last_add(X, Y, ZZ, ZZZ, cx, cy, y0, bad);
    if (badpair) {
        #pragma unroll
        for (int s = 0; s < 4; s++) { X[s] = 0; Y[s] = 0; ZZ[s] = 0; ZZZ[s] = 0; }
    }
}
#endif
