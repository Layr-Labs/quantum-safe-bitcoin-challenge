/* cg_ec_scalar.h -- scalar (4x64-limb) EC stage of the pinning co-grinder. Original code.
 * Included by cpu_cogrind.h inside namespace qcg. Same algorithm as the AVX2 stage in
 * cpu_cogrind_vec.h: candidates are processed in blocks of 4 ("lanes"); lane l of every block
 * belongs to product chain l, so each block issues 4 independent multiplications. */

struct sstate {
    uint64_t px[QSB_CG_BMAX][4], py[QSB_CG_BMAX][4], c[QSB_CG_BMAX][4];
};

/* 8 compressed-pubkey hashes (only H0), message words prepared per key */
static void pub_hash8_words(const uint32_t W9[8][9], uint32_t h0[8]);

/* x: canonical 4x64 -> message words W0..W8 of the 33-byte key */
static inline void pub_words(uint32_t *W, const uint64_t *x, int odd) {
    uint32_t X[8];
    for (int k = 0; k < 4; k++) { X[2 * k] = (uint32_t)x[k]; X[2 * k + 1] = (uint32_t)(x[k] >> 32); }
    W[0] = ((uint32_t)(0x02 | odd) << 24) | (X[7] >> 8);
    for (int j = 1; j < 8; j++) W[j] = (X[8 - j] << 24) | (X[7 - j] >> 8);
    W[8] = (X[0] << 24) | 0x00800000u;
}

template <class F>
static QCG_FE_INL void inv4(uint64_t inv[4][4], const uint64_t a[4][4]) {
    uint64_t p01[4], p23[4], p[4], ip[4], t0[4], t1[4];
    F::mul(p01, a[0], a[1]); F::mul(p23, a[2], a[3]); F::mul(p, p01, p23);
    scalar_inv(ip, p);
    F::mul(t0, ip, p23);                /* 1/(a0 a1) */
    F::mul(t1, ip, p01);                /* 1/(a2 a3) */
    F::mul(inv[0], t0, a[1]); F::mul(inv[1], t0, a[0]);
    F::mul(inv[2], t1, a[3]); F::mul(inv[3], t1, a[2]);
}

template <class F>
static __attribute__((noinline)) void ec_scalar(worker_t *w, sstate *ss) {
    shared_t *S = g_cg;
    const layout_t &L = S->lay;
    const int n = w->n;
    const int nb = (n + 3) >> 2;
    const tentry *T = S->table;
    uint64_t (*px)[4] = ss->px, (*py)[4] = ss->py, (*cc)[4] = ss->c;
    static const uint64_t ONE[4] = {1, 0, 0, 0};
    const int nw = L.nwin;
    /* window 0 */
    {
        const uint32_t *dg = w->dig[0];
        const tentry *T0 = T + L.off[0];
        for (int i = 0; i < 4 * nb; i++) {
            if (i + 4 * QSB_CG_PF < 4 * nb) __builtin_prefetch(T0 + dg[i + 4 * QSB_CG_PF]);
            const tentry *e = T0 + dg[i];
            memcpy(px[i], e->x, 32); memcpy(py[i], e->y, 32);
        }
    }
    uint64_t acc[4][4], u[4][4], inv[4][4];
    /* forward(1), ascending */
    {
        const uint32_t *dg = w->dig[1];
        const tentry *Tj = T + L.off[1];
        for (int l = 0; l < 4; l++) memcpy(acc[l], ONE, 32);
        for (int b = 0; b < nb; b++) {
            if (b + QSB_CG_PF < nb) for (int l = 0; l < 4; l++) __builtin_prefetch(Tj + (dg[4 * (b + QSB_CG_PF) + l] & QCG_IDXM));
            for (int l = 0; l < 4; l++) {
                const int i = 4 * b + l;
                uint64_t dx[4];
                if (QCG_ZERO(dg[i])) memcpy(dx, ONE, 32);
                else F::sub(dx, Tj[dg[i] & QCG_IDXM].x, px[i]);
                F::mul(acc[l], acc[l], dx);
                memcpy(cc[i], acc[l], 32);
            }
        }
    }
    int asc = 1;
    uint64_t xD[4], yD[4]; memcpy(xD, S->dx_w, 32); memcpy(yD, S->dy_w, 32);
    for (int s = 1; s < nw; s++) {
        inv4<F>(inv, acc);
        memcpy(u, inv, sizeof u);
        const uint32_t *dg = w->dig[s];
        const tentry *Tj = T + L.off[s];
        const int last = (s + 1 == nw);
        const uint32_t *dgn = last ? NULL : w->dig[s + 1];
        const tentry *Tn = last ? NULL : T + L.off[s + 1];
        for (int l = 0; l < 4; l++) memcpy(acc[l], ONE, 32);
        const int step = asc ? -1 : 1;
        int b = asc ? nb - 1 : 0;
        for (int it = 0; it < nb; it++, b += step) {
            const int bp = b + step;
            if (!last) {
                const int bf = b + step * QSB_CG_PF;
                if (bf >= 0 && bf < nb) for (int l = 0; l < 4; l++) __builtin_prefetch(Tn + (dgn[4 * bf + l] & QCG_IDXM));
            }
            /* every operation is issued for the 4 independent lanes back to back */
            const uint32_t *d4 = dg + 4 * b;
            const tentry *e[4];
            uint64_t dx[4][4], ik[4][4], dy[4][4], lam[4][4], l2[4][4], x3[4][4], t[4][4], y3[4][4], dxn[4][4];
            for (int l = 0; l < 4; l++) {
                e[l] = Tj + (d4[l] & QCG_IDXM);
                if (QCG_ZERO(d4[l])) memcpy(dx[l], ONE, 32); else F::sub(dx[l], e[l]->x, px[4 * b + l]);
            }
            if (it + 1 < nb) {
                for (int l = 0; l < 4; l++) F::mul(ik[l], u[l], cc[4 * bp + l]);
                for (int l = 0; l < 4; l++) F::mul(u[l], u[l], dx[l]);
            } else memcpy(ik, u, sizeof ik);
            for (int l = 0; l < 4; l++) {
                uint64_t yT[4];
                if (QCG_NEG(d4[l])) qcg_fe::fe_neg_c(yT, e[l]->y); else memcpy(yT, e[l]->y, 32);
                F::sub(dy[l], yT, py[4 * b + l]);
            }
            for (int l = 0; l < 4; l++) F::mul(lam[l], dy[l], ik[l]);
            for (int l = 0; l < 4; l++) F::sqr(l2[l], lam[l]);
            for (int l = 0; l < 4; l++) { F::sub(x3[l], l2[l], px[4 * b + l]); F::sub(x3[l], x3[l], e[l]->x); F::sub(t[l], px[4 * b + l], x3[l]); }
            for (int l = 0; l < 4; l++) F::mul(y3[l], lam[l], t[l]);
            for (int l = 0; l < 4; l++) {
                if (!QCG_ZERO(d4[l])) {
                    F::sub(y3[l], y3[l], py[4 * b + l]);
                    memcpy(px[4 * b + l], x3[l], 32); memcpy(py[4 * b + l], y3[l], 32);
                }
                const int i = 4 * b + l;
                if (!last) {
                    const uint32_t cn = dgn[i];
                    if (QCG_ZERO(cn)) memcpy(dxn[l], ONE, 32); else F::sub(dxn[l], Tn[cn & QCG_IDXM].x, px[i]);
                } else F::sub(dxn[l], xD, px[i]);
            }
            for (int l = 0; l < 4; l++) { F::mul(acc[l], acc[l], dxn[l]); memcpy(cc[4 * b + l], acc[l], 32); }
        }
        asc = !asc;
    }
    /* Q- = Q+ + D, hash both */
    inv4<F>(inv, acc);
    memcpy(u, inv, sizeof u);
    {
        const int step = asc ? -1 : 1;
        int b = asc ? nb - 1 : 0;
        for (int it = 0; it < nb; it++, b += step) {
            const int bp = b + step;
            uint32_t W9[8][9];
            for (int l = 0; l < 4; l++) {
                const int i = 4 * b + l;
                uint64_t dx[4], ik[4], dy[4], lam[4], l2[4], xm[4], t[4], ym[4], xp[4], yp[4];
                F::sub(dx, xD, px[i]);
                if (it + 1 < nb) { F::mul(ik, u[l], cc[4 * bp + l]); F::mul(u[l], u[l], dx); } else memcpy(ik, u[l], 32);
                F::sub(dy, yD, py[i]);
                F::mul(lam, dy, ik);
                F::sqr(l2, lam);
                F::sub(xm, l2, px[i]); F::sub(xm, xm, xD);
                F::sub(t, px[i], xm);
                F::mul(ym, lam, t);
                F::sub(ym, ym, py[i]);
                memcpy(xp, px[i], 32); memcpy(yp, py[i], 32);
                qcg_fe::fe_norm_c(xp); qcg_fe::fe_norm_c(yp); qcg_fe::fe_norm_c(xm); qcg_fe::fe_norm_c(ym);
#ifdef QCG_EC_HOOK
                if (i < n) QCG_EC_HOOK(w, i, xp, yp, xm, ym);
#endif
                pub_words(W9[l], xp, (int)(yp[0] & 1));
                pub_words(W9[4 + l], xm, (int)(ym[0] & 1));
            }
            uint32_t h0[8];
            pub_hash8_words(W9, h0);
            for (int k = 0; k < 8; k++) {
#if QSB_ZEROS_N >= 32
                const int pass = h0[k] == 0;
#else
                const int pass = (h0[k] >> (32 - QSB_ZEROS_N)) == 0;
#endif
                if (pass) { const int i = 4 * b + (k & 3); if (i < n) publish(w, i, k >> 2); }
            }
        }
    }
}

__attribute__((target("avx2"), noinline)) static void pub_hash8_avx2(const uint32_t W9[8][9], uint32_t h0[8]) {
    using namespace qcg_sha;
    v8u W[16], st[8];
    for (int j = 0; j < 9; j++) W[j] = _mm256_setr_epi32((int)W9[0][j], (int)W9[1][j], (int)W9[2][j], (int)W9[3][j], (int)W9[4][j], (int)W9[5][j], (int)W9[6][j], (int)W9[7][j]);
    for (int j = 9; j < 15; j++) W[j] = _mm256_setzero_si256();
    W[15] = _mm256_set1_epi32(264);
    for (int j = 0; j < 8; j++) st[j] = _mm256_set1_epi32((int)IV256[j]);
    s8_compress_full(st, W);
    _mm256_storeu_si256((__m256i *)h0, st[0]);
}
__attribute__((target("sha,sse4.1"), noinline)) static void pub_hash8_shani(const uint32_t W9[8][9], uint32_t h0[8]) {
    using namespace qcg_sha;
    for (int k = 0; k < 8; k += 2) {
        uint32_t wa[16], wb[16], sa[8], sb[8];
        memcpy(wa, W9[k], 36); memset(wa + 9, 0, 24); wa[15] = 264;
        memcpy(wb, W9[k + 1], 36); memset(wb + 9, 0, 24); wb[15] = 264;
        memcpy(sa, IV256, 32); memcpy(sb, IV256, 32);
        shani_compress2(sa, wa, sb, wb);
        h0[k] = sa[0]; h0[k + 1] = sb[0];
    }
}

/* word-major variant (Wt[j][k] = word j of key k), used by the AVX2 EC stage on SHA-NI hosts */
__attribute__((target("sha,sse4.1"), noinline)) static void pub_hash8_shani_wm(const uint32_t Wt[9][8], uint32_t h0[8]) {
    using namespace qcg_sha;
    for (int k = 0; k < 8; k += 2) {
        uint32_t wa[16], wb[16], sa[8], sb[8];
        for (int j = 0; j < 9; j++) { wa[j] = Wt[j][k]; wb[j] = Wt[j][k + 1]; }
        for (int j = 9; j < 15; j++) { wa[j] = 0; wb[j] = 0; }
        wa[15] = 264; wb[15] = 264;
        memcpy(sa, IV256, 32); memcpy(sb, IV256, 32);
        shani_compress2(sa, wa, sb, wb);
        h0[k] = sa[0]; h0[k + 1] = sb[0];
    }
}
static void pub_hash8_words(const uint32_t W9[8][9], uint32_t h0[8]) {
    const int m = g_cg->sha_mode.load(std::memory_order_relaxed);
    if (m == 2) { pub_hash8_shani(W9, h0); return; }
    if (m == 1 || (m == 0 && g_cg->has_avx2)) { pub_hash8_avx2(W9, h0); return; }
    for (int k = 0; k < 8; k++) {
        uint32_t wv[16], st[8];
        memcpy(wv, W9[k], 36); memset(wv + 9, 0, 24); wv[15] = 264;
        memcpy(st, qcg_sha::IV256, 32);
        qcg_sha::sha_compress_ref(st, wv);
        h0[k] = st[0];
    }
}
