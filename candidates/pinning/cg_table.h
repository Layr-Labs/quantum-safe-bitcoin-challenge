/* cg_table.h -- background build of the co-grinder's fixed-base table. Original code.
 * Included by cpu_cogrind.h inside namespace qcg.
 *
 * Window 0 (A folded in): entry d = d B + A,             d = 0 .. 2^w0 - 1.
 * Window j >= 1:          entry e = (e + 1) 2^pos_j B,   e = 0 .. 2^(w_j - 1) - 1.
 * Each window is cut into segments of QSB_CG_SEG entries; a segment starts from an OpenSSL
 * scalar multiple and proceeds in rows of TB_R entries: row 0 = start + k G_j (k G_j precomputed
 * per window), row r = row r-1 + TB_R G_j, one batched inversion per row (doubling handled).
 * Segments are shared out to SCHED_IDLE builder threads. The table is then spot-checked
 * against OpenSSL; any mismatch disables the co-grinder.
 */
#ifndef QSB_CG_SEG
#define QSB_CG_SEG 4096
#endif
#define TB_R 256

typedef uint64_t fe4_t[4];

struct tb_ctx {
    shared_t *S;
    fe4_t *kg[QSB_CG_MAXWIN];             /* kg[j][k] = (k+1) G_j, k = 0..TB_R-1 (x,y interleaved: 2 per point) */
    uint64_t seg_first[QSB_CG_MAXWIN + 1];/* prefix count of segments per window */
    std::atomic<uint64_t> next_seg;
    std::atomic<int> bad;
    int nthreads;
};
static tb_ctx g_tb;

static void bn_to_w(const BIGNUM *b, uint64_t *w) { uint8_t t[32]; BN_bn2lebinpad(b, t, 32); memcpy(w, t, 32); }

/* affine (x, y) of k * G_j (+ A if addA) through OpenSSL; k given as BIGNUM scalar of B */
static int ossl_point(EC_GROUP *grp, BN_CTX *ctx, const BIGNUM *sc, int addA, uint64_t *x, uint64_t *y) {
    int ok = 0;
    EC_POINT *P = EC_POINT_new(grp), *A = EC_POINT_new(grp);
    BIGNUM *bx = BN_new(), *by = BN_new(), *ax = BN_new(), *ay = BN_new();
    if (P && A && bx && by && ax && ay && EC_POINT_mul(grp, P, sc, NULL, NULL, ctx)) {
        ok = 1;
        if (addA) {
            const pinning2_params_t *pp = g_cg->pp;
            ok = BN_lebin2bn(pp->u2r_x, 32, ax) && BN_lebin2bn(pp->u2r_y, 32, ay) &&
                 EC_POINT_set_affine_coordinates(grp, A, ax, ay, ctx) && EC_POINT_add(grp, P, P, A, ctx);
        }
        if (ok) ok = !EC_POINT_is_at_infinity(grp, P) && EC_POINT_get_affine_coordinates(grp, P, bx, by, ctx);
        if (ok) { bn_to_w(bx, x); bn_to_w(by, y); }
    }
    EC_POINT_free(P); EC_POINT_free(A); BN_free(bx); BN_free(by); BN_free(ax); BN_free(ay);
    return ok;
}
/* scalar of B for entry e of window j: (e + off_j) * 2^pos_j * neg_r_inv mod n */
static void entry_scalar(BIGNUM *r, int j, uint64_t e, const BIGNUM *nri, const BIGNUM *order, BN_CTX *ctx) {
    const layout_t &L = g_cg->lay;
    BN_set_word(r, (BN_ULONG)(e + (j ? 1 : 0)));
    BN_lshift(r, r, L.pos[j]);
    BN_mod_mul(r, r, nri, order, ctx);
}

/* R[i] = P[i] + Q[i] for i < n (Q[i] = Q[0] if qsingle). Handles P == Q; fails on P == -Q.
 * All inputs canonical or lazy; outputs canonical. tmp: 3 n elements. */
template <class F>
static int batch_add(fe4_t *rx, fe4_t *ry, const fe4_t *px, const fe4_t *py, const fe4_t *qx, const fe4_t *qy, int qsingle, int n, fe4_t *tmp) {
    fe4_t *num = tmp, *den = tmp + n, *pre = tmp + 2 * n;
    for (int i = 0; i < n; i++) {
        const uint64_t *ax = px[i], *ay = py[i], *bx = qx[qsingle ? 0 : i], *by = qy[qsingle ? 0 : i];
        F::sub(den[i], bx, ax);
        F::sub(num[i], by, ay);
        if (qcg_fe::fe_is_zero_c(den[i])) {
            if (!qcg_fe::fe_is_zero_c(num[i])) return -1;        /* P == -Q */
            uint64_t x2[4], t[4];
            F::sqr(x2, ax); F::add(t, x2, x2); F::add(num[i], t, x2);   /* 3 x^2 */
            F::add(den[i], ay, ay);                                        /* 2 y */
            if (qcg_fe::fe_is_zero_c(den[i])) return -1;
        }
        if (i == 0) memcpy(pre[0], den[0], 32); else F::mul(pre[i], pre[i - 1], den[i]);
    }
    uint64_t u[4], ik[4], lam[4], l2[4], t[4], x3[4], y3[4];
    if (qcg_fe::fe_is_zero_c(pre[n - 1])) return -1;
    { uint64_t p[4]; memcpy(p, pre[n - 1], 32); qcg_fe::fe_inv<F>(u, p); }
    for (int i = n - 1; i >= 0; i--) {
        if (i > 0) { F::mul(ik, u, pre[i - 1]); F::mul(u, u, den[i]); } else memcpy(ik, u, 32);
        const uint64_t *ax = px[i], *ay = py[i], *bx = qx[qsingle ? 0 : i];
        F::mul(lam, num[i], ik);
        F::sqr(l2, lam);
        F::sub(x3, l2, ax); F::sub(x3, x3, bx);
        F::sub(t, ax, x3);
        F::mul(y3, lam, t);
        F::sub(y3, y3, ay);
        qcg_fe::fe_norm_c(x3); qcg_fe::fe_norm_c(y3);
        memcpy(rx[i], x3, 32); memcpy(ry[i], y3, 32);
    }
    return 0;
}

template <class F>
static int build_kg(int j, const uint64_t *gx, const uint64_t *gy) {
    /* kg[k] = (k+1) G, k = 0..TB_R-1, by doubling the known prefix: P[s+k] = P[k] + P[s] */
    fe4_t *X = (fe4_t *)malloc(sizeof(fe4_t) * TB_R), *Y = (fe4_t *)malloc(sizeof(fe4_t) * TB_R);
    fe4_t *tmp = (fe4_t *)malloc(sizeof(fe4_t) * 3 * TB_R), *qx = (fe4_t *)malloc(sizeof(fe4_t) * TB_R), *qy = (fe4_t *)malloc(sizeof(fe4_t) * TB_R);
    int rc = 0;
    memcpy(X[0], gx, 32); memcpy(Y[0], gy, 32);
    for (int s = 1; s < TB_R && rc == 0; s *= 2) {
        const int cnt = (2 * s <= TB_R) ? s : TB_R - s;
        /* (s + k + 1) G = (k + 1) G + s G, k = 0..cnt-1; s G = X[s-1] */
        rc = batch_add<F>(qx, qy, X, Y, &X[s - 1], &Y[s - 1], 1, cnt, tmp);
        for (int k = 0; k < cnt && rc == 0; k++) { memcpy(X[s + k], qx[k], 32); memcpy(Y[s + k], qy[k], 32); }
    }
    fe4_t *kg = (fe4_t *)malloc(sizeof(fe4_t) * 2 * TB_R);
    for (int k = 0; k < TB_R; k++) { memcpy(kg[2 * k], X[k], 32); memcpy(kg[2 * k + 1], Y[k], 32); }
    g_tb.kg[j] = kg;
    free(X); free(Y); free(tmp); free(qx); free(qy);
    return rc;
}

template <class F>
static int build_segment(int j, uint64_t seg, EC_GROUP *grp, BN_CTX *ctx, const BIGNUM *nri, const BIGNUM *order, BIGNUM *sc,
                         fe4_t *rowx, fe4_t *rowy, fe4_t *nx, fe4_t *ny, fe4_t *kx, fe4_t *ky, fe4_t *tmp) {
    shared_t *S = g_cg;
    const layout_t &L = S->lay;
    const uint64_t e0 = seg * QSB_CG_SEG;
    const uint64_t ne = (L.cnt[j] - e0 < QSB_CG_SEG) ? L.cnt[j] - e0 : QSB_CG_SEG;
    tentry *out = S->table + L.off[j] + e0;
    const fe4_t *kg = g_tb.kg[j];
    for (int k = 0; k < TB_R; k++) { memcpy(kx[k], kg[2 * k], 32); memcpy(ky[k], kg[2 * k + 1], 32); }
    /* row 0: start + k G_j, k = 0..TB_R-1 */
    uint64_t sx[4], sy[4];
    if (j >= 1 && seg == 0) {
        /* entries (k+1) G_j are kg itself */
        for (int k = 0; k < TB_R; k++) { memcpy(rowx[k], kx[k], 32); memcpy(rowy[k], ky[k], 32); }
    } else {
        entry_scalar(sc, j, e0, nri, order, ctx);
        if (!ossl_point(grp, ctx, sc, j == 0, sx, sy)) return -1;
        memcpy(rowx[0], sx, 32); memcpy(rowy[0], sy, 32);
        fe4_t px[1], py[1]; memcpy(px[0], sx, 32); memcpy(py[0], sy, 32);
        /* start + (k) G = start + kg[k-1], k = 1..TB_R-1 */
        for (int k = 1; k < TB_R; k++) { memcpy(nx[k - 1], sx, 32); memcpy(ny[k - 1], sy, 32); }
        if (batch_add<F>(rowx + 1, rowy + 1, nx, ny, kx, ky, 0, TB_R - 1, tmp)) return -1;
    }
    uint64_t done = 0;
    for (;;) {
        const uint64_t take = (ne - done < TB_R) ? ne - done : TB_R;
        for (uint64_t k = 0; k < take; k++) { memcpy(out[done + k].x, rowx[k], 32); memcpy(out[done + k].y, rowy[k], 32); }
        done += take;
        if (done >= ne) break;
        /* next row: + TB_R G_j = kg[TB_R-1] */
        if (batch_add<F>(nx, ny, rowx, rowy, &kx[TB_R - 1], &ky[TB_R - 1], 1, TB_R, tmp)) return -1;
        memcpy(rowx, nx, sizeof(fe4_t) * TB_R); memcpy(rowy, ny, sizeof(fe4_t) * TB_R);
    }
    return 0;
}

template <class F>
static void *builder_main_t(void *arg) {
    (void)arg;
    set_idle_priority();
    shared_t *S = g_cg;
    const layout_t &L = S->lay;
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *nri = BN_new(), *order = BN_new(), *sc = BN_new();
    fe4_t *buf = (fe4_t *)malloc(sizeof(fe4_t) * TB_R * 9);
    if (!grp || !ctx || !nri || !order || !sc || !buf) { g_tb.bad.store(1); return NULL; }
    EC_GROUP_get_order(grp, order, ctx);
    BN_lebin2bn(S->pp->neg_r_inv, 32, nri);
    const uint64_t total = g_tb.seg_first[L.nwin];
    for (;;) {
        const uint64_t s = g_tb.next_seg.fetch_add(1);
        if (s >= total || g_tb.bad.load() || S->stop.load()) break;
        int j = 0; while (s >= g_tb.seg_first[j + 1]) j++;
        if (build_segment<F>(j, s - g_tb.seg_first[j], grp, ctx, nri, order, sc,
                             buf, buf + TB_R, buf + 2 * TB_R, buf + 3 * TB_R, buf + 4 * TB_R, buf + 5 * TB_R, buf + 6 * TB_R))
            g_tb.bad.store(1);
    }
    free(buf); BN_free(nri); BN_free(order); BN_free(sc); BN_CTX_free(ctx); EC_GROUP_free(grp);
    return NULL;
}

/* compare a few entries of every window with OpenSSL */
static int table_spot_check() {
    shared_t *S = g_cg;
    const layout_t &L = S->lay;
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *nri = BN_new(), *order = BN_new(), *sc = BN_new();
    EC_GROUP_get_order(grp, order, ctx);
    BN_lebin2bn(S->pp->neg_r_inv, 32, nri);
    int bad = 0;
    uint64_t rs = 0x9E3779B97F4A7C15ULL;
    for (int j = 0; j < L.nwin && !bad; j++) {
        for (int t = 0; t < 6 && !bad; t++) {
            rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
            const uint64_t e = t == 0 ? 0 : t == 1 ? 1 : t == 2 ? L.cnt[j] - 1 : t == 3 ? QSB_CG_SEG + TB_R : rs % L.cnt[j];
            if (e >= L.cnt[j]) continue;
            uint64_t x[4], y[4];
            entry_scalar(sc, j, e, nri, order, ctx);
            if (!ossl_point(grp, ctx, sc, j == 0, x, y)) { bad = 1; break; }
            const tentry *te = S->table + L.off[j] + e;
            if (memcmp(te->x, x, 32) || memcmp(te->y, y, 32)) bad = 1;
        }
    }
    BN_free(nri); BN_free(order); BN_free(sc); BN_CTX_free(ctx); EC_GROUP_free(grp);
    return bad;
}

static void *table_helper(void *arg) {
    (void)arg;
    /* start() pins the GPU host thread before spawning this helper. Restore the
     * non-host worker cpuset here so all child table builders inherit it. */
    if (g_worker_set_on)
        pthread_setaffinity_np(pthread_self(), sizeof g_worker_set, &g_worker_set);
    set_idle_priority();
    shared_t *S = g_cg;
    const double t0 = mono_s();
    pthread_t th[QSB_CG_MAXW];
    int nt = g_tb.nthreads, started = 0;
    for (int b = 0; b < nt; b++) {
        if (pthread_create(&th[b], NULL, g_fe_asm ? builder_main_t<qcg_fe::FeAsm> : builder_main_t<qcg_fe::FeC>, NULL) == 0) started++;
        else break;
    }
    for (int b = 0; b < started; b++) pthread_join(th[b], NULL);
    if (!started) g_tb.bad.store(1);
    S->t_build = mono_s() - t0;
    if (g_tb.bad.load() || table_spot_check()) {
        printf("  CPU co-grind: table build failed its OpenSSL spot check; off\n");
        S->failed.store(1);
        return NULL;
    }
    S->ready.store(1, std::memory_order_release);
    return NULL;
}

/* allocate, compute constants, start the builders (returns at once) */
static int table_start(shared_t *S, int nw) {
    /* allocation with fall back to smaller layouts */
    const char *order_nm[5] = {"xlarge", "large", "medium", "small", "tiny"};
    int li = 0; while (li < 5 && strcmp(order_nm[li], S->lay.name)) li++;
    void *mem = NULL;
    for (; li < 5; li++) {
        layout_t L; layout_by_name(&L, order_nm[li]);
        const size_t bytes = (size_t)L.total * sizeof(tentry);
        const size_t al = 2u << 20;
        void *m = mmap(NULL, bytes + al, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (m == MAP_FAILED) continue;
        mem = (void *)(((uintptr_t)m + al - 1) & ~(uintptr_t)(al - 1));
        madvise(mem, bytes, getenv("QSB_COGRIND_NOTHP") ? MADV_NOHUGEPAGE : MADV_HUGEPAGE);
        S->lay = L; S->table = (tentry *)mem; S->table_bytes = bytes;
        break;
    }
    if (!mem) return -1;
    const layout_t &L = S->lay;
    /* A, D = -2A, window bases G_j = 2^pos_j B */
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *order = BN_new(), *nri = BN_new(), *sc = BN_new(), *bx = BN_new(), *by = BN_new();
    EC_POINT *A = EC_POINT_new(grp), *D = EC_POINT_new(grp);
    int ok = grp && ctx && order && nri && sc && bx && by && A && D;
    if (ok) ok = EC_GROUP_get_order(grp, order, ctx) && BN_lebin2bn(S->pp->neg_r_inv, 32, nri) &&
                 BN_lebin2bn(S->pp->u2r_x, 32, bx) && BN_lebin2bn(S->pp->u2r_y, 32, by) &&
                 EC_POINT_set_affine_coordinates(grp, A, bx, by, ctx) &&
                 EC_POINT_dbl(grp, D, A, ctx) && EC_POINT_invert(grp, D, ctx) &&
                 EC_POINT_get_affine_coordinates(grp, D, bx, by, ctx);
    if (ok) { bn_to_w(bx, S->dx_w); bn_to_w(by, S->dy_w); memcpy(S->ax_w, S->pp->u2r_x, 32); memcpy(S->ay_w, S->pp->u2r_y, 32); }
    /* per-window k G_j tables (TB_R points each), serial: ~1 ms per window */
    for (int j = 0; j < L.nwin && ok; j++) {
        uint64_t gx[4], gy[4];
        BN_one(sc); BN_lshift(sc, sc, L.pos[j]); BN_mod_mul(sc, sc, nri, order, ctx);
        ok = ossl_point(grp, ctx, sc, 0, gx, gy);
        if (ok) ok = (g_fe_asm ? build_kg<qcg_fe::FeAsm>(j, gx, gy) : build_kg<qcg_fe::FeC>(j, gx, gy)) == 0;
    }
    EC_POINT_free(A); EC_POINT_free(D); BN_free(order); BN_free(nri); BN_free(sc); BN_free(bx); BN_free(by);
    BN_CTX_free(ctx); EC_GROUP_free(grp);
    if (!ok) return -1;
    g_tb.S = S;
    g_tb.seg_first[0] = 0;
    for (int j = 0; j < L.nwin; j++) g_tb.seg_first[j + 1] = g_tb.seg_first[j] + (L.cnt[j] + QSB_CG_SEG - 1) / QSB_CG_SEG;
    g_tb.next_seg.store(0); g_tb.bad.store(0);
    g_tb.nthreads = nw < 1 ? 1 : nw;
    pthread_t ht;
    if (pthread_create(&ht, NULL, table_helper, NULL) != 0) return -1;
    pthread_detach(ht);
    return 0;
}
