/* Scalar safegcd from terrapinelf Subset 97f347a8 / aef1aef, after
 * libsecp256k1 modinv64_var (MIT; COPYING-secp256k1). Pinning boundary adapter. */
#pragma once
struct s62 { int64_t v[5]; };                         /* signed, 62-bit limbs (the top one carries the sign) */
struct trans2x2 { int64_t u, v, q, r; };
static const s62 S62_P = {{-0x1000003D1LL, 0, 0, 0, 256}};   /* p = 2^256 - 0x1000003D1 */
static const uint64_t S62_PINV = 0x27C7F6E22DDACACFULL;      /* p^-1 mod 2^62 */
/* 62 divsteps on the low limbs (f0, g0) of (f, g); the transition matrix t and the new eta. */
static inline int64_t divsteps_62_var(int64_t eta, uint64_t f0, uint64_t g0, trans2x2 *t) {
    uint64_t u = 1, v = 0, q = 0, r = 1, f = f0, g = g0, m; uint32_t w; int i = 62, limit, zeros;
    for (;;) {
        zeros = __builtin_ctzll(g | (~0ULL << i));    /* a sentinel bit counts zeros only up to i */
        g >>= zeros; u <<= zeros; v <<= zeros; eta -= zeros; i -= zeros;
        if (i == 0) break;
        if (eta < 0) {
            uint64_t tmp;
            eta = -eta;
            tmp = f; f = g; g = -tmp;
            tmp = u; u = q; q = -tmp;
            tmp = v; v = r; r = -tmp;
            limit = ((int)eta + 1) > i ? i : ((int)eta + 1);
            m = (~0ULL >> (64 - limit)) & 63U;
            w = (uint32_t)((f * g * (f * f - 2)) & m);   /* -g/f mod 2^6 (f odd: f*(2-f^2) = f^-1 mod 64) */
        } else {
            limit = ((int)eta + 1) > i ? i : ((int)eta + 1);
            m = (~0ULL >> (64 - limit)) & 15U;
            w = (uint32_t)(f + (((f + 1) & 4) << 1));      /* f^-1 mod 16 */
            w = (uint32_t)((-(uint64_t)w * g) & m);
        }
        g += f * w; q += u * w; r += v * w;
    }
    t->u = (int64_t)u; t->v = (int64_t)v; t->q = (int64_t)q; t->r = (int64_t)r;
    return eta;
}
/* (d, e) = t * (d, e) / 2^62 mod p, keeping them in (-2p, p) */
static inline void update_de_62(s62 *d, s62 *e, const trans2x2 *t) {
    const uint64_t M62 = ~0ULL >> 2;
    const int64_t d0 = d->v[0], d1 = d->v[1], d2 = d->v[2], d3 = d->v[3], d4 = d->v[4];
    const int64_t e0 = e->v[0], e1 = e->v[1], e2 = e->v[2], e3 = e->v[3], e4 = e->v[4];
    const int64_t u = t->u, v = t->v, q = t->q, r = t->r;
    int64_t md, me, sd, se; __int128 cd, ce;
    sd = d4 >> 63; se = e4 >> 63;
    md = (u & sd) + (v & se);
    me = (q & sd) + (r & se);
    cd = (__int128)u * d0 + (__int128)v * e0;
    ce = (__int128)q * d0 + (__int128)r * e0;
    md -= (int64_t)((S62_PINV * (uint64_t)cd + (uint64_t)md) & M62);
    me -= (int64_t)((S62_PINV * (uint64_t)ce + (uint64_t)me) & M62);
    cd += (__int128)S62_P.v[0] * md;
    ce += (__int128)S62_P.v[0] * me;
    cd >>= 62; ce >>= 62;                             /* the low 62 bits are zero by construction */
    cd += (__int128)u * d1 + (__int128)v * e1;
    ce += (__int128)q * d1 + (__int128)r * e1;
    d->v[0] = (int64_t)((uint64_t)(int64_t)cd & M62); cd >>= 62;
    e->v[0] = (int64_t)((uint64_t)(int64_t)ce & M62); ce >>= 62;
    cd += (__int128)u * d2 + (__int128)v * e2;
    ce += (__int128)q * d2 + (__int128)r * e2;
    d->v[1] = (int64_t)((uint64_t)(int64_t)cd & M62); cd >>= 62;
    e->v[1] = (int64_t)((uint64_t)(int64_t)ce & M62); ce >>= 62;
    cd += (__int128)u * d3 + (__int128)v * e3;
    ce += (__int128)q * d3 + (__int128)r * e3;
    d->v[2] = (int64_t)((uint64_t)(int64_t)cd & M62); cd >>= 62;
    e->v[2] = (int64_t)((uint64_t)(int64_t)ce & M62); ce >>= 62;
    cd += (__int128)u * d4 + (__int128)v * e4;
    ce += (__int128)q * d4 + (__int128)r * e4;
    cd += (__int128)S62_P.v[4] * md;                  /* p's limbs 1..3 are zero */
    ce += (__int128)S62_P.v[4] * me;
    d->v[3] = (int64_t)((uint64_t)(int64_t)cd & M62); cd >>= 62;
    e->v[3] = (int64_t)((uint64_t)(int64_t)ce & M62); ce >>= 62;
    d->v[4] = (int64_t)cd;
    e->v[4] = (int64_t)ce;
}
/* (f, g) = t * (f, g) / 2^62 over the low len limbs */
static inline void update_fg_62_var(int len, s62 *f, s62 *g, const trans2x2 *t) {
    const uint64_t M62 = ~0ULL >> 2;
    const int64_t u = t->u, v = t->v, q = t->q, r = t->r;
    int64_t fi = f->v[0], gi = g->v[0];
    __int128 cf = (__int128)u * fi + (__int128)v * gi;
    __int128 cg = (__int128)q * fi + (__int128)r * gi;
    cf >>= 62; cg >>= 62;
    for (int i = 1; i < len; ++i) {
        fi = f->v[i]; gi = g->v[i];
        cf += (__int128)u * fi + (__int128)v * gi;
        cg += (__int128)q * fi + (__int128)r * gi;
        f->v[i - 1] = (int64_t)((uint64_t)(int64_t)cf & M62); cf >>= 62;
        g->v[i - 1] = (int64_t)((uint64_t)(int64_t)cg & M62); cg >>= 62;
    }
    f->v[len - 1] = (int64_t)cf;
    g->v[len - 1] = (int64_t)cg;
}
/* r in (-2p, p) -> [0, p), negated first when sign < 0 */
static inline void normalize_62(s62 *r, int64_t sign) {
    const int64_t M62 = (int64_t)(~0ULL >> 2);
    int64_t r0 = r->v[0], r1 = r->v[1], r2 = r->v[2], r3 = r->v[3], r4 = r->v[4];
    int64_t cond_add = r4 >> 63;
    r0 += S62_P.v[0] & cond_add; r4 += S62_P.v[4] & cond_add;
    const int64_t cond_negate = sign >> 63;
    r0 = (r0 ^ cond_negate) - cond_negate; r1 = (r1 ^ cond_negate) - cond_negate; r2 = (r2 ^ cond_negate) - cond_negate;
    r3 = (r3 ^ cond_negate) - cond_negate; r4 = (r4 ^ cond_negate) - cond_negate;
    r1 += r0 >> 62; r0 &= M62; r2 += r1 >> 62; r1 &= M62; r3 += r2 >> 62; r2 &= M62; r4 += r3 >> 62; r3 &= M62;
    cond_add = r4 >> 63;
    r0 += S62_P.v[0] & cond_add; r4 += S62_P.v[4] & cond_add;
    r1 += r0 >> 62; r0 &= M62; r2 += r1 >> 62; r1 &= M62; r3 += r2 >> 62; r2 &= M62; r4 += r3 >> 62; r3 &= M62;
    r->v[0] = r0; r->v[1] = r1; r->v[2] = r2; r->v[3] = r3; r->v[4] = r4;
}
/* x = x^-1 mod p for x in [0, p) (0 -> 0); false if the divstep loop did not settle within 24 batches. */
static bool modinv_var(s62 *x) {
    s62 d = {{0, 0, 0, 0, 0}}, e = {{1, 0, 0, 0, 0}}, f = S62_P, g = *x;
    int len = 5; int64_t eta = -1;
    for (int it = 0; it < 24; it++) {
        trans2x2 t;
        eta = divsteps_62_var(eta, (uint64_t)f.v[0], (uint64_t)g.v[0], &t);
        update_de_62(&d, &e, &t);
        update_fg_62_var(len, &f, &g, &t);
        if (g.v[0] == 0) {
            int64_t cond = 0; for (int j = 1; j < len; ++j) cond |= g.v[j];
            if (cond == 0) { normalize_62(&d, f.v[len - 1]); *x = d; return true; }
        }
        const int64_t fn = f.v[len - 1], gn = g.v[len - 1];
        int64_t cond = ((int64_t)len - 2) >> 63; cond |= fn ^ (fn >> 63); cond |= gn ^ (gn >> 63);
        if (cond == 0) { f.v[len - 2] |= (int64_t)((uint64_t)fn << 62); g.v[len - 2] |= (int64_t)((uint64_t)gn << 62); --len; }
    }
    return false;
}

static void fe_inv(fe *r, const fe *a) {
    fe canon = *a; fe_normalize(&canon);
    if (fe_is_zero_norm(&canon)) { *r = canon; return; }
    uint64_t w[4]; fe_to_w(w, &canon);
    const uint64_t M = ~0ULL >> 2; s62 x;
    x.v[0] = (int64_t)(w[0] & M);
    x.v[1] = (int64_t)((w[0] >> 62 | w[1] << 2) & M);
    x.v[2] = (int64_t)((w[1] >> 60 | w[2] << 4) & M);
    x.v[3] = (int64_t)((w[2] >> 58 | w[3] << 6) & M);
    x.v[4] = (int64_t)(w[3] >> 56);
    if (!modinv_var(&x)) { fe_inv_fermat(r, &canon); return; }
    w[0] = (uint64_t)x.v[0] | (uint64_t)x.v[1] << 62;
    w[1] = (uint64_t)x.v[1] >> 2 | (uint64_t)x.v[2] << 60;
    w[2] = (uint64_t)x.v[2] >> 4 | (uint64_t)x.v[3] << 58;
    w[3] = (uint64_t)x.v[3] >> 6 | (uint64_t)x.v[4] << 56;
    fe_from_w(r, w);
}
