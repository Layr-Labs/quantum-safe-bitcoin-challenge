/* qsb_host_verify.h — exact host publication gate for the subset grinder (QSB_HOST_VERIFY=1).
 * Replaces kernel_verify_pair_hits: each GPU-nominated tentative hit is rebuilt from the
 * problem and its (epoch rank, lane) identity, hashed with OpenSSL SHA-256d from the committed
 * midstate, recovered with OpenSSL EC arithmetic, and published only if it passes exactly the
 * harness/verify.py predicate. The exact-recovery kernel (21,728 sm_89 instructions, ~35% of
 * the fatbin's JIT work, compiled by the driver inside the ranked window) leaves the binary.
 * Every published hit is still re-derived by the harness on the CPU. */
#ifndef QSB_HOST_VERIFY_H
#define QSB_HOST_VERIFY_H
#include <errno.h>
#include <unistd.h>
#include <openssl/sha.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>

typedef struct {
    EC_GROUP *grp; BN_CTX *ctx; BIGNUM *order; BIGNUM *nri; EC_POINT *Ru2;
    const digest_params_t *dp;
    uint8_t win3[QSB_SE_PER_EPOCH][QSB_SE_TWIN];
    int window_start, s_early;
} qsb_hv_t;

static int qsb_hv_init(qsb_hv_t *h, const digest_params_t *dp, const uint8_t win3[QSB_SE_PER_EPOCH][QSB_SE_TWIN],
                       int window_start, int s_early) {
    memset(h, 0, sizeof(*h));
    h->dp = dp; h->window_start = window_start; h->s_early = s_early;
    memcpy(h->win3, win3, sizeof(h->win3));
    h->grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    h->ctx = BN_CTX_new(); h->order = BN_new(); h->nri = BN_new();
    if (!h->grp || !h->ctx || !h->order || !h->nri) return 0;
    EC_GROUP_get_order(h->grp, h->order, h->ctx);
    BN_lebin2bn(dp->neg_r_inv, 32, h->nri);                    /* LE, as d_nri (PROBLEM.md .bin layout) */
    BIGNUM *x = BN_lebin2bn(dp->u2r_x, 32, NULL), *y = BN_lebin2bn(dp->u2r_y, 32, NULL);
    h->Ru2 = EC_POINT_new(h->grp);
    int ok = x && y && h->Ru2 && EC_POINT_set_affine_coordinates(h->grp, h->Ru2, x, y, h->ctx);
    BN_free(x); BN_free(y);
    return ok;
}

static int qsb_hv_zeros(const uint8_t *hh) {
    int n = 0;
    for (int i = 0; i < 32; i++) { if (hh[i] == 0) { n += 8; continue; } uint8_t b = hh[i]; while (!(b & 0x80)) { n++; b <<= 1; } break; }
    return n;
}

/* verify.py predicate: z = SHA256d(fixed_prefix || kept dummy sigs in storage order || tail || suffix),
 * Q = u1*G + (recid ? -R : R) with u1 = z*neg_r_inv mod n; hit iff lz(SHA256(compress(Q))) >= N. */
static int qsb_hv_check(const qsb_hv_t *h, const uint8_t skip[9], int recid) {
    const digest_params_t *dp = h->dp;
    uint8_t buf[4096]; size_t len = 0;
    if (dp->prefix_remainder_len) { memcpy(buf + len, dp->prefix_remainder, dp->prefix_remainder_len); len += dp->prefix_remainder_len; }
    for (uint32_t i = 0; i < dp->n; i++) {
        int skipped = 0;
        for (int j = 0; j < 9; j++) if (skip[j] == i) { skipped = 1; break; }
        if (!skipped) { memcpy(buf + len, dp->dummy_sigs + (size_t)i * SIG_PUSH_SIZE, SIG_PUSH_SIZE); len += SIG_PUSH_SIZE; }
    }
    memcpy(buf + len, dp->tail_section, dp->tail_section_len); len += dp->tail_section_len;
    memcpy(buf + len, dp->tx_suffix, dp->tx_suffix_len);       len += dp->tx_suffix_len;
    if (len + 72 > sizeof(buf)) return 0;
    if (((size_t)dp->total_preimage_len - len) % 64) return 0;  /* midstate covers whole blocks */
    uint64_t bits = (uint64_t)dp->total_preimage_len * 8;
    buf[len++] = 0x80;
    while (len % 64 != 56) buf[len++] = 0;
    for (int i = 0; i < 8; i++) buf[len++] = (uint8_t)(bits >> (56 - 8 * i));
    SHA256_CTX sc; SHA256_Init(&sc);
    for (int i = 0; i < 8; i++) sc.h[i] = dp->midstate[i];
    for (size_t off = 0; off < len; off += 64) SHA256_Transform(&sc, buf + off);
    uint8_t d1[32], d2[32];
    for (int i = 0; i < 8; i++) { d1[4*i] = (uint8_t)(sc.h[i] >> 24); d1[4*i+1] = (uint8_t)(sc.h[i] >> 16); d1[4*i+2] = (uint8_t)(sc.h[i] >> 8); d1[4*i+3] = (uint8_t)sc.h[i]; }
    SHA256(d1, 32, d2);
    BIGNUM *z = BN_bin2bn(d2, 32, NULL), *u1 = BN_new(), *qx = BN_new(), *qy = BN_new();
    EC_POINT *P = EC_POINT_new(h->grp), *Q = EC_POINT_new(h->grp), *R = EC_POINT_dup(h->Ru2, h->grp);
    int ok = 0;
    if (z && u1 && qx && qy && P && Q && R &&
        BN_mod_mul(u1, z, h->nri, h->order, h->ctx) && EC_POINT_mul(h->grp, P, u1, NULL, NULL, h->ctx)) {
        if (recid) EC_POINT_invert(h->grp, R, h->ctx);
        if (EC_POINT_add(h->grp, Q, P, R, h->ctx) && !EC_POINT_is_at_infinity(h->grp, Q) &&
            EC_POINT_get_affine_coordinates(h->grp, Q, qx, qy, h->ctx)) {
            uint8_t pub[33]; memset(pub, 0, sizeof pub);
            int nb = BN_num_bytes(qx);
            if (nb > 0 && nb <= 32) BN_bn2bin(qx, pub + 1 + (32 - nb));
            pub[0] = (uint8_t)(0x02 + (BN_is_odd(qy) ? 1 : 0));
            uint8_t hh[32]; SHA256(pub, 33, hh);
            ok = qsb_hv_zeros(hh) >= QSB_ZEROS_N;
        }
    }
    BN_free(z); BN_free(u1); BN_free(qx); BN_free(qy);
    EC_POINT_free(P); EC_POINT_free(Q); EC_POINT_free(R);
    return ok;
}

/* Rebuild the candidate from (epoch rank, lane) — never from the tentative combo bytes — check the
 * GPU's recid first and the other recid second, publish on success. Returns 1 published, 0 dropped, -1 io. */
static int qsb_hv_publish(const qsb_hv_t *h, uint64_t epoch_rank, unsigned lane, int recid_gpu, int fd, uint64_t *hit_counter) {
    uint8_t skip[9];
    qsb_host_unrank(epoch_rank, h->window_start, h->s_early, skip);
    for (int j = 0; j < 3; j++) skip[6 + j] = h->win3[lane & (QSB_SE_PER_EPOCH - 1)][j];
    int recid = recid_gpu & 1;
    if (!qsb_hv_check(h, skip, recid)) { recid ^= 1; if (!qsb_hv_check(h, skip, recid)) return 0; }
    char line[96];
    int wl = snprintf(line, sizeof line, "indices=%d,%d,%d,%d,%d,%d,%d,%d,%d recid=%d\n",
                      skip[0], skip[1], skip[2], skip[3], skip[4], skip[5], skip[6], skip[7], skip[8], recid);
    const char *wp = line;
    while (wl > 0) { ssize_t k = write(fd, wp, (size_t)wl); if (k < 0) { if (errno == EINTR) continue; return -1; } wp += k; wl -= (int)k; }
    (*hit_counter)++;
    return 1;
}
#endif /* QSB_HOST_VERIFY_H */
