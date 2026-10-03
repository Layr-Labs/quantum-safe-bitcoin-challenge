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

/* QSB_HV_JOINT (host only; kill switch): the publication gate forms Q = u1*G + (+-R) with one
 * two-term EC_POINT_mul. 0 = separate u1*G and EC_POINT_add. */
#ifndef QSB_HV_JOINT
#define QSB_HV_JOINT 1
#endif
/* QSB_CPU_FENCE (host only, not an image knob): the co-grinder
 * grinds the GPU's own 128 window patterns on the epochs [F, C(137,6)) above a static, batch-aligned fence F, and the GPU
 * walks [0, F) and idles at F until the stop signal (tree.cu, CpuGrindSubset.h). It is defined here, the first header both
 * publishers include, because this file holds the one process-wide set of published hits both publish paths share: a
 * candidate (sorted skip set, recid; verify.py's canonical key) is written at most once per process, and a second
 * publication is dropped and counted (qsb_pub_dups). Disjoint epoch ranges make that count 0 by construction; the set
 * turns an enumeration bug into lost hits instead of a rejected run (one duplicate voids a run). 0 = the base. */
#ifndef QSB_CPU_FENCE
#define QSB_CPU_FENCE 0
#endif
#if QSB_CPU_FENCE
#include <mutex>
#include <unordered_set>
static std::mutex qsb_pub_m;
struct QsbPubKeyHash { size_t operator()(uint64_t k) const { k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; return (size_t)k; } };
static std::unordered_set<uint64_t, QsbPubKeyHash> *qsb_pub_set = nullptr;   /* leaked on purpose: the process _exits */
static uint64_t qsb_pub_n = 0, qsb_pub_dups = 0;                            /* keys taken; publications dropped */
/* The key: the sorted skip set's lexicographic rank among the 9-subsets of {0..149} (C(150, 9) < 2^47), times 2, plus the
 * recid. Exact and collision-free in 64 bits. qsb_pub_once: true (and the key recorded) when this candidate was not
 * published before in this process; false (and qsb_pub_dups counted) for a second publication. */
static uint64_t qsb_pub_key(const uint8_t skip[9], int recid) {
    uint8_t s[9]; memcpy(s, skip, 9);
    for (int i = 1; i < 9; i++) { const uint8_t v = s[i]; int j = i; while (j > 0 && s[j - 1] > v) { s[j] = s[j - 1]; j--; } s[j] = v; }
    static uint64_t C[151][10];                       /* C[n][k], n <= 150, k <= 9 (< 2^47): filled on first use, under qsb_pub_m */
    if (!C[0][0]) for (int n = 0; n <= 150; n++) for (int k = 0; k <= 9; k++) C[n][k] = k == 0 ? 1 : n == 0 ? 0 : C[n - 1][k - 1] + C[n - 1][k];
    uint64_t r = 0; int prev = -1;
    for (int i = 0; i < 9; i++) { for (int j = prev + 1; j < s[i]; j++) r += C[150 - j - 1][9 - i - 1]; prev = s[i]; }
    return r << 1 | (uint64_t)(recid & 1);
}
static bool qsb_pub_once(const uint8_t skip[9], int recid) {
    std::lock_guard<std::mutex> g(qsb_pub_m);
    const uint64_t k = qsb_pub_key(skip, recid);
    if (!qsb_pub_set) { qsb_pub_set = new std::unordered_set<uint64_t, QsbPubKeyHash>(); qsb_pub_set->reserve(1u << 18); }
    if (!qsb_pub_set->insert(k).second) { qsb_pub_dups++; return false; }
    qsb_pub_n++;
    return true;
}
#endif
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

/* Fixed-width coordinate encoding; macro0/older libraries retain the original. */
#ifndef QSB_HV_PUBKEY_PAD
#define QSB_HV_PUBKEY_PAD 1
#endif
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
    if (z && u1 && qx && qy && P && Q && R && BN_mod_mul(u1, z, h->nri, h->order, h->ctx)) {
        if (recid) EC_POINT_invert(h->grp, R, h->ctx);
#if QSB_HV_JOINT
        /* Q = u1*G + 1*(+-R) in one interleaved wNAF pass instead of a constant-time ladder for u1*G
         * followed by an add: the same group element, about half the host field work per check. */
        const int qok = EC_POINT_mul(h->grp, Q, u1, R, BN_value_one(), h->ctx);
#else
        const int qok = EC_POINT_mul(h->grp, P, u1, NULL, NULL, h->ctx) && EC_POINT_add(h->grp, Q, P, R, h->ctx);
#endif
        if (qok && !EC_POINT_is_at_infinity(h->grp, Q) &&
            EC_POINT_get_affine_coordinates(h->grp, Q, qx, qy, h->ctx)) {
            uint8_t pub[33];
#if QSB_HV_PUBKEY_PAD && OPENSSL_VERSION_NUMBER >= 0x10101000L && !defined(LIBRESSL_VERSION_NUMBER)
            if (BN_bn2binpad(qx, pub + 1, 32) != 32) {
#endif
            memset(pub, 0, sizeof pub);
            int nb = BN_num_bytes(qx);
            if (nb > 0 && nb <= 32) BN_bn2bin(qx, pub + 1 + (32 - nb));
#if QSB_HV_PUBKEY_PAD && OPENSSL_VERSION_NUMBER >= 0x10101000L && !defined(LIBRESSL_VERSION_NUMBER)
            }
#endif
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
#if QSB_CPU_FENCE
    if (!qsb_pub_once(skip, recid)) return 0;              /* published before in this process: dropped, counted */
#endif
    char line[96];
    int wl = snprintf(line, sizeof line, "indices=%d,%d,%d,%d,%d,%d,%d,%d,%d recid=%d\n",
                      skip[0], skip[1], skip[2], skip[3], skip[4], skip[5], skip[6], skip[7], skip[8], recid);
    const char *wp = line;
    while (wl > 0) { ssize_t k = write(fd, wp, (size_t)wl); if (k < 0) { if (errno == EINTR) continue; return -1; } wp += k; wl -= (int)k; }
    (*hit_counter)++;
    return 1;
}
#if QSB_HIT_TELEMETRY
/* QSB_HIT_TELEMETRY (hit_telemetry.h): qsb_hv_publish with the write deferred. The same gate and the same first-publication
 * check; the line and its canonical key (nine indices, recid) go to *out, and the caller writes the batch's lines in its
 * chosen order. 1: published (counted), 0: not a hit or published before. */
static int qsb_hv_publish_line(const qsb_hv_t *h, uint64_t epoch_rank, unsigned lane, int recid_gpu, qtel::Line *out, uint64_t *hit_counter) {
    uint8_t skip[9];
    qsb_host_unrank(epoch_rank, h->window_start, h->s_early, skip);
    for (int j = 0; j < 3; j++) skip[6 + j] = h->win3[lane & (QSB_SE_PER_EPOCH - 1)][j];
    int recid = recid_gpu & 1;
    if (!qsb_hv_check(h, skip, recid)) { recid ^= 1; if (!qsb_hv_check(h, skip, recid)) return 0; }
#if QSB_CPU_FENCE
    if (!qsb_pub_once(skip, recid)) return 0;
#endif
    out->len = snprintf(out->text, sizeof out->text, "indices=%d,%d,%d,%d,%d,%d,%d,%d,%d recid=%d\n",
                        skip[0], skip[1], skip[2], skip[3], skip[4], skip[5], skip[6], skip[7], skip[8], recid);
    memcpy(out->key, skip, 9); out->key[9] = (uint8_t)recid;
    (*hit_counter)++;
    return 1;
}
#endif
#endif /* QSB_HOST_VERIFY_H */
