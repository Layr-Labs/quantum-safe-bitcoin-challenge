/* hit_recheck.cuh — host-only exact re-derivation of every tentative hit.
 *
 * WHY THIS EXISTS.  The filter path uses bounded truncations in the field
 * arithmetic (rp drops the reduction's g8/z9; QSB_TRUNC_FOLD additionally drops
 * f8 and sfc).  A truncation that fires produces a wrong pubkey x, whose hash
 * passes the leading-zero gate with probability 2^-24 — and `harness/verify.py`
 * rejects the WHOLE 1200 s run on a single unverifiable hit.  Unlike the subset
 * kernel, pinning has no exact GPU re-check, so the tolerable corruption rate
 * would otherwise be ~1e-12 per candidate.  This re-derives each tentative hit
 * exactly with OpenSSL, on spare cores, concurrently with the GPU, and emits
 * only hits that verify.
 *
 * SAFETY PROPERTY: this can only ever REMOVE a hit, never invent one.  A bug in
 * it costs score (fewer hits) and shows up immediately in validation as a
 * hit-count drop against the base tree; it cannot make a run invalid.
 *
 * COST: ~256 us per hit on one core (measured), against ~92 hits/s produced, so
 * a single worker has a ~40x margin.  The GPU-driving thread only enqueues.
 * Machine has 24 cores and is otherwise idle during a run.
 */
#ifndef QSB_HIT_RECHECK_CUH
#define QSB_HIT_RECHECK_CUH

#include <pthread.h>
#include <openssl/sha.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>

#ifndef QSB_RC_THREADS
#define QSB_RC_THREADS 2
#endif
#define QSB_RC_CAP 8192            /* ring capacity; steady-state depth is ~0 */

typedef struct { uint32_t seq, lt; int ri; } qsb_rc_item;

static struct {
    qsb_rc_item ring[QSB_RC_CAP];
    volatile unsigned head, tail;      /* head = next to pop, tail = next to push */
    pthread_mutex_t m, fm;
    pthread_cond_t cv;
    volatile int stop;
    FILE *out;
    const pinning2_params_t *p;
    pthread_t th[QSB_RC_THREADS];
    volatile long n_in, n_ok, n_drop;
    int started;
} qsb_rc = {};

static volatile int qsb_rc_found = 0;

/* --- exact re-derivation of one candidate ------------------------------- */
typedef struct {
    EC_GROUP *grp; BN_CTX *ctx; BIGNUM *order, *nri, *e, *u1, *qx, *qy;
    EC_POINT *u2r, *u2rn, *Q, *T;
} qsb_rc_ctx;

static int qsb_rc_ctx_init(qsb_rc_ctx *c, const pinning2_params_t *p) {
    memset(c, 0, sizeof *c);
    c->grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    c->ctx = BN_CTX_new();
    if (!c->grp || !c->ctx) return 0;
    c->order = BN_new(); c->e = BN_new(); c->u1 = BN_new();
    c->qx = BN_new(); c->qy = BN_new();
    if (!EC_GROUP_get_order(c->grp, c->order, c->ctx)) return 0;
    uint8_t be[32];
    for (int i = 0; i < 32; i++) be[i] = p->neg_r_inv[31 - i];
    c->nri = BN_bin2bn(be, 32, NULL);
    BIGNUM *ux = BN_new(), *uy = BN_new();
    for (int i = 0; i < 32; i++) be[i] = p->u2r_x[31 - i]; BN_bin2bn(be, 32, ux);
    for (int i = 0; i < 32; i++) be[i] = p->u2r_y[31 - i]; BN_bin2bn(be, 32, uy);
    c->u2r = EC_POINT_new(c->grp); c->u2rn = EC_POINT_new(c->grp);
    c->Q = EC_POINT_new(c->grp);   c->T = EC_POINT_new(c->grp);
    int ok = EC_POINT_set_affine_coordinates(c->grp, c->u2r, ux, uy, c->ctx)
          && EC_POINT_copy(c->u2rn, c->u2r) && EC_POINT_invert(c->grp, c->u2rn, c->ctx);
    BN_free(ux); BN_free(uy);
    return ok;
}

/* e = SHA256d(preimage with seq/lt patched); mirrors the kernel's tail block. */
static void qsb_rc_sighash(const pinning2_params_t *p, uint32_t seq, uint32_t lt,
                           uint8_t out[32]) {
    uint8_t buf[192]; memset(buf, 0, sizeof buf);
    uint32_t n = p->suffix_len;
    memcpy(buf, p->suffix, n);
    buf[p->seq_offset] = seq & 0xFF;       buf[p->seq_offset+1] = (seq >> 8) & 0xFF;
    buf[p->seq_offset+2] = (seq >> 16) & 0xFF; buf[p->seq_offset+3] = (seq >> 24) & 0xFF;
    buf[p->lt_offset] = lt & 0xFF;         buf[p->lt_offset+1] = (lt >> 8) & 0xFF;
    buf[p->lt_offset+2] = (lt >> 16) & 0xFF;   buf[p->lt_offset+3] = (lt >> 24) & 0xFF;
    buf[n] = 0x80;
    size_t total = ((size_t)(n + 1 + 8 + 63) / 64) * 64;
    uint64_t bitlen = (uint64_t)p->total_preimage_len * 8;
    for (int i = 0; i < 8; i++) buf[total - 1 - i] = (uint8_t)(bitlen >> (8 * i));
    SHA256_CTX sc; memset(&sc, 0, sizeof sc);
    for (int i = 0; i < 8; i++) sc.h[i] = p->midstate[i];
    for (size_t i = 0; i < total / 64; i++) SHA256_Transform(&sc, buf + 64 * i);
    uint8_t d1[32];
    for (int i = 0; i < 8; i++) {
        d1[4*i] = sc.h[i] >> 24; d1[4*i+1] = sc.h[i] >> 16;
        d1[4*i+2] = sc.h[i] >> 8; d1[4*i+3] = (uint8_t)sc.h[i];
    }
    SHA256(d1, 32, out);
}

static int qsb_rc_verify(qsb_rc_ctx *c, const pinning2_params_t *p,
                         uint32_t seq, uint32_t lt, int ri) {
    uint8_t eh[32]; qsb_rc_sighash(p, seq, lt, eh);
    if (!BN_bin2bn(eh, 32, c->e)) return 0;
    if (!BN_mod_mul(c->u1, c->e, c->nri, c->order, c->ctx)) return 0;   /* u1 = e*(-r^-1) */
    if (!EC_POINT_mul(c->grp, c->T, c->u1, NULL, NULL, c->ctx)) return 0;
    if (!EC_POINT_add(c->grp, c->Q, c->T, ri ? c->u2rn : c->u2r, c->ctx)) return 0;
    if (!EC_POINT_get_affine_coordinates(c->grp, c->Q, c->qx, c->qy, c->ctx)) return 0;
    uint8_t pub[33];
    pub[0] = BN_is_odd(c->qy) ? 0x03 : 0x02;
    if (BN_bn2binpad(c->qx, pub + 1, 32) != 32) return 0;
    uint8_t h[32]; SHA256(pub, 33, h);
    for (int i = 0; i < QSB_ZEROS_N / 32; i++)
        if (h[4*i] | h[4*i+1] | h[4*i+2] | h[4*i+3]) return 0;
#if (QSB_ZEROS_N % 32) != 0
    {   int w = QSB_ZEROS_N / 32;
        uint32_t v = ((uint32_t)h[4*w] << 24) | ((uint32_t)h[4*w+1] << 16)
                   | ((uint32_t)h[4*w+2] << 8) | h[4*w+3];
        if ((v >> (32 - (QSB_ZEROS_N % 32))) != 0) return 0; }
#endif
    return 1;
}

static void *qsb_rc_worker(void *arg) {
    (void)arg;
    qsb_rc_ctx c;
    if (!qsb_rc_ctx_init(&c, qsb_rc.p)) {
        fprintf(stderr, "ERROR: hit re-check context init failed\n");
        abort();                     /* loud, not silent: never emit unverified hits */
    }
    for (;;) {
        qsb_rc_item it;
        pthread_mutex_lock(&qsb_rc.m);
        while (qsb_rc.head == qsb_rc.tail && !qsb_rc.stop)
            pthread_cond_wait(&qsb_rc.cv, &qsb_rc.m);
        if (qsb_rc.head == qsb_rc.tail && qsb_rc.stop) { pthread_mutex_unlock(&qsb_rc.m); break; }
        it = qsb_rc.ring[qsb_rc.head % QSB_RC_CAP];
        qsb_rc.head++;
        pthread_mutex_unlock(&qsb_rc.m);

        int ok = qsb_rc_verify(&c, qsb_rc.p, it.seq, it.lt, it.ri);
        pthread_mutex_lock(&qsb_rc.fm);
        if (ok) {
            qsb_rc.n_ok++;
            /* one line per hit — the format gpu_wrap.py parses; unchanged */
            fprintf(qsb_rc.out, "sequence=%u locktime=%u recid=%d\n", it.seq, it.lt, it.ri);
            fflush(qsb_rc.out);      /* the run is killed by `timeout`: never buffer */
            qsb_rc_found = 1;
        } else {
            qsb_rc.n_drop++;
        }
        pthread_mutex_unlock(&qsb_rc.fm);
    }
    return NULL;
}

static int qsb_rc_start(const pinning2_params_t *p, const char *fname) {
    qsb_rc.p = p; qsb_rc.head = qsb_rc.tail = 0; qsb_rc.stop = 0;
    pthread_mutex_init(&qsb_rc.m, NULL);
    pthread_mutex_init(&qsb_rc.fm, NULL);
    pthread_cond_init(&qsb_rc.cv, NULL);
    qsb_rc.out = fopen(fname, "a");
    if (!qsb_rc.out) { fprintf(stderr, "ERROR: cannot open %s for hits\n", fname); return 0; }
    for (int i = 0; i < QSB_RC_THREADS; i++)
        if (pthread_create(&qsb_rc.th[i], NULL, qsb_rc_worker, NULL) != 0) {
            fprintf(stderr, "ERROR: cannot start hit re-check thread\n"); return 0; }
    qsb_rc.started = 1;
    return 1;
}

static void qsb_rc_push(uint32_t seq, uint32_t lt, int ri) {
    pthread_mutex_lock(&qsb_rc.m);
    if (qsb_rc.tail - qsb_rc.head < QSB_RC_CAP) {
        qsb_rc_item it; it.seq = seq; it.lt = lt; it.ri = ri;
        qsb_rc.ring[qsb_rc.tail % QSB_RC_CAP] = it;
        qsb_rc.tail++; qsb_rc.n_in++;
        pthread_cond_signal(&qsb_rc.cv);
    }   /* full ring: drop rather than stall the GPU thread (never happens at 92 hits/s) */
    pthread_mutex_unlock(&qsb_rc.m);
}

static void qsb_rc_stop(void) {
    if (!qsb_rc.started) return;
    pthread_mutex_lock(&qsb_rc.m);
    qsb_rc.stop = 1; pthread_cond_broadcast(&qsb_rc.cv);
    pthread_mutex_unlock(&qsb_rc.m);
    for (int i = 0; i < QSB_RC_THREADS; i++) pthread_join(qsb_rc.th[i], NULL);
    if (qsb_rc.out) fclose(qsb_rc.out);
    fprintf(stderr, "  hit re-check: %ld queued, %ld verified, %ld dropped\n",
            qsb_rc.n_in, qsb_rc.n_ok, qsb_rc.n_drop);
}
#endif
