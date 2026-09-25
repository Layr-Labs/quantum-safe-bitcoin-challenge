/* Exact two-recovery publication gate. Host code only.
 * SPDX-License-Identifier: GPL-3.0-only
 * Based on the existing Yukon candidate OpenSSL gate; no predicate changes.
 */
#ifndef QSB_EXACT_RECOVERY_PAIR_H
#define QSB_EXACT_RECOVERY_PAIR_H
#include <stdint.h>
#include <string.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/sha.h>

static int qsb_exact_pair(const uint8_t digest[32], int preferred, int zeros,
                         const EC_GROUP *grp, BN_CTX *ctx, const BIGNUM *order,
                         const BIGNUM *nri, const EC_POINT *Ru2) {
    if (preferred != 0 && preferred != 1) return -1;
    int accepted = -1;
    BN_CTX_start(ctx);
    BIGNUM *z = BN_CTX_get(ctx), *u = BN_CTX_get(ctx);
    BIGNUM *x = BN_CTX_get(ctx), *y = BN_CTX_get(ctx);
    EC_POINT *P = EC_POINT_new(grp), *Q = EC_POINT_new(grp);
    EC_POINT *R = EC_POINT_new(grp);
    if (y && P && Q && R && BN_bin2bn(digest, 32, z) &&
        BN_mod_mul(u, z, nri, order, ctx) &&
        EC_POINT_mul(grp, P, u, NULL, NULL, ctx)) {
        for (int attempt = 0; attempt < 2; ++attempt) {
            int ri = preferred ^ attempt;
            if (!EC_POINT_copy(R, Ru2)) break;
            if (ri && !EC_POINT_invert(grp, R, ctx)) break;
            if (!EC_POINT_add(grp, Q, P, R, ctx)) break;
            if (EC_POINT_is_at_infinity(grp, Q)) continue;
            if (!EC_POINT_get_affine_coordinates(grp, Q, x, y, ctx)) break;
            uint8_t pub[33] = {0}, hash[32];
            int nb = BN_num_bytes(x);
            if (nb > 32) break;
            if (nb) BN_bn2bin(x, pub + 33 - nb);
            pub[0] = (uint8_t)(2 + BN_is_odd(y));
            SHA256(pub, sizeof(pub), hash);
            int n = 0;
            for (int i = 0; i < 32; ++i) {
                if (!hash[i]) { n += 8; continue; }
                unsigned v = hash[i];
                while (!(v & 0x80u)) { ++n; v <<= 1; }
                break;
            }
            if (n >= zeros) { accepted = ri; break; }
        }
    }
    EC_POINT_free(P); EC_POINT_free(Q); EC_POINT_free(R);
    BN_CTX_end(ctx);
    return accepted;
}
#endif
