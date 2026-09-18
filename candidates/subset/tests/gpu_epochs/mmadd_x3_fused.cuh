#ifndef MMADD_X3_FUSED_CUH
#define MMADD_X3_FUSED_CUH

// One-chain classical X3 for the two-affine seed mixed-add.
//
// Interior / last-window madd already uses _ModX3Fused = a + b - 2c
// (V-form: X3 = R^2 + PPP - 2V). The seed path is the leftover Q-form
// X3 = R^2 - PPP - 2Q with Q = X1*PP. Q must stay Q (not rewritten as
// V) so deferred Y can remain R*(Q - X3) = R*PPP.
//
// Three sequential _ModSub256 walks each do a borrow + conditional +p.
// Fold a - b - 2c into one unsigned PTX carry chain. Bias is +4p so
// the accumulator stays non-negative: a 4-limb a - b - 2c can under-run
// by almost 3*2^256; +2p is not enough. secp256k1 p = 2^256 - K with
// K = 2^32 + 977, so 4p = 2^258 - 4K:
//   low 256 bits 0xFF..FFBFFFFF0BC, high overflow +3.
// The subsequent fold is the same t4*K reduction _ModX3Fused uses
// (non-canonical [0, 2p) is the convention every consumer already
// accepts). Included from GPUMath.h after the UADDO/USUBO macros and
// after _ModX3Fused.

__device__ __forceinline__ void _ModX3Classic(uint64_t *r,
                                              const uint64_t *a,
                                              const uint64_t *b,
                                              const uint64_t *c)
{
    uint64_t t0, t1, t2, t3, t4, d0, d1, d2, d3, d4;

    /* t = a + 4p ; 4p = 2^258 - 4K, K = 0x1000003D1 */
    UADDO(t0, a[0], 0xFFFFFFFBFFFFF0BCULL);
    UADDC(t1, a[1], 0xFFFFFFFFFFFFFFFFULL);
    UADDC(t2, a[2], 0xFFFFFFFFFFFFFFFFULL);
    UADDC(t3, a[3], 0xFFFFFFFFFFFFFFFFULL);
    UADD(t4, 3ULL, 0ULL);

    /* t -= b */
    USUBO1(t0, b[0]);
    USUBC1(t1, b[1]);
    USUBC1(t2, b[2]);
    USUBC1(t3, b[3]);
    USUB1(t4, 0ULL);

    /* t -= 2c */
    d0 = c[0] << 1;
    d1 = (c[1] << 1) | (c[0] >> 63);
    d2 = (c[2] << 1) | (c[1] >> 63);
    d3 = (c[3] << 1) | (c[2] >> 63);
    d4 = c[3] >> 63;
    USUBO1(t0, d0);
    USUBC1(t1, d1);
    USUBC1(t2, d2);
    USUBC1(t3, d3);
    USUB1(t4, d4);                          /* t4 in {0,1,2,3,4} */

    t4 *= 0x1000003D1ULL;
    UADDO1(t0, t4);
    UADDC1(t1, 0ULL);
    UADDC1(t2, 0ULL);
    UADD1(t3, 0ULL);
    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
}

#endif
