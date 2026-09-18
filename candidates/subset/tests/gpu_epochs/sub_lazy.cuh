#ifndef SUB_LAZY_CUH
#define SUB_LAZY_CUH

// Predicate-free field subtraction for the ranked XYZZ madd.
//
// HEAD already has _ModAddLazy (carry folds as K, no conditional -p) and
// _ModX3Fused (one chain for the V-form X3). The remaining _ModSub256
// walks on _PointAddXYZZ_def / _def_last still do a borrow + four-limb
// predicate mask + conditional +p. Those subtracts feed _ModSqr /
// _ModMult / the deferred-Y product, which already accept a
// non-canonical [0, 2^256) representative.
//
// r = a - b (mod p) as t = a + 2p - b, then the same t4*K fold
// _ModX3Fused uses. +2p is enough for 4-limb a, b in [0, 2^256):
//   min = 0 + 2p - (2^256-1) = 2^256 - 2K + 1 > 0
//   2p  = 2^257 - 2K = (t4 = 1) || 0xFFFFFFFDFFFFF85E
// The in-flight seed X3 fold (a - b - 2c) is a different identity and
// is not touched here. Included from GPUMath.h after _ModX3Fused.

__device__ __forceinline__ void _ModSubLazy(uint64_t *r,
                                            const uint64_t *a,
                                            const uint64_t *b)
{
    uint64_t t0, t1, t2, t3, t4;

    /* t = a + 2p ; 2p = 2^257 - 2K, K = 0x1000003D1 */
    UADDO(t0, a[0], 0xFFFFFFFDFFFFF85EULL);
    UADDC(t1, a[1], 0xFFFFFFFFFFFFFFFFULL);
    UADDC(t2, a[2], 0xFFFFFFFFFFFFFFFFULL);
    UADDC(t3, a[3], 0xFFFFFFFFFFFFFFFFULL);
    UADD(t4, 1ULL, 0ULL);

    /* t -= b */
    USUBO1(t0, b[0]);
    USUBC1(t1, b[1]);
    USUBC1(t2, b[2]);
    USUBC1(t3, b[3]);
    USUB1(t4, 0ULL);                            /* t4 in {0,1,2} */

    t4 *= 0x1000003D1ULL;
    UADDO1(t0, t4);
    UADDC1(t1, 0ULL);
    UADDC1(t2, 0ULL);
    UADD1(t3, 0ULL);
    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
}

#endif
