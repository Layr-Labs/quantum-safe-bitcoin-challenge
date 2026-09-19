// SPDX-License-Identifier: GPL-3.0-only
// Raw weighted cofactor leaf plus independent 2u finish sum.
// Uses the VanitySearch-derived field primitives in GPUMath.h.
#pragma once

// W = V*(a*U-X) as an exact [0, 2^256) residue. Internal tree products
// already keep qsb_field_mul's last reduction carry; normalizing only
// this leaf was leftover from the older canonical-boundary path. The
// next cofactor multiply and the pair inverse accept any 256-bit
// representative congruent mod p.
__device__ __forceinline__ void qsb_weighted_leaf(
    uint64_t *W, const uint64_t *V, const uint64_t *d) {
    uint64_t tmp[5];
    qsb_field_mul(tmp, const_cast<uint64_t *>(V), const_cast<uint64_t *>(d));
    Load256(W, tmp);
}

// After the group-root publishes J = b/T, a finish lane has
// u = tbar*J and v = vbar*I. Then l = u-v, m = u+v, so
// S = l+m is identically 2u. Doubling u does not wait on either
// difference. _ModAdd256 reduces the single carry, matching the
// previous (l+m) representative modulo p.
__device__ __forceinline__ void qsb_weighted_sum(
    uint64_t *sum, const uint64_t *u) {
    _ModAdd256(sum, u, u);
}
