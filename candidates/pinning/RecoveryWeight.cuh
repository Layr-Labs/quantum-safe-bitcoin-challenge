// SPDX-License-Identifier: GPL-3.0-only
// Weighted cofactor leaves and square-free finish sum.
//
// The public 128-leaf traversal already publishes I=1/T and J=b/T, so each
// finish lane forms u=tbar*J without a per-leaf multiply by the fixed
// ordinate b. Two remaining costs still sit on that weighted product:
//
// 1. The denominator W=V*(a*U-X) was canonicalized before the tree. Tree
//    products only need a congruent representative in [0,2^256); the exact
//    multiplier is carry-complete, so a zero leaf stays the all-zero word
//    and the existing zero-test / identity-fill is unchanged. Canonicalize
//    at the inverse and at the u,v add/sub boundary, not at every leaf.
//
// 2. The finish identity S=l+m is 2u. Forming S from l and m waits on both
//    reductions. Doubling the already-weighted u is the same field value
//    and is independent of l=u-v and m=u+v.
#pragma once

__device__ __forceinline__ void qsb_weighted_leaf(
    uint64_t *W, uint64_t *V, uint64_t *d) {
    uint64_t raw[5];
    qsb_field_mul(raw, V, d);
    Load256(W, raw);
    W[4] = 0;
}

__device__ __forceinline__ void qsb_weighted_sum(uint64_t *sum, const uint64_t *u) {
    _ModAdd256(sum, const_cast<uint64_t *>(u), const_cast<uint64_t *>(u));
}
