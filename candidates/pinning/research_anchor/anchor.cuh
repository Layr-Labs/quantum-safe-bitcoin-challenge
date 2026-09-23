// SPDX-License-Identifier: GPL-3.0-only
// Experimental second affine anchor; not part of the production include closure.
// D = X - xa*U; N = -Y - ya*V. Inputs ya and yb use inherited YOFF encoding.
// After an ordinary step D' = R^2 + P^3 - 3*xb*U' and N' = R*D'.
#pragma once
__device__ __forceinline__ void qsb_anchor_add(
    uint64_t *D, uint64_t *N, uint64_t *U, uint64_t *V,
    const uint64_t *xb, const uint64_t *yb,
    const uint64_t *xa, const uint64_t *ya, bool last) {
    uint64_t dx[4], sy[4], P[4], R[4], PP[4], PPP[4], Q[4], T[4];
    _ModAddLazyOff(sy,yb,ya);
    qsb_muladd_seed(R,sy,V,N);
    _ModSub256(dx,(uint64_t*)xb,(uint64_t*)xa);
    _ModMult(P,dx,U);
    _ModSub256(P,P,D);
    _ModSqr(PP,P);
    _ModMult(PPP,PP,P);
    _ModMult(U,PP);
    _ModMult(Q,(uint64_t*)xb,U);
    _ModSqrAddSub3(T,R,PPP,Q);
    _ModMult(V,PPP);
    _ModMult(N,R,T);
    if(last) _ModAddLazy(T,T,Q);
    Load256(D,T);
}
