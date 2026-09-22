// SPDX-License-Identifier: GPL-3.0-only
// Derived from the promoted VanitySearch-based XYZZ formulas.
// Negative deferred Y convention and seeded MAC: public PR1063.
// Keep the original positive-Y checkpoint ABI by restoring its sign at exit.
#pragma once
#include "SeededMAC.cuh"

__device__ __forceinline__ void qsb_negative_point_add(
    uint64_t *X1, uint64_t *Y1, uint64_t *ZZ1, uint64_t *ZZZ1,
    const uint64_t *X2, const uint64_t *Y2, const uint64_t *Yoff)
{
  uint64_t U2[4];
  uint64_t S2[4];
  uint64_t P[4];
  uint64_t R[4];
  uint64_t PP[4];
  uint64_t PPP[4];
  uint64_t Q[4];
  uint64_t T[4];

#if QSB_YOFF
  _ModAddLazyOff(S2, Y2, Yoff);        // offset ordinates: y2 + yoff (mod p)
#elif QSB_LAZY
  _ModAddLazy(S2, Y2, Yoff);
#else
  _ModAdd256(S2, (uint64_t *)Y2, (uint64_t *)Yoff);
#endif
  qsb_muladd_seed(R, S2, ZZZ1, Y1);   // Y1 stores negative deferred ordinate
  _ModMult(U2, (uint64_t *)X2, ZZ1);   // U2 = X2*ZZ1
  _ModSub256(P, U2, X1);               // P  = U2 - X1
  _ModSqr(PP, P);                      // PP = P^2
  _ModMult(PPP, PP, P);                // PPP = P*PP
  _ModMult(Q, U2, PP);                 // V  = U2*PP

#if QSB_FUSE_SQRADDSUB2
  /* xlib f297b0f9: one reduction for R^2 + PPP - 2V. */
  _ModSqrAddSub2(T, R, PPP, Q);        // X3 = R^2 + PPP - 2V
#else
  _ModSqr(T, R);                       // R^2
#if QSB_LAZY
  _ModX3Fused(T, T, PPP, Q);           // X3 = R^2 + PPP - 2V
#else
  _ModAdd256(T, T, PPP);
  _ModSub256(T, T, Q);
  _ModSub256(T, T, Q);                 // X3 = R^2 + PPP - 2V
#endif
#endif

  _ModMult(ZZZ1, PPP);                 // ZZZ3
  _ModMult(ZZ1, PP);                   // ZZ3 (after ZZZ3: lets ptxas keep every multiply
                                       // on the paired-carry schedule without predicate spills)
  _ModSub256(Q, T, Q);                 // X3 - V, reverse of positive-Y path
  _ModMult(Y1, Q, R);                  // negative deferred ordinate

  Load256(X1, T);                      // X3
}


__device__ void qsb_negative_point_seed(uint64_t *X3, uint64_t *Y3, uint64_t *ZZ3, uint64_t *ZZZ3,
                                 const uint64_t *X1, const uint64_t *Y1,
                                 const uint64_t *X2, const uint64_t *Y2)
{
  uint64_t P[4];
  uint64_t R[4];
  uint64_t Q[4];
  uint64_t T[4];

  _ModSub256(P, (uint64_t *)X2, (uint64_t *)X1);   // P = X2 - X1
  _ModSub256(R, (uint64_t *)Y2, (uint64_t *)Y1);   // R = Y2 - Y1
  _ModSqr(ZZ3, P);                                 // ZZ3  = PP  = P^2
  _ModMult(ZZZ3, ZZ3, P);                          // ZZZ3 = PPP = P*PP
  _ModMult(Q, (uint64_t *)X1, ZZ3);                // Q = X1*PP

  _ModSqr(T, R);                                   // R^2
  _ModSub256(T, T, ZZZ3);
  _ModSub256(T, T, Q);
  _ModSub256(T, T, Q);                             // X3 = R^2 - PPP - 2Q

  _ModSub256(Q, T, Q);                             // X3 - Q
  _ModMult(Y3, Q, R);                              // negative deferred R*(X3-Q)
  Load256(X3, T);                                  // X3
}
