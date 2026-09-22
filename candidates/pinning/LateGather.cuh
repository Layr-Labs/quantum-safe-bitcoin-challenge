// Rolled, late gather placement derived from the promoted point-add body.
// Public CHAIN_PIPE descriptions by ItlaStudent/terrapinelf motivated the
// single-body and register-budget comparison; this placement is independent.
#pragma once
__device__ __forceinline__ void qsb_point_add_late_gather(
    uint64_t *X1, uint64_t *Y1, uint64_t *ZZ1, uint64_t *ZZZ1,
    uint64_t *X2, uint64_t *Y2, uint64_t *Yoff,
    bool prefetch, const uint8_t *table, unsigned next_chunk, unsigned next_base)
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
  _ModMult(S2, ZZZ1);                  // S2 = (Y2+Yoff)*ZZZ1
  _ModSub256(R, S2, Y1);               // R  = S2 - Y1
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
  // Current X2 and the old anchor are dead; PP and PPP have also died.
  // Preserve the current ordinate as the next anchor before overwriting it.
  Load256(Yoff,Y2);
  if(prefetch) qsb_load_decoded(table,next_chunk,next_base,X2,Y2);
  // These operations depend only on R/Q/T, not on the incoming table point.
  _ModSub256(Q, Q, T);                 // V - X3
  _ModMult(Q, R);                      // R*(V - X3)
  Load256(Y1, Q);                    // deferred ordinate, same as <true>

  Load256(X1, T);                      // X3
}
