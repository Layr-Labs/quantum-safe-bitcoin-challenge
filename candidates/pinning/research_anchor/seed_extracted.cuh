__device__ void qsb_anchor_seed(uint64_t *X3, uint64_t *Y3, uint64_t *ZZ3, uint64_t *ZZZ3,
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

#if QSB_NEG_Y_MAC
  _ModSub256(Q, T, Q); // seed negative deferred ordinate
#else
  _ModSub256(Q, Q, T);
#endif                             // Q - X3
  _ModMult(Y3, Q, R);                              // deferred R*(Q-X3)
  Load256(X3, Q);                                  // X3
}
