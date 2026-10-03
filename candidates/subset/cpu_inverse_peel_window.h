/* GPL-3.0: host-only endpoint peeling for CpuGrindSubset's batch-affine
 * window. The weighted-prefix arithmetic derives from that header; unlike
 * the JL cuts, first/last tests are outside the hot group loops. Included
 * inside namespace qcpu, after qcpu_pf_rows. Default-off research path. */
#if QSB_CPU_INV_PEEL
template <bool Advance, class RowFn>
Q8TX static inline QCPU_AIF void ec8_peel_window_pair(
fe8 *X, fe8 *Y, const fe8 *D, const fe8 *PRE, fe8 *run,
int G, int g, int c, const RowFn *nxt, const pt **rpn, __mmask8 *ngn) {
const int hA = g + c, hB = hA - 1;
const pt **pA = nullptr, **pB = nullptr;
if (nxt) {
const int nA = G - 1 - hA, nB = nA + 1;
pA = rpn + (size_t)nA * 8; ngn[nA] = (*nxt)(nA, pA);
pB = rpn + (size_t)nB * 8; ngn[nB] = (*nxt)(nB, pB);
}
qcpu_pf_rows(pA, 0, 4);
fe8 lamA, lamB, x3A, x3B, tA, tB;
fe8_mul_lz(lamA, run[c], PRE[hA]); fe8_mul_lz(lamB, run[c - 1], PRE[hB]);
qcpu_pf_rows(pA, 4, 8);
if (Advance) {
fe8_mul_lz(run[c], run[c], D[hA]); fe8_mul_lz(run[c - 1], run[c - 1], D[hB]);
}
qcpu_pf_rows(pB, 0, 4);
fe8_sqr_subdx(x3A, lamA, D[hA], X[hA]); fe8_sqr_subdx(x3B, lamB, D[hB], X[hB]);
qcpu_pf_rows(pB, 4, 8);
fe8_sub_nf(tA, X[hA], x3A); fe8_sub_nf(tB, X[hB], x3B);
fe8_mul_sub(Y[hA], lamA, tA, Y[hA]); fe8_mul_sub(Y[hB], lamB, tB, Y[hB]);
fe8_cp(X[hA], x3A); fe8_cp(X[hB], x3B);
}

template <class RowFn>
Q8TX static void ec8_window_peeled(
fe8 *X, fe8 *Y, fe8 *D, fe8 *PRE, int G, const pt *const *rp,
const __mmask8 *ng, const pt **rpn, __mmask8 *ngn, const RowFn *nxt) {
const int NC = QSB_CPU_NCH, PF = QSB_CPU_PFD;
if (!G) return; // no output or row publications for the empty batch
fe8 run[4];
// Every live chain starts at one; copy the first factors. Raw weighted
// differences remain multiply-only IFMA inputs, as in JL FIRST=2.
for (int c = 0; c < NC; c++) {
if (c + PF < G) for (int j = 0; j < 8; j++)
_mm_prefetch((const char *)rp[(size_t)(c + PF) * 8 + j], _MM_HINT_T0);
fe8 tx, ty;
pt8_load(tx, ty, rp + (size_t)c * 8);
fe8_sub_d(D[c], tx, X[c]);
fe8_sub_sgn_nf(PRE[c], ty, Y[c], ng[c]);
fe8_cp(run[c], D[c]);
}
for (int g = NC; g < G; g += NC) for (int c = 0; c < NC; c++) {
const int h = g + c;
if (h + PF < G) for (int j = 0; j < 8; j++)
_mm_prefetch((const char *)rp[(size_t)(h + PF) * 8 + j], _MM_HINT_T0);
fe8 tx, ty, t;
pt8_load(tx, ty, rp + (size_t)h * 8);
fe8_sub_d(D[h], tx, X[h]); fe8_sub_sgn_nf(t, ty, Y[h], ng[h]);
fe8_mul_lz(PRE[h], run[c], t); fe8_mul_lz(run[c], run[c], D[h]);
}
QCPU_INVC(run);
for (int g = G - NC; g >= NC; g -= NC) for (int c = NC - 1; c >= 1; c -= 2)
ec8_peel_window_pair<true>(X, Y, D, PRE, run, G, g, c, nxt, rpn, ngn);
// Slopes and next-row publications still happen for g=0; only the dead
// chain products disappear. No first/last predicate in the interior.
for (int c = NC - 1; c >= 1; c -= 2)
ec8_peel_window_pair<false>(X, Y, D, PRE, run, G, 0, c, nxt, rpn, ngn);
}
#endif
