/* GPL-3.0: host-only peeled batch-affine final addition; derived from
 * ec8_final_cf_kh in CpuGrindSubset.h. Included inside namespace qcpu after
 * kh16_store. Same compressed-key words; only dead endpoint products cut. */
#if QSB_CPU_INV_PEEL
template <bool Advance>
Q8T static inline QCPU_AIF void ec8_peel_final_pair(
const fe8 *X, const fe8 *Y, const fe8 *D, const fe8 *PRE, fe8 *run,
int g, int c, const fe8 &MX, uint32_t *m16) {
const int hA = g + c, hB = hA - 1;
fe8 tA, tB, lamA, lamB, x3A, x3B, y3A, y3B;
fe8_mul_lz(lamA, run[c], PRE[hA]); fe8_mul_lz(lamB, run[c - 1], PRE[hB]);
if (Advance) {
fe8_mul_lz(run[c], run[c], D[hA]); fe8_mul_lz(run[c - 1], run[c - 1], D[hB]);
}
fe8_sqr_sub2(x3A, lamA, X[hA], MX); fe8_sqr_sub2(x3B, lamB, X[hB], MX);
fe8_sub_lz(tA, X[hA], x3A); fe8_sub_lz(tB, X[hB], x3B);
fe8_mul_sub(y3A, lamA, tA, Y[hA]); fe8_mul_sub(y3B, lamB, tB, Y[hB]);
kh16_store(m16 + (size_t)hA * 144, X[hA], fe8_parity(Y[hA]), x3A, fe8_parity(y3A));
kh16_store(m16 + (size_t)hB * 144, X[hB], fe8_parity(Y[hB]), x3B, fe8_parity(y3B));
}
Q8T static void ec8_final_cf_kh_peeled(
const fe8 *X, const fe8 *Y, fe8 *D, fe8 *PRE, int G,
const fe &mx, const fe &my, uint32_t *m16) {
const int NC = QSB_CPU_NCH;
if (!G) return;
fe8 MX, MY; fe8_bcast(MX, mx); fe8_bcast(MY, my);
fe8 run[4];
for (int c = 0; c < NC; c++) {
fe8_sub_lz(D[c], MX, X[c]); fe8_sub_lz(PRE[c], MY, Y[c]);
fe8_cp(run[c], D[c]);
}
for (int g = NC; g < G; g += NC) for (int c = 0; c < NC; c++) {
const int h = g + c;
fe8 t; fe8_sub_lz(D[h], MX, X[h]); fe8_sub_lz(t, MY, Y[h]);
fe8_mul_lz(PRE[h], run[c], t); fe8_mul_lz(run[c], run[c], D[h]);
}
QCPU_INVC(run);
for (int g = G - NC; g >= NC; g -= NC) for (int c = NC - 1; c >= 1; c -= 2)
ec8_peel_final_pair<true>(X, Y, D, PRE, run, g, c, MX, m16);
for (int c = NC - 1; c >= 1; c -= 2)
ec8_peel_final_pair<false>(X, Y, D, PRE, run, 0, c, MX, m16);
}
#endif
