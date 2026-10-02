// N=24 scored-path arm "peel": QSB_SHA_CONST_PEEL 1 — with QSB_SHA_CONST_IV
// (already 1 in this tree), rounds 0..7 of each constant block read the saved
// state directly, removing the 16 working-state copies per block per pair at
// the cost of one more 8-round body of code (tree.cu:307). The paired hash's
// four constant blocks run on the per-candidate SHA path.
// Documented bit-identical; never measured.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 1
#define QSB_SHA_CONST_PEEL 1
#include "subset.cu"
