// N=24 scored-path arm "sew256": QSB_SE_WINDOWS 128 -> 256 (the promoted
// value) — window omission sets per epoch and the size of WIN3. 256 windows
// need 54 distinct first blocks vs 128's 8, so kernel_build_first_flat costs
// more per epoch, but each epoch covers 2x the omission sets, halving epoch
// count per unit of candidate space (tree.cu:2494-2501). Host-side build cost
// vs GPU-side epoch throughput: the balance was promoted at 256 on the ranked
// line but this tree runs 128 — this arm prices the difference on the dev rig.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 1
#define QSB_SE_WINDOWS 256
#include "subset.cu"
