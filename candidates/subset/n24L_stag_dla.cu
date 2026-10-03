// N=24 scored-path arm "dla": QSB_DIVSTEP_LOOKAHEAD 1 — in the root's divstep
// loop (inverse_limbs.cuh) the next batch's decision (five table lookups on the
// new low limbs of f and g) is formed from the row accumulators before the
// carry chain, in the same basic block as this batch's correction, ballot
// carries, shift and termination vote, so ptxas overlaps the two dependency
// chains (tree.cu:919-926). The root is the B3AP-identified warp-3/7 stretch
// source (5.6K cycles, 72% of blocks), so this attacks the measured bottleneck.
// Documented exact by construction; never measured.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 1
#define QSB_DIVSTEP_LOOKAHEAD 1
#include "subset.cu"
