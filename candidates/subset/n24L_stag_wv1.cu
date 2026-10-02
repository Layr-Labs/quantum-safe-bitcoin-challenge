// N=24 scored-path arm "wv1": QSB_TAIL_WEAVE 1 — the FGFG order's middle two
// tail calls (G(A) F(B)) become one __noinline__ qsb_pair_weave3_value whose
// body is gate A followed by finish B in one basic block, so ptxas interleaves
// the gate's ALU rounds with the finish's fma-heavy products inside the warp
// (the same idea TAIL_STAGGER applies across warps; tree.cu:884-906).
// B's twelve words pass through the thread's own parkA rows (12 STS.64 +
// 12 LDS.64) instead of 24 argument registers that spilled the body.
// Form 1: recid-0 hash in the finish's first BB, recid-1 in its second.
// Builds at 128 regs / 0 stack / 0 spills (B8 round 5). Forces SC_LATE.
// Documented bit-identical (work/tail_weave_replay.py). Never measured.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 1
#define QSB_TAIL_WEAVE 1
#include "subset.cu"
