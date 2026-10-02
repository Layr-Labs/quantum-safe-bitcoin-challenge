// N=24 scored-path arm "s5": runtime-uniform FGFG control (value 1's codegen,
// every warp FGFG at run time). Ladder value 5 — isolates phase-mixing vs
// pure codegen shape of the split-tail branches.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 5
#include "subset.cu"
