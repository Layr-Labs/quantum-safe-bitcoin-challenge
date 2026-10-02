// N=24 scored-path arm "wv2": QSB_TAIL_WEAVE 2 — same woven callee as form 1,
// but both hashes sit ahead of the whole finish in one basic block
// (tree.cu:902-906). Builds at 127 regs / 0 stack / 0 spills. Never measured.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 1
#define QSB_TAIL_WEAVE 2
#include "subset.cu"
