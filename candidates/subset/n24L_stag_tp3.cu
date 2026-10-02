// N=24 scored-path arm "tp3": QSB_TAIL_PARK 3 — F(A)F(B)G(A)G(B) with both
// inverses formed first and nB parked in rows 0..11 before the first call.
// 127 regs / 0 stack / 0 spills, 20 slots/candidate more than the default 4
// (tree.cu:749-756) — the register-cheapest split form, never measured.
// Sensible only if the census shows value 4's 128-reg ceiling is what limits
// occupancy at the tail; otherwise dead on slot count.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 1
#define QSB_TAIL_PARK 3
#include "subset.cu"
