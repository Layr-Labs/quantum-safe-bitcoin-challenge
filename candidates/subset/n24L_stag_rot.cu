// N=24 scored-path arm "rot": QSB_SHA_FMA_ROT 4 — schedule sigmas s0/s1's two
// rotations as IMAD.WIDE.U32 products on the FMA pipe (ROR(x,32-k) = lo+hi of
// x*2^k, disjoint halves), keeping the logical shift on ALU. The SHA gate and
// the paired hash sit BESIDE the IMAD-bound chain loop, so moving rotations
// off the ALU pipe rebalances the same sub-partition overlap TAIL_STAGGER
// exploits (sha_gate_fma.cuh:52-58; bit 4 = the two rotations only).
// Documented exact for every x, 1<=k<=31. Never measured.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 1
#define QSB_SHA_FMA_ROT 4
#include "subset.cu"
