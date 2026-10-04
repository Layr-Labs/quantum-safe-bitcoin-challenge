#define QSB_REDRAW_09260102 1   /* inert re-measurement tag; unreferenced */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Keep the paired SHA constant-block loop compact on the ranked PTX route. */
#define QSB_PAIR_SHA_UNROLL_CONST 0
#define QSB_SHA_FMA_ADD 0
/* SHA adds on the ALU pipe (tree.cu, QSB_SHA_ALU_RT 2): the window block, the four constant blocks and the outer block's
 * schedule take a constant-bank zero third operand, so ptxas keeps them as IADD3 instead of IMAD.IADD, in every phase;
 * the gate's plain form (after main() clears QSB_GATE_FMA_C) does the same. Same words, same hits. */
#ifndef QSB_SHA_ALU_RT
#define QSB_SHA_ALU_RT 2
#endif
#include "tests/gpu_epochs/tree.cu"
