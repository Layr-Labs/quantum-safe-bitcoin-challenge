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
/* Heat-limit SHA adds (tree.cu, QSB_SHA_ALU_RT): after main() clears QSB_GATE_FMA_C, the four constant blocks and the
 * gate's plain form keep every add on the ALU pipe (a constant-bank zero third operand) instead of IMAD.IADD. Same
 * words, same hits; the code that runs before the clear is the record's. */
#ifndef QSB_SHA_ALU_RT
#define QSB_SHA_ALU_RT 1
#endif
#include "tests/gpu_epochs/tree.cu"
