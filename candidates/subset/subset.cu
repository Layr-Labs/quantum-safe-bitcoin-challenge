#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Paired SHA: window block and the four-block constant loop unrolled, each block's 8-round
 * inner loop rolled (terrapinelf): best steady-state vs cold-JIT tradeoff measured on this tree. */
#define QSB_PAIR_SHA_UNROLL_CONST 1
#include "tests/gpu_epochs/tree.cu"
