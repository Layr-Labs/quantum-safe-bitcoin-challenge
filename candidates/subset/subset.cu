#define QSB_REDRAW_09260102 1   /* inert re-measurement tag; unreferenced */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Test unrolling four constant SHA blocks; preserve block order and round schedule. */
#define QSB_PAIR_SHA_UNROLL_CONST 1
#define QSB_SHA_FMA_ADD 0
#include "tests/gpu_epochs/tree.cu"
