#define QSB_REDRAW_09241557 1   /* inert re-measurement tag; unreferenced */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* The ranked path loads the native sm_89 image and never JIT-compiles the compute_52 PTX,
 * so the paired SHA constant-block loop no longer needs to stay compact: unroll it. */
#define QSB_PAIR_SHA_UNROLL_CONST 1
#include "tests/gpu_epochs/tree.cu"
