#ifndef QSB_RESUB_0924HN1
#define QSB_RESUB_0924HN1 1 /* inert resubmission tag: identical build, fresh ranked draw */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Keep the paired SHA constant-block loop compact on the ranked PTX route. */
#define QSB_PAIR_SHA_UNROLL_CONST 0
#include "tests/gpu_epochs/tree.cu"
