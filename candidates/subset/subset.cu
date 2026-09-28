#define QSB_REDRAW_09260102 1   /* inert re-measurement tag; unreferenced */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Keep the paired SHA constant-block loop compact on the ranked PTX route. */
#define QSB_PAIR_SHA_UNROLL_CONST 0
#ifndef QSB_SHA_FMA_ADD
#define QSB_SHA_FMA_ADD 1
#endif
#define QSB_PK_HEAD_FMA 1
#define QSB_GLV_FP32_CARRY 1
#define QSB_PK_OUTLINE 1
#include "tests/gpu_epochs/tree.cu"
