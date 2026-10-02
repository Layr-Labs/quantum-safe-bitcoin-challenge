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
/* Every cold record, including direct-table gathers, gets the .L2::64B hint on all four 16-byte loads. */
#define QSB_S3_HINT_MODE 2
#ifndef QSB_DIRECT10
#define QSB_DIRECT10 1
#endif
#include "tests/gpu_epochs/tree.cu"
