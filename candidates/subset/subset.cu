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
/* Keep each CPU worker's batch working set smaller and avoid the largest random-access
 * signed-digit table. The 12-window / 2048-candidate geometry was measured on
 * the related co-grinder in public submission 97f347a; assess this composition
 * on the current promoted CPU pipeline with the official ranked run. */
#define QSB_CPU_BATCH 2048
#define QSB_CPU_LMIN 12
#include "tests/gpu_epochs/tree.cu"
