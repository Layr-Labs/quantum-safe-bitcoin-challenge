#define QSB_REDRAW_09260102 1   /* inert re-measurement tag; unreferenced */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Keep the paired SHA constant-block loop compact on the ranked PTX route. */
#ifndef QSB_PAIR_SHA_UNROLL_CONST
#define QSB_PAIR_SHA_UNROLL_CONST 0
#endif
#define QSB_SHA_FMA_ADD 0
/* Measurement submission: the in-run multi-arm A/B probe (QsbCarrier.h). The ranked build line
 * passes no -D, so the arm count is set here; 1 = the promoted single-image search. */
#ifndef QSB_ARMS
#define QSB_ARMS 7
#endif
#include "tests/gpu_epochs/tree.cu"
