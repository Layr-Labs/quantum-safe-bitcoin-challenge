#define QSB_REDRAW_09261927 1   /* inert re-measurement tag; unreferenced */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Keep the paired SHA constant-block loop compact on the ranked PTX route. */
#define QSB_PAIR_SHA_UNROLL_CONST 0
/* SHA-256 message-schedule/round additions routed through IMAD (measured +0.6% peak with the
 * lane-predicated root below on the 4090 against the d052bc3d crown). */
#define QSB_SHA_FMA_ADD 1
#ifndef QSB_ROOT_UNIFORM_WARP
#define QSB_ROOT_UNIFORM_WARP 0
#endif
#ifndef QSB_Q_MIX
#define QSB_Q_MIX 8
#endif
#include "tests/gpu_epochs/tree.cu"
