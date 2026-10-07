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
/* Host-only experiment: five early omissions and four window omissions.
 * -DQSB_CPU_FAMILY54=0 restores the promoted CPU candidate family. */
#ifndef QSB_CPU_FAMILY54
#define QSB_CPU_FAMILY54 1
#endif
#if QSB_CPU_FAMILY54
#ifndef QSB_CPU_DIAG_EPOCH
#define QSB_CPU_DIAG_EPOCH 0
#endif
#endif
#include "tests/gpu_epochs/tree.cu"
