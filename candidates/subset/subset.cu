#define QSB_REDRAW_09260102 1   /* inert re-measurement tag; unreferenced */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* EXP 2026-09-26: frontier forces 0 here ("keep the paired SHA constant-block
 * loop compact on the ranked PTX route"). Restoring the shared header's default
 * (1) so the constant-block loop is unrolled on the ranked route is the
 * experiment. */
#define QSB_PAIR_SHA_UNROLL_CONST 1
#define QSB_SHA_FMA_ADD 0
#include "tests/gpu_epochs/tree.cu"
