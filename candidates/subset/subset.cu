#define QSB_REDRAW_09260102 1   /* inert re-measurement tag; unreferenced */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Native sm_89 carrier route: the constant-suffix SHA blocks are fully unrolled (both the
 * four-block loop and each block's 8-round inner loop), so W+K are immediates and the rolled
 * loop's register-rotation moves and loop control disappear. The rolled form was chosen to cut
 * driver-JIT time on the PTX route (a75cf15a); the carrier is loaded without JIT. */
#define QSB_PAIR_SHA_UNROLL_CONST 1
#define QSB_PAIR_SHA_UNROLL_CONST_INNER 1
#define QSB_SHA_FMA_ADD 0
#include "tests/gpu_epochs/tree.cu"
