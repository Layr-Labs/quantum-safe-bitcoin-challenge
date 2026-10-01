#define QSB_COMBO_1002 1   /* inert tag; unreferenced */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Keep the paired SHA constant-block loop compact on the ranked PTX route. */
#define QSB_PAIR_SHA_UNROLL_CONST 0
#define QSB_SHA_FMA_ADD 0
#define QSB_CODE_ROLL 3   /* exact device switch: bit 1 (gate loop, ranked in-run probe f70a9cbd) + bit 0 (outer block in the front, measurement) */
#include "tests/gpu_epochs/tree.cu"
