#define QSB_DIVSTEP_LOOKAHEAD 1   /* exact knob of the tree: divstep lookahead in the root inverse */
#define QSB_PARK128 0             /* needed by QSB_PRE3_ROOT */
#define QSB_PRE3_ROOT 1           /* exact knob of the tree: pre3 root scheduling */
#define QSB_Q_MIX 8               /* 1 in 8 warps on the GLV12 Q layout (was 4) */
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
#include "tests/gpu_epochs/tree.cu"
