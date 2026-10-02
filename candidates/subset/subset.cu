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
/* Smaller independent digest CTAs, same window patterns and field arithmetic. */
#ifndef QSB_SE_BLOCK
#define QSB_SE_BLOCK 128
#endif
/* Keep twice the restored work per host iteration behind the four resident CTAs. */
#ifndef ZLAB_LAUNCH_BLOCKS
#define ZLAB_LAUNCH_BLOCKS 1048576
#endif
/* Retire both opposite-denominator live ranges before the common inversion. */
#ifndef QSB_DEN_CROSS_PRE
#define QSB_DEN_CROSS_PRE 1
#endif
/* Centered-square mode2 qualified against promoted device source. */
#ifndef QSB_K2S_CENTER_SQR
#define QSB_K2S_CENTER_SQR 2
#endif
/* Full startup self-check aliases immutable slot0 until comparison completes. */
#ifndef QSB_HP_CHECK_ALIAS
#define QSB_HP_CHECK_ALIAS 1
#endif
/* Restore the tight epoch-group bound for this doubled epoch capacity.
 * Host allocation only: native digest image and enumeration unchanged. */
#ifndef QSB_GROUP_CAP_TIGHT
#define QSB_GROUP_CAP_TIGHT 1
#endif
#include "tests/gpu_epochs/tree.cu"
