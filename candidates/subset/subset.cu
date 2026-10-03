#define QSB_REDRAW_10030408 1   /* inert re-measurement tag; unreferenced */
#define QSB_HIT_TELEMETRY 0   /* ours: no runtime telemetry in the hit order */
#define QSB_CPU_DIAG_EPOCH 0   /* ours: the co-grinder walks from epoch 0 (no diagnostic code) */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Keep the paired SHA constant-block loop compact on the ranked PTX route. */
#define QSB_PAIR_SHA_UNROLL_CONST 0
#define QSB_SHA_FMA_ADD 0
#define QSB_CODE_ROLL 2   /* the pair gate as a 2-trip loop over the two recids (ercumentyildirim PR 2441, ported) */
#define QSB_Q_MIX 2   /* ours: the record has 4; both lay out the same Q terms (bit-identical) */
#define QSB_P_MIX 1   /* the GLV12 P tail rows (16..21); with QSB_LAYOUT_RT their stops come from the launch argument */
#define QSB_LAYOUT_RT 1   /* wr's layout until the gate-form rate rule fires, then every warp on the GLV12 Q and P terms (C); same points, bit-identical hits */
#include "tests/gpu_epochs/tree.cu"
