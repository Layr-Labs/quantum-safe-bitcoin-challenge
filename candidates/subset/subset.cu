#define QSB_REDRAW_10030557 1   /* inert re-measurement tag; unreferenced */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Keep the paired SHA constant-block loop compact on the ranked PTX route. */
#define QSB_PAIR_SHA_UNROLL_CONST 0
#define QSB_SHA_FMA_ADD 0
#define QSB_HIT_TELEMETRY 0   /* ours: no runtime telemetry in the hit order */
#define QSB_CPU_DIAG_EPOCH 0   /* ours: the co-grinder walks from epoch 0 (no diagnostic code) */
#define QSB_CODE_ROLL 2   /* the pair gate as a 2-trip loop over the two recids (ercumentyildirim PR 2441, ported) */
#define QSB_Q_MIX 2   /* ours: the record has 4; both lay out the same Q terms (bit-identical) */
#include "tests/gpu_epochs/tree.cu"
