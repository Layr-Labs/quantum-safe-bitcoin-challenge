#define QSB_REDRAW_ERC_1002125301 1   /* inert tag; unreferenced */
#define QSB_R_CBANK 1   /* the recovery point R from the constant bank in the paired front and tail (bit-identical) */
#define QSB_Q_MIX 2   /* the record's Q-table layout 2 (same values; layout only) */
#define QSB_HIT_TELEMETRY 0   /* host only: no NVML log in the hit order (hit set unchanged) */
#define QSB_CPU_DIAG_EPOCH 0   /* host only: the co-grinder walk starts at epoch 0 (enumeration only) */
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
