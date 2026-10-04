#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Keep the paired SHA constant-block loop compact on the ranked PTX route. */
#define QSB_PAIR_SHA_UNROLL_CONST 0
#define QSB_SHA_FMA_ADD 0
#define QSB_SHA_WROLL_PIPE 1   /* the record's: window block as one 8-round loop, W+K loads half a trip ahead (bit-identical) */
#define QSB_R_CBANK_TAILS 1    /* the record's: R's words from the constant bank in both tails (bit-identical) */
#define QSB_HIT_TELEMETRY 0    /* host only: no runtime telemetry in the hit order */
#define QSB_CPU_DIAG_EPOCH 0   /* host only: the co-grinder's ranges start at epoch 0 (no diagnostic code) */
#define QSB_CPU_BATCH_ODD 2048 /* host only: odd co-grinder workers use 2,048-candidate batches, even ones the record's 1,024 */
#include "tests/gpu_epochs/tree.cu"
