#define QSB_REDRAW_ERC_THERMAL_ROLL_1002 1   /* inert tag; unreferenced */
#define QSB_R_CBANK 1   /* the record's switch (pair_shared.cuh): R from the constant bank in the paired front/tail, bit-identical */
#define QSB_Q_MIX 2   /* the record's Q-layout switch (tree.cu): one warp in two decodes Q with the six GLV12 terms (the record: one in four) */
#define QSB_RT_THERMAL 2   /* host only: the record's gate-form switch also fires once NVML reports thermal slowdown in 2 consecutive 1 s samples */
#define QSB_CODE_ROLL 2   /* exact rolled recovery gate from the 31edf branch */
#define QSB_HIT_TELEMETRY 1   /* thermal streak source; retained for the ranked sampler */
#define QSB_CPU_DIAG_EPOCH 0   /* no diagnostic walk prefix */
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
