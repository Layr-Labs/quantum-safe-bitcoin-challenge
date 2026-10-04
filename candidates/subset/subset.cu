#define QSB_REDRAW_ERC_1003P2R8 1   /* inert tag; unreferenced */
#define QSB_R_CBANK 1   /* the record's switch (pair_shared.cuh): R from the constant bank in the paired front/tail, bit-identical */
#define QSB_Q_MIX 2   /* the record's Q-layout switch (tree.cu): one warp in two decodes Q with the six GLV12 terms (the record: one in four) */
#define QSB_QMIX_RT 1   /* the record's run-time Q-layout switch, on: Q_MIX 2 at the start */
#define QSB_QMIX_RT_TARGET 1   /* the one write selects the GLV12 terms for every warp (mask 0) */
#define QSB_RT_THERMAL 2   /* host only: the record's gate-form switch also fires once NVML reports thermal slowdown in 2 consecutive 1 s samples */
#define QSB_SHA_WROLL_PIPE 1   /* the record's switch (tree.cu): the window block as one 8-round loop with its W+K loads half a trip ahead */
#define QSB_PAIR_SHA_UNROLL_WINDOW 0   /* required by QSB_SHA_WROLL_PIPE (window_schedule_shared.cuh) */
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
