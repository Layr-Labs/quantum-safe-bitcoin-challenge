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
/* Phase-1 trims, both existing switches of this tree. Ranked hit-order telemetry read the GPU rate of the 450 W
 * phase (before the card reaches 90 C) about 0.25% higher without the rolled gate and 0.15% higher with R in the
 * constant bank, with the thermal-phase cycles per candidate unchanged. Same words, same operations, same hits. */
#ifndef QSB_CODE_ROLL
#define QSB_CODE_ROLL 0   /* tree.cu: the gate's two H0 hashes inline (the record: 2, the rolled gate) */
#endif
#ifndef QSB_R_CBANK
#define QSB_R_CBANK 1     /* pair_shared.cuh: the paired front and tail read R from the constant bank (the record: 0) */
#endif
#include "tests/gpu_epochs/tree.cu"
