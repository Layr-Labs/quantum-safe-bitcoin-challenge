#define QSB_REDRAW_TRS5_COMBPOOL_0930 1   /* inert tag; unreferenced */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Keep the paired SHA constant-block loop compact on the ranked PTX route. */
#define QSB_PAIR_SHA_UNROLL_CONST 0
#define QSB_SHA_FMA_ADD 0
/* Host-only: split the four longest MRG IFMA chains into two accumulators (switch already in CpuGrindSubset.h, default 0). */
#define QSB_CPU_MRGS 1
#include "tests/gpu_epochs/tree.cu"
