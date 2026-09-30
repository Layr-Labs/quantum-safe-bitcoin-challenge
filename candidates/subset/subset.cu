#define QSB_REDRAW_ERC_0930044151 1   /* inert tag; unreferenced */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Keep the paired SHA constant-block loop compact on the ranked PTX route. */
#define QSB_PAIR_SHA_UNROLL_CONST 0
#define QSB_SHA_FMA_ADD 0
#define QSB_DIVSTEP_LOOKAHEAD 1   /* offsw screen */
#define QSB_HP_SKIP 1   /* host producers off: the GPU producers build every batch (host-only) */
#include "tests/gpu_epochs/tree.cu"


// Yukon reuse package v1; original inventory SHA-256: 9cbbecef7516fc0c9175ffc045b4c8f7d9d4f42f884a25608a41e050ce96d3ef
