#define QSB_REDRAW_09241557 1   /* inert re-measurement tag; unreferenced */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Keep the paired SHA constant-block loop compact on the ranked PTX route. */
#define QSB_PAIR_SHA_UNROLL_CONST 0
#include "tests/gpu_epochs/tree.cu"


// Yukon reuse package v1; original inventory SHA-256: a6d7e11c427a851abf1b870b895184c3cb6bc3fed7d239958611d39c0de5399b
