#define QSB_REDRAW_10021304 1   /* inert re-measurement tag; unreferenced */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Keep the paired SHA constant-block loop compact on the ranked PTX route. */
#define QSB_PAIR_SHA_UNROLL_CONST 0
#define QSB_SHA_FMA_ADD 0
#define QSB_DIVSTEP_LOOKAHEAD 1   /* on: kshitij-hash's bit-identical lookahead root inverse */
#define QSB_CODE_ROLL 2   /* the pair gate as a 2-trip loop over the two recids (-20.9 KB of digest code) */
#define QSB_WSEC_L1LAST 0   /* off: only QSB_CODE_ROLL 2 is taken (as in DPZZxlz 7843167a) */
#include "tests/gpu_epochs/tree.cu"
