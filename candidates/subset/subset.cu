#define QSB_REDRAW_09241557 1   /* inert re-measurement tag; unreferenced */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Keep the paired SHA constant-block loop compact on the ranked PTX route. */
#define QSB_PAIR_SHA_UNROLL_CONST 0
#define QSB_SHA_FMA_ADD 1
/* Schedule-sigma logical shifts (x>>3, x>>10) of the pubkey hash as IMAD.HI on the FMA-heavy
 * pipe: one ALU SHF becomes one IMAD.HI, no extra instruction; rotations stay on SHF. 0 = off. */
#define QSB_SHA_FMA_ROT 8
#include "tests/gpu_epochs/tree.cu"
