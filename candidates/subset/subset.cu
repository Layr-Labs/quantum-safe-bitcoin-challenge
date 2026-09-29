#define QSB_REDRAW_ERC_0929134104 1   /* inert tag; unreferenced */
#ifndef QSB_FKLEAN_TAG_0924
#define QSB_FKLEAN_TAG_0924 1 /* fk minus the IPC/pipe-routing switches */
#endif
#ifndef QSB_REMEASURE_TAG_0921R3
#define QSB_REMEASURE_TAG_0921R3 1 /* no-op: exact-source PR897 remeasurement */
#endif
/* Keep the paired SHA constant-block loop compact on the ranked PTX route. */
#ifndef QSB_PAIR_SHA_UNROLL_CONST
#define QSB_PAIR_SHA_UNROLL_CONST 0
#endif
#define QSB_SHA_FMA_ADD 0
/* Measurement submission: the in-run multi-arm A/B probe (QsbCarrier.h). The ranked build line
 * passes no -D, so the arm count is set here; 1 = the promoted single-image search. */
#ifndef QSB_ARMS
#define QSB_ARMS 5
#endif
/* Slices per arm: QSB_ARMS x QSB_PROBE_SLOTS x ~0.715 s must exceed the 1200 s run (5 x 384 x 0.715 = 1373 s), and a slot
 * (C(137,6) / QSB_ARMS / QSB_PROBE_SLOTS = 4.28M epochs) must hold one slice at the ranked rate (~3.6M epochs). */
#ifndef QSB_PROBE_SLOTS
#define QSB_PROBE_SLOTS 384
#endif
/* kshitij-hash's bit-identical lookahead root inverse, on in every arm unless an arm sets it (terrapinelf 9b0c36bb). */
#ifndef QSB_DIVSTEP_LOOKAHEAD
#define QSB_DIVSTEP_LOOKAHEAD 1
#endif
#include "tests/gpu_epochs/tree.cu"
