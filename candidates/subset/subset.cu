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
/* Q layout mix: one warp in eight (instead of one in four) decodes Q with the six-term GLV12 layout, the
 * rest with the five-term P18 layout. Per candidate this is 9.125 field additions and 7.75 cold table
 * records instead of 9.25 and 7.5: fewer additions for a little more DRAM traffic, which pays on a card
 * whose clock is limited by core temperature rather than board power. Same points, same hit set. */
#define QSB_Q_MIX 8
/* Exact scheduling switches already in the tree: warps 1-7 move the recovery denominator's pre3 multiply
 * out of their fronts into the root window, where they otherwise wait at the down-sweep barrier (PARK128
 * is written for the path without it, so it is off), and the root's divstep loop forms the next batch's
 * decision before the carry chain (lookahead). */
#define QSB_DIVSTEP_LOOKAHEAD 1
#define QSB_PRE3_ROOT 1
#define QSB_PARK128 0
#include "tests/gpu_epochs/tree.cu"
