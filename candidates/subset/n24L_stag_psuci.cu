// N=24 scored-path arm "psuci": QSB_PAIR_SHA_UNROLL_CONST_INNER 1 — roll the
// 8-round inner loop of the paired epoch SHA's four constant blocks (the four-
// block loop stays unrolled; window_schedule_shared.cuh:212-216). Default 0 =
// fully unrolled constant blocks on the per-candidate paired path. Pricing the
// I-cache footprint of the fully unrolled form against loop overhead on the
// hottest SHA path. Never measured.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 1
#define QSB_PAIR_SHA_UNROLL_CONST_INNER 1
#include "subset.cu"
