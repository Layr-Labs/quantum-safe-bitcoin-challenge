// N=24 scored-path arm "psha": ZLAB_PAIRSHA 1 — hash both recovery public
// keys with one interleaved pair of one-block SHA-256 compressions instead of
// the sequential loop (tree.cu:122-127; independent implementation of an idea
// from unpromoted 5605ad8 @nullforest8200, isolated in 558d022 @DPZZxlz).
// Start-up work (2 recoveries per problem), so the effect shows as a shorter
// warm-up, not steady-state rate — priced as start-up latency only.
// Never measured on this tree.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 1
#define ZLAB_PAIRSHA 1
#include "subset.cu"
