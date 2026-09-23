#ifndef QSB_REMEASURE_TAG_0921R1
#define QSB_REMEASURE_TAG_0921R1 1 /* no-op: exact-source remeasurement re-roll */
#endif
/* Exact PR1027 scalar/table rescaling; command-line 0 restores A/2 + 2k. */
#ifndef QSB_RECODE_BASE_A
#define QSB_RECODE_BASE_A 1
#endif
/* PR1093 filter-only carry cut; command-line 0 restores the carried limb. */
#ifndef QSB_DROP_Z2_EARLY
#define QSB_DROP_Z2_EARLY 1
#endif
/* Public PR1137 compact paired schedule, composed with the negative-Y stack.
 * Each kill switch keeps the original fully expanded schedule available. */
#ifndef QSB_PAIR_SHA_UNROLL_CONST
#define QSB_PAIR_SHA_UNROLL_CONST 0
#endif
#ifndef QSB_PAIR_SHA_UNROLL_CONST_INNER
#define QSB_PAIR_SHA_UNROLL_CONST_INNER 0
#endif
/* Separate reductions shorten live ranges: native sm89 has no digest spills. */
#ifndef QSB_FUSE_X3
#define QSB_FUSE_X3 0
#endif
#include "tests/gpu_epochs/tree.cu"
