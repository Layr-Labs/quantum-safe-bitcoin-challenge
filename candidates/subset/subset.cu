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
#include "tests/gpu_epochs/tree.cu"
