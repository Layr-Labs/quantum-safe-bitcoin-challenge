#ifndef QSB_REDRAW_TAG_QSD3_0925A
#define QSB_REDRAW_TAG_QSD3_0925A 1 /* no-op: disclosed exact-source redraw QSB_REDRAW_TAG_QSD3_0925A */
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
