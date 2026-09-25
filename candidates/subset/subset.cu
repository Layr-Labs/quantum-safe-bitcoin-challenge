#define QSB_REROLL_0925_142216UTC 1 /* inert re-roll tag */
#ifndef QSB_REMEASURE_TAG_0921R1
#define QSB_REMEASURE_TAG_0921R1 1 /* no-op: exact-source remeasurement re-roll */
#endif
/* Production d32 + L1' root-inverse LDS LUT (QSB_L1_LDS_LUT) with the warp-ordering fix
 * (QSB_L1_ROOT_SYNCW), both DEFAULT-ON so the ranked build (no -D) ships them. S1 stays OFF.
 * Build with -DQSB_L1_LDS_LUT=0 to reproduce crown_stack_d32 byte-for-byte. */
#ifndef QSB_L1_LDS_LUT
#define QSB_L1_LDS_LUT 1
#endif
#ifndef QSB_L1_ROOT_SYNCW
#define QSB_L1_ROOT_SYNCW 1
#endif
#include "tests/gpu_epochs/tree.cu"
