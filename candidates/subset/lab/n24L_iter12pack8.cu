/* Iteration12 full-capacity compact first-state pitch screen.
 * Reuses the audited lossless schedule packing with current centered-square
 * arithmetic; does not shrink ZLAB_LAUNCH_BLOCKS or pipeline capacity. */
#define QSB_LOCAL_SM86 1
#define QSB_FIRST_PACK8 1
#include "../subset.cu"
