// Double candidates per launch while preserving the verified 128-thread CTA.
#define QSB_LOCAL_SM86 1
#define QSB_SE_BLOCK 128
#define ZLAB_LAUNCH_BLOCKS 524288
#define QSB_DEN_CROSS_PRE 0
#include "subset.cu"
