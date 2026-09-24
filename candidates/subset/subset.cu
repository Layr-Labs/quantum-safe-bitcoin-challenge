// Split paired consumer; preserve the promoted arithmetic and exact replay.
#define QSB_SPLIT_PIPELINE 1
#define ZLAB_LAUNCH_BLOCKS 131072
#include "tests/gpu_epochs/tree.cu"
