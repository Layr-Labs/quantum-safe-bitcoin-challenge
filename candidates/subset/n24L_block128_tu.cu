// Compile-time 128-thread up/down tree levels on the verified smaller CTA.
#define QSB_LOCAL_SM86 1
#define QSB_SE_BLOCK 128
#define QSB_TREE_UNROLL 1
#include "subset.cu"
