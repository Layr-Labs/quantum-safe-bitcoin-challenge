// N=24 scored-path A/B arm "P3": fused pre3-root hypothesis (QSB_PRE3_ROOT 1)
// on the current default config + dev sm_86 geometry (GLV12, 9.13 GiB).
// UNTESTED on subset: folds the pre3 root computation into the kernel root,
// needs wave top + K2S3M + DUAL_EPOCH_SHA (all on in this lineage).
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 1
#define QSB_PARK128 0
#define QSB_PRE3_ROOT 1
#include "subset.cu"
