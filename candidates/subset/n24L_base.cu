// N=24 scored-path A/B arm "F": OLD PROMOTED config (stagger OFF, PARK128 ON)
// on the dev sm_86 geometry (GLV12 six-term table, 9.13 GiB — the only one
// that fits this 24 GiB 3090; native GLV11 needs 22.7 GiB and OOMs solo).
// Knobs are #ifndef-guarded in tree.cu, so defines here DO take effect.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 0
#define QSB_PARK128 1
#include "subset.cu"
