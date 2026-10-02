// N=24 scored-path arm "s4": hardware-subpartition stagger — FFGG iff
// (warpid>>2)&1, i.e. phase by the SM's actual warp scheduler slot instead of
// the software `half` split. QSB_TAIL_STAGGER ladder value 4 (untested axis).
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 4
#include "subset.cu"
