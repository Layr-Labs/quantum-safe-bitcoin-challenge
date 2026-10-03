// N=24 scored-path A/B arm "C2": CURRENT default config (stagger ON, PARK128
// OFF) on the dev sm_86 geometry (GLV12, 9.13 GiB). tree.cu defaults supply
// stagger=1/park=0; the define is explicit for readability.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 1
#define QSB_PARK128 0
#include "subset.cu"
