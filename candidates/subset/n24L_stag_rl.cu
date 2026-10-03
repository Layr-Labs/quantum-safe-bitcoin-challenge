// N=24 scored-path arm "rl": QSB_ROOT_LUT_SMEM 1 — the warp-0 root inverse's
// 832-word divstep table read from shared memory (LDS.64 broadcast ~30c) instead
// of constant bank 3 (63-66c waits on the root's serial chain; table > 2KiB L1).
// Documented in tree.cu:655 but never measured. Independent warp (root) from the
// tail-stagger mechanism (warps 1..7), so gains should stack if real.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 1
#define QSB_ROOT_LUT_SMEM 1
#include "subset.cu"
