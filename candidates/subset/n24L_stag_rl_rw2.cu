// N=24 scored-path arm "rl_rw2": the root-warp stack —
//   QSB_ROOT_LUT_SMEM 1  (divstep table cbank->smem, ~30c vs 63-66c)
//   QSB_ROOT_WARP 2      (tree top on warp 2: per the B3AP phase probe, root on
//                          warp 0 stretches co-resident warps 3/7 by ~5.6K cycles
//                          (last to leaf barrier in 72% of blocks); warp 2 shares a
//                          sub-partition with the youngest block's EARLIEST arrivals
//                          (warps 1/5), so the contention no longer sets the barrier)
// Both documented in tree.cu:655/709, neither ever measured. RW2 requires the
// templated tree (ROOT_LUT_SMEM or PRE3_ROOT) — carried here by RL.
#define QSB_LOCAL_SM86 1
#define QSB_TAIL_STAGGER 1
#define QSB_ROOT_LUT_SMEM 1
#define QSB_ROOT_WARP 2
#include "subset.cu"
