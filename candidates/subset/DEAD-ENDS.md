# Subset dead ends — 2026-10-01 night session

Every entry below was measured on the dev RTX 3090 (sm_86, GLV12 9.13 GiB geometry, `QSB_LOCAL_SM86 1`) unless noted. N=4 = the fast local ABBA protocol; N=24 = ranked difficulty.

| lever | setting | measurement | verdict |
|---|---|---|---|
| `QSB_OUTER_LITK` | 1 (literal-K on the outer SHA block) | 4-round ABBA at N=4: A(C2) 318.45±0.95 vs B(D2) 319.18±0.15, delta −0.21% | DEAD solo (noise). The earlier +2.7% (319.4 vs 311) was a mid-run epoch-line read during warm-up drift — both arms read ~318-319 at steady state. |
| `QSB_PRE3_ROOT` | 1 | q-pairs −0.46% (4/4 neg), kernel census −0.60%, smokes −3.7% | DEAD (3 reads). Spills 20/24 B + forces SC_LATE. |
| Native GLV11 geometry on the dev rig | `QSB_LOCAL_SM86 0` | table 22,688,113,472 B = 21.1 GiB; OOMs solo (20.7 GiB free with desktop session) | IMPOSSIBLE on 24 GiB 3090. All N=24 dev work must ride the GLV12 9.13 GiB geometry. "GLV12 table allocation failed" is printed by BOTH geometries — the label is not diagnostic. |
| Two concurrent scored runs in one checkout | 1025+1026 | "judge or problem files modified during run" + seed mismatch REJECT | Scored runs are strictly serial; they share `benchmark-results/problem/` and `results/`. |
| `bash stat %.9Y` stamps for gpu_wrap | (rig bug) | prints `1790855154.253761415` (dot) vs python `st_mtime_ns` `1790855154253761415` (integer) — never equal | Always write stamps with python, or the harness plain-rebuilds (and on this rig, OOMs). |
| Pre-include `#define` overrides of knobs tree.cu defines | — | WRONG THEORY (retracted): the four knobs ARE `#ifndef`-guarded; wrapper defines and `-D` DO take effect. The real failure was geometry (row 3). | Keep wrapper .cu files; they are valid. |

## Confirmed-live levers (for stacking)

| lever | setting | N=24 scored (seed 1789110211) | status |
|---|---|---|---|
| `QSB_TAIL_STAGGER` | 1 (software-half FFGG/FGFG) | scored 4 pairs: F 366.78 vs C2 371.57 = +1.31%; kernel census 4x120 ABBA: F 311.27 (311.7/310.1/304.7/318.6) vs C2 317.55 (316.6/317.1/317.2/319.3) = **+2.02%**, 4/4 pairwise negative | live (committed 257e92c); census says kernel gain > end-to-end gain |
| `QSB_ROOT_LUT_SMEM` | 1 | census queued (job 1046) | unmeasured anywhere |
| `QSB_ROOT_WARP` | 2 | census queued (job 1049) | unmeasured anywhere; probe-backed (B3AP: root-on-0 stretches warps 3/7 by 5.6K cycles, 72% of blocks) |
| `QSB_TAIL_STAGGER` ladder 4 | hardware warpid phasing | census running (job 1044) | in flight |

| `QSB_R_CBANK_TAILS` | 1 | Earlier N=4 ABBA 4x120 (job 1003): A 308.60±1.52 vs B 307.65±2.06 GPU-only = −0.31% | DEAD solo at N=4 (neutral-to-negative); the queued N=24 census (1051) will confirm before removal. |

| `QSB_TAIL_STAGGER` ladder 4 (hardware warpid phasing) | s4 vs C2 | N=24 kernel census 4x120 ABBA: C2 316.18±0.92 vs s4 317.40±0.87 = +0.39%, 4/4 pairwise positive but inside the 1.5% noise band | MARGINAL (not a keeper solo). Warpid phasing ≈ software-half phasing; the s4-vs-s5 control (running) tells whether any of it is phase-diversity at all. |
