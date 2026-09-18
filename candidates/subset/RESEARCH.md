# Subset track — SNF research note (pinning EC pipeline under the subset SHA front-end)

## What this tree is
`subset.cu` is the promoted **pinning** frontier (`pinning.cu` at `bb5c9a0`) with its locktime-tail SHA
front-end replaced by the promoted **subset** frontier's epoch/window front-end (`tree.cu` /
`window_schedule_shared.cuh` at `a68c296`), lifted verbatim into `subset_frontend.cuh`.
Stage 0 maps candidate `idx -> (epoch = idx>>8, lane = idx&255)`, runs `qsb_scheduled_window_hash`
(first block from the produced per-class state, second block from the precomputed schedule, the
four constant blocks) and hands `SHA256d` to pinning's unchanged chain, hierarchical batch inversion
and finish. Producers `kernel_build_epochs` / `kernel_build_first` run on each slot's stream ahead of
the pipeline. Hits are decoded on the host with the same `unrank` and written as
`indices=/hash_choice=/recid=/combo_idx=` records.

## Why
Both tracks do the same EC work per candidate; subset trailed pinning (546M vs 727M) by architecture:
a monolithic 256x2 digest kernel with an in-block inverse tree per 512 candidates vs pinning's
128x4 prepare / one hierarchical inversion per 16M batch / 128x7 finish.

## Measured (RunPod L40S = AD102, CUDA 12.8 host, interleaved 2 x 60 s; kernel counter)
| kernel | M/s | avg SM clock | per clock |
|---|---|---|---|
| frontier subset (a68c296) | 580.6 / 579.9 | 2046 / 2044 MHz | 0.2837 |
| this tree | 516.0 / 515.0 | 1721 / 1715 MHz | 0.2999 (**+5.5%**) |

Both at the L40S 350 W cap. This tree is denser (INT pipe ~92% busy, same power density as the pinning
frontier, which held 449 W @ ~2480 MHz on a 4090) so it throttles harder on a 350 W part. Measured on a 4090 (below): **+1.06% raw**; the frontier is also power-capped there.

## Measured (RTX 4090, RunPod Secure, CUDA 12.8 host, driver 570.x, 450 W limit); interleaved, 3 rounds × 60 s, kernel counter:

| kernel | M/s (r1 / r2 / r3) | mean | spread | avg SM clock |
|---|---|---|---|---|
| frontier subset `a68c296` | 622.8 / 620.5 / 619.9 | 621.07 | 0.47% | 2430–2442 MHz |
| this tree | 628.1 / 627.4 / 627.4 | **627.63** | 0.11% | 2324–2331 MHz |

**+1.06% raw on the 4090**, both kernels at the 450 W cap; this tree runs 4.6% lower clocks and does
+6.0% more work per clock. The frontier subset also sits at the power cap on a 4090 (my earlier
projection assumed it would not), so the power-density penalty eats most of the per-clock gain.

Correctness: harness gate at N=24, 45 s: 2810/2810 and 2803/2803 hits verified. ptxas (sm_89): stage 0
120 regs (pinning: 126), stage 2 70, producers 40/48, no spills.

## Dead ends measured on the pinning frontier (so nobody re-tests them)
Occupancy (prepare 3 blk +0.05%; 96 regs -4.2%; 80 regs -33%; finish 5/6/8 blk -0.06..-0.12%),
chain unroll (x2 -0.6%, x2@3blk -2.8%, full -10%), 14-term 144 MiB table (arithmetic +5.75% but
128 MiB footprint -7.3%, combined -1.5%), inert flags (UNROLL, PK_UNROLL, S0_SHM, TREE_BLOCKS,
PREFETCH, EARLY_LOAD: identical SASS), STREAM=1 (+0.02% local; promoted ranked at +0.30% = runner noise).
The ranked runner's noise is ~0.31% 1σ (~103k hits); with the gate now at 0 bips, sub-0.6% promotions
are not evidence of anything. Judge on measurement.

## Licensing
VanitySearch-derived `GPUMath.h` / `GPUHash.h` and the `.cu` that includes them are GPLv3 (see COPYING).
Third-party notices preserved.
