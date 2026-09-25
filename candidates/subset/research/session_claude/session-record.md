# Session record — 2026-09-25, Subset track

Written by the Claude Code session that produced submissions `e898c1a` and the
follow-up package in this tree. Everything here is local process evidence, not an
official result.

## What was found

- The promoted Subset frontier (`7aef224a`, 623,518,629) builds from `compute_52`
  PTX with no `-arch`. Two consequences the frontier source did not exploit:
  1. the 64-byte table-record load cannot carry an L2 prefetch-size qualifier, and
  2. the driver JIT-compiles the whole module inside the timed window on every
     ranked run, because a ranked run is a fresh identity with fresh PTX.
- The pinning track's promoted tree already solves both (native sm_89 carrier,
  `.L2::64B` on every GLV table load, no compute_52 module touch). The Subset track
  had neither.

## Measurements that decided the package

| Experiment | Result |
|---|---|
| Cold-JIT cost, `CUDA_CACHE_DISABLE=1`, 60 s arms, harness wall clock | module 69.53 s vs image 60.17 s → the JIT costs **9.36 s** of the window |
| Steady state, 60 s arms OFF/ON/OFF/ON | 1855, 1877, 1836, 1877 verified hits → **+1.71%** |
| Qualifier isolated, native sm_86 both arms, 300 s A/B/B/A | 8981, 8981 vs 9183, 9152 → **+2.08%** |
| Host `cudaLimitMaxL2FetchGranularity=64` | 7335 vs 7286 / 7305 → neutral-to-negative, not the mechanism |
| Hit-set containment, 60 s arms | every hit the module arm found, the image arm found too; 0 missing |

## The bug worth remembering

Redirecting only `tree.cu`'s symbol uploads to the image left
`window_schedule_shared.cuh`'s six uploads writing the module alone. The image's
window schedule stayed zero, the paired epoch `z` was wrong, and the kernel ran at
a *normal* rate while publishing **zero** hits — the failure mode is a rejected run,
not a slow one. `build_carrier.sh` now refuses to emit a header whose image is
missing any of the fourteen globals, and the local wrapper compares device-printed
intermediates between arms when a discrepancy appears.

## What is not claimed

No RTX 4090 measurement exists here. The transfer factor for memory-side changes on
this track is unknown, and the previous submission on this track
(`fd16dfa4`, +1.9% locally) scored 500,784,234 on the ranked 4090 against
623,518,629 — a −19.7% transfer. That is why the package keeps the device change to
one load qualifier, keeps the startup claim separate from the memory claim, and is
accepted or rejected by the official run alone.
