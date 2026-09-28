# Subset: terrapinelf's `a33e04c3` (the `521075fe` device side byte for byte: `QSB_Y_PAIR`, `QSB_Q_MIX` 4, image `003e3d39`; round-9 co-grinder with KH16 + MRG, v3 producers, 9-window table) with the co-grinder grinding 80 of its 158 window patterns (whole block-0 groups, new here)

Prepared with Claude Opus 5.5 in Claude Code. This host has no GPU and no AVX-512. What I did myself is the composition, one host-only co-grinder change, builds and byte comparisons, execution checks under emulation, and an analysis of public ranked artifacts. Everything else in this tree is terrapinelf's `a33e04c3`, byte for byte.

## Summary

- **Base:** terrapinelf's `a33e04c3` (ranked 704.27: GPU 640.18 + co-grinder 64.08), unchanged, except for one host-only change in `CpuGrindSubset.h`.
- **Change (`QSB_CPU_PAT_MINGRP`, default 5):** after `hash_plan`, the co-grinder keeps whole block-0 groups of at least 5 patterns, in the largest prefix whose total is a multiple of 16. That is 80 of its 158 window patterns, in 16 groups. `hash_plan` then runs again on the kept set.
- **Unchanged:** the device code and the native sm_89 image. The rebuilt cubin is byte-identical to `a33e04c3`'s and the promoted `521075fe`'s (`003e3d39…`).
- **Purpose of this draw:** the same host package without this change has three ranked draws (co-grinder parts 64.08, 64.50, 63.90), so this is a clean ranked A/B of the 80-pattern selection.

## Why this base: ranked ground truth (public hit lists)

Every ranked subset run uploads its verified hits. The GPU grinds 128 of the C(13,3) = 286 window patterns and the co-grinder the other 158, so each score splits exactly into a GPU part and a co-grinder part.

**Co-grinder parts by host package.** Grouped by the exact contents of `CpuGrindSubset.h` and `host_producers.h`:

| host package | ranked draws | co-grinder part, M/s |
|---|---|---:|
| terrapinelf `a33e04c3` (KH16 + MRG + round-9 lane + 9-window table + v3 producers) | `a33e04c3` 64.08, `ec9a648e` 64.50, `cb336fbe` 63.90 | **64.16** |
| `a141df2b` co-grinder + v3 producers (the promoted `5c7e36c5` host, via `4da17ebc`) | 6 draws, 61.9–63.7 | 62.7 |
| the same with S16, shift-form Sigmas and 80 patterns (our `4a197f06`) | 1 draw | 63.71 |
| the same co-grinder with the older producers | 3 draws | 60.7 |
| `521075fe`'s own co-grinder | 3 draws | 57.5 |

**GPU work without hit luck.** The GPU walks epochs in order, so the largest epoch among a run's GPU hits, × 128 / 1,200 s, measures the GPU's candidate rate without the Poisson noise of the hit count. I compared each run with the mean of its two previous and two next runs, to remove the card's slowly drifting phase (about ±6 M/s). Over 42 P18 draws from 2026-09-27 05:44 to 2026-09-28 07:26 UTC:

| device side | draws | GPU work minus neighbours, M/s |
|---|---:|---:|
| `QSB_Y_PAIR`, `QSB_Q_MIX` 4 (image `003e3d39`) | 15 | +1.05 ± 0.66 |
| `QSB_Y_PAIR`, `QSB_Q_MIX` 2 (image `f7454842`) | 10 | +0.49 ± 0.63 |
| no `QSB_Y_PAIR` (`d052bc3d` chain) | 17 | −1.27 ± 0.63 |

- `QSB_Y_PAIR` is worth about +2 M/s of GPU part on the ranked card, about 2.5σ.
- `QSB_Q_MIX` 4 leans ahead of 2 by +0.6 ± 0.9. That is not significant, but it points the same way as terrapinelf's luck-free comparison in `a33e04c3`'s note, so this package keeps `a33e04c3`'s `QSB_Q_MIX` 4 and its image.

## The change: 80 patterns

Within an epoch, a candidate's SHA-256 work after the epoch state is block 0 plus five later blocks. Block 0 holds the epoch's 8 buffered bytes and the first 56 bytes of kept window pushes. `hash_plan` already computes block 0 once per group of patterns with the same block-0 bytes. The 286 window patterns split by block 0 into groups of 35, 6 × 15, 21 × 5 and 56 × 1. The GPU's 128 are the 35-group, the six 15-groups and 3 members of one 5-group (the rule in `tree.cu`). The co-grinder's 158 are therefore 20 × 5 + 1 × 2 + 56 × 1: 77 block-0 compressions per epoch for 158 candidates, 0.49 per candidate.

With the change, the co-grinder keeps the first 16 of the twenty 5-groups, 80 patterns, so block 0 costs 16/80 = 0.2 compressions per candidate. The 4-lane SHA-NI groups (20) and 16-lane chunks (5) carry no padding lanes. The per-epoch fixed cost (prefix re-hash, per-group schedules, block 0) drops from about 0.56 to about 0.29 compression-equivalents per candidate. The model gain is about +1% of co-grinder rate before the epoch walk's extra overhead, about +0.5% net. The walk covers about 1.98× more epochs, about 9.5e8 of the 8.2e9 in 1,200 s. The kept patterns are a subset of the complement of the GPU's, so no candidate is ground twice. Every co-grinder hit still passes the exact OpenSSL gate `qsb_hv_check`.

## Checks done here

All builds ran in an amd64 container with CUDA 12.8.93, the ranked runner's toolkit.

| check | result |
|---|---|
| `build_carrier.sh 24` | cubin sha256 `003e3d39b7a6283fa61c3f9d60e2c8560e5b916b6d9abcf43a1f445c4236dc06`, 462,496 bytes, byte-identical to `a33e04c3`'s and `521075fe`'s; 0 spills |
| the harness's build line (`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`) | builds with no errors |
| image `qsb_carrier_knobs` against host `QSB_CARRIER_KNOBS` | identical, so the native image loads |
| selection, replayed in Python on the problem's pushes | 77 groups (56 × 1, 1 × 2, 20 × 5); 80 kept in 16 groups; all disjoint from the GPU's 128 |
| co-grinder under `qemu-x86_64 -cpu max` (SHA-NI + scalar EC), CPU-only harness with tree.cu's loader and 128-pattern rule, `QSB_ZEROS_N=12`, 40 s, against a brute-force `qsb_hv_check` oracle | 80 patterns: 170 = 170 hits over epochs < 4,096, 0 missing, 0 extra, 0 duplicates, all in the kept set. `QSB_CPU_PAT_MINGRP=1`: 302 = 302, exact. No crash. The start line prints `80 of 158 window patterns kept (block-0 groups of >= 5: 16 groups, was 77)` |
| the same selection on another co-grinder lineage (`3ff68d21`'s header) under Intel SDE inside a qemu-system x86_64 guest (8-lane IFMA + 16-lane AVX-512 paths) | exact against the oracle in every mode (88 = 88 hits over 2,048 epochs) |

Caveat: under emulation the 80-pattern build processed about 1.6% fewer candidates in the same time than the 158-pattern build. Emulated scalar EC costs are nothing like Zen 4's, where SHA-256 is a much larger share of the co-grinder's time, so this says little about the ranked host. It does mean the sign of the gain is not settled, which is why this draw is set up as a clean A/B.

## Expected result

- **GPU part:** about 632–644, following the card's phase.
- **Co-grinder part:** about 64–65 if the selection pays, about 63.5–64 if it does not.
- **Total:** about 696–708. The promotion bar is 715.50, so promotion would need an exceptionally fast phase. The co-grinder part of this draw is the measurement that matters.

## Kill switches

- `-DQSB_CPU_PAT_MINGRP=1` (or `QSB_CPU_ALLPAT` in the environment): all 158 co-grinder patterns, which is `a33e04c3` exactly.
- Everything else is as in `a33e04c3`: `-DQSB_CPU_KH16=0`, `-DQSB_CPU_MRG=0`, `-DQSB_CPU_AINL=0`, `-DQSB_CPU_TRY9=0`, `-DQSB_CPU_TAB9_FRAC=0.40`, `-DQSB_CPU_GRIND=0`, `-DQSB_HOST_PRODUCERS=0`; at run time `QSB_HP_BLOCKSYNC=0`, `QSB_HP_PLACE=1`, `QSB_CPU_THREADS_ENV=0`.
- Any device knob change needs `NVCC=<CUDA 12.8 nvcc> ./build_carrier.sh 24` again.

## Reproducing

```
./setup.sh subset
./benchmark.sh subset
```

To get the GPU/CPU split of any ranked run: download its `benchmark-diagnostics-subset-<run>` artifact, count hits per `skip[6:9]`, take the 128 most frequent patterns as the GPU's, and compute rate = hits × 2^23 / `elapsed_s`. For the luck-free GPU work, take the largest lexicographic epoch rank of the first six skip indices among the GPU hits, × 128 / `elapsed_s`.

## Base and attribution

- **terrapinelf** (co-author): this whole package (`a33e04c3`). That covers the round-9 co-grinder lane with KH16 and MRG and the 9-window table gate, the v3 host producers, the three-way `tree.cu` merge, the luck-free `QSB_Q_MIX` comparison, and the earlier lineage (`de5739c9`, `82d8493f`, `92a51c8c`).
- **ercumentyildirim** (co-author): the 9-window gate fraction matched here (`bfe57794`, `a141df2b`, `789aed1b`) and the ranked GPU/co-grinder split method.
- **HyeokxC** (co-author): blocking GPU waits with the host core given to the co-grinder (`0735233a`, `888f5fce`).
- **RealAdii**: the promoted `521075fe` device side and host loop that `a33e04c3` keeps.
- **kshitij-hash**: `QSB_Y_PAIR` and `QSB_SC_PARK` (`8f99a3e9`, promoted in `521075fe`) and the `d052bc3d` composition.
- **fkiene**: the GLV11 P18 chain and the per-warp Q-layout mix (`413f83e7`).
- **jacklightChen**: the promoted `5c7e36c5`.
- **Through the base:** Meganpark980320, newjordan, i34-9, Ryun1, Akashneelesh and every contributor their notes credit. The GPU arithmetic headers derive from VanitySearch (GPLv3, `COPYING`); the co-grinder's field and scalar code follow libsecp256k1 (MIT, `COPYING-secp256k1`).
- **Mine:** the 80-pattern selection, its emulation checks, the luck-free GPU-work comparison across 42 draws, the co-grinder grouping by file contents, and this composition.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
