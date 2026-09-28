# Subset: terrapinelf's `a33e04c3` (the `521075fe` device side byte for byte: `QSB_Y_PAIR`, `QSB_Q_MIX` 4, image `003e3d39`; round-9 co-grinder with KH16 + MRG, v3 producers, 9-window table) with the co-grinder on the 100 patterns of the twenty 5-groups (i34-9's selection) and one contiguous epoch range per worker (new here)

Prepared with Claude Opus 5.5 in Claude Code. This host has no GPU and no AVX-512. What I did myself is the composition, the co-grinder pattern selection and the contiguous epoch walk (both host-only), builds and byte comparisons, execution checks under emulation, and an analysis of public ranked artifacts. Everything else in this tree is terrapinelf's `a33e04c3`, byte for byte.

## Summary

- **Base:** terrapinelf's `a33e04c3` (ranked 704.27: GPU 640.18 + co-grinder 64.08). The only changes are host-only, in `CpuGrindSubset.h`.
- **Change 1, 100 patterns (`QSB_CPU_PAT_MINGRP` 5, `QSB_CPU_PAT_ALIGN` 4):** the co-grinder keeps only the twenty block-0 groups of 5 patterns, 100 of its 158.
  - This is i34-9's selection in `b67487a1`, which extended my 80-pattern selection in `4a197f06`.
  - ercumentyildirim measured both on a rented Zen 4 EPYC 9554 in `86c643ae`: 100 patterns +0.8%, 80 patterns +0.2% over all 158.
  - On ranked, this base with 80 patterns drew 65.97 (`80212db2`) and 64.79 (my `2c00655f`), against 64.16 without them.
- **Change 2, contiguous epochs (`QSB_CPU_EPOCH_CONTIG` 1), new here:** worker t walks one contiguous range of epochs, one epoch at a time, instead of epochs t, t + T, t + 2T, …
  - Consecutive epochs then share a longer prefix, so less of it is re-hashed per epoch.
  - Each epoch's omissions follow from the previous epoch's, with no unrank.
- **Unchanged:** the device code and the native sm_89 image. The rebuilt cubin is byte-identical to `a33e04c3`'s and the promoted `521075fe`'s (`003e3d39…`).

## Why this base: ranked ground truth (public hit lists)

Every ranked subset run uploads its verified hits. The GPU grinds 128 of the C(13,3) = 286 window patterns and the co-grinder the other 158, so each score splits exactly into a GPU part and a co-grinder part.

**Co-grinder parts by host package**, grouped by the exact contents of `CpuGrindSubset.h` and `host_producers.h`. These are the 45 ranked runs from 2026-09-27 13:04 to 2026-09-28 09:45 UTC:

| host package | co-grinder patterns | ranked draws | co-grinder part, M/s |
|---|---:|---|---:|
| terrapinelf `a33e04c3` + 80-pattern selection (`80212db2` also with `QSB_R_CBANK` 1) | 80 | `80212db2` 65.97, my `2c00655f` 64.79 | 65.38 |
| terrapinelf `a33e04c3` (KH16 + MRG + round-9 lane + 9-window table + v3 producers) | 158 | `a33e04c3` 64.08, `ec9a648e` 64.50, `cb336fbe` 63.90 | **64.16** |
| ercumentyildirim's `a141df2b` line + terrapinelf's KH16/MRG + 100 patterns | 100 | `0cdb0861` 64.12, `ca9b2e27` 63.93 | 64.03 |
| `a141df2b` line + 100 patterns (two other packages) | 100 | `b67487a1`, `78691035`, `86c643ae`, `85caa156` | 63.33–64.08 |
| `a141df2b` co-grinder + v3 producers (the promoted `5c7e36c5` host) | 158 | 10 draws, 61.83–65.13 | 63.06 (sd 0.95) |
| `521075fe`'s own co-grinder | 158 | 7 draws | 57.73 (sd 0.77) |

- **Draw-to-draw spread:** one byte-identical host package drew co-grinder parts from 61.83 to 65.13 over 10 draws (sd 0.95 M/s). The Poisson part of that is about 0.66 M/s, so the ranked host adds its own run-to-run spread.
- **What that means for single draws:** one draw is worth about ±1 M/s. The two 80-pattern draws average +1.2 M/s over `a33e04c3`, but with that spread the Zen 4 measurements (+0.2% on ercumentyildirim's EPYC, +1.0% on terrapinelf's Ryzen) are the better estimates.

**GPU work without hit luck.** The GPU walks epochs in order, so the largest epoch among a run's GPU hits, × 128 / 1,200 s, measures the GPU's candidate rate without the Poisson noise of the hit count. I compared each run with the mean of its two previous and two next runs, to remove the card's slowly drifting phase (about ±6 M/s). Over 42 P18 draws from 2026-09-27 05:44 to 2026-09-28 07:26 UTC:

| device side | draws | GPU work minus neighbours, M/s |
|---|---:|---:|
| `QSB_Y_PAIR`, `QSB_Q_MIX` 4 (image `003e3d39`) | 15 | +1.05 ± 0.66 |
| `QSB_Y_PAIR`, `QSB_Q_MIX` 2 (image `f7454842`) | 10 | +0.49 ± 0.63 |
| no `QSB_Y_PAIR` (`d052bc3d` chain) | 17 | −1.27 ± 0.63 |

- `QSB_Y_PAIR` is worth about +2 M/s of GPU part on the ranked card, about 2.5σ.
- `QSB_Q_MIX` 4 leans ahead of 2 by +0.6 ± 0.9. That is not significant, but it points the same way as terrapinelf's luck-free comparison in `a33e04c3`'s note, so this package keeps `a33e04c3`'s `QSB_Q_MIX` 4 and its image.
- The card's phase decorrelates within about 1 h (lag-1 autocorrelation 0.64 at 27.5 min, 0.15 at 1.4 h). With a queue of several hours, no submission can aim at a phase.

## The changes

**Where the co-grinder's per-epoch cost goes.** Within an epoch, a candidate's SHA-256 work after the epoch state is block 0 plus five later blocks. Block 0 holds the epoch's 8 buffered bytes and the first 56 bytes of kept window pushes. `hash_plan` computes block 0 once per group of patterns with the same block-0 bytes.

- **Groups:** by block 0, the 286 window patterns split into groups of 35, 6 × 15, 21 × 5 and 56 × 1.
- **GPU's 128:** the 35-group, the six 15-groups and 3 members of one 5-group (the rule in `tree.cu`).
- **Co-grinder's 158:** 20 × 5 + 1 × 2 + 56 × 1.
- **Other per-epoch work:** each epoch unranks its early omissions and re-hashes the part of its prefix (the kept early pushes) that differs from the worker's previous epoch.

**100 patterns.** The twenty 5-groups keep block 0 at 0.2 compressions per candidate instead of 0.49. The per-epoch work is also shared by 100 candidates instead of 80.

- **Why a multiple of 4 is enough:** this lane hashes the tails 4 lanes wide (SHA-NI) and batches its 16-lane key-hash schedules by key, not by pattern. My earlier `4a197f06` required a multiple of 16 and so kept only 80.
- **Measured:** on ercumentyildirim's EPYC 9554, 100 patterns ran +0.8% and 80 ran +0.2% over 158 (their `a141df2b` line).

**Contiguous epoch ranges.** Worker t now walks [base + t·span, base + (t+1)·span) one epoch at a time. Here span is the epoch space above the diagnostic base divided among the workers: about 1.9e8 epochs each at 30 workers, against about 2.6e7 walked in 1,200 s.

- **Prefix re-hash:** in lexicographic order, consecutive epochs differ in the last omission. I replayed the walk on this problem's prefix: the re-hash drops from about 6.1 SHA-256 blocks per epoch at stride 30 to about 3.4.
- **No unrank:** the next epoch's omissions come from the previous epoch's by the usual next-combination step. The binomial unrank runs about 100–130 dependent iterations per epoch, and it now runs only at the start of each range.
- **Disjointness:** the ranges are disjoint and the kept patterns are a subset of the complement of the GPU's, so no candidate is ground twice. Every co-grinder hit still passes the exact OpenSSL gate `qsb_hv_check`.
- **Diagnostic:** worker 0 still starts at the base, so the smallest co-grinder hit still carries the diagnostic code.

**Expected gain.** Change 1 is about +0.8% of co-grinder rate (measured on an EPYC 9554 by ercumentyildirim). Change 2 removes about 3 SHA-256 blocks and one unrank per epoch, a few hundred cycles shared by 100 candidates. That is about +0.3–0.5% of co-grinder rate. I estimated it and did not measure it: I have no Zen 4 host. Together they add about 0.7 M/s of co-grinder part.

## Checks done here

All builds ran in an amd64 container with CUDA 12.8.93, the ranked runner's toolkit.

- **Emulation harness:** the co-grinder ran under `qemu-x86_64 -cpu max` (SHA-NI + scalar EC) in a CPU-only harness with tree.cu's loader and 128-pattern rule.
- **Settings:** `QSB_ZEROS_N=12`, 40 s per run.
- **Reference:** each run was compared against a brute-force `qsb_hv_check` oracle.

| check | result |
|---|---|
| `build_carrier.sh 24` | cubin sha256 `003e3d39b7a6283fa61c3f9d60e2c8560e5b916b6d9abcf43a1f445c4236dc06`, 462,496 bytes, byte-identical to `a33e04c3`'s and `521075fe`'s; 0 spills. Only the informational source hash in the header changes |
| the harness's build line (`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`) | builds with no errors |
| image `qsb_carrier_knobs` against the host binary's `QSB_CARRIER_KNOBS` | byte-identical (1,794 bytes), so the native image loads |
| selection, replayed in Python on the problem's pushes | 77 groups (56 × 1, 1 × 2, 20 × 5); 100 kept in 20 groups; all disjoint from the GPU's 128 |
| next-combination step against the binomial unrank, 20,000,000 consecutive epochs from 400 starts (including the first and last epochs) | 0 mismatches; carries at every index 0..5 exercised |
| 1 thread, 100 patterns | all hits over epochs < 4,096 match the oracle (199 = 199), 0 missing, 0 extra, 0 duplicates, all in the kept set. The start line prints `100 of 158 window patterns kept (block-0 groups of >= 5: 20 groups, was 77)` |
| 4 threads, ranges capped to 4 × 1,024 epochs (`-DQSB_CPU_EPOCH_CAP=4096`) | every worker stops at its range end: exactly 409,600 candidates, 199 = 199 hits, 0 duplicates |
| 4 threads, full ranges | epochs < 4,096 (worker 0): exact. Worker 2's first 2,048 epochs (from 4,109,236,362): 87 = 87 against a separate oracle run. 0 duplicates |
| `-DQSB_CPU_EPOCH_CONTIG=0`, 4 threads | exact (the old stride walk) |
| `-DQSB_CPU_PAT_ALIGN=16` / `QSB_CPU_ALLPAT` | 80 patterns: 170 = 170; all 158: 302 = 302 |

Emulated timings say nothing about Zen 4, so I did not measure speed here.

## Expected result

- **GPU part:** about 632–644, following the card's phase.
- **Co-grinder part:** about 64.5–65.5, with a single-draw spread of about ±1.
- **Total:** about 697–710. The promotion bar is 715.50, so promotion needs a fast phase.

## Kill switches

- `-DQSB_CPU_EPOCH_CONTIG=0`: the stride walk of `a33e04c3`.
- `-DQSB_CPU_PAT_ALIGN=16`: 80 patterns. Together with `-DQSB_CPU_EPOCH_CONTIG=0`, this is `2c00655f` exactly.
- `-DQSB_CPU_PAT_MINGRP=1` (or `QSB_CPU_ALLPAT` in the environment): all 158 co-grinder patterns.
- Everything else is as in `a33e04c3`:
  - build: `-DQSB_CPU_KH16=0`, `-DQSB_CPU_MRG=0`, `-DQSB_CPU_AINL=0`, `-DQSB_CPU_TRY9=0`, `-DQSB_CPU_TAB9_FRAC=0.40`, `-DQSB_CPU_GRIND=0`, `-DQSB_HOST_PRODUCERS=0`;
  - run time: `QSB_HP_BLOCKSYNC=0`, `QSB_HP_PLACE=1`, `QSB_CPU_THREADS_ENV=0`.
- Any device knob change needs `NVCC=<CUDA 12.8 nvcc> ./build_carrier.sh 24` again.

## Reproducing

```
./setup.sh subset
./benchmark.sh subset
```

To get the GPU/CPU split of any ranked run:

1. Download its `benchmark-diagnostics-subset-<run>` artifact.
2. Count hits per `skip[6:9]`. The 128 most frequent patterns are the GPU's.
3. Compute rate = hits × 2^23 / `elapsed_s`.
4. For the luck-free GPU work, take the largest lexicographic epoch rank of the first six skip indices among the GPU hits, × 128 / `elapsed_s`.

## Base and attribution

- **terrapinelf** (co-author): this whole package (`a33e04c3`), and the first ranked draw of the 80-pattern selection (`80212db2`). That covers the round-9 co-grinder lane with KH16 and MRG and the 9-window table gate, the v3 host producers, the three-way `tree.cu` merge, the luck-free `QSB_Q_MIX` comparison, and the earlier lineage (`de5739c9`, `82d8493f`, `92a51c8c`).
- **i34-9** (co-author): the 100-pattern selection (`b67487a1`).
- **ercumentyildirim** (co-author): the EPYC 9554 measurements of the pattern sets and batch sizes (`86c643ae`, `0cdb0861`), the 9-window gate fraction matched here (`bfe57794`, `a141df2b`, `789aed1b`) and the ranked GPU/co-grinder split method.
- **HyeokxC** (co-author): blocking GPU waits with the host core given to the co-grinder (`0735233a`, `888f5fce`).
- **RealAdii**: the promoted `521075fe` device side and host loop that `a33e04c3` keeps.
- **kshitij-hash**: `QSB_Y_PAIR` and `QSB_SC_PARK` (`8f99a3e9`, promoted in `521075fe`) and the `d052bc3d` composition.
- **fkiene**: the GLV11 P18 chain and the per-warp Q-layout mix (`413f83e7`).
- **jacklightChen**: the promoted `5c7e36c5`.
- **Through the base:** Meganpark980320, newjordan, Ryun1, Akashneelesh and every contributor their notes credit. The GPU arithmetic headers derive from VanitySearch (GPLv3, `COPYING`); the co-grinder's field and scalar code follow libsecp256k1 (MIT, `COPYING-secp256k1`).
- **Mine:**
  - Co-grinder: the block-0 pattern selection (80 in `4a197f06`, here with a multiple of 4), the contiguous epoch walk with its next-combination step, and their emulation checks.
  - Analysis: the luck-free GPU-work comparison across 42 draws, and the co-grinder grouping by file contents across 45 draws.
  - This composition.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
