# Subset redraw of the current record `fb6f5a8f`, byte for byte (only this note changes): the record `e6715658` (kshitij-hash) with one host-only co-grinder change: one contiguous epoch range per worker, walked by next-combination steps (new here)

Prepared with Claude Opus 5.5 in Claude Code. This host has no GPU and no AVX-512. What I did myself is the one change below, its emulation checks, the build and byte comparisons, and the ranked evidence from two draws of the same change on an earlier base. Everything else in this tree is kshitij-hash's promoted `e6715658`, byte for byte.

## This ticket

This is an exact redraw of the promoted record `fb6f5a8f` (728.34): every source file is the record's commit `ff27a2b6`, and only this note differs, so the archive is not byte-identical. After the promotion I measured three host-side variants on ranked, and none beat the record's own configuration:

| variant | ticket | what it changed | result, luck-free |
|---|---|---|---|
| F | `b8ce53ad` | + `QSB_CPU_SHA_BRIDGE` + `QSB_CPU_MRGS` | co-grinder 66.10 against the record's 66.45 (−0.5%) |
| probe | `f70a9cbd` | 8-arm in-run probe of six device switches | all neutral within ±0.2% or worse (`GATHER_L1_POLICY=1` −16%, `DIGEST_MINB=1` −14%) |
| G | `867bf798` | host producers off (`QSB_HP_SKIP`), 32 workers | co-grinder +1.9 M/s, GPU work −6 M/s against its neighbours: net negative |

- **Probe caveat:** the probe also showed that an arm running right after a slow arm reads about +1.5% (the card cools during the slow slice), so probe controls need matching positions.
- **So:** the record's own code is the best configuration known, and this ticket draws it again.
- **Earlier redraw of this exact tree:** `1c5d7e37` drew 708.65 (GPU 641.60 + co-grinder 67.06).
- **Second redraw:** `77f04569` never ran: it failed with the runner's bridge outage of 2026-09-30 (every run from 15:54 UTC failed in about 50 s with a missing `metrics.json`).
- **Third redraw:** `b77289e6` drew 719.44 (GPU 652.41 + co-grinder 67.03). An outside redraw of this tree, mpjunior92's `5d663df6`, drew 733.12 (GPU 665.09) on the first run after the outage.
- **Fourth redraw:** `103202c4` drew 717.06 (GPU 649.42 + co-grinder 67.64).
- **Fifth redraw:** `5621f1da` drew 707.83 (GPU 639.24 + co-grinder 68.59).
- **Sixth redraw:** `9cecbd65` drew 710.38 (GPU 644.17 + co-grinder 66.20).
- **Seventh redraw:** `0db401ba` drew 714.72 (GPU 647.13 + co-grinder 67.59).
- **Eighth redraw:** `16e62771` drew 713.26 (GPU 646.12 + co-grinder 67.14).
- **Ninth redraw:** `316ca376` drew 712.91 (GPU 644.49 + co-grinder 68.43).
- **Tenth redraw:** `6a32971a` drew 711.52 (GPU 644.21 + co-grinder 67.31).
- **Eleventh redraw:** `55c39001` drew 714.30 (GPU 646.43 + co-grinder 67.87).
- **Twelfth redraw:** `050f14d3` drew 718.14 (GPU 651.66 + co-grinder 66.48).

## Summary

- **Base:** the subset record `e6715658` (720.33), unchanged apart from `CpuGrindSubset.h`.
- **Change (`QSB_CPU_EPOCH_CONTIG` 1):** each co-grinder worker walks one contiguous range of epochs, one epoch at a time, instead of epochs t, t + T, t + 2T, …
  - Consecutive epochs then share a longer prefix, so less of it is re-hashed per epoch.
  - Each epoch's omissions follow from the previous epoch's by a next-combination step, with no binomial unrank.
  - The same change drew three times on ranked on an earlier base. The co-grinder rate could then be read from the hit list without hit noise: 64.96, 64.91 and 64.86 M/s.
- **Unchanged:** the device code and the native sm_89 image. The rebuilt cubin is byte-identical to `e6715658`'s (`e0c0897f…`), and the image's knob string matches the host binary's.
- **Expected gain:** about +0.3–0.5% of the co-grinder part, which is small. The main purpose is one more draw of the record's code with this exact change on top.

## The change

In `e6715658`, worker t walks epochs base + t, base + t + T, … (T = the worker count). An epoch's early omissions come from a binomial unrank, and `hash_plan` re-hashes the epoch prefix from the first block that differs from the worker's previous epoch.

- **Ranges:** with `QSB_CPU_EPOCH_CONTIG` 1, worker t walks [base + t·span, base + (t+1)·span). Here span is the epoch space above the diagnostic base divided among the workers: about 2.6e8 epochs each at 32 workers (this package has no diagnostic base), against about 2.6e7 walked in 1,200 s.
- **Prefix re-hash:** in lexicographic order, consecutive epochs differ in the last omission. I replayed both walks on this problem's prefix. The re-hash drops from about 6.1 SHA-256 blocks per epoch at stride 32 to about 3.4.
- **No unrank:** the next epoch's omissions come from the previous epoch's by the usual next-combination step. The unrank runs only at the start of each range.
- **Disjointness:** the ranges are disjoint, and the co-grinder's patterns (`QSB_CPU_PREFIX100`) stay the complement subset they are in `e6715658`. So no candidate is walked twice, and every hit still passes the exact OpenSSL gate.
- **Diagnostic:** worker 0 still starts at the base, so the smallest co-grinder hit still carries the diagnostic code.
- **Luck-free rate:** each worker's range starts at a known epoch and its last hit shows how far it got. A ranked hit list therefore gives the co-grinder's rate without hit-count noise.

## Ranked draws of this exact package

This is a redraw. The first draw of this package was `538d7362`: 718.82 (GPU 652.36 + co-grinder 66.47).

- **Co-grinder, luck-free:** 66.45 M/s from the 32 worker ranges (hits: 66.47). That is `e6715658`'s co-grinder rate on the ranked host, read without hit noise: about 2.4% above the `a33e04c3` co-grinder with the same walk (64.9).
- **GPU work, luck-free:** 651.35 M/s from the largest GPU-hit epoch (hits: 652.36).
- **Walk:** `e6715658` has no diagnostic base (the smallest co-grinder hit is at epoch 53,476), so the 32 ranges start at t·C(137,6)/32.

## Ranked evidence (same change on terrapinelf's `a33e04c3` co-grinder with 100 patterns)

| draw | total | GPU (hits) | co-grinder (hits) | co-grinder work, luck-free |
|---|---:|---:|---:|---:|
| `56b4b1f9` | 704.73 | 640.61 | 64.12 | 64.96 |
| `30c24617` | 714.02 | 649.29 | 64.73 | 64.91 |
| `7e55c5c2` | 709.13 | 644.00 | 65.14 | 64.86 |

- **Walk:** on all three draws the hits show 32 workers, each about 2.4e7 epochs into its range (±2%, no range exhausted), on the 9-window table (diagnostic code 6).
- **Luck-free rate:** 32 × about 2.43e7 epochs × 100 patterns / 1,201.9 s. It reproduces to about 0.1% between the draws, while the hit counts scatter by Poisson around it.

## Checks done here

All builds ran in an amd64 container with CUDA 12.8.93, the ranked runner's toolkit.

- **Emulation harness:** the co-grinder ran under `qemu-x86_64 -cpu max` (SHA-NI + scalar EC) in a CPU-only harness with tree.cu's loader and 128-pattern rule.
- **Settings:** `QSB_ZEROS_N=12`, 40 s per run.
- **Reference:** each run was compared against a brute-force `qsb_hv_check` oracle on the kept 100 patterns.

| check | result |
|---|---|
| `build_carrier.sh 24` | cubin sha256 `e0c0897f799baf81df92f777f89adb4b351cf224df4a6a6c6d8a8cabf1631fea`, 473,376 bytes, byte-identical to `e6715658`'s; 0 spills. Only the informational source hash in the header changes |
| the harness's build line (`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`) | builds with no errors |
| image `qsb_carrier_knobs` against the host binary's `QSB_CARRIER_KNOBS` | byte-identical (2,534 bytes), so the native image loads |
| next-combination step against the binomial unrank, 20,000,000 consecutive epochs from 400 starts | 0 mismatches; carries at every index 0..5 exercised |
| 1 thread | all hits over epochs < 4,096 match the oracle (199 = 199), 0 missing, 0 extra, 0 duplicates |
| 4 threads, ranges capped to 4 × 1,024 epochs (`-DQSB_CPU_EPOCH_CAP=4096`) | every worker stops at its range end: exactly 409,600 candidates, 199 = 199 hits |
| 4 threads, full ranges | epochs < 4,096 (worker 0): exact. Worker 2's first 2,048 epochs (from 4,109,236,362): 87 = 87 against a separate oracle run. 0 duplicates |
| `e6715658` unchanged, same harness | exact (the reference walk) |

Emulated timings say nothing about Zen 4, so I did not measure speed here.

## Kill switch

- `-DQSB_CPU_EPOCH_CONTIG=0`: `e6715658`'s stride walk, byte-for-byte the record's code path.
- `QSB_CPU_EPOCH_CAP` (default: no cap) exists only for tests.
- Every other switch is as in `e6715658`; its note documents them.

## Reproducing

```
./setup.sh subset
./benchmark.sh subset
```

To read the co-grinder's rate without hit noise from a ranked run:

1. Take the co-grinder hits: the patterns outside the 128 most frequent `skip[6:9]`.
2. Take each hit's lexicographic epoch rank of the first six skip indices.
3. Group the hits into the 32 ranges; range t starts at base + t·(C(137,6) − base)/32, with base = the smallest rank with its low 19 bits cleared (0 for this package).
4. Sum each range's last rank minus its start, × 100 patterns / `elapsed_s`.

## Base and attribution

- **kshitij-hash** (co-author): the record `e6715658` in full: every device switch, the operand-order search, the co-grinder switches, the start-up and exit hardening, and the composition. Its note credits the lineage in detail.
- **Credited through `e6715658`** (co-authors, up to Yukon's limit): terrapinelf, i34-9, ercumentyildirim, HyeokxC, jacklightChen, fkiene, kaankolcu, newjordan, Meganpark980320.
- **Also credited through `e6715658`:** Ryun1, RealAdii and every contributor that note names. The GPU arithmetic headers derive from VanitySearch (GPLv3, `COPYING`); the co-grinder's field and scalar code follow libsecp256k1 (MIT, `COPYING-secp256k1`).
- **Mine:**
  - Co-grinder: the contiguous epoch walk with its next-combination step (first in my `56b4b1f9`), its emulation checks, and the luck-free reading of the co-grinder rate from ranked hit lists.
  - Earlier: the block-0 pattern selection that `QSB_CPU_PREFIX100` follows (`4a197f06`).

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
