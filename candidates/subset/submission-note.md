# Subset: SHA-NI double-hash handoff and host-producer scheduling on the latest promoted frontier

Model: GPT 6 Sol. Effort: medium. Harness: Codex.

This submission starts from the promoted Subset source `ff27a2b66990a3eb554a1d4453e896c0397337ba` ([PR2419](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/2419), cefika), whose official score is 728,337,167 verified candidates/s. The two source changes are the CPU SHA-256d bridge in `CpuGrindSubset.h` and `QSB_HOST_PRODUCERS=0` at the existing candidate entry point. The GPU device routines and native sm_89 image, continuous epoch walk, field arithmetic, candidate ownership, and exact-hit verifier are inherited unchanged. No score is claimed for this new composition before the official remote run.

The bridge is the same independently developed mechanism as our completed [PR2385](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/2385), represented by [commit aa937fc](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/aa937fc0aaca2408894dd0d294fc9940732b4d49). That earlier package officially scored 720,739,773 against its actual reference 720,332,123, recomputed +0.056592%. It was measured on the preceding promoted source, so that small whole-package result is a reason to test this composition, not a prediction of its performance on cefika's new frontier. Our later CPU-bridge plus GPU-divstep bundle [PR2427](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/2427) scored 710,092,271 against its 720,332,123 reference, a -1.4215% whole-package result; this submission does **not** carry its GPU divstep switch or replacement native image. That result cannot identify which component caused the loss.

The second change is a separate host scheduling experiment informed only by the public description of in-flight [PR2441](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/2441); none of its in-flight source was accessed. The promoted `QSB_HOST_PRODUCERS` kill switch already provides GPU-only batch construction. Setting it to 0 avoids starting the host producer threads and leaves each batch to the existing GPU fallback producer path, potentially returning CPU time to the co-grinder. PR2441 reports probe-based CPU gains and a small possible GPU loss; these are not an official result for this switch or for our combination. The native image knob list does not include this host-only switch, so this package retains the promoted native image. This is a new combined remote experiment, not an assertion that disabling producers is faster.

The promoted four-lane planned SHA path already computes each first SHA-256 digest in SHA-NI chaining registers. Previously it stores each digest into a four-by-16-word scratch array; the fixed 32-byte second SHA immediately reloads those words into its message ring. `QSB_CPU_SHA_BRIDGE=1` converts the same register pairs into the existing four-lane second-block ring, fills the unchanged fixed padding and length words, and calls the unchanged four-lane compression/output routines. It removes that intermediate digest store/reload in source, but actual machine traffic, register spills and instruction-cache effects are unknown. `QSB_CPU_SHA_BRIDGE=0` retains the old route. The change is guarded by the existing `QSB_CPU_SHC` and `QSB_CPU_SHA4` switches, so other configurations retain their old path.

The identical register-word mapping and fixed 32-byte SHA padding were checked again by a pure-Python 2,000×4-lane audit. For this rebased source, the CPU patch applied cleanly to `ff27a2`; the producer switch selects the already-existing GPU fallback. The exact-hit gate `gate_publish_exact` still calls `qsb_hv_check`, and static scope/diff checks are recorded with this submission. These checks establish source scope and data mapping; the official remote verifier is the first actual execution of this exact composition. No local C++ or CUDA compilation and no local GPU benchmark were run. Public in-flight submissions were screened from their descriptions only; no in-flight source was obtained. The `QSB_CPU_MRGS` single-switch [PR2383](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/2383) scored -0.863352% against its actual reference and was not added; GPU-side in-flight proposals require matching native images and were not imported; the host-producer change remains unverified until this remote run.

The SHA bridge was developed by jacklightChen with GPT 6 Sol/Codex. The host-producer scheduling cue comes from ercumentyildirim's public PR2441 description, independently implemented here using the promoted source's pre-existing kill switch. The initial read-only research cue was the user's Pinning staging candidate `b0a450b`, which had no official throughput result at the time; its 16-lane AVX-512 software SHA and locktime packing were not imported. The new promoted contiguous-epoch base belongs to cefika, built on kshitij-hash's promoted SHA and device work and the earlier contributions named below. The existing GPL and secp256k1 licenses and all source notices remain. Credit here records research lineage, not participation in this submission; no other co-author field is used.

The remainder of this note is the inherited PR2419 historical explanation of the contiguous-epoch base. Its phrases such as “new here,” old score, build claims and test environment refer to that prior promoted package, not to this SHA-bridge composition or a new local measurement.

---

# Historical inherited note: the e6715658 base with cefika's contiguous-epoch change

Prepared with Claude Opus 5.5 in Claude Code. This host has no GPU and no AVX-512. What I did myself is the one change below, its emulation checks, the build and byte comparisons, and the ranked evidence from two draws of the same change on an earlier base. Everything else in this tree is kshitij-hash's promoted `e6715658`, byte for byte.

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
