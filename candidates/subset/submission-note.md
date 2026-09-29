# Subset: the record `fb6f5a8f` with its host producers off (ercumentyildirim's `QSB_HP_SKIP`) and all 32 CPUs given to the co-grinder

Prepared with Claude Opus 5.5 in Claude Code. This host has no GPU. The device side and the native image are the record's, byte for byte; two host-only lines change.

## Summary

- **Base:** the current subset record `fb6f5a8f` (728.34) = kshitij-hash's `e6715658` + my contiguous co-grinder epoch walk.
- **Change 1, `QSB_HP_SKIP` 1** (ercumentyildirim, PR 2441 / `dea321f0`): the host producers are not started. The GPU producers build every batch on the slot stream, the fallback the search loop already takes whenever a host-built batch is not ready. The co-grinder keeps the producers' CPU time.
- **Change 2, `QSB_CPU_RESERVE` 0:** with the producers on, the main thread is narrowed to one CPU and the co-grinder gets the other 31 plus the main thread's (32). With them off, the main thread keeps its full set, so the count falls to ncpu − `QSB_CPU_RESERVE` = 30 (the hit lists of `dea321f0` and `f70a9cbd` show exactly 30 worker ranges). The main thread sleeps in blocking waits and every worker is `SCHED_IDLE`, so this package reserves nothing: 32 workers.
- **Unchanged:** the device code and the native sm_89 image (cubin `e0c0897f…`, byte-identical; knob strings identical).

## Ranked evidence

Every package with the contiguous walk shows its co-grinder rate without hit noise (per-worker ranges, see below):

| draw | host producers | workers | co-grinder, luck-free (M/s) |
|---|---|---:|---:|
| `538d7362` (this record's code) | on | 32 | 66.45 |
| `dea321f0` (ercumentyildirim) | off | 30 | 67.98 |
| `f70a9cbd` (my probe) | off | 30 | 68.26 |

- **CPU:** with the producers off, each worker walks about 9% more epochs, so the 30 workers deliver +1.5 to +1.8 M/s over the 32 with producers. Two more workers should add about another +1 to +2 M/s.
- **GPU:** the cost of GPU-built batches is not resolved by a single draw. `dea321f0` read 639.4 M/s of GPU work against 640.7 for its neighbours, inside the card's phase drift. This draw measures the co-grinder exactly and adds one GPU reading.

## The record's contiguous walk (unchanged)


In `e6715658`, worker t walks epochs base + t, base + t + T, … (T = the worker count). An epoch's early omissions come from a binomial unrank, and `hash_plan` re-hashes the epoch prefix from the first block that differs from the worker's previous epoch.

- **Ranges:** with `QSB_CPU_EPOCH_CONTIG` 1, worker t walks [base + t·span, base + (t+1)·span). Here span is the epoch space above the diagnostic base divided among the workers: about 2.6e8 epochs each at 32 workers (this package has no diagnostic base), against about 2.6e7 walked in 1,200 s.
- **Prefix re-hash:** in lexicographic order, consecutive epochs differ in the last omission. I replayed both walks on this problem's prefix. The re-hash drops from about 6.1 SHA-256 blocks per epoch at stride 32 to about 3.4.
- **No unrank:** the next epoch's omissions come from the previous epoch's by the usual next-combination step. The unrank runs only at the start of each range.
- **Disjointness:** the ranges are disjoint, and the co-grinder's patterns (`QSB_CPU_PREFIX100`) stay the complement subset they are in `e6715658`. So no candidate is walked twice, and every hit still passes the exact OpenSSL gate.
- **Diagnostic:** worker 0 still starts at the base, so the smallest co-grinder hit still carries the diagnostic code.
- **Luck-free rate:** each worker's range starts at a known epoch and its last hit shows how far it got. A ranked hit list therefore gives the co-grinder's rate without hit-count noise.

## Ranked draws of the record's code

The record's code drew `538d7362`: 718.82 (GPU 652.36 + co-grinder 66.47), and `fb6f5a8f`: 728.34 (GPU 662.35 + co-grinder 65.99), promoted.

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

## Kill switches

- `-DQSB_HP_SKIP=0` (or delete the `#define` in `subset.cu`): the host producers run as in the record.
- `-DQSB_CPU_RESERVE=2`: 30 workers when the producers are off.
- `-DQSB_CPU_EPOCH_CONTIG=0`: `e6715658`'s stride walk.

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

- **ercumentyildirim** (co-author): `QSB_HP_SKIP` (PR 2441, `dea321f0`) and the first ranked draw with the producers off.
- **kshitij-hash** (co-author): the record `e6715658` in full: every device switch, the operand-order search, the co-grinder switches, the start-up and exit hardening, and the composition. Its note credits the lineage in detail.
- **Credited through `e6715658`** (co-authors, up to Yukon's limit): terrapinelf, i34-9, HyeokxC, jacklightChen, fkiene, kaankolcu, newjordan, Meganpark980320.
- **Also credited through `e6715658`:** Ryun1, RealAdii and every contributor that note names. The GPU arithmetic headers derive from VanitySearch (GPLv3, `COPYING`); the co-grinder's field and scalar code follow libsecp256k1 (MIT, `COPYING-secp256k1`).
- **Mine:**
  - Co-grinder: the contiguous epoch walk with its next-combination step (first in my `56b4b1f9`), its emulation checks, and the luck-free reading of the co-grinder rate from ranked hit lists.
  - Earlier: the block-0 pattern selection that `QSB_CPU_PREFIX100` follows (`4a197f06`).

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
