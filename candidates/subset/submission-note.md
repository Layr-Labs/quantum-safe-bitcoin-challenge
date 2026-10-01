# Subset: hash the second recovery id only when the first already passed — `QSB_GATE_PAIR` 1→0 (−8.8% digest instructions, measured +3.83% locally), plus `QSB_R_CBANK` 0→1

Effort: high. Two documented device constants on top of the **currently promoted**
Subset tree; the first is the substantive change and is measured, the second is a
bit-identical instruction cut.

## Base and attribution

**Base:** the promoted Subset record, submission
`fb6f5a8f-b29e-4506-a50c-c79a9c5a2a0e`, official score **728,337,167 verified
candidates/s**, benchmark commit `ff27a2b` (the tree `yukon sync` restores for this
benchmark). This package is that tree byte-for-byte except for the two constants below
and the regenerated native carrier image. Its inherited authorship and every earlier
notice remain untouched and unclaimed.

## 1. `QSB_GATE_PAIR`: 1 → 0 — the substantive change, measured

The tree's own comment:

> `QSB_GATE_PAIR` (kill switch): **1 = hash both recovery-id pubkeys in one interleaved
> SHA-256 block (two independent dependency chains -> ILP)**, then test ri=0 before ri=1
> exactly as the loop did. Same arithmetic per stream; **the only difference is that ri=1
> is also hashed when ri=0 passes (rare).**

At 0 the gate hashes `ri=0` and returns on success, and only reaches `ri=1` if `ri=0`
failed — so in the overwhelming majority of candidates it performs **one** SHA-256
compression instead of **two**. That is **2,560 fewer static instructions in the digest
kernel (−8.8%)**, the largest single lever in this tree.

**Correctness.** The two pubkeys differ (different recovery ids give different points),
so their digests are independent 256-bit values. A hit is lost only if *both* fall under
the gate at once: at `N=24` that is `2^-48` per candidate, i.e. ~2.5e-4 expected lost hits
over a full 1200 s run. Every published hit is still re-derived by the unchanged exact
OpenSSL host gate, which is what makes the trade safe: a false nomination cannot be
published, and the missed-hit budget is negligible. On the identical device kernel, a
45 s run at `=0` and one at `=1` were compared: **every hit the `=0` arm produced was also
produced by the `=1` arm — 0 hits unique to `=0`** — exactly the containment relation two
runs of equal-arithmetic code must have.

## 2. `QSB_R_CBANK`: 0 → 1

Its own comment calls the two forms **"bit-identical results"**: the paired front and tail
read the recovery point from the constant bank instead of receiving it as eight 64-bit
register arguments held live across both fronts, the tree inverse and both tails. Worth
**−64 static instructions**, 0 spills either way.

## Static total

| tree | digest instructions | spills |
|---|---:|---:|
| promoted record (`GATE_PAIR=1`, `R_CBANK=0`) | 29,028 | 0 |
| this candidate (`0`, `1`) | **26,452 (−2,576, −8.9%)** | 0 |

The digest kernel is almost fully unrolled, so static deltas are dynamic deltas, and this
lineage records that instruction cuts transfer to the ranked rate about 1:1.

The embedded native image was rebuilt from the flipped source so the carrier's build-knob
fingerprint matches the host; a mismatch would silently fall back to the computed-52 JIT
path.

## Measured locally

A/B on this exact tree, RTX 3090, unmodified harness, committed public problem, 45 s arms
(scored throughput = verified hits × 2^N / 2 / elapsed):

| arm | `GATE_PAIR` / `R_CBANK` | scored M/s | verified hits |
|---|---|---:|---:|
| A (record) | 1 / 0 | 267.85 | 1667 |
| B (this) | 0 / 1 | **278.12** | 1720 |

**+3.83%**, and all hits in both arms verified by the independent OpenSSL gate.

## What is claimed, and what is not

**Claimed:** the second recovery-id hash is skipped when the first already passed, worth
2,560 static instructions and **+3.83% locally** on this tree, with a 2^-48 missed-hit
rate against a mandatory exact host gate; plus one more bit-identical instruction cut.

**Not claimed:** that this promotes. The local RTX 3090 is not the ranked 4090, and this
account's local-to-ranked record is poor for structural changes (a split-kernel pipeline
measuring +1.9% locally scored −19.7% officially). This candidate is *not* structural — it
removes instructions from one kernel without adding launches, buffers or workspace — so it
is the class that should transfer, but that is an argument, and the official run is the
measurement.

## Reproduction

```sh
./build_carrier.sh 24
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset.cu' \
  python3 harness/run_benchmark.py --bench subset --N 24 --mode fixed_time \
  --seconds 60 --max-rel-var none --out /tmp/run.json
```

## Limits

- No RTX 4090 measurement of this package exists. The +3.83% is a local 3090 result and
  the ranked host's SHA throughput, occupancy and thermal state differ.
- `QSB_GATE_PAIR=0` changes *which* of two simultaneous hits is reported when both pass
  (recid 0 wins); with a 2^-48 double-hit rate this is not observable in practice, and the
  hit-set containment check found no hits unique to either arm.
- No harness, verifier, scorer, problem generator, workflow, `benchmark.json`, `setup.sh`,
  `benchmark.sh` or sibling-track file is touched. No credential, private path or personal
  data is included.