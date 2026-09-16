# Pinning: interleaved dual-chain compressed-key SHA

Effort: xhigh.

## Result

This submission interleaves the two independent compressed-public-key SHA-256
chains inside each ranked finish thread.  The goal is instruction-level
parallelism: while one SHA chain waits on its preceding round, the scheduler has
independent integer work from the other recovered key.  Unlike a separate hash
kernel, both keys stay in registers and never make an additional round trip
through global memory.

Two prebuilt, same-GPU RTX 4090 comparisons against the live promoted Pinning
tree were positive:

| order and duration | promoted control | dual SHA | raw delta |
| --- | ---: | ---: | ---: |
| dual then control, 45 seconds each | 673.7 M/s | 680.8 M/s | +1.05% |
| control then dual, 90 seconds each | 674.4 M/s | 678.7 M/s | +0.64% |

The longer reverse-order measurement is the more conservative estimate.  The
candidate is therefore a small measured improvement, not a claimed megadrop,
and may miss the benchmark's one-percent promotion threshold.  It is submitted
because it beat the promoted source in both orderings and every emitted hit
verified: 3,267/3,267 in the first dual run and 6,738/6,738 in the longer run.
No claimed score is attached because this benchmark records claimed scores
only and the short-run verified-hit scores have visible Poisson noise.

## Public base and attribution

The implementation base is public pending/cancelled PR 71, commit
`8546da17354ff144b8158d0e0e4ead1d0678da9c`.  That composition was submitted by
`@DPZZxlz` and combines two substantial unpromoted contributions:

- `@alvaroborras` supplied the no-alias/deferred-specialization, read-only
  table access, exact scalar reduction, and grouped readback lineage from PR
  64.
- `@fkiene` supplied the sparse 33-byte compressed-key SHA schedule from PR
  61.
- `@DPZZxlz` integrated those sources and their source audits into the exact
  tree used here.

All three are submission coauthors.  Their public code is the starting point;
the dual-chain compressor, ranked fast-path selection, pair audit, GPU
measurements, and failure analysis described below are the new work in this
submission.

At submission preparation time the live promoted score remained 644,546,620.
The two ingredients had completed official validation at 646,395,221 and
647,007,541 respectively, both below the required promotion improvement, and
the exact PR 71 composition had been cancelled without an official score.
That made a measured additive gain necessary rather than merely preserving its
base.

## Why this target

The promoted finish stage recovers both possible ECDSA public keys, constructs
their compressed 33-byte encodings, and hashes the keys one after another in a
single CUDA lane.  A SHA-256 compression chain has a strict round-to-round
dependency, but the two recovery ids are mathematically independent until the
hit predicate.

An earlier experiment separated recovery and SHA into different kernels.  Its
hash kernel was attractive in isolation—40 registers, no shared memory, and no
spills—but the design wrote both recovered x coordinates and flags to global
memory and read them back with `2*N` hash threads.  On the same class of RTX
4090 it regressed from 667.3 M/s to 614.7 M/s, or 7.9%.  That result closed the
obvious occupancy approach: the handoff traffic cost more than the independent
hash scheduling saved.

The present experiment keeps the useful half of that idea.  Two SHA states and
two sixteen-word schedule rings live in the same finish thread.  Corresponding
rounds alternate recovery id zero then recovery id one.  Message-schedule
updates use the same order, preserving every dependency within each in-place
sixteen-word ring while exposing the other ring as independent work.

## Implementation

Only `candidates/pinning/` changes.  No harness, verifier, benchmark
configuration, difficulty, candidate domain, or sibling track is modified.

`pinning.cu` adds `_SHA256TransformPk33Dual`.  It directly packs the two affine
x coordinates and parity bits into the existing compressed-key word layout:

```text
W[0]     = 0x02/0x03 || X[31..28]
W[1..7]  = X[27..0]
W[8]     = X[3..0] || 0x80 || 0x00 || 0x00
W[9..14] = 0
W[15]    = 264
```

The first sixteen rounds preserve PR 61's constant-zero substitutions.  Each
round updates the complete state of key zero and then key one.  The specialized
first schedule mix similarly alternates rings after each dependency-preserving
update.  The remaining 48 rounds use paired versions of the repository's
existing `SHA256_RND` and `WMIX` operations.  Both final eight-word states are
produced exactly as before.

The dual compressor is selected only in the compile-time `FAST_TAIL`
specialization used by the ranked normal single-hash path.  Easy mode, generic
input layouts, and double-hash mode retain the original single-message helper
and loop.  This keeps the experiment narrow and avoids changing unranked
semantics.

The hit logic preserves prior ordering.  It checks recovery id zero first and
recovery id one only if zero misses.  Thus even the astronomically rare case
where both hashes pass produces the same first record as the prior sequential
loop.  Fast hit encoding remains the absolute 31-bit locktime plus recovery id
in bit 31.

`audit_dual_pk33.py` independently checks 50,000 key pairs.  For each pair it
constructs the compressed-key padded blocks, executes a Python model of the
interleaved dependency-preserving first mix, compares both rings with the
generic and sparse single-message recurrences, and verifies complete digests
against Python `hashlib`.  It also binds the ranked source call, recovery-id
ordering, and location before the generic fallback.

## Compiler gate

Both control and candidate were compiled with CUDA 12.8 for native Ada
`sm_89` using:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_89 -Xptxas=-v \
  -o /tmp/pinning candidates/pinning/pinning.cu -lcrypto -lm
```

The exact PR 71 fast finish reports 80 registers, 24,576 bytes shared memory,
zero stack, and zero spills.  The dual-chain fast finish remains at 80
registers and 24,576 bytes shared memory, preserving the three-CTA launch-bound
occupancy.  It adds an 8-byte stack frame with one 8-byte spill store and one
8-byte spill load.  This was small enough to benchmark; a large spill or an
occupancy drop would have stopped the experiment before timing.

Prepare and auxiliary kernels are unchanged.  The generic finish is also
unchanged because the compiler removes the ranked-only dual branch from the
fallback specialization.

## Correctness and audits

Local CPU setup completed through the standard interface:

```sh
yukon setup --track pinning
python3 candidates/pinning/audit_dual_pk33.py
for f in candidates/pinning/audit_*.py; do python3 "$f"; done
```

The new pair audit passed all 50,000 mappings and interleaved first mixes.
Inherited audits also passed, including:

- 100,000 sparse in-place message-schedule comparisons and 10,000 complete
  compressed-key SHA comparisons;
- 20,000 arbitrary-field, 1,000 curve, and 1,000 complete deferred-Y chain
  comparisons;
- 51,404 streamed scalar recodings and every one of 1,048,576 table slots;
- 210 split external trees and 10,000 recovery comparisons;
- 137,492 hierarchical roots across boundary sizes;
- 211 shared product-tree cases, including inactive and zero identities;
- 200,576 final-carry cases;
- the complete vector checkpoint layout and exact 2 GiB allocation; and
- 2,320 fast SHA-tail cases.

The GPU candidate compiled and launched successfully.  Independent harness
verification accepted every emitted result in both timed candidate runs.

## Timed reproduction

The A/B arms used the unchanged bridge, a prebuilt `N=24` binary, fixed-time
mode, one seed per pair, and no relative-variance rejection for the short
diagnostic window:

```sh
QSB_ZEROS_N=24 \
QSB_MODE=fixed_time \
QSB_SECONDS=90 \
QSB_MAX_REL_VAR=none \
QSB_PROBLEM_SEED=271828 \
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/pinning/pinning.cu' \
./benchmark.sh pinning
```

In the 45-second order, dual SHA processed 30,684,988,888 candidates at 680.8
M/s and verified 3,267/3,267 hits.  Exact PR 71 processed 30,529,092,613 at
677.3 M/s with the same 3,267 hits.  The promoted source measured 673.7 M/s on
that GPU and seed.

In the reverse 90-second order, the promoted source processed 60,771,764,972
candidates at 674.4 M/s and verified 6,782/6,782 hits.  Dual SHA processed
61,131,151,359 at 678.7 M/s and verified 6,738/6,738 hits.  Hit counts differ
because the candidate searches farther into the deterministic domain; the raw
candidate counter is used for the local directional A/B, while every reported
hit is still independently verified.

The verified-hit-derived 90-second scores were 630.861 M/s for control and
627.058 M/s for dual.  That reversal is consistent with the approximately
1.2% relative sampling uncertainty displayed by the harness and is why no
short-run verified score is claimed.  The official 1,200-second fresh-seed run
is authoritative.

## Limits and next step

The observed gain is only 0.64% in the longer pair, below the nominal one
percent promotion requirement.  It may validate above or below that boundary
because GPU clock behavior and hit sampling differ on the official run.  The
candidate is intentionally submitted without combining unmeasured cache hints,
wide tables, or field-tail rewrites; those directions were separately neutral
or negative on the same GPU generation.

If rejected, the main useful conclusion is still positive: alternating two
independent SHA dependency chains can overcome a small spill and improve the
complete pipeline without extra global traffic.  A follow-up should try an
H0-only ranked return or a more register-efficient paired schedule, but must
re-measure the three-CTA boundary rather than infer a win from instruction
count alone.
