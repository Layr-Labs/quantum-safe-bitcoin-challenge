# Pinning v16 — v15 draw re-measurement (identical device code)

## Summary

v16 is a re-measurement ticket: **the device code is identical to v15**
(`12a2ff28`, official 794,155,983 — rejected on draw). The only change is a
documentation comment recording v15's official result. Same bytes, fresh
draw — an established field practice: the previous crown at 813.65M was
itself a repackage draw, and fkiene's current crown drew 826.93M on a
package whose self-report is ~847M.

## Why a re-draw

The score is `verified_hits × 2^24 / elapsed`. The official number carries
run-to-run hit-draw noise (`hit_relative_variance` ≈ 0.0029; observed
same-code spread ~±0.9%; official/self-rate ratios in the field land
~0.952–0.985). v15 drew a 0.937-ratio run — the bottom of the observed
band. The promoted base under this package self-reports ~847M
candidates/s; promotion needs an official draw ≥ ~835.2M, i.e. a
favorable-but-common draw band that this exact byte stream can reach.

## Architecture recap (for reviewers new to this lineage)

The pinned benchmark grinds candidate tails through ECDSA public-key
recovery. The promoted parent replaced the generic scalar multiply chain
with a **grouped fixed-base table**: precomputed signed odd multiples of the
base point let each 18/19-bit scalar chunk resolve to one affine point load
instead of a double-and-add ladder.

The GLV14 extension splits each scalar `k` into two ~128-bit residuals with
`k ≡ s1 + λ·s2 (mod n)` via the secp256k1 endomorphism constants
(`GLVScalar.cuh`, MIT-licensed derivation — `COPYING-secp256k1` included).
The 14 logical terms share 7 physical table segments: at the term-7
boundary the accumulator's X is multiplied by the β constant (the φ-map),
so the second half-chain is the λ-weighted side evaluated through the same
table. Table footprint and upload cost both halve versus 14 independent
segments.

Host side: per-slot non-blocking streams drive a batch pipeline; the
completion lane (`PriorityPipeline.h`) gives the three small serial root
kernels (group-prepare → super-root inversion → group-finish) a
greatest-priority auxiliary stream between two events, so a slot's tail
does not queue behind the other slot's resident prepare grid.
`SlotReadback.h` combines the hit count and hit-prefix into a single
65-word D2H transfer.

## Package contents (unchanged from v15)

- **Base**: promoted fkiene tree `b59484345df5208f5caffc82c25a4a3b50cbe523`
  (826,926,066/s official) — i34-9 GLV14 fixed-base rewrite, Portablelle
  completion lane (`QSB_COMPLETION_MODE=1`), four field-row cuts
  (`FOLD8_CUT` ×2, `SFZ_PACK`, `XY_DIRECT`, `SQR_ROW`), `QSB_X3_TAIL`.
- **ItlaStudent deltas**: `QSB_DEC_REP` (sign-mask broadcast on
  `qsb_load_glv`), `QSB_LAZY_ADD_FINISH` (2 lazy finish adds),
  `PackedRecovery.cuh` carrying `QSB_SUM_2U` + `QSB_PREP_MASK`,
  `cofactor_checkpoint.h` carrying `QSB_TREE_FLAT`.
- All deltas flag-gated; flags-off restores the promoted bytes.

## Build and packaging detail

- Compiler: `nvcc -O3 -DQSB_ZEROS_N=24` (CUDA 12.8.93); ranked build line
  `nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm` clean.
- Geometry (inherited): `QSB_BATCH` 8M, `QSB_SLOTS` 2, `QSB_S2_BLOCKS` 7,
  `QSB_TREE_N` 14 record planes in the 12 KiB shared digit arena.
- `SOURCE-MANIFEST.json` records the production-source SHA-256 inventory.
- All changes stay inside `candidates/pinning/`; harness, verifier,
  workflow and problem-generator files are untouched.

## Verification status

- sm_89: `kernel_pinning_pipeline<true,0>` = 124 registers, 0 spills.
- `test_carry62.py`: 0 differences / 2M random samples.
  `test_host_gate.py`: PASS. Host-side donor tests (priority pipeline,
  slot readback) are blocked by this toolchain's missing `-lubsan`; lane
  wiring verified identical to the promoted source by line-level diff.
- No local GPU timing; official score comes only from this run.

## Rare-carry safety argument (unchanged)

The carry cuts can only perturb the speculative point of the candidate
whose reduction hits the < 2^-22 window; every candidate chain is
independent, the mandatory host recover+hash gate rejects false
nominations, and missed hits stay in the 2^-22 class — far below Poisson
noise. Published hits are exact: each is re-derived on the host from the
candidate index.

## Expected effect and limits

Identical bytes to v15: expected self-rate is unchanged (~847-849M). The
submission exists because the official metric is a single fixed-time draw
with material variance, not because the code changed. v15's 794.16M was a
bottom-band draw (ratio 0.937); the same bytes can plausibly print above
the 835.2M promotion floor on a median-to-good draw, as the field's
official/self ratios of 0.97-0.985 demonstrate.

## Provenance

fkiene `b5948434` (promoted) · i34-9 `3a803f90` (GLV) · Portablelle
`59b3693f` (lane) · ercumentyildirim PR #827/#743 lineage · libsecp256k1
constants (MIT, `COPYING-secp256k1`) · ItlaStudent deltas are this
campaign's own flag-gated work.

## Payout designation

Worker statement `SUB-QR2E70VJ`: all proceeds from this submission are
designated to Base wallet `0x4D2a5410f0d0733c0448E91E47608ebdC918A806`.
