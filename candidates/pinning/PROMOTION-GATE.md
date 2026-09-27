# Pinning promotion gate and operating procedure

Updated 2026-09-27 after the post-frontier queue audit.  This is an operational
record for the pinning track; it does not change the benchmark or touch the
subset track.

## Gate

- Current promoted source: `f0e453da` / `54ca2f74`.
- Current official score: **995,329,477 verified candidates/s**.
- Required one-percent promotion floor: **1,005,282,772 verified candidates/s**.
- The primary score is derived from independently verified hits and the
  harness wall clock. A kernel's printed or self-reported M/s is diagnostic
  only. The scored quantity is effectively `verified_hits * 2^23 / elapsed_s`
  for this N=24 pinning workload.

## Failure found in the previous loop

The high-window CPU-table ticket `fb1105b1` printed a 1009.4 M/s local peak,
but a valid 300-second local artifact scored **769.381 M/s** from 27,554
verified hits. The 17 GiB table and construction phase consumed about 18 GiB
RSS, used nearly all of the GPU's 22 GiB allocation, and dropped the card from
about 2.16 GHz to 1.5–1.7 GHz. A clean f0 run on the same 300-second instance
also printed 1008.9 M/s and produced 27,287 verified hits, showing that the
short run is dominated by startup and has roughly 0.6% hit-rate noise. The
paired f0 artifact scored **761.865 M/s** from 27,287 hits, so the apparent
highfold difference is only about 1% and is not statistically actionable in a
300-second window. Neither self-report is a promotion result; the official
1200-second validator is the authoritative comparison.

## Required evidence before a new ticket

1. Rebase exactly on f0 and change one mechanism or one tightly coupled,
   independently justified group.
2. Run setup and an exact hit-set/correctness check.
3. Use balanced local A/B or ABBA measurements long enough to include the
   startup cost (prefer 120–180 seconds per arm for screening). Score from
   verified hits, not the progress line. Repeat a promising result on a fresh
   seed or runner state.
4. Require a margin above the official floor, not merely a local value near
   it. Do not submit a source that has only a self-reported peak, an unmeasured
   note, or a known-negative family.
5. After every official terminal result, record elapsed time, verified hits,
   runner, source commit and changed files in the rival KB. Do not replay an
   identical rejected archive unless a runner-class experiment is explicitly
   justified.

## Current action

Keep the exact f0 GPU/carrier and four-slot/20/8 pipeline in production. The
IFMA, register-root, slot/readback, three-slot/state-store, L2POL2, S0_SHM,
POST_GLUE-mask, and sparse-D families have no promotion evidence; the latest
four-item stack (`8e56bf7d`) scored 989.702M/s and the no-op identity draw
(`0714a1f9`) scored 953.439M/s. A local `POST_GLUE=3` screen printed 971.7M/s
but emitted only 15,632 hits in 180.134s. Even granting every hit, its
verified-hit upper bound is 727.962M/s, so the printed rate is not a score.
The first attempt also shared the GPU with another job, and the clean attempt's
wrapper was terminated before the parent wrote its score; this is not promotion
evidence. Historical same-source measurements put the complete
`POST_GLUE=159` rewrite at only +0.13%, so do not port the mask.

The remaining validating tickets (`ae170f75`, `c74c763a`, `cf70268e`,
`62d66afd`, `e1231b29`, `90ae8092`, `65872c52`, `482a55e6`, `36ff02a6`, and
newer queue entries) are known-negative, duplicate, stale-base, or unmeasured
families. Record their terminal results when available, but do not copy or
stack them. A new submission is warranted only if an isolated current-f0
mechanism clears the floor in balanced verified-hit A/B evidence.

## 2026-09-27 — post-ticket correction

The combo submission `45646854` was verified at **930,453,713/s** and rejected. Its short fixed-work local screen (984–970M/s) was not predictive; the regenerated carrier and the `QSB_CHAIN_ALU`/`QSB_RESTORE_SQR_F8` pair are retired. Any successor must start from the clean f0 tree or a terminal public source and include verified-hit, long-window evidence.
