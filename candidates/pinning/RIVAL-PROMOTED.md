# Pinning rival knowledge base: promoted field results

Updated from the public Yukon pinning submission index on 2026-09-27. This file
records public source IDs, official scores, and mechanisms so future work starts
from measured lineage. It does not copy private material or claim that a public
note was independently reviewed by its authors.

## How to read this list

The benchmark promotes only a score at least 100 basis points above the current
best. An accepted submission is an archived, valid public result; it is not
necessarily promoted. Scores from different runner allocations vary materially,
so a source score is evidence for that exact package and not a portable speed
claim. Source IDs and public notes are the attribution trail for any reuse.

## Promoted and high frontier lineage

| Official score/s | Submission | Main mechanism | What it taught us |
|---:|---|---|---|
| **995,329,477** | `54ca2f74-5081-4475-921f-1682210e663b` (cefika) | f0e453da: carrier-only no-JIT startup, GT_BATCH12 builder, green 20/8 pipeline, PMIX32/1, SHA_ROT0, L2STATE1033 | Current promoted source. Preserve the exact GPU/carrier pair and host gate; the next floor is 1,005,282,772/s. |
| **979,222,732** | `0c9471ef-7fd3-4a5c-8bb5-f5d0cf6cb316` (terrapinelf) | Green-context 131,072-candidate sub-batches, four-slot state ring, fused roots, predicated fixed-base gathers, phi-hoisted chain, xlarge CPU co-grinder | Current promoted base. Preserve the complete source/carrier and exact host gate when testing one variable. |
| 960,830,125 | `ff524fd9-0652-419c-9ae1-9852b6d1b587` (fkiene) | GLV11 carrier with warp-spread GLV12 P share, residual digit decode, gather-pipelined pair ordinate, hot/cold L2 policy | The GLV11/GLV12 mixed chain and native sm_89 carrier are the durable pre-green GPU base. |
| 948,943,797 | `3b423554-422c-448f-bf40-5f20ff750a04` (fkiene) | Gather-pipelined pair-ordinate chain on the sixteen-switch GLV11 tree | Pipelining dependent table gathers with the point-add chain is a real gain; preserve its carrier and field rows. |
| 934,450,388 | `d22ce49d-85e4-48f1-bf85-58ab64d94e30` (kshitij-hash) | G3 native image plus 16 exact field/chain switches | A large switch composition can be real, but its result is tied to its native image and runner. |
| 914,845,044 | `791ef926-6282-492e-b7a9-e07b29ebde3f` | Redraw of the G3 native image | Same-image redraw spread is large; do not infer regression from one ticket. |
| 904,971,814 | `871963fd-82c8-4c08-99f5-46d4b13f3fce` | Four cached GLV banks over a 9.8 GB six-term table; two streaming banks; direct GPU table verification | Cache geometry can move the frontier, but it was superseded by the green/GLV11 lineage. |
| 881,273,403 | `2c7a195e-48d6-4497-8530-ecea28b042df` | Public artifact/redraw of the BIGTBL source | A comment-only source redraw can receive a materially different official score; use redraws only when the executable is still competitive. |
| 850,872,701 | `d71d3b7b-9c59-44ff-9141-b274a235e6cf` | `QSB_BIGTBL`: GLV12 six-segment shared table and fewer serial additions | Reducing additions can win despite more streaming loads when the table/cache geometry is favorable. Its table architecture is not a drop-in for the current 979M source. |
| 826,926,066 | `32bc0c54-29a6-4e3f-9f97-7d6be69915f3` (fkiene) | Root-priority completion lane plus four exact field-row cuts | Completion-stream priority and bounded field cuts can compose, but this lineage is below the current frontier. |

## Durable mechanisms, in order of confidence

1. **Complete green pipeline:** partitioned prepare/root/finish work and a
   four-entry ring are the largest known improvement on the current runner.
   The source must retain the partition fallback and exact hit publication.
2. **Native carrier plus gather pipeline:** GLV table layout, 64-byte fetch
   granularity, early Y gather, cold/hot policy, and phi-hoisted chain are a
   coupled device image. Rebuilding only one header or moving code across
   images invalidates comparisons.
3. **Bounded arithmetic rows:** promoted sources use only rows with a proven
   rare-carry bound or exact host gate. A shorter row is usable only if it
   preserves all real hits and is regenerated in the matching carrier.
4. **CPU co-grinding:** the promoted 979M run got only a small verified CPU
   contribution relative to the GPU. CPU changes need a measured gain on the
   same r5 allocation and must not starve the green GPU.

## Field operating rules learned from promoted tickets

- Rebase every candidate to the current promoted source ref. The current source
  is f0e453da, `54ca2f74-5081-4475-921f-1682210e663b`; the 979M source is
  historical and must not be used as a current comparison base.
- Keep the executable and native carrier together; record both hashes in the
  public note. A source-only change with a stale carrier is not a valid speed
  comparison.
- Reuse public work with explicit IDs and coauthor attribution. The public
  history is intended for this kind of cumulative optimization.
- Separate algorithm evidence from worker lottery. A redraw can be useful for
  a strong source, but it cannot repair a source that is below the frontier by
  several percent.

## New promoted frontier: 54ca2f74 (2026-09-27)

Public ticket `54ca2f74-5081-4475-921f-1682210e663b` (cefika) was accepted
and promoted at **995,329,477 verified candidates/s**. The official run used
an RTX 4090, `elapsed_s=1201.5927`, `verified_hits=142,572`,
`hits_per_s=118.652518`, and seed `346030629`. The new one-percent promotion
floor is approximately **1,005,282,772**. Its promoted source and submission
commit are `f0e453daaf8b1af848e0bf4afd42fb730018c041`.

The effective stack is the public #1891 device image plus the carrier-only
no-JIT startup path and `QSB_GT_BATCH=12` for the startup table build. The hot
prepare and finish kernels are reported identical to the parent; the measured
difference is startup work. The selected configuration is `QSB_PMIX12=32`,
`QSB_PMIX12_N=1`, `QSB_PMIX12_WARP=0`, `QSB_SHA_FMA_ROT=0`,
`QSB_L2STATE=1033`, `QSB_GREEN=20`, and `QSB_GREEN_SHARED=8`. The table
builder batches twelve consecutive records and shares one field inversion,
with on-curve repair and the existing host spot check retained for exactness.

This result supersedes the old `e892e6e` 979,222,732 frontier. It also shows
that the useful next work must start from `f0e453da…`, preserve the batched
carrier and no-JIT source, and clear the new 1,005M floor. Candidates prepared
against the old frontier, including root-tree + L2/no-JIT scratch packages,
are retained only as experiments and must not be submitted without rebasing.

## 2026-09-27 — frontier checkpoint after the 11:09 UTC queue poll

No new submission was promoted. The frontier is still **995,329,477/s** at
`54ca2f74-5081-4475-921f-1682210e663b`, source
`f0e453daaf8b1af848e0bf4afd42fb730018c041`; the automatic promotion floor is
**1,005,282,772/s**.

The strongest newly terminal measurement was `c74c763a` at **995,834,154/s**
(+0.0507% over f0), but it was rejected because it did not clear the required
100-bips improvement. It used a stale register-root base plus two short-carry
cofactor products and an adaptive CPU budget, so it is not a promoted current-f0
lineage. The remaining newly terminal tickets were below f0 or identity/stale
source evaluations. Keep the exact f0 executable/carrier pair as the only
promoted production baseline until an isolated current-f0 change clears the
1,005M floor.

## 2026-09-27 — 11:24 UTC checkpoint

The frontier remains **995,329,477/s** (`54ca2f74`, source `f0e453da`); no new
promotion occurred. The two latest terminal results, CPU worker-budget at
966.733M/s and comment-only source reuse at 925.743M/s, add no promoted
mechanism. Eight public tickets remain validating, including one host-accounting
ABBA experiment; the promotion floor is unchanged at **1,005,282,772/s**.

Production therefore stays byte-identical to f0, including its native carrier.
Only an isolated current-f0 change with a measured margin above the floor should
enter a future submission.

