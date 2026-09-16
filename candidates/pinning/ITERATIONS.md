# Pinning experiment ledger

## Research priority

The user requested large improvements only. Future research should target
structural gains, using roughly 25%+ as a working prioritization target rather
than chasing small increases above the 1% promotion floor. Predicted gains need
a concrete cost argument and remain unconfirmed until official GPU evaluation.
Current priority follows submission availability: free track first; with both
pending, prepare the likely first finisher. Progress is stronger evidence than
submission age, which is only a tentative ordering heuristic. The recurring
follow-up now permits research and preparation, but no upload or cancellation.

## Initial setup — 2026-09-16

- Work directory: repository root returned by `yukon clone`.
- Yukon: `v2026.09.12-1`; schema v2; selected track `pinning`.
- Base commit: `1776cde0ffbc0c3b6ddb8b4748708cdc2017c8e2`.
- Frontier: `ae99b9ad-82b9-49fe-ac8d-4d3bd68896d5`, `1b62e99`,
  197,764,166 verified candidates/s on RTX 4090.
- Setup: passed CPU verifier smoke; CUDA unavailable locally.
- Ranked baseline: attempted, unavailable because official bridge is absent.
- CPU diagnostic (seed 0, N=6, fixed_hits=3): 134 candidates; 3/3 verified;
  11.8 s; score 8 is CPU diagnostic only, not a claimed GPU score.

## Experiment 1 — defer affine normalization

- Change: carry homogeneous `u1*G` into recovery; normalize the two recovered
  points together. Saves one inversion and two multiplications per candidate.
- Retain `(256, 2)` launch bounds and 1,048,576-candidate batches.
- Local check: 547 scalars, 1,094 recovered keys, affine debug wrapper all
  match OpenSSL; one inversion per production recovery.
- CUDA compilation, PTX arithmetic and speed require the official GPU runner.
- Submission: `8150e0be-d5f7-4a2c-bcb7-bec3d0a4cc64`; **rejected**;
  no GPU score claimed. Exact attribution: GPT 6 Astra xhigh / Codex.
- Submitted `pinning.cu` SHA-256:
  `797440c9e3c72c54bbe2cfee246472731d737879aa04c5ee612d81e3a6afdd97`.
- Process improvement: source-extracted mathematical checks and explicit
  separation of CPU diagnostics from candidate GPU results.
- Official result: **226,444,961 verified candidates/s**, +14.5% over its
  original 197,764,166 baseline but 2.98% below the current 233,402,654 frontier.
  This completed evaluation frees the pinning slot. Subset PR27 remains pending,
  making pinning the current research/submission priority.
- The original results-only schedule was superseded by the user's subsequent
  research and two-track prioritization instructions. The existing 20-minute
  heartbeat now checks both tracks and prepares successors under that policy.

## Experiment 2 — strongest pending pipeline, under review

- Strongest inspected base: PR24 `6e76a74fed8e6e5b8439e64ec20f586085f37d52`,
  alvaroborras's specialization of nullforest8200 PR17. PR38 independently ports
  the same development frontier; its author reports 645,625,292 verified/s on
  a 90-second RTX 4090 run. This is public author evidence, not our GPU result.
- PR41's 24-bit/10.7-GB table reports 363,988,688 verified/s, below the stronger
  pipeline. Other inspected entries largely duplicate mechanisms already in it.
- The best pipeline already uses shared direct XYZZ recovery and hierarchical
  inverse trees; do not count these as new improvements. Inspect checkpoint
  compression and wider mixed windows for incremental gains, and retain the
  known final-carry correction before selecting an arithmetic implementation.

## Ongoing goal research — field base and affine screen

- Refreshed both tracks: subset PR27 still validating with no later scored
  subset entry observed. Pinning PR11 promoted at 249,134,266; its mechanisms
  remain weaker than the strongest inspected pending pipeline. Newjordan's
  911f664 note reviewed; no new mechanism beyond that stronger base.
- Preserved PR24 source with provenance and produced an isolated corrected
  header. Both host and actual-inline-PTX semantic models reproduce the old
  carry defect and pass after repair: 100,900 host calls and 4,360 PTX-model
  calls. A targeted mutation also catches missing `.cc` on the final addition.
  Source details and limitations: `research/FIELD_BASE.md`.
- Main candidate source and prepared subset source remain unchanged. No CUDA
  result or performance claim. Only Apple Clang is installed; its listed
  targets exclude NVPTX, and no local container runtime is present.
- Screened affine batching as a larger architecture, with a nominal point-chain
  reduction from 95M+28S to70M+14S before other costs. Naive repeated checkpoints
  are too costly to justify selection; a fused schedule needs an explicit
  traffic/lifetime proof. See `research/AFFINE_PIPELINE.md`. Goal remains active.
