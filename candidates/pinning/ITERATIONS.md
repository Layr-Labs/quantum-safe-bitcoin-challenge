# Pinning experiment ledger

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
- Submission: being prepared; no GPU score claimed.
- Process improvement: source-extracted mathematical checks and explicit
  separation of CPU diagnostics from candidate GPU results.
