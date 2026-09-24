# Pinning table and inversion comparison

Latest official result: PR74 was rejected at **407376555** verified candidates/s
against **644546620** (-36.80%). Its valid run does not establish a cause for
the regression. Any earlier pending/expected-best wording below records the
pre-result experiment. No successor performance win is established.

Prepared from our pending PR74 (`66c031c6`, source fingerprint
`ed600d1c31cc168b0010419da1dfafc78d1be147b816e78dbab52cd37f2e5e54`).
Production remains byte-identical. The promoted pinning frontier is still
644,546,620 verified candidates/s. This is an isolated research candidate,
not a demonstrated speed improvement or a new submission.

PR74 already selects external versus per-CTA inversion but always uses a
16 GiB table. This prototype compares four complete search paths: compact
64 MiB/external, compact/fused, wide/external and wide/fused. Each available
path warms once, then reverse/forward trials run on distinct real batches.
All hits and ranges pass through the ordinary handler. Selection uses elapsed
GPU time per candidate and requires a 2% advantage over compact/external.
Invalid timings retain compact/external; unavailable wide memory leaves a
two-path comparison. Generic modes use compact/external.

The compact runtime-base builder comes from our PR86 subset candidate,
derived from the promoted PR62 mixed geometry. Both use the same corrected
pinning field arithmetic and sign-mask loader. Compact has 15 lookup terms
(14 point additions, 95M+28S); wide has 10 terms (9 additions, 60M+18S).
That arithmetic reduction is inherited from PR74. The new selector supplies
no additional arithmetic gain and cannot guarantee a substantial improvement
over our pending submission. Its purpose is to test the cache/arithmetic and
inversion tradeoffs together, without nesting two potentially coupled choices.

The compact table and actual search state/trees are allocated first. Wide is
optional only when its 16 GiB plus 64 MiB builder reserve fit afterward.
Only allocation OOM falls back; other CUDA failures remain fatal. Both tables
are built from the live problem's recovery coefficient and spot-checked.
Startup and driver JIT costs remain unmeasured and are outside tuner timing.

Validation on this Mac:

- Each geometry: 12,769 extracted recoding cases, 414 OpenSSL curve chains and
  822 recovered-key comparisons passed. Virtual table entries use OpenSSL;
  this does not execute the CUDA field primitives or full table allocation.
- Actual selection/event code passed synthetic timing tests under UBSan,
  covering all winners, ties, margin, unequal counts, warmups, invalid timings,
  disabled mode and early destruction.
- Real CUDA 12.8.93 compilation passed for `sm_89` and default flags, without a
  GPU. Wide prepare/finish/fused retain PR74's 122/78/126 registers and zero
  spills. Compact prepare/fused use 128/126 registers, also zero spills.
  Static instruction counts are not throughput measurements.

Reproduce with this directory's `prepare.py`, `check_geometry.py --mode compact`,
`check_geometry.py --mode wide`, `check_policy.py` and `check_resources.py`.
From the repository root, compile using
`python3 candidates/pinning/research/compile_local.py --source candidates/pinning/research/adaptive_geometry/candidate --default-build`.
See source-bound JSON reports and provenance. The current evidence establishes
comparison readiness, not a substantial performance lead over PR74. Preserve
both pending evaluations.
