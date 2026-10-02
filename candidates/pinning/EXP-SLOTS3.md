# EXP-SLOTS3: QSB_SLOTS 2 -> 3 (slot-pipeline depth) — WASH/NEGATIVE

Rationale: the slot pipeline overlaps batch k's host tail with batch k+1's GPU
work on QSB_SLOTS independent streams. Depth 2 is the crown default; depth 3
adds a third in-flight batch (state memory scales: 3x 768 MiB checkpoints)
to test whether the residual gap is inter-batch bubble.

Method (local 3090, direct kernel invocation, NOT via benchmark.sh — bridge
grinder is unavailable locally; consistent with prior local AB methodology):
- A = crown d59a969 source, prebuilt binary `candidates/pinning/pinning`
  (setup.sh build, sm_52, stamp QSB_ZEROS_N=24).
- B = same source, `nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_SLOTS=3` -> separate
  binary (never installed over the crown build; deleted after the experiment).
- Fixed problem instance: benchmark-results/ab-2lane-20260925/A1/problem/
  pinning.bin (seed 2476060521 instance). Order A1,B1,A2,B2, 70 s timeout per
  run, self-reported cumulative rate from the last progress line.

Results (self-reported M/s, last progress line ~62 s):
  A1 333.8  A2 320.4  (mean A 327.1)
  B1 326.2  B2 321.7  (mean B 323.95)   -> B -0.96% vs A

Conclusion: WASH within the 3090's ~1.5% warm-rep noise, slight negative lean.
Depth 2 already hides the host tail; a third slot only adds state footprint
(512->768 MiB of checkpoints). QUB_SLOTS=3 closed as a lever. Hits were not
verified in these raw runs (direct kernel invocation bypasses the verifier);
this is a rate-only screening A/B, same as EXP-SPLIT-SHA's local methodology.
Artifacts: benchmark-results/ab-slots3-20260925/*.log (4 runs).
