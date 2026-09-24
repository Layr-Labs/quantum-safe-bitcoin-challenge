# Compact first-state experiment, 2026-09-25

Outcome: correct, no demonstrated >1% performance improvement; not submitted.
Prior goal turn classification: progress (batch-size experiment and audited plan).
Current frontier before and after: 623,518,629 candidates/s, official Subset.
Base public commit: 7e95c40c99e57bded233ce57c7f453fbde9fd21c.

The candidate changes only QSB_FIRST_SLOTS from 16 to 8 for 128 windows;
256 windows keep 64. All 128 triples enumerate eight first-block classes
with cardinalities 35,15,15,15,15,15,15,3. Allocation decreases 512 to 256 MiB.
All producer, consumer, exact replay, and allocation strides were inspected.
The runtime first_distinct bound guard is preserved. Default capacity restored
following the non-winning trial; a guarded override remains for reproduction.

All four runs are RTX 3090 diagnostics, NOT RTX 4090 official scores. Each uses
harness/gpu_wrap.py and independent CPU hit verification, N=24, 40 seconds,
seed=2026092501. Runs hold flock -x /tmp/qsb-gpu.lock. ABBA results below and
results-summary.json distinguish verified score from diagnostic counters.
The first candidate run had a longer startup/first-use interval and completed
fewer batches. The second candidate and first baseline returned exactly the
same 1,141-hit set. Every candidate-b hit is also in that set. No missing hits
were observed within the common completed epoch range.

| Run | Verified hits | Wall s | Hit-derived M/s | Self-reported max M/s | Completed candidates |
|---|---:|---:|---:|---:|---:|
| baseline-a | 1141 | 40.1395 | 238.453507 | 244.0 | 9529458688 |
| candidate-b | 1032 | 40.1222 | 215.767052 | 244.5 | 8724152320 |
| candidate-c | 1141 | 40.1306 | 238.50619 | 244.3 | 9529458688 |
| baseline-d | 1160 | 40.134 | 242.457186 | 247.5 | 9663676416 |


OpenSSL first-stage audit: 2,546 compression states, 20,368 active words,
13,936 untouched-padding words, zero errors. The research audit adapts the
old first_stage_audit.cu to the actual flat producer (the old audit referenced
a disabled kernel) and uses class counts 1,3,7,8 over 67 epochs, two fixtures.
See first-audit-result.txt and first_audit.cu.

Mechanical helper: first external call was rejected by auto-review. Both
source files were then fetched unauthenticated from the exact public GitHub
commit and SHA256 matched against git show HEAD:<path>. Only public-original
excerpts plus a natural-language slot-count hypothesis were sent, no edited
local source. Accepted retry of DeepSeek via Claude Code timed out after
100 seconds without a response and was killed. No helper result is credited.
Public source copies were subsequently re-extracted directly with git show.
Astra independently performed the structural enumeration and GPU checks.

Reproduction: PATH=/usr/local/cuda/bin:$PATH python3
candidates/subset/research/compact_first/trial.py build; then use flock -x
/tmp/qsb-gpu.lock python3 candidates/subset/research/compact_first/trial.py
baseline --label=-a (and candidate --label=-b, candidate --label=-c,
baseline --label=-d). Rebuild wrappers after any included-header change.

Next experiment: protect reused 64 MiB fixed-base point-table cache lines
from the streaming producer working sets using CUDA access-policy hints.
This does not change arithmetic, enumeration, or hit validation.
