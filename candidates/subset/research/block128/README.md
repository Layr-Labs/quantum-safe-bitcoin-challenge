# 128-thread CTA inverse-tree experiment

Status: rejected locally for speed after a warm repeat. All emitted hits pass
independent verification. The production tree edge audit failed; baseline
reproduction and exact-mode geometry audits are tracked below. No submission.
Previous goal turn classification: progress (compact-state correctness audit,
non-winning ABBA measurements, cache-policy implementation and live queue).

The authoritative baseline is public commit 7e95c40 with a current official
score of 623,518,629 candidates/s. Measured local hardware is RTX 3090.
Stage profiling placed 99.1745% of kernel time in kernel_digest, with all
producers plus exact replay below 1%. This directs the experiment to main
kernel scheduling rather than the already-rejected first-state/caching changes.

Change: 128 threads per CTA instead of 256, and proportional shared arrays:
parkA[12][128], products[4][256], inverses[4][128]. The launch bound becomes
128 threads / four minimum blocks, retaining a 128-register budget. The
baseline is 256 threads / two blocks, with twice the shared storage. Actual
compiled default-sm52 resources: both use 128 registers, no stack; shared
memory decreases from 49,152 to 24,576 bytes. The actual 3090 JIT can differ.
Potential mechanism: more resident CTAs can hide the serial cooperative-root
inverse latency. Tradeoff: twice as many root inversions per candidate count.
This is a scheduling hypothesis; resource counts are not throughput proof.

Every CTA still assigns exactly 128 unique windows to each epoch. Old blocks
cover four epochs, new blocks cover two, so physical launch blocks double
262,144 -> 524,288 to keep 1,048,576 epochs and 134,217,728 candidates per
batch. Descriptor, group-map, first-state allocations and host readback
frequency are therefore unchanged. Existing odd-tail aliases and activity
guards remain unchanged. The inverse code already accepts power-of-two block
sizes <=256; only storage bounds are specialized, with dynamic n=blockDim.x.
All upward/downward indices remain within 2*n product and n inverse slots.
Point-chain arithmetic, SHA code, table construction, output format and
independent exact replay are unchanged.

Initial 60-second candidate: 1,673/1,673 verified hits, 233.322393 hit-derived
M/s; baseline: 1,758/1,758, 245.269842 M/s. The candidate first-use run has
longer startup/JIT and a peak self-reported 250.0 M/s versus baseline 243.8;
its 46-second accumulated rate 245.5 versus 243.5 does not yet prove a 1% win.
Do not submit based on the peak counter. The candidate repeat is required.
All 1,673 candidate hits exactly match the baseline hits inside its completed
epoch range (108,003,328 epochs).

The GPU audit adapts the existing OpenSSL-backed tree_audit.cu for block
sizes 32/64/128, tests both sides of 128/256 boundaries and 8,191-element tails,
and retains canonical multiply, root inverse, bounded/fallback checks.
It is not yet complete; inspect the actual pending session.

Helper: DeepSeek V4.1 Flash [1m] through Claude Code successfully performed
only arithmetic on sanitized local stage timings and resource numbers.
Astra independently verified the returned sum 545.35581791 ms, percentages
and 192 shared bytes/thread in both variants. No unpublished source or keys
were sent. All optimization design and implementation is GPT-6 Astra high
in Codex. See helper-prompt.txt, helper-response.json, helper-verification.txt.

Reproduction: PATH=/usr/local/cuda/bin:$PATH python3
candidates/subset/research/block128/trial.py build, then flock -x
/tmp/qsb-gpu.lock python3 candidates/subset/research/block128/trial.py
candidate --label=-a / baseline --label=-b / candidate --label=-c.
Tests use N=24, seed 2026092501, 60 seconds, unchanged harness/gpu_wrap.py and
independent CPU verification. There is no root score artifact. All outputs
are local diagnostics regardless of the harness static RTX_4090 label.

## Follow-up outcome

Warm candidate: 1,724/1,724 independently verified hits, 240.530330 M/s,
14,227,079,168 completed candidates versus baseline 245.269842 M/s and
14,495,514,624 candidates. No speed improvement; do not submit this geometry.

The first tree audit returned errors on adversarial edge inputs while the
canonical multiplier and individual root inverse tests passed. Inspection
found that default QSB_TREE_MUL uses the inherited speculative qsb_filter_mul
(via QSB_SHORT_CARRY3=1), not the canonical multiplier. A 256-thread baseline
audit and a precise-multiplier 128-thread audit are pending to isolate this.
Disabling QSB_SHORT_CARRY3 alone does not compile because the inherited parity
fallback directly calls qsb_fmul/qsb_fadd; the exact audit also disables
QSB_K2S_PARITY_WINDOW. These two switches are audit-only, not proposed changes.
Do not describe the production edge-case tree audit as passed.

Next scheduling direction: 512-thread CTAs with 96 KiB dynamic shared memory,
halving root-inverse count at unchanged per-launch candidate count. Evidence
is under ../block512; the production entry point still uses the baseline256.

The256-thread baseline production audit also failed the adversarial tree
inputs, while its canonical multiply and standalone roots passed. The128
geometry audit with exact multiplication completed with zero wrong values
for every tested block size and tail count. Both terminal logs are preserved
as tree-audit-baseline-result.txt and tree-audit-exact-result.txt. This isolates
the edge failures to inherited speculative arithmetic, not resized storage.
