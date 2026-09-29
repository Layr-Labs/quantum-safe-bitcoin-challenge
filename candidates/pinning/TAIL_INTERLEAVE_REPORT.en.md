# SHA/finish partition investigation and partial-offload experiment

Status: local investigation and validation completed. Both partial-offload
variants were rejected. The fused-SHA scheduling candidate improved all four
paired local inputs, with pooled completed-work throughput +2.8300%.
It is retained for review; RTX 4090 performance is unverified.
No new public submission has been made.

## Isolated stage evidence

The diagnostic loads the exact failed submission's embedded native cubin
(SHA-256 52df8f900a3198190aa50474e1d84b1557f3f0cc12f5fe920a55bc5a632b7cf0).
It runs on the user's RTX 4060 Ti, with 131,072 candidates per launch.
OpenSSL independently supplied 262,144 compressed public keys for valid
secp256k1 points. Every recovery result agrees. A known officially verified
hit exercises the native finish output path; every timed batch produces
exactly the expected hit count. All 131,072 SHA digests match OpenSSL.

| Partition and mode | Median time per batch/pair | Stage capacity |
|---|---:|---:|
| 28 SM, SHA alone, initial run | 79.935 us | 1,639.732 M/s |
| 28 SM, finish alone, initial run | 79.674 us | 1,645.104 M/s |
| 28 SM, both stages, ring dependencies | 160.522 us | 816.536 M/s |
| 28 SM, repeated ring test | 156.421 us | 837.944 M/s |
| 32 SM, ring test A | 137.318 us | 954.513 M/s |
| 32 SM, ring test B | 137.676 us | 952.033 M/s |
| 20 SM, finish only control | 109.507 us | 1,196.927 M/s |

Both 28-SM stages together consume nearly the sum of their separate service
times. Concurrent launch does not double throughput within a saturated shared
partition. Increasing its SM count helps isolated capacity, but takes SMs away
from EC prepare in the complete program.

This supports the SHA/finish-capacity hypothesis. It does not prove the exact
cause or size of the RTX 4090 regression: there is no official device profile,
EC workload, or full GLV11 table in this diagnostic. Input fixtures use valid
affine points P0+iG and exact cofactor state, with OpenSSL reference recovery.
Only P0 is tied to a real published pinning candidate; this is deliberately
a stage diagnostic and does not run the scoring harness.

SHA output and finish input are separate immutable/replayed fixtures, with two
8-MiB buffers per stage (32 MiB total). The ring4 mode retains the SHA->finish
and finish->reuse dependencies but omits EC/root latency. It is a service
capacity experiment, not an end-to-end pipeline benchmark.

The user closed other GPU jobs before these runs. No compute processes were
listed; idle memory was 635 MiB, SM clock 210 MHz and power about 8 W. WSL
still reported 37-38% utilization at idle, so this was a desktop-idle condition,
not a verified zero-utilization exclusive device. Full clock/power/temperature
telemetry was saved for every run. The earlier busy-GPU probe remains excluded.

## Updated base and next hypothesis

Remote main advanced to 8d07d3ebad41a017dfaa5906b164f883a9b59348.
Yukon reported a new current best of 1,008.206828 M/s. The latest pinning
source was downloaded by commit and hashed into an isolated research copy.
A normal git fetch could not write .git/objects; no repository permissions
were changed. origin/main therefore remains stale locally, although the
remote ref and source were independently read and recorded.

The new base includes six sub-batch ring entries, three host slots, register
root inversion, two root streams, CPU co-grind changes and device arithmetic
changes. The proposed candidate is based on this exact source.

Hypothesis: retain the promoted 116/20/shared8 SM allocation and offload only
one in eight SHA sub-batches. The other seven retain the original fused
prepare. This limits added SHA work on the finish partition and does not
take any prepare SMs away. Every candidate performs the complete calculation.

The local compact comparison uses 30/6/shared2 on both arms. It necessarily
uses GLV12, three hot banks, PMIX12=0, PDEC_Z=0, GLV_GLUE=5, TBL_L2POL=0,
and DECODE_CUT=0. The last switch disables a new optimization that requires
a decoder absent in the compact configuration. These differences are a major
limit on transfer to the 4090; a local win would not establish an official win.

Local fused-prepare and finish instruction texts are identical between arms
(7,448 and 4,048 instructions respectively). The local EC-only kernel has
128 registers and no stack/spills; official EC-only has 126 registers and no
stack/spills. SHA uses 44 registers. All source files and GPL notices remain
within candidates/pinning/research/local_4060ti.

## Initial correctness check

One complete sequence per arm: 1,244,600,000 candidates each.
Both found the identical 154 hits; all 308 independently verified.
Elapsed time was 7.0867 seconds for the base and 7.1343 for the candidate.
This short check is not a performance win.

Raw logs and structured results:
benchmark-results/sha-overlap-20260928/shared-stage/

## Complete partial-offload comparisons

Every timed arm completed six full sequences (7,467,600,000 candidates).
CPU co-grind was disabled equally in both arms so the GPU work count is fixed.
Each input was generated after the candidate was frozen. The unchanged CPU
verifier checked every hit, and paired runs produced exactly the same hit sets.
These wall times include setup and output; they are not official scores.

| Variant | Paired changes | Aggregate change | Verified output records | Decision |
|---|---:|---:|---:|---|
| Periodic one-in-eight SHA offload | -1.2349%, +0.3797% | **-0.4266%** | 3,536 / 3,536 | Reject |
| Same fraction, alternating prepare lanes | -1.7517%, +0.4003% | **-0.6793%** | 3,564 / 3,564 | Reject |

The first schedule sends every offloaded batch to prepare lane 0. The balanced
schedule uses batch indices 0, 9, 16, 25, ... to alternate the prepare lane while
retaining the one-in-eight fraction. The native GPU image is byte-identical
between the two partial-offload variants. The balanced experiment included a
separate six-sequence warmup (849 verified hits) excluded from its comparison.

The first ABBA test's median SM clock was 2,670 MHz in all four arms; power was
about 159.5 W and temperature rose during the run. No clock adjustment or
post-hoc score correction was applied. The sign changes between pairs and the
negative aggregate do not establish a repeatable improvement.

Neither variant is active or selected for submission. Their source and raw
evidence are retained solely to reproduce rejected experiments.

## Fused-SHA scheduling follow-up

The next single-change hypothesis interleaves only the locktime-tail SHA
message expansion and compression rounds 32..63. The exact same arithmetic,
constant-zero ALU routing, fused prepare kernel, table, ring and SM partition
are retained. No kernel, event, buffer or memory transfer is added.

This is an experiment in instruction scheduling, not a claim that SHA rounds
can be omitted. NVIDIA's [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
explains the tradeoff between register use, occupancy and instruction-level
parallelism; it does not predict a speedup for this specific change.

Standalone validation covered 1,048,576 inputs across eight midstates and
tail templates, including locktime 0, the top of the uint32 range, and actual
search-range starts. All first and second digests (2,097,152 total) matched
OpenSSL, and the first digests matched the unmodified GPU function.

Both local and official candidate native images were rebuilt with CUDA 12.8.
Prepare remains at 128 registers and 14,336 bytes of shared memory.
Static instruction counts remain 7,448 local and 7,336 official, with changed
instruction text/order. Finish and register-root instruction texts are
identical to their respective bases. Both prepare kernels report zero stack
frame, zero spill stores and zero spill loads.

The initial ABBA test completed six sequences per arm, with fresh seeds
1010775324 and 240148833. Pair gains were +1.2334% and +1.6219%; aggregate
throughput improved +1.4275%. All 3,592 output records verified, with identical
paired hit sets. All logs explicitly report the intended native carrier active.

The candidate was not changed after this result. Independent confirmation
uses two further fresh seeds, twelve full sequences per arm, a separate warmup,
and the reverse BAAB order. It completed successfully.

| Comparison | Seeds | Sequences per run | Pair gains | Aggregate gain | Verified output records |
|---|---|---:|---:|---:|---:|
| Initial ABBA | 1010775324, 240148833 | 6 | +1.2334%, +1.6219% | +1.4275% | 3,592 |
| Independent BAAB | 1239722078, 182650957 | 12 | +1.8834%, +5.2265% | +3.5427% | 7,220 |
| Pooled, excluding warmups | All four | 36 per arm total | All four positive | **+2.8300%** | **10,812** |

Each arm completed 44,805,600,000 candidates in the pooled comparison.
The base took 247.805219 seconds (180.809751 M/s); the candidate took
240.985234 seconds (185.926744 M/s). All paired hit sets are identical.
Every output record passed the unchanged verifier.

The +5.2265% pair is larger than the other three. It is included, not selected
as the expected gain. The small sample and observed spread do not establish
a >=3% improvement. This is a reproducible positive local direction, not a
precise prediction for another GPU.

During confirmation, median SM clocks were 2,640 and 2,625 MHz for candidate
runs, and 2,655 MHz for both base runs. Median power stayed near 159.5-159.7 W;
median temperature was 74-75 C. Candidate median clocks were not higher.
Clocks were not locked and there was no post-hoc correction of timing.

## Normal-path validation

The official-configuration host executable also compiled successfully.
It was not executed with its 21.1 GiB table on the 8 GiB card.

The unchanged CPU reference smoke test passed 3/3 hits (N=6, seed 1132677321).
The compact candidate then passed the unchanged N=24 timed harness with
QSB_TEST_SEQUENCES unset and default CPU co-grind enabled:
469/469 hits verified, consisting of 443 GPU and 26 CPU results, seed 334852122.
This was a 20-second local functional run, not an official performance
comparison. The harness's configured GPU label says RTX_4090, but the physical
device was the RTX 4060 Ti; the label must not be read as 4090 evidence.

An initial normal-path invocation failed before search because its relative
problem path was interpreted under the wrapper's changed working directory.
It produced zero hits in 0.1 seconds and was correctly rejected. Repeating
with absolute problem/output paths passed. Both logs are preserved; the
harness was not edited or bypassed.

## Decision and remaining risk

The previous split-SHA design's added competition on the finish partition
is absent from this candidate: SHA stays fused with prepare. There is no
separate digest buffer exchange or added launch/event dependency. This removes
that introduced contention path by construction, while preserving the
promoted base's resource allocation. It does not eliminate all GPU bottlenecks.

The exact cause of the old official regression is still not proven by a 4090
profile. This new candidate has only local performance evidence. The official
GLV11 table, larger L2, memory bandwidth, SM count and stage balance may change
or erase the benefit. It may fail the official promotion threshold and no
ranking, reward or >=3% official gain is claimed.

Retain the fused-SHA candidate for review. Keep both partial-offload variants
rejected. Active production sources remain at f0e453d; latest-base and candidate
copies remain isolated in research. Existing .pinning.build changes were preserved.
A local review ZIP is prepared from the latest-base official source, including
all original license notices. Public submission still needs new user approval.

Reproduction: prepare_interleave.py, interleave_unit.cu, run_interleave.py.
Native resources, logs and comparisons are under the raw artifact directory
listed above. The full official configuration was compiled, not timed on a 4090.
