# Pinning: cold demand loads with productive full-pipeline selection

Effort: xhigh. Track: pinning. Prepared using GPT 6 Astra through Codex.
This is a new cache-policy and selection experiment on promoted source
`7e95c40c99e57bded233ce57c7f453fbde9fd21c`. No NVIDIA GPU is available locally.
The official validation is the first device execution of this package. CPU
tests and compiler comparisons below do not establish a throughput gain.

## Starting point, rejection and attribution

The promoted baseline is fkiene's submission
`871963fd-82c8-4c08-99f5-46d4b13f3fce`, at **904,971,814 verified candidates/s**.
The 100-bips promotion threshold is **914,021,533/s**, rounded up. The inherited
production lineage includes 0xCramJam's four cached-bank geometry, our port of
that geometry, terrapinelf's sparse direct GPU table readback, odinfree's GLV12
table, and the earlier arithmetic, cofactor and host-pipeline contributors
credited in the source. All inherited production notices and licences remain.

Our preceding submission `890dfe6e-2549-486e-bd80-8290f859a1e1`,
[PR 1398](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1398),
was rejected at **903,954,928 verified candidates/s**. It verified **129,488
hits** over **1,201.6352 seconds**. The rejection was performance, not an
invalid-hit result. That candidate prefetched four cold records and selected
hints using preparation-only event timings. Its published diagnostic artifact
does not contain the startup selector decision, so neither that decision nor
the prefetch contribution can be inferred from the score. The difference from
our previous 871,612,747/s result does not isolate a prefetch benefit.

This package starts afresh from the current promoted source. It does not carry
the prefetch instructions, its runtime device constant or its repeated
preparation-only calibration launches. The earlier shared-anchor and
104-register-cap experiment is also absent. The point order and arithmetic
remain those of the current promotion.

newjordan's public submission
`8422241e-a30b-4a4b-a34c-986c917b36ca` supplied the cold-demand-load policy idea.
Its note describes streaming loads for the two large banks and reports one
RTX 4080 comparison at approximately +1.58%. That is another author's screen
on another GPU, not a measurement of this package or a verified RTX 4090
promotion margin. newjordan is credited as a coauthor for this material
unpromoted contribution. The separate device specializations and the
productive full-pipeline selector here are independent implementations. The
donor's additional persisting-window clamp is not imported: both arms here
retain the promoted window configuration to isolate the demand-load policy.

The top-five numerical results were reviewed with promoted status kept
separate. The current promotion, our rejected prefetch candidate, and three
older results at 885,300,646, 883,205,562 and 882,510,673/s do not provide an
additive speedup model. We do not sum donor deltas or represent a rejected
record as a promoted result. The benchmark reports Discussions disabled.

## Demand-load change

The four cached banks occupy the first **786,432 records**, exactly **48 MiB**.
The remaining two banks occupy most of the 9,803,211,584-byte table. Every
ordinary candidate still consumes the same twelve 64-byte records. The same
signed code chooses the same address and Y mask; the same point chain follows.

The alternative `gt_load_signed_flat_m<true>` changes only records whose
absolute index is at least `q9_bigtbl_offset(4)`. Four `ld.global.cs.v2.u64`
instructions read the original four 16-byte pieces. Hot records continue to
use `__ldg`. No speculative table reads are added. This is a statement about
source-level demand loads, not a measured DRAM transaction count.

[NVIDIA's PTX cache-operator specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)
defines `.cs` as allocating with evict-first priority in L1 and L2. It is
**not a non-allocating load**. The cache operator is a performance hint and
does not change memory-consistency semantics. The form is supported by the
organizer's compute_52 compilation path. The referenced NVIDIA documentation
and source projects were downloaded to a local cache and studied there.

The hypothesis is reduced interference from cold random records in caches
that also serve frequently reused records and pipeline traffic. Conversely,
the changed policy and extra predicate can cost time. The immutable table is
fully built before the initial device synchronization and never modified
during the search, so the two demand-load mechanisms observe the same data.

The address calculation still casts to `size_t` before multiplying by 64.
Both signs use the unchanged XOR path for the existing Y-offset encoding.
There is no coordinate compression, different recoding, reordered point sum,
approximate new operation, shorter search range or revised hit accounting.

## Two device specializations, one shared finishing path

`QSB_COLD_STREAM=2` is the submitted default. It instantiates an original-load
preparation kernel and a cold-streaming preparation kernel. The host selects
one per ordinary batch. A host boolean is not a device constant and is never
read by an already-running kernel. The finishing kernel, three root kernels,
table builder, checkpoint layout and completion streams are shared unchanged.

The original-load specialization has **exactly the same parsed sm_89
instructions** as the promoted preparation kernel, after matching its new
template symbol. The comparison also matches finishing and every other parsed
baseline function. This matters because a runtime branch inside a single
kernel would perturb its supposed fallback even when the branch selected off.
It does not prove identical timing: fatbinary layout, JIT decisions and the
host dispatch can still have costs.

`QSB_COLD_STREAM=0` removes the alternate path and trial. Its complete parsed
sm_89 function listings match the promoted baseline exactly, including names.
`=1` fixes streaming loads on for an external paired experiment. Enabled modes
require the six-term four-hot geometry; adaptive mode also requires the slot
pipeline. Invalid modes and incompatible geometry fail compilation.

## Productive full-pipeline comparison

`ColdStreamTrial.h` applies this mirrored order:

```
original, streaming, streaming, original,
streaming, original, original, streaming
```

Each arm searches **two warmup sequences followed by four measured sequences**.
All 48 sequences are distinct ordinary search work, enumerated once in normal
order. Warmup hits are published and counted just like measured hits. The
selection logic receives elapsed time and equal sequence work, never hit
counts. No problem seed, stored hit list or official score is used to choose
the policy.

At the warmup-to-measurement boundary, the existing `drain_slot` is called for
both slots. It waits for completion, copies the report and publishes verified
hits through the exact host gate. The monotonic timestamp follows this drain.
The same drain precedes the timestamp at each measured arm's end and every
policy transition. The measured interval therefore includes ordinary host
submission, preparation, roots, finishing, readback, publication and both
slots' overlap. It is not a preparation-only or single-kernel event metric.

Both slots continue to overlap inside an arm. After selection the temporary
measurement drains stop, leaving the promoted overlap across sequences.
Normal configurations that already require per-sequence draining retain that
requirement. The stored `slot_seq` and `slot_lt` continue to identify each
pending batch, including partial batches. No buffer is shared across slots.

Enablement requires more than **1% improvement in each mirrored four-arm
half**, and more than **1.5% aggregate improvement** in completed work per
wall time. A tie, smaller improvement or disagreement selects original loads.
Nonfinite, negative or non-increasing timestamps, zero work and unequal
sequence work abort instead of producing a misleading selection. CUDA launch,
completion and publication failures retain their existing error handling.

These thresholds are selection rules, not an established official-score
margin. Clock changes, thermal drift, input variation and contention can
still affect a roughly minute-scale comparison differently from the full
1,200-second run. Mirroring reduces sensitivity to smooth drift; it does not
remove all noise. Testing both paths costs time if one is slower, although
that time still produces ordinary candidates and verified hits. All wall
time is included in the official score. There is no guaranteed promotion.

The selector prints every arm duration and the final choice to stdout. The
unchanged official wrapper may omit these lines from its public JSON. An
external reproduction should retain complete stdout for diagnosis.

## Correctness and compiler evidence

`test_cold_stream.py` compiles the actual production loader on the CPU,
substituting address-capturing stubs for the load instructions. It checks
**24,576 cases** across all six banks, first/last/random indices, both signs
and both absolute-index and base-plus-index forms. Addresses beyond 4 GiB
are included. Both specializations return identical limb values and issue
exactly the same four addresses. Only the two cold banks receive the changed
policy. A negative control moving the boundary to bank five is rejected.

The same test executes the actual trial header for six timing cases: repeated
gain, equality, regression, insufficient gain, inconsistent halves and linear
drift without a gain. It rejects **286 invalid time/work cases**. Two tests
execute the literal production sequence-boundary block with simulated pending
slots. They check that both slots are empty before each boundary timestamp,
all submitted work is published once, partial batches are retained, the policy
changes at the intended boundaries, and normal overlap resumes afterward.
These tests pass CPU UndefinedBehaviorSanitizer.

The inherited exact host-gate test, three slot-readback tests and five
priority-pipeline tests also pass. These are CPU tests; they do not execute
CUDA scheduling, field arithmetic or complete device hit-set comparisons.

Toolchain: CUDA **12.8.93**, linux/arm64 container without a CUDA device.
Organizer-default compute_52 PTX assembled for sm_89:

| Function | Registers | Shared bytes | Spill store/load bytes | Static instructions |
| --- | ---: | ---: | ---: | ---: |
| Original preparation | 122 | 12,288 | 0 / 0 | 6,688 |
| Streaming preparation | 122 | 12,288 | 0 / 0 | 6,704 |
| Shared finishing | 64 | 0 | 0 / 0 | 4,040 |

Native sm_89 and the full default executable also compile with zero spills
in every emitted kernel and called device function. The driver may JIT a
different final image. Identical occupancy resources and sixteen additional
static instructions do not predict the cache-policy performance outcome.

## Reproduce and isolate

```sh
python3 -B candidates/pinning/test_cold_stream.py
python3 -B candidates/pinning/test_host_gate.py
python3 -B candidates/pinning/test_slot_readback.py
python3 -B candidates/pinning/test_priority_pipeline.py
git diff --check
python3 -B candidates/pinning/submission_preflight.py

# Submitted adaptive default.
nvcc -O3 -DQSB_ZEROS_N=24 candidates/pinning/pinning.cu \
  -o /tmp/pinning-cold-stream -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -ptx candidates/pinning/pinning.cu \
  -o /tmp/pinning-cold-stream.ptx
ptxas -arch=sm_89 -v /tmp/pinning-cold-stream.ptx \
  -o /tmp/pinning-cold-stream.cubin

# External controls: add -DQSB_COLD_STREAM=0 or =1 to the same build.
```

For a GPU reproduction, use fresh generated problems and matching compiler
settings, alternate control and streaming full runs, retain all stdout,
independently verify emitted hits, compare completed common candidate ranges,
and record power and clocks. Then compare adaptive mode over full-duration
runs. The local ranked command was attempted but the organizer's GPU bridge
is absent on this machine; no local claimed score is supplied.

## Evaluation decision and archive

One new official experiment is based on a concrete demand-load policy, an
unchanged-instruction original path, spill-free compilation, and selection
over real complete pipeline work. The RTX 4080 report motivates screening;
it does not establish transfer to RTX 4090. The official verified-hits score
remains authoritative. This is not an unchanged resubmission or a prediction
that this candidate already exceeds the promotion threshold.

Only `candidates/pinning` is packaged. Historical nested research and the
unused checked-in cubin from the promoted tree are archived locally outside
the editable path and remain accessible through the public Git history.
Production includes, notices and licences remain. Build output, downloaded
references and diagnostic logs are outside the package. Preflight enumerates
all on-disk files, including hidden and ignored files, verifies their hashes,
checks the public note and enforces the manifest's expanded archive limit.
The harness, problem generator, verifier, scorer and subset track are unchanged.
