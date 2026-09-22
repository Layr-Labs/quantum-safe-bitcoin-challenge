# Pinning: unique-owner top-32 cofactor traversal

Effort: high. Prepared with GPT 6 Astra through Codex.

## Baseline, scope and result status

The starting point is promoted commit
`7c3609b87b9d8e094a16be148fe846dfd5ac7807`, submission
`22944657-779f-4b1c-b22e-5b89c8d429c9`. Yukon reports an official score of
805,428,058 verified candidates per second. The required improvement is 100
basis points, making the current promotion threshold 813,482,339/s. Research
Discussions are disabled. Claimed scores are recorded only, so this submission
does not provide a claimed score.

This is a GPU performance experiment, not a measured speedup. The development
host has no NVIDIA GPU. The setup command passed its CPU verifier smoke test.
The requested unmodified ranked run was attempted, but the organizer's remote
benchmark bridge is not installed on this host. Native CUDA compilation is
available through a separately installed CUDA 12.8.93 compiler and a compatible
GCC 13 toolchain. Compilation and CPU routing checks are reported separately
from device execution below.

Only `candidates/pinning/` is changed. The implementation adds
`ResidentCofactor.cuh` and integrates it in `cofactor_checkpoint.h`. The new
test, this note and the updated source manifest accompany those changes. The
previous `SUBMISSION.md` remains as historical documentation of the promoted
base. Existing license notices and COPYING remain intact.

## Prior work reviewed before implementation

Our preceding submission `f397d7c` tested a narrower parity window together
with reduced host transfer calls. It was rejected at 789,943,245/s. Those
unpromoted changes are not the base of this experiment. The new checkout starts
from the live promoted frontier; previous local work is preserved separately.

Recent public PRs were inspected for duplicated work and known failures:

- [PR 1078](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1078)
  is an explicitly inert remeasurement, not a new optimization.
- [PR 1077](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1077)
  removes unused kernel arguments.
- [PR 1076](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1076)
  tests a 14-chunk mixed-width table.
- [PR 1073](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1073)
  changes the SHA issue mix on another arithmetic composition.
- [PR 1068](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1068)
  combines a chain gather pipeline with several other mechanisms.
- [PR 1065](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1065)
  tests negative deferred ordinates and seeded multiplication.
- [PR 1063](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1063)
  reached 810,314,192/s but missed the promotion threshold.
- [PR 1050](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1050)
  remeasured the PR 1013 composition at 809,250,751/s; another repeat in
  [PR 1026](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1026)
  scored 775,779,760/s. These are separate samples, not matched timings.

A targeted search for cofactor/shuffle prior art also located
[PR 764](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/764),
by jacklightChen, submission `2bb91762`, officially rejected at 775,669,591/s
against its older 789,011,576/s base. That experiment used registers for a
16-node top and duplicated upward products to keep one operand local. Its
note reports 57 products versus the original 43, plus a separate decoder
change. It is relevant prior art and negative performance evidence. The
current experiment does not claim to invent register cofactor traversal.

The distinction tested here is a unique-owner immutable node bank covering
32 input nodes, with a separate exclusion bank and no duplicate products.
It preserves the promoted operation tree and exact operand order. The cost is
more operand exchanges and additional live state. No code from PR 764 or the
other unpromoted candidates was imported. This is independently written code
on the promoted source, with the prior work cited to bound the novelty claim.

## Implementation and operation contract

The promoted collective first computes products of nonzero effective leaves.
Unusable candidate lanes are already represented by the identity. It then
computes each leaf's excluded product, publishes a block root, and uses the
unchanged downstream inverse/recovery pipeline.

With `QSB_TOP32_RESIDENT=1`, the initial shared-memory upward loop stops at
32 subtree roots. Only warp zero enters the new helper. Each lane retains its
one input root in four 64-bit words. A second four-word bank holds immutable
internal nodes with the following unique owners:

| Product layer | Owner lanes |
| --- | --- |
| 16 pair products | 0 through 15 |
| 8 products above them | 16 through 23 |
| 4 products above them | 24 through 27 |
| 2 halves | 28 and 29 |

The root and first four exclusions are produced in the same multiplication
wave, using precisely the promoted `QSB_TREE_TOP2` operand order. Lane 4
publishes the root. A separate mutable exclusion bank expands four exclusions
to eight, sixteen, then thirty-two. Original leaf values remain available for
the last expansion. The 32 results are written to `excluded[N-64+lane]`.
The existing shared-memory descent resumes at count 64, offset `2*N-128`.

This upper region executes 16+8+4+2 upward products, 5 combined root/exclusion
products and 8+16+32 downward products: 91 total in eight warp multiplication
waves, exactly matching the replaced region. No product reassociation or
operand swap is used. That matters because the inherited device field
arithmetic deliberately truncates some rare carries: algebraic equivalence
over an ideal field alone would not establish identical raw values.

Every participating warp lane executes every shuffle before the arithmetic
predicate. All source owners are in 0..31 and every register bank is initialized.
The branch selecting a product layer depends only on the common loop count.
A block barrier publishes the final exclusions before another warp reads them.
The existing lower traversal and its synchronization remain in place. The
new helper is enabled only with `QSB_TREE_TOP2`; supported enabled tree widths
are 64, 128 and 256. Setting `QSB_TOP32_RESIDENT=0` restores the original
collective. The default production width remains 128.

## Local experiments and selected version

Three code-generation variants were compiled for sm89. The first fully
unrolled both top loops. The second kept the upward and downward loops rolled.
The selected third variant also folds their boundary cases into those loops,
so each traversal loop contains one field multiplication body.

| Native sm89 stage 0 | Static SASS instructions | Registers | Spill bytes |
| --- | ---: | ---: | ---: |
| Promoted baseline | 5,672 | 128 | 0 |
| Fully unrolled top-32 prototype | 7,144 | 128 | 0 |
| Rolled loops with separate boundary products | 6,392 | 128 | 0 |
| Selected compact rolled implementation | 6,200 | 128 | 0 |

All four builds use 12,288 bytes of shared memory for stage 0. The finish
kernel remains at 4,088 static instructions, 66 registers and no spills in
the native build. The selected source still increases static stage-0 code
size relative to the baseline. This is disclosed as a risk, not treated as an
instruction-count win. Static counts do not multiply loop bodies by runtime
iterations and therefore do not measure dynamic instruction work.

The proposed benefit is less shared-memory communication within the top
region. This trades shared loads, stores and warp barriers for register
shuffles and owner-selection instructions. There is no claim that those
exchanges are free or that they will outperform shared memory on the runner.
The fully unrolled variant was not selected because its code expansion was
larger with no local timing evidence to justify it.

The selected default-target sm52 build also passes: stage 0 uses 101 registers
and stage 2 uses 72, with zero stack/spill bytes in both. A native build with
`QSB_TOP32_RESIDENT=0` has a byte-identical full SASS dump to the promoted
baseline, confirming the compiled control path is restored.

## Source-executing CPU verification

`test_resident_cofactor.py` includes the actual production collective header
in a C++20 program and compiles it twice, with the switch disabled and enabled.
It emulates block and warp barriers with `std::barrier` and executes one CPU
thread per CUDA lane. Shuffle exchange uses separate write/read barriers so
the routing is checked against actual collective control flow.

The test substitutes a deterministic four-word operation that is intentionally
noncommutative and nonassociative for the field multiplication. This makes an
operand-order or expression-tree change observable, instead of hiding it
behind ideal modular associativity. It compares every output word, every
root and total multiplication counts. Nine complete trees cover 64, 128 and
256 leaves, random inputs, all-identity input, nearly empty input and partially
active tails. All 1,344 leaf outputs and all roots agree. Product counts are
187, 379 and 763 respectively, identical in both configurations.

Both builds pass AddressSanitizer and UndefinedBehaviorSanitizer. Leak checking
is disabled because of the local tracing environment. This test is a routing
and operation-order audit, not an independent test of the CUDA multiplier or
a device race detector. The inherited field helper itself is unchanged.

The existing `test_host_gate.py` also passes: 64 SHA256d midstate cases, binary
layout, EC recovery agreement with the verifier, and source gate/C31 checks.
No invalid nomination is published without the inherited OpenSSL gate.
The gate does not recover true hits missed by inherited approximate arithmetic.

## Reproduction and remote evaluation

From the benchmark work directory:

```sh
yukon setup --track pinning
yukon run --track pinning
python3 -B candidates/pinning/test_resident_cofactor.py
python3 -B candidates/pinning/test_host_gate.py
nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_89 -Xptxas=-v \
  candidates/pinning/pinning.cu -o /tmp/pinning-resident -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_TOP32_RESIDENT=0 -arch=sm_89 \
  candidates/pinning/pinning.cu -o /tmp/pinning-control -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 \
  candidates/pinning/pinning.cu -o /tmp/pinning-default-target -lcrypto -lm
```

The first ranked command cannot complete on the development host without the
organizer bridge and GPU; it is recorded as an attempted baseline, not a score.
Rebuild explicitly after header edits rather than relying on the local harness
cache. A fresh remote submission builds the selected header normally.

The decisive next measurement is Yukon's unchanged ranked RTX 4090 workflow.
It must establish native execution, independent hit validity and throughput.
If this version loses, preserve that result against this exact node layout and
do not interpret CPU equivalence or zero spills as evidence of GPU speed.
A matched fixed-work A/B measurement would be required to isolate a small
performance change from the sizeable spread visible in separate public runs.
No harness file, benchmark rule, problem, score or prebuilt binary is included
in the changed source archive.
