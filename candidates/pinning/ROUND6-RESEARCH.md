# Pinning: conditional bias in fused square reduction

This submission changes the fused field operation used in each mixed XYZZ
addition. The existing implementation forms the wide integer corresponding to
`a*a + e - 2*q` and adds `3*p` before reducing it. The new GPU path adds that
bias only when the signed wide intermediate is negative. The nonnegative path
therefore omits a ten-word carry chain. Both paths retain the existing final
pseudo-Mersenne folds.

This is a measured, correctness-tested experiment, not a claim that a local
RTX 3070 result predicts an RTX 4090 score. Earlier local comparisons showed a
small positive average that was smaller than the observed run-to-run drift.
Official evaluation of this exact source archive is needed to establish whether
the change helps the scoring hardware. No claimed score is supplied.

The final comparison on the current 8M-batch base was slightly negative
(-0.4616% on the RTX 3070). Thus this submission is an explicit cross-device
performance experiment, not a demonstrated local improvement. Both measured
signs of the effect are reported below.

Development used GPT-6 Astra, effort xhigh, through Codex. The current official
Yukon checkout and the prescribed build and hit-reporting interfaces are used.
No harness, verifier, workflow, problem generator, difficulty setting, or scoring
rule is included in the archive. Research Discussions were disabled when checked.

## Base and inherited implementation

The baseline is promoted commit
`b62eb79d21ac6d1db6bff3732448f20ac84ce30b`, submission
`886874a0-4e5c-41bd-b2ec-eddd87fbf77c`, with the displayed official score
739,180,224. That revision halves the candidate batch to 8,388,608. Its launch
size, two-stream pipeline, 64 MiB signed fixed-base table, 128-leaf cofactor tree,
global root inversion, and evict-first state traffic are retained.

The field square and fused reduction are inherited from the public xlib fused
square/add/subtract lineage and the subsequently promoted interleaved square
schedule. The original attribution and GPLv3-only notices remain in the sources.
This work changes the placement and condition of the bias; it does not claim
authorship of the underlying multiplication, square, or recovery algorithms.

The submission also carries forward two independently audited changes from our
previous source `5e7280d3128e68edd8febfe15f10aa15677813d9`: a borrow-chain parity
helper and bounded raw-product additions in the recovery finish. That previous
submission was valid but was rejected with score 710,087,672. The earlier result
is not presented as evidence that this combination will beat the current leader.

## Arithmetic change

Write `B = 2^256`, `K = 2^32 + 977`, and `p = B-K`. The existing first fold of
the 512-bit square replaces its high part by its product with `K`. Let the
resulting nonnegative wide value be `S`. Before the final fold, the desired
integer is `T = S + e - 2*q`, with the two other input values represented in
`[0,B)`. This integer is congruent to the desired field result.

The old schedule adds `3*p` unconditionally. The new schedule first adds `e`
and subtracts `q` twice, retaining the wide carry/borrow words. It then tests
the highest word as a signed integer. A negative `T` is greater than `-2*B`,
so adding `3*p` makes it positive. A nonnegative `T` needs no bias. In either
case adding or omitting `3*p` leaves the residue modulo `p` unchanged, and the
remaining reduction receives a nonnegative wide integer within its bound.

The condition is on the computed arithmetic value. It does not depend on a
problem seed, difficulty, runtime duration, device name, scoring state, or a
known solution. Small squares and near-modulus operands exercise the negative
path explicitly in the audit; it is not removed on a probability assumption.

The CPU reference in `GPUMath.h` retains the unconditional, congruent schedule.
The production GPU path is tested against independently computed OpenSSL
BIGNUM values rather than merely comparing it with that reference. The final
result can use the existing full-width residue convention; consumers and the
recovery normalization boundaries are preserved.

## Recovery finish carried forward

`BorrowParity.cuh` computes the subtraction borrow through an explicit scoped
PTX chain. The parity of the reduced difference is the XOR of the two low bits
and the borrow, because the field modulus is odd. This preserves equality and
all limb-boundary comparisons while avoiding a nested magnitude comparison.

The two x-coordinate products in `PackedRecovery.cuh` retain the exact raw
multiplier, then use `qsb_add_boundary` before adding the fixed canonical
x-coordinate. If that addend's high limb is not all ones, the sum is less than
`2*p`, so the following one-subtraction modular addition is sufficient. The
extreme upper addend range still normalizes the product first. This change
does not remove rare-boundary handling. Streaming stores from the new base are
retained; the old finish file is not copied over the new base wholesale.

## Development hardware and measurement method

Local GPU: RTX 3070, 8 GiB, WSL2 Ubuntu. Compiler: CUDA 12.8.93. Local binaries
target `sm_86`; the driver on this machine cannot execute the compiler's newer
default PTX. Independent compile checks also cover native `sm_89` and the
ordinary, architecture-unspecified setup command. The submitted archive contains
source, documentation, and audits, with no generated executable or build stamp.

Candidate throughput estimated from sparse hits is too noisy for short local
optimization experiments. Separate source copies outside the submission archive
were instrumented to stop only after draining an exact number of complete
sequence traversals. The device code, candidate generation, hit tests, and hit
collection were left unchanged. Four complete traversals search 4,978,400,000
candidates; sixteen search 19,913,600,000. These fixed-stop copies are not the
production entry point.

The fixed-work problem seed is 817231 and the difficulty is 24 bits. Comparisons
run in A/B/B/A order on one GPU. Complete hit sets must match, and the unchanged
independent harness verifier checks the hits. The final run also uses the normal
unmodified timed harness on a different seed. All GPU measurements are serialized.

The local card reaches its 220 W power limit. Clock and temperature drift are
visible, including changes from approximately 1695 MHz at 76 C to 1635 MHz at
80 C during sustained tests. Small percentages must therefore be treated with
care even when an A/B/B/A mean is positive. No official throughput or promotion
is inferred from a local arithmetic audit or a fixed-work timing alone.

## Earlier experiments and negative results

Before this submission, alternative layouts moved the deferred ordinate anchor
and some projective coordinates into shared memory. A five-block resident
variant reduced register use from 119 to 96, but the extra memory work outweighed
the occupancy improvement: its paired fixed-work comparison was -1.3032%.
Compressing digit planes and changing the tree to 64 lanes also did not yield a
convincing improvement. Those changes are absent from this archive.

A one's-complement encoding fused table-ordinate sign handling with the signed
pair addition. Its CPU audit passed 400,576 cases, and the full grinder returned
the same 585 hits as the control, but the paired timing was -1.6322%. It was
discarded. A warp-vote version of the new conditional bias also passed the GPU
boundary audit, but its complete paired test was -0.6076%. This submission uses
the ordinary conditional branch, not that warp-vote variant.

A separate radix-2^29 multiplier kept nine limbs across repeated operations to
avoid representation conversions in the measured loop. It passed 131,072
OpenSSL cases and an aliasing check. It was nevertheless roughly twice as slow
as the existing multiplier in the microbenchmark. Disassembly showed 371 static
instructions in the test kernel versus 147 for the existing arithmetic. Fewer
carry dependencies did not compensate for the additional normalization, shifts,
and masks. That implementation is not part of the submission.

On the previous 16M-batch promoted base, the selected conditional-bias plus
finish combination completed a long A/B/B/A comparison:

| Arm | Million candidates/s | Verified hit set |
|---|---:|---:|
| control A1 | 177.060680 | 2344 |
| candidate B1 | 176.088845 | 2344 |
| candidate B2 | 177.741668 | 2344 |
| control A2 | 175.271562 | 2344 |

The means were 176.166121 and 176.915257 M/s, a +0.4252% difference. All hit sets
were identical. This is a small exploratory result, not a statistically
established speedup. The new 8M-batch base is re-tested below before submission.

## Reproduction and archive checks

Production is built and invoked through the unchanged official commands:

```sh
yukon setup --track pinning
yukon run --track pinning
```

For a local RTX 3070 correctness run, use the ordinary harness with a native
build and an explicitly local configuration. The native build command is:

```sh
nvcc -O3 -arch=sm_86 -DQSB_ZEROS_N=24 \
  candidates/pinning/pinning.cu -o candidates/pinning/pinning -lcrypto -lm
```

The standalone audits include the actual production source:

```sh
nvcc -O3 -arch=sm_86 -DQSB_ZEROS_N=24 \
  candidates/pinning/tests/fusion_bias_audit.cu -o fusion-audit -lcrypto -lm
./fusion-audit
nvcc -O3 -arch=sm_86 -DQSB_ZEROS_N=24 \
  candidates/pinning/tests/boundary_audit.cu -o boundary-audit -lcrypto -lm
./boundary-audit
```

The fusion audit covers 262,144 inputs in three output/alias arrangements, with
zero mismatches. It includes all combinations of eight boundary values and a
large group of deliberately small-square inputs that force the negative-wide
path. The finish audit covers 32,768 boundary/parity cases with zero mismatches.
Both compare against OpenSSL BIGNUM and fail on CUDA errors or mismatches.

`SOURCE-MANIFEST.json` records this candidate's source hashes, base, compiler
context, and model/harness attribution. The prior subset implementation is
preserved outside this track's archive. No mutable output directories, local
measurements, GPU binaries, or authentication material are submitted.

The acceptance criterion remains the authoritative official result for this
archive. A valid run below the current best is a failed performance experiment,
not a correctness failure or a reason to alter the evaluator.

## Final validation on the 8M-batch base

The production sources were re-audited after applying the change to the exact
current base. The 262,144-case, three-alias fused-operation audit and the
32,768-case recovery boundary/parity audit both passed with zero mismatches.
The four long fixed-work runs were executed serially in one process sequence:

| Arm | Million candidates/s | Independently verified hits |
|---|---:|---:|
| baseline A1 | 182.983596 | 2344 |
| candidate B1 | 180.813542 | 2344 |
| candidate B2 | 179.724225 | 2344 |
| baseline A2 | 179.226242 | 2344 |

Each arm searched 19,913,600,000 candidates. All four hit sets were identical.
The baseline mean was 181.104919 M/s and the candidate mean was 180.268884 M/s,
a **-0.4616%** difference. The downward drift between the two control arms
limits the precision of this comparison; it does not justify converting this
negative result into a positive claim. No 4090 was available locally.

The unchanged ordinary harness then used an independently generated problem,
seed 439987, with a requested 120-second run and native sm_86 production binary.
It reported `verified: true`, 2690 verified hits, 124.9277 seconds elapsed,
180.627254 M verified candidates/s, and hit relative variance 0.019281. The
local configuration explicitly labels the GPU RTX_3070_LOCAL. This sparse-hit
estimate is a correctness check, not a comparison with the official score.

Native sm_89 compilation passed. The binary and build stamp were moved out,
and `yukon setup --track pinning` then passed using the ordinary prescribed
architecture-unspecified build. Its generated executable and stamp were also
removed from the submitted source tree. No build products remain in the track.

One additional input-bound variant moved the bias predicate before the affine
terms. It passed the same 262,144-case GPU audit and returned the same 585 hits
in a short four-arm comparison, but its mean change was -0.0982%. It is not
selected or included in this archive. The submitted predicate remains the
direct signed-wide test documented above, with all exceptional inputs handled.
