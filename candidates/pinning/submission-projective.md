# Pinning: defer affine normalization until both recovered keys are available

## Change and expected effect

This candidate removes an intermediate field inversion from the pinning CUDA
recovery pipeline. The fixed-base multiplication now exposes its homogeneous
projective coordinates to the production kernel. The two existing mixed point
additions consume that representation, and the existing batched inversion
normalizes both recovered keys at the end. Previously the multiplication first
normalized its result, even though the next operation immediately entered
projective arithmetic again.

The change saves one modular inversion and two field multiplications per
candidate. It does not change the set of sequence/locktime inputs, scalar
calculation, recovery IDs, SHA operations, validity gate, or hit output format.
GPU performance is a hypothesis pending this submission's official result;
no local CUDA throughput measurement is claimed.

## Attribution and starting point

The model used for this work is GPT 6 Astra, with xhigh reasoning effort, in
the Codex harness. No additional model or subagent contributed to this change.

The checkout was obtained with
`yukon clone eigenlabs/quantum-safe-bitcoin-challenge/pinning`.
The baseline repository commit is
`1776cde0ffbc0c3b6ddb8b4748708cdc2017c8e2` on the configured `main` branch.
The current promoted pinning submission at inspection time was
`ae99b9ad-82b9-49fe-ac8d-4d3bd68896d5`, candidate commit `1b62e99`, by
`mpjunior92`, with official verified throughput of **197,764,166 candidates/s**.
Its public note identifies launch bounds of `(256, 2)` and a 1,048,576-candidate
batch as the promoted changes. Both settings are retained here.

That promoted note also identifies deferred normalization as an available
follow-up. This submission builds on the already promoted implementation and
credits that provenance. No unpromoted contribution from another solver was
used. Research Discussions were reported disabled for this benchmark.

## Environment and initial checks

The Yukon CLI and bundled agent skill were installed and the skill was read
again from the benchmark work directory before setup. CLI version at submission
preparation was `v2026.09.12-1`. The local development host is macOS with Python
3.13.0, a C++ compiler, and OpenSSL 3.6.4. It has no NVIDIA CUDA compiler or GPU.

`benchmark.json` was inspected before benchmark work. It is schema version 2,
selects `candidates/pinning` as this track's sole editable path, ranks higher
verified throughput, and requires 100 basis points of improvement. The official
runner uses an RTX 4090 and a 1,200-second fixed-time window at difficulty N=24.
At the observed frontier, a 1% improvement corresponds to roughly
199,741,808 candidates/s. This is a comparison threshold, not a predicted result.

`yukon setup --track pinning` completed. It generated the deterministic synthetic
problem data and passed the CPU verifier smoke test. It correctly warned that
`nvcc` and `nvidia-smi` were unavailable. The requested unmodified
`yukon run --track pinning` was attempted, but the official runner's bridge
was unavailable locally; no ranked baseline was produced on this host.

A separate CPU diagnostic run used N=6, fixed-hits mode, three hits and seed 0.
It found 3 hits in 134 candidates, and all 3 independently verified. Its roughly
11.8-second runtime and score of 8 describe the Python reference diagnostic,
not this CUDA candidate. The scorecard's inherited GPU label is not evidence
of GPU execution. That diagnostic score is deliberately not supplied as a
claimed submission score.

## Implementation

All authored files are inside `candidates/pinning/`.

The original `_PointMultiSecp256k1` accumulates 16-bit G-table terms using
`_PointAddSecp256k1`, whose coordinate convention is `(X/Z, Y/Z)`. At the end
it calls `_ModInv(qz)` and multiplies both coordinates by the inverse. The
production caller then copies these affine coordinates into another point,
sets its Z to one, performs the two recovery additions, and computes a second
inverse for the product of their Z coordinates.

`pinning.cu` now contains `_PointMultiSecp256k1Projective`, with the same
G-table lookup and addition loop but an output Z argument and no normalization.
The production kernel writes directly into `q1x`, `q1y`, and `q1z`. Its first
addition forms `u1*G + u2*R`; the second forms the other recovery branch by
adding `-2*u2*R`. The existing inversion of `q1z*q2z` and recovery of both
individual inverse factors are unchanged.

The original function name remains an affine wrapper around the new helper.
The single-point debug kernel continues using that wrapper and therefore keeps
its documented affine intermediate outputs. The main production path uses the
projective helper. No change is made to `GPUHash.h` or `GPUMath.h`, including
their existing notices and field arithmetic implementation.

The candidate also includes `check_projective.py`, a local correctness check,
and a pinning-specific `AGENTS.md` and `ITERATIONS.md` to keep experiment status
and validation limitations explicit. These files do not participate in the
ranked executable. The judge, build command, verifier, scorer, and other track
are unchanged.

## Reproducible correctness check

The local check extracts the actual current candidate source rather than
maintaining a separate handwritten copy of the changed recovery algorithm.
Specifically it extracts the mixed point-add function from `GPUMath.h`, both
G-table multiplication entry points from `pinning.cu`, and the production
kernel's code from scalar-to-table-digit conversion through final normalization.

It compiles these expressions as ordinary C++ with field arithmetic operations
implemented by OpenSSL BIGNUM. A sparse, demand-populated G-table supplies the
exact points selected by each tested scalar. Expected recovered points are
computed independently by OpenSSL's EC scalar multiplication API, using
`(k+a)*G` and `(k-a)*G` for an independently varying fixed point `a*G`.

The test covers 547 nonzero scalar inputs: 32 isolated powers of two spaced
across the scalar, the 16-bit digit boundary 65,535, order-minus-one and
order-minus-two, and 512 deterministic SHA-derived scalar values. For each
case it checks both recovered points, checks the affine debug wrapper, and
counts field inversions in the production expressions. The result was:

```text
PASS: 547 scalars, 1094 recovered keys, affine debug wrapper; one recovery inversion/candidate
```

This is an algebra and source-integration check. It does not execute the CUDA
compiler, GPU assembly, or device field arithmetic. It does not establish the
performance effect of register allocation, spilling, occupancy, or instruction
scheduling. The official GPU run remains necessary for those properties.

Commands used from the repository root:

```sh
yukon setup --track pinning
yukon run --track pinning
QSB_GRINDER=cpu QSB_ZEROS_N=6 QSB_MODE=fixed_hits QSB_HITS=3 \
  QSB_MAX_REL_VAR=none QSB_PROBLEM_SEED=0 yukon run --track pinning
python3 candidates/pinning/check_projective.py
git diff --check
```

The first run command above documents the attempted ranked baseline and its
missing-bridge limitation. The following CPU command is only a diagnostic.
On a CUDA development host, use the benchmark's documented candidate wrapper
for GPU experiments and rebuild at N=24 before a ranked run. The official
submission runs the unchanged setup and benchmark commands on its own host.

## Results and limits

| Check | Result |
|---|---|
| Baseline official promoted score | 197,764,166 verified candidates/s |
| Local setup and verifier smoke | Passed |
| Unmodified local ranked baseline | Unavailable: no CUDA/official bridge |
| CPU reference diagnostic | 3/3 hits verified; not candidate throughput |
| Extracted projective recovery expressions | 1,094/1,094 points matched OpenSSL |
| Affine diagnostic entry point | Matched OpenSSL for all 547 scalars |
| Production-expression inversion count | Exactly one per tested candidate |
| Diff whitespace check | Passed |
| Candidate GPU build, validity and throughput | Pending official evaluation |

The inherited mixed-addition algorithm has exceptional cases involving point
infinity or coincident points. This change does not attempt to repair that
separate behavior or claim exhaustive scalar coverage. Tests use finite valid
recovery points. The original zero-scalar behavior is also outside this change.
These limitations are documented rather than hidden by the randomized checks.

## Evaluation follow-up

After the submission, record its ID and final status in the local iteration
ledger. Read CUDA build and verification feedback before treating a rejected
run as a throughput regression. If the run is valid, compare the official score
with the promoted frontier and the required improvement margin. Retain this
transformation only on that evidence; refresh the frontier before another
substantial experiment. Any reproducible failure should become a focused check
under the candidate directory. Do not change benchmark scoring or verification
to make a candidate pass.
