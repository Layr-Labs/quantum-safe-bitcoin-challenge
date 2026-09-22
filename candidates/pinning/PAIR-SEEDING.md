# Two-pair affine seeding with productive route selection

This package changes how the fifteen signed table points are combined. Two
independent affine sums replace the first three serial XYZZ additions, using
one denominator inversion shared by a 128-thread block. The resulting two
points seed the existing XYZZ chain. A finite comparison on ordinary search
batches selects between the promoted implementation and three alternatives.
Every trial batch contributes its actual candidates and verified hits.

This is an experiment requiring the ranked GPU, not a claim of a measured
speedup. The authoring machine has no GPU. CPU algebra tests and CUDA resource
reports are evidence about correctness and compilation, not GPU throughput.

Effort: high. Implementation and audits used GPT 6 Astra through Codex.

## Starting point and prior evidence

The source starts from promoted commit
`7c3609b87b9d8e094a16be148fe846dfd5ac7807`. The current best score observed before
submission is 805,428,058 verified candidates/s; the 100-basis-point improvement
requirement means a new score must reach approximately 813,482,339/s. The
promoted arithmetic, cofactor tree, parity window, recovery isomorphism,
publication gate, search space and finish implementation remain the control.

Our previous [PR #1082](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1082)
tried a resident TOP32 cofactor schedule. It was verified but rejected at
766,632,513/s. That schedule is absent from this package. The new worktree was
cloned from the promoted source so that an unsuccessful tree experiment does
not become the baseline for another change.

The historical review included open and rejected submissions. Public
[PR #1063](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1063)
reported isolated negative-Y MAC local gains of 0.214% and 0.362% on its own
composition, with identical fixed-work hit sets. Its official result was
810,314,192/s. An executable-equivalent repeat, PR #1081, obtained 797,694,029/s.
These different ranked runs are not an A/B experiment. PR #1065 high-parity,
PR #1068 pipelined gathers, PR #1073 SHA FMA and PR #1076 mixed fourteen-chunk
attempts also failed to advance the frontier. This is why the present package
compares routes on the same device within one run instead of interpreting a
different problem seed and runner as a causal speedup.

The harness computes its self-reported candidate estimate using the maximum
printed rate. A peak rate extrapolated over the whole run can exceed completed
work as clocks change. For PR #1082, the search position of the last verified
hit implies approximately 763.96 million completed candidates/s, close to the
766.63 million hit-derived official score. The larger self-report is therefore
not evidence of a five-percent arithmetic recall failure. All scoring and
harness code is unchanged here.

## New affine prefix

Let the first four signed table points be P0, P1, P2 and P3. Compute
`d0 = x1-x0`, `d1 = x3-x2`, and the corresponding ordinate differences.
Each lane contributes `d0*d1` to a block product tree. After one root inverse,
the descending tree returns `h = 1/(d0*d1)` to each lane. The two slopes are
`(y1-y0)*h*d1` and `(y3-y2)*h*d0`. Their independent affine additions produce
P01 and P23, which are combined with the existing deferred-Y XYZZ seed. The
eleven remaining table points use the original mixed-add chain.

For each affine pair, the helper computes
`xnew = slope^2 - 2*x - dx` and
`ynew = slope*(x-xnew) - y`. The table uses offset ordinates `y+c`, where
`c=(2^32+976)/2`, to make signed loads cheap. Differences retain their usual
meaning; the affine finish subtracts the offset before forming its ordinate,
normalizes that result, and adds `c` without reducing the offset representation
modulo p. The positive deferred-Y checkpoint ABI is preserved.

For a block of N lanes, the inverse tree uses N initial products, N-1 upward
products and 2(N-1) downward products. At N=128, the nominal prefix plus tail
cost is `90.9765625M + 26S + I/128`, compared with the baseline's `95M + 28S`.
This saves about 4.02 multiplications and two squares per candidate, but adds
sixteen block barriers, a serial root inversion, and offset normalization.
The operation count alone does not establish a speedup.

The tree stores products and then inverses in one 8 KiB heap. During descent,
one parent owns both child writes and reads both products before replacing
them. A barrier separates levels. This avoids cross-parent overwrite races.
The 12 KiB digit arena remains independent because later digits are still live.
Zero denominators contribute one to the collective and use the original scalar
chain for that lane, isolating exceptional input from other lanes.

Two implementations expose the important register/cache tradeoff:

* Route 2 reloads the two anchor points after the inverse. It performs seventeen
  table loads rather than fifteen, with a four-block launch bound.
* Route 3 retains those points in registers and uses a three-block launch bound.
  It keeps fifteen table loads but allows fewer resident blocks.

A first version retaining the points with the four-block bound spilled on
ordinary lanes, not just the root-inverse lane. A shared-memory stash still
spilled and repeated the digit decode. Those versions are not included.

This is narrower than the seven-pair global-checkpoint design in public
[PR #669](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/669),
which regressed substantially. It does not add kernels or global checkpoints.
It also differs from fourteen successive block inversions in public
[PR #1097](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1097),
which was still awaiting dispatch when reviewed. Neither historical approach
is claimed as a successful performance precedent.

## Other routes and attribution

Route 0 is the promoted scalar chain. Route 1 uses negative deferred Y and a
seeded integer multiply-add to replace the mixed-add multiplication followed
by subtraction. `SeededMAC.cuh` is copied from PR #1063 commit
`758c1fe451b705b243dd73851826289b1da7a5b9`, with its GPL notice retained.
That public source credits Saviour1001 and Portablelle for the mechanism from
[PR #1060](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1060).
Credit also goes to terrapinelf for the isolated PR #1063 source and evidence.
The present adaptation restores positive actual Y before checkpointing, so
all four routes share the original recovery and finish kernels.

The finite productive comparison was independently implemented here, informed
by the precedent of PR #1060. It is not a GPU speedup claim for that prior PR.
Other promoted-source provenance remains in the original headers and
`SUBMISSION.md`. No TOP16, TOP32, narrow-parity, K32 or raw-finish composition
from an unpromoted submission was imported with the seeded MAC.

An alternative eight-word seed decomposition was rejected before submission.
For the same CUDA 12.8 / sm89 arithmetic microcase, the promoted multiply then
subtract used 152 instructions and 44 registers; the imported four-64-bit-seed
MAC used 140 and 38; the proposed eight-32-bit-seed form used 155 and 40.
Its integer algebra was correct, but zero extension and carry lowering made
it worse. A negative-X representation likewise moved a subtraction chain
into the ordinate expression instead of removing it. Neither is active here.

## Productive comparison and accounting

The tournament begins with route 0. It compares challengers 1, 2 and 3 against
the incumbent in that order. Each contest uses eight warmup batches for each
implementation, followed by eight 128-batch cohorts in ABBA BAAB order. Each
cohort drains every pending stream slot before its ending timestamp; the next
cohort starts with no inherited GPU work. Its measured cost is elapsed wall
time divided by its actual completed candidate count, including partial batch
tails, sequence preparation, hit copies and publication work.

A challenger replaces the incumbent only when the geometric gain is at least
1.01, at least three of four adjacent comparisons favor it, and no pair is
worse than 0.995. Invalid timing fails closed to the incumbent. Once all three
contests finish, the chosen route remains fixed. `QSB_SCALAR_ROUTE=-1` is the
default; compile with a value from zero through three to force a route for
independent reproduction using the normal slotted pipeline.

All warmup and measured batches are ordinary disjoint search batches. No
candidate is replayed, skipped, added to a fictitious count or excluded from
hit publication. Slot ownership and sequence/locktime attribution do not
change. CUDA event synchronization errors are checked directly before reading
the completed buffers. Monotonic-clock failures invalidate a contest.

The full calibration is 3,120 real batches, roughly 26 billion candidates or
33 seconds if all routes sustain 800 million candidates/s. Slow alternatives
increase that duration and consume real opportunity cost, even though their
work is retained. The selection guard reduces the risk of a persistent
regression; it is not a guarantee of a ranked gain or immunity to later thermal
changes. Diagnostics go to stderr. The current harness does not archive raw
grinder output, so its public result alone may not expose the selected route.

## Validation and reproduction

The installed Yukon skill and benchmark manifest were read in the new cloned
work directory. `yukon setup --track pinning` passed the CPU verifier smoke
test. `yukon run --track pinning` was attempted; this host lacks both a GPU and
the organizer's external benchmark bridge, so there is no local ranked score.
Only `candidates/pinning/` is modified.

Tests shipped with the source:

```sh
python3 candidates/pinning/test_productive_tune.py
python3 candidates/pinning/test_seeded_mac.py
python3 candidates/pinning/test_negative_point.py
python3 candidates/pinning/test_pair_seed.py
```

The point tests require g++, Boost headers and OpenSSL development headers.
Temporary external headers were used on the authoring host; no dependency
installation or test executable is part of the submitted runtime.

* The policy test passes 56 scenarios, including drift, unequal work, invalid
  measurements, ties, regressions and finite completion, under UBSan.
* The seeded MAC test interprets its actual PTX: 20,648 exact 512-bit product
  comparisons, 636 device alias checks and a seed-corruption negative control.
  All integer products match. The retained baseline C31 reduction has 1,139
  accounted-for directed boundary differences and zero random differences in
  10,000 samples. There are no unexplained differences. This operation is not
  claimed to be an exact field multiplier on every possible input.
* The negative-point test extracts the actual new and baseline point helpers.
  Each of two configurations checks 2,048 fifteen-point chains, 28,672 states,
  30,720 signed loads and 6,656 projective rescalings against exact arithmetic
  and OpenSSL, including the production sign-restoration instructions. An
  incorrect-sign mutation is detected.
* The pair-prefix test checks both anchor strategies with and without ordinate
  offsets, using exact arithmetic and OpenSSL. It covers 511 chains, 36
  exceptional fallbacks and 343 affine boundary cases per configuration.
  The actual tree body also runs with CPU threads and barriers at 128 and 256
  lanes, including isolated, multiple and all-zero-denominator inputs. An
  incorrect offset-reduction mutation is detected.

These point tests replace field primitives with exact arithmetic; they do not
execute CUDA or prove the inherited approximate PTX correct for all inputs.
The unchanged OpenSSL publication gate rejects false nominations but cannot
restore genuine hits missed by an approximate GPU path. The independent ranked
verifier remains necessary.

CUDA 12.8.93 builds complete both with the organizer-default command and with
native sm89 code generation:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_89 -Xptxas=-v \
  -o pinning-sm89 pinning.cu -lcrypto -lm
```

Native sm89 stage-0 resources are:

| Route | Registers | Shared bytes | Spill stores/loads | Stack bytes |
| --- | ---: | ---: | ---: | ---: |
| 0 promoted | 128 | 12,288 | 0 / 0 | 0 |
| 1 negative Y | 122 | 12,288 | 0 / 0 | 0 |
| 2 pair reload | 128 | 20,480 | 0 / 0 | 120 |
| 3 pair retained | 168 | 20,480 | 0 / 0 | 120 |

The control stage-0 and shared stage-2 SASS instruction listings match the
clean promoted build exactly: 5,672 and 4,088 instructions respectively.
The shared stage-2 kernel remains at 66 registers with no spill. The default
sm52 cubin has 101-register routes 0/1, a 128-register route 2 with 64/32 bytes
of spill stores/loads, and a 168-register route 3 with 12/12 bytes. Its stage 2
uses 72 registers. RTX 4090 execution of that default build uses its embedded
PTX rather than the sm52 machine code. Native resource reports therefore
cannot substitute for the remote default-build timing. As an additional static
check, the exact compute52 PTX extracted from the default binary was assembled
for sm89 with CUDA 12.8 ptxas. All four routes again had zero spill and the same
register, stack and shared-memory figures as the native table above. This is
still offline assembly, not execution by the runner's driver JIT.

No ranked score is attached as a local claim. The new arithmetic reduction,
its memory tradeoffs and the selection policy are concrete changes; whether
they advance the frontier is for the Yukon GPU run to determine. If the
collective paths fail the same-device comparisons, the package retains its
incumbent instead of imposing a long-running speculative slowdown.
