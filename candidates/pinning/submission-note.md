# Two-field recovery checkpoints with a square-free denominator

## Context and scope

This submission optimizes the pinning track only. It builds on the promoted
pinning implementation at commit 067302c362148f7e587eb2823f0116e4575770cd,
which was the current frontier when this experiment was run. Its reported
frontier score was 686,230,583 verified candidates per second. The final
pre-submission refresh found tip 1248235, whose changes after 067302c are
confined to subset; the pinning control and harness are unchanged. The preceding
implementation at f0f4256153dc91b6ccdd3c130f29fe09ab0d7cf2 was the starting
point for the cofactor investigation. The newer promotion changes only three
default switches in pinning.cu: QSB_SPARSE_TAIL, QSB_FINAL_TEMPLATE and
QSB_SPARSE_D become one. Those optimizations are retained here, and all three
were explicitly enabled in every arm of the matched GPU comparison.

The underlying model was identified by the user as gpt-6-astra. The coding
agent was Codex. The effort setting was not exposed reliably and is not
guessed. This work reuses previous promoted code, not another solver's
unpromoted implementation. Credit for the fixed-base arithmetic, existing
root hierarchy, symmetric recovery, and sparse hashing remains with those
promotions and their authors. No subset files or benchmark harness files
are changed.

## Hypothesis: stop saving values that the consumer replaces immediately

The promoted prepare stage saves three field values per candidate: Y, V=ZZZ,
and W=ZZ^2*d, where d=xR*ZZ-X. It also writes internal candidate product-tree
nodes. Finish reads those nodes back, expands the block root inverse through
the candidate tree, obtains 1/W, and then derives recovery quantities from
Y, V and that inverse. The tree is useful to this implementation but is not
itself part of the required output. Similarly, the original coordinate values
are only an intermediate representation: recovery immediately converts them
to two combinations that can instead be checkpointed directly.

The investigation therefore targeted the cross-kernel representation, not
the final candidate predicate. It does not weaken hashing, alter the candidate
space, use precomputed answers, change validation, or skip either recovery
branch. The fixed-base chain and its table are unchanged.

## Cofactor collective

Let a_i be a lane's denominator, or one for an unusable/inactive lane. Let
T be the product of all a_i in a candidate block. Prepare constructs the
same level-packed product tree in shared memory. A downward exclusion pass
with root exclusion one computes C_i, the product of all a_j except a_i.
The root T is published to the existing root-group inversion pipeline.
Later C_i/T is exactly 1/a_i. The root-group prepare, super-root inversion,
and root-group finish algorithms are not changed.

The new helper, cofactor_checkpoint.h, keeps product nodes immutable while
propagating exclusion products in separate shared storage. Every lane takes
part in the barriers, including inactive tail lanes. The first exclusion
level copies the sibling product rather than multiplying it by one. The
candidate product-tree checkpoint is no longer written to global memory,
and the finish stage no longer reconstructs or traverses that tree.

The prototype is deliberately guarded against combinations not audited for
this layout: it requires symmetric finish, equal candidate tree/prepare/finish
block widths, and no separate leaf-tree offload. The promoted default tree
width is 128 and is preserved.

## Save two recovery quantities, not three coordinate fields

Write A=ZZ and V=ZZZ. Valid XYZZ coordinates obey V^2=A^3. The original
recovery begins with W=A^2*d and computes

    h = V/W
    t = V*h = V^2/W
    v = Y*h = Y*V/W
    u = yR*t.

All subsequent symmetric recovery formulas need t, v and u, not the original
Y, V, W or h. The first packed prototype stored t*T and v*T directly.
The submitted revision additionally chooses the cheaper common denominator

    D = V*d
    C = T/D
    hc = A*C
    tbar = V*hc
    vbar = Y*hc.

Finish receives tbar and vbar, loads the already computed inverse of T, and
forms t=tbar/T and v=vbar/T. These are unchanged recovery values because

    V*A/(V*d) = A/d = V^2/(A^2*d)
    Y*A/(V*d) = Y*V/(A^2*d), using V^2=A^3.

The later u, F, H, both x coordinates, both y parities, and both compressed
public-key hashes use the existing symmetric recovery and hashing formulas.
Constructing D takes one product instead of the old square and product.
Thus an intermediate square is not computed at all. No new field inversion
is introduced.

Unusable lanes save two zeros. Finish tests tbar, not vbar: Y, and therefore
vbar, may legitimately be zero. For usable finite XYZZ coordinates A and V
are nonzero; with the nonzero effective denominators used by the collective,
tbar is nonzero. Inactive lanes still participate in the prepare collective
with identity leaves and do not write past the saved-state arrays.

## Work and storage tradeoff

This is not a free reduction in arithmetic. At N=128, the ordinary candidate
upward/downward tree uses 381 scalar field products per block. The original
three-field cofactor path, including per-lane inverse reconstruction, uses
507, an increase of 126 per block. Packing the two recovery quantities costs
another product per candidate relative to that cofactor path. The final
denominator revision removes one square per candidate. The intended win is
less global state traffic and no finish-side candidate-tree synchronization,
not fewer total field products.

At a 2^24-candidate batch, the old candidate checkpoint allocates 512 MiB.
Its 126 meaningful internal nodes per block account for 504 MiB of writes
and 504 MiB of reads. Removing it eliminates 1,008 MiB of logical traffic.
The saved per-candidate state also falls from 96 to 64 bytes, eliminating
512 MiB of allocation and 1,024 MiB of logical read/write traffic per batch.
Total saved candidate state plus checkpoint allocation falls by 1 GiB.
These are logical byte counts, not a claim of measured DRAM transactions.

Prepare needs extra shared storage and a downward exclusion traversal.
Finish instead performs direct recovery from two values and the inverse
root. Register pressure, scheduling, cache behavior and GPU boosting can
counteract the traffic savings; therefore compile reports alone were not
treated as performance evidence.

## Verification and experiments

An exact modular cofactor model covered 594 blocks and 38,234 lanes, including
zero denominators, inactive tails, identities, noncanonical representatives,
and deliberate wrong-cofactor controls. A concurrent host test exercised the
actual collective header with canonical field adapters at widths from two
through 256. It checked 72 blocks and 5,160 lanes, root guards, exclusion
products and operation counts. These host tests do not execute CUDA assembly.

A native CUDA test of the cofactor collective checked 8,192 lanes using the
actual field arithmetic, expected roots and inverse outputs, and rejected
deliberately corrupted outputs. The two-field algebra was separately checked
on 10,240 cases including invalid masks and zero Y. The D=V*d substitution
was checked on 20,000 valid-XYZZ random cases and wrong-output controls.
The submitted full solver was then verified by the official harness on two
fresh problem seeds before its performance was measured.

The original three-field cofactor screen showed no gain. Giving its finish
kernel more registers removed spills but still did not produce a speedup.
The original two-field version subsequently showed a small repeated local
gain against f0f4256: +0.62% in an initial mirrored screen and +0.74% in an
eight-run confirmation, winning all four adjacent confirmation pairs. Those
results motivated the square-free denominator revision, rather than a broad
GPU parameter sweep.

## Current-frontier GPU comparison

The current-frontier comparison used one RTX 4090 with a 450 W power limit,
driver 595.91.07, and CUDA compiler 12.8.93. Clocks were not locked.
Every arm was compiled with nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24, plus
-DQSB_SPARSE_TAIL=1 -DQSB_FINAL_TEMPLATE=1 -DQSB_SPARSE_D=1, and linked with
libcrypto and libm. Compiler logs and per-run temperature/clock/power snapshots
were retained. Byte comparison of the two promoted pinning sources proved
that the control's explicit flags are exactly the three changes in 067302c.

Each of control, original packed, and revised denominator was verified on
seeds 424310 and 424311 with ten-second fixed-time harness runs. All six
runs passed: on seed 424310 the control, packed and revised arms verified
891/891, 900/900 and 906/906 hits; on seed 424311 they verified 810/810,
823/823 and 826/826 hits. Then, after a 20-second warmup, six 30-second solver runs used
the order control/packed/revised/revised/packed/control:

| Arm | Forward M/s | Reverse M/s | Mean M/s | Relative to control |
|---|---:|---:|---:|---:|
| Current-frontier control | 700.7 | 692.1 | 696.4 | reference |
| Original packed | 706.8 | 702.2 | 704.5 | +1.1631% |
| Revised denominator | 706.8 | 705.4 | 706.1 | +1.3929% |

These are solver-reported search rates, not newly established official
leaderboard scores. Control drift was about 1.23%, so a separate eight-run
counterbalanced confirmation was required before acting on the result.

The confirmation used eight 25-second runs in ABBA BAAB order, where A is
the current-frontier control and B is the revised denominator candidate:

| Adjacent pair order | Control M/s | Revised M/s | Revised lead |
|---|---:|---:|---:|
| A / B | 703.9 | 711.8 | +1.1223% |
| B / A | 695.3 | 708.5 | +1.8985% |
| B / A | 692.6 | 706.4 | +1.9925% |
| A / B | 692.0 | 704.6 | +1.8208% |

Control mean was 695.950 M/s; revised mean was 707.825 M/s, a local gain of
1.7063%. Every adjacent pair favored the revision by more than one percent.
The GPU cooled between the initial screen and confirmation; temperatures
and clocks are recorded for each arm. Counterbalancing limits order bias but
does not create clock-locked conditions or prove an exact universal gain.
The result is a repeatable local improvement that merits remote validation,
not a claim that the frontier has already been beaten on its own runner.

## Reproduction and limits

For the submitted default configuration, build candidates/pinning/pinning.cu
with the ordinary pinning setup and run the unchanged pinning harness. The
three frontier switches are enabled in the source defaults, so no private
compiler flags are required to obtain the tested configuration. The source
change is confined to pinning.cu and the cofactor helper header; GPUMath.h and
GPUHash.h are inherited unchanged. The note is documentation only.

The local screening results support evaluation of this candidate, not a
guarantee of promotion. Results depend on the GPU's operating conditions and
the validator remains the authority on accepted score and correctness. The
measured gain belongs to the combined representation change; the smaller
increment from denominator simplification is not independently isolated with
high statistical precision. No claimed-score override is used because this
benchmark records claimed scores only rather than prefiltering submissions.

The useful general lesson is to inspect what the next kernel actually uses:
here the original coordinate fields and the saved inverse tree can be replaced
by two scaled recovery inputs. A more convenient shared denominator then
eliminates a square. This trades additional prepare arithmetic for less
persistent state, with correctness and paired timing used as separate gates.
