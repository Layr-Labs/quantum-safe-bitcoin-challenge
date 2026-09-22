# Pinning: collective affine comb with exact per-round inversions

Prepared with GPT 6 Astra, xhigh effort, using Codex. This is an experimental
successor built and checked locally, prepared for official RTX 4090 evaluation.
No local GPU throughput has been measured.

## Source and attribution

The promoted comparison base is shared main
7c3609b87b9d8e094a16be148fe846dfd5ac7807, pinning submission
22944657-779f-4b1c-b22e-5b89c8d429c9, score 805,428,058 candidates/s.
The immediate predecessor is our negative-Y, parity-replay and adaptive-SHA
submission 8fd91df0-4fba-44d7-afa4-4bc349723931, submitted package commit
457e4a15ef2f823de24f857c7284405a180b3887. This successor starts from its local
receipt checkpoint 260ae84. New implementation milestone: f898592.
SOURCE-MANIFEST.json records the full milestone and current archive hashes.

All changes are within candidates/pinning. The benchmark, scoring, harness,
problem data and sibling track are unchanged. Existing source and GPLv3 notices
remain intact. Portablelle remains a required coauthor for the substantial
unpromoted narrow-parity code inherited from
1e8f3b02cba580490960655b1a5e1711c9d5b4f6. The affine comb and its tests are new
work. hm39_divstep.cuh and hm41_quad_inverse.cuh are byte-identical copies from
the promoted subset source in 7c3609b, preserving their VanitySearch attribution.
That promoted reuse is cited here and does not introduce another unpromoted
coauthor.

## Algorithmic change

The existing fixed-base comb loads fifteen signed table points and combines
them using projective deferred-Y arithmetic. The new produced-scalar route
combines the same fifteen points in affine coordinates. Every ordinary affine
addition requires division by qx-x. A block of 128 candidates shares that cost:
a product tree forms all denominators, four lanes invert its root, and a
reverse traversal supplies the exact inverse for every lane.

For an ordinary addition, lambda=(qy-y)/(qx-x),
xnew=lambda^2-x-qx, and ynew=lambda*(x-xnew)-y. Fourteen successive rounds
combine the fifteen table terms. This exchanges projective multiplications
and squarings for collective inversions and synchronization. It is a materially
different algorithm, not an instruction-count claim about a known winner.

For each round the product/reverse tree uses 3*(128-1) field multiplications
per block. The point update uses three multiplications per candidate, including
the square implemented as a multiplication. An ordinary round therefore costs
approximately 5.977 multiplications per candidate plus one root inversion per
128 lanes, before carry operations, normalization, loads, stores and barriers.
Across fourteen rounds that is about 83.67 multiplications plus fourteen shared
inversions. This arithmetic count must not be converted directly into a score.
The roots serialize tree phases and most lanes are idle at small levels.

## Exact arithmetic and exceptional points

The new affine path uses full-carry canonical addition and subtraction.
It does not add speculative first-fold or short-carry cuts. Its multiplication
uses the inherited exact qsb_field_mul and normalizes the result where affine
formulas require a canonical field value. The actual inline PTX add/sub
sequences are separately interpreted and checked against integer arithmetic.

The signed recoder and GTable selections retain the predecessor's identities.
The copied decoder changes only which shared arena holds its digit planes.
The table's offset ordinate is normalized and the exact offset subtracted
before affine arithmetic. Table selection and sign action are tested against
independent OpenSSL points, including the scalar boundaries 0, 1, n-1, n and
2^256-1.

The code handles an infinity accumulator by selecting Q, equal points by
doubling, opposite points by producing infinity, and doubling with zero
ordinate by producing infinity. Exceptional lanes contribute a neutral
denominator of one so that no zero enters the collective inversion.
Inactive lanes also participate in every collective and remain masked from
publication. There is no early return that strands active threads at a barrier.

The completed affine point is exported with U=V=1, or U=V=0 for infinity.
When QSB_NEG_Y_MAC is enabled, the ordinate is negated at the checkpoint
boundary so the inherited packed recovery convention remains consistent.
The subsequent recovery, sparse parity replay, both recid interpretations,
OpenSSL publication gate and hit writer are unchanged.

## Shared arena and synchronization

The affine route reserves one 16 KiB shared arena per block. Its first 7.5 KiB
stores fifteen planes of signed digit codes; a 0.5 KiB pad separates them from
an 8 KiB destructive product tree. Once the comb finishes, the existing
12 KiB cofactor traversal reuses the same arena. PackedRecovery.cuh gained only
a compile-time arena selection; its arithmetic body remains unchanged.
The promoted cofactor_checkpoint.h stays byte identical.

Each parent thread loads both children before overwriting either in the
reverse traversal. This avoids sibling read/write races without another
product buffer. The upward tree uses a block barrier while multiple warps
produce data, then warp synchronization for levels wholly inside warp zero.
The downward tree restores block barriers before the reader set expands past
one warp. For the supported 128-lane geometry, each inverse has four full
block barriers, instead of seventeen with the conservative all-block variant.
The next round synchronizes all newly written leaves before reading them.
A full barrier already guards the final handoff to cofactor recovery.

QSB_AFFINE_WARP_SYNC=0 restores full-block synchronization at each level.
The host collective test uses separate warp and block barriers to exercise
this ownership schedule. It is not a hardware race sanitizer and cannot prove
all CUDA execution behavior on its own.

The root uses the promoted four-owner delayed-divstep inverse. Only lanes
0..3 join its shuffle operations, with mask 15. This removes the serial
_ModInv call stack from the new main search kernel. QSB_AFFINE_QUAD=0 retains
the serial inverse as a research comparison; it has a call stack and is not
the selected release path.

## Integration and conservative runtime choice

Only the scalar-producer consumer, pipeline stage 3, invokes the new affine
comb. The fused stage 0 still uses the predecessor's projective negative-Y
chain. The SHA producer, saved-state allocation, host slot streams, replay
queue and final slot drain are unchanged. The new shared arena is chosen at
compile time, so the fused route does not reserve it.

The inherited adaptive policy compares ordinary distinct search batches,
including host verification and writeout after all slots drain. It warms both
routes and compares eight cohorts in ABBA BAAB order. The produced route is
selected only if its geometric mean gain is at least 1.015, at least three
paired ratios exceed 1.01 and none is below 0.995. Here that route now includes
the affine comb. The fused reference includes our prior unpromoted negative-Y
and parity work; this is not a direct 1.5% test against the promoted winner.

The selector preserves real search work and all verified hits. It does not
alter the harness clock, scoring or verifier. Trial batches can still reduce
the final score if affine execution is slow. Keeping a fallback is not enough
to call this a confident leaderboard submission. The calibration/reference
limitation remains explicit.

## Compiler results and rejected screens

CUDA 12.6.20 compiled native sm_89 and compute_52 PTX reassembled for sm_89.
The latter checks an additional code-generation path, not the exact production
driver JIT. The current default uses a four-block bound for the EC consumer.
The two compilations agree on the search kernel census:

| Kernel | Registers | Shared bytes | Stack / spill stores / spill loads |
|---|---:|---:|---:|
| Fused projective prepare | 126 | 12288 | 0 / 0 / 0 |
| Affine produced prepare | 108 | 16384 | 0 / 0 / 0 |
| Scalar SHA producer | 40 | 0 | 0 / 0 / 0 |
| Recovery finish | 62 | 0 | 0 / 0 / 0 |
| Sparse parity replay | 72 | 0 | 0 / 0 / 0 |

The inherited GTable builder and super-root inversion retain a 120-byte call
stack and zero spills. They are not described as stack-free. CUDA-CENSUS.json
records all kernels and hashes of external compiler/SASS artifacts.

A five-block EC bound reduced the affine register cap to 96 but introduced
an 8-byte stack with 8-byte spill stores and 8-byte spill loads in the final
screen. It is rejected. Earlier attempts to tighten the old projective bound
also spilled. The default remains four blocks. Initial experiments using only
QSB_S0_BLOCKS were overridden by the source geometry and were not valid tests;
QSB_PIN_EC_BLOCKS now supplies an independent stage-3 bound. No gain is claimed
from a flag that did not actually change the compiler result.

## Correctness evidence

test_affine_block.py executes the actual affine control flow with 128 CPU
threads, independent OpenSSL field arithmetic and a virtual OpenSSL GTable.
It passed 384 complete collective comb comparisons: two groups of scalar
boundary/random cases, then a group forcing doubling, opposite points and
temporary infinity. It also passed 80,000 canonical add/sub and alias cases.

That test's root inverse is an OpenSSL shim. It does not emulate the copied
four-owner CUDA arithmetic. The test separately checks that both inverse
headers match the promoted source byte for byte. Device field instructions,
shuffle scheduling and target performance remain separate evidence limits.

test_affine_field_ptx.py executes an interpreter over the actual inline PTX
add/sub strings. It passed 42,048 directed/random comparisons, including carry
and borrow boundaries. It covers the device instructions that the CPU fallback
in the complete-comb test does not execute.

The unchanged integration tests were rerun: test_sha_producer.py checked
2,712 SHA scalar digests and 66 buffer-order cases; test_parity_replay.py
checked 111,190 finish executions, 17,990 replay records and 388 matching hits;
test_sha_path_tuning.py checked nine synthetic decisions and ninety fully
drained cohorts. Synthetic rates are not measured performance.

Native QSB_PIN_AFFINE=0 and QSB_AFFINE_WARP_SYNC=0 builds passed. The affine-off
EC consumer returns to 122 registers and 12 KiB shared with zero stack/spills.
All Python checks use -B and temporary build directories outside the archive.

## Build, expectations and release status

    nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 -Xptxas=-v -o /tmp/pinning candidates/pinning/pinning.cu -lcrypto -lm

QSB_PIN_AFFINE=0 restores the predecessor's produced projective route.
QSB_SHA_PRODUCER=0 retains fused execution only. The inherited parity-replay,
narrow-window and negative-Y switches remain available for controlled
comparisons. Keep exact publication verification enabled.

The last refreshed frontier was 805,428,058/s and its 100-bips floor was
813,482,339/s. No numerical score prediction is justified by these local tests.
The opportunity is fewer point-combination field operations; the principal
risk is the latency of fourteen dependent collective inversions. Shared-memory
traffic, canonical normalization and inactive lanes at upper tree levels may
outweigh the saving even though the kernel has zero spills.

The predecessor has now been rejected at 754,794,582 candidates/s, 108,053
verified hits, approximately 6.29% below the unchanged 805,428,058 leader.
This is a regression of the whole predecessor package; the aggregate result
does not isolate negative-Y arithmetic, parity replay or scalar splitting.
The affine successor retains that package's fused fallback and finish, so it
inherits this performance risk. Its local correctness/compiler checks do not
provide evidence that the new comb recovers the loss.

The solver requested this already-built affine candidate be submitted as the
next official experiment while the regression is investigated separately.
The algorithm and tested defaults remain unchanged. Archive checks pass,
the live frontier was refreshed, and no claimed score is supplied. This is
an explicit experimental evaluation, not a claim of confident promotion.
The next independent pinning investigation returns to the promoted baseline
to avoid treating the rejected composite as a known-good reference.
