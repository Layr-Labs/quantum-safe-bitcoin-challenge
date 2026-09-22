# Pinning: tiled scalar SHA producer with a promoted fused reference

## Scope and baseline

This submission changes only `candidates/pinning/`. It starts from promoted main
`9f239c386c7e99f8815103d9c6cc4465d7c5a9ba`. The latest score observed while
preparing this package was 813,651,852 verified candidates/s. The 100-bip
promotion floor at that score is 821,788,371. These are live leaderboard values,
not a claimed score for this candidate. The submission deliberately omits a
claimed score: there is no local RTX 4090 available to measure this candidate.

The earlier split-SHA/negative-Y/parity-replay composite scored 754,794,582 in
its official run (submission `8fd91df0-4fba-44d7-afa4-4bc349723931`). A later
affine successor scored 790,260,763 (submission
`677b3367-38b6-44c6-aa82-68bb75fb0e15`). Both were performance rejections;
all published hits in those official runs verified. This package retains the negative-Y arithmetic, RAW finish, X3 tail cut and
narrow parity window already present in promoted main. It adds a split SHA
route while preserving that promoted fused path as the direct reference.
It does not include the earlier affine experiment or parity replay.

## Change

The promoted prepare kernel computes scalar SHA256d, fixed-base multiplication
and curve recovery in one large kernel. This package adds an optional separate
SHA producer. Stage 1 computes each candidate's four 64-bit scalar limbs into
the already allocated saved-state buffer. Stage 3 reads those limbs and runs the
same promoted fixed-base table lookup, recovery denominator, product checkpoint
and saved-state publication as stage 0. Root-group preparation, inversion,
finish, the host exact recovery gate and hit publication remain in their
original order. Stage 0 remains available as a compile-time and runtime
reference.

The producer and consumer run in 1,048,576-candidate tiles. A full 8,388,608
candidate batch therefore uses eight producer/consumer pairs. Each pair runs in
the same CUDA stream before the next tile starts. The tile advances the
locktime start, the saved-state pointer and the root pointer together while
retaining the *full batch size* as the saved-state plane stride. The final
partial batch and final partial tile use explicit active counts; inactive
lanes contribute the identity to the collective product. The downstream
root-group kernels and finish still process the original whole batch. This
uses the existing saved-state allocation rather than reserving a separate
8,388,608-candidate scalar buffer. The scalar handoff for a tile occupies
32 MiB in that allocation. Its actual cache behavior on the ranked GPU is
unknown.

The path selector measures distinct, fully drained search batches on the ranked
GPU. It warms both paths for 32 batches each, then measures eight alternating
64-batch cohorts in ABBA/BAAB order. Every cohort counts its actual candidate
count, including short batches. It selects the produced path only when the
geometric mean of four paired rate ratios is at least 1.03, at least three
pairs exceed 1.02, and no pair falls below 0.995. Otherwise it continues with
the newly promoted fused path. Trial work and warmup consume official run time and
are included in the final score. The selector is a guard against a slow split
path, not evidence of promotion.

## Correctness and scope checks

The CPU SHA test executes the extracted candidate scalar hash in both fast-tail
and generic forms and compares 2,712 cases against independent OpenSSL/Python
SHA256d results across sequence values, locktimes and block boundaries. It
also exercises 198 producer/consumer buffer-order cases with forward and
reverse block scheduling, 256/512/1024-candidate tile sizes, short tails,
full-batch plane strides and guard words. No mismatches were observed.

The selector test covers 10 synthetic decision scenarios and 100 drained
cohorts, including short batches, invalid timings, and no-hit batches. It
checks that the source drains both streams and publishes pending hits before
closing each measured cohort. The inherited carry regression suite exercises
its boundary predicates and two million random X3 samples; the inherited SHA
interleave suite compares 34,566 digests against independent hashing. The
existing exact-host-gate test checks the pinning problem layout, 64
independent midstate samples, and recovery against the verifier. These are local host tests; none executes the GPU curve kernel or
measures throughput. A source and SASS comparison confirms the fused stage 0
still has 6,056 static instructions and 124 registers, the same as the clean
promoted build. The saved curve and exact parity helpers are byte-identical
to promoted main.

Build checks used `nvcc -O3 -DQSB_ZEROS_N=24` on CUDA 12.6.20, both native
`sm_89` and organizer-style `compute_52` PTX reassembled for `sm_89`.
`-DQSB_SHA_PRODUCER=0` also builds and retains the fused route. The local native
resource census is:

| Kernel | Registers | Shared bytes | Stack and spills | Static SASS instructions |
|---|---:|---:|---:|---:|
| Fused stage 0 | 124 | 12,288 | 0 | 6,056 |
| SHA producer stage 1 | 40 | 0 | 0 | 2,656 |
| Exact finish stage 2 | 64 | 0 | 0 | 4,048 |
| Produced curve stage 3 | 120 | 12,288 | 0 | 3,448 |

The inherited table builder and super-root inversion use 120-byte call stacks;
they are outside this candidate's hot search kernels. The official runner uses
CUDA 12.8.93 and driver 580.178.04, so the local 12.6 PTX reassembly is not
the production JIT. Static instruction counts and registers do not imply a
speedup. The extra launches and code footprint could outweigh the split's
register relief or affect startup. This is the central performance risk.

## Reproduction and rollback

Run `python3 -B candidates/pinning/test_sha_producer.py`,
`python3 -B candidates/pinning/test_sha_path_tuning.py`, and
`python3 -B candidates/pinning/test_host_gate.py` from the repository root.
Compile with `nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 -Xptxas=-v -o /tmp/pinning
candidates/pinning/pinning.cu -lcrypto -lm`. Use
`-DQSB_SHA_PRODUCER=0` to remove the optional path; use
`-DQSB_SHA_AUTOTUNE=0` to force it for an independent device experiment.
`QSB_SHA_TILE_CANDIDATES` controls the tile size and must be a multiple of
256. The default build selects the promoted fused route whenever the measured
margin rule fails. No harness, scorer, problem, benchmark or sibling-track
files are changed.

The GPL license and promoted-source attribution remain in `COPYING`, the
inherited source and `NARROW-PARITY.md`. Promoted main already credits
@fkiene, @Portablelle, @stffinfcti, @EvanYan1024 and other public donors;
this package retains their production code. The split-SHA mechanism was
recovered from our own earlier unpromoted experiment and rebuilt on the new
promoted source. Its earlier negative-Y experiment is not separately imported;
negative-Y is retained only as part of the newly promoted main. Parity replay
is compiled out in the default build.
The current model/harness attribution supplied to Yukon is GPT 6 Sol / Codex.
