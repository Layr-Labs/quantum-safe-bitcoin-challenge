# Subset: exact negative-Y arithmetic and productive SHA schedule selection

Effort: max

## Scope and ancestry

This candidate combines an exact negative deferred-ordinate multiply-add with
four complete ways to schedule the same SHA and curve work. A bounded initial
comparison chooses a schedule using productive batches of the actual input.
The implementation is a performance experiment. It has independent host
correctness and complete CUDA compiler evidence; it has no local NVIDIA timing
or device execution result. Only the official evaluation can establish its
verified throughput and eligibility for promotion.

The implementation baseline is public main
`7c3609b87b9d8e094a16be148fe846dfd5ac7807`, whose subset source is
`9ac2515450446dbadbe061e98ebfc317c36d4999`. That subset promotion belongs to
Akashneelesh's job `7aef224a-e3ff-43f9-9877-50cdbda3f653` and recorded
623,518,629 verified candidates per second. This is the baseline's historical
result, not a result for this candidate. Earlier source notices and attribution
are retained. All changes are confined to `candidates/subset`.

Additional sources and credit:

- chengxuandi, PR1047, source `902d48c00d597a00f76163713a7870e36caee5cd`:
  an independent staged-SHA design, explicit buffer interfaces and a useful
  device comparison for that family. Its official job
  `a6bc32dd-9200-48b8-919f-8c6a71c57a02` was subsequently rejected at
  581,482,329 verified candidates per second, below the 623,518,629 frontier.
  That receipt is evidence against assuming staging improves the ranked
  workload. This candidate uses a different producer and arithmetic, and
  retains a productive measured choice with a fused fallback; none of those
  differences establishes a gain before this archive is evaluated.
  <https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1047>
- Saviour1001, PR1072, source `50fda34b2819c350fdda8939c10c648a76c5dc1c`:
  negative deferred-Y multiply-add arithmetic, dense single-candidate SHA
  production, reconstruction of cheap consumer identities, and the source lead
  for selecting schedules using productive batches. The arithmetic port was
  independently checked; the tile ownership and four-route policy here are
  different implementations.
  <https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1072>

Chengxuandi and Saviour1001 are credited as coauthors for this composition. Their notes
and compiler observations do not constitute an official result for this code.
No previous result from this account is attributed to the unsubmitted changes.

## Exact arithmetic changes

The deferred ordinate previously represented the actual ordinate as
`Yactual = Ycore - Yoff * ZZZ (mod p)`, for secp256k1's
`p = 2^256 - 2^32 - 977`. The adapted representation stores `N = -Ycore`, so
`Yactual = -N - Yoff * ZZZ`. The next slope numerator becomes
`(Y2 + Yoff) * ZZZ + N`. Its addition can be seeded directly into the low words
of the raw 512-bit product. For unsigned 256-bit operands, `a*b+c < 2^512`;
the implementation propagates the introduced carries before the existing
secp256k1 reduction. The final deferred subtraction is reversed consistently,
and the positive-Y convention is restored once at the chain boundary before
the existing affine resolution. The exact replay implementation is unchanged.

The SHA implementation is restored byte for byte from the promoted baseline.
An earlier combination included DPZZxlz's exact SHA integer multiply-add change
from PR1046, source `b16a873572dd75cd12c0f05ebaa9cceffdbd2100`.
Its official job `c4fe4f41-4bcb-4043-bef5-07f029af5aff` was rejected at
592,373,803 verified candidates per second, below the 623,518,629 frontier.
That public cross-run result is evidence against carrying the change forward;
it is not a matched estimate of its isolated causal effect. The SHA multiply-add
change is absent from this archive. The earlier experiment is credited here to
make the removal and its provenance clear:
<https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1046>

A preceding submission from this account combined the paired SHA producer,
full-batch staging and SHA multiply-add changes. Its job
`bada91dc-af6c-4ddc-9848-68b0176ffc28` (PR1057, source
`d31978c5290b327bcd46f161b837a4735cc4f1ec`) passed 83,170 independently
verified hits and was rejected at 580,878,662 candidates per second on the
RTX 4090, seed 330636717, N24, over 1,201.0779 seconds. Its verified epoch
coverage implied a lower-bound rate of 581,308,617 candidates per second,
consistent with its hit-derived score. This establishes a loss for that
combination; it does not isolate the cause. That archive did not contain the
negative-Y arithmetic, dense single producer, tiled routes or productive
selection introduced here. The current archive restores the promoted SHA and
can select fused execution, but still has no established positive performance
margin. No score from the preceding submission is assigned to this archive.
<https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1057>

The scalar remains the complete SHA256d value in four little-endian uint64
limbs. The existing preimage construction, prefix states, 128 window choices,
fixed-base tables, recovery choices, curve filter and exact hit replay retain
their mathematical meaning. This change introduces no new probabilistic field
shortcut. The inherited filter retains its existing behavior and limitations.
Every published hit still follows its independent exact replay path.

## Dense SHA production and scratch layout

The selected producer uses 128 threads per block and launch bounds of twelve
blocks per SM. Each thread computes one epoch/window scalar using the existing
scheduled-window hash and second-SHA helper. It writes four planes of 128 uint64
values per epoch. This retains our per-epoch scalar layout; it does not use the
donor's global eight-plane layout. The alternative paired producer remains
available behind its compile-time switch but is not an arm of the default
runtime comparison.

Consumer addresses and nomination identities can be reconstructed from CUDA
thread and block indices after long field calls. This avoids keeping those
inexpensive values alive across the calls. The staged consumer still runs with
256 threads and the original paired-epoch geometry. Four epochs, each with
128 window choices, belong to a full consumer block.

A full nominal batch has 262,144 consumer blocks, 1,048,576 epochs and
134,217,728 candidate preimages. Full-batch staging uses 4 GiB of scalar scratch.
The tiled routes instead use two buffers of 2,048 epochs each: 8 MiB per buffer,
16 MiB total. A full tile has 512 consumer blocks. The logical scalar handoff
still writes and reads 32 bytes per candidate. Reduced allocation does not
prove reduced DRAM traffic, L2 residency or faster execution.

## Four productive schedules

The default executable includes these routes:

0. Fused hashing and curve work in one consumer kernel, with the exact
   negative-Y arithmetic described above and the promoted SHA implementation.
1. Dense SHA production for the complete batch, followed by curve consumption
   on the default stream.
2. Dense SHA production and consumption in bounded tiles, serialized on the
   default stream.
3. Bounded tiles with a nonblocking SHA producer stream and event dependencies
   that permit overlap with default-stream curve work.

The first-state producer runs before these alternatives. All consumers in a
batch use the same hit buffer in default-stream order. Exact replay and normal
publication occur once after the complete original batch, regardless of route.
Tuning does not rewind an epoch, duplicate a candidate range, run a different
problem, change the work counter, or discard its valid hits.

There are two warmup batches for each route, followed by three rounds of the
symmetric order `0,1,2,3,3,2,1,0`. Each measured phase consists of three full
productive batches. The comparison therefore uses eighty batches total,
twenty per route. Monotonic host timing includes epoch and first-state
production, SHA, curve work, exact replay, readback and normal hit publication.
The comparison uses elapsed time per searched preimage. A non-fused route must
beat the paired fused observations in every round and improve the mean
normalized time by at least one percent; otherwise the fused route is retained.
Among qualifying routes, the lowest mean time is selected.

This is a bounded engineering selection rule, not a confidence interval or a
proof of sustained superiority. Symmetric ordering reduces a simple linear
trend's effect; nonlinear drift, interference, input-dependent phase costs and
noise can still cause a poor selection. The choice is made once. Unused scalar
allocations are freed after selection, after the completed batch's readback.
Invalid timing falls back to the fused route. Allocation errors are reported
rather than being interpreted as successful tests.

## Tile dependency and tail rules

The producer stream first waits for completed first states. Before reusing one
of its two scalar buffers, it waits for the previous consumer of that buffer.
After writing a tile it records scalar readiness; the default stream waits for
that record before consuming the tile. The consumer records completion before
the slot is reused. Event waits capture the record current when the wait is
issued. Later event records do not retarget an already submitted wait, as
specified by the CUDA runtime:
<https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__EVENT.html>

The producer stream uses `cudaStreamNonBlocking` so the legacy default stream
does not add implicit synchronization that defeats the intended overlap:
<https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/stream-sync-behavior.html>
These API guarantees establish the intended dependency graph, not achieved
concurrent execution. Two curve blocks already consume the full SM register
budget, so SHA residency can displace curve work. Extra launches and events can
also make tiled routes slower.

Tiles shift the epoch and first-state pointers locally. The consumer adds the
tile's original batch offset only when encoding a nominated hit. Exact replay
therefore still indexes the original complete batch. Full tiles preserve the
four-epoch grouping. Partial final tiles keep all collective lanes present;
inactive lanes use the inherited safe aliases and cannot publish nominations.
The last odd epoch does not read or write a nonexistent second scalar.

## Correctness evidence and limits

The arithmetic port was checked against independent integer and group oracles.
The actual extracted PTX multiply-add passed 20,560 raw product cases in each
of two carry variants, 41,120 checks total. The corpus included boundaries,
bit patterns and fresh random values. It also passed 4,096 random PTX point
states and 256 complete fifteen-addend chains generated from fresh
libsecp256k1 points and compared with an independent affine group oracle.
Extracted-source OpenSSL checks with undefined-behavior and bounds sanitizers
passed 2,048 positive/negative chains and 1,024 recovered points.

Before removing the SHA multiply-add change, complete-workload checks used
synthetic seed `202609221919`. All four
forced routes matched the promoted control's complete hit set: eight hits each
in the ordinary two-batch case and eight each at the actual final odd tail.
Four additional tests exercised the entire eighty-batch policy, then two more
batches after selection. Synthetic timings in those isolated host projections
forced each possible final selection; each test matched forty-three control
hits. AddressSanitizer was enabled, and those projections performed real host
scratch frees to check subsequent use after selection.

Across those twelve tests, 240,640 scalar outputs were compared with independent
hashlib SHA256d reconstruction of the complete 9,906-byte preimage. The oracle
independently unranked early omissions, verified window identities and checked
unique epoch/window coverage within each run. Inputs repeat across alternative
routes to make their comparisons exact; 240,640 is the number of checked
outputs, not the number of independent input fixtures.

After that removal, a new synthetic seed `202609222020` exercised every route
again. Each of the four full-host tests matched nine control hits, with no
candidate-only or control-only result. All 36,864 scalar outputs across those
four comparisons matched independent complete-preimage SHA256d reconstruction.
Only `GPUHash.h` changed between these two validation sets, and its new bytes
match the promoted baseline exactly. The field helpers, route policy, tile loop,
work counters and output code are unchanged, so the earlier targeted policy,
curve and dependency checks apply to those same implementations.

A separate test extracted the actual host tile loop and modeled CUDA's event
capture rules. It checked ownership and happens-before edges across ordinary,
small-tail and nominal full-capacity batches, including repeated slot reuse.
Both serialized and nonblocking-stream cases passed adversarial and random
legal schedules. Deliberately removing the first-state wait, removing the reuse
wait or supplying an incorrect global hit offset caused rejection. This model
is not GPU racecheck and does not execute CUDA memory operations.

The selection policy passed 213 synthetic cases covering each winner, losing
routes, ties, subthreshold improvements, linear drift, bounded noise, misleading
outliers, invalid durations and work conservation. Synthetic timings validate
the policy's behavior; they are not timing evidence for a GPU route.

## Complete compiler evidence and ablations

CUDA 12.8.93 compiled and linked the complete default source using
`nvcc -O3 -DQSB_ZEROS_N=24 -o subset_cuda subset.cu -lcrypto -lm`.
Separate inspection emitted default frontend PTX, assembled it for `sm_89` and
disassembled the result. The fused and staged specializations each use 128 registers. Both report 49,152 bytes of shared memory and zero stack
frame, spill stores and spill loads. The dense producer uses 39 registers with
zero stack or spills. These are generated-code properties, not measured
occupancy, device correctness, elapsed time or verified throughput.

With SHA staging, runtime selection and the negative-Y MAC disabled, the
complete cubin is byte-identical to the promoted control. Its SHA256 digest is
`994014fb153b73eb4338b013d373734dc8350041378b2c39ed0199987cc38ae8`.
Earlier composition checks also recovered the fixed tile, serial dense and
positive-Y variants byte for byte as mechanisms were removed. These are
isolation checks; they do not make separately observed gains additive.

The negative-Y composition shortened a repeated EC backedge from 1,059 to
1,044 static instructions in the local disassembly, while the enclosing front
function's total static count changed from 3,415 to 3,416. It would be incorrect
to describe this as a measured throughput gain or a reduction in total static
function size. The complete official workload remains the decisive test.

## Submission identity

Runtime differences from the stated baseline are confined to
`hit_filter_field_sc.cuh`, `tests/gpu_epochs/tree.cu`, and the new
`sha_stage.cuh`, `sha_tile_pipe.cuh` and `schedule_tuning.cuh` headers in that
directory. `SOURCE-MANIFEST.json` records hashes of the selected source files,
including this note. The evaluator, wrapper, problem generator, scoring rule,
counting semantics and output contract are unchanged. No executable, credential,
trace, machine-specific path or local research history is part of this archive.
The official result, if any, must be associated with the exact uploaded source
and server-issued job receipt.
