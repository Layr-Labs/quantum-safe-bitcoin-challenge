# Subset: dense single/paired SHA producers, streaming scalars and negative-Y MAC

## Source and attribution

This candidate starts from shared main 7c3609b87b9d8e094a16be148fe846dfd5ac7807. Its subset source is submission 7aef224a-e3ff-43f9-9877-50cdbda3f653, promoted at 9ac2515450446dbadbe061e98ebfc317c36d4999 by Akashneelesh. The current recorded score is 623,518,629 verified candidates/second. The 100-bips floor is ceil(623518629*1.01) = 629753816. Origin/main remained at 7c3609b when this package was prepared.

The final implementation milestone is b8f921b609fdc6b8947da5bfe7509c0b97745a81, following paired-producer checkpoint 67052be and arithmetic checkpoint 922ef84. This work was prepared with GPT 6 Astra, xhigh effort, using Codex. All changes belong to candidates/subset. The benchmark manifest, scoring, harness, fixed problem inputs, generator, workflows and sibling track are unchanged. The archive contains source and documentation, not compiled binaries or claimed score files.

All upstream source notices and the VanitySearch GPLv3 license are retained. Existing point-chain, epoch, SHA, inverse, recovery and verification contributions remain attributed in their source files. The producers invoke the promoted single and paired SHA functions without changing their rounds. The negative-Y representation and timing policy extend our own pinning research. That pinning submission is still pending and supplies no measured performance evidence for subset. No substantial additional unpromoted implementation from another solver is introduced, so this submission claims no new coauthor.

## Motivation and prior evidence

Our previous subset host/startup/SHA composition verified but scored 616,008,463/s against the 623,518,629/s frontier. Its historical donor-ratio projection of about 632.3M/s was wrong. That projection was never a local GPU measurement. This candidate therefore avoids estimating throughput by multiplying unrelated public ratios or treating instruction counts as elapsed time.

The promoted kernel performs the paired remaining SHA calculations in the same resource-constrained context as its curve computation. The compiled block has 256 threads, 128 registers per thread and 49,152 bytes of shared memory. Two new scalar producers execute existing SHA calculations separately: the paired version uses 64 registers, and the single-candidate version uses 39, both with no shared memory. The hypothesis is that more concurrent SHA execution can outweigh added global-memory traffic and another kernel launch. A smaller change removes work from the repeated curve loop. The implementation compares the combined candidate with promoted arithmetic during real search batches on the target GPU.

## Negative deferred ordinate

The promoted representation satisfies Yactual = Ycore - Yoff*ZZZ. The new representation stores N = -Ycore, so Yactual = -N - Yoff*ZZZ. Its next slope numerator is R = (Y2 + Yoff)*ZZZ + N modulo p, where p = 2^256 - 2^32 - 977.

Instead of multiplying, reducing and subtracting Ycore, the f2 multiply seeds N into the low four 64-bit even accumulators. Its outgoing carry is included in e4. For arbitrary 256-bit a, b and c, the maximum a*b+c is 2^512-2^256, so the raw integer operation fits exactly in 512 bits. The remaining promoted product and reduction sequence is retained.

The final deferred product changes R*(Q-Xnew) into R*(Xnew-Q), and the affine seed reverses its corresponding subtraction. One negation after the last point addition restores the positive deferred ordinate before the existing affine-Y resolution. Recovery therefore receives the original coordinate convention. Exact replay and the gate authorizing hit publication are unchanged.

This is still the inherited speculative filter architecture. Existing short-carry and first-fold approximations remain; the newly seeded raw integer product is exact. Unexpected nominations must survive independent exact verification before publication. Lost nominations would lower the actual score. The flag is QSB_SUBSET_NEG_Y_MAC. The isolated repeated loop contains 1,044 static SASS instructions versus 1,058 with the flag disabled. That observation is not a throughput measurement.

## SHA producers, register pressure and memory layout

kernel_subset_scalars invokes the existing qsb_pair_epoch_z_value. kernel_subset_scalars_single instead invokes the promoted qsb_scheduled_window_hash and qsb_pair_second_sha_z for one epoch/window per thread. The single version launches four 128-thread blocks for each 256-thread consumer block: both producers fill exactly 512 scalar records per consumer block, including safe inactive aliases.

A screening build using launch_bounds(128,8) needed 47 registers. Changing the minimum-block target to 12 reduced allocation to 39 with the same static instruction count, no spills and no shared memory. Under the documented [Ada register, warp and block limits](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#occupancy), the single producer fits 48 resident warps by these resource limits, compared with 32 for the paired producer and 16 for the fused curve/SHA block. This is a resource calculation, not a device occupancy measurement or a speedup factor. Dependent instructions, bandwidth, scheduling and launch overhead still matter.

 Each consumer block owns eight planes of 256 uint64 words. Planes 0..3 hold epoch A's scalar and planes 4..7 hold B's scalar. Word index is ((size_t)block*8 + word)*256 + thread. Adjacent threads therefore use adjacent addresses.

At the default 262,144 blocks, the allocation is 4,294,967,296 bytes. The producer writes 32 bytes per searched candidate and the consumer reads 32 bytes. Both accesses use PTX .cs hints, which compile to STG.E.EF and LDG.E.EF in native and reassembled sm_89 SASS. The intent is to evict transient scalar data before reusable curve-table data. The [PTX cache-operator documentation](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html#cache-operators) describes the streaming eviction policy; it does not guarantee retention of the table or eliminate traffic. QSB_SUBSET_SCALAR_STREAM=0 restores normal accesses. First states, epoch descriptors, lookup tables and verification buffers are preserved. If the optional allocation fails, execution falls back to the fused path.

The producer retains the original epoch/lane mapping. An odd B tail aliases A safely; an inactive A group aliases epoch zero. These aliases are masked from the searched-candidate count and nomination path. The tests cover empty, odd, block-boundary and multi-block tails, guard words and exact active-identity coverage.

B remains in the immutable global scalar buffer until its curve calculation needs it. The produced path avoids parking B through shared memory. The fused reference retains its original parking. Both paths invoke the same exact hit-verification kernel, with original first states and epoch descriptors. Their dispatch arguments, searched ranges and output buffers are identical.

The first combined build spilled 20 bytes of stores and 16 bytes of loads in the consumer. Loading B later alone did not solve that; whole-front inlining also failed. The successful change reloads inexpensive thread/block identity after long field calls and reconstructs nomination identity only on the cold hit path. Volatile CUDA special-register reads prevent early hoisting. The submitted defaults have no local-memory spill instructions in either search route.

## Target-device comparison

baseline_filter_control.cuh contains the promoted point seed, deferred add, final resolve and chain bodies. Only function-name suffixes differ. A source test compares all four bodies with 7c3609b. The fused reference uses these positive-Y bodies, and the produced candidate uses the new negative-Y bodies. SHA, table geometry, inverse, recovery finish and exact publication remain shared.

QSB_SUBSET_SHA_AUTOTUNE defaults on. Two independent trials compare paired production against fused, then single production against fused. In each trial, two batches warm the fused route and two warm the candidate route. Eight four-batch measurement cohorts then run F,S,S,F,S,F,F,S. Thus the complete selection spans 72 ordinary search batches. Every batch searches a new range and every completed hit is verified and written normally. No benchmark-only candidate repetitions, discarded hits, altered score inputs or harness timing changes are involved.

CLOCK_MONOTONIC timing spans normal epoch and first-state production, optional scalar production, curve computation, exact verification, blocking D2H copy and host output writes. Rates use actual candidate counts, including short batches. Decisions happen only between completed batches. There is no asynchronous pending slot to drain or reinterpret. An additional hit-tail D2H error is checked before those bytes can be published.

Each trial forms four adjacent candidate/reference ratios. A producer is eligible only when their geometric mean is at least 1.025, at least three ratios exceed 1.02, and none falls below 0.995. Invalid or inconsistent timings make that producer ineligible. The eligible producer with the larger aggregate gain wins; a tie retains paired production. If neither passes, select the promoted fused reference. There is no upper cap on the gain being sought. The chosen path is then fixed and cohort clocks stop. Diagnostics print each trial's ratios, aggregate and eligibility, then the selected route.

This is a runtime safeguard, not a guarantee of promotion. Startup, JIT, finite-sample timing noise, thermal variation and verified-hit sampling can affect the full-run score. If the candidate does not win its comparison, it may finish near baseline and be rejected. There is no defensible offline expected score. The 2.5% selection guard is not a performance projection: increasing it does not make the implementation faster. The performance opportunity comes from separating SHA resource usage, enabling the denser single-candidate producer and avoiding unnecessary scalar cache retention. The final score must still come from the ranked run.

## Local validation

The available machine is a Mac without an NVIDIA GPU. Builds use the local arm64 qsb-build:latest Docker image with CUDA 12.6.20. Compiled outputs are outside the editable tree. No local candidate-rate measurement is reported.

~~~sh
git diff --check 7c3609b
python3 -B candidates/subset/test_negymac_ptx.py
python3 -B candidates/subset/test_negymac_source.py
python3 -B candidates/subset/test_scalar_producer.py
python3 -B candidates/subset/test_subset_sha_tuning.py
nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 -Xptxas=-v \
  -o /tmp/subset-native candidates/subset/subset.cu -lcrypto -lm
nvcc -O3 -arch=compute_52 -DQSB_ZEROS_N=24 --ptx \
  -o /tmp/subset.ptx candidates/subset/subset.cu
ptxas -O3 -arch=sm_89 -v /tmp/subset.ptx -o /tmp/subset.cubin
cuobjdump --dump-sass /tmp/subset-native
~~~

The PTX audit executes the actual preprocessed inline assembly. Both lean and old carry variants pass 8,272 directed/random raw 512-bit MAC cases each. Four representation/carry combinations pass 1,024 random point updates in total. Another 256 complete 15-addend PTX chains agree with independent affine addition, with zero mismatches.

The extracted-source integration audit runs the actual seed, update and final resolve using exact CPU field primitives. Its 2,048 positive/negative chains agree, and 1,024 recovered affine points match OpenSSL sums. An early test failure came from accidentally reusing final ZZ/ZZZ as the next seed; the driver was fixed. A small-scalar fixture also hit an inherited incomplete-addition collision. The independent random-scalar fixture avoids that exceptional case. These tests complement the device assembly audit and do not emulate GPU execution timing.

The scalar audit first compares the complete output buffers of the production single and paired producers byte for byte. It then runs actual consumer loads against independent OpenSSL compression and hashing. It passes 36,864 comparisons for 128-window geometry and 70,656 for 256-window geometry, totaling 107,520 with zero mismatches. Fourteen batch shapes per geometry include empty and odd tails. Guard words remain intact, reversed consumer-block order is safe, and active identities cover each epoch/lane exactly once.

The policy audit runs with single production both enabled and disabled. Each setting passes 11 ratio-decision scenarios and 7 route-selection scenarios. It checks 73 final-batch boundaries with both trials and 37 with one, covering regression, marginal gains, noisy ratios, invalid times, ties, matching dispatch arguments, allocation fallback and completed-output ordering. C++ source audits use undefined-behavior and bounds sanitizers. Synthetic rates test decisions, not performance.

## Compiler census and rollback

Final default native sm_89 and compute_52 reassembled to sm_89 agree:

| Kernel | Registers | Shared bytes | Stack | Spill stores/loads |
|---|---:|---:|---:|---:|
| Fused promoted reference | 128 | 49,152 | 0 | 0 / 0 |
| Produced-scalar curve consumer | 128 | 49,152 | 0 | 0 / 0 |
| Paired SHA producer | 64 | 0 | 0 | 0 / 0 |
| Single-candidate SHA producer | 39 | 0 | 0 | 0 / 0 |

All four have zero LDL/STL instructions in the inspected native and reassembled SASS. Existing startup, epoch and exact-verification helper call stacks remain; not every helper has zero stack. CUDA-CENSUS.json records the final source milestone, register/stack/spill data and hashes of external compiler evidence. This compile check approximates the harness's PTX route but cannot promise identical driver JIT behavior.

Independent feature-disable builds all compiled successfully: QSB_SUBSET_SHA_SINGLE=0 restricts runtime selection to paired versus fused; QSB_SUBSET_SCALAR_STREAM=0 restores ordinary global loads/stores; QSB_SUBSET_SHA_PRODUCER=0 isolates negative-Y fused execution; QSB_SUBSET_SHA_AUTOTUNE=0 forces production if allocation succeeds; QSB_SUBSET_NEG_Y_MAC=0 restores the positive ordinate on the candidate path. That last counterfactual producer combination can spill and is not shipped or used as the runtime reference. The actual reference is the separate zero-spill promoted fused route. Disable both producer and negative-Y flags for a complete arithmetic rollback.

SOURCE-MANIFEST.json contains source hashes and actual attribution. The preflight checks every on-disk file, note and archive sizes, hashes and edit scope. YUKON-SUBMISSION.md preserves the upload procedure and earlier packaging/performance lessons. The actual ranked RTX 4090 run remains the authority for throughput and promotion.
