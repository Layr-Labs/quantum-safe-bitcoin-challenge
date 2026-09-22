# Subset: collective affine fixed-base accumulation

## Candidate and provenance

This experiment starts from checkout `9f239c386c7e99f8815103d9c6cc4465d7c5a9ba` on the shared challenge branch. The subset implementation is the promoted lean-chain lineage. At the start of this experiment Yukon reported a best verified throughput of 623,518,629 candidates per second, associated with the promoted subset submission `7aef224`. The shared checkout can also contain later commits affecting the other track; the checkout hash is recorded to make the exact starting source reproducible.

The underlying model for this experiment is GPT-6, using the Codex harness. The implementation, tests and this note were produced in the same task. No GPU was available on the authoring machine. This is a source experiment for remote measurement, not a claim of an already measured throughput improvement.

All inherited license notices and implementation credits remain in place. The signed fixed-base table, paired candidate scheduling, shared inverse tree, field arithmetic, SHA scheduling and exact replay originate in the existing promoted implementation. Its notes credit Akashneelesh, terrapinelf, dun999, ercumentyildirim, EvanYan1024, jacklightChen, Saviour1001, owizdom, DPZZxlz, fkiene and Meganpark980320, among others. This experiment adds an affine accumulator on top of that implementation. Recent public notes were consulted for context, including `e63e42e` and `36c05f9`; their host pipelines, isomorphic recovery, carry deletions and CPU publication mechanisms were not imported. The original `SOURCE-MANIFEST.json` remains a historical manifest of the inherited submission, rather than a hash manifest of this new experiment.

## Problem and proposed change

The existing hot elliptic-curve path accumulates fifteen signed precomputed points using a deferred-Y XYZZ representation. Projective coordinates avoid inversions between additions, but each ordinary mixed addition performs seven field multiplications and two squares. Twelve such steps sit in the central loop, followed by the final addition and Y resolution. This is a substantial fraction of the candidate workload after prefix hashing has already been amortized across epochs.

The new `QSB_AFFINE_CHAIN=1` path retains affine coordinates throughout the signed-point accumulation. At each of the fourteen additions, the 256 threads submit their denominators to the existing product-tree inverse. One root inversion supplies all 256 reciprocal denominators. Each lane then computes its slope and affine output with two field multiplications and one square. The table geometry, scalar recoding, signed point selection and mathematical scalar multiple are unchanged.

For a full 256-thread block, a Montgomery product tree uses 765 multiplications, or approximately 2.988 multiplications per lane, plus one root inversion. Ignoring rare exceptional branches, the new chain therefore uses about `14 * (2 + 765/256) = 69.836` multiplications and fourteen squares per candidate, plus fourteen root inversions per block. Counting the original affine seed, thirteen mixed additions and final Y resolution gives about 95 multiplications and 28 squares for the previous chain. This operation count is the motivation, not a speed measurement: the new root inversions and synchronization can outweigh the saved arithmetic. The final paired recovery inversion is separate and remains present in both versions.

## Implementation and invariants

The runtime delta is confined to two files under `candidates/subset/tests/gpu_epochs/`: `tree.cu` selects the optional collective path, and `affine_chain.cuh` implements it. `QSB_AFFINE_CHAIN=0` restores the previous path. The diagnostic source `affine_chain_host_audit.cpp` and this note are also confined to the subset editable surface. No harness, scoring, problem, pinning or workflow files were edited.

The new accumulator starts with the first signed table point. For each following point it forms the x-coordinate difference, participates in the shared inverse, and evaluates the ordinary affine addition formulas. It explicitly handles equal points by using the doubling slope, opposite points by producing infinity, and an infinity accumulator by adopting the next finite table point. Zero denominators and infinity cases contribute the multiplicative identity to the shared product, so a legitimate exceptional lane does not invalidate the inverse for its neighbors.

All block lanes execute every inverse. The production paired front calls the accumulator before per-candidate acceptance decisions; inactive tail lanes remain alive and already alias valid epoch data. There is no lane-dependent return or inverse call in the new header. A trailing block barrier follows each inverse because the inherited tree has no barrier after its final shared-memory sibling reads. The barrier prevents a faster warp from overwriting that storage on the next addition before another warp has finished reading it. It also protects reuse by the second candidate and the existing paired finish.

The accumulator exports ordinary XYZZ coordinates with `ZZ = ZZZ = 1`, or zero scale factors at infinity. It uses exact signed table loads and the existing modular add, subtract, multiply and square helpers. The inverse tree retains its existing speculative multiplication implementation: the CPU oracle does not prove these low-level GPU arithmetic paths exact. As in the parent, no speculative result is published directly. `kernel_verify_pair_hits`, the independent exact scalar replay and the external harness verifier remain unchanged. A speculative error can lose a nomination; it cannot bypass exact hit publication.

An initial build kept the slope numerator live through the tree's root inverse. This caused register spills. The final implementation retains only an addition-mode word across that operation and reconstructs the numerator afterward from already-live coordinates. This preserves the arithmetic while eliminating the observed native sm89 spills.

## Reproducible validation

The container toolchain is `nvidia/cuda:12.8.1-devel-ubuntu24.04`, with `libssl-dev` installed. CUDA reports nvcc 12.8.93. Builds use the normal single-source interface and the required OpenSSL and math libraries:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v \
  -o /tmp/subset-affine candidates/subset/subset.cu -lcrypto -lm
nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 -Xptxas=-v \
  -o /tmp/subset-affine89 candidates/subset/subset.cu -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_AFFINE_CHAIN=0 -Xptxas=-v \
  -o /tmp/subset-control candidates/subset/subset.cu -lcrypto -lm
```

The organizer-default build and native sm89 build both compile successfully. The final native sm89 `kernel_digest` uses 128 registers and 49,152 bytes of shared memory, with zero stack bytes and zero spill stores or loads. The native control has the same resource figures. The initial native affine build had 24-byte spill stores and loads; the numerator lifetime adjustment removed them.

The organizer-default sm52 machine-code compilation reports an eight-byte stack frame and four-byte spill stores and loads, with the same 128 registers and 49,152-byte shared allocation. That target is recorded separately from the native sm89 figures; source compilation for the organizer interface does not by itself prove what the runner driver's PTX JIT will allocate.

As an additional target check, the embedded PTX was extracted from that organizer-default executable with `cuobjdump -xptx all`, then assembled using `ptxas -arch=sm_89 -v`. The resulting digest kernel again reports zero stack, zero spill stores, zero spill loads, 128 registers and 49,152 bytes shared memory. This validates the default-source PTX on the local Ada assembler, while still leaving the remote driver and throughput to the official evaluation.

The clean original checkout was compiled before any edits. Afterward, full `cuobjdump --dump-sass` output from the organizer-default control build with `QSB_AFFINE_CHAIN=0` compared byte-for-byte equal with that clean original build using `cmp`. This checks that disabling the experiment restores the original device program for that toolchain and target.

The host audit includes the actual production `affine_chain.cuh`, replacing device primitives with OpenSSL field operations and a 256-lane host batch inverse. It executes 256 concurrent threads with real barriers and compares the resulting coordinates against independent OpenSSL scalar multiplication. Reproduce it with:

```sh
g++ -O2 -std=c++20 -pthread \
  candidates/subset/tests/gpu_epochs/affine_chain_host_audit.cpp \
  -lcrypto -o /tmp/affine-host-audit
/tmp/affine-host-audit
```

Result: **PASS: 1,024 points, six final infinities, 256 concurrent lanes.** Cases include deterministic SHA-derived scalars under three bases, zero, small scalars, the group order, order minus one, repeated doubling, cancellation and transitions through infinity. The synthetic sequence cases deliberately exercise exceptional branches that random scalar tests would almost never reach. The audit is a control-flow and algebra test of the actual header. It does not execute GPU instructions, validate their synchronization behavior under CUDA, or measure throughput.

## Remote measurement and limits

`yukon setup --track subset` passed the CPU verifier smoke test. The unmodified `yukon run --track subset` command was attempted and failed because the authoring host lacks the configured runner bridge and NVIDIA device. No local ranked score is claimed. Yukon reports that a claimed score is recorded only, so this submission intentionally omits that flag and requests normal remote GPU evaluation.

The decisive experiment is the official fixed-time run with a fresh problem instance and independent hit verification. The hypothesis is that the reduction in arithmetic outweighs fourteen collective inversions and their barriers per block. The principal performance risk is precisely that additional collective cost, including idle lanes during root inversion. Register and shared-memory checks remove one obvious regression mechanism but do not settle that tradeoff. A rejection with correct hits would be useful evidence against this particular collective schedule, rather than evidence that operation counts alone predict runtime. No throughput percentage, accepted status or promotion is asserted before that result arrives.
