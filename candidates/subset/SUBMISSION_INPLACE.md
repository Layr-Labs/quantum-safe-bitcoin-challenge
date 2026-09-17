# Subset: experimental in-place packed batch inversion

## Status and objective

This submission asks the remote GPU runner to evaluate an experimental
structural change to the subset candidate. The target is the promoted score
of 495,193,826 verified candidates per second. No higher local score is claimed.
There has been no local CUDA build, GPU execution, or independent hit
verification of this modified candidate. The development machine is macOS
ARM64 without nvcc or an NVIDIA GPU. Remote compilation and ranked evaluation
are required to determine whether the experiment is correct and beneficial.

The ranked entry point explicitly selects ZLAB_TREE=3. This is intentional:
submitting a new helper without selecting it would only measure the old code.
The user requested submission to the existing runner after the local host
checks, knowing that GPU validation remains pending. No claimed-score value
is attached, and neither the target nor a host-test result is represented as
a measured score for this submission.

## Starting point and provenance

The checkout begins at commit 372a3251e707616011ff6dc0c961cf944d565149 on
the challenge main branch. Yukon reported the current subset frontier as
submission 99ce841 at score 495193826. Its public note describes the promoted
field squaring, packed inverse layout, direct digit extraction, launch
geometry, and reduced host hit-path overhead. That promoted implementation
is the starting point retained here. This submission builds on promoted work;
it imports no additional unpromoted patch or external candidate.

The setup manifest is schema version 2. Only candidates/subset is edited.
The pinning candidate, problem generator, verifier, scoring harness, benchmark
configuration, setup scripts, workflows, and benchmark manifest are unchanged.
Third-party GPL notices and the existing COPYING file are retained.

## Change and hypothesis

The existing variant 2 allocates products[4][512] and inverses[4][256] in
shared memory, for 24,576 bytes inside the batch inversion helper. The proposed
variant 3 keeps only products[4][512], for 16,384 bytes. During the downward
pass, internal product slots become inverse slots once their old values have
been consumed. Leaves remain products until the final per-lane multiply.

This saves 8,192 bytes of source-declared shared storage per block in this
helper. It is not a measured reduction in the full kernel's final compiled
resource use: that depends on nvcc allocation and all other shared objects.
The hypothesis is that lower shared-memory pressure may allow better
occupancy or scheduling on the ranked GPU. The alternative explanation is
that the change loses performance because each active thread performs more
sequential work in the downward pass. The runner must resolve that tradeoff.

The field multiplier, normalization function, modular inverse, SHA routines,
elliptic-curve formulas, fixed-base table, candidate enumeration, recovery
IDs, validity gate, hit encoding, and search difficulty remain unchanged.
The experiment does not skip any candidate computation or cache valid hits.

## Algorithm and storage ownership

The upward pass uses the existing packed level arrangement. For a block of
n threads, leaves occupy indices 0 through n-1. The next level starts at n
and has n/2 products. Subsequent levels continue in packed order until the
two root children start at 2*n-4. Each multiplication has the same field
semantics as the promoted helper. The root is normalized before inversion.

Lane zero loads both root children, multiplies them, normalizes and inverts
their product, and forms the two child inverses. It writes those inverses
back to the former root-child slots. Both input values are in private
registers before either shared slot is overwritten.

For every subsequent internal downward level, one thread owns a pair of
children. It loads the parent inverse and both child products before any
write, then computes parent_inverse times right_product for the left child
and parent_inverse times left_product for the right child. The same thread
stores both results in the former child-product slots. No other thread at
that level needs the overwritten old values. This ownership is the key
structural difference from variant 2, where separate threads produce the
individual child inverses into a second array.

Barriers publish each level to its consumers. Warp barriers are used only
when the writers and the next internal readers reside in warp zero. Block
barriers are used when consumers cross warp boundaries and before the final
leaf stage reads internal inverses. Nonparticipating warps still execute
their corresponding warp barriers. The final stage reads untouched leaf
products and their parent inverses, computes each lane's inverse, and clears
the fifth limb. Supported block sizes are powers of two from 32 to 256.
Inactive tail lanes contribute the identity and participate in all barriers.

## Files and entry points

- tests/gpu_epochs/tree_inverse_inplace.cuh contains the new helper.
- tests/gpu_epochs/tree_inverse.cuh dispatches ZLAB_TREE=3 to that helper;
  existing variants remain available.
- subset.cu explicitly selects variant 3 for the fixed ranked build command.
- subset_inplace.cu provides an equivalent local experiment entry point.
- subset_baseline.cu explicitly selects variant 2 for local comparisons.
- tests/gpu_epochs/inverse_schedule_host.cpp executes the actual selected
  helper with host threads and emulated warp and block barriers.
- tests/gpu_epochs/test_inverse_storage.py checks the preprocessed helper's
  declared shared-memory allocation.
- INPLACE_INVERSE.md records the experiment, reproduction steps, and limits.

No prebuilt executable is included. The normal setup step must build the
candidate with the benchmark's fixed nvcc command and ranked difficulty.
All quoted candidate includes remain within the subset editable surface.

## Local verification and limitations

The storage test was run before the implementation. It failed because the
selected code declared 24,576 bytes instead of the required maximum 16,384.
After implementing variant 3, the same test passed. This establishes a
source-level resource property, not GPU speed or actual occupancy.

The host schedule audit includes the actual tree_inverse.cuh selection and
new helper rather than a separate reimplementation of the tree. It supplies
CPU threads, one reusable barrier for the block, and independent reusable
barriers for each warp. Substitute field operations use the prime 2^61-1
and redundant words to check propagation of all four array components.
They support the aliasing used by the helper. This audit intentionally does
not emulate PTX instructions or validate the secp256k1 field implementation.

For each of 32, 64, 128, and 256 threads, cases cover one active lane, all
but one active lane, and a full block. Inputs include identity padding and
values near the test prime. All 1,440 resulting lane inverses passed the
independent condition input times output equals one modulo the test prime.
The same audit passed for baseline variant 2. Variant 3 also passed with
ThreadSanitizer and no reported host data races. These are bounded host
checks; GPU synchronization and compiler behavior still require validation.

Reproduction from the benchmark directory:

```sh
python3 candidates/subset/tests/gpu_epochs/test_inverse_storage.py
clang++ -std=c++17 -O1 -g -pthread -fsanitize=thread -Wno-unknown-pragmas \
  candidates/subset/tests/gpu_epochs/inverse_schedule_host.cpp \
  -o /tmp/qsb-inverse-schedule-tsan
/tmp/qsb-inverse-schedule-tsan
git diff --check
```

## Remote evaluation and next steps

The initial unmodified local yukon setup --track subset completed its CPU
verifier smoke check. The unmodified local yukon run --track subset could
not execute the configured ranked bridge: the local machine lacks the bridge,
passwordless sudo configuration, CUDA compiler, and NVIDIA hardware. That
failure produced no candidate score and is not evidence for or against the
new inverse implementation.

The remote runner should build the selected variant and evaluate it using
the unchanged subset verifier and scoring settings. Additional GPU audits
are documented in INPLACE_INVERSE.md, using the existing tree_audit.cu with
-DZLAB_TREE=3 and CUDA synccheck/racecheck. Those audits have not been run
locally. Compilation failure, incorrect hits, or a slower score would reject
the current performance hypothesis; none should be hidden or treated as a
successful improvement. If performance is promising, matched baseline and
candidate runs plus fresh-seed repeats should establish reproducibility.

The authoritative result is the runner's verified score. Until that result
exists, this submission is an unmeasured optimization experiment whose
confirmed benefits are limited to the smaller declared storage footprint.
