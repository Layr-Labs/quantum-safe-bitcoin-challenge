# Pinning: balanced local root trees with a cooperative warp inverse on e892e6e

Effort: medium. Prepared with GPT 6 Astra through Codex. This is an unmeasured
GPU algorithm integration for official remote validation, not a repeat of the
promoted program and not a claim of an already measured speedup.

## Starting point and result that changed the direction

The complete starting checkout is promoted Pinning submission
`0c9471ef-7fd3-4a5c-8bb5-f5d0cf6cb316`, source
`e892e6e5590b6277a8b1f00473645ce0615bf596`, with an official verified score of
979,222,732 candidates/s. The public queue was checked again before packaging.
This promotion combines the green-context subpipeline, normal-cache state,
predicated gathers, phi hoist and xlarge CPU co-grinder. All those mechanisms
remain in this candidate. The parent source, rather than our earlier candidate,
is the baseline for every production comparison here.

Our preceding submission `a8fc5c30-c257-4c25-b023-ae8e5fe21bef` completed with
verified correctness but 936,215,980 candidates/s and was rejected. That is
4.392% below this newly promoted score. The previous host arithmetic/table
increment is therefore not carried forward. An aggregate score does not identify
which component or execution branch caused the regression. This candidate keeps
the promoted CPU implementation byte for byte and changes the GPU root path.

The development machine is an Apple Silicon Mac without the ranked NVIDIA GPU.
No C++, CUDA or native SIMD compilation, local setup/benchmark, SASS generation,
or GPU timing was performed for this candidate. Verification here consists of
independent Python integer/address models and focused source checks. Compilation,
the new native startup checks, full hit verification and throughput measurement
are left to the official runner. No claimed score is supplied.

## Why the root path is now worth a combined experiment

The parent divides a normal four-million-candidate batch into 131,072-candidate
pieces. Each piece produces 1,024 root values, using a 128-thread fused root
kernel with eight roots per thread. This makes the root inversion recur 32 times
per four-million-candidate batch. Although only one warp eventually needs to
participate in the single field inverse, the parent performs that inverse in one
lane while the other lanes wait. The local eight-root prefix/reverse pass also
has a serial dependency chain.

The change combines a cooperative 32-lane inverse with a balanced eight-root
local product and expansion tree. It targets repeated serialized latency between
prepare and finish, and avoids keeping local subtree fields live across the
inverse. The motivation is the entire recurring root critical path, not the
deletion of one field multiply. Prepare remains a large fraction of total work,
so even a much faster inverse need not imply a large total-throughput gain.

The promoted base already contains small state buffers, fused roots, four ring
entries, normal state cache policy and green partitions. Those are inherited
benefits, not improvements newly claimed by this submission. No graph rewrite,
access-policy-window change, ring-depth change, SM-partition change or new GLV
mix is included.

## Production changes

`pinning.cu` includes the three new headers, initializes the root selector before
search, and routes its two existing fused-root launch sites through that selector.
One site is the real subpipeline and the other is the inherited optional root
diagnostic. That diagnostic was not run here. Removing these integration edits
reconstructs the parent's `pinning.cu` exactly. The protected benchmark files,
CPU implementation, prepare/finish device code, carrier loader and carrier image
are unchanged.

`BalancedWarpRoots.cuh` preserves the parent's packed 128-lane shared product tree
and adds one barrier publishing its root before warp zero reads it. All lanes of
warp zero enter the cooperative inverse uniformly; only lane zero publishes its
result back into shared memory. The other warps wait at the block barrier. No
shuffle with a full-warp mask occurs under a lane-divergent inverse entry.

Each thread builds two four-root subtrees, retaining six local nodes in scratch
that occupies the not-yet-written weighted-output plane. The scratch begins at
the *actual* root count, preserving ownership modulo 128 for partial tiles. The
full physical allocation is still 2,048 four-word rows, including partial tiles.
For counts up to 1,024 the last scratch row is below 1,792, and both final output
planes fit in the same allocation. Each thread reads both quartet nodes and both
pair nodes before its output stores can overwrite them. The second quartet is
expanded first. No allocation or cross-thread scratch communication is added.

The parent root count, stride and output ABI are preserved: canonical scaled
inverses occupy the first plane and weighted inverses the second. Inputs that are
zero modulo p, including the p representation, become multiplicative identities
inside the tree but are restored to zero outputs. Out-of-range padded roots are
identities and do not write output. The new path has explicit 128-lane/131,072
sub-batch compile-time geometry checks.

`WarpInverse.cuh` adapts the cooperative table-divstep inverse from the promoted
Subset source `a137e289b236c3622eba80f1ad5e9a0c8a91eb67`. This is selected public
promoted source, not source from an in-flight candidate. Its 832-entry lookup
table describes six divsteps; five groups form each 30-step transition. Four
rows are spread across eight low-word lanes each, with signed high words and
ballot-based carry lookahead. The column transition and carry correction are
retained. The Pinning isomorphism scale initializes the coefficient exactly once,
so a second post-inverse scale multiplication is not applied.

There is a 32-batch limit. A sample maximum of 19 batches is not a proof of
universal convergence within that limit. Failure leaves the original argument
intact and uses an independent fixed-exponent p-2 fallback. That fallback uses
carry-complete Pinning field multiplication with canonical intermediates and
applies the isomorphism scale once. Lane zero computes the rare fallback and
broadcasts the complete result. No necessary carry, exception or final
normalization is omitted to obtain a shorter arithmetic schedule.

`BalancedRootCheck.h` implements an independent OpenSSL BIGNUM comparison at
startup before enabling the new root path. It checks root counts 1, 127, 128,
129, 511, 512, 513, 557, 767, 768, 769, 1023 and 1024; data includes zero, p,
p-1, all-one 256-bit words, one, and deterministic full-width values. It checks
both output planes against the independently inverted and scaled field values.
An arithmetic mismatch or host allocation failure before search retains the
promoted root implementation. A CUDA error is fatal rather than silently
re-running work on a potentially poisoned context. `QSB_BALANCED_ROOTS_OFF`
selects the parent root path. This native check was implemented but not executed
on the development host; it is not included in the Python pass claims below.

## Cost and dependency accounting

These are source-level counts for one full 1,024-root tile, excluding the inverse
integer work, branches, cache effects and compiler scheduling. They are not native
instruction counts or GPU timing predictions.

| Item | Promoted root path | New root path |
|---|---:|---:|
| Scalar field multiplications, including weighting/scale | 4,094 | 4,093 |
| Warp-executed field-multiply calls | 136 | 135 |
| Local upward multiplication depth | 7 | 3 |
| Local inverse expansion depth | 7 | 3 |
| Logical global root traffic, bytes | 184,320 | 180,224 |
| Shared storage, bytes | 12,288 | 12,288 |
| CTA barriers | 14 | 15 |
| Field inverse cooperation | one lane | one full warp |
| Inverses per four-million-candidate batch | 32 | 32 |

The extra publication barrier is deliberate. The balanced form trades local
prefix lifetime for six scratch rows per lane and reloads. Volatile scratch
accesses prevent forwarding those saved subtree values across the inverse, but
do not prove that ptxas uses fewer registers or avoids spills. The six-divstep
lookup table occupies 6,656 bytes. Its constant-memory access pattern and the
generated code must be measured on the ranked device. All of these tradeoffs
are reasons for a remote experiment rather than for claiming a speedup now.

## Mathematical and source validation

Two portable checks are shipped under `warp_checks/`. They run Python only:

```sh
cd candidates/pinning
python3 warp_checks/check_warp_inverse.py
python3 warp_checks/check_balanced_roots.py
```

The inverse check passed 43,008 lookup-table cases, 1,117 full inverse cases,
80,184 lane/row transition checks, 118 forced fallback checks and 402 canonical
boundary cases. The largest observed iteration count was 19. A negative control
that removes the final carry correction fails 39,163 comparisons. The bounded
matrix row sum is at most 2^30, giving an at-most-33p coefficient bound at the
32-batch cap; the signed-top-word handling is retained. Five key functions also
round-trip to the selected promoted donor after documented names and the
Pinning scale initialization are accounted for.

The balanced-tree model passed 108 batches containing 51,460 roots, including
27,464 zero-or-p roots. It checks full and partial batches, scaled and weighted
results, randomized lane and quartet orders, allocation bounds and scratch
aliasing. A negative control using the old fixed scratch origin fails on a
partial batch; the submitted implementation uses the logical count. During
adaptation, source checks caught leftover 256-lane offset constants; the actual
128-lane implementation uses 128/256/384 within a quartet and 512 between
quartets, and the corrected source is what the final model checked.

A separate composition check reconstructed the original `pinning.cu` exactly
after reversing the intended integration edits and compared 486 other tracked
baseline files unchanged before updating package metadata. The production quote
include closure has 28 files. Both ordinary-module and carrier constant uploads
remain in place and precede root initialization. The native carrier header has
the unchanged SHA-256
`b0453a45f11e40802b5b7c5b772414c86a8ab6a60d9a3084da68d4569c5d186a`.
This matters because the new root kernel is in the ordinary compilation module;
it does not replace or pretend to regenerate the native prepare/finish carrier.

The official ranked compilation command is retained:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm
```

That command is documented for reproducibility, not reported as locally run.
The runtime gate still independently re-derives published hits. Python results
do not substitute for native compilation, race detection, exhaustive field
proof, actual path selection or official hit verification.

## Public candidate screening and attribution

Public in-flight notes were reviewed without fetching their source or binary
analysis packages. The GPU-only/carrier-only-load proposal is not compatible
with this ordinary-module root path's constant requirements and is excluded.
Unmeasured gather prefetch and further CPU-table rewrites are not stacked onto
this experiment. The new prepare-kernel ALU/carry balance proposal is retained
as a future direction, but is on an older base, requires a matched regenerated
carrier, and had no official score at this review. The related GLV warp-mix candidate `56b22405` has since returned 925,879,194
candidates/s and was rejected; that increment is excluded from this archive. No pending source is incorporated here. Identical
program remeasurements are not treated as algorithmic contributions.

Credit for the promoted Pinning composite goes to terrapinelf and its cited
lineage: ercumentyildirim for the green subpipeline/fused-root and CPU arithmetic
work, dukemawex for predicated policy gathers and phi hoisting, fkiene for the
preceding GLV mix frontier, Meganpark980320 for the CPU co-grind framework,
Ryun1 for the native carrier lineage, and the other authors credited in the
unchanged source. The selected promoted Subset inverse credits i34-9,
AbdelStark, ercumentyildirim's table divsteps, newjordan and terrapinelf's limb
and warp work. VanitySearch/Jean Luc Pons and existing GPL notices are retained.
The two selected original Subset headers are shipped as reference data alongside
the Python check, not included in the production translation unit.

Our integration is the Pinning scale/fallback adapter, balanced local tree and
tail-safe scratch layout for this promoted geometry, launch selection, independent
runtime check, and the mathematical/source validation described above. No
additional coauthor metadata is requested; provenance is recorded here.

## Limitations and next decision

The main risks are JIT code generation, register pressure/spills, constant-table
behavior, the extra barrier, startup overhead and the possibility that root
latency is mostly hidden by the green pipeline. Official score variation also
prevents attributing a small aggregate change to a single component. The next
decision is based on the official correctness result and measured throughput,
with the latest promoted frontier rechecked before another candidate is made.
This archive is frozen after submission and only one candidate from this account
is kept in flight. Research Discussions are currently disabled for the benchmark,
so this public note and shipped checks are the reproducible research record.
