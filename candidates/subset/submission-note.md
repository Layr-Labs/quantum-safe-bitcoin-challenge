# Subset: exact GLV10 recoding, batched table construction, and disjoint continuation

Effort: high. Developed with Codex GPT-6 Astra at high effort, with separate
Herdr panes for implementation, arithmetic research, and independent validation.

## Problem and starting point

The starting subset record is promotion `9ac2515450446dbadbe061e98ebfc317c36d4999`,
submission `7aef224a-e3ff-43f9-9877-50cdbda3f653`, at 623,518,629 verified
candidates/s. The campaign target is an additional 60%, or 997,629,806.4/s.
That number is a target, not a measured or projected result of this candidate.
The earlier native-loading experiment `4284f910-a4d4-4604-bc03-5de91c935064`
was submitted separately with byte-identical baseline device arithmetic.
It was canceled before dispatch to prioritize this candidate within the account's
one-submission limit. There is no official native-only result. This candidate
therefore tests native loading and GLV together; isolated native uplift cannot
be attributed from its score.

The baseline fixed-base multiplication uses a 64 MiB table and fifteen signed
point selections. Its base already incorporates the problem-dependent
`neg_r_inv`; it does not perform an extra scalar multiplication by this inverse
for every candidate. A useful endomorphism transfer must therefore remove
point additions, rather than assume the benefit of halving a generic doubling
ladder which this implementation does not have.

The promoted pinning implementation at `1fe5a8e40008befcd917668ea9b1a23c6ee590c4`
provides an exact GLV scalar split and a twelve-selection table design. We
reuse its scalar decomposition, retaining the original attribution and license.
The later `1fe5a8e` change compared with `1ec687f` is an ordinary comment deletion
in GPUMath.h; it is not a new executable GLV optimization. Historical score
variation between those revisions is not evidence for another arithmetic gain.

## Table design and exact decomposition

GLV writes the scalar as `p + lambda*q mod n`, where p and q are signed bounded
magnitudes and the curve endomorphism multiplies affine X by beta. We retain
the donor's exact coefficient fallback and 129-bit signed residual handling.
Each component is represented with five table terms. The signed component
magnitudes are decoded on demand rather than parked in an additional shared
array, preserving the digest kernel's shared-memory occupancy budget.

The selected geometry is the exact minimum-record layout within this biased
first-segment design:

| Segment | Shift | Entries | Offset |
|---|---:|---:|---:|
| 0 | 0 | 16,777,216 | 0 |
| 1 | 24 | 33,554,432 | 16,777,216 |
| 2 | 50 | 33,554,432 | 50,331,648 |
| 3 | 76 | 33,554,432 | 83,886,080 |
| 4 | 102 | 21,319,972 | 117,440,512 |

The top odd center is 42,639,943. Segment zero represents
`bias+i`, with `bias = 42,639,944*2^101 - 2^23`; subsequent segments represent
`(2*i+1)*2^(shift-1)`. There are 138,760,484 affine records, each 64 bytes,
for 8,880,670,976 bytes (8.270769 GiB). The table's actual base is
`A = neg_r_inv*G`, unlike the baseline replay table's half-scaled base.

The point chain first accumulates q, applies beta to X once, and adds p's five
terms. Under subset's deferred-Y XYZZ invariant, beta changes X while Y, ZZ,
ZZZ, and the affine-Y anchor remain valid. The last addition resolves the
existing deferred Y. This preserves subset's positive-Y convention; pinning's
finish conventions are not copied implicitly.

A zero q component skips its entire chain. A zero p component still has a
valid biased five-term encoding when added to nonzero phi(q). The all-zero
scalar is handled explicitly. Exact integer prefix bounds and lattice bounds
rule out the incomplete-add exceptional cases for the intended nonzero chain
paths; sampled random points are supplementary evidence, not that proof.

The gross arithmetic removal is five mixed additions, or 35 field multiplies
and ten squares, before paying the endomorphism multiplication, scalar split,
changed indexing, and memory latency. Ten 64-byte random records per candidate
can request up to 640 bytes of table data before cache effects. No instruction
count establishes the final throughput on the ranked GPU.

## Construction and memory budget

The runtime builds the instance-specific table after receiving the fresh
problem. No table points, benchmark seed, hit list, or recovery result are
embedded in the native image. Two 8,192-point ladders per segment cover the
13-bit low/high decomposition, using about 5 MiB total ladder storage.

A large table makes the inherited one-inverse-per-record builder an unattractive
startup dependency. A separate exact affine builder uses a 256-lane product
and inverse tree for the nonzero X differences. Tail lanes and copy-only
high-zero records provide identity denominators and participate in every
barrier. Actual unexpected zero denominators are errors. The geometry proves
that H and L are neither equal nor negatives for every active addition.

The affine output needs two multiplies and one square after the inverse; the
block tree adds 765 multiplies per 256 records and one inverse. This is about
4.988 multiplies plus one square and 1/256 inverse per output in active-lane
work. Underfilled tree levels make the warp issue cost higher, about 6.5
multiply-equivalents plus one square and a sparse root inverse. We do not
claim a 256-fold measured speedup. The ordinary builder remains a comparator.

The large table is checked by gathering selected records on the GPU into a
small host buffer for OpenSSL comparison. Copying the entire 8.88 GB table to
host solely for a few hundred samples was deliberately removed. An additional
87-scalar cold startup audit exercises the actual GLV scalar/point chain against
OpenSSL on the current runtime base, including boundary and zero cases.
These audits count as GPU evidence only when the GPU executes them.

The existing 64 MiB table is built separately for exact tentative-hit replay.
The hot filter receives the GLV10 table; independent replay retains its original
recoder and original table geometry. A poisoned or incorrect filter cannot
publish an unchecked candidate through the host path. Replay is not a cure for
false negatives, so point-chain and nomination tests remain necessary.

The minimum layout has substantial room within 24 GiB even with the current
large stack reservation and expanded first-state buffers. A proposed 17.114
GiB layout with a 32 MiB hot first bank was deferred: adding the modeled 6 GiB
stack reservation and 64-slot buffers exceeds 24 GiB. Runtime memory telemetry
is required to distinguish explicit allocations from driver reservations.
The first GLV package retains the checked 32,768-byte stack limit and logs memory
before/after the request. Final static census is recorded below.

## Search-space continuation

The promoted fast 128-window family has exactly
`C(137,6)*128 = 1,051,964,508,672` distinct skip sets. At the campaign target it
would exhaust in approximately 1,054 seconds, short of the benchmark's
1,199-second minimum elapsed duration. Merely waiting or wrapping epoch ranks
would not deliver the requested score.

The continuation retains the original family and original lane order, then
uses a disjoint second set of 128 suffix triples. It selects the largest
remaining first-block classes first, with deterministic lexicographic ties.
Family A uses eight first classes; family B requires 47 and uses six second
classes. The first-state capacity is therefore raised from 16 to 64 consistently
in the constant dimensions, allocations, and producer/consumer/replay strides.
At the existing epoch capacity this adds 1.5 GiB of allocated first-state space.

Transition occurs only after exact replay, blocking readback, and publication
of the last family-A batch. A checked device drain precedes replacement of
WIN3 and all window schedules/class maps. Epoch rank restarts at zero for B;
cumulative search and hit counts continue. Every used first state is rebuilt.
The two suffix triple sets are disjoint, so repeating an early rank does not
repeat a nine-index skip set or recovery identity. Both families have bounded
partial final batches. The domain is a selected subset of C(150,9), not the
entire combinatorial space.

## Native integration and reproducibility

The native runtime route is inherited from the separately submitted startup
experiment. It selects one complete module before any instance-specific global
initialization and routes every symbol read/write and kernel launch through
that choice. Declared kernel parameter types determine launch storage, avoiding
integer-zero versus null-pointer argument-width errors. Missing native payload
or unsupported architecture selects the original compiled source pipeline;
corruption, incompatible ABI, or failed operations fail closed.

GLV kernels and changed first-state dimensions require a newly generated
payload and manifest. The earlier native-v1 payload is not reused. The generator
records source hashes, exact entry parameter widths, globals, CUDA version,
PTX hash, and cubin hash. The native image is generic sm89 code compiled from
the included source. No benchmark harness or sibling-track file is changed.

Authoring uses a CPU-only macOS host and an existing CUDA 12.8.93 container.
No local GPU validity or timing result is claimed. Official evaluation must
establish named-global lookup, actual code loading, audit results, memory
capacity, startup time, steady rate, and independently verified hit score.

## Validation ledger and commands

Independent CPU evidence completed before integration:

- Exact geometry dynamic-programming minimum and 315,460 compiled signed digit
  codes, including carry and boundary cases.
- 58 complete affine point cases and 116 recovery branches, covering all nine
  split sign/zero classes; analytic incomplete-add prefix/lattice bounds.
- 426 independent builder records and 2,550 inverse-tree leaves, including
  high-zero copies, segment boundaries, noncanonical residues and padded tails.
- 512 compiled family selector triples, five invalid-input cases, 1,152 actual
  host schedule outputs on three seeds through A-to-B-to-A changes, and 296
  tail/continuation models. The real final epoch batch contains 782,612 epochs.

The independent GPU fixtures provide 87 generic point cases, 174 finish
branches, and eleven field zero/carry edges. They are synthetic arithmetic
fixtures, not cached benchmark hits. Actual source audits and compiled runtime
checks are recorded in the final ledger after source freeze.

Rebuild the fixed-command candidate:

```sh
cd candidates/subset
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm
```

Regenerate the native payload with CUDA 12.8.93 and run the host-routing suite:

```sh
python3 candidates/subset/regenerate_native.py --docker qsb-cuda
python3 candidates/subset/tests/native_runtime/check.py --docker qsb-cuda
```

The Docker commands require an existing CUDA container with the repository
mounted at `/work`; local toolchain forms are documented with the generator.
The final archive is checked against the 8,388,608-byte expanded limit. Runtime
stage telemetry goes to stderr and does not change the hit-record contract.

## Results and limits

Final source iteration recomputes each hit tag from block/lane indices only
when a hit is emitted. This shortens the epoch identifier lifetime across the
point-chain calls. The initial build had a small caller/return-address spill;
the recomputed-tag build removes both stores and loads without changing the
logical tag mapping, including guarded partial epoch tails. Final sm89 census:

| Kernel | Registers | Shared memory | Cumulative stack | Spills |
|---|---:|---:|---:|---|
| Digest | 128 | 49,152 B | 0 B | none |
| Batch table builder | 118 | 16,384 B | 120 B | none |
| Cold chain audit | 144 | 0 B | 160 B | none |

The first GLV package preserves the checked 32,768-byte stack request and
records actual free/total device memory around it. A smaller request is reserved
for a separate memory-layout experiment after checking the exact native image.
The maximum static cumulative stack above does not by itself measure driver
reservations or unsupported-architecture source fallback behavior.

The final fixed-command N=24 build passed with CUDA 12.8.93. The regenerated
native image is 1,209,952 bytes with SHA-256
`08739e1c55956c1c5a8601c53e01f48e80c395b5a2f09fa805345656bba00a66`.
The corresponding PTX SHA-256 is
`c7f66ab1264d1fdbb9c898c775f87e6324451247af9a77f0c645cff14df2c6cf`.
All 27 recursive source hashes match the packaged files; the generated module
contains eleven kernels, twenty globals, and thirteen host-accessed globals.
Twelve isolated native-runtime mock cases pass, exercising 22 launches with all
argument bytes and symbol round trips. The final payload is byte-identical to
the independently inspected zero-spill recomputed-tag artifact.

The packaged CPU contract and coverage commands pass against the actual headers.
The standalone CUDA audit compiles and links for sm89, with 87 actual scalar-split
fixtures, two forced singular finishes, and 1,335 builder records including a
partially populated final block. This is compile-only evidence. Independent
review reports no blocking defect in the default chain, builder, tags, exception
routing, or continuation. Reproduce the included checks with:

```sh
python3 candidates/subset/tests/glv10/check_contract.py
python3 candidates/subset/tests/glv10/check_coverage.py
python3 candidates/subset/tests/glv10/gpu_audit.py --docker qsb-cuda
```

See `tests/glv10/README.md` for explicit GPU execution and its required binary
problem input. The test generator replaces SHA generation with an explicit
scalar seam in copied exact-replay bodies; it does not claim to test the complete
SHA-to-nomination pipeline. No compiler output, test result log, or host executable
is included in the submission archive.

CPU models, static instruction counts, and successful compilation are not a
verified GPU throughput improvement. Official runtime validity and ranked
throughput are pending. The extra 60% campaign goal remains open.


## Primary references

- The repository problem and scoring specifications define fresh instances,
  canonical skip/recid identities, fixed build, and verified-hit scoring.
- [CUDA 12.8 library management](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__LIBRARY.html)
  and [NVIDIA runtime loading example](https://developer.nvidia.com/blog/dynamic-loading-in-the-cuda-runtime/)
  support the native loader interface.
- [NVIDIA stack-reservation discussion](https://forums.developer.nvidia.com/t/cudadevicesetlimit-bug/320656/6)
  motivates distinguishing the per-thread limit from explicit buffer allocation;
  runtime memory telemetry remains authoritative for this candidate.

The unpromoted startup hypothesis from terrapinelf's `a75cf15a` remains credited
through the inherited native-loading experiment. Promoted pinning scalar and
host-ladder sources retain their original notices; those promotions are the
arithmetic donors, not fresh measured subset results.
