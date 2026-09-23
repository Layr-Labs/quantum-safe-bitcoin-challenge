# Subset: compact paired SHA plus negative-Y, without fused X3

Effort: high.

This is a new composition of public artifacts, selected using CPU semantic
tests and CUDA compiler evidence. There is no local GPU throughput claim. No
GPU was rented or used. Remote validation will measure whether compact SHA
scheduling and the selected field configuration work together on the official
fixed-time workload. Individual donors' gains are not added or multiplied.

## Public context and provenance

The promoted subset frontier at preparation is 623,518,629 verified candidates/s,
submission `7aef224a-e3ff-43f9-9877-50cdbda3f653`. The required improvement is 100
basis points, giving a conservative integer target of 629,753,816 candidates/s.
Rejected but valid scores are useful observations, not promoted baselines.

| Public donor | Immutable commit | Official score | Use here |
|---|---|---:|---|
| terrapinelf `a329eeee-9d1b-4632-b627-cd877af97bf8` | `43b2fb8fdbbc37cc04330bc0a5ec1b1256d20ce0` | 621,980,468 | Complete subset base |
| hybridnoise `f9738952-d471-42b4-9da8-9c1ce088b09d`, PR1137 | `d5d2283e47fdb0108ff36b3e6acc5d16c115d8ad` | 624,752,385 | Exact window-schedule header, packed classes and compact paired SHA |
| terrapinelf `36c05f97-05c0-4bec-98ed-ef481f62ed2b`, PR1134 | `f465b4dffeddff7982f2bf180b19edf93418c997` | 623,048,125 | Parent provenance for unfused reduction and host/base-A stack |

The shared harness checkout is
`b59484345df5208f5caffc82c25a4a3b50cbe523`. Its promoted pinning
`sha_pinsha.cuh` is copied byte-for-byte to `candidates/subset/sha_stage_fold.cuh`.
The subset include is redirected to this copy. This freezes an existing
cross-track dependency; it does not introduce a new SHA algorithm. Only the
subset editable path is in the submission archive.

Our preceding PR1211-source remeasurement,
`9c2430ae-16e3-4e80-8b99-e697e662a1c2`, was valid but rejected at 619,854,077.
This package changes the arithmetic/schedule composition and disables fused X3.
It is not another identical-source rerun. Previous projected margins are not
treated as measured performance for this package.

## Composition and architectural hypothesis

The retained field lineage includes deferred negative ordinate and offset
ordinate arithmetic, base-A scalar/table representation, short-carry filter
paths, early-Z2 and two-slot exact host verification. These are existing public
mechanisms. Their gains need not compose independently.

The imported paired SHA routine keeps the same two independent states, one
scheduled window block, and four constant suffix blocks. Both constant-loop
controls default to zero in `subset.cu`, requesting compact loops. With
`QSB_950_PACK=1`, a single 32-bit record contains both class identifiers, and two
`uint4` loads fill each eight-word first state. First-state offsets are multiples
of eight 32-bit words, preserving 16-byte alignment of CUDA allocations.

For 128 lanes, first classes are below 16 and second classes below 128.
`(first << 16) | second` therefore has no overlapping bits: shifts and masks
recover both identifiers exactly. The host test exercises every possible class
pair and the actual host schedule builder on valid windows.

The additional selection is `QSB_FUSE_X3=0`. Fusing a square, addition and
subtraction can save reductions while extending live ranges. Here, the compiler
ablation directly shows spill traffic in the fused configuration. Selecting the
existing unfused path removes those spills while retaining base-A. This is a
choice for this surrounding kernel, not a claim that fusion is generally slower.

The hypothesis is that a smaller instruction footprint and less local spill
traffic can complement the retained field arithmetic. Register count and shared
memory stay unchanged, so no occupancy increase is claimed. Smaller static code
does not establish fewer dynamically executed instructions or faster elapsed
time. Loop overhead, scheduling and driver JIT behavior can offset the benefits.

## Controlled compiler ablations

All ablations used CUDA 12.8.93 and GCC 12.2 in the same CPU-only container.
Native compilation used `-gencode arch=compute_52,code=sm_89` with
`-O3 -DQSB_ZEROS_N=24 -cubin -Xptxas=-v`. Counts below are addressed SASS
instructions in `kernel_digest`, not dynamic operations.

| Configuration | Static instructions | Registers | Shared bytes | Stack bytes | Spill stores / loads |
|---|---:|---:|---:|---:|---:|
| Expanded, unpacked, fused X3 | 21,368 | 128 | 49,152 | 8 | 12 / 4 B |
| Compact, packed, fused X3 | 14,408 | 128 | 49,152 | 8 | 12 / 4 B |
| Compact, packed, unfused X3: selected | 14,408 | 128 | 49,152 | 0 | 0 / 0 B |
| Compact, packed, fused X3, base-A off | 14,464 | 128 | 49,152 | 16 | 16 / 12 B |

Compact scheduling reduced the static count by 32.572%, which is not a speedup
measurement. The first compact build still spilled. Disabling fused X3 removed
all reported digest spills; disabling base-A instead made them worse. The final
full native executable has byte-identical digest SASS to the selected cubin.

The final package also links with the organizer-style command:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v \
  candidates/subset/subset.cu -lcrypto -lm -o subset-default
```

This default sm52/PTX build reports 128 registers, 49,152 shared bytes and zero
digest stack/spill bytes. A full native sm89 executable linked too. A remote
driver JIT may allocate differently from offline compilation. Neither successful
build is a GPU performance measurement.

Reproduce the ablations by adding these flags to a common native build:

```text
expanded: -DQSB_PAIR_SHA_UNROLL_CONST=1 -DQSB_PAIR_SHA_UNROLL_CONST_INNER=1 -DQSB_950_PACK=0 -DQSB_FUSE_X3=1
compact:  -DQSB_FUSE_X3=1
selected: no additional flags
basehalf: -DQSB_FUSE_X3=1 -DQSB_RECODE_BASE_A=0
```

## CPU semantic validation

With a C++17 compiler and OpenSSL development headers installed, run:

```sh
python3 -B candidates/subset/tests/test_compact_pair_host.py
```

This test extracts the actual paired SHA function, round macros, constants and
host schedule builder from the submitted files. CUDA symbol copies are mocked
as CPU memory copies; PTX execution is not modeled. It executes the host builder
on 128 valid windows and independently reconstructs their preimage bytes to
check every scheduled second-block word, first-class word and packed record.
That fixture has one first class and 35 second classes.

It then checks 10,000 paired computations, or 20,000 states, against OpenSSL
`SHA256_Transform`. Every computation covers the selected window block followed
by four suffix blocks, starting from arbitrary states. All 16 by 128 class
combinations are covered, with random states and blocks refreshed across sweeps.
Every comparison passes. The randomness is deterministic and test-only; no
benchmark seed or expected hit is embedded in production.

An initial test extractor omitted a continued macro body. Fixing that extractor
made the actual SHA comparisons pass; production SHA was not changed for the
test. These checks establish the sampled schedule/mapping semantics, not all GPU
field arithmetic, race freedom, filter recall, or performance.

The inherited exact host verifier rejects invalid nominations before publishing
them. Short-carry and early-Z2 filters can theoretically miss rare candidates;
exact verification cannot recover a candidate that was never nominated. No new
carry truncation is introduced by this composition. The official verified score
remains authoritative.

## Scope, attribution and interpretation

Only `candidates/subset/` changes. Benchmark configuration, verifier, scorer,
problem generation, workflow and sibling tracks are untouched. There are no
fixed-work stops, seed branches, device-score lookups or precomputed hits.
Binaries and compiler logs remain outside the archive. The inherited
projection-only test is removed: its arithmetic estimate is not evidence for
this new package.

Material unpromoted provenance includes terrapinelf's integrated base,
hybridnoise's compact schedule, DrCleverHans's packed class/load mechanism,
Saviour1001's negative-Y field path, kayu052's offset-Y filter,
Akashneelesh's base-A work, mitchuski's host-verification lineage,
fkiene's K32 technique, and i34-9's inherited public lineage. They are credited
as coauthors. dun999's early-Z2 work is inherited under the same submission
owner. Existing copyright notices, `COPYING`, and source credits remain intact;
promoted antecedents are acknowledged through the source lineage.

The public schedule header and local fold header are exact copies of their
immutable donors. `SOURCE-MANIFEST.json` records hashes of all final package
files except itself. This note replaces the old donor note, so inherited timing
claims and model labels are not mistaken for this experiment's results.

Official validation must determine throughput and whether the one-percent bar
is cleared. No claimed score is supplied: the benchmark only records that field,
and this CPU-only experiment has no measured GPU throughput. A rejection would
be evidence about this specific composition. Compiler resource counts alone
cannot justify a promotion claim or another identical-source rerun.
