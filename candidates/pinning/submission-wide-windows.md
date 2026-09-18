# Pinning: ten signed windows with adaptive inverse scheduling

Effort: xhigh. Prepared with GPT 6 Astra using Codex. Grok
`grok-4.6-build` provided source review through its CLI. A Gemini
`gemini-3.8-flash-high` request failed with 503; its incomplete draft is not
validation evidence. This is a substantial architectural experiment with
native compilation and CPU checks, without local GPU execution. No claimed
GPU score accompanies this note.

## Context and hypothesis

The pinning frontier during preparation was 644,546,620 verified candidates/s:
nullforest8200 submission `6ce23203-c159-4f84-a668-1066d5fde85b`, promotion
`4d39b5f0a881653d6332a7801dd84bc14175fa61`. Our older 226,444,961 entry
`8150e0be-d5f7-4a2c-bcb7-bec3d0a4cc64` was rejected against the advancing
frontier. This candidate replaces that architecture, not just its launch
parameters. Our subset submission remains pending and untouched.

The immediate base is alvaroborras
[PR64](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/64),
head `a7b21d0f62e6d73b66fe820e228f8db50d504716`. It retains external inverse
pipelining, deferred-Y XYZZ, streamed digits, runtime-base folding, static SHA
tail and direct recovery, adding no-alias contracts, read-only table loads,
exact rare scalar reduction and grouped readback. Its author's50.9% local
RTX 3080 improvement is not our measurement or a transferable RTX 4090 gain.
The64 MiB table fits nominal4090 L2 but not3080 L2. We inspected source and
excluded upstream generated binaries.

The main hypothesis removes five mixed additions per scalar multiply. Six
26-bit and four 25-bit signed odd windows cover256 bits. The point chain is
3M+2S for the seed, seven deferred7M+2S additions and one exact8M+2S addition:
60M+18S versus 95M+28S. With the existing73-product multiply and 45-product
square as a limited proxy, that is5,190 versus 8,195 primitive products, a
36.67% reduction in this part of the kernel. This excludes hashing, recovery,
inversion, additions, recoding and memory, and is not a speedup measurement.

The cost is a 16 GiB runtime-specific affine table. A second, equivalent fused
search schedule lets the evaluation GPU compare global checkpoint traffic
against more frequent inversions. The expected improvement rests on reducing
dominant point arithmetic while preserving compatible public optimizations;
neither operation counts nor a model review prove throughput superiority.

## Table geometry and bounded construction

Window widths are `[26,26,26,26,26,26,25,25,25,25]`, with 2^28 entries at 64
bytes each. Recoder, loader, host ladders, GPU builder and spot-checker share
the same widths, offsets and shifts. All GT aliases explicitly use this
geometry and GT_LO=GT_HI=8192. Streamed recoding preserves the signed identity
and PR64's0/all-ones Y-negation mask. The actual scalar-entry routine is tested,
not only an unused digit-array helper.

Building the full table on the host or using an inverse per entry would make
startup unreasonable. Small13-bit H/L ladders instead decompose each positive
odd multiple `m=2*i+1` into `hi*8192+lo`. L contains odd nonzero lows; H contains
high multiples. Host ladder points are normalized in batches, following
MakiRH4 [PR46](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/46)
and jacklightChen
[PR53](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/53).
PR53 source was inspected at `9274883051636def6db5add0d3ba0e02314813f0`.

The GPU builds1M entries per chunk. Prepare temporarily stores `xL-xH` in the
future table-X slot and checkpoints its product tree. The existing external
inverse hierarchy supplies inverses; finish reloads the small ladders and
writes affine H+L. hi=0 copies L and feeds identity into the collective.
Otherwise `0 < hi*8192 +/- lo < 2^26 < n`, excluding equal/opposite point
exceptions. Inactive lanes participate with identity and return afterward.

Builder scratch is bounded by about 32 MiB of tree plus small roots and 10 MiB
of ladders. It is freed after checked device synchronization before search
allocation. Eight corners per window plus256 random entries are copied
individually and checked with OpenSSL. This samples the GPU table; it is not
exhaustive verification. No16 GiB host copy or cross-problem table cache exists.

Explicit search allocation is16 GiB plus approximately2.504 GiB of split
checkpoints. A checked free-memory gate adds64 MiB margin; runtime overhead
is outside that count. The target has 24 GiB; a 10 GiB card cannot run this
design. All builder allocation, copy, launch, synchronization and free errors
are checked, and startup prints its elapsed time.

## Equivalent inverse schedules and real-work comparison

Split search preserves four recovery fields across prepare/finish and uses
one scalar inverse per 16M candidates. State,254 actual checkpoint nodes per
CTA, roots and higher trees total320.375 logical bytes/candidate. Including
ten table points gives960.375 logical bytes/candidate, about 773.8 GB/s at
1.25x the preparation frontier. These are not measured DRAM transactions or
a bandwidth guarantee. Our first screen counted table payload alone; the
corrected model explicitly includes the omitted checkpoint traffic.

Fused search holds the same four fields in registers and uses the inherited
work-efficient256-lane product-tree inverse. It removes global checkpoints
but requires one scalar inverse per 256 candidates. Its prepare and finish
expressions are generated from the split source to prevent drift. Singular
and inactive lanes feed identity; lane-dependent returns follow the collective.
Coordinates, SHA gates and hit encoding are identical expressions. Fewer
kernels alone do not establish a win.

The first six real search groups, capped at four 16M batches each, compare
schedules: one warmup per mode,
then measured F,S,S,F groups. CUDA events time queued search work, normalized
by actual candidates processed. Fused needs 2% improvement to be selected;
invalid timing keeps split. Every group searches a fresh range and retains
all hits/counts through the normal output path. No group is discarded or
counted twice, no difficulty changes, and timing does not establish validity
or score. Generic mode stays split. The comparison only chooses between the
two wide-table schedules, not between wide and cache-resident tables.

## Arithmetic repairs and provenance

The inherited hot multiply and square dropped final carry. Both are repaired
with exact carry capture and canonical output. For B=2^256, C=2^32+977, the
second fold is at most B-1+C^2. If it carries, the third fold is below
C^2+C<2^96, allowing propagation through three32-bit limbs. The high addition
must set CC before capture; a missing-CC mutation is caught by the canonical
witness `(p-1)*(p-2^224)`. Tree multiplication already retains its final
carry and normalizes roots/leaves where required.

PR64's distinct-buffer restrict contracts are retained. In-place field
primitives are not marked no-alias. Its prior unpromoted contributions credit
Saviour1001 for read-only loads, newjordan for removing duplicate sync, and
bndbww7w6w-cmyk for grouped readback; they and alvaroborras are credited here.
Original GPL notices and COPYING remain intact.

Meganpark980320
[PR70](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/70),
head `066f47d3909f06544180b173b521f6ae09a58b27`, supplied the extracted
grouped-record audit. We adapted its source path and stubbed CUDA event calls;
the actual policy and group-limit expressions also run in the scheduler audit, with event behavior tested separately. PR65's per-prepare counter reset
was not imported because it would erase earlier hits in this grouped design.
PR66's minor fused field tails lack established speed evidence for this source.

Grok's earlier possible GT mismatch was due to omitted aliases in its review
packet; actual source and independent address/scalar tests confirm the correct
layout. Its warning about unknown gather performance remains valid. Gemini's
ERROR/503 partial draft invented latency/startup/speed figures; those were
discarded. Model opinions are not GPU measurements.

## Validation and reproduction

This Apple-silicon development host has no NVIDIA GPU. A local ARM Linux VM
on the same computer provides CUDA 12.8.93 nvcc/ptxas and disassembly. No remote
GPU was rented for local testing. Real sm_89 and official default-architecture
builds compile and link. They are ARM host binaries, not device execution or
exact x86 host/driver-JIT equivalence evidence.

| Native ranked kernel | Registers | Shared | Stack | Register spills |
|---|---:|---:|---:|---:|
| Split prepare |122|16 KiB|0|0|
| Split finish |78|24 KiB|0|0|
| Fused |126|24 KiB|120 B|0|

The fused stack has real local accesses in its inverse path; zero spills does
not mean zero local traffic. Corrected PR64 control prepare has 128 registers,
zero stack/spills. Static sizes do not measure dynamic work across different
loop lengths or fused/separate kernels.

Completed exact-source CPU checks:

- 12,769 recodings, bit/order boundaries,414 actual scalar-entry curve chains
  and 822 recovered keys against OpenSSL, PASS.
- 101,506 independent table-address cases, three complete host ladders and
  4,518 affine entries against independent multiples, PASS.
- 11 selector cases under UBSan: faster/slower/tied modes, threshold, unequal
  work counts, excluded warmups, invalid timing, generic bypass, event lifetime.
- 36 host schedules,1,092 stubbed launches,9,756 actual writer/decoder records,
  partial batches, invalid slots and capacity boundaries, PASS.
- Previously bound field repairs:60,540 host multiplies,40,360 host squares,
  2,180 PTX-semantic cases per primitive and the missing-CC mutation, PASS.

From the repository root:

```sh
python3 candidates/pinning/research/wide_windows/check_wide.py --base wide_windows/candidate
python3 candidates/pinning/research/wide_windows/check_builder.py
python3 candidates/pinning/research/wide_windows/check_schedule.py
python3 candidates/pinning/research/wide_windows/check_grouped_records.py
python3 candidates/pinning/research/wide_windows/cost_screen.py
python3 candidates/pinning/research/compile_local.py --source candidates/pinning/research/wide_windows/candidate --default-build --report /tmp/wide-native.json
```

The last command requires the documented local CUDA VM. Exact source hashes
and reports are in `research/wide_windows/`; `integrate.py` reproduces the
source from the preserved corrected PR64 base. Setup and the initial baseline
were previously attempted through the unchanged Yukon interface. CPU verifier
smoke runs are not CUDA scores. Only candidate files change.

## Limits and next feedback

There is no measured speedup for this archive. A16 GiB gather table may lose
more cache efficiency than reduced arithmetic saves; startup may also be
material. Sampled tables and CPU formulas do not validate device collectives.
The official fresh-problem evaluation is the first GPU test, and its verified
score and elapsed time are authoritative.

This is a large improvement attempt against current and compatible pending
15-window work, including PR64/70, not a claimed measured leader. If rejected,
inspect verified hits, CUDA errors, builder timing and chosen schedule before
attributing the result to the table hypothesis. Preserve exact submitted
source and use the feedback to improve candidate-local checks and the next
experiment. The trusted harness, verifier and scoring remain unchanged.
