# Subset: runtime choice between 64 MiB and 16 GiB fixed-base methods

Effort: xhigh. GPT 6 Astra in Codex implemented and checked this candidate.
Gemini 3.8 Flash High (`gemini-3.8-flash-high`) provided a read-only integration
review. Grok 4.6 reviewed earlier adaptive components; its final combined-source
request timed out without a completed result. Muse Spark 1.3 Contributor Free
in OpenCode supplied public research proposals, which were independently screened
as duplicate, inapplicable or deferred. Those proposals are not validation or
implemented novel mechanisms. No local GPU throughput or claimed score is supplied.

## Starting point, frontier and provenance

Our immediate subset parent is promoted PR60, submission
`65fb673d-5014-4ee8-866d-97d972e7b1f0`, promoted commit
`8e5cd89b50dafa0dbef41ce92cab1f7974e531c2`. It scored **451135044 verified
candidates/s**. Its complete production/audit source fingerprint is
`44f9d5808add05e204b169194af8893dc91a9c01d31448362b0e01c667c9fd06`.
That result covers the full composite, not an ablation of external inversion.

The current promoted frontier is welttowelt's
[PR62](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/62),
submission `80a850f0-3bd5-484b-b890-70faec8a39d0`, promoted commit
`106a6826ecf8998e684169f46e8de9dce1cf3f0f`, at **477182283 verified
candidates/s**. Its measured binary uses a 64 MiB interleaved mixed-window
method and monolithic per-CTA inversion. This candidate adopts that table
geometry into our external-inversion pipeline. The public score does not
belong to our independently integrated compact branch.

The 16 GiB geometry and bounded builder come from our pending pinning
[PR74](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/74),
submission `66c031c6-16e4-484e-ad4d-0e7e0ef3f38f`, validation commit
`154b2bd97a9c918ac3d992c31a6cf7901aeab1a0`. Pinning remains unchanged.
That work substantially extends alvaroborras's unpromoted
[PR64](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/64),
head `a7b21d0f62e6d73b66fe820e228f8db50d504716`, with batched ladder
construction informed by MakiRH4's
[PR46](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/46)
and jacklightChen's
[PR53](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/53),
head `9274883051636def6db5add0d3ba0e02314813f0`. These substantial unpromoted
contributors are credited as coauthors. Pinning-only readback changes were
not transplanted into subset.

The promoted subset parent already credits jacklightChen, DPZZxlz,
nullforest8200, alvaroborras and Meganpark980320, with earlier epoch work
and its upstream contributors. Retained mechanisms include streamed signed
recoding, deferred-Y XYZZ, ranked SHA reuse, direct recovery and the external
checkpoint/root hierarchy. Original GPL notices and COPYING remain intact.
Those inherited mechanisms are not claimed as new improvements here.

The selected complete source fingerprint is
`6bfe0e61fbbcabce237877c95e238f0bd22914ed6ceade6baf97e058c5e51ac6`.
`research/adaptive_mixed/provenance.json` identifies the frozen adaptive and
compact donors. `validation-summary.json` binds reports to the selected source.

## Substantial mechanism and its cost

The compact branch has fifteen signed windows `[18,17,17,17,17,17,17,17,17,17,17,17,17,17,17]`.
It stores 2^20 affine entries, 64 bytes each, in a 64 MiB interleaved table.
The wide branch uses `[26,26,26,26,26,26,25,25,25,25]`: 2^28 entries and 16 GiB.
Both depend on the current problem's runtime base and recode the actual
SHA-derived scalar. Neither caches answers, preimages or results across problems.

The fixed-base chain alone changes from **95M+28S** in compact to **60M+18S**
in wide: five fewer mixed additions. Logical table payload drops from 960 to
640 bytes per candidate. The earlier PR60 chain used 102M+30S and 1024 bytes.
These counts exclude hashing, inversion, recovery, field additions, checkpoints
and startup. Wide can increase physical DRAM traffic by losing cache reuse.
No measured cache hit rate, latency overlap or bandwidth utilization is assumed.

This is the basis for expecting a substantial opportunity beyond public pending
micro changes. It is not a predicted numerical score. Public PR68/77 geometry,
dispatch and recovery source was inspected; it combines compact geometry with
overlapping or small changes. PR78's exact source retains sixteen windows and
adds direct XYZZ recovery, already present here. PR75 and PR82 notes describe
small schedule/packing effects. These pending scores are unknown, so expected
superiority remains a hypothesis rather than an established ranking.

The final queue refresh also found DPZZxlz's `fa6e0a3` entry, combining the
exact promoted mixed-15 core with our promoted external inversion and narrow
field carry repairs. Its note was read before upload. That is a direct compact
competitor, not a novel wide-table option. It retains a different specialized
square schedule, so our compact method cannot be assumed to equal or beat it.
Our differentiating substantial hypothesis remains the additional wide branch
and actual runtime comparison. The new entry reports CPU/projection checks but
no native CUDA build or GPU timing. Its field schedule is not imported without
source-level device arithmetic validation.

The compact path is always available. In the ranked single-GPU mode, wide is
considered only if free memory after compact allocation covers its 16 GiB plus
a 2 GiB search reserve. Only optional allocation exhaustion falls back; unrelated
CUDA failures remain fatal. The recoverable allocation error is cleared before
later launch checks. Generic modes use compact.

Both builders create small runtime-dependent H/L ladders on the host, batch
their affine normalization, then build the table on the GPU in at most 2^20-entry
tiles. Compact uses 8-bit lows; wide uses 13-bit lows. Scratch is bounded and
freed after checked synchronization before search allocation. Table samples
are compared against OpenSSL. There is no 16 GiB host table copy. Samples and
native compilation do not establish correctness of every device entry.

## Runtime comparison retains useful work

The ranked loop evaluates six distinct real search batches in compact, wide,
wide, compact, compact, wide order. The first two warm their methods. The final
four measure both alternatives in balanced order; this reduces simple ordering
bias but cannot guarantee cancellation of thermal, cache or clock effects.

CUDA events cover the epoch producer, point preparation, checkpoint/root
inversion and recovery/hit pipeline on the same stream. Each method's total
GPU time is divided by its actual candidate count. Wide is selected only if
its measured time per candidate is more than 2% lower. Ties and invalid timings
select compact. Events are destroyed when comparison ends, including short-run
cleanup, and cannot be copied accidentally. If compact wins, wide storage is
released after the existing device synchronization.

All six batches advance the ordinary range cursor and counters and feed their
hits through the normal output path. No calibration candidate is replayed or
discarded. The inherited 64-record host drain cap remains unchanged. The explicit
dispatch flag selects the correct prepare kernel and table pointer; both methods
use the same state layout, root hierarchy and finish kernel. Both X/Y coordinates
are interleaved in one allocation; the unused Y pointer is null for both branches.

Construction, JIT and host I/O are outside the event comparison. The benchmark's
process wall time still includes startup. A wide method that loses after building
therefore incurs overhead that selection cannot undo. This policy is not a
guarantee against total-process regression or sustained-performance changes.

## Validation and reproducible checks

Development used an Apple silicon Mac and a local ARM Linux CUDA 12.8.93
compiler VM, with no NVIDIA GPU. The trusted harness and sibling pinning sources
were preserved. Native compilation is separate from CUDA execution and timing.

For each actual fixed-base branch, extracted source executes against OpenSSL
field operations and independent curve multiplication: 12769 scalar recodings,
414 curve chains and 822 recovered-key comparisons pass. Actual interleaved
loaders pass 570 compact and 380 wide cases. Next-load prefetch addresses match
5382 compact and 3312 wide logical targets. Sparse virtual mappings permit large
offset tests without allocating the full resident wide table on the Mac.

Source-derived host ladder and builder affine tests check three runtime bases,
window boundaries, low/high decomposition and random entries. Compact covers
101491 decodes and 4473 curve entries; wide's exact counts are in its report.
These do not execute CUDA builder collectives. Bounded scratch lifetimes and
all-lane participation are separately checked from source.

The actual host loop, dispatch, policy and allocation handling are projected to
C++ with mocked CUDA computation: 20 host-flow scenarios, 10 timing-policy cases
and 7 allocation cases pass. They cover both winners, tails, short runs, invalid
timings, range continuity, hit routing, pointer selection and release. Mocked
hits do not validate rare GPU hits.

The general candidate audit now accepts an isolated source root and actual
mixed-window geometry. On the selected source it passes 3840 inverse outputs,
6144 complete SHA256d comparisons, 10262 signed recodings and 8086 combination
unrankings. The earlier sixteen-window production control also passes after
the checker update. This prevents checking a stale production tree while
working on an isolated candidate.

The pipeline audit was also updated to select either actual prepare kernel.
Each branch freshly passes 1282 hierarchical leaf inverses, 2000 direct
recoveries, 768 controlled pipeline candidates, three singular lanes and
237 generated hit records. This tests shared transport/recovery with controlled
producer points, while the separate chain tests exercise actual table geometry.

The corrected `GPUMath.h` and `square32.cuh` are byte-identical to the tested
PR60 primitives, retaining near-p carry witnesses and stale-condition-code
mutation detection. The SHA/inverse headers and shared pipeline bodies are
unchanged; the new compact prepare differs only in its name and separately
tested fixed-base call. `inherited-evidence.json` records this exact equivalence
and the reused report hashes. OpenSSL-substituted curve tests alone would not
validate the actual PTX field primitives.

Native production builds pass for explicit sm89 and the default official-style
flags. The audit translation unit also compiles for sm89. Both prepare kernels
use 128 registers and 24576 shared bytes with no spill stores/loads; common
finish uses 80 registers, 24576 shared bytes and no spills. Compiler counts
measure neither active occupancy, device scheduling, cache behavior nor speed.
No CUDA sanitizer or GPU audit execution occurred.

From the benchmark directory, reproduce the CPU checks with:

```sh
python3 -B candidates/subset/check_candidate.py --source candidates/subset/research/adaptive_mixed/candidate --report /tmp/qsb-core.json
PYTHONPATH=candidates/subset:candidates/subset/research python3 -B candidates/subset/research/wide_windows/check_wide.py --source candidates/subset/research/adaptive_mixed/candidate --report /tmp/qsb-wide.json
PYTHONPATH=candidates/subset:candidates/subset/research python3 -B candidates/subset/research/wide_windows/check_wide.py --source candidates/subset/research/adaptive_mixed/candidate --compact --report /tmp/qsb-compact.json
PYTHONPATH=candidates/subset:candidates/subset/research python3 -B candidates/subset/research/wide_windows/check_builder.py --source candidates/subset/research/adaptive_mixed/candidate --compact --report /tmp/qsb-compact-builder.json
PYTHONPATH=candidates/subset:candidates/subset/research python3 -B candidates/subset/research/wide_windows/check_builder.py --source candidates/subset/research/adaptive_mixed/candidate --report /tmp/qsb-wide-builder.json
python3 -B candidates/subset/research/adaptive_tables/check_adaptive.py --source candidates/subset/research/adaptive_mixed/candidate --report /tmp/qsb-flow.json
python3 -B candidates/subset/check_pipeline.py --report /tmp/qsb-pipeline-wide.json
python3 -B candidates/subset/check_pipeline.py --compact --report /tmp/qsb-pipeline-compact.json
```

Use a clean source copy with `nvcc -O3 -DQSB_ZEROS_N=24 subset.cu -lcrypto -lm`
for production and add `-arch=sm_89 -Xptxas=-v,--warn-on-spills` for the reported
resource target. Compile `tests/gpu_epochs/tree_audit.cu` separately. On a CUDA
GPU host, run `yukon setup --track subset` and `yukon run --track subset` after
invalidating any generated wrapper cache following header changes.

## Review corrections and interpretation of the next result

Gemini found no blocking integration defect, but several supporting claims were
rejected: invented startup seconds and score ranges, guaranteed cache residency
or regression protection, incorrect epoch count, and confused builder tile units.
A tile holds one million entries, not one MiB. C(137,6) is 8218472724. The GPU
exposes many scheduled blocks, not tens of thousands of simultaneously active
warps. Model agreement is not evidence for any of these properties.

The builder test initially included device-only routines in its host projection,
then mistook a helper definition for its later call in an ordering assertion.
Both checker extraction errors were corrected without changing candidate source;
the final tests execute the intended extracted routines. Their reports bind to
the selected fingerprint rather than merely recording process success.

The official result must determine whether compact geometry plus external
inversion, or wide's arithmetic saving, overcomes memory and startup costs.
Record the selected runtime method and compare verified throughput with both
the historical PR60 result and the frontier at evaluation time. If it loses,
retain the source and selection log as evidence; do not infer that all wider
table geometries or all inversion schedules are ruled out by one integration.
