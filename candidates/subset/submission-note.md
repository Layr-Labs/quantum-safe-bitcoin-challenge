# Subset: twelve-term GLV with the promoted six-bank geometry

Effort: high. Developed with Codex GPT-6 Astra at high effort using separate
Herdr implementation, architecture-review, and independent-validation panes.

This package is selected for official evaluation after the valid native-control
result and independent source, arithmetic, ABI, and resource reviews. It has no
GLV12 GPU result yet.

## Starting evidence and purpose

The campaign baseline is subset promotion
`9ac2515450446dbadbe061e98ebfc317c36d4999`, at 623,518,629 verified
candidates/s. The campaign target remains 997,629,806.4/s, an additional 60%.
No result in this note establishes that target or predicts that this candidate
will reach it.

The preceding minimum GLV10 candidate,
`aa4d949d-6773-4cf6-b698-ba3fa09f01a6`, produced 62,949 verified hits in
1200.829171492 seconds, for an official score of 439,741,553/s. Its rejection
was for insufficient score improvement. The assigned source and all 68
archived files matched. An independent rank audit found a searched-prefix
lower bound of 529,891,590,144 candidates, with 63,168 expected hits under the
random-hit model versus 62,949 observed. This is consistent with ordinary
count variation; it is not proof of perfect filter recall or an attribution
of the throughput loss to any particular hardware stage.

The native-only control `7311c0aa-424a-4ad5-8197-31ad8db50a50` retains the
promoted subset arithmetic and 64 MiB table. It completed with 87,134 verified
hits in 1201.128028308 seconds and an official score of 608,538,767/s, about
2.4% below the campaign baseline. Its rejection was solely for insufficient
score improvement. The package did not establish a native-loading gain.
Its purpose was to test the native-loading pipeline independently of the large
GLV table. An earlier native submission,
`4284f910-a4d4-4604-bc03-5de91c935064`, was canceled before dispatch and has
no GPU result. These are distinct submissions.

This preparation takes the reviewed minimum's exact scalar split, table
builder, replay separation, startup audits, and disjoint search continuation,
then substitutes the already promoted pinning GLV12 geometry from
`1fe5a8e40008befcd917668ea9b1a23c6ee590c4`. The smaller table trades two more
point additions against less table construction and a different cache demand.
It is a bounded geometry transfer, not a new scalar approximation.

## Geometry, indexing, and table ownership

The scalar split is the donor's exact `p + lambda*q mod n` decomposition,
including the HIGH15 ambiguous-boundary fallback and signed 129-bit residual
handling. Each signed component uses six biased/signed table selections.
The table has the following logical and physical layout:

| Logical bank | Shift | Records | Physical record offset |
|---|---:|---:|---:|
| 0 | 0 | 262,144 | 0 |
| 1 | 18 | 262,144 | 262,144 |
| 2 | 37 | 262,144 | 524,288 |
| 3 | 56 | 8,388,608 | 6,116,425 |
| 4 | 80 | 8,388,608 | 14,505,033 |
| 5 | 104 | 5,329,993 | 786,432 |

Physical order is `[0,1,2,5,3,4]`. The allocation contains 22,893,641
canonical affine records of 64 bytes, totaling 1,465,193,024 bytes, about
1.365 GiB. Banks 0, 1, and 2 occupy a contiguous 48 MiB prefix. The logical
top bank is physically in the middle of the allocation; treating logical
bank numbers as ascending offsets would build incorrect points.

One portable geometry header supplies shifts, sizes, offsets, digit encoding,
and the interval-aware physical-record decoder. Builders, gathered samples,
and host reference points use this same mapping. The inherited `glv10_*`
filenames and kernel names are retained for interface continuity; the selected
constants explicitly specify six chunks and twelve terms.

The runtime base is `A = neg_r_inv*G`, derived from the fresh problem. Bank 0
contains `(bias+i)A`, where
`bias = 10,659,986*2^103 - 2^17`. Ordinary banks contain
`(2*i+1)*2^(shift-1)A`. The top odd digit center is 10,659,985. The promoted
donor's recoding is copied exactly and checked literally against the new
portable header.

## Point-chain and exact replay contracts

The filter accumulates q's six terms, applies one endomorphism multiplication
to projective X, and adds p's six terms. Subset's positive-Y convention and
deferred affine-Y anchor are retained. Endomorphism changes X while preserving
the relevant Y, ZZ, ZZZ, and anchor invariant. Pinning's alternative table or
finish coordinate transformations are not imported.

Zero q uses only p's six terms. Zero p with nonzero q still has a valid
six-term biased encoding. The whole-zero scalar, raw-p/zero denominator
nominations, and singular exact finishes retain the reviewed minimum behavior.
The scalar splitter and the 87-scalar fixture header remain byte-identical to
that minimum reference. Existing speculative field/filter operations remain
speculative; exact recoding is not a claim that every filter operation is
universally exact.

The old 64 MiB table remains a separate allocation for exact tentative-hit
replay. The filter receives the GLV12 table and the verifier receives the old
table. Every published identity passes exact replay. This prevents publication
of unchecked nominations, but it cannot recover hits that a filter fails to
nominate. Independent chain and exceptional-case checks therefore remain part
of validation.

## Construction, cache policy, and memory

Six pairs of 4,096-point ladders support the 12-bit low/high decomposition,
using 3 MiB of device ladders plus 3 MiB of host ladders during setup.
Ordinary high-zero records copy their low-ladder
point without reading the unpopulated high-zero entry. Biased bank-0 low zero
is a valid point and is handled separately from ordinary odd magnitudes.

The exact 256-lane affine builder is inherited from minimum GLV10. Active
noncopy denominators feed an exact product/inverse tree; padding and copy
lanes use identity denominators while participating in all barriers. The last
table block has 73 live lanes and 183 padded lanes. It ends in logical bank 4,
not logical bank 5. Gathered table samples and the ordinary builder provide
independent comparisons without copying the entire allocation to the host.

The checked 32,768-byte stack policy is retained and applied before large
allocations. The small static stack census is not used as a reason to change
the runtime reservation. Native/source fallback and driver allocation remain
distinct from compile-time resource records.

After table construction and audits, the code requests a persisting-L2
setaside and an exactly 48 MiB access-policy window for the first three banks
on the actual consumer stream, stream 0. Device and window caps, operation
returns, actual setaside, and every read-back policy field are checked. The
selected hit ratio requires sufficient actual setaside. A successful request
is a retention hint, not proof of residency or a latency measurement.

Twelve full records represent 768 bytes of logical point demand per ordinary
candidate. Six accesses refer to the three small banks and six to the larger
banks. A conditional model in which the small banks hit gives 384 bytes of
cold point demand. That model is not a DRAM measurement, and it does not
include other loads, stores, transaction effects, or eviction. The minimum
GLV10 layout used ten records and 640 logical bytes; the prepared dense19
layout used a larger table with a different protected prefix. No bandwidth
ratio is multiplied by an arithmetic ratio to predict speed.

## Coverage and native module

The minimum's disjoint A/B continuation is unchanged. The original 128-window
family covers `C(137,6)*128 = 1,051,964,508,672` candidates. That family alone
would exhaust before the minimum benchmark duration at the campaign target.
Family B supplies another disjoint set of 128 suffix triples after complete
replay, readback, publication, and device drain. Global schedules are replaced,
producer counters reset, and first states rebuilt with a consistent 64-slot
stride. Search/hit totals continue across the transition. The combined domain
is 2,103,929,017,344 candidates, a chosen family rather than all of `C(150,9)`.

The generic native sm89 image is regenerated from included CUDA sources using
CUDA 12.8.93. The image contains no problem-dependent points, seed, or hits.
Runtime selection is frozen before global initialization; all symbol reads,
writes, and typed launches target that selected pipeline. The generated
manifest records 27 recursive source hashes, eleven kernels, twenty total
globals, and thirteen host-accessed globals.

The generated image is 1,211,360 bytes, SHA-256
`edcd351da72e3458051c728149a24b80d9e12b76541bee69885c4bd942a09cb6`.
The PTX SHA-256 is
`9212de12d47a3a5918732243b08f6a2dc07363a39c1ad62897a7f2051dc623ae`.
The fixed N=24 executable builds successfully; it has not been executed locally.
Source fallback remains possible under the documented native-loader conditions.
Public benchmark artifacts discard module-selection stderr, so package identity
and a valid hit result alone do not directly observe the selected route.

## Validation status and reproduction

Completed CPU checks against the actual header include 378,552 signed codes,
145 affine cases/290 recovery signs, all nine split sign/zero classes, all 87
production fixtures, and six forced HIGH15 boundaries for each reciprocal.
The donor comparison agrees on 2,520 codes. The physical bank intervals,
invalid-record sentinel, and 73/183 tail are checked. Existing A/B checks cover
512 selector triples, 1,152 literal host-schedule lane outputs, and 296 tail
models. Twenty mock cases exercise the real stack and 48 MiB policy setup
bodies, including failures and readbacks.

The native build and actual-cubin resource comparison preserve all eleven
kernel resource tuples relative to minimum. Digest uses 128 registers,
49,152 bytes shared memory, zero stack, and zero spills. The exact batch
builder remains 118 registers, 16,384 bytes shared memory, and 120 bytes
cumulative stack. The cold audit remains 144 registers and 160 bytes
cumulative stack. All emitted functions have zero reported spills. These are
compile-time properties, not throughput measurements.

Final independent native validation passes twelve mock cases, 22 typed launches,
and thirteen symbol roundtrips. All eleven actual-cubin ABI layouts match the
minimum reference. The standalone real-source CUDA audit compiles and links,
covering the 87 scalar fixtures, two additional singular finishes, zero/raw-p
handling, and 1,602 builder records. Its compact sample tail has 66 live and
190 padded lanes; the full production table tail remains 73/183. Mandatory
production startup checks compare 240 table records and the 87 chain fixtures.
No GPU execution is claimed for this prepared candidate.

```sh
cd candidates/subset
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm
```

Portable source checks and regeneration, from the repository root:

```sh
python3 candidates/subset/tests/glv10/check_contract.py
python3 candidates/subset/tests/glv10/check_coverage.py
python3 candidates/subset/tests/glv10/check_policy.py
python3 candidates/subset/tests/glv10/gpu_audit.py --docker qsb-cuda
python3 candidates/subset/tests/native_runtime/check.py --docker qsb-cuda
python3 candidates/subset/regenerate_native.py --docker qsb-cuda
```

Docker examples assume an existing CUDA container with the repository mounted
at `/work`; local toolchain options are documented in the included scripts.
Only the intended subset source, native payload, tests, and public provenance
belong in a submission. Local executables, logs, and experiment artifacts do
not belong in the archive.

## Interpretation and attribution

This candidate is a moderate alternative to the score-regressed minimum and
the larger, unmeasured dense19 layout. The native control's valid result near
the promoted baseline reveals no concrete shared loading or correctness defect
requiring a detour. It also offers no measured native uplift and does not
identify the cause of the minimum GLV10 regression. Those distinctions support
one smaller-table experiment, not a prediction that GLV12 will win. No new
gain or successful target result is claimed here.

Official prior evaluations:

- [Minimum GLV10](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/actions/runs/35947804787).
- [Native-only control](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/actions/runs/35960960900).

The promoted subset implementation and promoted pinning scalar/geometry donors
retain their original license notices. The unpromoted startup hypothesis from
terrapinelf's `a75cf15a` remains credited through the inherited native experiment;
neither its score gap nor the native package establishes a causal JIT timing.
The repository scoring and fresh-seed contracts remain unchanged.
