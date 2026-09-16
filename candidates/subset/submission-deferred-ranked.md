# Subset: deferred-Y arithmetic, composed inversion and a ranked kernel

Effort: xhigh. The implementing model is GPT 6 Astra through Codex. This
submission combines substantial public pending work on top of the promoted
GPU-epoch architecture. It is an expected-performance experiment with CPU
checks, not a measured GPU improvement. No local score is claimed.

## Starting point and contribution

The official promoted frontier observed during selection is **433346795
verified candidates per second**, submission
`873ed724-9815-4e13-a02f-072f21e3f992` by nullforest8200, promoted at
`cfc0d9cf5dd7cc8c607a8ff53b0dadb90e6424c1`. That score belongs to the base.
The source is the [production challenge repository](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge).
Historical development numbers in `TREE_INVERSE.md` are preserved upstream
reports and do not measure this candidate.

Our earlier pending subset submission `c9a85a87-e2b8-4ab8-8d4f-c79d96546d03`,
[PR27](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/27),
introduced four-lane shuffle levels around a smaller shared inverse tree.
This successor incorporates that work through a newer public integration.
The contribution here is composition of two pending candidates, source review,
additional execution of point helpers against independent arithmetic, and
reproducible verification records. The underlying optimizations are credited
to their original authors rather than presented as new inventions.

Substantial unpromoted sources used:

- **jacklightChen, PR36**, commit
  `472b536106d2e2aae2b927cbf217678a58bebb73`:
  [streamed deferred-Y and composed inverse integration](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/36).
  Its `GPUMath.h`, fixed-base accumulator and composed inverse helper supply
  the arithmetic base for this submission. Its independently modeled audit is
  retained as `audit_integrated.py`.
- **DPZZxlz, PR40**, commit
  `08a4b7b44aee9cddc5b8c2852f9ed2e6b4f3d5f8`:
  [ranked short-epoch template specialization](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/40).
  Its complete small `tree.cu` patch applies to PR36 with zero fuzz. The
  specialization and three explicit launch instantiations are preserved.
- **nullforest8200, PR17**, commit
  `647698377478bf4898f86679791c651e683cf5c3`:
  [streaming recoding and deferred affine anchor](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/17).
  This is separate unpromoted pinning work adapted into subset by PR36, beyond
  the author's promoted subset base.
- **alvaroborras, PR24**, commit
  `6e76a74fed8e6e5b8439e64ec20f586085f37d52`:
  [deferred/final helper specialization](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/24)
  and arithmetic-model work inherited through PR36.
- **Meganpark980320, PR28**, commit
  `d03c1fc50d4d0983ca206718cdf8537374a085a4`:
  [fused inverse-tree levels](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/28),
  applied by PR36 to the shared levels above our four-lane reduction.

All five contributors are coauthors. GPL notices and COPYING are retained.
The promoted architecture also carries its earlier upstream credits, including
the GPU epochs described in the historical documentation.

## Why select this combination

The promoted implementation already avoids the dominant repeated hashing of
the long common prefix. Six early omissions define an epoch; 256 lanes visit
distinct triples among the last thirteen pushes. It uses a runtime-generated
signed fixed-base table, shared SHA schedules, folded scalar coefficient,
projective recovery and block inversion. Replacing it with an older projective
kernel would discard those gains.

PR36 reduces actual field arithmetic and shared-tree synchronization. PR40
exposes the scored mode as a compile-time constant, allowing the compiler to
remove generic hashing and diagnostic paths from that instantiation. These
changes affect different parts of the kernel and preserve its data flow. This
is our best expected candidate among the reviewed public alternatives; it is
not evidence that the combined binary beats PR36, PR40 or the official frontier.

The large opportunity relative to our previous pending submission is the
deferred multiplication chain. Relative to PR36, the incremental hypothesis is
only specialization: its benefit could be small, zero or negative. The archive
does not claim that merging two promising changes adds their speedups.

The scan covered all public subset submissions visible before selection,
including PR28, PR33, PR36, PR39, PR40 and the newer PR42. PR39's streaming
recoding alone is already present in PR36. PR33 specializes flags on an older
base. PR42 reports about 99-104M verified candidates/s on an RTX 4080 using
another two-pass epoch path; these are author-reported measurements on different
hardware, not a controlled comparison with this candidate. Its main mechanism
is already present in the promoted epoch architecture. Earlier projective,
constant-data and launch-size changes did not justify replacing that base.

## Arithmetic and synchronization changes

`_FixedBaseSignedProj` now consumes the raw four-limb SHA scalar and streams
the existing signed recoder. It avoids materializing the sixteen-entry digit
array. Table geometry remains sixteen windows with the existing 32 MiB layout.
The table is still built from each runtime problem; no answers or hits are
precomputed. Both recovery signs are still evaluated.

The deferred chain stores `Ycore = Yactual + anchor*ZZZ`. The next addition
therefore obtains the correct slope numerator from
`(Ynext + anchor)*ZZZ - Ycore`. The affine seed omits one multiplication and
retains the first point's ordinate as its anchor. Chunks 2 through 14 call
the deferred helper and update the anchor. Chunk 15 resolves the deferred
term with the final helper. Conversion to homogeneous coordinates stays
`(X*ZZZ, Y*ZZ, ZZ*ZZZ)`, which preserves the recovery caller's convention.

The sixteen-point chain changes from **116M+30S to 102M+30S**, with the same
three further multiplications for homogeneous output. Fourteen field
multiplications are removed per candidate. This is a source operation count,
not a measured runtime percentage or a claimed end-to-end speedup.

For inversion, the lowest two levels stay within groups of four neighboring
lanes using full-warp shuffles. The remaining shared tree then fuses two
levels in each direction, with disjoint subtrees owned by individual threads.
Thus PR27 and PR28 compose at different levels. An earlier local research
note incorrectly treated the two entire helpers as mutually exclusive; that
categorical conclusion has been corrected.

At 256 threads, the final inverse helper has **4096 bytes of shared tree
storage, 12 block barriers, 765 multiplications and one inverse**. The promoted
helper uses 16384 bytes and 18 barriers with the same arithmetic count; our
previous four-lane helper uses 4096 bytes and 14 barriers. These are helper
counts, not total kernel shared-memory, register or occupancy measurements.
The helper contract remains power-of-two whole-warp blocks from 32 to 256,
with identity factors for inactive lanes and nonzero actual denominators.

## Ranked specialization and fallback semantics

`kernel_digest` becomes `template<bool RankedShortEpoch>`. The short-epoch
launch selects `<true>`, while generic enumeration and explicit tile paths
select `<false>`. Host selection requires single GPU, no explicit tile,
single_hash, no easy/calibrate diagnostics, n=150, t=9, and the existing
42/218/44-byte input-shape checks with full preimage length 9906.

The true instantiation selects the epoch descriptor and scheduled hash path,
uses the ordinary leading-zero gate and skips the second SHA transform of
each public-key digest. This does **not** skip either recovered public key:
the loop over both recovery signs remains. It selects the existing early-six
plus WIN3 hit-index mapping. The false instantiation preserves generic runtime
branches, including double hashing and diagnostics.

No new compiler standard, launch geometry, candidate distribution or hit
format is introduced. `single_hash` was added to the host short-epoch guard
as in PR40 so a double-hash request cannot accidentally enter the specialized
single-hash path. Additional template instantiation can increase compile/JIT
cost; dead-path removal and register reduction are hypotheses until measured.

## Local checks and reproducibility

The work host is an Apple Silicon Mac without nvcc or an NVIDIA GPU. Earlier
`yukon setup --track subset` passed its CPU verifier smoke. The baseline
`yukon run --track subset` could not start the Linux CUDA grinder locally.
The user authorized selection by expected performance using this Mac only.
No CUDA compile, sanitizer run, GPU timing, occupancy or spill count is claimed.

Commands from the benchmark work directory:

```sh
python3 candidates/subset/check_candidate.py --report /tmp/subset-source-audit.json
python3 candidates/subset/audit_integrated.py
python3 candidates/subset/check_deferred_source.py
python3 candidates/subset/preflight.py --note candidates/subset/submission-deferred-ranked.md
# On a CUDA host, additionally compile fresh production and arithmetic-audit sources:
python3 candidates/subset/preflight.py --cuda
yukon setup --track subset
yukon run --track subset
```

The first audit executes extracted production inverse, recoding, epoch and
scheduled SHA functions through a temporary C++ library. OpenSSL replaces
field primitives and CPU threads emulate barriers/shuffles. It passed 3840
inverse outputs, 6144 complete SHA256d digests from two fresh synthetic problems,
10262 signed recodings and 8086 combinadic unrankings. The measured CPU-emulated
barrier counts are 6, 8, 10 and 12 for 32, 64, 128 and 256 lanes respectively.
Every inverse case has exactly `3*n-3` multiplications and one inverse.

The inherited independent Python model passed 19200 inverse outputs, 20000
arbitrary-field chains, 1000 on-curve chains, 10262 scalar recodings and 28
complete runtime-folded multiplies. The new `check_deferred_source.py` goes
beyond that model by compiling the actual two point-helper bodies with
OpenSSL field operations and comparing every intermediate state with the
ordinary non-deferred reference. It passed 2000 arbitrary-field chains and
128 on-curve chains, totaling 31920 intermediate-state comparisons; the final
curve results also match independent affine addition. These tests exclude
singular denominators, an inherited limitation of the incomplete group law.

The final production/audit include closure has nine files and fingerprint
`d0ccab7c66830429f8bc39dc619ad4b369883d0ea9c7929b64a569592c65512e`.
Its `tree.cu` SHA256 is
`6694341256a400aacf344b48013aeabdff988481593b615d0d3da6e6d96ed832`.
Source-bound reports prevent an older passing audit being reused after a
header change. CPU checks do not validate GPU instructions, races, template
code generation or the complete emitted-hit pipeline.

## Research retained but not selected

Grok through Grok CLI (model metadata `grok-4.6-build`) and Gemini through
Antigravity CLI (`gemini-3.8-flash-high`) reviewed relevant source and research
in read-only sessions. Their recommendations were checked against code;
unsupported claims about measured runtime fractions, compilation, spills and
guaranteed speedups are not results. Codex made the implementation and selection.

The research directory also contains a generator for an isolated SHA/EC kernel
split. It writes 32-byte digests between kernels, adding 64 bytes of transfer
per candidate plus tiled launches. Its hash/scalar transport and candidate
identity checks passed on the CPU, including wraparound and guarded partial
tiles. It is not in the production include closure. The EC-only stage may
retain the same register bottleneck, so the extra kernels have no demonstrated
benefit. A bounded epoch-geometry scan likewise found under 1% modeled SHA-only
gain among its sustainable configurations. Neither experiment is selected.

## Risks and next evidence

Saving arithmetic and barriers does not guarantee a faster GPU binary.
Deferred anchors, streamed scalars and shuffled sibling products extend live
register state. Compiler allocation or spills could offset the source savings.
Earlier separate upstream streaming/specialization experiments reportedly
regressed slightly in another source context; that warning is retained.
The score also includes runtime costs such as JIT and table preparation.

Official validation must establish compilation, verified hits and the full
ranked score. Compare the result with the 433346795 promoted frontier and with
PR36/PR40 when their results appear. Treat a regression as evidence to isolate
components, not as something static operation counts can refute. Returned
failures should strengthen candidate-local checks without changing trusted
scoring. The result monitor remains results-only and does not authorize future
code changes or submissions by itself.
