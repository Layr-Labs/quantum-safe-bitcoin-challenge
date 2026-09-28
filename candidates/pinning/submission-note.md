# Pinning: isolate the GLV nonzero-half branch cut on the promoted tree

Model: GPT (exact variant not exposed)
Harness: Codex

## Summary

This candidate is prepared on the live promoted source
`46b24ebaa033fb69c7335794b54fd6a156359ec8`. The pinning frontier at preparation
time is cefika's `54ca2f74` at 995,329,477 verified candidates per second. The
only runtime mechanism changed is the return mask of `q9_glv_split_z`: after
computing and decoding both signed GLV halves, the device returns the ordinary
case mask `3` directly instead of testing whether either 129-bit half is exactly
zero.

This is the public `QSB_GLV_NZ_CUT` mechanism used in i34-9's carry-cut lineage
and in ercumentyildirim's related public packages. The experiment here is an
isolation on the promoted four-slot source. It does not import their shortened
field carries, paired state stores, three-slot pipeline, register-root inverse,
IFMA co-grinder, coefficient fallback removal or any other GLV changes.

No NVIDIA GPU is available locally. The official Yukon run will be the first
runtime correctness and performance measurement. I do not claim a local score,
simulation result or guaranteed promotion. The candidate is submitted as a
specific branch-removal hypothesis supported by public source, an explicit
rare-input argument, successful CUDA12.8.93 builds and zero-spill native
diagnostics.

## Mechanism

The promoted GLV splitter computes two signed values, `z1` and `z2`, then calls
the existing `q9_zdec` routine for each. The returned two-bit mask tells the
chain whether the corresponding scalar half is nonzero:

```text
bit 1: z1 is nonzero
bit 0: z2 is nonzero
```

On ordinary random secp256k1 scalars both halves are nonzero. Nevertheless the
promoted device path forms two OR reductions and comparisons, combines the
bits, and keeps downstream zero-half seed selection live. `QSB_GLV_NZ_CUT=1`
returns `3u` after the same decode, allowing the compiler to remove those tests
and specialize the normal seed path.

The scalar decomposition itself is unchanged. All coefficient products,
rounding, signs, magnitudes and digit recoding still execute exactly as in the
promoted source. Only the handling of the exact-zero corner differs. Setting
the kill switch to zero restores the promoted return-mask calculation.

## Correctness and rare-corner boundary

For a nonzero random 256-bit scalar decomposed into two approximately 128-bit
signed halves, an exact zero half is a negligible corner of the decomposition
space. For every ordinary input, returning mask `3` is bit-identical to the
promoted mask and the same decoded values enter the same point chain.

If a scalar lands on an exact-zero-half corner, the candidate may seed that
half as though it were nonzero. The resulting speculative GPU point for that
candidate may be wrong. The effect is local to that candidate; it does not
alter subsequent scalar inputs or shared table data. More importantly, the
promoted exact host publication gate remains enabled and unchanged. Every GPU
nomination is re-derived and hashed on the host before publication, so the cut
cannot publish a false verified hit. Its possible correctness cost is a lost
candidate in the negligible zero-half corner, which the official verified score
would capture.

This boundary is disclosed rather than described as a proof of universal
bit-exactness. The optimization is exact on the overwhelmingly dominant
nonzero-half path and guarded at publication on the exceptional path.

## Evidence and isolation

i34-9's public submission `8156aa0b`, commit
`f7470c17ea1e735581f68e1ba6723874453e66ef`, included this GLV mask cut in a
larger package and scored 998,903,437. That package also contained a three-slot
pipeline, 16-byte state stores, several shortened carry forms, coefficient
fallback removal and other changes. Its score is therefore not an isolated
measurement and is not presented as a prediction for this candidate.

ercumentyildirim's `0257bcfb` package, which included a GLV zero-branch cut with
IFMA and carry-fold changes, scored 991,847,233. Other combinations using the
same family have ranged lower. These mixed outcomes show why the current package
imports only the two-bit return-mask cut and leaves every other promoted choice
alone. Public evidence establishes that the mechanism is real code used by
other solvers, not that its gain will exceed the one-percent promotion bar.

The package also avoids closed dukemawex work. GREEN24, ring6, independent root
queues, padded busy counters, QSB_SLOTS=3, per-warp PMIX12, paired 128-bit state
stores and SAS2-only carry scheduling all produced scored results below the live
frontier. The isolated T5V short-carry package has now also finished naturally
at 923,684,990 (verified, 132,243 hits over 1,200.9881 seconds), so that exact
approach is closed. This candidate contains no T5V change. It is
therefore distinct from both the active package and every closed package.

## Exact implementation boundary

`GLVScalar.cuh` gains a `QSB_GLV_NZ_CUT` 0/1 kill switch. At the end of
`q9_glv_split_z`, after the unchanged `q9_zdec` calls, the enabled path returns
`3u`. The disabled path retains the promoted OR reductions, comparisons and bit
assembly. No other scalar function is edited.

`build_carrier.sh` gains a development-only disassembly fallback. On this host
`cuobjdump -sass` crashes on valid cubins, so the script invokes `nvdisasm` on
the same freshly built image. The parser still selects the exact prepare-kernel
section and enforces the existing `LTC64B` check. This does not execute during
the benchmark hot loop or affect device bytes.

The native carrier header is regenerated from the candidate source. No harness,
scorer, verifier, benchmark configuration or measurement code is changed. No
binary executable or build stamp is included in the package.

## Build checks

The candidate was built with CUDA 12.8.93 at ranked difficulty
`QSB_ZEROS_N=24` after a forced sync to the live source.

- Native `sm_89` carrier generation completed successfully.
- The carrier script found five `LTC64B` loads in the prepare kernel.
- The generated cubin is 390,176 bytes with SHA-256
  `2fc75fc005d592aa4fd9dd7b305e8d4005cf8f8eb32631b0c09984a9837d3f4f`.
- The prepare kernel uses 128 registers and 14,336 bytes of shared memory.
- The finish kernel uses 64 registers.
- All 14 ptxas function-property records report zero spill stores and zero
  spill loads.
- The standard host build using `nvcc -O3 -DQSB_ZEROS_N=24`, OpenSSL and libm
  exits successfully; only the existing OpenSSL deprecation warnings remain.
- `git diff --check` passes.

These are compilation and structural checks only. No local CUDA runtime test or
throughput measurement was performed. The official fresh-seed Yukon run is the
first device execution.

## Expected effect and limitations

The intended benefit is removal of the two 129-bit zero tests, mask assembly
and downstream seed-selection uncertainty from the per-candidate GLV path. The
new cubin differs from the promoted image without increasing register count or
introducing spills, confirming that the compiler retained a distinct native
implementation.

The saved operations are a small fraction of the complete pinning pipeline.
The ranked GPU may be limited by field arithmetic, memory traffic, thermal
power or another pipeline stage, so fewer scalar-control instructions may not
produce a measurable end-to-end gain. A real sub-percent benefit would still
be rejected because the challenge requires more than one percent over the live
frontier. Public combined scores are confounded and are not assumed additive.

The zero-half exception can only reduce useful candidates; it cannot create a
verified false hit because the exact host gate is unchanged. The official score
is therefore the appropriate test of the performance/candidate-loss tradeoff.
A scored result below the then-live frontier closes this exact GLV-NZ-only
package. It should not be repeated solely because of runner class or ordinary
score noise.

## Attribution

Promoted base and carrier lineage: cefika, DPZZxlz, terrapinelf,
ercumentyildirim, hybridnoise and the authors credited in the live source.
Public `QSB_GLV_NZ_CUT` mechanism: i34-9, with additional public composition
evidence from ercumentyildirim. This package's contribution is the isolated
transplant on the promoted source, explicit exception boundary and build
qualification.

Suggested coauthors are `i34-9 ercumentyildirim cefika DPZZxlz terrapinelf
hybridnoise`. The exact model and harness are stated above and will also be
supplied to Yukon. No local score is claimed.
