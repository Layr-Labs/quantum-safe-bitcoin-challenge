# Complete ten-window candidate — current status

The complete implementation is now in `candidate/`, generated from corrected
PR64. `submission-wide-windows.md` two directories above is the detailed draft
with architecture, provenance, evidence and limitations. No submission yet.
The exact prepared closure is now copied to production; the previous rejected
source is preserved in `../previous_rejected_source/`.

The point chain is60M+18S instead of95M+28S, using a16 GiB table. The new
chunked GPU builder needs bounded scratch and small batched host ladders; it
never allocates a whole16 GiB host copy. A runtime selector compares external
and CTA-local inverse schedules on distinct real groups, retaining all hits.
It excludes one warmup per mode, then measures four bounded groups of4 batches in F,S,S,F order and requires2% gain to
choose fused. This does not prove the wide table beats the public64 MiB base.

Current source-bound checks all PASS:

- `cpu-results-integrated.json`:12,769 recodings,414 curve chains,822 keys.
- `builder-results.json`:101,506 address cases,3 ladders,4,518 affine entries.
- `schedule-results.json`:11 synthetic timing-policy cases under UBSan.
- `grouped-records-results.json`:36 schedules,1,092 launches,9,756 hit records;
  audit adapted with attribution from Meganpark980320 PR70.
- `adaptive-native-results.json`:full sm_89 and default CUDA builds. Ranked
  split prepare122regs/16KiB,finish78regs/24KiB,zero stack/spills; fused126regs,
  24KiB,120-byte stack,zero register spills. No GPU execution or timing.

`integrated-split-native-results.json`, `fused-native-results.json` and earlier
reports are historical and refer to their own source hashes. The complete
candidate's builder and actual scalar-entry/recovery code were audited; the
prototype-only limitations below are historical, not the current work state.

Corrected `cost-results.json` includes320.375 logical checkpoint bytes per
candidate as well as640 table bytes. Split totals960.375 logical bytes, about
773.8 GB/s at1.25x the current644,546,620 frontier. Physical memory traffic,
startup time and throughput remain unknown. Fused removes checkpoints but
uses one root inverse per256 candidates rather than one per16M. Its stack
traffic is real even though the compiler reports no register spills.

Grok's earlier GT-alias concern was an omitted-definition issue in its packet:
actual aliases use the ten-window geometry and both ladder dimensions8192.
Independent address mapping and actual Scalar tests verify that composition.
Gemini returned ERROR/503; its partial draft's invented timing estimates are
not evidence. Grok confirmed selector ordering and hit preservation; its startup-cost concern
led to the four-batch trial cap. It did not endorse a predicted GPU win. Our
performance hypothesis remains an expectation, not reviewer-provided evidence.

Reproduce: run `integrate.py`, `check_wide.py --base wide_windows/candidate`,
`check_builder.py`, `check_schedule.py`, `check_grouped_records.py`, and
`cost_screen.py` in this directory. Compile from repository root with the
local VM documented in `../LOCAL_CUDA.md`:

```sh
python3 candidates/pinning/research/compile_local.py --source candidates/pinning/research/wide_windows/candidate --default-build --report /tmp/wide-native.json
```

## Historical prototype notes

# Ten-window point-chain experiment

This is an isolated research prototype, not the production candidate or a
submission. The current pinning frontier is 644,546,620 verified candidates/s.
The source base is the field-corrected PR24 closure recorded in FIELD_BASE.md.
Production `pinning.cu` and all subset source remain untouched.

## Hypothesis and concrete progress

Replace the mixed 15-window signed table with six 26-bit and four 25-bit
windows. The widths still sum to 256. There are 2^28 stored odd multiples,
requiring 16 GiB at 64 bytes per affine point. Retain streamed signed recoding,
the deferred-Y XYZZ chain, the corrected field primitives, and eventually the
same external inversion/recovery pipeline.

The point chain falls from 95M+28S to 60M+18S: seed3M+2S, seven deferred
additions7M+2S, and one resolving addition8M+2S. There are five fewer table
lookups and mixed additions. This is a source-level arithmetic count, not a
measured speedup. Existing runtime coefficients must remain folded into the
table base; no cross-instance table cache is allowed.

`wide_geometry.cuh` implements the exact widths, offsets and recoding step.
`wide_probe.cu` uses the actual corrected point primitives and includes both
the original and experimental scalar-entry point chains in comparable kernels.

`check_wide.py` executes the extracted recode and point-chain expressions on
the CPU. It covers 12,769 scalar recodings, including every bit boundary and
values around the group order, and 414 whole curve chains with varying runtime
bases. Sparse OpenSSL table generation checks the actual base/index sequence;
an independent OpenSSL scalar multiplication checks final coordinates and
infinity. All pass. No large table is allocated. The backend checks formula
ordering and aliasing, not GPU field arithmetic or memory behavior.

Both native point-chain probes compile for sm_89 with CUDA12.8.93 at 128
registers, zero stack and zero spills. Static instruction slots are 4,224 for
the reference and 4,240 for the ten-window probe. These loops execute different
numbers of iterations, so similar static sizes do not mean similar dynamic
work. The prototype contains additional mixed-width recoding expressions.
The complete search prepare kernel has not yet been integrated or compiled.

PR64 arrived after this first probe. Its corrected copy is now the preferred
integration base (`../pr64_corrected/`), with provenance in that directory.
`check_wide.py --base pr64_corrected` also passes12,769 recodings and414 curve
chains using PR64's revised scalar reduction, sign-mask decoder and point
helper. The native wide probe still includes PR24; do not describe this CPU
composition check as an integrated PR64 CUDA result. The new decoder returns
0/all-ones, so new table loaders must preserve that mask contract rather than
interpreting it as a0/1 value and negating it again.

The complete corrected PR64 (still15 windows) passes sm_89 and default CUDA
builds: ranked prepare128registers/16KiB and finish78registers/24KiB, both zero
stack/spills, static sizes7,808 and4,608 slots. Finish is now explicitly
unrolled, so its larger static size is not a dynamic-work regression claim.
Its source retains the original field-carry defect; our copy applies the two
already-proved corrections. See `../pr64-native-results.json`.

Reproduce from the benchmark root:

```sh
python3 candidates/pinning/research/wide_windows/check_wide.py
python3 candidates/pinning/research/wide_windows/cost_screen.py
# Start the local VM as documented in ../LOCAL_CUDA.md, then:
python3 candidates/pinning/research/compile_local.py --source candidates/pinning/research --entry wide_windows/wide_probe.cu --report /tmp/qsb-wide-native.json
```

Source-bound evidence is in `cpu-results.json` and `native-results.json`.
`cost-results.json` lists exact logical bytes and explicitly assumed timing
scenarios. Neither these scenarios nor native code generation is a GPU result.

## Why reconsider a large table

Public PR41, source20b85098c8a900e488f070765da69fde7739a35b, reports a different
older homogeneous/warp-inverse implementation improving from318M/s at16-bit
windows to362M/s at24-bit windows, despite a10.7GB table; all are author kernel
clock figures. Its complete verified result363,988,688 was below the now much
stronger frontier. This does not establish a gain on our base, but contradicts
the categorical assumption that a table outside L2 must lose. Its software
prefetch variant regressed; do not copy that prefetch.

[PR41 source and author experiment notes](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/41)

The earlier 14-window/144MiB proposal saved only one addition; this design
tests a larger arithmetic reduction. With the existing16M search checkpoint
buffers, explicitly tracked device allocations total about18.504GiB before
runtime overhead. At a hypothetical1.25x frontier rate, table payload is
515.6GB/s of logical reads. Physical traffic, miss latency and usable bandwidth
are unknown; that number is not proof of feasibility. The cost model shows how
quickly a smaller point-chain fraction or added memory cost erases the margin.

## Required next implementation and gates

1. Integrate the ten-window chain into an isolated complete corrected PR64 copy.
   Native-build the ranked prepare/finish kernels and compare resources; the
   standalone probe is insufficient for full-kernel occupancy claims.
2. Replace the original table builder and whole-table host spot-check copy.
   The original per-entry inverse and16GiB host allocation must not be carried
   forward unchanged. Use small13-bit H/L ladders and chunked batch inversion:
   `m=2*index+1`, `hi=m>>13`, `lo=m&8191`, `H=hi*8192*base`, `L=lo*base`.
   L is never the identity; H[0] is handled by copying L while feeding identity
   into the inverse collective. Distinct positive bounded coefficients ensure
   other H/L sums are valid; audit this bound and builder arithmetic.
3. For each bounded chunk, form `xL-xH`, checkpoint its product tree, apply the
   existing root hierarchy, then finish affine H+L using the inverse. Reload
   the small ladders instead of storing all input coordinates. The final
   table may temporarily hold its denominator in its X slots. Explicitly
   account for scratch lifetimes and check allocation/launch/sync errors.
4. Build host ladder points with batched affine normalization or another
   verified bounded method. Inspect PR53/MakiRH4 source before any import and
   attribute borrowed unpromoted work. Do not needlessly run individual
   inversions for every ladder point. Spot-check only sampled64-byte entries
   copied from the GPU, including chunk edges, against OpenSSL.
5. Audit exact builder decomposition and formulas, scalar/window boundaries,
   full direct recovery/hashes and the complete source include closure. Keep
   carry repairs. Inspect latest current/pending entries before judging whether
   the combined candidate is expected to win substantially.

No upload yet: table construction and full-pipeline integration are required
work, not a request for otherwise-unavailable GPU permission. Existing user
authority allows submission without GPU timing when the completed candidate's
evidence supports an expected substantial win.
