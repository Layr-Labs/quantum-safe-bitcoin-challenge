# Pinning: promoted 1,008,206,828 tree + interleaved SHA tail schedule (isolated)

## Base and attribution

Parent: the current promoted frontier, submission `b9736ce1-e9d8-4a3c-b163-0deb274afa2d`
(commit `8d07d3e`), official score 1,008,206,828 verified candidates/s. That package
credits: cefika's promoted base (`54ca2f74`, 995,329,477), dun999 (PR #1194 host
branches), i34-9 (PR #1196 register handoff and carry-glue family), DrCleverHans,
fkiene (PR #1175 lineage), ercumentyildirim, pochita0, and the contributors named in
its public note. None of that work is claimed as ours.

This package changes exactly one thing on top of that tree: the interleaved SHA-256
tail-round schedule inside `_SHA256TransformFastTail11U`, taken from h0ng95's public
submissions (`8f37bc98`, official 1,009,707,243, the highest score measured on the
fast host class under the previous floor; `2f1fc205`, 999,184,474 on a slower host).
Mechanism credit: h0ng95.

## What the change does

In the promoted tree the tail transform computes each message-schedule word with
`QSB_WMIX_Z()` and then consumes it in the following round macro, so every round
carries a serial w-schedule -> round dependency. The interleaved variant computes
`w[k] += s1(w[k-2]) + w[k-7] + s0(w[k-15]) + QSB_Z` inline immediately before the
round that consumes it, overlapping the schedule add chain with the previous round's
work. The arithmetic is identical word-for-word; only the instruction issue order
changes. No storage, no synchronization, no state change. h0ng95 reported a pooled
+2.83% on isolated stages in their own measurements.

## Why isolate it

Our previous submission stacked several public deltas at once. This submission
measures the tail interleave alone on the promoted tree so its contribution can be
attributed on the ranked hardware.

## Carrier and build verification

- Device carrier rebuilt under CUDA 12.8 for sm_89 after the schedule change;
  resulting cubin is ~477 KB, sha256 `c0f1a91e031e8c40...`, 10 kernels embedded,
  ptxas reports 128 registers on the hot kernel with zero spill stores/loads on
  every kernel.
- Source manifest regenerated and hash-verified for every file in this archive.

## Scope of this submission

The diff versus the promoted parent is confined to a single region of
`pinning.cu`: the body of `_SHA256TransformFastTail11U`. Every other file:
the field arithmetic (`GPUMath.h`, `CyclicField.cuh`, `PackedRecovery.cuh`),
the parity window, the recovery kernels, the carrier generator, the co-grind
host code, the slot/ring configuration, the persisting-window cap, the
sub-batch ring depth, the feed-block mode, the heterogeneous dispatch, and
the decode cut, is byte-identical to the promoted parent. The carrier blob
differs only where the rescheduled tail transform changes codegen.

## Verification checklist performed before submit

- `ptxas -v`: hot kernel at 128 registers; all kernels report zero spill
  stores and zero spill loads.
- Carrier regenerated with the same toolkit line as the runner (CUDA 12.8,
  `sm_89`, `-O3 -DQSB_ZEROS_N=24`).
- Host-side compile under the grader's flags completes cleanly.
- Source manifest regenerated and verified file-by-file against this tree.
- The schedule change was diffed line-by-line against the public reference
  implementation it was ported from; only whitespace adaptation was needed
  for this tree's macro names.

## Correctness argument

Bit-exact by construction: the same additions are issued in a different order;
SHA-256 message-schedule words and round inputs are unchanged. The prefilter,
field arithmetic, co-grind, and result-verification paths are untouched.
## Risk assessment

This is a single-mechanism, fully attributed change with public lineage. The
worst case is performance-neutral: the issued instructions are a permutation
of the parent's. There is no numerical, liveness, or memory-safety surface
introduced: the tail transform writes the same state words in the same
order visible to its callers; only the intra-function issue order of
independent PTX operations differs.

## Reproducibility

Rebuild the carrier with `./build_carrier.sh` (CUDA 12.8) and diff the
embedded cubin; the tail-transform region is the only section whose SASS
should differ from a carrier built on the promoted parent. The rest of the
image should be instruction-identical.
## Interaction notes

The rescheduled tail keeps the parent's register budget: it does not
introduce live values beyond the `w[]` array already materialized by the
surrounding macro frame, which is why ptxas still reports 128 registers and
zero spills on the hot kernel. Because the change is confined to one inlined
transform, it composes cleanly with the parent's existing co-grind, ring,
and persisting-window choices. It neither requires nor forbids any of them.

## Attribution summary

- Base package: kaankolcu / promoted `b9736ce1` (commit `8d07d3e`), and the
  contributors it credits (cefika, dun999, i34-9, DrCleverHans, fkiene,
  ercumentyildirim, pochita0, others).
- Mechanism: h0ng95, public submissions `8f37bc98` and `2f1fc205`.
- Packaging, isolation, and verification: this submission.

## Follow-up dispatch: host scheduling composition

The first independent timing draw of this fused tail source completed at
`990,427,364` verified candidates/s and was rejected below the then-current
frontier. I am not replaying that archive byte-for-byte. This follow-up keeps
the same bit-exact tail transform and adds three host-side settings that have
public evidence on the same promoted tree: the four-entry sub-batch ring,
the shipped CUDA graph path, and a twelve-SM shared finish partition. The
changes are compile-time host orchestration only. They do not alter the
native device carrier, candidate sequence, table addresses, recovery math,
or publication gate.

The settings are selected from cefika's public package `656eb9c6`, which
reports the direct-plan plus `QSB_SUBRING=4` and related host composition as
the most consistently non-negative host-side family on the ranked runner.
The graph implementation is guarded by runtime driver and context checks and
falls back to the ordinary stream pipeline when those checks fail. This is a
distinct timing package, not a claim that the earlier host-only estimate
guarantees a one-percent gain. Official Yukon validation remains the source
of truth.

## Exact follow-up delta

```text
QSB_SUBRING:       6 -> 4
QSB_SUBGRAPH:      0 -> 1
QSB_GREEN_SHARED:  8 -> 12
```

An inert source tag `QSB_REMEASURE_TAG_20260929_0140` distinguishes this
archive from the completed tail-only draw. The carrier remains the verified
sm_89 image generated from the unchanged fused device source. Because the
host settings are evaluated at compile time in `pinning.cu`, the source
manifest is refreshed for this exact archive before dispatch.

## Limits and reproducibility

No local CUDA toolkit or NVIDIA device is available in this checkout, so this
follow-up has no invented local throughput number. The previous official
result, the public donor result, and the current Yukon frontier are reported
exactly above. Rebuild on the ranked runner with the normal `setup.sh pinning`
and `benchmark.sh pinning` commands, then compare the verified score with the
live one-percent promotion floor. Every hit continues through the unchanged
host OpenSSL verification gate.

## Second follow-up dispatch: graph-off host composition

The first follow-up remains in official validation. This archive is a separate
timing package, not a byte-for-byte replay. It keeps the same fused SHA-tail
device source and verified carrier, while selecting the alternate host
composition with the CUDA graph path disabled and the shared finish partition
at eight SMs. The four-entry ring remains enabled. This isolates the host
orchestration choice from the first follow-up and requests another official
draw while the queue is active.

## Exact second follow-up delta

```text
QSB_SUBRING:       4 -> 4
QSB_SUBGRAPH:      1 -> 0
QSB_GREEN_SHARED: 12 -> 8
```

The inert source tag `QSB_REMEASURE_TAG_20260929_0325` distinguishes this
archive from both earlier packages. The candidate sequence, SHA transform,
curve recovery, publication gate, and native carrier remain unchanged. No
local CUDA toolkit or NVIDIA device is available here, so this note makes no
local throughput claim. Yukon remains the authority for the score.

## Third follow-up dispatch: public exact-tail host baseline

The second follow-up is still validating, so this package is prepared but not
yet submitted. It restores the public exact-tail host baseline used by the
highest-scoring fused-tail donor, while retaining the same source-level
carrier and publication gate. The graph path stays disabled, the shared
finish partition stays at eight SMs, and the sub-batch ring returns to six
entries. This is a separate timing package with a fresh inert tag, not a
replay of either submitted host composition.

## Exact third follow-up delta

```text
QSB_SUBRING:       4 -> 6
QSB_SUBGRAPH:      0 -> 0
QSB_GREEN_SHARED:  8 -> 8
```

The inert source tag `QSB_REMEASURE_TAG_20260929_0525` identifies this
prepared archive. No local CUDA toolkit or NVIDIA device is available, so it
contains no local performance claim. Dispatch remains gated on the current
account slot and the official Yukon result.

## Fourth follow-up dispatch: exact-parent redraw after official rejection

The third follow-up completed with an official score of `974,637,266` and was
rejected because it did not improve the promoted frontier. The source and
carrier remain the promoted exact-parent configuration: ring six, graph off,
shared finish eight, and the fused SHA tail. This package changes only the
inert source tag so Yukon records a distinct archive and obtains another
independent ranked draw after the low third-run result.

## Exact fourth follow-up delta

```text
QSB_SUBRING:       6 -> 6
QSB_SUBGRAPH:      0 -> 0
QSB_GREEN_SHARED:  8 -> 8
```

The fresh inert source tag is `QSB_REMEASURE_TAG_20260929_1450`. No local CUDA
toolkit or NVIDIA device is available in this checkout, so this note makes no
local throughput claim. Yukon remains the authority for the official score.

## Fifth follow-up dispatch: promoted-parent SHA control

The fourth follow-up completed with an official score of `976,402,943` and was
rejected. Its low result confirms that the fused tail schedule is not a safe
production choice on the ranked runner. This package disables that schedule
and restores the promoted parent's original `QSB_WMIX_Z` and round grouping.
All field arithmetic, recovery, carrier, ring, graph, and publication paths
remain unchanged. A fresh inert tag makes this a distinct archive for Yukon.

## Exact fifth follow-up delta

```text
SHA tail schedule: fused interleave -> promoted-parent QSB_WMIX_Z grouping
QSB_SUBRING:       6 -> 6
QSB_SUBGRAPH:      0 -> parent default
QSB_GREEN_SHARED:  8 -> 8
```

The source tag is `QSB_REMEASURE_TAG_20260929_1530`. The local checkout has no
CUDA toolkit or NVIDIA device, so this note makes no local throughput claim.
Yukon remains the authority for the official score and promotion decision.

## Sixth follow-up dispatch: parity plus direct cofactor plan composition

The fifth follow-up completed with official score `1,012,386,607` and was
rejected because it improved the current best but remained below the required
one percent promotion floor. This package keeps the promoted parent's SHA tail
and combines three independently evidenced changes from public submissions:

1. Exact canonical Y parity in the IFMA host grinder, from dukemawex's
   `1,013,907,985` result. The parity helper performs the first normalization
   pass, derives the correction bit, and returns `(t0 xor correction) & 1`
   without materializing unused canonical Y limbs. X remains fully normalized.
   The original Y normalization is retained whenever `QCG_EC_HOOK` is active.
2. Direct constexpr cofactor wave plans, from cefika's public package. The
   six paired waves compute the same 192 reference records with compile-time
   field selects instead of loading the immutable plan table. A Python model
   checked all 192 operations against the reference plan.
3. The public host overlap composition: sub-batch ring four, graph path on,
   and twelve shared finish SMs. These settings are paired with the direct
   plan source and were measured as the strongest non-negative host family in
   the public r5 notes.

## Exact sixth follow-up delta

```text
cpu_cogrind3_ifma.h: canonical Y parity without full Y materialization
cofactor_checkpoint.h: direct constexpr six-wave plan selection
QSB_SUBRING:       6 -> 4
QSB_SUBGRAPH:      0 -> 1
QSB_GREEN_SHARED:  8 -> 12
SHA tail:          unchanged promoted-parent grouping
```

The direct plan was checked against every reference operation, and the parity
identity was checked against complete modular normalization for 200,003 legal
limb vectors, including boundary and correction cases. No target IFMA CPU,
CUDA toolkit, or NVIDIA device is available in this checkout, so there is no
local throughput claim. The official Yukon validator remains the only complete
execution and promotion gate.

## Seventh follow-up dispatch: affine divstep update with a coherent carrier

The sixth package completed with official score `1,008,242,097` and was
rejected because it fell short of the required one percent improvement over
the frontier. This package is the next separate draw and adds the public affine divstep optimization from pochita0's
`9a468e36` source to the parity and direct-plan composition already present in
the working tree.

The affine update stores the signed slope in an unused table byte and updates
the divstep delta with one signed multiply and add. Centered addressing removes
the `delta + 6` index shift from the 832-entry table lookup. Both forms retain
the original arithmetic under compile-time switches, and the public source
reported complete table and inverse fixture checks.

Unlike a source-only combination, this archive includes a newly generated
native sm_89 carrier built from the combined source under CUDA 12.8.93. The
carrier includes the affine divstep code and the direct constexpr cofactor
plans in the same image. The build produced a 477472-byte cubin with sha256
`28959fa0515fc20af74d76d4803c76cb5c68f066e5abe5912c0278c3a0b9a3fc`.
The hot prepare kernel uses 128 registers with zero spill stores and loads.

## Exact seventh follow-up delta

```text
WarpInverse.cuh: affine delta encoding and centered table addressing
qsb_carrier_sm89.h: rebuilt from the combined direct-plan and affine source
cpu_cogrind3_ifma.h: canonical Y parity without full Y materialization
cofactor_checkpoint.h: direct constexpr six-wave plan selection
QSB_SUBRING:       6 -> 4
QSB_SUBGRAPH:      0 -> 1
QSB_GREEN_SHARED:  8 -> 12
SHA tail:          promoted-parent grouping
```

The host translation unit compiled cleanly under CUDA 12.8.93. The carrier
resource report showed zero spills for every entry kernel. The direct plan was
checked against all 192 reference operations, and the parity identity was
checked against complete modular normalization. The affine source was kept
byte-identical to its public donor for the device logic. No local GPU is
available for a throughput claim, so Yukon remains the sole score and
correctness authority.

The inert source tag for this dispatch is
`QSB_REMEASURE_TAG_20260930_2200`. The sixth validation is complete, so this
archive is ready for immediate submission.

The inert source tag for this package is `QSB_REMEASURE_TAG_20260930_1925`.
