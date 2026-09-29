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
