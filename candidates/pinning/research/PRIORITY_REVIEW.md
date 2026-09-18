# Pinning priority review — 2026-09-16

The user wants major improvements and preparation timed to submission-slot
availability. Subset c9a85a87 remains validating; pinning 8150e0be finished
rejected at 226,444,961. Pinning therefore has priority. When both tracks have
our pending entries, use visible evaluation progress to choose the likely first
finisher; submission time is only a fallback heuristic.

## Public work inspected

- PR17, nullforest8200, commit 647698377478bf4898f86679791c651e683cf5c3:
  byte-exact development frontier port; author cites former official 650,602,569.
- PR24, alvaroborras, commit 6e76a74fed8e6e5b8439e64ec20f586085f37d52:
  same pipeline plus compile-time deferred-Y specialization; author reports
  5.8% better candidate counter on RTX 3080. This is the preferred research base.
- PR38, welttowelt, commit ba37c4007890cbeedba14a2aba69a9aff376ce49:
  same development port; author reports 645,625,292 verified/s in 90 seconds on
  RTX 4090. This corroborates the architecture but is not our measurement.
- PR41, teddyjfpender, commit 20b85098c8a900e488f070765da69fde7739a35b:
  24-bit table, GPU builder and warp inversion; author reports 363,988,688
  verified/s. Its much larger table does not establish superiority to PR24.
- Notes for PR20,26,29,31,32,34,37,44,45,46,47 and newest 85b0bac were inspected.
  Their principal mechanisms are already present in the stronger pipeline or
  have weaker performance evidence. Do not merge changes merely because newer.

Source copies were downloaded read-only to temporary directories. No upstream
agent instructions were installed. Public PR24 includes compiled artifacts;
exclude its binary and build stamp from any future import. Preserve licenses.

## What PR24 already includes

Fifteen streamed mixed signed digits; 64 MiB interleaved point table folded with
the runtime scalar coefficient; deferred-Y XYZZ chain costing 95M+28S; direct
shared-denominator recovery; prepare/finish kernels with vector checkpoint I/O;
hierarchical root inversion; sequence-dependent SHA midstate; ranked fast-tail
specialization. These are inherited mechanisms, not our proposed contribution.
Its hot multiply and square still drop the final carry; the inverse-tree
multiplier handles that carry. Read the sibling research/ARITHMETIC.md before
reusing these primitives. The point-model audits do not test the hot PTX core.

## Reviewed hypotheses and disposition

1. Fourteen points with four 19-bit and ten 18-bit windows need 144 MiB and
   save 7M+2S from the point chain. This requires new recoding/table-map checks
   and changes memory behavior. Historical RESEARCH.md records an earlier
   15-point variant losing 1.26% when doubling a 32 MiB table to 64 MiB, before
   later pipeline changes made the 15-point design competitive. That is a risk
   signal, not proof that all larger tables fail. Do not submit solely from
   the operation count; no credible large incremental gain is established.
2. Keep only 62 upper product-tree nodes and reconstruct quartet products in
   finish. A 16M batch uses a padded 128 MiB checkpoint instead of 512 MiB,
   saving 768 MiB of combined writes/reads at a cost of 192 additional field
   multiplications per 256 candidates (0.75M/candidate). Independent Python
   model passes 116 blocks / 29,696 leaf inverses, including tails, zeros and
   field-modulus representations mapped to identity. This is a mathematical
   model, not extracted CUDA source or evidence of a speedup. Park as a
   supporting hypothesis until a sufficiently strong overall candidate exists.

`python3 candidates/pinning/research/check_checkpoint_model.py` reproduces the
second model and work counts. Production source has not been replaced.

## Grok and Gemini review, critically checked

Actual signed-in CLIs reviewed the source: grok-4.6-build and
Gemini gemini-3.8-flash-high. Both discourage the wider table; their cache and
performance conclusions remain hypotheses without local GPU measurements.
Gemini favors checkpoint compression; Grok rejects it as too small/uncertain.
Neither opinion is a benchmark.

Corrections to their responses:

- Existing source uses eight CTA barriers in prepare and eight in finish,
  not sixteen per phase. A straightforward quartet rewrite is expected to use
  six and seven respectively; do not quote a count until actual source exists.
- Existing packed products pair lanes across warps (half=128, then64).
  Adjacent-lane shuffles require rewriting BOTH checkpoint passes and their
  numbering. They are not a drop-in replacement. The standalone final root
  inversion can retain its own ordering because its external interface is
  ordered roots and their inverses; not every helper needs the same internal
  tree layout.
- A smaller checkpoint is only part of total pipeline traffic. Memory savings
  alone do not establish a major throughput gain or even a positive one.

## Decision and next gate

No new upload: neither reviewed hypothesis presently supports a confident
expectation of beating the strongest pending pipeline by a substantial amount.
This decision is about the evidence and user's large-improvement preference,
not a requirement for unavailable CUDA testing. Existing authority permits a
well-supported expected-best submission without such testing. Preserve PR27,
continue pinning research, and refresh new pending work before choosing.

Before a future import, retain the public base/source hashes, isolate and test
both field-carry corrections, keep performance claims separate from inherited
results, and build source-derived checks for any rewritten recoder/checkpointer.

## Current continuation: 644.5M frontier and compiler access

Subset 65fb673d (PR60) now validates; original subset PR27 returned rejected.
Pinning still has our free slot. Public pinning 6ce23203 promoted at 644546620,
source 4d39b5f. The related PR24 specialization remains a useful research base,
but copying it plus minor changes no longer meets the major-improvement aim.

New notes 51,53,54,56,58 inspected: 51 reports 539.8M from a subset-core port;
53 combines PR24 with batched host-ladder normalization;54 modifies a slower
monolithic 4080 base;56 stops inverse expansion early and assigns two leaves
per owner;58 remains an older deferred-normalization design. All are author
claims/source descriptions, not our GPU measurements. PR56's serialization
and extra W reload have uncertain tradeoffs; do not adopt it just to reduce
barriers after our subset tree regression.

Re-reading the public development record found an already rejected L2 cache
persistence experiment (-0.5289% end-to-end) and a neutral cache microbenchmark.
Do not reopen that unchanged proposal. Affine batching remains a larger
possibility but requires a bounded traffic design. A limited Karatsuba field
multiplier is another candidate: 48 rather than 64 primitive 32-bit products at
one recursion, with carry/recomposition/register costs still to be counted.

A new local verification route is being prepared: portable Lima 2.2.0, Ubuntu
24.04 ARM virtual machine on this Mac, 4 CPUs/4 GiB memory, 12 GiB sparse disk, no
container runtime. It mounts only dedicated temporary task build storage.
CUDA 12.8 ARM compiler/disassembly tools are installing inside the VM, without
a GPU driver. This may provide real CUDA codegen/resource evidence while
respecting the Mac-only restriction. No GPU runtime result is implied.

## Native compiler results and corrected research screen

The compiler environment is now working. PR24 reference and field-corrected
closures compile and link for sm_89 with CUDA 12.8.93; the corrected closure
also builds with official default architecture flags. Ranked prepare remains
126 registers/16 KiB shared and ranked finish 80 registers/24 KiB shared, with
zero spills or stack. The correction expands static code from 7,200 to 7,816
slots in prepare and 3,096 to 3,288 in finish. This is a correctness cost with
unchanged resource counts, not a speed measurement. Details and source hashes
are in `native-base-comparison.json` and workflow in `LOCAL_CUDA.md`.

Re-reading RESEARCH.md lines 1528 onward found the specific prior Karatsuba
implementation: 73 to 57 wide products but 168 to 240 SASS slots, 42 to 93
IADD3 instructions and 40 to 44 registers. Do not recreate that same losing
schedule. A fundamentally different recomposition remains a hypothesis; fewer
partial products alone no longer qualifies as a credible expected gain.

Actual Grok and Gemini CLI reviews were also checked. Grok identified this
prior Karatsuba result, but repeated an unreachable reduction counterexample
(low remainder near 2^256 while final carry is one). The proven remainder
bound in FIELD_BASE.md excludes that case. Gemini's claimed arithmetic lower
bound and invented resource counts were unsupported; reject those claims.
Neither review establishes that all future field arithmetic gains are
impossible. Use exact native probes and independently checked bounds.

The immutable pending subset PR60 production include closure also passed
real sm_89 and official-default CUDA builds. Its ranked prepare reports 128
registers, 24 KiB shared, and one 8-byte spill store/load; ranked finish uses
80 registers, 24 KiB shared and zero spills. This is useful successor feedback,
not a reason to cancel the pending evaluation or a proof of its throughput.
Production source remains untouched while pinning is selected.

Its GPU audit also compiles (12 kernels, no execution); source-bound reports
`subset-pr60-native.json` and `subset-pr60-audit-native.json` are retained here
because pinning is the selected track. Both report closures still match the
pending source. The local compiler VM is stopped after use to release memory.

Latest public notes PR61–63 were inspected. Pinning PR61 specializes the sparse
33-byte compressed-key SHA block but provides no native or GPU comparison;
constant folding in the actual base must be checked before assuming removed
source expressions reduce instructions. Treat as a minor supporting option.
Subset PR62 reports paired RTX4090 90-second runs improving its base by 10%
with deferred XYZZ, streamed recoding, direct recovery and a 15-window table.
Our pending source already contains the first and third mechanisms plus
external inversion; streamed recoding and mixed table geometry remain useful
future research when subset is selected. Do not add reported gains as if
independent. PR63's table-negation and scalar recid changes have author native
codegen evidence only and are small supporting mechanisms. Our split finish
already stores coordinates as scalars. Neither later subset entry has completed
a scored GPU evaluation as of this refresh.

## Ten-window prototype and stronger PR64 base

The resident-affine schedule was reviewed with explicit synchronization/root
latency budgets; no credible major-win schedule emerged. Grok's concrete
critical-path objections were useful; Gemini's ERROR/503 partial draft used
the wrong inverse algorithm and invented latency/resource figures, which were
discarded. Details and boundaries are in AFFINE_PIPELINE.md.

The new ten-window projective probe in `wide_windows/` cuts five additions,
from95M+28S to60M+18S, with a16GiB signed table. It passes12,769 scalar recodings
and414 curve chains against OpenSSL, and both reference/experimental native
probes use128registers with no spills. It is not a complete candidate: the table
builder, bounded-memory spot-checking, and full pipeline integration remain.
The required measured/predicted evidence is specified in that directory.

New public PR64,072d9b8e,head a7b21d0f62e6d73b66fe820e228f8db50d504716, reports
a50.9% local RTX3080 candidate-counter gain from no-alias annotations and a
bundle including grouped readback, exact rare scalar reduction, read-only
loads and recid unrolling. This is author evidence on another GPU, not a
ranked RTX4090 gain. Its 64MiB table exceeds3080 L2 but fits nominal4090 L2,
so its speedup must not be transferred numerically. It is nevertheless the
strongest compatible next base to inspect and combine.

The four source/license files were inspected and preserved, with provenance,
under `pr64_corrected/`. Exact comparison shows GPUMath differs from PR24 only
in point-helper restrict qualifiers; both hot carry defects remain upstream.
Our copy retains the proven multiply/square repairs. Changed restrict call
sites use distinct coordinate/local arrays and separate kernel allocations;
the intentionally in-place field primitives were not annotated.

Complete corrected PR64 passes explicit sm_89 and default CUDA builds.
Ranked prepare:128registers,16KiB shared,zero stack/spills,7,808 static slots.
Ranked finish:78registers,24KiB shared,zero stack/spills,4,608 static slots.
The unrolled finish cannot be compared to the old loop by static size alone.
The ten-window CPU composition with its new recoder and sign-mask helper also
passes12,769 recodings/414 curves. `pr64-native-results.json` and the wide-window
reports bind all claims to exact source hashes.

Priority remains pinning while subset PR60 validates. Next implement the
complete ten-window candidate on corrected PR64 and its bounded, batched
table builder; do not resubmit a small annotation-only duplicate. Carry forward
public-source attribution and the new sign-mask contract. No new submission
has been made in this research step.
