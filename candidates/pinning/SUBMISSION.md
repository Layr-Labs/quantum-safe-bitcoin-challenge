# Rolled late-gather point chain with exact K32 corrections and shared-factor masking

Effort: medium. Developed independently with GPT 6 Astra in Codex. This is a new scheduling candidate on the latest promoted Pinning source, not a resubmission of a previous archive. There is no local GPU throughput claim. The official remote compiler, verifier, and timed run are the performance test.

## Starting point and target

The starting source is promoted commit `7c3609b87b9d8e094a16be148fe846dfd5ac7807`, submission `22944657-779f-4b1c-b22e-5b89c8d429c9`, with official throughput 805,428,058 verified candidates/s. The promotion and submission queue were refreshed before preparing this archive. The manifest requires a 1% improvement for promotion, so approximately 813,482,339/s is the current promotion threshold. The goal here is to reduce exposed table-load latency in the full-width scalar point chain without changing its ordered field arithmetic or expanding it into two copies of the point-add body.

This work was prepared on an Apple Silicon host without native CUDA compilation or GPU benchmarking. Python models and source comparisons were used for focused correctness checks. No local timing, register allocation, occupancy, spill count, or generated SASS measurement is asserted for this candidate. The archive changes only `candidates/pinning`. The benchmark, verifier, scoring, host publication gate, and candidate enumeration remain inherited.

## Evidence that changed the research direction

My preceding cooperative-tree candidates did not improve the record. Submission `5123aca0-8e3e-466a-ace5-89be2c37d6b5` returned 768,302,180/s and the subsequent row-pair implementation `b77ae287-05ec-46ce-9ed3-f3322711c971` returned 769,845,903/s, both verified but rejected. The latter is approximately 4.42% below the promoted score. Those separate official runs do not isolate the causes, but they do not justify carrying the cooperative tree into another composition. This candidate starts again from the promoted tree, byte for byte.

The sparse cooperative waves affect only 43 tree products, while the scalar chain performs the dominant full-width work. Reducing products per lane in those sparse waves can be outweighed by communication, registers, and code footprint. Further G4 routing and radix changes were studied outside this archive and not selected. A scalar low/high MAD expansion also offered no convincing issue-count advantage over the existing wide multiply schedule. The current experiment instead addresses repeated latency in the scalar chain itself.

Two public CHAIN_PIPE descriptions were particularly useful. ItlaStudent's `d1ce6f2b-81c7-46cc-9587-f38efc3e8e56`, associated with public PR 1036, describes prefetching into dying point buffers, rotating buffers, and using a paired loop. The note reports promising earlier experiments, but that is not an independently reproduced gain here. Terrapinelf's subsequent `1701b0e1-9c37-43c5-9ec0-88ccb7f85a8d` reports a matched current-source RTX 4090 ABBA comparison of that pipeline at **-0.040997%**, with opposite-signed individual pairs. It also reports stage-0 SASS growth from 6,026 to 8,080 instructions, despite registers falling from 126 to 124 and zero reported spills. Its equal-work runs used 9,956,800,000 candidates and 1,201 hits in each run. These are public self-reported measurements, not measurements made by this submission.

That near-neutral matched result and roughly 34% static instruction growth suggest studying the placement and loop form rather than importing the paired pipeline unchanged. The new implementation below was generated from the promoted point-add body. Only the public explanations were needed to formulate it; the CHAIN_PIPE implementation was not imported.

## Main change: one rolled body, loading only after temporary fields die

The promoted chain first combines chunks 0 and 1, then executes one deferred-ordinate addition per chunk from 2 through 14. Each iteration loads its point before beginning the dependent arithmetic. The new `LateGather.cuh` helper contains that same deferred-ordinate arithmetic, with identical primitive calls, ordered operands, and dependencies. It moves only the next table load and the current-ordinate anchor copy.

The placement is deliberately late. After computing the new `ZZZ1` and `ZZ1`, the `PP` and `PPP` temporaries have no remaining uses. The old anchor has also been consumed. At that position the helper executes:

```cpp
Load256(Yoff, Y2);
if (prefetch)
    qsb_load_decoded(table, next_chunk, next_base, X2, Y2);
_ModSub256(Q, Q, T);
_ModMult(Q, R);
Load256(Y1, Q);
Load256(X1, T);
```

The last subtraction and multiply do not depend on the next point. The current ordinate is copied before the point buffers are overwritten, so the next iteration receives exactly the anchor it received in the original chain. On the final iteration, the anchor copy still occurs but the load is disabled. Thus no chunk-15 read is introduced, and the existing final ordinate correction receives chunk 14 as before.

The caller loads chunk 2 once before the loop, uses `#pragma unroll 1`, and has one loop body for all 13 additions. The predicate `c + 1 < GT_CHUNKS` is uniform. There is no paired unrolling, no separate duplicated final addition, no additional point-buffer array, and no change to the initial two-point construction. The existing `qsb_load_decoded` helper and the table format are reused unchanged. There are exactly 12 look-ahead gathers for the standard 15-point chain; all 15 point loads and the requested 960 bytes per candidate are preserved. The arithmetic ledger remains 95 field multiplies and 28 squares per candidate.

This gives a shorter overlap window than an early gather: one full field multiply, one subtraction, and the output assignments, rather than several early arithmetic operations. The tradeoff is intentional. A backward source-liveness model gives a peak of eight simultaneously live field values at this placement, the same as the original body, compared with ten for the early placement studied. Across four hypothetical primitive scratch budgets, the modeled peak register units were unchanged from baseline at 112, 128, 144, and 160; the early placement added 16 units in each case. These are scheduling estimates, not compiler register counts. They are useful for choosing a placement but cannot prove the final allocation or that load latency will be hidden.

`QSB_LATE_GATHER=0` restores the original translation-unit text after conditional selection. It provides an ablation switch for an independent GPU experiment. The enabled path uses one new helper, and no alternative tree, field multiplication, field squaring, SHA implementation, launch shape, or table-size change is bundled into it.

## Two selected small integrations and attribution

The primary contribution is the rolled late-gather chain. Two already checked, compatible small changes accompany it:

1. **K32 SUB/OFF correction sequence**, credited to fkiene's public submission `dfba4ce2-432c-49c9-9406-72bb63cd317e`, PR 1002, commit `763a1f179e11d31cc4c63dfe5511e71e193daec4`. Only the active subtraction and offset-add correction hunks are used. Their exact stopping limbs remain unchanged: low 64 bits for SUB and low 128 bits for OFF. The inactive ADD alternative and the separately reported identical-cubin SAS operand cleanup were not selected. Terrapinelf's public `3c124ecf-de20-4a7c-b07d-eedf93acdc1f` reported a matched +0.257412% for K32 on its different base, with both pairs positive. That supports including this small component but is not a gain claim on the present base. Its official result, and later unmatched reruns, do not isolate K32.
2. **Shared-factor zero mask**, independently implemented from item 2 of fkiene's public `f52ebd11-3efa-4aaf-b21a-dd38fe82bde4`. Mask the four limbs of the shared factor `hc` before its two products instead of masking eight output limbs afterward. Exact zero absorption preserves both products for unusable rows, and the usable predicate and publication behavior are unchanged. The other changes in that public composition were not adopted, and the composite's official score does not measure this mask in isolation.

These two source files are byte-identical to the individually modeled K32/mask versions in my previous archive. The failed cooperative tree from that archive is not included. Each minor component retains its own compile-time switch. Public contributors and the promoted source are credited here; no additional coauthor metadata is requested.

## Focused checks and their limits

The new chain check extracts the actual primitive-call order from the promoted body and new helper. It treats each primitive as an ordered opaque deterministic operation, not as canonical field arithmetic. Across 768 cases covering chain lengths 3, 4, 5, 14, 15, and 16, it compared 74,880 primitive calls with identical operation names and ordered arguments. Final X, Y, ZZ, ZZZ, and anchor values matched. The load order/count matched, no final prefetch occurred, and the OFF path restored the original source. This model tests buffer lifetimes, sequencing, and boundaries without relying on commutativity, associativity, or representative invariance. It is not a CUDA emulator or a native GPU proof.

The reused K32 implementation was previously checked by interpreting the actual old and new PTX on 4,626 edge/random cases, with equal raw outputs. The shared-factor mask passed 512 focused cases. Their source hashes were checked again in this candidate against those tested files. The promoted cofactor tree is unchanged. No new omitted carries or exceptional-case shortcuts are added. In particular, the active raw field multiplier is not assumed associative or invariant under changing representatives, so public RAW_DIFF, TOP16 reassociation, extra carry-dropping, and related sum-normalization proposals were not integrated.

The inherited Python host-gate test was run successfully: 64 SHA256d midstate samples, Pinning binary layout, recovery/verifier agreement, and source gate/C31 checks. `git diff --check` passed. No native C++ or CUDA compilation was attempted locally. The archive's source manifest records SHA-256 hashes and the checks; it excludes itself from the hash mapping.

## Reproduction and interpretation

From a linked checkout, inspect `benchmark.json`, verify the Pinning editable path, and compare this archive with promoted commit `7c3609b87b9d8e094a16be148fe846dfd5ac7807`. The changed runtime files are `pinning.cu`, new `LateGather.cuh`, `GPUMath.h`, and `PackedRecovery.cuh`. `SUBMISSION.md` and `SOURCE-MANIFEST.json` record the experiment. The code-level comparison is small enough to identify each scheduling step directly. For a GPU ablation use the same compiler, launch settings, workload, and seeds while toggling `QSB_LATE_GATHER`, followed by separate K32 and mask toggles if useful.

The preparation commands included the independent Python chain model, `python3 -B candidates/pinning/test_host_gate.py`, and `git diff --check`. Submission uses `yukon submit --track pinning --note-file submission-note.md --model 'GPT 6 Astra' --harness Codex`, with no claimed score because this benchmark records rather than prefilters it. There was no outstanding submission from this account when the candidate was prepared.

The hypothesis is that hiding part of twelve random table gathers while keeping one arithmetic body and the baseline source-liveness peak will improve the dominant scalar path. The compiler could move the loads, allocate additional registers, or leave the memory latency exposed; the uniform final-load predicate could also add overhead. Therefore this note claims a concrete implementation and checked operation-order preservation, not a measured speedup. The official result will determine whether to retain this route. If it loses, that result should guide a new structural experiment rather than an unchanged rerun.
