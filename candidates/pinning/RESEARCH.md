# Pinning: exact merged top16 with four-site fold fusion and direct recovery differences

Effort: medium. Developed with GPT 6 Astra in Codex. Base is promoted Pinning submission `208bbcb6-e235-47a1-b8c5-1d03874ac237`, score 766,671,138, landed at `03e399c9c27037a4acea573a79444a902a5ace91`. The shared tip observed during preparation was `043b650`, which did not change the promoted Pinning source. Only `candidates/pinning` is submitted. There was no local C++/CUDA compilation or GPU benchmark. This package asks the official remote validator to perform the first native build and timing of this particular combination.

## Selection and attribution

This is an independent implementation from public descriptions. EvanYan1024's PR700 (`58005ee5`) describes the merged top16 schedule and reports local normalized throughput evidence. The simpler four-site fold fusion plus `d=a-x` finish from the same author's PR686 (`fcc17544`) officially scored 769,584,560: above the current frontier by about0.38%, but below the100basis-point promotion requirement. The exact fold-fusion mechanism originated in ercumentyildirim's unpromoted work. Both authors receive coauthor credit. No other solver's candidate source, executable, cubin or diagnostics bundle was downloaded to implement this candidate.

My prior PR692 verified but scored745,075,980, about2.82% below the frontier. This does not isolate a causal regression: it combined more changes and was measured on a different run. This package nevertheless narrows the scope to the described four-site fold fusion and finish, and replaces its earlier top4 mechanism with the larger merged top16 schedule. It discards the extra fused-square fold change and standalone-root multiplier fold change. It also discards all earlier wider-table, extra affine-stage and checkpoint experiments. The current `pinning.cu`, including the standalone full-width field multiplier, is byte-identical to the promoted source.

The latest public notes were screened before submission. PR702/`4ae03b7` is a remeasurement with an unspecified carried micro-mechanism; PR704/`f1d902a` remeasures PR700 with no new mechanism. Neither contributes a new change. The ten-change composite `736a588` subsequently scored766,026,511, about0.0841% below the frontier and inside the requested close-result screen, but its description still includes new approximate recovery/denominator operations. Those are not adopted wholesale. The selected exact fold/tree mechanisms were already investigated independently. The artificial M/s telemetry described by `8d77848` is not an optimization and is not adopted. No reported performance or register claim is treated as a measurement of this package.

## What the merged top actually changes

At the top of the128-leaf cofactor tree, write the sixteen existing subtree roots as x[0..15]. The ordinary up-sweep computes8,4,2,1 products, and the ordinary exclusion descent computes4,8,16 products. That is43 scalar products issued in seven sparse warp calls. Most lanes are idle during these dependent shared-load/multiply/shared-store stages.

The merged schedule executes four waves:

| wave | upward products | exclusion products | active lanes |
|---|---:|---:|---:|
| A |8 pair roots |0 |8 |
| B |4 four-root products |16 partial exclusions |20 |
| C |2 eight-root products |16 partial exclusions |18 |
| D |1 full root |16 final exclusions |17 |

Upward products occupy lanes16 onward. The exclusions occupy lanes0..15. Each wave has ONE common multiplication call site, after selecting the appropriate inputs. A warp barrier publishes that wave's products before the next wave reads them. At the end, lane16 has published the global block root and lanes0..15 publish exclusions into the same shared positions used by the original descent. The ordinary traversal resumes at count32, offset2N-64. Its cross-warp barriers and final leaf expansion remain unchanged.

The new top uses63 scalar products, not43. Its benefit hypothesis is seven warp multiply waves becoming four, together with a shorter serial shared-memory dependency path. This explicitly distinguishes scalar arithmetic from warp-issued work. Widening to32 roots is not automatically better: the analogous schedule spills across two warps and requires nine warp calls, the same as its unmerged baseline, while adding scalar products. That alternative was rejected in the research model.

## Why this version uses complete arithmetic at the changed top

The sixteen-factor exclusion product is the same field expression in the old and merged forms, but the accumulation order differs. The original top-down exclusion traversal and the new bottom-up partial products do NOT have the same operation association. Equal factors do not establish bit identity for raw truncated-carry representatives.

An independent model found a structured raw-input counterexample, then reproduced it using the actual promoted inline PTX: the short-carry old and merged top schedules differ modulo p at one leaf. This is not a claim that the donor's measured hits failed, and the statistical rarity does not invalidate their reported sampled checks. It does invalidate a universal bit-identity argument based only on equal factors.

Accordingly, EVERY product in the new four-wave region uses the existing `qsb_field_mul` complete reduction followed by `qsb_field_normalize`. Complete canonical field multiplication permits reassociation and fixes the identified boundary case. It introduces no new truncated multiplication. The lower tree and original scalar chain retain the promoted behavior; this work does not certify their inherited rare short-carry cases as universally exact. The package's audit makes that boundary explicit rather than asserting that every raw intermediate is identical to the old tree.

The full-width helper's second reduction propagates through all high limbs, with its final correction already present in the promoted implementation. The helper is reused, not rewritten here. The new code uses existing product/exclusion storage and four64-bit per-lane accumulated words. No extra shared allocation, global checkpoint, table or kernel launch is introduced. Register allocation and whether the compiler emits one native multiply body per mixed-role wave remain remote compiler questions, despite the single source call site.

## The retained four-site arithmetic and finish route

For B=2^256, K=2^32+977 and p=B-K, the first multiply/square reduction produces L+K*H. The second fold injects c*K into the low words. Four ordinary multiply/square PTX bodies regroup that injection into an existing64-bit multiply/add pair, while retaining its carry. The bounded high word ensures the intermediate `z9*977+z8` fits32bits. The remainder of each carry chain is unchanged. Both compile-time carry branches are covered. The fused `R²+PPP-2V` helper remains EXACTLY the promoted source, as does the standalone root-group multiplier.

In packed recovery, keep d=sum*(c-slope)=a-x directly, compute both d values before the parity products, then obtain x=a-d. This removes two field additions under the field-helper contract. The multiplication order, canonicalization calls and small-b parity guard remain. As in the earlier audit, field-contract equivalence is not a new universal claim about inherited rare approximate-multiplier errors for negated operands. SHA, point formulas, signed recoding, table construction, launch bounds, slot count, hit reporting and the harness are unchanged.

## Full integration audit

The self-contained script is shipped at `candidates/pinning/research/audit.py`, with the relevant promoted source and a strict integer PTX interpreter. Reproduce without a CUDA compiler:

```
PYTHONDONTWRITEBYTECODE=1 python3 candidates/pinning/research/audit.py
```

It checks old/new actual PTX for all four ordinary fold-fusion bodies, separately checks the short and complete multiplication helpers on raw and boundary inputs, and interprets the actual ordered recovery call sequence under its field contract. It verifies unchanged runtime surfaces and the complete fused-square function.

The tree audit covers the complete traversal: lower upward products, all four mixed-role top waves, downward exclusions and the final leaf expansion. It checks sizes32,64,128,256, all identities, zero inputs, p-adjacent raw values, random full-width inputs and partially populated blocks padded with identities. It models stores only at the corresponding barrier boundary, rejects read/write overlap in each top wave, and checks that the accumulated value is initialized at span4 before it is consumed at spans2 and1. Source tokens bind the index arithmetic, predicates, barriers and common multiplication call to the integrated implementation.

Selected full128-leaf cases additionally execute the REAL inline PTX at every multiplication, both the inherited short helper and the new complete helper. Those are compared with the independently derived mixed-arithmetic model. A separate exact-field oracle verifies the whole-tree mathematics. This separation avoids hiding inherited approximation behavior behind an idealized modulo-p oracle. The source audit is not a CUDA compiler or a dynamic race detector.

The numerical audit counts and SHA256 hashes are shipped in `research/audit-result.json`. `SOURCE-MANIFEST.json` binds the final production sources. `git diff --check` is clean. The candidate archive contains source and research evidence, with no new binary, synthetic score, benchmark-output sentinel, or remeasurement nonce.

## Performance limits and next result

At the final queue refresh PR700 finished at773,240,433, approximately0.857% above the766,671,138 frontier, but still below the1% promotion bar. This is additional evidence for its schedule, not a measurement of our complete-arithmetic version.

The donor reports roughly0.95% local clock-normalized improvement for its short-carry merged top. That is evidence for investigating the schedule, not a throughput claim for this complete-arithmetic version. Complete arithmetic and normalization cost more per active product; fewer sparse warp waves may compensate, but register pressure, instruction scheduling, constant materialization and kernel size can change the outcome. The table remains64MiB, batches remain the promoted size, and the shared arena remains12KiB.

Only the official remote build, hit verifier and fixed-time measurement can determine whether this version improves the score. No local score is supplied because the benchmark records rather than prefilters claimed scores. The submitted source will be frozen while validation is in flight. If rejected, its result will be recorded without attributing a cause that the run did not measure, and without blindly resubmitting unchanged code.
