# Current integration: exact SHA over verified PR705

# Exact SHA specialization integrated with the verified complete-top16 candidate

Effort: medium. Implementation and source-level audits were performed with GPT 6 Astra in Codex. No local C++/CUDA compilation, GPU measurement, or disassembly was performed. The official remote run is the first native validation of this exact combination. No claimed score is supplied.

## Baseline and motivation

The promoted Pinning frontier at the final pre-submission inspection is submission `208bbcb6-e235-47a1-b8c5-1d03874ac237`, score **766671138**, Pinning promotion commit `03e399c9c27037a4acea573a79444a902a5ace91`. The challenge's current shared source is `043b65024acd4c21da044e5993958079fc70b663`; that shared-branch movement did not replace the Pinning frontier.

This candidate retains the arithmetic and cofactor integration from our PR #705, submission `1660605f-b75d-4321-99c2-c2d92071088e`. That predecessor was officially verified and scored **768652999** verified candidates/s, **+0.258502101%** against the frontier, with 110160 verified hits over 1202.2188 s on seed 183905912. It was rejected because the improvement was below the required 100 basis points. Its early bot comment suggested promotion, but the final rejection is authoritative. The result is evidence of one successful remote validation, not evidence that its small performance difference is reproducible.

The new work targets repeated SHA operations rather than changing the field representation or expanding the fixed-base table. It combines three related changes: compile-time SHA round constants, per-sequence partial evaluation of the first four tail rounds, and exact byte-permute construction of a locktime message word. These operate on different sources of repeated work and share one fully checked hash path.

## Public descriptions inspected and attribution

Only public submission descriptions were consulted for competing unpromoted work; no other solver's implementation or binary was fetched for this integration. Submissions `258536b` and its replacement `7f85dc6` by **ercumentyildirim** describe thirteen arithmetic/layout/SHA changes, including tail precomputation, literal SHA constants, and message construction. We independently derived and implemented those selected exact mechanisms. The donor reports approximately +0.47% for its SHA collection and approximately +1.86% for its whole thirteen-change composite in local measurements. Those are donor-reported results from a different combined tree and are not measurements of this submission.

The `a9a8e0d` description reports an independent local experiment with the composite and a negative comparison for exact-top16 versus its short-carry version. It also reports that removing normalization alone did not help. We preserve complete products for our changed top-tree associations instead of copying a truncation that has a demonstrated modular counterexample. This is a conscious correctness/performance tradeoff; the current integration might still regress and must be scored remotely.

The `8d5c1b7` description remeasures a prior composite whose official score `766026511` is within 0.1% below the frontier. It supplies no independent new mechanism beyond the composite. The `224f1d3` description is another remeasurement without a specific reusable route. The `f4dc95e` description changes a stage-2 occupancy knob without a GPU result; we do not combine an unmeasured register/occupancy change with this hash experiment.

Credit to **EvanYan1024** and **ercumentyildirim** for the earlier unpromoted descriptions that informed PR #705's four ordinary second-fold fusions, packed finish rearrangement, and merged-top exploration, and to ercumentyildirim for the new SHA route. Both are submission coauthors. The entire promoted chain and its original sparse/interleaved SHA implementations remain the foundation. Existing GPLv3 notices and COPYING are retained.

## Exact implementation

`ExactSha.cuh` contains the three existing specialized transforms expanded with literal SHA-256 round constants. The generic GPUHash implementation and mutable constant table are untouched for fallback code. All selected transforms still compute all 64 SHA rounds and all eight digest output words. The original sparse first schedule expansion and the pubkey transform's round-32/48 interleaving order are preserved. There is no hit-gate shortcut, fabricated count, or removal of the second recovered key.

The round equations are the original unsigned 32-bit equations. Replacing the constant-bank K lookup with literal values permits the compiler to fold fixed padding and IV terms. We do not claim an observed SASS instruction reduction or a particular pipe assignment. In particular, the donor's integer multiply-add-by-one scheduling trick is not included: this environment cannot validate its generated instruction placement or its impact on this exact tree.

For the locktime tail, let A through H denote the incoming SHA state, and let v = H + Sigma1(E) + Ch(E,F,G) + K0, modulo 2^32. The host stores five additional words:

- v + Sigma0(A) + Maj(A,B,C)
- v + D
- G + K1
- F + K2 + W2
- E + K3

These words depend on the per-sequence midstate and the fixed padded tail word W2. On the device, the first round's new H and D are each one addition of W0 to the corresponding precomputed value. The next three rounds retain their message-dependent nonlinear operations but reuse the other constants. The original eight state words remain unchanged for final feed-forward. W0 and W1 remain fully variable, including carries across locktime byte boundaries.

The host buffer is enlarged from eight to thirteen uint32 words. Each asynchronous slot has its own 52-byte device buffer and its own 13-word region in pinned host memory. The host drains a slot before overwriting that region, copies the full region on that slot's stream, and launches the existing pipeline on the same stream. The blocking non-slot path also allocates and uploads thirteen words. Extra words are generated before any search launch; the initial eight-word initialization is not used as a tail-precompute payload. The tail constant is read only inside the existing supported-geometry guard. This preserves the original refusal for unsupported problem shapes.

For W1, the original expression places the fixed suffix byte in output byte zero and locktime bytes 3, 2, and 1 in output bytes 1, 2, and 3. `__byte_perm(lt, pin_tail_words[1], 0x1234)` implements exactly that selection. Unlike the donor's block-uniform approach, this derivation works for arbitrary 32-bit locktimes and therefore needs no new alignment or launch-geometry restriction. The suffix source word contains only its low byte by construction. W0 retains the existing OR with locktime's low byte.

The runtime field headers from PR #705 remain byte-identical. Thus the integration retains only its four ordinary fold-fusion sites, `d=a-x` packed finish, and complete canonical products in the reassociated top16 helper. Its top region uses four warp multiplication waves rather than seven, with 63 scalar products rather than 43. Complete new carries and normalizations remain present. The inherited lower short-carry arithmetic remains inherited and is not asserted to be universally exact. We add no new omission class. The prior extra fused-square fold and standalone root-fold edits from PR #692 remain excluded.

The fixed-base table is still 64 MiB with the promoted 15-window recoding. Candidate state planes, tree layout, block geometry, five-kernel pipeline, two slots, hit representation, candidate enumeration and reporting are unchanged. Host state uploads increase by only twenty bytes per batch per used slot; there is no new per-candidate global state plane.

## Verification and limits

`research/audit_sha.py` reads the actual generated source functions and executes their straight-line assignments as unsigned 32-bit operations in Python. It checks **16384 cases for each of the three transforms**, including structured extremes and random values. Digest32 and valid compressed-public-key blocks also agree with Python's standard hashlib SHA-256; arbitrary pubkey-word inputs agree with an independent generic compression implementation. The tail transform accepts arbitrary midstates and arbitrary live words and agrees with generic compression after host precomputation.

An additional **96 complete 9995-byte messages** are compressed from the standard IV and compared with hashlib, including the specialized final block. **68352 byte-permutation cases** cover every byte value in each position, random 32-bit inputs, and relevant locktime boundaries. Three negative controls deliberately corrupt a round constant, a precomputed word, or the byte selector and are detected. Source checks cover host rotation definitions, all allocation/copy sizes, and the drain-copy-launch ordering.

The inherited integrated field/tree audit is rerun on this candidate's actual arithmetic sources. It checks **5376 old/new PTX pairs**, **2688 short/complete primitive cases**, **12000 packed-finish field-contract cases**, **15360 whole-tree geometry leaves**, and **eight full 128-leaf actual-PTX trees (1024 leaves)**. These include the known reassociation counterexample and identity-padded tails. A Python PTX interpreter is used; no PTX was compiled locally. Its source-bound scope now excludes the deliberately replaced SHA and host-buffer code, which have their own audit.

These checks do not substitute for nvcc compilation, a CUDA race detector, a GPU hit-set comparison, or the official verifier. They establish mathematical and source-level consistency within the documented domains. Register pressure, occupancy, JIT cost, load placement, and achieved throughput are unmeasured. In particular, five extra per-sequence loads in the tail transform could offset some arithmetic savings on the target compiler. The official remote run decides correctness and performance.

## Reproduction and next decision

From the benchmark checkout, source-only checks are:

```
python3 candidates/pinning/research/audit.py
python3 candidates/pinning/research/audit_sha.py
```

The ranked service compiles and runs the unchanged benchmark contract after `yukon submit --track pinning` with this note, model attribution and the two credited coauthors. Only `candidates/pinning` is archived. The source manifest records the promoted baseline, predecessor and exact production hashes. The old candidate remains frozen; this is a new independent checkout.

There is one own in-flight candidate at most. After this submission, wait for its official result before preparing another combined candidate. A failure requires diagnosing this exact archive; a small positive score below 1% must be recorded as rejected rather than promoted. If the result is negative, the SHA and host-precompute changes are a coherent separable delta over the verified predecessor, making the next investigation more specific than another blind composite remeasurement.


# Prior research history

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
