# Pending frontier source review

Use **PR217 as the strongest correctness-oriented pending base**, keeping accepted PR189 as the official performance control and our PR200 untouched. This is a source-review recommendation, not qualification or a measured win. The supplied queue records PR189 at **512,865,536 verified candidates/s**, evaluated commit `3c3d748e7e46bbf68fcd22d179c44b231eaa15de`, promoted as `df2fb8b8f5f3e47ff7a6a1848ccebf40e902f634`. PR217/216/220 have no official scores in that snapshot.

## Exact sources and reusable mechanisms

| PR | Exact current head | Disposition |
|---|---|---|
| [217, Meganpark980320](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/217) | `5b4a5054ab494586dded4806c72f025784ea8da4` | Best pending base: corrected selected field arithmetic, bounded warp root and fallback, PR201 codegen/drain |
| [220, AbdelStark](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/220) | `d1c5d7cba08bf8f6c67e67fbea51a84bed98cfd3` | Substantial paired-candidate inverse amortization; separately port and measure after correctness repair |
| [216, i34-9](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/216) | `8c99d934cf67f29e7ba010853d7ea27f2cda78e6` | Different 4-lane root; weaker boundedness/evidence than PR217, not additive to HM43 |

Heads were read with GitHub API, then source blobs fetched by immutable blob SHA. Full fetched-file SHA256 values and queue binding are in `public-review.json`. Temporary read-only copies are under `/tmp/qsb-frontier-public-review/{217,220,216}/candidates/subset`.

## PR217: strongest starting point, with precise limits

Actual `GPUMath.h` carries out the final reduction carry in selected multiplication and squaring. A cheap high-word prefilter calls `qsb_field_cold_correction`, whose exact test retains false positives and adds C=2^32+977 for a carry or value >=p. The root helper caps complete batches at16, returns uniformly without publishing partial output, then lane0 computes fixed-exponent x^(p−2), clears word4 and broadcasts. This addresses the known carry defect and the missing unbounded accumulator premise rather than inheriting those risks from official acceptance. The dormant `square32.cuh` remains the old a973 file; default selected pinning-square code is repaired. Alternate square selection must not be treated as qualified.

Source also confirms `__ldg` table reads, a compile-time deferred-add boolean, the rare scalar-reduction path, and a checked blocking default-stream hit-buffer copy replacing the extra ranked synchronize. Completed-work diagnostic publication follows successful copy. The table is **mixed15/64MiB, [18]+[17]*14**, not regular16/32MiB. The selected finish is PR150's **10M0S squaring-free recovery**, unlike our a360's retained10M1S finish. Changing to regular16 therefore requires an explicit geometry/builder/ordering port; it is not already supplied by this base.

The author reports matched1200s RTX4090/CUDA12.8.93 runs: exact PR150572,237,097/s versus candidate581,990,636/s, +1.70446%, all165,331 hits verified. Their native/JIT claim is100 registers/no stack versus128/120B control. **These are author results, not reproduced here, and the comparison is against PR150, not accepted PR189.** Since HM43 is already promoted, that combined gain cannot be attributed solely to corrections/codegen. It supports choosing a tested base, not promising another1.7% over the current frontier.

An isolated weak ordinary point-add experiment on this base is compatible in principle. Preserve canonical seed, exceptional guards, recovery/root inputs and finish; prove all weak add/sub folds, normalize the four accumulators at the boundary, and check actual selected helper and alias/restrict contracts. Retain the original geometry as the first control or explicitly qualify a regular16 port. Source and compiler/JIT/resource evidence must bind the resulting candidate, not be inherited numerically from the donor.

## PR220: real amortization, unresolved traffic tradeoff

Actual source runs two candidates per thread, parks candidate A's three finish fields and denominator, forms W_A*W_B, invokes the existing256leaf tree once, and splits the inverse using two multiplies. Extra arithmetic is3M/pair, or1.5M/candidate. It includes a CTA barrier before reusing first-state shared storage and clamps the odd-tail descriptor while suppressing inactive hits. Hit mapping preserves `(2*block+candidate)*256+tid`.

`volatile savedA[16]`, `savedB[4]` and `savedBf[12]` represent256B of parked source state per thread/pair. The author's128register/200Bstack/8Bspill report is not a dynamic local-traffic measurement; the spill counter does not count every explicit local access. Their claimed2.5–6% gross opportunity assumes an inverse-time fraction and excludes added multiplication/traffic costs. The pre-HM43 root-free ablation cannot establish that fraction after HM43. Their source still uses defective GPUMath91a067 and unbounded HM43. Porting pairing onto PR217 could be useful later, but combining it immediately with weak arithmetic would obscure both qualification and performance attribution.

## PR216: distinct root alternative

The new helper uses30bit batches, nine32bit signed limbs, four participating lanes, mask0xF exchanges and sparse modulus correction. This is a real design alternative, not just a renamed HM43. However, its outer loop remains unbounded and canonicalization assumes |state|<2^261 without a proof established here. Its surrounding selected field files retain the known carry defect. The source comment claims39,851→28,155 root cycles against the old scalar inverse; that is neither comparison with HM43 nor an official whole-workload gain. Do not adopt it over bounded PR217 merely from that microbenchmark.

## Remaining queue triage and attribution

PR223 host-drain and PR224 rebased PR201 codegen mostly overlap PR217. PR212's external first-SHA producer removes consumer shared state/barrier at the price of global states and an extra launch; its +2.63% is an author attempted-work measurement, not the verified score. PR211 is the previously reviewed external inverse/checkpoint family. PR215 enables144MiB14term tables with a material cache tradeoff. PR204/208's direct constant argument is a small codegen probe; source syntax cannot guarantee zero registers or higher occupancy. None displaces the bounded corrected base.

Credit Meganpark980320 for unpromoted PR217 reuse, with its retained PR201/scarletbright and PR200/hybridnoise provenance; retain accepted AbdelStark/PR189 and inherited arithmetic notices. Any later reuse of unpromoted PR220 or PR216 needs the relevant author/coauthor disposition. Acceptance does not validate every field input or erase prior authorship.

No candidate, stage, production, queue or submission was changed. No compilation, device run, formal proof or independent throughput reproduction was performed. PR200 remains preserved.
