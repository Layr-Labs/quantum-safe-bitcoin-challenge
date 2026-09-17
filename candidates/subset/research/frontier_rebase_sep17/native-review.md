# Frontier native comparison

**Prefer the canonical final-guard variant over the weak-region experiment.** The guard retains PR217's ordinary common-path instruction-site count, zero local traffic and the same rounded register allocation. The weak experiment does not demonstrate a codegen gain: its smaller loop image hides the fact that PR217 skips cold correction envelopes. This is a static native screen, not a GPU performance result or standalone correctness qualification.

All four existing production reports passed native sm89 and default-flag compilation with CUDA12.8.93. This review verified every reported source hash against its corresponding frozen local source and binds the SASS/PTX/report hashes in `native-review.json`. No build or GPU run was started here.

| Source | Registers | Shared B | Stack / spills B | Whole-image nonNOP |
|---|---:|---:|---:|---:|
| Accepted PR189 | 126 | 32768 | 0 / 0 | 10175 |
| Exact pending PR217 | 100 | 32768 | 0 / 0 | 12800 |
| Weak-region experiment | 119 | 32768 | 0 / 0 | 14187 |
| Canonical final guard | 104 | 32768 | 0 / 0 | 14197 |

With256register-per-warp allocation,100 and104 registers both allocate3328 registers/warp;119 rounds to3840 and126 to4096. At256threads, all four permit two register-resident CTAs, not three, under65536registers/SM. This is static resource legality, not measured occupancy. The guard's four-register reported increase does not cross an allocation threshold. A default build pass does not prove the driver-JIT image equals this native sm89 image.

## Actual repeated ordinary loop

PR217, weak and guard each traverse12 ordinary additions for chunks2 through13. Counter initialization is2, increment is1, and the emitted exit compares against14. The separately emitted final chunk14 is outside these loop regions. All loops contain four128bit table loads and no LDL/STL instructions.

| Image | Inclusive PC region | Static sites | Cold bypassed inline sites | No-correction region sites before digit-selection skips |
|---|---|---:|---:|---:|
| PR217 | `0xff90–0x15db0` | 1507 | 154 | 1353 |
| Weak | `0xff50–0x15630` | 1391 | 0 | 1391 |
| Guard | `0xff80–0x15d80` | 1505 | 152 | 1353 |

PR217 has nine conservative field-correction checks. When every active lane avoids the cold prefilter, their forward branches skip inline operand/call/result envelopes of18,14,18,18,18,18,14,18,18 sites. These skip154 sites plus the out-of-line correction callees. The guard's corresponding envelope widths are18,15,18,18,18,18,11,18,18, totaling152. Thus guard's two fewer static sites are cold-envelope allocation changes: the ordinary no-correction region count remains1353.

The weak loop removes those cold checks, but uses unconditional weak carry folds and two-fold add/sub arithmetic. It adds38 common-path region sites per ordinary add, or456 across12 adds, before differences after the loop. The raw−116 static-site comparison would wrongly imply a7.7% work reduction. Same no-correction assumption and equivalent digit-control branches give16183 traversed loop sites for PR217 versus16639 for weak, counting predicated-false instruction sites but excluding the final untaken backedge. These are not cycles. There is no local traffic for weak arithmetic to eliminate in this baseline.

Digit extraction has equivalent selection branches at positions36+17*i. Word0 selections skip12sites twice, word1 selections skip7sites four times, for52skipped sites across either12-add loop. The common-path delta therefore survives this control adjustment. If any active lane enters the conservative correction prefilter, PR217/guard execute a different path; no frequency or timing for that event was measured.

Accepted PR189's loop at `0xf8d0–0x15340` has1448 sites, but combines12 ordinary and one final addition in the same loop with a conditional extra product. It is not an apples-to-apples13×1448 executed-work comparison. Its inherited hot arithmetic also has the known near-p defect, and its HM43 is unbounded. Acceptance supplies performance evidence, not full-domain arithmetic proof.

## Surrounding work and selection

The emitted `kernel_build_epochs` instruction sequence is identical across all four images, including3120sites/3108nonNOP and its own stack operations. This proves producer codegen preservation for that kernel; it does not separately measure the consumer's SHA phase. The root source and inverse-tree source match PR217 in both experiments. Register renaming and layout mean source equality alone is not a claim of identical root latency.

The weak experiment's combined postloop final-point/guard/normalization/recovery-prepare/tree-leaf region grows1486static sites through the first tree barrier (`0x15640–0x22d30` versus PR217 `0x15dc0–0x1d7d0`). This includes exceptional branches and cannot be charged to every ordinary candidate. Both experiments add complete final-point handling absent from the original baseline; their whole-image growth is therefore not a dynamic regression estimate.

The evidence supports rejecting the weak experiment as the selected optimization and retaining the canonical guard variant for separate arithmetic, complete-chain, audit and packaging gates. The guard introduces no observed ordinary-loop site-count or local-memory regression and no register-allocation cliff. Its final guard may still cost ordinary predicates, alter scheduling or instruction-cache behavior. It has **no measured throughput result**. No candidate/source/stage was edited by this review, and no submission action was taken.

Reproduction is read-only: parse `sass.txt` from each native report's build directory, select the `Function : _Z13kernel_digest...` section, match instruction lines `/*hexPC*/ instruction ;`, and count the inclusive regions above. For each forward cold bypass, count instruction addresses strictly between the branch and its join; the JSON records all nine branch/join/call PCs. Compare complete `kernel_build_epochs` instruction strings for the producer equality check. Native report and SASS hashes prevent accidental cross-image comparisons.
