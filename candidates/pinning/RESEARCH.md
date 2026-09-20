# Pinning: four-wave exact tree top, boundary-only normalization, and complete fused reduction

Effort: medium. Developed with GPT 6 Astra in Codex. This candidate starts from the current promoted Pinning frontier, submission `52cd275a-d385-401b-815b-49a6ab4fc0af`, official score **778,624,395**, landed as `7b0a15bedac7dd533eaf1359e6d102ba8bb4275f`. Only the Pinning editable directory is changed. Native compilation and throughput validation of this composition are delegated to the official remote runner; no local CUDA/C++ build or GPU benchmark was performed.

## What is integrated, and why now

The newly promoted source already contains the previous thirteen-change arithmetic, representation and SHA composition. It supersedes the older 766,671,138 baseline. Its production `pinning.cu`, `GPUMath.h`, hash headers and recovery headers remain byte-identical here. In particular, this candidate retains the current frontier's offset ordinates, fused square constant split, existing packed finish, literal SHA specialization and host pipeline. It does not transplant our earlier PR718 SHA implementation: that combination verified but officially scored 759,448,215 and is not a sound default to restack.

Instead the integration is confined to the upper cofactor tree. The public PR700 description by EvanYan1024 supplied the four-wave top16 scheduling route; we previously independently implemented a complete-arithmetic version in PR705. That earlier package also changed other arithmetic and officially scored 768,652,999 on the older frontier, insufficient for promotion. It is not evidence of a speedup on today's source. This version addresses a concrete overhead in that complete implementation: it needlessly canonicalized every intermediate result, even though its complete multiplier accepts every raw 256-bit operand.

We now keep exact raw residues throughout the four-wave top, canonicalize only its 17 boundary outputs, and independently apply the public second-fold fusion mechanism to a dedicated complete multiplier for this top. The fusion description originates with ercumentyildirim and the subsequent public follow-up by EvanYan1024. Both receive coauthor credit for substantial unpromoted mechanisms. The already promoted source lineage is credited separately as our baseline. Only public descriptions and our own prior implementation were used for these additions; no other solver's unpromoted code or binary was fetched.

## Scheduling and the actual work tradeoff

Let x[0..15] be the sixteen subtree roots already published by the unchanged lower up-sweep. The current frontier issues six sparse warp multiplication waves across this region: three upward waves, its combined root/top-two exclusion wave, and two downward waves. It performs 43 scalar multiplications. The new schedule uses four common multiply call sites across its unrolled waves:

| wave | upward products | exclusion products | active lanes |
| --- | ---: | ---: | ---: |
| span 8 | 8 | 0 | 8 |
| span 4 | 4 | 16 | 20 |
| span 2 | 2 | 16 | 18 |
| span 1 | 1 | 16 | 17 |

This increases scalar multiplication count to 63, while reducing the dependent warp multiply waves from six to four. The 20 extra scalar products occupy previously idle lanes. This is not a claim that scalar work falls or that GPU time improves in proportion to the wave count. Register allocation, code expansion, inactive-lane behavior and instruction scheduling can erase the benefit. Those uncertainties are precisely what this remote submission evaluates.

The source uses one common `qsb_top16_mul(out,a,b)` call for upward and exclusion lanes within each wave. Lanes 0..15 carry exclusion accumulators; lanes 16..23 compute the shrinking upward frontier. Each wave publishes its new products before the next full warp synchronization. The root is written once by lane 16 in the last wave. The sixteen canonical exclusion outputs are then published before the original down-sweep resumes at count 32 and offset `2*N-64`.

No additional shared arena or per-candidate global state is allocated. Original cross-warp barriers remain in the lower sweeps, and all block lanes still enter the collective. The supported production widths are unchanged (128 and 256); the helper also audits width 32 and 64. Tail and unusable leaves retain their existing multiplicative-identity input contract. There is no change to host batch size, slot count, table size, candidate enumeration, hit gate or output format.

## Exact raw intermediate residues

Write B=2^256, K=2^32+977 and p=B-K. The dedicated multiplier accepts all a,b in [0,B), returns a raw result in [0,B), and preserves a*b modulo p. Therefore each intermediate result is a valid input to the next complete multiplication without first subtracting p. An induction over the four waves gives the same final field root and exclusions as the previous fully canonical top16 schedule.

Canonicalization remains at all 17 outputs: the root plus sixteen exclusions in the last wave. Thus downstream consumers receive exactly the same canonical values as the complete top16 reference, even when intermediate products are p or p+1. The code removes 46 scalar normalizations relative to our earlier complete top16 implementation, reducing normalization wave count from four to one. Compared with the current frontier, this is still a different top schedule using more complete arithmetic; do not confuse these two baselines.

The reassociation deliberately does not use the inherited short-carry multiplier. We previously found a structured counterexample where short-carry reassociation changes the field result. That fixture remains packaged and tested. The unchanged lower tree and point chain still have their inherited approximations; this submission does not assert they are universally exact and does not introduce an additional omitted carry.

## Complete second-fold fusion

`Top16Mul.cuh` is a dedicated variant of the current source's complete `qsb_field_mul`. It preserves the convolution, first fold and final overflow correction. Only the second-fold prefix is fused, following the public mechanism already studied on the short-carry path.

For the first-fold value F=L+K*H, its high part is represented by z8+2^32*z9. Combining the low output word z0 with `sfq=z8+977*z9` forms a 64-bit addend for the wide product `977*z8`. The outgoing carry is added to z9 and then propagated through every remaining limb. Crucially the final 2^256 carry is still captured and folded back as K; it is not dropped.

A sufficient full-domain bound is `F < (K+1)*B`: H<=B-2 and L<=B-1. Hence z9 is at most one, and when z9=1, z8<=977. Consequently `z8+977*z9` cannot overflow its 32-bit word. If z9=0, that conclusion is immediate. The retained final correction uses the existing bound that a carried second fold has a small low residue, so its K correction fits the existing three-word correction. No operand-distribution assumption or independent-randomness assumption is needed.

An early audit draft used an unnecessarily tight `F<K*B` assertion, which was not justified by its independent L/H bound. The assertion was corrected to the sufficient `(K+1)*B` bound above before final validation. A negative-control draft also mistakenly expected the raw result of `(p-1)^2` to equal 1: it can be p+1, which is precisely why the final normalization matters. The test now checks that raw representative explicitly and uses `(B-1)^2` to exercise the retained final carry. Neither correction required dropping a carry or changing production arithmetic.

## Validation and limits

The packaged Python semantic interpreter executes the actual inline PTX extracted from the current frontier and candidate. It rejects unknown instructions, uninitialized registers and unsupported declarations. It is a source-level integer model, not a CUDA compiler, native GPU emulator or race detector.

The audit compares the complete fused multiplier bit-for-bit with the frontier complete multiplier, checks both against arbitrary-precision integer reduction, and covers Cartesian limb boundaries, p/B edges, uniform raw operands and an oversampled near-B band where omitted high carries become visible. It separately checks the actual inherited short multiplier against its modeled raw contract.

The tree interpreter is bound to the production source's index expressions, wave predicates, call site and barriers. It checks undefined accumulator reads, read/write conflicts within a wave, the canonical final root and sixteen exclusions, and the complete continuation through lower-tree leaf outputs. It compares boundary-only normalization against a canonical-every-wave reference and checks the full geometry against exact integer field arithmetic. Selected full 128-leaf cases execute the actual PTX through every multiplication, including raw extremes, zero/identity inputs, the prior reassociation counterexample and partially active tails.

The exact final counts and source hashes are recorded in `research/audit-result.json`. Reproduce with `python3 -B candidates/pinning/research/audit.py`; this does not compile C++ or CUDA. `SOURCE-MANIFEST.json` records the runtime sources. There are no executable binaries, benchmark-result overrides, cached scores or Python bytecode in the package.

No performance percentage, register count, SASS count or occupancy claim is made for this candidate. The official build, independent hit verifier and 1200-second ranked evaluation decide whether the composition is useful. If it regresses, preserve the result as a negative for this complete schedule on this frontier rather than repeatedly re-uploading an identical tree.

## Final pre-submit audit

PASS: 9936 complete-multiplier PTX pairs and the same number of inherited-short model comparisons; 168 full-tree cases / 20160 geometry leaf checks; 11 actual-PTX full 128-leaf trees / 1408 leaves. Both negative controls passed. Nine original manifest production files are byte-identical to the latest frontier.

The final public in-flight screen read `960da80` (identical bounded-truncation description to its cancelled predecessors), `f034a9c` (enable existing SHA table/shared-schedule switches), `9f78b02` and `61278ed` (remeasurement/micro or inert descriptions). No new carry truncation is adopted. The SHA-switch proposal has no native throughput evidence yet and changes a separate resource profile; it is retained for result-based follow-up, rather than combined with the audited tree change. No unpromoted source was fetched.
