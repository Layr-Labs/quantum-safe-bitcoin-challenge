# Hierarchical checkpoint for one-layer affine pairing

Status: the accompanying HybridPair.cuh implements this structural design. Python source-call and PTX semantic audits are supplied; no local native compile or GPU timing. Historical design alternatives below are distinguished from the implemented path.

## State and arithmetic reduction

For each candidate, let the seven nonzero pair denominators be d0..d6 and T=product(d). Run the existing global block-cofactor/inversion hierarchy on one T per candidate. Store just original scalar k and C=product(other candidates' T), 64B total. With block inverse I, C*I=1/T.

In the second prepare stage reconstruct only prefixes p0=d0 through p5=d0...d5. This costs five multiplications; no need to reconstruct full T. Starting r=1/T, for j=6..1 emit inverse(dj)=r*p[j-1], then r*=dj. The final r is inverse(d0). Cost: first T6M + global cofactor approximately3M + local inverse1M + recomputed prefixes5M + backward12M =27M/candidate excluding O(1/N) tree savings and super-root hierarchy. This compares with28M for seven separately checkpointed cofactor leaves. Add seven affine pair finishes14M+7S and eight-point chain46M+14S:87M+21S, versus current chain95M+28S. Recovery work common to both is excluded. Not a GPU throughput prediction; new exact arithmetic may cost more perM than inherited short-carry chain.

31,640 inverse checks passed with candidate counts2/32/64/128. Shuffled in-place per-lane checkpoint overwrite passed. Global state is64B/candidate write plus64B read, not the old256B each way. Existing final recovery state also occupies64B, so the same allocation can be overwritten lane-wise after reads. This does not make the extra global traffic free.

## Kernel sequence and ownership

1. First prepare: SHA scalar as promoted, regular recode, read x coordinates for pairs(0,1)..(12,13), form7denominators and T. Existing N128 cofactor collective on T. Save scalar and excludedT in the four existing128-bit state planes; publish block root.
2. Existing root hierarchy prepares groups, inverts super-roots, expands block inverses. Its weighted inverse output is unnecessary in this phase but can initially remain for compatibility. No inverse is executed by one lane inside the main candidate CTA.
3. New second prepare: read scalar/checkpoint and block I, derive r. Reconstruct local prefixes from x table loads. Load carry point14; visit pair6..0, forming affine pair sums and immediately accumulating into existing deferred-Y XYZZ chain. The first pair and carry use the existing3M+2S two-affine seed. Six remaining pair sums use mixed additions. Resolve final Y once. Then form existing recovery denominator and call packed_prepare, replacing intermediate state with ordinary vbar/tbar, and roots with recovery roots.
4. Run existing root hierarchy again, followed by original finish/hash kernel.

Adds four kernel launches per full batch (new prepare plus one root hierarchy); not per32candidate cohort or perwave. Must audit both slotted host streams and tail/inactive lanes. Do not modify verifier, harness, scoring, or output order.

Root alias hazard: a late lane must not read first-phase inverse after another lane publishes recovery root. Existing packed_prepare begins with a block barrier and calls collective barriers, which should protect these earlier reads if EVERY lane reaches it and no early publication/return is introduced. A negative control detects unsafe publication before all old-root reads. Do not add or remove barriers without source-level lifecycle audit. Distinct blocks access distinct root indices.

## Shared-memory/register layout

Naively storing six prefixes and a scalar takes28KiB/N128, which weakens the residency budget. Instead use five prefix fields p0..p4 (20KiB) plus recode state M (4KiB). Keep p5 only temporarily in registers until pair6 inverse consumes it, before the long chain accumulator is established. Total24KiB. Inverse r survives across the reverse loop; scalar digits are extracted from shared M on demand, not a15-entry register array or separate7.5KiB digit arena. After the pair chain, reuse the arena for existing12KiB cofactor tree, protected by its entry barrier. No whole-array prefix lifetime extends into recovery.

At each pair, construct lambda, xpair, ypair, then kill its operands before invoking mixed-add. Do not keep all seven affine pair results. Table decode should load only necessary recode limbs, rather than keeping all four64-bit scalar limbs live across field asm. This is a design to limit liveness, not evidence that ptxas fits128registers. Preserve exact correction carries in any new field helper; measure actual register/performance only through official remote validation as user forbids local native compilation.

## Exact pair and reordered-chain boundaries

The baseline reduces raw hash k once modulo N. Seven pair points have scalar coefficients with different2-adic valuations; |a+b| and|a-b| are less thanN, so they cannot be equal/opposite moduloN. For pair indices up to6 the largest is below2^240. A/2 is nonzero of prime orderN. Thus all paired affine denominators are nonzero by proof, not probability.

For reversed accumulation (carry14, pair6,...,pair0), pre-final equality/opposite is excluded by nonzero2-adic valuation and magnitude<2N: it cannot equal0 or±N. At final pair define q=sum(first two signed terms), |q|<2^35. D=2k-N. Opposition is D=±N, hence k=0 modN. Doubling requires D-2q=±N, hence k=q or N+q. The exact two-digit recurrence gives q=sign(D)*(abs(D) mod2^36 -2^35), yielding the two nonzero critical residues ±c, c=(N mod2^36)-2^35=0x4d0364141. Raw hash representatives N and N+c also fit below2^256 and must be included.

20,125 scalar coefficient checks /140,875 pair checks passed, including raw hashes>=N. Exact point prototype now processes the intended reverse order, with a stable sort by orientation only. Earlier prototype sorted same-tag points by coordinate as an accidental secondary key; its sum checks were still valid, but did not test this specific order. That modeling issue was fixed and all point tests rerun:77full scalars +4explicit exceptional cases PASS,5reference-fallback cases. No frozen submission was changed.

Implementation must correctly handle0,±c moduloN. Possible route for ±c: exact one-time scalar guard and a slow existing-order scalar path that avoids the newly reordered singular addition; this still needs direct point verification and must not write a shared digit layout overlapping other lanes' prefix data. The infinity case must preserve appropriate baseline behavior or explicitly handle the recovery of infinity. Do not silently discard new exceptional inputs. A generic complete exceptional handler is another option; evaluate its effect on hot-path liveness. Proof-domain checks and exception model are not a CUDA correctness proof.

## Next concrete work

Implement an independent current-frontier candidate only after inspecting the live helper ordering and shared-arena definitions. Start with the64B checkpoint and24KiB arena layout, carry-complete new pair arithmetic, on-demand direct digit extraction, scalar exceptional route, and original recovery/hash finish. Keep a source-level Python interpreter/differential audit tied to implemented functions. Review exact helper cost vs inherited short-carry cost and record table traffic: first x-only pair loads, prefix x-only reload, then full pair reload. Do not submit if implementation shows a clear occupancy/traffic regression; fix the identified cause rather than adding minor scheduling patches. A sound structural candidate can go to official validation without local GPU timing, labeled unmeasured.
