# Bounded HM43 prototype: source contract and outstanding qualification

Frozen production/audit closure: `f47ad8241adac7aac14e63ad947735ada3bf3565ae60266fa7153b6744d5583a` (19files). This source is an isolated, **unqualified** prototype. The unbounded6cb control, ready3ac and submission stage are preserved. Source construction is not a CPU, native or GPU result.

`prepare.py` copies the exact6cb include closure and GPL notice into a fresh directory strictly below `bounded/`. It refuses to overwrite an existing candidate. Its two changes reverse exactly to the control: `hm43_warp_inverse.cuh` gains a finite uniform budget/boolean return, and the EC192 root call site gains an original-root scalar fallback. All other include hashes remain identical. Original JeanLucPons/VanitySearch arithmetic and AbdelStark donor provenance remain in the source/receipt; this local increment changes budgeting and fallback, not the matrix algorithm.

The helper API is `bool hm43_warp_inverse(uint64_t result[5],int lane)`. Every one of32 lanes participates, including guard lanes. `true` means the algorithm completed; the original noninvertible-input zero result remains valid for the helper's test extensions. `false` means the budget was exhausted and **all caller result arrays are untouched**. The production inverse tree supplies canonical nonzero roots.

`HM43_WARP_MAX_BATCHES` defaults to16. Compile-time values0..16 are allowed for bounded fallback tests; larger or negative values are rejected. The counter is uniform. After at most16 complete62-bit matrix updates, failure returns before any next-batch collective. A success on the sixteenth update exits normally before that failure guard. Cap0 still executes the initial full-warp input broadcasts, then returns false uniformly without writing results. No unconditional8/16-batch convergence claim is made.

The caller retains the original shared tree[1] until publication. Only ecid0 handles false: reload tree[0..3][1], explicitly zero root[4], call unchanged `_ModInv`, then execute the existing four-word publication. All32 first-EC-warp lanes reconverge before the existing root subgroup join. Other EC warps remain at that join; the tree's five dynamic subgroup joins,192-owner participation, identity padding and descent are unchanged.

The independent range review is `../range-review.md/json`: terminal coefficient-row L1≤2^62; U/V remain≤p with predivision magnitudes<2^318; R/S aftert≤16 batches satisfy maxabs≤1+t·p<2^260; row products plus modular correction remain<2^322, inside signed384 bits. Exact quotients fit signed320 bits. The cap closes the new finite-width argument; it does not prove convergence, and capped cases retain the baseline scalar-inverse assumption.

Outstanding qualification is owned separately:

- Execute the actual bounded helper with all32 cooperative host lanes at default16, plus forced0 and forced1 on roots that need more than one batch. Check uniform booleans and untouched five-word outputs on false. Detect premature result publication and false-success mutations in addition to carry/guard mutations.
- Execute the actual bounded EC192 caller/tree at default16 and forced0/1 with a disclosed fallback backend. Verify original-root use, fifth-word reset, exact field inverses, all owners/identity padding, unchanged joins and finite progress. If exact legacy `_ModInv` projection is used, identify its source and arithmetic modeling; an OpenSSL fallback oracle must not be mislabeled execution of that legacy helper.
- The false-output contract itself preserves the initialized private root, so merely deleting the defensive reload or root[4] reset is observationally redundant under that contract. A mutation test that poisons private root data before fallback must disclose that extra injection; it tests defensive reload behavior rather than showing the valid helper corrupts its output. No-reload or missing-clear alone should not be advertised as a detected numerical defect if it passes.
- Bind default and sm89 production/audit builds, union19 source hashes, and actual native root/loop/producer resources to this new fingerprint. Unbounded6cb or ready3ac compiler results cannot qualify this source. No compiler/VM operation is performed by preparation.
- Source-bound CPU tests are not GPU collective scheduling, race or throughput evidence. Preserve ready3ac until a separately reviewed qualification/selection decision.

No submission, cancellation, queue mutation, runner change or external call is part of this preparation.
