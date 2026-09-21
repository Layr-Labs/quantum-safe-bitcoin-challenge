# Pinning: deferred point-add fusion with retained even-fold carries

Effort: medium. Prepared with GPT 6 Astra in Codex. No local CUDA/C++ compilation and no GPU benchmark were performed. The official remote build and ranked run are the first native evaluation of this exact integration. No speedup or claimed score is asserted.

## Base and decision

This candidate starts from the latest promoted Pinning source, submission 07009ac3-94a4-428e-b030-1f6ce317ccb7 by terrapinelf, promoted commit 94abdd0d72847b780c7d4f99da4f367e6f9f0fd1, official score 797,446,582 verified candidates/s. It retains the 64 MiB, 15-window signed table, decoder layout and volatile accesses, isomorphic recovery xR=plus/minus1, field header, bounded 27-product parity, cofactor tree, pipeline, SHA and OpenSSL host publication gate.

Our preceding cache/schedule integration, 40c5c4de, passed verification but scored 751,799,654, with 107,622 verified hits in 1200.8502 seconds. That is approximately 5.724% below the current promoted frontier. The public table-CG donor d2b1ca8 separately scored 762,928,439 and the schedule donor 8aa35d5 scored 777,631,992. These separate runs do not isolate each component under identical conditions, but they provide no basis to carry the losing bundle forward. This archive starts fresh: it does not contain table .cg loads, relaxed digit reads, cofactor predicate deletion, factor masking, our former 32 MiB table, or our former 25-product parity.

The new mechanism is a single PTX register scope for each deferred mixed point addition. It targets all thirteen additions in the serial scalar chain. The total number of mathematical field multiplications is not reduced. The potential gain is reduced intermediate materialization, carry-register capture, register lifetime and instruction scheduling cost over the entire serial chain. The official run will determine whether this pays after the compatibility repairs below.

## Selected public mechanism and attribution

The fused structure is derived from Saviour1001's public Pinning submission 7c159de9-d226-465d-b31a-ff98dd32492a, PR930, immutable source 66912b5a49fe853cc9b991f47e1f9c5408554887:
https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/930

The public description was screened first; only then was the selected Pinning implementation fetched. No other track's implementation, author task, private artifact, binary or diagnostic archive was inspected. The donor note describes earlier provenance; retained notices and COPYING continue to apply. Attribution is written in this note and the new header. No additional coauthor metadata is requested by the account owner.

The donor reports a one-body native sm89 version using 116 registers without spills, while its duplicated rotated-body variant spills. Those are donor observations, not measurements of this archive. We retain the promoted single-body loop and its original decoder rather than import the donor's complete decoder/seed-register/paired-layout stack. Thus neither the donor register count nor its modeled throughput gain can be transferred to this candidate as a measured result.

## Implementation and compatibility repairs

Only pinning.cu and a new point_add_integrated.cuh implement runtime changes. GPUMath.h, all recovery headers, the parity window, SHA headers and the cofactor tree remain byte-identical to the promoted frontier.

The promoted scalar loop still decodes the same digits, loads the same table coordinates, seeds with _PointAddXYZZ_mm, visits the same thirteen remaining chunks, and resolves the deferred ordinate once at the end. Each _PointAddXYZZT<true> invocation is replaced by qsb_filter_point_add<true>. Its output includes the next affine ordinate anchor, so the caller's four-word anchor copy is folded into the same scope. No point or digit is prefetched, and no extra table point is held across iterations.

The selected header is intentionally not copied wholesale. Unused donor filtering helpers and seed routines are excluded. The one-body active configuration is extracted into a dedicated header. On device builds the fused arm requires the promoted YOFF, C31, short-carry and fused-square configuration. Other compile-time configurations and the nondeferred arm use the original promoted point-add helper, with anchor update after consumption when required.

First, six lean sites in the donor's active body discarded an even-fold carry that the promoted field primitive still retains. These include five multiplication sites and the first square. They are repaired. Where the odd fold formerly interrupted that carry, the independent odd fold is moved ahead of the even fold; the final even add writes CC and the next addc consumes it directly into z8. This keeps the desired carry scheduling without treating the carry as zero. The two other multiply sites already retained that carry. Both inputs and the raw output representative of all seven multiply bodies are preserved relative to the promoted raw-multiply contract.

A simple boundary illustrates why this repair matters. Let B=2^256 and K=2^32+977. For a=B-1 and b=2, the product's low half is B-2 and high half is 1. The even fold overflows B. Retaining its contribution is necessary for the second-fold result; omitting it changes the raw value by K. A probabilistic rarity argument is not used to delete this carry.

Second, the donor's separate second-square and signed X3 correction are replaced with the promoted _ModSqrAddSub2 PTX expression itself, remapping only the input/output bindings to the surrounding point registers. This retains the promoted R^2+PPP-2Q fused reduction, its compile-time macro policy and its existing rare-boundary behavior. It avoids introducing another reduction boundary in the critical X3 formula. No changes are made to the promoted field header to achieve this insertion.

Third, the fused subtraction sites use the donor's two-limb correction arm. The current promoted C31 helper uses only one corrected limb; this integration retains one additional borrow propagation limb rather than introducing a shorter tail. The broader inherited approximate arithmetic remains present. These strengthened correction cases mean that this entire point helper is not claimed universally byte-identical to the promoted C31 helper on rare boundary inputs. The seven raw multiplications and first square have the narrower exact-representative check described below.

The first square uses carry-ordered doubling of its cross terms instead of fifteen funnel shifts. For T=sum_(i<j) ai*aj*2^(32(i+j)), the low word is zero and 2T plus the diagonal terms equals a^2. Full word-to-word carry is retained. This moves work between instruction classes rather than removing a square. The second square is the promoted fused body, not the previous exploratory all-square rewrite.

## Necessary checks

A Python semantic model executes the extracted integer PTX from seven active multiply sites and the first square. Each site passes 1,100 directed and random cases, 8,800 site evaluations in total. Directed inputs include zero, one, two, K-1, p-1, p, B-2, B-1 and 128-bit boundaries. The reference is the inherited raw short-carry multiplication schedule, including its documented reduction limitations, not merely equality modulo p. Each repaired carry reaches its consumer without an intervening CC-writing instruction. This is a source/PTX model check, not execution on a CUDA device.

The preceding independent square identity check covered 10,515 actual-square inputs and 10,000 arbitrary 512-bit words with zero low word. It established equality of the shift and carry-doubling steps; it did not establish a GPU performance advantage. The full promoted fused-square source expression is reused for X3 instead of replacing it with a newly derived modular identity.

The unchanged Python host-gate test passes: 64 SHA256d midstate continuations, problem binary layout, both public-key recovery outputs against the verifier, and source checks for the exact publication gate. An initial attempt to run that test from a standalone research folder lacked the harness import path; it was rerun successfully from this complete independent benchmark checkout. No native compilation was substituted for the user's remote-validation workflow.

A source inventory records every submitted file. The production diff has no changes to benchmark inputs, protected harness, verifier, difficulty, candidate enumeration, time limit, allocation geometry or host publication behavior. The cache/schedule submission remains frozen separately and is not the source of this archive.

## Performance limits and next decision

Seven multiplies and two squares per deferred addition remain. All candidate lanes execute the same fused arithmetic, so there is no sparse-lane speedup claim. Table traffic stays 960 requested bytes per candidate. Shared digit storage and cofactor traffic stay unchanged. The new register scope may improve allocation and remove transfers, or it may constrain scheduling and cause spills; remote compilation and the ranked measurement decide. Restoring the donor's missing carries and using the promoted fused square also changes the instruction balance relative to the donor census.

The exact OpenSSL gate checks every nomination before publication. The inherited approximate GPU arithmetic can still miss real nominations; successful host verification does not prove universal recall. This candidate adds no deliberate probability-based carry omission. No unchanged rerun, inert marker or cosmetic source change is presented as an optimization.

If this integration fails to build, that is recorded separately from a scored regression and fixed using the remote compiler evidence. If it verifies but regresses, the combined fusion is set aside instead of being stacked onto the next candidate. Any successor starts from the then-current promoted frontier and uses the official result and current public descriptions to select compatible mechanisms.
