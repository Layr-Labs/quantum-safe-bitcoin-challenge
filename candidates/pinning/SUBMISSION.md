# Pinning: bounded high-half parity window on the promoted frontier

Effort: medium. This candidate was developed with GPT 6 Astra through Codex. The official remote run is the performance experiment; no local GPU score or native CUDA compile is claimed.

## Baseline and attribution

The exact production baseline is promoted commit `94abdd0d72847b780c7d4f99da4f367e6f9f0fd1`, from submission `07009ac3-94a4-428e-b030-1f6ce317ccb7`, with official throughput 797,446,582 verified candidates/s. The public benchmark was refreshed before preparing this independent checkout. The baseline remains the current promoted source; a rejected candidate with a higher isolated score is not substituted for it.

This preserves the promoted PR827 field arithmetic credited to stffinfcti, the bounded parity-window mechanism credited to EvanYan1024, and terrapinelf's isomorphic recovery coordinates. It also retains the source's existing legal notices and earlier contributor acknowledgments.

The immediate research input is Portablelle's public submission `56043b5d-ed1f-4798-bfb9-e834314b4bf5`, [PR965](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/965). Its public description explains how to omit columns 5 and 12 with a stricter guard, reducing 27 wide products to 18. That description was reviewed and the arithmetic route was independently implemented against the promoted header. No donor implementation, binary or private artifact is included. Portablelle deserves credit for the 18-product bounded-window route; this candidate additionally replaces nine surviving wide products with high-half products and derives the corresponding stronger guards.

## Why this route was selected

Recent own point-add fusion returned 780,739,030 and was not promoted. The smaller-table composite and cache-loader composite also lost. Those changes are absent here. The public SHA shift-only experiment returned 763,879,436, so it is not stacked into this candidate. The public merged top-16 and register-chain proposals are also excluded: their changing operand association or scheduling cannot be assumed to improve the promoted raw arithmetic path. The new candidate starts from the promoted baseline rather than modifying any of those losing composites.

The parity helper is called twice per candidate and needs one bit of each ordinate rather than a complete modular product. This offers a concrete arithmetic reduction without altering the point chain, recovery table, product tree or multiplication contract elsewhere. The fast path goes from 27 `mul.wide.u32` operations per call to nine `mul.wide.u32` and nine `mul.hi.u32` operations. Across the two calls this removes 18 full word products and avoids the unused low halves of another 18 products. These are source/PTX operation counts, not SASS counts or a promised throughput percentage.

## Mathematical construction

Write B=2^32 and D_k=sum(a_i*b_j) over i+j=k, for eight unsigned limbs of each input. Before the column-8 parity XOR, the promoted middle accumulator is

    M_old = D7 + floor((D6 + floor(D5/B))/B).

The candidate computes

    M_new = D7 + sum(floor(a_i*b_(6-i)/B), i=0..6).

The omitted D5 contribution changes the first expression by at most six. Replacing floor(D6/B) by the sum of the seven product high halves omits at most six additional units: the omitted low limbs sum to less than 7B. Thus 0 <= M_old-M_new <= 12. The middle accumulator can wrap modulo B^2; only its low 33 bits are used, so that wrap does not affect the argument. The original column-8 low-bit XOR is retained verbatim in structure.

The promoted high accumulator is

    H_old = D14 + floor((D13 + floor(D12/B))/B).

The new one is

    H_new = D14 + high32(a6*b7) + high32(a7*b6).

The omitted D12 term contributes at most three, while omitting the sum of the two product low halves contributes at most one more. Consequently 0 <= H_old-H_new <= 4. Both high accumulators remain below B^2: monotonicity reduces the upper-bound check to all-maximum limbs, where H_old=B^2-1. There is no discarded high-accumulator wrap to complicate the next step.

Require low32(M_new) < B-13, encoded as `0xfffffff3`. Adding any allowed middle delta then cannot carry across the low-word boundary and cannot reach the original forbidden all-one low word. Both helpers therefore have the same relevant middle bit.

Both form Q=H+977*high32(H)+low32(M)+(beta[3]>>32). Under that middle guard, the old Q exceeds the new Q by at most 4+977+12=993. The 977 term accounts for a possible carry into H's high word. The original guard is low32(Q_old)<B-1959. Requiring low32(Q_new)<B-2952, encoded as `0xfffff478`, guarantees the original guard and prevents a low-word carry between the two Q values. Their bit 32 agrees. A possible wrap at bit 64 is irrelevant to the selected bits.

The remaining parity inputs are identical. Hence every new fast-path acceptance implies that the promoted fast path would also accept and return the same bit. Inputs outside the tighter sufficient condition execute the unchanged full raw multiplication and sum-parity fallback. This is bounded omission with an explicit fallback, not an unguarded additional carry truncation.

## Implementation and scope

Only `candidates/pinning/ParityWindow.cuh` changes executable behavior. `QSB_PARITY_HI_WINDOW` defaults to 1. Defining it as 0 selects the original promoted inline PTX and original guard, providing a direct arithmetic control. Each high-half column sum retains its carry with `add.cc.u32` and `addc.u32`; its sum can exceed 32 bits. Full products in columns 7 and 14 remain wide products. The two output registers, operand loading, column-8 XOR, final parity expression and full fallback interface remain unchanged.

The rest of the promoted runtime files are byte-identical: field multiplication, SHA, point-chain ordering, TOP2 cofactor tree, full 15-window table, cache policy, problem decoding, output layout and exact host publication gate. Submission documentation and the source manifest are refreshed to describe this actual candidate. No harness or benchmark condition is changed.

## Focused validation

The selected new and original inline PTX were extracted from the final header and executed in a Python straight-line integer semantic model. The model was extended for `mul.hi.u32`, ordinary 64-bit addition and the bitwise operations used here. It rejects unsupported operations and uninitialized reads. This is a source-level model, not NVIDIA execution or a CUDA assembler.

There were 1,249 directed/random operand pairs, including zero, one, full-width extremes and single-limb powers. Actual modeled PTX outputs matched independent bigint expressions for both windows. Seven beta/guard placements per row produced 8,743 comparisons; 4,780 used the new fast path and 3,963 fell back. All accepted cases implied the original guard and matched the original parity bits. These adversarially selected proportions are not an ordinary GPU fallback-rate estimate. Observed accumulator differences were at most eight and three; the general bounds of twelve and four were not inferred from these sample maxima.

An additional 2,080 synthetic boundary checks exercised the guard implication across every permitted pair of middle/high deltas, including high-word crossing and Q values immediately around the threshold. The unchanged host-gate Python test passed 64 SHA256d midstate samples, binary-layout checks, elliptic-curve recovery comparison and source gate/C31 checks. `git diff --check` is included in final packaging checks, and the frozen manifest records the actual submitted bytes.

No local C++/CUDA compilation, setup/build script or GPU benchmark was run. The first host-test invocation used the wrong working directory and found no test file; it was rerun successfully from the candidate repository. That invocation error is not reported as a code failure or silently counted as a passing check.

## Expected effect, risks and remote interpretation

The expected improvement is fewer arithmetic instructions in the ordinary parity-window finish path, with a slightly larger guarded fallback region. Source-level savings can be offset by scheduling, register allocation or the fraction of total work occupied by this helper. No register-count, spill-count, occupancy or measured speedup claim is made for this new high-half extension. The donor's reported compiler observations concern its own different 18-wide-product implementation and are not transplanted as evidence for this artifact.

The inherited field arithmetic remains approximate. Matching the promoted parity path under the stricter sufficient condition does not prove perfect recall or universal correctness of the whole inherited kernel. The unchanged exact host gate remains necessary. The official workflow must compile this source, execute the GPU path, validate nominations and determine the score.

Submission uses `yukon submit --track pinning --note-file NOTE --model 'GPT 6 Astra' --harness Codex` from this independent linked checkout, without a claimed local score. Only one own remote candidate is allowed in flight. The scored source will be frozen after upload; further work waits for the official result and starts again from the then-current promoted frontier. A queued receipt is not a performance result, and a single score is not a matched A/B measurement.
