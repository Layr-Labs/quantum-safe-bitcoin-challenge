# Pinning: exact whole-lane selection through native ternary logic

Model: GPT (exact variant not exposed)
Harness: Codex

## Candidate scope and live starting point

This is a prepared research candidate for the pinning track, based on promoted repository source 8d07d3ebad41a017dfaa5906b164f883a9b59348. The live pinning frontier at preparation on 2026-09-28 is 1,008,206,828 verified candidates per second, owned by kaankolcu through submission b9736ce1-e9d8-4a3c-b163-0deb274afa2d. This note does not claim that the candidate beats that score or its promotion margin. The user authorizes compile-qualified, evidence-backed experiments evaluated by Yukon. There is no local throughput measurement for this change.

The sole executable mechanism is QSB_CG_FSEL3=1 in candidates/pinning/cpu_cogrind3_ifma.h. It changes the five-limb selection helper fsel in the existing four-lane AVX512 IFMA CPU co-grinder. The promoted operation uses byte-wise variable blending. The candidate uses the AVX512F/VL 256-bit ternary-logic operation with immediate 0xca. The change depends on the existing helper contract that each 64-bit mask lane is either all zero bits or all one bits. All call sites meet that contract, as described below. A compile-time fallback retains the exact promoted helper body under QSB_CG_FSEL3=0; the prepared default is one, with no sweep of values.

No field multiplication, square, normalization, reduction, digit generation, table geometry, prefetch distance, inversion chain, CPU worker count, work allocation, timing controller or publication policy changes. GPU kernels, the prepared state layout and GPU arithmetic remain promoted. In particular this package does not contain active PACK2, rejected split45, rejected symmetric square, rejected VL SHA, or any of the previously closed GPU carry or root experiments. It preserves all code inherited in the live promoted tree without reopening those investigations.

## Mechanism and specific reason to investigate

The old helper applies _mm256_blendv_epi8 to each of the five 256-bit limbs. Its mask semantics are per-byte high-bit selection. For the actual all-zero/all-one lane masks, this is exactly equivalent to the bitwise expression (mask & a) | (~mask & b). Native ternary logic expresses that boolean operation directly. The immediate 0xca encodes output a when mask is one, otherwise b, with argument order mask, a, b. It is not a numeric approximation or a changed exception policy.

The existing sign-selection operation in bk_pre calls fsel for every processed table stage: after gathering yT, it computes ny using unchanged fneg, selects ny or yT from the digit sign, then subtracts py and weakly normalizes. The candidate therefore has a concrete repeated use even when exceptional zero-digit lanes are uncommon. Other fsel calls restore unchanged points or replace zero denominators with one using zero-digit masks. Those semantics remain exact. The motivation is to expose the all-bit mask contract directly to the compiler and replace variable byte blends with native boolean selection, potentially changing instruction scheduling or execution-resource demand in this CPU component.

This motivation is intentionally narrower than a speed claim. The isolated assembly comparison has more total static vector instruction sites in the candidate, because extra moves appear. It does not prove a net improvement. Register allocation, move elimination, code inlining, hardware scheduling and the contribution of this CPU path to the mixed CPU/GPU score all matter. Yukon is needed to decide whether any reduction in boolean-selection work outweighs those costs. This candidate is not being submitted simply because a queue is free, and it is not a replay of a scored package.

## Correctness contract and call-site audit

The helper comment on the promoted source explicitly specifies an all-ones or zero mask per lane. zmask4 constructs each lane with a conditional expression selecting ~0ULL or zero based on QCG_ZERO. negmask4 loads four signed digit words, performs a 32-bit arithmetic shift right by 31 and sign-extends each result to 64 bits. Its only possible lane values are therefore zero and all ones. The selected negative digit representation does not alter that argument: the sign bit alone determines which complete mask is produced.

Every current fsel call in this header uses either zmask4 or negmask4. In bk_pre the mask sm comes directly from negmask4. The zero-digit path for c.dx, the x3 and y3 restoration paths, the next-stage c.dxn path, and the forward denominator path use zmask4. None provides a fractional-bit or byte-wise mixed mask. This matters because a byte blend with arbitrary masks would not equal the bitwise expression; that broader contract is neither assumed nor claimed. Future call sites introducing partial masks must revisit this specialization.

For valid masks, each bit in every one of the five limbs is selected from exactly the same input as before. There is no arithmetic and therefore no new carry, overflow, field-range or normalization requirement. The loop retains limb order and uses the original loads/stores at the same positions. Aliasing of r with a or b remains valid: each limb's inputs are read before that limb is stored, and no limb refers to another limb. Exact pointer aliases used in existing call sites are preserved. This does not introduce a guarantee for arbitrary overlapping misaligned objects outside the original helper's contract.

The IFMA runtime dispatch already requires AVX512F, AVX512VL and AVX512IFMA with the operating-system vector-state support checked through the compiler's CPU feature mechanism. The helper retains its existing target string and inlining annotation. The intrinsic used here requires F/VL, already covered by that gate. No new dispatch branch, environment override or benchmark-hardware detection is introduced. The AVX2-only fallback and non-IFMA routines remain untouched.

## Structural checks and their limits

A deterministic Python structural model checked the eight possible single-bit triples against the ternary immediate with truth-table index (mask << 2) | (a << 1) | b. All eight outputs matched selection. A second model checked 10,032 four-lane cases using all 16 combinations of lane masks, with boundary operands and seeded random 64-bit limbs. The old operation was represented explicitly as per-byte high-bit selection; the new operation was represented as bitwise selection. Every lane matched. The model also examined boundary and seeded digit sign cases for the sign-mask invariant.

These are mathematical and representation checks, not execution of AVX512 instructions and not a hardware simulation. They do not supply timing data or validate the complete worker at runtime. The all-bit-mask proof is the main exactness argument. The model is a supporting implementation check. The normal final host publication gate and the independent organizer verifier are unchanged; neither is used as an excuse to intentionally lose exceptional candidates in this package.

## Actual-source compiler inspection

Isolated wrappers used the exact old helper loop and exact new helper loop with the same vfe/V definitions and the existing avx2,avx512f,avx512vl,avx512ifma target. GCC 13.3.0 at -O3 compiled both successfully to assembly. The promoted wrapper contained five vpblendvb sites, one vpcmpgtb, one vpxor and eleven vmovdqa sites. The candidate contained five vpternlogq sites, zero vpblendvb, zero vpcmpgtb, zero vpxor and fourteen vmovdqa sites. Total static vector sites were 18 versus 19. Both isolated wrappers had zero stack-reference sites.

The extra three moves are disclosed rather than hidden behind a count of only the favorable operations. These are static source-wrapper counts, not dynamic instruction counts, micro-operations, hardware latency, full-worker register pressure or measured performance. Inlining into the complete co-grinder may give different allocation. This evidence confirms that the intended operation changes and that the isolated expression is not compiled back into an identical instruction sequence. It cannot establish the ranked benefit of the package.

## Build evidence and unchanged GPU image

CUDA 12.8.93 native sm89 carrier generation and the standard host nvcc build both exited zero in the preparation environment. Native ptxas produced 15 spill diagnostic records; each reported zero spill stores and zero spill loads. The hot prepare kernel used 128 registers and finish used 64. Exact kernel-section checking found five prepare LTC64B loads. These are GPU compilation diagnostics, not a claim that the complete CPU co-grinder has no spills.

The regenerated native device image is 476,832 bytes with SHA256 913a97b2a8e6354e7632a2f41a997a05bf0e61e5a8b9a5c5c793f931e983b463, equal to the promoted device image. This is expected for a CPU-only change. It is not a frontier repackage: the CPU selection helper in the standard host executable changes, while the GPU program does not. Build logs, isolated wrappers and the structural evidence are kept outside the executable candidate package or in the fork's research archive.

The development carrier script includes the previously reviewed nvdisasm fallback for the local cuobjdump -sass crash. It disassembles the same cubin and retains the exact symbol/section and LTC64B gates. No trusted benchmark or measurement code is modified. The standard host command remains nvcc -O3 -DQSB_ZEROS_N=24 with the candidate pinning.cu and the original crypto/math link libraries. No binary executable or build stamp is packaged as candidate source.

## Runtime uncertainty, lineage and dispatch

This environment has no GPU and its CPU lacks the target IFMA/SHA-NI capabilities. The complete candidate has not been run locally. No local CPU throughput, GPU throughput, official score, percentage gain or promotion probability is asserted. The most material risks are extra moves after ternary selection, different allocation when inlined, negligible CPU contribution, and hardware-specific scheduling. The code may regress. Official validation must finish naturally; a scored result below the current frontier closes this exact package rather than inviting an identical retry due to noise.

The base and its existing notices are retained. Credits: kaankolcu terrapinelf ercumentyildirim cefika DPZZxlz hybridnoise i34-9 ItlaStudent. They credit the live pinning, CPU co-grinder and inherited implementation lineage; this note does not claim that those contributors measured this isolated ternary helper. The present integration and structural/compiler review were performed with GPT (exact variant not exposed) through Codex.

At original preparation, PACK2 was validating. It naturally rejected at 987866809 on 2026-09-28 16:39:52 UTC and its exact approach is now closed. The own pinning queue is empty at this dispatch review. This package excludes PACK2. A fresh queue check will be made immediately before submission. Before any eventual submission, re-query the own pinning queue, confirm then-live promoted source and owner/score, inspect result and overlap, synchronize, restore only this isolated cut, and rebuild if the source changed. The recovery patch omits the public note and generated carrier; restore this note and regenerate the carrier when recovering it. The current package is a waiting hypothesis, not a completed Yukon validation.



## Dispatch review — 2026-09-28

Fresh official query and sync confirmed 8d07d3ebad41a017dfaa5906b164f883a9b59348, score1008206828 from kaankolcu b9736ce1. No live source movement or promoted overlap was found. The candidate is rebuilt for this dispatch. Subset KH16 remains independently validating. No measured performance or guaranteed promotion is claimed.
