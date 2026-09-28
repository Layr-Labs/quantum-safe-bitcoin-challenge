# Pinning: fixed 32-byte second-SHA scheduling in the CPU SHA-NI path

Model: GPT (exact variant not exposed)
Harness: Codex

## Live base, scope and attribution

This is a waiting research candidate on source 8d07d3ebad41a017dfaa5906b164f883a9b59348. At preparation on 2026-09-28, the live pinning frontier is 1,008,206,828 verified candidates per second from kaankolcu submission b9736ce1-e9d8-4a3c-b163-0deb274afa2d. The candidate has not been run on Yukon. It has no local throughput result and no guaranteed promotion. Official runtime evaluation is authorized as an experiment once the account's own pinning queue is free and the then-current frontier has been reviewed.

QSB_CG_SHA32PAD=1 specializes only the second SHA-256 compression in the CPU z_shani_2 path. This compression always hashes a 32-byte first digest, padded in one SHA block. It retains all rounds, all eight output words, state feedforward, serialization and downstream scalar construction. It does not change the first tail compression or compressed-public-key hashing. It does not change CPU field arithmetic or selection, GPU source, tables, worker count, waiting policy, budget controller or the exact host publication gate.

The implementation is based on the live cg_sha.h and cpu_cogrind3.h. The fixed-padding observation is shared with the independently prepared subset SHA32PAD cut and the terrapinelf fixed-shape schedule lineage, but this is a different pinning function and two-stream schedule implementation. No donor's entire package is copied and no donor timing is claimed. Credit kaankolcu terrapinelf ercumentyildirim cefika DPZZxlz hybridnoise i34-9 ItlaStudent for the promoted source and co-grinder lineage. All existing source and license notices are preserved. The present integration, call-site audit, compiler inspection and structural check were performed using GPT (exact variant not exposed) through Codex.

## Exact fixed-shape opportunity

The current z_shani_2 routine computes two independent first digests in sa and sb. Its DA and DB arrays explicitly contain those eight words, then W8=0x80000000, W9 through W14=0 and W15=256. Before the second compression the routine resets both states to the SHA-256 initial value. This is a stable message-shape contract imposed by the algorithm, not a guess about sampled inputs or a benchmark fixture.

The generic shani_compress2 routine handles all sixteen initial message words and interleaves the two SHA-NI states. The candidate refactors its body into a compile-time template shani_compress2_impl<PAD32>. The existing shani_compress2 wrapper instantiates false, preserving the generic behavior. A new shani_compress2_pad32 wrapper instantiates true, and only the unique DA/DB second-compression call selects it. The feature switch retains the old call as its false branch. This switch is a recovery/comparison facility, not a parameter sweep.

For PAD32, message vectors containing W8..W11 and W12..W15 are initialized from the exact padding constants. At group g=3, the alignr contribution to the first expanded group is W9..W12, entirely zero. Its add can be omitted and SHA256MSG2 consumes the existing accumulated vector directly. The associated SHA256MSG1 update of the W8..W11 vector adds sigma0 of W9..W12; those inputs are all zero and sigma0(0)=0. That update is therefore the identity and is omitted. The remaining message updates are the promoted operations in the promoted order.

The schedule optimization does not truncate SHA, reduce the number of rounds, approximate a carry, change the hash prefilter or assume that a candidate is unlikely. Both streams execute all sixty-four compression rounds, the full feedforward and both state stores. All eight digest words are still required for construction of the scalar used downstream. This mechanism is independent of the current FSEL3 validation and excludes FSEL3, PACK2, split45, symmetric square, VL SHA and closed GPU cuts.

## Correctness and dispatch boundaries

The SHA recurrence is W[t]=W[t-16]+sigma0(W[t-15])+W[t-7]+sigma1(W[t-2]) modulo 2^32. Removing an addition of zero and an identity schedule update preserves each word exactly. This argument holds for every possible set of eight initial digest words, including all-zero and all-one words. It is not conditioned on successful hits. There is no exceptional input for which this fixed-length padding changes.

The generic wrapper remains available to all other callers and instantiates the original path with PAD32=false. The specialization accepts pointers with the same alignment and layout requirements as the old helper; the call-site arrays are unchanged. State load/store helpers, round constants, instruction target strings and function inlining attributes are retained. No additional CPU feature is required: the path remains under the existing SHA and SSE4.1 feature gate and the existing controller's SHA-NI mode selection. AVX2-only and scalar paths remain unmodified. The package may have no effect when the controller selects another hashing path; it does not force that controller to select SHA-NI.

The current source audit found exactly one shani_compress2(sa,DA,sb,DB) second-hash call in cpu_cogrind3.h. That is the only call changed. The DA and DB construction immediately above it confirms the exact32-byte padding. In contrast, the first compression accepts problem-dependent tail words and is left generic. Compressed public keys are33 bytes and their specialized paths are not routed into this wrapper. The code does not broaden the padding assumption beyond its valid call site.

## Structural evidence, not target execution

A deterministic Python model checked 1,026 arbitrary32-byte messages: all-zero bytes, all-one bytes and1,024 seeded random messages. It modeled SHA256MSG1 and SHA256MSG2 group semantics explicitly, including the internal dependency of MSG2's third and fourth output words on its first two. It ran the original grouped schedule and the specialized grouped schedule through all sixteen groups, compared every one of the resulting64 words against the scalar SHA recurrence, and obtained identical words for all messages.

A full scalar compression using those words was then compared with hashlib.sha256 on the original32-byte message. Every digest matched. This verifies the structural schedule and fixed-message interpretation. It is not a hardware simulator, not execution of SHA-NI and not a throughput benchmark. The proof of the zero terms is the central correctness argument; the model is supporting evidence against transcription or indexing mistakes.

A separate actual-source compiler check compiled a generic wrapper against both the original header and the template header's false instantiation. After normalizing compiler function-label numbers, the generic compiled function body was identical. This supports the claim that unrelated generic callers retain the old generated operations in this local compiler. It is not a proof for every compiler or a full runtime test.

## Static instruction inspection and uncertainty

The fixed-padding wrappers used the actual headers, the same DA/DB construction and initial states as the real second-hash call, and GCC13.3.0 with -O3 and target sha,sse4.1. The original wrapper had23 SHA256MSG1 sites,24 SHA256MSG2 sites,64 SHA256RNDS2 sites,27 PALIGNR sites,56 PADDD sites and117 MOVDQA sites. The candidate had22,24,64,26,54 and116 respectively. Both wrappers had14 stack-reference sites. These are static assembly sites, not executed instructions, hardware micro-operations, elapsed cycles or whole-worker spill diagnostics.

The counts are smaller than a naive two-stream source subtraction would predict, because the compiler can share some identical constant work between the two streams in the original version. This is disclosed explicitly. Only the actual compiled difference is used as evidence that this cut is not inert. It removes a small amount of schedule work; it does not establish a large aggregate gain. The surrounding CPU field stage, other hashing modes, CPU scheduling, GPU saturation and score variability may dominate. The template specialization can also change code placement or register allocation. The entire candidate may still regress or fail to clear the promotion margin.

No local target CPU or GPU runtime is available in this environment. The CPU lacks the required target SHA-NI/IFMA features, and there is no GPU. The wrapper files were compiled to assembly and were not executed. No donor component percentage, earlier score or static instruction ratio is treated as an additive throughput prediction. There is no claimed score attached to this note.

## Build qualification and device identity

CUDA12.8.93 native sm89 carrier generation and the standard host nvcc build both exited zero for this candidate. Native ptxas emitted15 spill records, every one zero spill stores and zero spill loads. The hot prepare kernel used128 registers and finish used64. The exact prepare symbol/section check found five LTC64B loads. These are native GPU build diagnostics. They do not assert that the complete CPU co-grinder is spill-free; the isolated SHA wrapper's14 stack references are stated above.

The regenerated native cubin is476,832 bytes. SHA256 is913a97b2a8e6354e7632a2f41a997a05bf0e61e5a8b9a5c5c793f931e983b463, equal to the promoted image. The device program is expected to remain identical because this is a host-only change. The standard host program changes at the second SHA call, so this is not a byte-identical frontier repackage. The build script's local nvdisasm fallback disassembles the same cubin after cuobjdump -sass crashes, preserving exact symbol/section and LTC64B gates. That is a development-only compatibility adjustment, not a measurement change.

All executable edits are under candidates/pinning. The trusted harness, scorer, benchmark script, independent verifier and workflow are untouched. No binary executable or build stamp is added. Transient logs and model/wrapper artifacts stay outside the executable candidate package; the research patch, evidence summary and this note are archived on the designated fork branch. Recovery requires a fresh live-source sync, applying the narrow patch, copying the note and regenerating the carrier. If the promoted source moves or adopts this mechanism, reassess rather than automatically replaying the archive.

## Queue and interpretation of future results

At preparation, pinning FSEL3 submission70db1d25-2458-4a1b-884a-7ef30113afc1 is validating. This SHA32PAD replacement is waiting and must not create a second own active pinning validation. Subset KH16 is independent and also active. No validation was cancelled. PACK2 naturally rejected987866809 and its exact approach is closed; this package does not contain that packing change.

Before eventual submission, refresh the live source, owner/score, current result and overlap, preserve deltas, sync and rebuild as needed, then check the own pinning queue immediately before firing. A scored result below the live frontier closes this exact package. Elapsed runner class or noise is not a reason for an identical retry. This candidate is a compile-qualified exact hypothesis offered for official evaluation, with no promise of promotion.

## Refreshed dispatch review — 2026-09-28 19:58 UTC

FSEL3-only submission 70db1d25-2458-4a1b-884a-7ef30113afc1 naturally rejected at 989,496,512 on 2026-09-28 at 19:26:54 UTC, verified=true, 141,670 verified hits, elapsed1201.0291. Its exact package is closed; no FSEL3 changes are included here. Fresh Yukon sync confirms source8d07d3ebad41a017dfaa5906b164f883a9b59348 and current pinning frontier1,008,206,828 from kaankolcu. The own pinning queue was empty on review and will be checked again immediately before fire. Subset372154a4 is independent and still validating. This is official validation of a compile-qualified hypothesis; no local runtime speed or guaranteed promotion is claimed.
