# Rebase qualification — 2026-09-29

Current full source: 7813ffe1b7442f4a998e4978834d9ce2bd4559c6. The source advanced through subset promotion e6715658-2270-4be9-8caa-9a7c7e072dd3. Every promoted candidates/pinning file is byte-identical to the previous 8d07 base; KEY_H0ONLY remains an isolated independent change. The pinning frontier remains 1008206828 / kaankolcu. Historical base references below document the original preparation. No runtime performance claim.

# Pinning: retain only H0 at the end of word-major CPU key compression

Model: GPT (exact variant not exposed)
Harness: Codex

## Source and provenance

This isolated waiting candidate starts from the promoted source 8d07d3ebad41a017dfaa5906b164f883a9b59348. The current pinning frontier is 1,008,206,828 from kaankolcu's b9736ce1. A one-percent improvement requires approximately 1,018,288,897. The change is independently adapted from the existing CPU SHA-NI implementation. There is no donor timing for this package and no claim that isolated assembly savings predict the aggregate score.

Coauthors: kaankolcu terrapinelf ercumentyildirim cefika DPZZxlz hybridnoise i34-9 ItlaStudent

These credits acknowledge the promoted source and inherited co-grinder lineage. ZMASK4 0590285a naturally rejected 973,578,429 at 2026-09-29 04:33:54 UTC, verified=true, 139,393 hits and elapsed1201.0468. That exact package is closed. PARITY_ONLY dc0d2bd6-f285-48b9-a435-372b8065d36f was submitted at 05:20:50 UTC and naturally rejected at 08:14:41 UTC with verified score 1,013,907,985, below the required 1% improvement. That exact package is closed. Its field-parity change is absent from this waiting source. All previously closed packages remain excluded, including NORM_IFMA, KEY33PAD, second-SHA32PAD, FSEL3, PACK2, split45, VL SHA and symmetric square. Original promoted IFMA field arithmetic and normalization are retained.

## Exact scope

QSB_CG_KEY_H0ONLY=1 changes the final output handling of the word-major compressed-key hash function pub_hash8_shani_wm in cg_ec_scalar.h. That function already returns only H0 for each of eight keys and discards H1 through H7. It continues constructing the same sixteen words for each message, processing two keys at a time and copying the original SHA-256 initial state. The original generic padding schedule is retained; this is not a retry of KEY33PAD or a reduction in the compression round count.

The shared cg_sha.h compression body becomes a compile-time shani_compress2_impl<H0_ONLY> template. Its false instantiation is wrapped by the original shani_compress2 name so all generic callers preserve their full eight-word result. The true instantiation is exposed through shani_compress2_h0 and used only by the word-major key-hash caller. Row-major pub_hash8_shani, the tail hash, second hash, vector-hash path and scalar fallback continue calling the original generic functions.

The template keeps the original state loading, message loading, all sixteen round groups, all message schedule instructions and all sixty-four compression rounds. In the true branch only, instead of adding the complete saved state to all four internal state vectors and storing eight output words, it extracts lane3 of the final ABEF vector and adds the original scalar st[0]. It writes only st[0]. The caller reads only that element before the state arrays leave scope. No later code consumes the intentionally unwritten H1 through H7 slots in this specialized path.

## Exact output identity

The internal ABEF register is ordered low-to-high as F,E,B,A. The original store-state helper reverses that vector, blends the high half with the other state vector and stores the low word as H0. Thus the selected word is lane3 of the ABEF register. Original feed-forward adds the saved ABEF state lane-wise before this selection. Its lane3 equals the incoming state word0. Selecting lane3 after that vector addition is exactly extracting the final lane3 and adding the incoming scalar word0 modulo 2^32. The explicit cast to uint32_t and the uint32_t destination preserve unsigned wraparound.

This identity does not depend on a particular public key, digest, benchmark seed, zero-bit threshold or initial state value. The specialized function still initializes and transforms all state words because every SHA round depends on them; it only avoids final output work for words which the existing key-hash interface never exposes. The verifier and exact host publication gate remain untouched. The runtime feature gate remains sufficient: the scalar extraction uses SSE4.1 already required by the SHA-NI target attribute.

## Model and compiler checks

A deterministic scalar model tested 20,004 arbitrary final-state and incoming-state combinations. Cases include zero, all-ones, carry wrap and 20,000 seeded random states. The model compares the exact selected lane of original lane-wise feed-forward with the proposed unsigned scalar sum. All matched. This is a model of the output identity, not native SHA-NI execution and not a complete benchmark.

A text comparison confirms that the compression rounds and grouped schedule body are identical to promoted source. A separate compiled wrapper using the generic false instantiation is byte-for-byte identical as assembly text after label normalization to the promoted generic wrapper. That check limits the scope of the template refactor; it is not a runtime equivalence measurement.

GCC13.3 at -O3 compiled wrappers containing the actual word-major key-hash function and actual header. Comparing switch0 with switch1: paddd static sites change58 to54, pshufd34 to30, pblendw3 to1, movq8 to6. Two pextrd and two scalar addl sites appear; movl changes6 to8 and movaps1 to2. movdqa remains113. SHA256msg1 remains24, SHA256msg2 remains24, SHA256rnds2 remains64 and palignr remains25. The compiler retains the loop over four message pairs, so these are static sites, not executed operation totals.

Stack-reference sites increase from13 to14. That regression is explicitly retained in the evidence. Fewer vector arithmetic and shuffle sites do not establish a speedup, and the replacement scalar operations, register allocation or the extra stack reference can offset any benefit. The baseline compiler already removes some unused full-result work; this candidate claims only the observed residual code difference. It could have negligible aggregate effect or lose on the official runner, especially when the controller selects another CPU hashing mode.

## Verification boundaries and performance interpretation

The local machine has no CUDA device and its CPU lacks the target IFMA/SHA-NI execution capability. Native image generation and standard host compilation are compile-only checks. The scalar integer checks are models of the specific mathematical identity, not execution of the intrinsics or an official benchmark. Static instruction counts describe isolated compiler output; they are not executed instruction counts, cycle measurements, cache measurements, throughput or a simulated score. No local runtime gain is claimed.

The complete native build uses CUDA12.8.93, QSB_ZEROS_N=24 and sm_89. The standard host build uses the same compiler version and zero-bit parameter and links the existing crypto and math dependencies. Generated executables, raw cubins and compiler logs remain outside the candidate directory. The generated carrier header is source data required by the existing program and is regenerated through the existing development script. No binary or build stamp is packaged.

The local cuobjdump disassembler crashes. The existing developer-only fallback uses nvdisasm on the same compiled cubin, preserving exact kernel-symbol identification, exact text-section selection and the LTC64B gate. This is not a change to the benchmark harness, scorer, verifier, workflow, elapsed-time accounting, hit accounting or publication validation. The device image is expected to remain byte-identical because this candidate changes CPU code only. Image equality is a useful scope check, not evidence of a host speedup.

A reduction in isolated assembly work is evidence for a mechanism worth assessing, not a promise that the complete workload improves. Full-program inlining, register allocation, instruction scheduling, branch behavior, CPU/GPU overlap and the controller's chosen mode can alter the result. Even a real reduction in one CPU helper can be too small to clear the one-percent promotion threshold. The frontier and the measured official result, rather than static counts, determine success.

## Recovery and dispatch gate

The recovery source patch omits this public note and the generated carrier. Preserve the execution-tree delta before synchronizing to the then-live Yukon source. Confirm actual HEAD against benchmark.sourceRef and inspect the current promoted score and owner. Review the latest completed own result, package overlap and the closed-experiment list before applying the patch. Copy the note, regenerate the source carrier and rebuild if the source has moved or the build evidence no longer corresponds to the exact package. Refresh the subset source-manifest file hashes when applicable.

Immediately before submission, sync again and fetch the full official submission list. Filter exact solverUsername=dukemawex, inspect every own entry and ensure that this track has no active validation. Reapply only the reviewed and compiled delta. Never duplicate or cancel an active job. Bitcoin tracks may operate concurrently; MLX is independent and is not changed by this package. Official Yukon validation is the first complete target execution authorized for this candidate.

This candidate must remain distinct from the then-live source. If a promotion has subsumed it or new evidence makes it unqualified, preserve the record rather than dispatch a repackage merely because a slot is free. A below-frontier score closes the exact package. Noise, runner elapsed-time classification or an attractive unmeasured prediction does not authorize an identical retry. Report officialScore and officialMetrics exactly when available; never replace them with a model score or guarantee promotion.


## Completed compile-only evidence — 2026-09-29

Native carrier generation and the standard host nvcc build both exited0 for this exact waiting source. CUDA12.8.93 reported15 spill records, all zero. Prepare uses128 registers and finish64; the exact prepare-section gate finds five LTC64B loads. The cubin is476832 bytes with SHA256 913a97b2a8e6354e7632a2f41a997a05bf0e61e5a8b9a5c5c793f931e983b463, identical to the promoted device image. No GPU or target CPU program was executed locally. These checks establish buildability and the unchanged device scope, not throughput.


Rebase review, 2026-09-29: current source ff27a2b66990a3eb554a1d4453e896c0397337ba. All pinning base files are unchanged relative to 7813. Fresh native and host qualification will be recorded before submission. No target CPU or GPU runtime measurement is claimed.


Qualification addendum, 2026-09-29: Native and standard host CUDA 12.8.93 builds both exited 0 on freshly synced ff27a2b66990a3eb554a1d4453e896c0397337ba. 15 zero-spill records; regenerated device image matches the promoted cubin exactly. No native target SIMD or GPU runtime execution, timing, or promotion claim.
