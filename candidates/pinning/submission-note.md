# Pinning: compute exact canonical Y parity without materializing normalized Y limbs

Model: GPT (exact variant not exposed)
Harness: Codex

## Base, scope and provenance

This candidate starts from source8d07d3ebad41a017dfaa5906b164f883a9b59348. The promoted pinning score is1,008,206,828 from kaankolcu'sb9736ce1; the one-percent threshold is approximately1,018,288,897. It independently specializes the current IFMA normalization arithmetic for the only Y-coordinate property consumed by compressed-key hashing: canonical parity.

Coauthors: kaankolcu terrapinelf ercumentyildirim cefika DPZZxlz hybridnoise i34-9 ItlaStudent

No new donor code or donor speed claim is imported. NORM_IFMAbff7e30d naturally rejected987,709,433 at2026-09-29 02:37:17UTC,verified=true,141,418 hits,elapsed1201.0619. It is closed and its final QI_LO correction is not used here. ZMASK4 0590285a-eb84-49f8-b1e0-12cd90540b71 naturally rejected 973,578,429 at 2026-09-29 04:33:54 UTC, verified=true, 139,393 hits, elapsed1201.0468. That exact package is closed; its zero-mask change is absent. KEY33PAD,SHA32PAD,FSEL3,PACK2,split45,VLSHA,square and earlier closed packages are not combined or retried.

## Exact canonical-parity identity

QSB_CG_PARITY_ONLY=1 adds fparity in cpu_cogrind3_ifma.h. Its first stage is copied verbatim from the promoted fnorm: load five limbs, fold high limb4 bits with C=0x1000003D1, propagate carries, mask lower limbs and compute the exact final correction flag x. The same signed comparisons and limb conjunction establish x in{0,1} for the documented input domain, with each incoming limb below2^62. No comparison is removed and no exceptional input is assumed away.

Full fnorm then adds C to t0 if x is one, propagates carries again and stores every normalized limb. The hash needs only bit0 of normalized t0 for Y. Because C is odd, that bit is exactly (t0 XOR x)&1. Carry propagation into higher limbs and masking to52 bits cannot change bit0. The new helper returns this parity without constructing the full canonical Y output. It preserves both the x=0 and x=1 cases, including the field-modulus boundary.

The production call still fully normalizes both X coordinates before serializing them. It keeps the original Y values, then calls fparity on each in hash_block and uses the same plus/minus packing and low-bit mask as before. No EC recurrence, field product, scalar inversion, point construction, candidate count, key ordering, SHA function or publication gate changes. The old parity code is retained under the zero setting of the compile-time switch.

When QCG_EC_HOOK is defined, the original full Y normalization remains before the hook so its canonical coordinate outputs are preserved. Calling fparity on an already canonical Y still returns the same bit. No hook implementation or test harness is edited. All callers of hash_block in this IFMA path are reviewed; the original field normalization remains available for other uses and has no NORM_IFMA modification. AVX2-only and scalar implementations are unchanged.

## Structural and compiler evidence

A scalar model checked20,010 legal limb vectors, including zero,one,p-1,p,p+1,2p-1,2p,the maximum256-bit value,the maximal permitted62-bit limbs and20,000 seeded random vectors. It compares the candidate parity with the low bit of complete promoted normalization and the parity of the represented big integer modulo p. All matched. The correction flag was zero in20,005 cases and one in five boundary cases. This explicitly reports coverage of the rare final correction rather than implying random cases establish it frequently.

The model is scalar integer arithmetic, not execution of IFMA instructions. The exact same normalization prefix is shared by the modeled old and new derivations, and the final identity follows from C being odd. Target execution of the complete program remains an official-validation responsibility.

Actual-header wrappers compiled with GCC13.3 -O3 compare full fnorm followed by n0&1 against fparity. Both wrappers already allow the compiler to eliminate unused higher-limb outputs, so this comparison does not claim that all second-pass carries were formerly executed. vpaddq sites change5->4,vpand6->5,vpternlogq2->3,vpxor1->0,vpor1->0 and vpsubq1->0. vpmadd52luq remains1,shifts remain6,comparisons remain3,and stack-reference sites remain zero. These small static changes are evidence of distinct generated code, not a measured throughput result. The compiler already removes much of the unused full-normalization work; the residual benefit could be negligible.

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

Both the full native carrier generation and standard nvcc host build completed with exit code 0 for this exact source. CUDA 12.8.93 reported 15 spill records, all zero for stores and loads. The exact prepare section passed the 5 LTC64B-load gate. Prepare and finish use 128 and 64 registers respectively. The cubin is 476832 bytes, SHA256 913a97b2a8e6354e7632a2f41a997a05bf0e61e5a8b9a5c5c793f931e983b463, byte-identical to the promoted device image. No executable was run on a GPU or on the target SIMD CPU. These results establish compilation and scope only, not performance.

Dispatch recheck, 2026-09-29: after ZMASK4 completed naturally, source8d07 was synchronized and this isolated parity package was rebuilt. Both native and host builds exited0 again, with15 zero-spill records,prepare128/finish64,five LTC64B and the same476832-byte device image. The then-live frontier is1,008,206,828. No measured local gain is asserted.
