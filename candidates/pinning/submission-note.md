# Pinning IFMA canonical normalization: construct the correction predicate in mask registers

Model: GPT (exact variant not exposed)
Harness: Codex

## Exact runtime change

QSB_CG_NORM_MASK=1 changes only construction of the exact final-subtraction predicate inside v4i::fnorm. The promoted code obtains three full-width vector comparison results, combines them with vector AND operations and a vector of ones, and ORs that value into t4 >> 48. This candidate uses three AVX-512VL comparison masks, intersects their four relevant lane bits, and materializes a vector of zero or one with maskz_set1_epi64. It then performs the same OR with t4 >> 48. The first comparison uses unsigned t0 >= 0xFFFFEFFFFFC2F, equivalent to the old signed t0 > 0xFFFFEFFFFFC2E under the existing normalized limb bound. The other two comparisons retain exact equality of t4 with M48 and m with M52.

The source's first folding stage, IFMA multiplication, carry propagation, m accumulation, final t0 += (0-x)&C correction, second carry chain, and canonical stores are unchanged. No normalization is skipped. Both X and Y continue to be fully normalized where the promoted caller requires them. The feature gate remains avx2,avx512f,avx512vl,avx512ifma; the new mask intrinsics do not introduce an unchecked CPU capability. There is no change to the meaning of x: it is exactly zero or one in each lane, as before.

The equivalence is lane-local. After the original first carry pass, t0 is masked to 52 bits, so it is nonnegative in signed 64-bit representation. Signed greater-than against P0-1 therefore equals unsigned greater-than-or-equal to P0. A true vector comparison previously contributed all one bits; AND with the vector value one reduced it to one. The new mask intersection contributes one mask bit in precisely the same lane, and maskz_set1_epi64 contributes the same integer one. False cases contribute zero. The unchanged top carry is ORed with that result in both forms. High unused mask bits cannot create additional vector lanes.

## Independence and closed experiments

This package is based on ff27a2b66990a3eb554a1d4453e896c0397337ba. It does not contain the H0BOUND predicate change submitted as 33b88396-81e4-433f-968a-4edba12f9607. That job naturally rejected on 2026-09-29 at 21:50:29 UTC with verified official score 974,263,417; its exact package is closed. This replacement targets the field canonicalization predicate, while H0BOUND targeted the later hash prefilter. It retains the original hash prefilter.

The previously rejected NORM_IFMA experiment changed the arithmetic form of the final correction into QI_LO. This candidate retains the promoted final arithmetic verbatim and only changes comparison representation; it does not retry or combine NORM_IFMA. Closed PARITY_ONLY, ZMASK4, KEY_H0ONLY, KEY33PAD, SHA32PAD, FSEL3, PACK2, split45, VL SHA and square changes are absent. KEY_H0ONLY naturally rejected with verified score 989,216,415 on 2026-09-29 at 19:55:21 UTC. Its output specialization is not reintroduced. Every prior blacklist remains in force.

## Specific evidence and limitations

The deterministic integer model passed 20,026 groups of four legal five-limb vectors, 80,104 vectors total, each input limb below 2^62. It checked every one of the sixteen four-lane correction-predicate patterns explicitly, boundary values around p and 2p, zero, one, maximum permitted limbs, and random vectors. Original and new full canonical normalization results matched each other and independent big-integer reduction modulo p. The correction distribution was 80,052 zero and 52 one cases. This executes scalar integer math, not IFMA or AVX-512 instructions.

An actual-header GCC 13.3 -O3 wrapper copies the input field into a local field value, applies full normalization, and stores all five result limbs. Comparing the switch off/on shows vpand sites 12 to 11 and vpternlogq 2 to 1. The three vector comparisons vpcmpgtq plus two vpcmpeqq become one vpcmpuq and two vpcmpq mask comparisons. Stack memory-reference sites are zero in both wrappers. Other arithmetic sites remain unchanged. The compiler can use predicated mask comparisons to combine conditions, so source-level mask ANDs must not be counted as guaranteed separate machine instructions. These are static counts and a concrete representation change, not measured cycle savings or a throughput prediction.

The candidate may be neutral or slower. Mask-register dependencies, compiler integration into the surrounding caller, instruction scheduling and field workload frequency determine whether the isolated static change helps. A micro-wrapper does not reveal the complete controller's throughput, and another CPU path may bypass this function. The official evaluator alone determines correctness and performance. No target execution, local runtime gain, claimed score or guaranteed promotion is reported.

Credits: kaankolcu terrapinelf ercumentyildirim cefika DPZZxlz hybridnoise i34-9 ItlaStudent. These credits recognize the promoted implementation and its lineage. Original licenses and comments remain. The change and scalar/static evidence are independent work, with no donor speed claim.

## Scope and provenance

This is a CPU-side kernel experiment for the Quantum Safe Bitcoin challenge, prepared from freshly synchronized official source ff27a2b66990a3eb554a1d4453e896c0397337ba on 2026-09-29. Model: GPT (exact variant not exposed). Harness: Codex. Solver: dukemawex. The contribution is an independently derived, narrow representation change in existing candidate code. It is not a copy of the frontier offered as a new optimization, a repeat of a scored package, or a claim to reproduce someone else's measured speedup.

All inherited promoted components remain in place. The existing work distribution, candidate enumeration, elliptic-curve recurrence, SHA-256 rounds, message padding, key ordering, exact OpenSSL gate, publication protocol, and GPU controller are preserved. No scorer, harness, verifier, measurement code, timer, success stamp, result file, or CI workflow is changed. The candidate does not recognize benchmark text, precompute answers, guess outputs, use hidden cases, or change the accepted result set. Every claimed equivalence concerns arbitrary machine words under the documented input bounds, not particular benchmark values.

## Evidence limits

The development environment has no GPU and lacks the native CPU SHA-NI/IFMA capabilities required for executing this path. No target SIMD instructions or GPU runtime were executed locally. The scalar model checks only the changed identity. It is not an execution of SHA-256 rounds, field arithmetic, or an end-to-end benchmark. Compilation of actual-source wrappers shows the compiler's static choices, not timing or dynamic instruction counts. Static differences do not establish a throughput improvement, and their effect can be masked by hashing, field work, table loads, controller mode, memory pressure, or the unchanged GPU component.

This is a compile-qualified experiment only after both the native carrier build and standard host build complete successfully. The native build regenerates the candidate's normal source carrier using CUDA 12.8.93. A development-only fallback uses nvdisasm because local cuobjdump --dump-sass crashes; exact required symbols, kernel section selection, and the digest LTC64B gate remain enforced. This fallback changes how local disassembly evidence is read, not the submitted computation or validation rules. The regenerated GPU image is checked against the promoted image because the intended runtime change is confined to CPU code.

## Reproduction and acceptance criteria

Use the benchmark's current source as the base, not an archive branch's execution tree. Apply only this candidate's recovery patch, copy this note to the track's submission-note.md, and regenerate the carrier. Use CUDA 12.8.93 and the ordinary candidate build_carrier.sh 24 command. Then perform the standard host compile with nvcc -O3 -DQSB_ZEROS_N=24, the track's .cu source, and -lcrypto -lm. Building succeeds independently of running the resulting executable. Do not run the executable on this development host and do not interpret its existence as a runtime result.

Recheck the official source reference, exact solver queue, frontier, overlap, and closed-package list immediately before dispatch. A waiting package is not permission to submit from a stale base. If source moves, inspect the relevant helper and callers and rebuild after reapplying the change. Maintain at most one active submission for this solver on each track; let every active job finish naturally. Do not cancel or duplicate an active experiment. The concurrent other track is independent, and the separate MLX work is untouched.

Only officialScore and officialMetrics from the official evaluator may establish performance and verification. No local score is supplied. A positive static result does not promise a positive official result, and a small positive official difference can still miss the required promotion margin. This package must not be resubmitted merely to obtain a luckier run. If rejected, close the exact package and move to a distinct, evidence-supported change. Previously rejected isolated imports remain closed even where newer promoted sources legitimately inherit related components.

## Review boundaries

The feature switch defaults on only for the named helper. Its off branch preserves the original implementation for source and assembly comparison. It does not select a benchmark-specific answer, change input-dependent correctness, or skip required computation. Runtime feature detection and the existing fallback paths remain as inherited. The model covers edge values deliberately, while the source-level identity supplies the general argument beyond sampled inputs. Existing attribution and license headers remain intact. The public note explicitly distinguishes scalar modeling, static assembly inspection, compilation, official verification, and official performance; none of those forms of evidence substitutes for another.

## Completed build qualification — 2026-09-29

Native and standard host CUDA 12.8.93 builds both exited 0. Native logs contain 15 zero-spill records, prepare128/finish64 and five LTC64B loads. The 476832-byte device image SHA256 913a97b2a8e6354e7632a2f41a997a05bf0e61e5a8b9a5c5c793f931e983b463 equals promoted. No local target execution or timing was performed.

Requalification on 2026-09-29: fresh sync returned ff27a2b66990a3eb554a1d4453e896c0397337ba. Native and host builds passed again after restoring the matching CUDA 12.8.93 compiler package; native image and all gates match the previous qualification. No target runtime execution or local speed claim.
