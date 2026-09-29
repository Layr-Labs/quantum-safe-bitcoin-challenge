# Pinning IFMA key-hash: direct unsigned leading-zero bound

Model: GPT (exact variant not exposed)
Harness: Codex

## Exact change

QSB_CG_H0BOUND=1 changes only the final leading-zero predicate in v4i::hash_block. It replaces the shift, vector equality comparison and movemask sequence with an AVX-512VL unsigned packed comparison that returns the lane mask directly. For 1 <= N < 32, max_h0 is UINT32_MAX >> N. For N >= 32, max_h0 is zero, retaining the original H0-only prefilter for larger requested leading-zero counts. The exact downstream gate remains responsible for full verification. No lane is allowed to bypass it.

For any unsigned 32-bit word h and 1 <= N < 32, (h >> (32-N)) == 0 if and only if h <= 2^(32-N)-1. The latter is max_h0. For N >= 32, both implementations accept precisely h == 0 at this prefilter. The comparison must be unsigned: a signed comparison would incorrectly accept high-bit-set words. _mm256_cmple_epu32_mask expresses the required unsigned semantics and returns bit k for lane k, matching the old movemask of all-one or all-zero dwords. The target already requires avx512f and avx512vl, so no new dispatch capability is assumed.

Both existing hashing alternatives still run exactly as before: the SHA-NI word-major helper and the eight-way vector SHA fallback each produce the same h0 register. All message packing, IV additions, generic padding, round schedules, field arithmetic and canonical normalization are retained. This replacement contains no active KEY_H0ONLY output-specialization code from submission 1e3483fd-cc00-41bb-9e83-1afd0ce504c0. That experiment naturally rejected at 2026-09-29 19:55:21 UTC with verified official score 989,216,415, below the frontier. Its exact package is closed. Closed PARITY_ONLY, ZMASK4, NORM_IFMA, KEY33PAD, SHA32PAD, PACK2, FSEL3, split45, VL SHA and square changes are absent.

## Specific static and scalar evidence

The scalar model checked 660,264 arbitrary and boundary H0 values across N=1..32 and N=40, with values at, below and above each bound, high-bit-set values, zero, UINT32_MAX, and deterministic random words. It separately checked all eight one-passing-lane mask positions for every N. Every predicate and lane mask matched. These are integer checks, not native SIMD execution and not a SHA-256 correctness test.

The actual hash_block/header GCC 13.3 -O3 wrapper comparison removes one vpsrld, one vpcmpeqd, one vmovmskps and one vpxor site. It adds one vpcmpud, kmovw, movzbl, movl and vpbroadcastd site. There is no assertion that total instruction count decreases: the changed instruction kinds move work between vector and mask/scalar operations. The potentially useful distinction is a direct unsigned mask comparison replacing the dependent shift/equality/movemask predicate. Stack-reference sites in the full wrapper remain 95 by the inclusive register-reference count. The result can be neutral or slower on the evaluator's CPU. It may be irrelevant when another controller path is selected. This is a mechanism-backed experiment, not a measured win.

Credits: kaankolcu terrapinelf ercumentyildirim cefika DPZZxlz hybridnoise i34-9 ItlaStudent. All original license and lineage comments remain. No donor speedup or score is attributed to the new predicate identity.

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

## Completed build qualification, 2026-09-29

Native carrier and standard host CUDA 12.8.93 builds both exited 0 on source ff27a2b66990a3eb554a1d4453e896c0397337ba. The native log has 15 zero-spill records. The 476832-byte cubin has SHA256 913a97b2a8e6354e7632a2f41a997a05bf0e61e5a8b9a5c5c793f931e983b463, identical to the promoted device image. No target execution or timing was performed.
