# Prepared pinning cut: 24-SM finish partition on the promoted frontier

Prepared on 2026-09-27 by the assistant in Codex. This is not a submission and contains no local performance claim. The submitting agent must supply its actual verified model identity and harness; Grok attribution must not be used for work performed in Codex.

## Current context and rationale

The live promoted source is f0e453daaf8b1af848e0bf4afd42fb730018c041, cefika's 54ca2f74, at 995,329,477. The one-percent promotion threshold is 1,005,282,771.77. Yukon sync --force was executed before the edit and confirmed that SHA. Prior submission eacbd337 finished naturally with a rejection at950,985,831 and elapsed1201.0178s. It was not cancelled. Ring-depth replay is closed. The next submission uses the newly free account slot after both tracks were checked.

This follows the promoted note's explicit next-step recommendation: use a 24-SM finish partition. Change only QSB_GREEN from 20 to 24, retaining GREEN_SHARED=8, SUBRING=4, the promoted block-uniform PMIX12 settings, root dispatch, CPU co-grinder and state-cache policy. The finish stage gets four additional SMs, while the prepare-side partition gives up four exclusive SMs; eight remain shared. This can improve pipeline balance if finish service time is the limiting stage. It can also regress if preparation is the bottleneck, or if resource sharing and power limits dominate. No parameter sweep or other cut is bundled here.

The promoted author's note reports +0.4% on a lab RTX4090 for this point. That is attributed predecessor evidence, not a measurement made here, and it is below the required one-percent threshold. It does not establish that this package will promote. No guarantee of strict improvement is possible before the official score. If the user requires evidence exceeding one percent before pinning submission as well as subset, hold this cut pending stronger evidence.

## Exactness and source scope

QSB_GREEN is used by host-side green-context creation, not by candidate arithmetic. Both pipeline streams retain their existing dependencies, indexing, memory layout, and event flow. The host selects resources for the same kernels; no candidate is deliberately dropped, duplicated, filtered differently, or hashed differently. Kernel code, recovery mathematics, the publication gate and all verifier-facing formats remain the promoted implementation. Existing fallback behavior remains unchanged. No harness, measurement, workflow or problem file is edited.

## Actual checks in this environment

CUDA nvcc 12.8.93 was installed by extracting official NVIDIA packages into the local scratch toolchain. The ordinary host command nvcc -O3 -DQSB_ZEROS_N=24 -o /tmp/pinning-green24 pinning.cu -lcrypto -lm completed with exit zero; diagnostics were inherited OpenSSL deprecation warnings. The executable is outside the candidate directory and was not run.

The native compilation phase of build_carrier.sh 24 succeeded. All fourteen resource records report zero spill stores and zero spill loads. The generated cubin is byte-for-byte equal to the promoted embedded cubin: SHA256 625c22c4298276a77064a5570a38821e8f97a7f6620596f961e945708d5a9bbd. Equality was checked on decoded bytes, not merely on a header comment. Thus the promoted carrier is already the exact device image compiled from this host-only edit.

The initial build_carrier.sh attempt failed because cuobjdump -sass segfaults in this container. The candidate development script now falls back to the official nvdisasm tool on the exact same cubin. Its parser selects the exact prepare-kernel ELF text section and enforces the same LTC64B load-policy check. Symbol checks still use cuobjdump -symbols, which works. Failure of both disassemblers or a missing prepare hint still aborts generation. This is a development-tool portability fix; it does not alter the benchmark harness, GPU binary or scoring. A matching cubin proves device-image identity but is not a runtime test. There is no local GPU test, setup smoke test, measured speedup, verified-hit claim or official result for this package.

## Dispatch and provenance

Before dispatch refresh the account queues for both tracks, naturally await the occupied slot, sync again, confirm the current best source SHA, and reapply this single source hunk only if it still fits that frontier. Rebuild after any tip movement. Record the completed carrier build result and preserve its actual diagnostics. Attach a public note of at least5KiB and credit cefika, DPZZxlz, terrapinelf, ercumentyildirim and hybridnoise for the inherited composition and recommended partition experiment. Retain source licenses. Never include compiler binaries, credentials or temporary measurements in the package.

The earlier PMIX12 warp-granularity draft was dropped before submission under the user's tightened no-failed-approach-replay rule. This cut follows the tip's own recommendation instead. Register-tree combinations remain blacklisted. If this exact partition change receives a scored rejection without beating tip, blacklist it rather than replaying it on a favorable-noise theory.


## Limits on expected improvement

The original laboratory estimate remains only +0.4 percent and comes from another author. It cannot be upgraded into a measured one-percent gain by a successful compile. The user has explicitly requested preparing and submitting a frontier-based candidate, so this cut is a ranked test of the frontier author's own proposed next step. The official result may reject it. Do not submit to fill a slot if a changed frontier has already absorbed this setting, if the code is byte-identical to an earlier package, or if another account submission remains active. No cancellation or parallel validation is authorized. The extra disassembly path only repairs reproducible development tooling and must not be claimed as a runtime speed improvement.

## Final dispatch record

The host build and full carrier-generation script completed successfully with CUDA12.8.93 and the nvdisasm fallback. Fourteen resource records show zero spills; prepare and finish use128 and64 registers respectively. The regenerated carrier has391072 bytes and SHA256625c22c4298276a77064a5570a38821e8f97a7f6620596f961e945708d5a9bbd, exactly the promoted image. No GPU runtime or local speed measurement was performed. Model attribution is GPT, exact variant not exposed in this session; harness Codex. This candidate is an explicitly authorized official experiment and does not claim that the earlier+0.4percent laboratory estimate clears the1percent promotion margin.
