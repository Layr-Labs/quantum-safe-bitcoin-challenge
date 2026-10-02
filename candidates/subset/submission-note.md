# Subset: QSB_CPU_SWAP1_SHUF=1, exact adjacent-lane shuffle

Model: GPT (exact variant not exposed)
Harness: Codex

Only fe8_swap1 changes: _mm512_permutex_epi64 with 0xB1 becomes _mm512_shuffle_epi32 with 0x4E. fe8_swap2, fe8_swap4 and fe8_inv_lanes are unchanged. The currently enabled scalar-inversion path still calls this helper; promoted lane-zero canonicalization and every other inherited JL component remain intact. This is not a change to KH16 output packing.

The actual full-output fe8_swap1 helper wrapper changes five vpermq sites to five vpshufd. All other instruction counts are unchanged. The fe8_swap2 and fe8_swap4 control wrappers retain their instruction counts. Inclusive stack-reference sites are two in both versions of each wrapper. This is a helper-level static comparison; no claim is made that all surrounding inversion or worker assembly is identical.

## Equivalence and scope

Each input vector is viewed as pairs of 32-bit words making arbitrary unsigned 64-bit values. Within every 128-bit half, immediate 0x4E selects 32-bit words 2, 3, 0, 1. It therefore swaps two neighboring 64-bit values, keeping each value's high and low words together. In a four-lane vector the result is b,a,d,c. In an eight-lane vector it is b,a,d,c,f,e,h,g. This is exactly the lane-index XOR-one operation of the promoted 64-bit permutation. It is valid for every possible lane bit string; the argument does not depend on field bounds, signed values, a particular benchmark seed, or an observed output. Applying the operation twice returns the original vector.

The implementation processes all five limbs. It retains the original expression under the feature-switch off branch so the compiler comparison is reproducible. No multiplication, carry, normalization, inversion, candidate enumeration, hash round, padding word, prefilter or exact verification gate is removed. The resulting field values and their lane order are unchanged. The new intrinsic is supported by the existing target attribute and feature gate. No new hardware feature is assumed. Every neighboring helper and caller retains its inherited implementation.

The hypothesis is that a shuffle confined to 128-bit portions is a useful expression of the actual required routing. It is a substitution of five instruction sites, not a reduction in total instruction count. We do not claim a target latency, port advantage or throughput improvement. The path does relatively little of this permutation work compared with field multiplication and hashing, so the end-to-end effect may be negligible, negative, or inactive under another controller mode. The user has authorized compile-qualified experiments without local measured speed, and this package is submitted under that evidence standard. Promotion is not promised.

## Current source and earlier work

This package was prepared after a successful Yukon sync to full source SHA 2f57d80b8877a9e63b6af220da913236886a7ce5 on 2026-10-02. The latest source was inventoried before editing; an archive branch was not used as executable baseline. The full official submission lists showed no own queued or validating work at preparation. Another fresh sync and exact-own queue check are required immediately before dispatch. If the live source moves, the runtime diff must be reviewed and rebuilt on the new base.

The source legitimately inherits optimizations from earlier promotions, including changes related to previously closed isolated experiments. That inheritance is preserved. It does not reopen an old package. This candidate does not combine a prior near-threshold score with another old change, repeat an identical submission, repackage the frontier, or tune on hidden output text. It is a separate, explicit machine-word identity with source and compiler evidence.

The previously submitted pinning NORM_MASK, H0BOUND and KEY_H0ONLY packages are closed; their exact changes are absent from this delta. Subset KH16_H0ADD is closed. The later own subset MRGS and pinning SUBGRAPH packages also rejected and are not retried. The old subset H0PACK4 waiting patch is now subsumed by the new promoted register-mask path and is not dispatched. These closures do not modify whatever related code is legitimately part of the current promoted tree.

## Validation limits

The deterministic scalar model explicitly splits each arbitrary 64-bit input into two 32-bit words, applies the new immediate's lane selections, reconstructs the output values and compares them against XOR-one indexing. Boundary groups contain zero, all-one values, distinct high/low lane markers, every single-bit position and complements. Twenty thousand seeded random groups supplement these cases. The result is a check of the local lane identity, not a SHA, field inversion or end-to-end execution. No target SIMD instructions were executed by that model.

The static evidence uses GCC 13.3 at -O3 with wrappers made from the actual helper source. It reports generated instruction sites and inclusive stack references. These are not dynamic instruction counts, cycle estimates or measured performance. Full native and standard host CUDA 12.8.93 builds are also required. Their success establishes compilation and the carrier gates, not target runtime correctness or speed. The local environment has no GPU and lacks the required target CPU execution capabilities, so none of the compiled target executables is run locally.

The GPU image is regenerated through the normal candidate carrier builder. Its required symbols and exact kernel section are checked, including the existing LTC64B gate. Local cuobjdump disassembly can crash; the existing development-only nvdisasm fallback retains those same gates. That fallback affects local inspection only. The source change is CPU-only, so the native cubin is required to equal the promoted image. Any different image would need investigation before submission rather than being dismissed as incidental.

## Submission boundaries and provenance

Only candidate-directory runtime source and the candidate's ordinary generated carrier or development builder are involved. The harness, scorer, verifier, timers, result files, success stamps, workflow configuration and official measurement protocol are not edited. Inherited host scheduling and telemetry are preserved. No local claimed score is supplied. Only officialScore and officialMetrics can establish verified performance, and even a positive official score difference may fail the required promotion margin.

One own active job per track is maintained. Each current job must finish naturally; no cancellation or duplicate dispatch is used to free a slot. Pinning, subset and the separate MLX effort have independent gates. A waiting recovery patch is not permission to submit from a stale source. The patch, model, actual-source wrapper, compiler counts and build evidence are saved so the preparation can be reconstructed and checked. After a rejection, this exact package is closed rather than rerun for a luckier sample.

The work is an independent adaptation of the existing helper. Existing attribution and source licenses remain intact. Model: GPT (exact variant not exposed). Harness: Codex. Solver: dukemawex. No donor performance result is claimed as evidence for this change. The note deliberately distinguishes a general identity, scalar modeling, static compiler inspection, compilation, official verification and official performance: none of these is silently substituted for another.

Coauthors: cefika kshitij-hash terrapinelf i34-9 ercumentyildirim HyeokxC jacklightChen fkiene kaankolcu newjordan

## Completed qualification

The scalar model passed 20,068 groups, covering 160,544 arbitrary 64-bit lane values. Native and host builds both exited zero. The native build has 15 zero-spill records. Its cubin is 516,960 bytes, SHA256 cef81a9f525fae26edb6808ff84299887cef1472fa5cc9467ba8f1c34753cde3, byte-identical to the promoted image. The normal symbol/section/LTC64B gate passed. No target runtime or local performance measurement was performed.
