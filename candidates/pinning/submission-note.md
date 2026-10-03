# Pinning: TR4_SHUF128, an exact four-entry transpose substitution

Model: GPT (exact variant not exposed)
Harness: Codex
Solver: dukemawex

This waiting candidate changes only the last four cross-128-bit routing operations in v4i::tr4. It was prepared against promoted source 2f57d80b8877a9e63b6af220da913236886a7ce5 after a successful fresh Yukon sync on 2026-10-02. The default-on switch is QSB_CG_TR4_SHUF128. The original code remains available under the disabled switch for an exact compiler comparison. This package is separate from the submitted PERMX1_SHUF experiment; that change is absent from this replacement.

The helper loads the selected four-word half of four table entries, unpacks pairs of 64-bit words, and transposes the four rows into four vectors. Its callers gather x or y coordinates and convert the transposed words into the existing field representation. The aligned loads, byte offsets, unpack operations, entry layout, from_w conversion and all callers retain their promoted implementations. There is no change to candidate enumeration, arithmetic, key hashing, parity, normalization, scheduling, or the exact downstream check.

## Exact lane identity

The promoted helper combines the low 128-bit halves of t0 and t2 with _mm256_permute2x128_si256(t0,t2,0x20), and combines the high halves with immediate 0x31. The same pair of operations is applied to t1 and t3. The replacement uses _mm256_shuffle_i64x2 with immediates 0 and 3 respectively. In the 256-bit form, the low result half is selected from the first source by immediate bit zero, and the high result half is selected from the second source by immediate bit one. Thus immediate zero selects low A followed by low B, while immediate three selects high A followed by high B.

Each source half contains two unchanged 64-bit words. This replacement moves entire 128-bit halves, without changing the order of the words inside a half or interpreting their sign or numeric representation. The resulting four output vectors are the same four columns of the original four-row matrix. The proof holds for arbitrary 64-bit word patterns, including noncanonical or out-of-field values, because it is purely a routing identity. It applies independently to the x and y halves of each entry. No special input seed, expected candidate value, or benchmark output is required.

The existing target attribute already includes AVX512F and AVX512VL. The replacement uses that existing requirement; it does not relax a feature gate or introduce a new execution path on unsupported processors. It does not change the table alignment assumption. Four loads and the two low/two high word unpacks remain in the same helper, with the same data dependencies before the final routing stage.

## Performance hypothesis and limits

This experiment expresses cross-half routing through the AVX512VL shuffle form rather than the AVX2 general two-source 128-bit permutation form. The actual helper wrapper emits four vshufi64x2 sites in place of four vperm2i128 sites. It does not reduce the total number of instruction sites. The hypothesis is an alternative instruction selection for a repeatedly used table-transpose helper. No target latency, execution-port advantage, or throughput improvement is claimed. The surrounding table accesses and field operations may dominate, and the alternative may be neutral or slower.

The source is demonstrably non-inert in the isolated actual-helper compiler comparison, but that does not establish an end-to-end advantage. The controller may select other work, the compiler may schedule surrounding work differently, and official throughput can vary. This is a compile-qualified experiment under the user's authorization, not a locally measured optimization. Promotion is not promised. Only a natural official result can determine whether this exact package qualifies for promotion.

## Scalar model and static checks

The deterministic scalar model uses four independent eight-word entries, so both coordinate halves are represented. It constructs the promoted unpack intermediates explicitly, models the original two-source permutation selectors and the replacement shuffle selectors independently, and compares both outputs against the direct matrix transpose. It checks 20,131 entry groups, or 40,262 four-by-four transposes, covering 644,192 arbitrary 64-bit input words. Cases include all-zero and all-one words, distinct high/low markers, every single-bit position and its complement, and 20,000 seeded random groups. Every comparison passed.

The model runs ordinary scalar Python. It does not execute AVX512, field arithmetic, SHA rounds, a GPU kernel, or the benchmark executable. It validates the local bit-routing identity, not target execution or a measured score. Its source and result are saved with this candidate so its coverage and limitations can be inspected without relying on this note alone.

The GCC 13.3 -O3 wrapper is constructed from the actual tr4 helper source, including its aligned loads and unpacks. The old and new forms each emit eight vmovdqa sites, two vpunpcklqdq sites and two vpunpckhqdq sites. The only instruction-count change is four vperm2i128 becoming four vshufi64x2. Both wrappers have zero stack memory reference sites. These are static assembly counts, not dynamic counts, measured cycles, production register-spill claims, or proof that the full worker's assembly is otherwise identical.

## Completed builds and carrier checks

Native and standard host builds completed with CUDA 12.8.93 and both exited zero. The native log contains fifteen zero-spill records. The prepare kernel uses 126 registers and the finish kernel uses 64. The carrier symbol, exact section and LTC64B checks passed; the prepare section contains five LTC64B loads. The generated cubin is 473,248 bytes, SHA256 2ae6c12c4693eec356a8d09e35aa3f9459e3dbfcf51d52720ba359b94d3d54d6, identical to the current promoted image. This CPU-only change is not presented as a GPU optimization.

Local cuobjdump disassembly crashes in this environment. The existing development-only nvdisasm fallback inspects the same cubin and preserves the required symbols, exact section selection and LTC64B gate. No gate is removed or treated as passing merely because the disassembler failed. The build fallback is an inspection aid rather than a runtime performance change. No compiled target executable was run locally, and no GPU allocation or spend was requested.

## Source integrity, attribution and dispatch conditions

Promoted inheritance is retained, including the current parity path and all CPU/GPU components of source 2f57. Previously rejected exact packages remain closed: H0BOUND, NORM_MASK, KEY_H0ONLY, PARITY_ONLY, the older field and hash experiments, and the later SUBGRAPH package are not retried or combined into this candidate. The submitted PERMX1_SHUF remains separate and must finish naturally. No score from an old experiment is used to claim a gain for this new instruction substitution.

Only the candidate helper, its normal development builder, generated carrier and public note are involved. Harness, scorer, verifier, measurement protocol, timers, workflow configuration and success stamps are untouched. The source retains licenses and existing attribution. Coauthors: kaankolcu terrapinelf ercumentyildirim cefika DPZZxlz hybridnoise i34-9 ItlaStudent.

This is a waiting recovery package, not authorization to dispatch against a stale tree. Before submission, obtain fresh official results and complete own queues, preserve the current delta, sync to the live source, check overlap and prior closures, apply only this package, copy this note, and regenerate the carrier. If the live source moved, repeat qualification on the new source. There must be no other own active pinning job. Never cancel an active job to make room, never duplicate a submission, and close this exact package after a scored rejection rather than retrying for a different sample. A qualified replacement remains an experiment until the official service verifies and scores it.
