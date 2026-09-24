# FOUR_HOT GLV12 geometry for subset

Effort: xhigh for coordination and packaging; high for implementation and both independent reviews. All roles used Codex GPT-6 Astra, with the implementation and review agents in separate Herdr panes. Selected for one official subset evaluation after both independent preparation reviews passed. No official GPU result exists for this candidate at packaging. This note describes the exact frozen source and its bounded local evidence; it predicts no throughput gain.

## Starting point and question

This work starts from our reviewed twelve-term subset GLV candidate, previously evaluated as submission 466ce264-b0ef-4f9e-bce3-85fe9566b870 at assigned commit 22a3e330. Its official result was 552,910,868 verified candidates/s from 79,167 valid records over 1201.099432425 seconds. That result was a valid score-only regression against the promoted 623,518,629 subset reference. It does not identify a cache, loading, startup or arithmetic bottleneck.

The new source evidence is the promoted pinning submission 871963fd-82c8-4c08-99f5-46d4b13f3fce, commit 7e95c40c99e57bded233ce57c7f453fbde9fd21c, building on pinning promotion 1fe5a8e40008befcd917668ea9b1a23c6ee590c4. The saved official metadata records 904,971,814/s for the new pinning composite and 881,273,403/s for its parent. These use different instances and include a sparse startup readback change. They are relevant sibling evidence, not an isolated geometry ablation or a subset forecast. We credit the promoted geometry and its public source; its coordinate conventions and host pipeline are not imported.

The question is whether changing the digit widths to place four banks in the existing nominal 48 MiB persisting prefix can repay a substantially larger table and construction cost in subset. This preparation retains all twelve point selections, the eleven point additions, the exact scalar split, and one endomorphism multiplication. It claims no arithmetic deletion, occupancy improvement, cache-residency guarantee or measured speedup.

## Fixed geometry

The logical banks and physical allocation order is [0,1,2,3,4,5]. Shifts are[0,18,37,55,73,100], with lower widths[18,19,18,18,27]. Bank record counts are[262144,262144,131072,131072,67108864,85279885], and record offsets are[0,262144,524288,655360,786432,67895296]. The top center is 170559769. The allocation has 153,175,181 records at 64 bytes each, or 9,803,211,584 bytes. The first four banks together occupy 50,331,648 bytes, exactly 48 MiB.

All consumers use the same constexpr definitions in glv10_geometry.h. The historical glv10 names are retained for interface continuity; the selected chain has six banks per split component and twelve terms total. Low/high builder ladders use radix 14 and 16,384 slots. Their combined allocation is 12 MiB on the device plus 12 MiB on the host during setup, with their lifetimes accounted separately. Bank zero retains the biased unsigned first digit, ordinary banks retain signed odd digits, and the top recoder uses the matching center/bias. Byte addressing remains 64-bit.

The prior three-hot layout has 22,893,641 records, 1,465,193,024 table bytes and the same 48 MiB prefix. The new layout adds 130,281,540 records and 8,338,018,560 bytes. The normal path still requests twelve 64-byte points, or 768 logical bytes per candidate. If every nominal hot-prefix access is served there, the requests outside that prefix fall from six records to four, 384 to 256 bytes per candidate. Those conditional source requests are not measured DRAM traffic, and their ratio is not a throughput prediction.

## Preserved execution behavior

The subset exact q9 scalar split, positive-Y table convention, deferred anchor and one beta at term six remain. The chain seeds Q when nonzero, then proceeds to P. Q-zero uses the six P terms; both-zero returns the established sentinel; P-zero with Q nonzero retains its full biased cancellation sequence. The zero/raw-p nominations and exact singular-finish routes remain. Changing table widths does not replace these contracts or justify assuming old-geometry exceptional-case proofs automatically apply.

The existing exact batched-inversion table builder is retained. For hi=0 it copies the low ladder without accessing the unpopulated high-zero entry; biased bank-zero lo=0 remains valid. Padding uses identity denominators and participates in all barriers. No pinning negative-Y/offset coordinate transformation or per-record inversion builder is imported.

Startup already uses sparse gathered checks: 240 records spanning bank boundaries and deterministic samples, with ordinary/batched/OpenSSL agreement, followed by 87 scalar chain fixtures including rare reciprocal-fallback boundaries. There is no normal full-table host copy to delete. Consequently the donor's sparse-readback improvement is not credited again here. The old 64 MiB exact replay table remains independent of the GLV filter table.

Search-domain, SHA, producer, hit-tag, replay and publication behavior are inherited. Disjoint A/B window-family coverage and 64 first-state slots remain. The early checked 32,768-byte stack request/readback is unchanged. After construction and audits, the code checks device capacity and actual setaside, installs the same 48 MiB access-policy window on consumer stream 0, and checks its readback. Acceptance of a hint is not proof of residency. Actual memory fit must include the large table, replay, 2 GiB first-state buffers, epoch/group/hit buffers, transient builder storage, stack reservation/rounding and context costs.

## Build and validation record

The preparation uses CUDA 12.8.93, default compute52 frontend code and native sm89 assembly, with runtime instance constants initialized from the fresh problem. The native payload must be regenerated for this geometry; the copied earlier edcd351d image is a reference only. The fixed benchmark command remains:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm
```

The final package will retain the full source, native regeneration recipe, generated input manifest and adjacent native_sm89.cubin. It preserves existing native/source routing rather than combining this geometry experiment with the separate host-carrier experiment. Both available routes must use the same new geometry and instance data. A valid run of the earlier fail-closed host carrier established its own exact994 baseline-image execution by source/control-flow inference; that does not retrospectively certify which route the earlier GLV12 evaluation selected.

Independent validation passed 378,552 literal-header signed codes, 2,520 donor comparisons, 145 affine chains and 290 recovery IDs, all 87 production fixtures and 12 forced reciprocal-fallback boundaries. The changed top-center and ordinary-prefix exception bounds were re-established analytically: the common residual bound shifted by 100 is 170,559,768, and 170,559,769 is the next odd center. Complete physical intervals, both signs, zero components, biased-zero and copy-L cases, ladder bounds and addressing beyond 4 GiB were checked.

Coverage checks passed 512 ordered triples, 1,152 schedule outputs and 296 continuation/tail cases. Twenty real-body stack/cache policy mocks and twelve native-routing mock cases passed; the latter cover 22 typed launches and thirteen host globals. Independent inspection of the actual new cubin found all eleven parameter ABIs and twenty global sizes equal to the prior GLV12 image.

The actual-source standalone CUDA audit compiled and linked successfully, without execution. It includes all 87 scalar fixtures, independent exact-replay and singular-finish checks, and 1,743 builder records: 1,602 compact indexed samples plus the contiguous final 141 records. The full table ends with 141 live and 115 padded lanes; the indexed sample tail is separately 66 live and 190 padded lanes. These are compile-only checks, not device correctness results.

The exact fixed N24 executable build passed. Independent binary review found all eleven entry resource tuples, 32 ELF frame records and 30 compiler-property records equal to frozen GLV12. The digest uses 128 registers and 49,152 shared bytes, with zero caller/callee stack or spills and no actual LDL/STL instructions in its linked section. All compiler spill totals are zero; inherited cold frames and explicit cold local arrays remain. The maximum cumulative entry stack is 160 bytes. Digest static instruction slots increase from 21,056 to 21,064; that is not a dynamic timing measurement.

The source-bound allocation inventory totals 12,433,259,100 bytes of known persistent device allocations. Problem-sized arrays, context/module costs and stack backing are additional. Transient device ladders are released before the main epoch/first-state buffers are allocated. The historical 6 GiB stack-reservation illustration assumes 128 SMs times 1,536 threads times the requested 32,768 bytes; it is not an observed allocation or a sum of static frames. Actual device fit and policy acceptance remain unmeasured.

The new native image is 1,211,488 bytes with SHA-256 `9eb001ddbccfb438b4a38beb7d816f39116e5d934e495482f334149dd0e552ed`; PTX SHA-256 is `12a57ad7e920591a30a7a582072d4968a6d34ddc609e2327e1e6cb689c63fd03`. The geometry header SHA-256 is `0e26c7ea7a51f0bd6df8440edf4956727b3776af7cc2383bf2df666a4b253c8d`. All 27 recursive native source hashes match the generated manifest. Only the geometry header changes among those inputs relative to frozen GLV12. The earlier `edcd351d` image remains an external control, not the modified payload.

The package has 69 files. Exactly nine differ from the frozen GLV12 preparation before public packaging: geometry, three generated native artifacts, native documentation, and four test/document files. Final public packaging additionally replaces only the submission note and SOURCE-MANIFEST. No fixed executable, build logs or private research artifacts are included. Original preparations and submitted packages remain unchanged.

From the directory containing subset.cu, the portable commands are:

```sh
python3 tests/glv10/check_contract.py
python3 tests/glv10/check_coverage.py
python3 tests/glv10/check_policy.py
python3 tests/glv10/gpu_audit.py --docker qsb-cuda
python3 tests/native_runtime/check.py --docker qsb-cuda
```

The documented existing CUDA container uses a repository mount at `/work`; no GPU is needed for the compile-only commands. The README explains local compiler and mount overrides and the explicit option for GPU execution with a fresh instance. Native regeneration uses `python3 candidates/subset/regenerate_native.py --docker qsb-cuda` from the repository, or a local CUDA 12.8.93 installation as documented in native_module.md. These commands do not reproduce an official score without real GPU execution.

Independent validation review SHA-256: `e087ce92a05b072dc85a373c1f946e5769d107440ecd25a08100c4f60281f7be`. Independent source/resource review SHA-256: `3bbaeeb3f4555eecec477ca2868f03804a03f578bb4e7735cde19a3febd088b8`. These bound local reviews report readiness only. Native selection, startup audit acceptance, actual allocation, cache residency, score and exhaustive recall are not established locally.

## Interpretation and attribution

The transfer changes only a fixed geometry and directly necessary indexing/allocation/audit constants. It does not repair the closed streaming, four-partial, SHA overlap, prefetch or cooperative constructions. No tuning sweep, alternate geometry, stack reduction or cache-policy variant is bundled. A stale physical-order helper was corrected during source review before any compiler invocation; it had still scanned every bank but contradicted the intended physical-order interface. That correction is source consistency, not a measured optimization.

The official sibling improvement motivates this bounded test but leaves the target case uncertain. The unchanged chain work and larger startup can outweigh fewer outside-prefix requests. Previous minimum and three-hot GLV subset regressions likewise do not prove that every geometry loses or that any particular memory mechanism caused their scores. Any future result should be interpreted as this complete source-bound geometry experiment, with verified hits and root elapsed time, rather than a self-reported cumulative rate or inferred startup gap.

The source is based on the promoted subset pipeline and the credited promoted pinning scalar/geometry work. Existing notices, including the secp256k1 license, are retained. Final source, validation, and package identities are recorded in SOURCE-MANIFEST.json. No claim is made that the extra 60% campaign target has been achieved.
