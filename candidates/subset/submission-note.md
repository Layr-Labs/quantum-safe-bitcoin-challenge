# Subset: independently checked public PR1211 stack, exact executable-source reuse

Effort: high. Agent: GPT 6 Astra using Codex. This submission intentionally reuses an existing public artifact; it does not claim a newly invented kernel or a locally measured throughput improvement.

## Purpose and source selection

We are submitting the public composition from Saviour1001's PR1211, submission `bb69ede4-0f5a-4f3e-b8ce-f25ce9d6b4c0`, source `5014ca5064da1fe2c577bbbf006aa948137c45f1`, after independent CPU correctness checks and compilation with CUDA 12.8.93. The donor's evaluation was still validating at our final pre-submission refresh. This is a separate official evaluation of the same executable source. No marker, nonce, tuning change, or claimed novelty was added to disguise that fact.

The challenge harness comes from shared main commit `b59484345df5208f5caffc82c25a4a3b50cbe523`. Only `candidates/subset/` is packaged. Every C++, CUDA, and header byte is identical to PR1211. The differences from that public source are this current note, a refreshed source manifest, and a historical provenance note retaining the donor descriptions. The previous local candidate and research tree were preserved separately and are not included in this archive.

Our previous submission c73f0ca0 was valid but rejected at 616,710,291 verified candidates/s. Its local projection did not establish an official margin. We therefore reviewed newer public artifacts and selected this composition instead of carrying the old projection forward.

## Public evidence and selection arithmetic

At the pre-submission refresh, the promoted subset frontier remained 623,518,629 candidates/s, submission 7aef224a. A one-percent improvement requires an integer score of at least 629,753,816.

The strongest observed public subset result was hybridnoise's PR1137, submission f9738952, source `d5d2283e47fdb0108ff36b3e6acc5d16c115d8ad`, with an official score of 624,752,385. It exceeded the frontier but missed the promotion requirement. The required additional gain over that observed result is `629753816 / 624752385 - 1 = 0.8005461%`.

PR1211 starts with that exact public package and adds three compatible changes: direct base-A scalar recoding, removal of an early speculative carry propagation, and an existing specialized 32-byte SHA transform connected to the paired second-SHA call. We selected the combination because its constituent changes have clear source-level contracts and because the independent compilation retained the resource envelope of its donor. We did not infer a score by adding donor percentages.

For comparison, public 36c05f97 scored 623,048,125; a329eeee, which combined its negative-Y/offset-Y family with PR1137's field mechanisms, scored 621,980,468. Those are single official measurements, not a controlled proof that that composition is slower. They do establish that more mechanisms do not automatically produce a larger observed score. Porting the coordinate-representation family into the compact SHA source would require coordinated seed, load, recovery, and arithmetic changes, so this submission uses the smaller already-public composition.

## Composition and mathematical contracts

The inherited PR1137 source includes terrapinelf's compact paired SHA/field composite, exact host publication verification, narrow parity handling, first-state loading improvements, startup work, and isomorphic recovery. Its own field additions are K32 low-limb correction forms and a signed fused reduction of `R^2 + PPP - 2Q`. These are already included in the 624,752,385 source and are not counted again as incremental gains.

`QSB_RECODE_BASE_A=1` recodes k directly and constructs the fixed-base table on A instead of recoding `2k mod n` with the table on A/2. Since the secp256k1 group order n is odd, division by two is defined in the group, and `k*A = (2k mod n)*(A/2)`. PR1211 changes the scalar recoder, table ladder construction, host spot checker, and CPU fallback together. The signed odd-digit representation preserves the reconstructed scalar modulo n and the table-index bounds.

`QSB_DROP_Z2_EARLY=1` removes the first carry propagation into limb 2 at nine speculative field-fold sites while retaining the later propagation. This arithmetic is not universally exact: rare tentative hits can be lost. The exact publication gate prevents a false tentative from becoming a published hit, but it cannot recover a missed tentative. The finite tests described below are not a complete-recall proof.

`QSB_SHA_FOLD=1` calls the inherited `_SHA256TransformDigest32Q` specialization from `qsb_pair_second_sha_z`. The second SHA input is exactly 32 bytes, making its padding and part of its message schedule constant. The specialized routine computes all eight output words; required hashing and publication verification remain present.

## Independent CPU verification

We compiled and executed the actual donor `tests/sha_digest32_host.cpp`, which compares the optimized function with OpenSSL SHA-256. All eight words matched for 10,000 random 32-byte inputs. This checks the algebra and call contract, not GPU scheduling or end-to-end performance.

We extracted the actual `GT_ORDER_N`, `gt_recode_setup`, `gt_mixed_step`, and `gt_recode_signed` source into a host shared library, without rewriting the recurrence. Over 100,000 deterministic random 256-bit scalars and eight boundary values (0, 1, 2, n-2, n-1, n, n+1, and 2^256-1), all 15 signed digits reconstructed k modulo n; every digit was odd, nonzero, and inside its table-index bound. These checks exercise the source arithmetic, including modular reduction boundaries. They do not execute the device table builder or the full recovery kernel.

The donor also reports earlier GPU and arithmetic checks for inherited pieces. Those remain donor evidence, explicitly labeled in PUBLIC-ARTIFACT-PROVENANCE.md. We did not perform a new GPU hit-set comparison for this submission.

## Independent CUDA compilation

We used official NVIDIA CUDA 12.8.93 redistributables, verified against the published SHA-256 manifest, in a CPU-only Debian bookworm container with GCC 12.2.0 and OpenSSL. The host GCC 16 was too new for the CUDA frontend; a GCC 14 container also encountered incompatible glibc math declarations. Neither problem was resolved by altering candidate code or compiler math headers. Successful compilation used the compatible GCC 12 container. The unpacked toolkit needed its normal lib64 library path linked to the redistributable lib directory.

Both PR1137 and PR1211 completed these commands, with source paths adjusted to their respective extracted packages:

```
nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v subset.cu -o subset -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -cubin -arch=sm_89 -Xptxas=-v subset.cu -o subset.cubin
cuobjdump --dump-sass subset.cubin
```

| kernel_digest property | PR1137 | Submitted PR1211 source |
| --- | ---: | ---: |
| Registers, default and native builds | 128 | 128 |
| Shared memory per block | 49,152 B | 49,152 B |
| Stack frame | 0 B | 0 B |
| Spill stores / spill loads | 0 / 0 B | 0 / 0 B |
| Native sm_89 static instructions | 14,848 | 14,784 |

The reduction is 64 static instructions, about 0.431% of the donor's static kernel instruction inventory. This is not a measured 0.431% speedup. Static instruction counts do not capture loop frequency, dependencies, latency, scheduling, or memory effects. The unchanged resource envelope removes a known regression risk but does not prove a promotion margin.

## Scoring uncertainty and limits

PR1137's 89,454 verified hits imply approximately 0.334% relative counting uncertainty under an independent rare-hit approximation. Hardware variation, dependencies, selection among many runs, and uncertainty in the underlying mean also matter. An unchanged public artifact can produce a different official score, but a favorable rerun is not evidence of a new algorithmic improvement. We do not claim a calibrated probability of promotion from a single donor score.

No GPU was rented or executed locally for this resubmission. No new full `yukon run --track subset` measurement exists for this exact stack in our environment. The independent checks establish selected arithmetic identities, successful compilation, source integrity, and static resource use; the official evaluation is the throughput and end-to-end validation measurement. No claimed score is attached because this benchmark records claims only and the inherited score belongs to a different source.

## Attribution and reproducibility

Direct unpromoted contributions are credited to Saviour1001 (PR1211 composition), hybridnoise (PR1137 field additions), terrapinelf (the PR1088 compact composite and its lineage), fkiene (K32 correction technique), mitchuski (material inherited SHA/parity work), and Akashneelesh (material inherited recoding/pipeline work). We also preserve the historical notes crediting DrCleverHans, Babbaragga, dun999, Meganpark980320, ercumentyildirim, EvanYan1024, owizdom, DPZZxlz, jacklightChen, and the promoted contributors. All source license notices and COPYING remain intact.

The implementation is reproducible by checking out `5014ca5064da1fe2c577bbbf006aa948137c45f1` and selecting `candidates/subset/`. SOURCE-MANIFEST.json records the SHA-256 of each packaged file except itself. No generated binaries, local diagnostics, private paths, credentials, or GPU rental configuration are included. All changes relative to the latest shared harness are confined to the subset candidate directory.

Sources: [PR1211](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1211), [PR1137](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1137), [PR1088](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1088), [PR1134](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1134), and the [NVIDIA redistributable manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json).
