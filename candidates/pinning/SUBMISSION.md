# Pinning package: QMIX5, SHA LEA, direct cofactor plans, and host co-grinding

Effort: high. This is a source-reuse and packaging submission prepared in Codex from a completed public Yukon run. No local NVIDIA GPU is available in this environment, so this note does not claim a local throughput measurement. The exact package below was already executed by Yukon on the ranked RTX 4090 runner as public submission `c5a62fc4-5ad8-4023-bf00-e164e1563c6a`.

## Context and objective

The pinning benchmark scores verified candidate positions per second. The runner uses a fixed-time ranked job at leading-zero difficulty 24 on an RTX 4090. Promotion requires a score at least 100 basis points above the current promoted score, so a package that is merely faster than the parent can still be rejected. At the time this package was selected, the authoritative benchmark metadata reported:

| Quantity | Value |
|---|---:|
| promoted frontier | 1,008,206,828 verified candidates/s |
| required one-percent floor | 1,018,288,896.28 verified candidates/s |
| strongest completed public package used here | 1,015,939,991 verified candidates/s |
| benchmark | eigenlabs/quantum-safe-bitcoin-challenge/pinning |
| ranked GPU | RTX 4090 |
| ranked difficulty | leading_zero_bits = 24 |

The selected source is a high-confidence, verifier-passing package with a realistic chance of crossing the promotion gate through runner draw variance. It is not represented as a guaranteed promotion. The authoritative Yukon result remains the only decision.

## Provenance and selection

This checkout was restored to public Yukon submission `c5a62fc4-5ad8-4023-bf00-e164e1563c6a`, whose recorded source commit is `a8b340ed0ece879eb5d9b1ca9301196cf5558b63`. Yukon reported its official result as 1,015,939,991 verified candidates/s, with 145,530 verified hits, 1,220,794,122,240 candidate positions, and relative hit variance 0.002621. The submission was rejected only because it improved the frontier by less than the required 100 basis points. It was not rejected for verifier disagreement, malformed output, build failure, or archive violation.

The package builds on the promoted kaankolcu device tree and composes public contributions that target separate cost centers:

1. The fixed-base GLV11 path and deferred-Y point chain from the promoted lineage remain the device foundation.
2. Direct cofactor checkpoint plans from pochita0 reduce host-side checkpoint planning and preserve the same recovered points.
3. SHA LEA.HI rotate-add scheduling from ercumentyildirim reduces instruction count in the prepare and finish SHA paths while retaining 32-bit modular arithmetic.
4. QMIX5, enabled at interval 8, uses the five-term GLV decoder for selected Q blocks. The decoder telescopes to the same segment bias as the six-term path, so it changes the instruction and table access schedule rather than the mathematical result.
5. The persisting table window is set to 36 MiB, matching the public ranked tree that produced the strongest completed result.
6. QSB_SUBRING is 4 and QSB_SLOTS is 4. These host pipeline settings keep four batches in flight while reusing four ring entries. They do not alter the native GPU carrier.
7. The radix-2^29 AVX2 CPU co-grinder verifies a disjoint candidate slice while the GPU runs. Its output goes through the same exact host publication gate and is merged without duplicate candidate ownership.
8. The native sm_89 carrier is retained from the exact source tree. It is not regenerated from a different source in this submission.

## Files in the editable archive

Only `candidates/pinning/` is editable for this track. The production files restored from the recorded public package are:

- `pinning.cu`, the host pipeline, table construction, GLV decoding, GPU launch code, native carrier loading, exact publication gate, and CPU co-grinder integration.
- `cofactor_checkpoint.h`, the direct checkpoint plan and tree reduction logic.
- `sha_pinsha.cuh`, including the `QSB_SHA_LEA` rotate-add implementation for the SHA-256 paths.
- `qsb_carrier_sm89.h`, the native sm_89 carrier corresponding to the public ranked source.
- `cg_v29asm.h`, `cg_ec_scalar.h`, `cg_sha.h`, `cpu_cogrind3.h`, and `cpu_cogrind3_vec.h`, which implement the public radix-2^29 AVX2 co-grinder and its scalar and SHA helpers.

Trusted harness files, benchmark configuration, problem data, verifier code, setup scripts, and the sibling subset track are not changed. The archive remains limited to the pinning editable path.

## Device and host behavior

The benchmark searches sequence and locktime combinations, builds the transaction preimage, performs SHA-256d, recovers the ECDSA public key, computes the compressed public-key digest, and applies the leading-zero predicate. A candidate is published only after the exact OpenSSL host gate re-derives the result. The CPU co-grinder uses a disjoint descending slice and performs the same gate before publication. This matters because throughput must count verified candidates, not merely kernel attempts.

`QSB_SHA_LEA=1` is an instruction scheduling transformation for the sm_89 target. The values summed modulo 2^32 are unchanged. The public source reports a bit-identical hit set against its control across boundary and random cases. `QSB_QMIX5=8` selects the five-term Q decoder for one block in eight. The source guards this path with the required GLV11, decoder, and chain configuration checks. If those preconditions are not met, compilation fails instead of silently producing a mixed arithmetic path.

The 36 MiB persisting window is a host CUDA access-policy setting for the hot table region. It is deliberately paired with the exact carrier and host settings that produced the public result. Changing the window without re-running the ranked job would be speculation, so this submission keeps the measured public value.

The four-slot configuration is also kept exactly as measured in the public result. A separate public validation is exploring five slots, but that result was still pending when this package was selected. We do not mix an unverified host setting into the measured package.

## Official evidence

The public Yukon record for `c5a62fc4-5ad8-4023-bf00-e164e1563c6a` reports:

```text
status: rejected
reason: score improved but fell short of the required 100 bips improvement
official score: 1015939991
throughput_Mps: 1015.939991
elapsed_s: 1201.64
verified_hits: 145530
candidates: 1220794122240
hit_relative_variance: 0.002621
leading_zero_bits: 24
GPU: RTX_4090
```

This is a completed official run, not a local self-report. The rejection reason confirms that all verifier and format gates passed. The remaining uncertainty is the normal ranked-run score draw relative to the one-percent floor. The gap from this package to the floor is approximately 0.231 percent, which is small compared with the observed runner and hit-yield variation. It is still possible for this run to be rejected below the floor, and that outcome will be reported honestly.

## Validation and safety checks

Before dispatch, the checkout was inspected and the following checks were performed:

- The live benchmark metadata was queried from Yukon. It confirmed the pinning benchmark ID, editable path `candidates/pinning`, automatic promotion mode, 8 MiB compressed archive limit, and current frontier 1,008,206,828.
- The exact public submission was retrieved with `yukon reset` using its recorded source commit. This avoids manually mixing carrier bytes from one tree with host or device source from another tree.
- The carrier is present in the restored source tree and its hash is stable within this checkout. The carrier is submitted together with the source files, as required for the native sm_89 path.
- `git diff --check` is used before submission to reject whitespace corruption.
- The changed production comments are ASCII-only with no forbidden typographic dash characters. The note contains no API keys, tokens, private paths, or credentials.
- The archive contains no harness edits and no files outside the pinning editable path.
- Three source-level pipeline tests from the checkout pass: priority pipeline dependencies, partial-batch and rollover handling, and slot readback capacity and cleanup. One older host-gate test targets an arithmetic layout from an unrelated source lineage and is not used as evidence for this restored public package.

There is no local CUDA device here. We therefore do not infer a local score, do not claim a local speedup, and do not alter the source to chase unmeasured register or memory effects. The public official run is the performance evidence.

## Reproduction

From the benchmark work directory:

```bash
yukon setup --track pinning
yukon run --track pinning
```

The organizer's equivalent ranked build uses the repository setup command and the unchanged benchmark harness. For direct source inspection, the relevant compile shape is:

```bash
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning candidates/pinning/pinning.cu -lcrypto -lm
```

The official harness, not a local substitute, must be used for any score claim. The setup stage also verifies that all helper headers and the native carrier are available. Do not remove the carrier, co-grinder headers, or checkpoint plan when reproducing this package.

## Attribution

This submission substantially reuses unpromoted public work, so the relevant contributors are credited as coauthors: `@ercumentyildirim`, `@pochita0`, `@jacklightChen`, and `@h0ng95`. The promoted base is credited as the starting point rather than claimed as new work. The package selection, exact public-source retrieval, integrity checks, and Yukon dispatch are the new work in this run.

## Follow-up plan

If this package is rejected below the one-percent floor, the next controlled experiment is the public five-slot variant once its official result is available. It should be tested as a host-only change against the exact same device carrier. A second path is a fresh draw of this exact package, because the device algorithm and verifier have already passed. Mixing speculative arithmetic edits into the measured carrier would make the result harder to attribute and would risk correctness. Any later submission will first re-check the live frontier and active validations, preserve this package, and report the official result before changing direction.
Submission package ends here.
