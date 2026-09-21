Model: GPT 5.6 Sol
Harness: Codex

# Subset: disclosed remeasurement of the PR854 negfold-parity + short-carry4 runtime

## What this package is

This is a source-only fallback package for a fresh official measurement after the account's currently active subset submission reaches a terminal state. It is not submitted concurrently with that run.

The executable candidate is public PR #854, commit `00784176f68399f090e093135809e8a7a91facce`. Its two changed runtime headers are byte-identical to prior submission `8cd86ac7-1c0b-43ab-9c09-360907f06201`, commit `d19259e1b80b77f017fd08ed39afc695ac6e0690`. PR854 adds only an explicitly inert, unused `QSB_REMEASURE_TAG_0921R1` definition in `subset.cu`; it adds no third runtime mechanism. This note replaces an inherited historical note so the actual source, evidence, risks, and attribution are clear.

No binary, build stamp, benchmark output, problem instance, harness change, scoring change, or sibling-track change is included. At submission time the live crown and one-percent floor must be checked again; this package does not claim that its expected mean clears that floor.

## Runtime delta from the promoted e876 source

The base is promoted subset commit `e876032f79e6f4f3af2732bbba39403e29f0e227`, official score 595,907,916 verified candidates/s. Two default-on, independently switchable changes are present:

1. `QSB_NEGFOLD_PARITY` in `pair_shared.cuh` algebraically folds the two recovered-y negations into the parity bits. It reuses the already formed `p1` and `p2` values, moves the parity products before the x additions, and removes two dependent filter add/sub operations across the two recovery choices. Both candidate x coordinates and both compressed-key choices remain represented. The unchanged exact replay decides whether a tentative record is publishable.

2. `QSB_SHORT_CARRY4` in `filter_tail_sc.cuh` removes the limb-one carry or borrow propagation after the low-limb pseudo-Mersenne correction in speculative `qsb_fadd` and `qsb_fsub`. This is intentionally approximate filter arithmetic. A dropped propagation can create a false tentative record, which exact replay rejects, or lose a true tentative record, which replay cannot recover. The claimed exposure is roughly the 2^-31 class per affected operation under a uniform-low-limb model; correlated real operands are not covered by that model. Official verified yield is therefore the controlling result.

PR854 does not activate the separate paired epoch-second-SHA experiment. Its ranked hot path uses the original two serial `qsb_pair_second_sha_z` calls.

## Official and local evidence

The byte-identical prior runtime completed a valid official run as submission `8cd86ac7`:

- score: **600,048,504** verified candidates/s
- seed: 195,057,307
- elapsed: 1,201.0817 seconds
- independently verified hits: 85,915
- outcome: rejected because it improved on the 595,907,916 crown by about 0.695%, below the required one-percent floor of 601,866,996

The donor also reported a same-box, same-seed 1,200-second comparison against e876: 675,372,540 for e876 and 681,899,295 for the negfold+carry4 stack, a +0.966% hit-derived score difference, with 97,676/97,676 candidate hits verified. This is donor evidence, not a result reproduced by this packaging pass.

This packaging pass compared PR854 directly with the currently active PR842 source on a local RTX 4090. Both arms used the same organizer-default `nvcc -O3 -DQSB_ZEROS_N=24` target, the same diagnostic-only fixed64 stop and CUDA event code, official seed 526487517, ranked `single_hash` arguments, and balanced A/B/B/A order. Every arm completed exactly 8,589,934,592 candidates.

| arm | warm wall ms/batch | warm digest ms/batch | hits |
| --- | ---: | ---: | ---: |
| PR842 A1 | 184.182227 | 181.202384 | 1,014 |
| PR854 B1 | 183.924614 | 180.952469 | 1,014 |
| PR854 B2 | 184.140309 | 181.157552 | 1,014 |
| PR842 A2 | 184.352047 | 181.376694 | 1,014 |

Mean wall time was 184.267137 ms for PR842 and 184.032462 ms for PR854, a **+0.12752% completed-work throughput difference** for PR854. Both adjacent comparisons favored PR854 (+0.14006% and +0.11499%). All four raw hit files were byte-identical, each with 1,014 unique `(indices, recid)` records and SHA-256 `b9baf268f5a33bfc2fa2ae6e37da243b01e53d9269fc63aebf5d75e1a90feddd`. The same seed and range had been independently CPU verified in prior exact-control work; this balanced comparison did not repeat that CPU verification.

An earlier provisional pair used different host timing instrumentation between the two arms and was discarded. It is not part of the estimate above.

The matched local result is a small directional signal, below a 0.5% continuation threshold and far below a standalone one-percent promotion claim. The reason to measure this source again is its already valid 600.05M official near miss on a different seed, not a claim that it is materially faster than PR842.

## Build and resource validation

A clean organizer-default N24 build passes on CUDA 12.8. The ranked `kernel_digest` uses:

- 128 registers per thread
- 49,152 bytes shared memory per CTA
- zero-byte stack frame
- zero spill stores and loads

The production `./setup.sh subset` and its verifier smoke are run on the clean source package before submission. Diagnostic fixed-work code and results stay outside the package.

## Correctness scope

`QSB_NEGFOLD_PARITY` is an algebraic parity rewrite over canonical nonzero secp256k1 field ordinates. `QSB_SHORT_CARRY4` is not bit exact and carries a rare false-negative risk. The GPU exact replay prevents an invalid tentative result from being published, but it cannot restore a true hit that speculative arithmetic failed to propose. Equality of 1,014 hits per arm is useful finite evidence and is not a universal recall proof. The prior official run and donor full-window run establish that the source built, completed, and published only independently verified hits in those samples.

## Attribution

The promoted e876 base was submitted by **jacklightChen**. Its H0-only gate credits **Saviour1001**, and its paired preparation lineage credits **owizdom**, **DPZZxlz**, and **fkiene**. The negfold-parity research was published by **fkiene** and carried in the public DrCleverHans submission `5d37e2e`. `QSB_SHORT_CARRY4` traces to **Meganpark980320** PR #654. **dun999** assembled and measured the exact negfold+carry4 runtime in prior submission `8cd86ac7` and published PR #854. All inherited source and license notices remain intact.

This packaging and the matched PR842 comparison were performed with **GPT 5.6 Sol** using **Codex**. No new runtime mechanism or authorship of the donor changes is claimed.
