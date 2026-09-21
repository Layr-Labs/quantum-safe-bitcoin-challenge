Model: GPT 5.6 Sol
Harness: Codex

# Subset: PR891 parity window, x-isomorphic recovery, fused root scale, SHORT_CARRY6, and speculative first-fold carry cut

## Additional exact recovery mechanisms

This package starts from the exact PR891 candidate source at local commit `ce248ca0d493f9db5c177fde690f3a38ebbe1695`. It adds two measured exact recovery changes and two independently measured speculative-filter carry truncations. No harness, scoring, problem, sibling-track, fixed-work diagnostic, or generated artifact is changed.

For every fresh problem recovery point `R=(xR,yR)`, the host chooses the curve isomorphism `x'=u^2*x, y'=u^3*y` with `u^2=+/-1/xR`. Since the secp256k1 field prime is 3 modulo 4, exactly one sign is a square, so transformed `xR'` is always +1 or -1. The hot `xR*ZZ` field multiplication becomes a copy or modular negation. The small host-built fixed-base ladders and recovery point are scaled into the same isomorphic curve; the million-entry table is then built and spot-checked through the existing GPU path.

The transformed preparation numerator and denominator scale by `u^9` and `u^12`. Scaling the batch inverse root by `1/u` therefore gives each leaf the factor required to recover the original affine slope after the existing `n*ZZ*inverse` product. The post-recovery path reloads original `xR,yR`, so affine x, y parity, compressed pubkeys, hashes, and hit records remain unchanged.

The second change folds that once-per-CTA `1/u` scale into the active four-lane zinv32 extended-GCD coefficient initialization. The branch named `HM43_WARP_ROOT` actually calls `zi_inverse_quad`; its u/v decision state is unchanged, while the input coefficient starts at `1/u` instead of one. This returns `(1/u)/root` directly and removes a full field multiply plus its call frame. The fixed-exponent fallback applies the same factor explicitly. `QSB_ISO_FUSED_ROOT_SCALE=0` restores the separately measured x-isomorphism implementation.

## New correctness evidence

An independent host differential generated 64 fresh problems and 64 valid fixed-base points per problem: 4,096 candidates and 8,192 recovery outputs covering both transformed x signs. Original and transformed paths matched affine x, y parity, compressed pubkey bytes, and SHA-256 exactly; 1,024 independent transformed group-law closure cases also passed. The GPU-built transformed table passed its OpenSSL spot check.

The shipped zinv32 fusion was compared directly with OpenSSL big integers for eight independent scale constants, including 1 and p-1. Normal bounded zinv32 produced 32,768/32,768 exact `(1/u)/root` values. A forced fixed-exponent fallback produced 4,096/4,096 exact values. Full inverse-tree checks had zero wrong residues; four cases per suite used the existing documented `expected+p` lazy representation.

Every timed arm below completed exactly 8,589,934,592 candidates and emitted the same normalized 1,096-record hit set, symmetric difference zero, SHA-256 `1b5c727cb1380c6ae0438e74011582cb652620bc0d26b7d14ccea59075572c06`.

## New matched measurements

PR891 versus x-isomorphism, fixed64 A/B/B/A on the same N24 problem, measured wall means 179.482930 to 179.041067 ms/batch: **+0.246794% completed-work throughput**, with both adjacent comparisons positive (+0.190606%, +0.302978%). Digest throughput improved +0.248102%.

The separate-root x-isomorphism versus fused-root x-isomorphism measured wall means 180.0506375 to 179.7881325 ms/batch: **+0.146008%**, again with both adjacent comparisons positive (+0.081355%, +0.210594%). Digest throughput improved +0.145602%.

The two same-problem matched ratios compose mechanically to **+0.393162% over PR891**. This number is a projection from two balanced comparisons, not a direct official score. Applied to the independently measured PR891 center projection of 605,982,191 candidates/s, it gives about 608,364,685 candidates/s; the ranked run remains the arbiter.

## New build evidence

With CUDA 12.8 at the organizer-default sm52 target, fusing the root scale changes the x-isomorphism digest from 128 registers, 49,152 bytes shared, a 40-byte call frame and zero spills to the same registers/shared memory with zero frame and zero spills; SASS falls from 50,202 to 50,130 instructions. At native sm89 it keeps 128 registers, 49,152 bytes shared and the same 16-byte-store/12-byte-load spill class, reduces the frame from 56 to 16 bytes, and reduces SASS from 21,656 to 21,600 instructions.

## SHORT_CARRY6 speculative filter mechanism

`QSB_SHORT_CARRY6` shortens four 64-bit `K=2^32+977` correction tails in every deferred-Y mixed add: the fold of `Y2+Yoff` and the folds of `U2-X1`, `S2-Y1`, and `V-X3`. It keeps the correction in limb 0 and omits only its rare carry or borrow into limb 1. The chain executes 13 such mixed adds, so this removes 52 propagation instructions per candidate. `QSB_SHORT_CARRY6=0` restores the fused-root control.

This change is deliberately confined to `qsb_filter_point_add<true>` in the speculative filter. `qsb_replay_chain_trial` uses the separate unchanged `chain_replay_field.cuh` arithmetic. `kernel_verify_pair_hits -> qsb_pair_verify_candidate -> qsb_k2s_front_exact` replays through that exact path before a record can be published. A shortened-fold event can therefore lose a tentative proposal; it cannot authorize an incorrect record.

The dropped propagation condition was modeled independently over 2,000,000 random four-limb states and 14 constructed threshold cases. The random comparison had zero differences. The directed cases locate the complete exceptional interval: when the correction flag is active, the limb-0 word lies in an interval of exactly `K` values out of `2^64`; the short result then differs from the full result by exactly `2^64` across the limb boundary. Under a uniform-low-word model this is `K/2^64 = 2.328306966171e-10 = 2^-31.999999672` per fold. A conservative 52-fold union without discounting the correction flag is `1.210719622409e-8 = 2^-26.299559954` per candidate. The interval and directed difference are exact; the probability statement depends on the low-word distribution.

## SHORT_CARRY6 matched measurement and build evidence

The fused-root package and SHORT_CARRY6 were compared on the same public N24 problem with sequence `2445458527`, locktime `2228745406`, 64 fixed batches per arm, first three batches excluded, and balanced A/B/B/A order. Every arm completed exactly 8,589,934,592 attempts and emitted the same normalized 1,096-record hit set, symmetric difference zero, SHA-256 `1b5c727cb1380c6ae0438e74011582cb652620bc0d26b7d14ccea59075572c06`.

| order | arm | warm wall ms/batch | digest ms/batch | exact-replay hits |
| ---: | --- | ---: | ---: | ---: |
| 1 | fused-root A1 | 178.880086 | 177.051881 | 1,096 |
| 2 | SHORT_CARRY6 B1 | 178.198875 | 176.369965 | 1,096 |
| 3 | SHORT_CARRY6 B2 | 179.010642 | 177.176485 | 1,096 |
| 4 | fused-root A2 | 179.892440 | 178.057347 | 1,096 |

Mean wall time fell from 179.386263 to 178.6047585 ms/batch, a **+0.437561% completed-work throughput improvement**. Both adjacent comparisons favored SHORT_CARRY6, by +0.382276% and +0.492595%; digest throughput improved **+0.442029%**.

At the organizer-default sm52 target, both variants retain 128 registers, 49,152 bytes shared memory and zero frame/spill; digest SASS falls from 50,130 to 50,112 instructions. At native sm89, SHORT_CARRY6 retains 128 registers and 49,152 bytes shared memory while removing the fused-root package's 16-byte frame, 16 bytes of spill stores, and 12 bytes of spill loads; digest SASS falls from 21,600 to 21,504 instructions.

Composing the measured SHORT_CARRY6 ratio with the prior x-isomorphism plus fused-root ratio gives **+0.832443% over PR891**. Applied mechanically to the independently measured PR891 center projection of 605,982,191 candidates/s, this gives about **611,026,650 candidates/s**. This is a composition of same-problem balanced ratios, not a direct official score; the ranked run remains the arbiter.

After this package was prepared, the exact PR891 base source completed its ranked Action. Digest-verified artifact `10633867510` scored **608,061,438** verified candidates/s on seed 216,208,103, with 87,061 independently verified hits over 1,201.0638 seconds; the artifact ZIP SHA-256 is `739980080a08c98274b62b0a17032ff6e9968dc39917f84f304cb3cd957b08b7`. Yukon artifact ingestion then failed with a platform `ENOSPC` error, so that valid floor-clearing result was not entered as a promotion. Applying the measured +0.832443% composition factor to this official base draw gives a mechanical **613,123,206** projection. That remains a projection for this composed source rather than an official score.

## Speculative first-fold top-carry cut

The only new runtime delta relative to the clean SHORT_CARRY6 package is in `hit_filter_field_sc.cuh`. Eleven inline PTX first-fold instances, covering speculative multiply and square paths, omit the carry bit above 256 bits after adding the even high-product limbs times 977. The next 32-bit word is formed from the shifted odd-product word and its own carry. This removes a dependency between two carry chains. The separate `chain_replay_field.cuh` and exact pair-hit verifier are unchanged; no record can be published without exact replay. A missed speculative proposal remains possible.

For one first fold, the omitted bit is one exactly when `r3 + 977*x14 + c3 >= 2^64`, where `r3` is the top 64-bit low-product word, `x14` is a 32-bit high-product word, and `c3` is the preceding carry bit. The exceptional interval has at most `977*(2^32-1)+1 = 4,196,183,047,216` top-word values. If `r3` were uniform conditional on the other words, its probability per fold would be at most `2.274755388e-7`. This distribution assumption is not a universal recall proof; a hit discarded by this speculative filter cannot be recovered by replay. The direct measurements below check finite candidate sets and exact published records.

A directed CUDA differential confirms the approximation and its exact delta. For canonical multiply operands `a=3` and `b=floor((2*2^256-1)/3) < p`, the old speculative result is `0x2000007a0`, the shortened result is `0x1000003cf`, and their difference is `K=2^32+977`; neither path sets its `bad` flag. A separately constructed canonical square with low product `2^256-7` produces the same `K` difference. These vectors are a deliberate disclosure of the exceptional branch, not evidence of universal recall.

At the organizer-default N24 build, the digest keeps 128 registers/thread, 49,152 bytes shared/CTA, zero frame and zero spills on sm52 and sm89. Complete digest SASS changes from 50,112 to 50,118 instructions on sm52 and from 21,504 to 21,464 on native sm89; the native select count falls by 41. The static count alone does not predict the measured dependency benefit.

Two independent N24 problems were compared with the ranked `single_hash` argument, the same fixed 64 batches per arm, the first three batches excluded, and A/B/B/A order. Every arm of the first problem searched 8,589,934,592 candidates and published the same 1,096 exact-replay records; all four sorted hit sets have SHA-256 `f57aeac25e8437ad7b1ea0473dd7e9a4339ee70725c11726bdf882edbe4f7023`. Every arm of the second problem searched the same number and published the same 998 records; all four sorted hit sets have SHA-256 `7cfb3ca7ce9cc95756fd0941c1722cb347250fd77ca56ae92f649aa3a7fe5610`.

| problem SHA-256 prefix | control wall ms/batch A1, A2 | candidate wall ms/batch B1, B2 | completed-work gain | adjacent gains |
| --- | --- | --- | ---: | --- |
| `02dbb614` | 178.499307, 179.325223 | 178.334360, 178.449169 | +0.291774% | +0.092493%, +0.490926% |
| `bb68c4d2` | 179.345548, 179.628982 | 178.541767, 178.756520 | +0.469144% | +0.450192%, +0.488073% |

The pooled equal-work wall ratio is **+0.380523%**, with all four adjacent comparisons positive. Composing it mechanically with the prior +0.832443% package ratio and the digest-verified PR891 official draw gives a **615,456,279** candidates/s projection. It is not an official score; the ranked run and finite-hit recall are the remaining uncertainties.

## Source and scope

This package starts from the PR871 source package, local commit `347394e9f6567d33b405621fccc12dc8a070c62f` and public commit `ac53faff38a192d78e9a63f839b6e97c64dc2b66` (submission `f8197832-fb79-4765-8b17-70034edaa492`). PR871 itself starts from public PR #854, commit `00784176f68399f090e093135809e8a7a91facce`, and ports the two exact, default-on mechanisms published in PR #868, commit `0d1b013accb3f691432450cd3b048cba955afa7e` (submission `174476ce-4b4f-4166-8253-4dff18eff8d6a`).

The only new runtime change here ports the parity-window core from public PR #885, commit `3e166ba462b03affb785852589606398b10f32b2`, into the two parity-only products in the subset negfold recovery tail. It adds one header and nine guarded lines in `pair_shared.cuh`; the kill switch `QSB_K2S_PARITY_WINDOW=0` restores the PR871 source path. No binary, generated problem, benchmark output, harness change, scoring change, or sibling-track change is included.

PR868's optional H0 scheduling experiment remains excluded. The inherited PR854 H0 gate and all exact replay/publication code remain unchanged.

PR854 contributes its existing `QSB_NEGFOLD_PARITY` and `QSB_SHORT_CARRY4` filter stack. Its byte-identical predecessor, submission `8cd86ac7-1c0b-43ab-9c09-360907f06201`, scored **600,048,504** verified candidates/s on seed 195,057,307 with 85,915 independently verified hits. That was about +0.695% over the 595,907,916 promoted crown, but below the then-current one-percent floor of 601,866,996.

## Exact mechanisms added from PR868

`QSB_EPOCH_FAST` removes producer-only work while preserving every epoch descriptor bit for bit. It emits big-endian SHA words directly instead of byte stores plus byte swaps, and publishes each group's already-known omission prefix and rank so the per-epoch kernel replaces repeated combination unranking with one coalesced group-index load. This changes neither the candidate SHA/EC digest nor the candidate family.

`QSB_SE_WINDOWS=128` selects 128 valid omission triples from the same `C(13,3)` pool. Those triples use eight distinct first-block schedules instead of the 54 used by the 256-window layout. The digest CTA remains 256 threads, 49,152 bytes of shared memory, and two resident CTAs per SM; warps 0..3 process one epoch pair and warps 4..7 process a second epoch pair. Each thread still represents exactly one `(epoch, window)` candidate, and the hit tag is decoded as `epoch * 128 + lane`.

The ranked family contains exactly `C(137,6) * 128 = 1,051,964,508,672` distinct candidates. A 1,200-second run at 600M/s needs about 720 billion candidates, and even 604M/s needs about 725 billion, so the reduced family retains substantial headroom without wraparound.

## Exact parity-window mechanism added from PR885

The negfold recovery needs only the low parity bit of each of two products, `p1*m1 + yR` and `p2*m2 + yR`; their full 256-bit residues are discarded. The parity-window helper evaluates the 27 cross-products that determine that bit after secp256k1 reduction, instead of running two complete 8-by-8-limb field multiplications and reductions. Its bounded carry test identifies the rare rows where the partial window is insufficient and sends those rows through PR871's unchanged `qsb_fmul` plus `qsb_fadd` path. In a 16,777,216-row CUDA differential, 16,777,207 rows used the fast path, nine used the fallback, and the result disagreed with the inherited path zero times.

The helper's contract is the one exercised by this recovery tail: arbitrary full-width product operands, a canonical nonzero `yR` addend, and a requested sign bit. It is filter-only. Every tentative hit is still recomputed by the unchanged exact recovery kernel before publication.

## PR871 base local matched measurement

The PR868 composition was compared with PR854 on an RTX 4090 at the organizer-default `nvcc -O3 -DQSB_ZEROS_N=24` target, official subset seed 526487517, ranked arguments, fixed 64 batches per arm, and balanced A/B/B/A order. Each arm completed exactly 8,589,934,592 candidates. A is PR854 and B is this composite.

| order | arm | warm wall ms/batch | epoch ms | first-state ms | digest ms | exact-replay hits |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | PR854 A1 | 184.237451 | 1.061899 | 1.791768 | 181.256810 | 1,014 |
| 2 | composite B1 | 183.158792 | 1.150580 | 0.539826 | 181.341217 | 1,048 |
| 3 | composite B2 | 183.213873 | 1.150842 | 0.539791 | 181.397150 | 1,048 |
| 4 | PR854 A2 | 184.193130 | 1.065747 | 1.790725 | 181.208169 | 1,014 |

Mean wall time was 184.215291 ms for PR854 and 183.186333 ms for the composite, a **+0.56170% completed-work throughput improvement**. Both adjacent comparisons favored the composite, by +0.58892% and +0.53449%. The expected trade is visible directly: first-state construction falls by about 1.25 ms/batch, while the larger epoch producer rises by about 0.09 ms/batch and digest time remains essentially flat.

The fixed-work timing code and its identical chain-side diagnostic overlay existed only in isolated test copies and are absent here. Public PR868 separately reported +0.703% for `QSB_EPOCH_FAST + QSB_SE_WINDOWS=128` on its e876-derived source, with two-seed verification; that is donor evidence rather than a result reproduced by this package.

## Parity-window matched measurement

The new parity-window port was compared directly against PR871 with identical diagnostic source, official subset seed 526487517, organizer-default N24, and fixed 64 batches in A/B/B/A order. Every arm completed exactly 8,589,934,592 candidates and published the same byte-identical 1,096-hit file.

| order | arm | warm wall ms/batch | digest ms/batch | exact-replay hits |
| ---: | --- | ---: | ---: | ---: |
| 1 | PR871 A1 | 180.463236 | 178.564401 | 1,096 |
| 2 | parity window B1 | 179.584988 | 177.684009 | 1,096 |
| 3 | parity window B2 | 179.617404 | 177.718394 | 1,096 |
| 4 | PR871 A2 | 180.906521 | 179.008001 | 1,096 |

Mean wall time fell from 180.684879 to 179.601196 ms/batch, a **+0.60338% completed-work throughput improvement**. Both adjacent comparisons favored the parity window, by +0.48904% and +0.71770%; digest throughput improved by +0.61058%. Fixed-work timing and raised hit-cap diagnostics are absent from this source-only package.

## Correctness checks

The 128-window layout searches a different valid subset of the full window family, so its complete fixed64 hit set is not expected to equal PR854's 256-window hit set. Correctness was checked in four ways:

- The parity-window CUDA differential tested 16,777,216 random and directed rows against the inherited full-product path: zero mismatches, 16,777,207 fast rows, and nine exact-fallback rows.
- All four PR871/parity-window fixed64 arms produced the byte-identical 1,096-record hit file, raw SHA-256 `8dda10dcf41d84faca98fb0da3de20763709a3c15cb4f002da8ae3cc3bba7c4f`; the normalized set SHA-256 is `1b5c727cb1380c6ae0438e74011582cb652620bc0d26b7d14ccea59075572c06`.
- The unchanged PR871 base previously passed independent CPU verification of **1,048 / 1,048** fixed-work hits, zero failures, when its 128-window mechanism was introduced. The parity-window ABBA additionally traverses the unchanged exact replay for every one of its 1,096 published records.
- All 128 selected windows are a subset of PR854's 256 windows. Restricting PR854 to those windows and restricting the composite to PR854's first 33,554,432 epochs produced **541 versus 541** normalized hits with symmetric set difference **zero**.

The mapping also has a direct accounting proof: a 256-thread block covers four consecutive epochs as two warp-aligned 128-lane epoch pairs; full launches therefore cover the same 134,217,728 candidate count as PR854 without overlap or gaps.

## Build and risk

The clean organizer-default sm52/N24 build retains the ranked digest resource class: 128 registers per thread, 49,152 bytes shared memory per CTA, one barrier, zero stack, and zero spills. The two fast epoch producer kernels use 48 registers and zero spills. The matched control and candidate also had the same resource class. The complete digest SASS contains 49,304 instructions in control and 49,926 in the candidate because the cold full-product fallback remains in the binary; the guarded hot path avoids the two full products. `QSB_PROBLEM_SEED=777 ./setup.sh subset` and its CPU verifier smoke pass on the clean package.

The two PR868 mechanisms and the guarded PR885 parity-window calculation ported here are exact under their stated input contracts. The inherited PR854 `QSB_SHORT_CARRY4` speculative filter remains approximate and can theoretically lose a rare true proposal; exact replay prevents false records from being published but cannot restore a missed proposal. The prior 85,915-hit official run, the local CPU verification above, and the common-domain equality are finite evidence rather than a universal recall proof.

## Attribution

The promoted e876 base was submitted by **jacklightChen**. Its H0-only gate credits **Saviour1001**, and its paired preparation lineage credits **owizdom**, **DPZZxlz**, and **fkiene**. The negfold-parity research was published by **fkiene**. `QSB_SHORT_CARRY4` traces to **Meganpark980320** PR #654. **dun999** assembled and measured the exact PR854 negfold + carry4 runtime. **ercumentyildirim** authored and published the PR868 fast epoch producer and 128-window/two-pair CTA mechanism. **EvanYan1024** published the PR885 parity-window mechanism; its public validation commit also credits **terrapinelf** and **DrCleverHans**. All inherited source and license notices remain intact.

This subset port, vector differential, balanced local comparison, and packaging were performed with **GPT 5.6 Sol** using **Codex**. Co-authorship should retain dun999, fkiene, Meganpark980320, and ercumentyildirim for PR871 and add EvanYan1024 for the parity-window mechanism.

The speculative first-fold carry cut, two-problem comparison, and final package update were performed with **GPT 5.6 Sol** using **Codex**.
