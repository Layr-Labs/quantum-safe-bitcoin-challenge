Model: GPT 5.6 Sol
Harness: Codex

# Subset: PR871 composite with exact parity-window recovery

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
