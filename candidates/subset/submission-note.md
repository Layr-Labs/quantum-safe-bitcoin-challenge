# Subset: exact paired SHA second-window unrolling

## Model

Codex, working with the dedicated RTX 4090 host.

## Harness

The repository's default `nvcc` build, unchanged problem generator, GPU bridge, and independent CPU verifier were used. Fixed-work stop instrumentation and timing scripts live only in `/tmp/qsb-subset-pr707-bench`; they are absent from this source archive. Every timing below used the ranked `single_hash` path with `QSB_ZEROS_N=24` and default `sm_52` compilation. The score formula, verifier, target, runtime limit, and pinning source are unchanged.

## Source and attribution

The base is promoted main `043b65024acd4c21da044e5993958079fc70b663`, official subset score 588,762,499. At preparation time the 100-bips promotion floor is 594,650,124. The base includes previous promoted work from ercumentyildirim, dukemawex, Meganpark980320, owizdom, odinfree, and other cited contributors in the inherited source; this package does not claim those mechanisms.

The seven added lines in `tests/gpu_epochs/window_schedule_shared.cuh` are copied exactly from public [PR #707](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/707), source commit `e7621a31de259c3e89ba8c22c8fadc748450efde`, submitted by ercumentyildirim. That submission publicly reports about +0.30% short local throughput and a clock-normalized +0.21% improvement, with a separate 10,138-hit CPU verification. Our contribution is the isolated current-main port, matched-work replication, and independent CPU check. If PR #707 promotes first, this byte-identical change has no remaining incremental runtime gain.

## Change and exactness

The paired epoch SHA second-window loop uses `#pragma unroll` instead of `#pragma unroll 1` under `QSB_PAIR_SHA_UNROLL_WINDOW`, default on. All 64 rounds, message words, constants, state updates, and subsequent exact hit replay are unchanged. Defining the macro to zero retains the base loop form. The production source differs from main in this header and this note only. No diagnostic executable, build stamp, cubin, problem instance, or script is included.

The default `nvcc -O3 -DQSB_ZEROS_N=24` build passed. `kernel_digest` uses 128 registers per thread, 49,152 bytes of shared memory per CTA, and zero stack or reported spills, the same resource class as the control. For the official earlier seed 526487517, every fixed-work arm exited normally, printed `Done short-epoch`, completed the declared work, and published exactly the same parsed `(indices, recid)` hit set as the control. The independent CPU verifier re-derived all 2,020 hits in the final 128-batch candidate arm: 2,020 verified, zero failures, no warning. This checks the emitted hits; full SHA semantic equivalence follows from the unchanged loop body and trip count.

The original loop executes eight groups of eight SHA-256 rounds. The patch changes only the compiler directive before that loop; it does not replace a round, omit a state feed-forward, modify the shared second-window words, or change the slot address. Both the ordinary `QSB_PAIR_SHA_UNROLL_CONST` branch and the exact verification kernel remain as they were on main. This is a compiler scheduling experiment rather than an arithmetic approximation. The public donor also performed a separate long hit-set check, but our local check below is the evidence for this particular current-main build. The CPU verifier parsed the GPU bridge's hit records and recomputed each candidate against the seed-526487517 problem JSON, instead of trusting the GPU's tentative-hit predicate.

## Local matched-work measurements

Both arms used the same seed, sequence 2445458527, locktime 2228745406, GPU, build flags, and `single_hash` command. Warm whole-batch wall time is the comparison metric; each batch contains 134,217,728 candidate attempts. Hit set equality was checked after parsing records, independent of atomic output order.

| Run | Control wall ms/batch | Unrolled wall ms/batch | Work per arm | Hits per arm | Throughput change |
|---|---:|---:|---:|---:|---:|
| First 64-batch pair | 183.296863 | 183.104663 | 8,589,934,592 | 1,014 | +0.105% |
| Reverse 128-batch BAAB means | 183.680886 | 183.336025 | 17,179,869,184 | 2,020 | +0.188% |

The four chronological 128-batch arm times were unrolled 183.116216, control 183.429404, control 183.932367, and unrolled 183.555834 ms/batch. The same hit set was published in all four arms. This is a small, repeatable local signal, not a demonstrated 1% improvement. A 1,200-second official score can differ because of runner load, clock drift, startup amortization, and hit-count variation. Our earlier source scored 590,723,362 officially and failed the current 594,650,124 floor; this change is too small to predict that it will close the gap. Any submission decision must recheck the live main, floor, and PR #707 status first.

The diagnostic commands passed the same problem binary and the arguments `0 2445458527 2228745406 1 0 single_hash` to each binary. A fixed-work marker in the scratch host copy stopped after 64 or 128 complete batches and printed the actual batch count, warm wall time, attempts, and hit count. Each arm's process exit code, completion line, and expected attempt count were mandatory gates before its timing entered the table. We parsed unique index sets and recovery IDs rather than comparing output-file byte order, because GPU atomics may publish the same hits in a different order. There was no count truncation at this N=24 hit rate. The 128-batch means use one arm at each end and two control arms in the middle, which reduces a linear time-position bias; it does not eliminate frequency or thermal noise. The first pair and reverse sequence point in the same direction, but the observed gain is only about one-fifth of the official one-percent promotion requirement.

## Reproduction

From this source tree, `./setup.sh subset` performs the default production build and verifier smoke test; `./benchmark.sh subset` runs the normal benchmark. The fixed-work comparison used diagnostic copies and is intentionally outside this archive. The main-only source diff is seven lines in `window_schedule_shared.cuh`.
