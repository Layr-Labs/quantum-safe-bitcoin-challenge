# Pinning: isolate exact GLV coefficient arithmetic on the promoted four-hot grinder

## Candidate, attribution, and scope

This submission is for the **pinning** track only. Its base is the promoted pinning source at `7e95c40c99e57bded233ce57c7f453fbde9fd21c`, associated with fkiene's submission `871963fd-82c8-4c08-99f5-46d4b13f3fce`. That source scored **904,971,814 verified candidates per second** with **129,635 independently verified hits**. Its four-hot fixed-base table came from 0xCramJam's earlier proposal and the ports and checks by Saviour1001 and fkiene. The field operations, search pipeline, SHA code, exact publication gate, and verifier contract remain inherited work.

The new active change is a selective port of the exact signed-GLV coefficient arithmetic published by ItlaStudent in [pinning PR #1264](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1264), commit `d6ee1d8a37ed6f8a74bba42e4878c9825b4022a7`. That public note also credits earlier lean-multiply work by kaankolcu and Portablelle. The port was made to the promoted **FOUR_HOT** source without importing PR #1264's older table geometry. The three active switches in `GLVScalar.cuh` are `QSB_GLV_LEAN=1`, `QSB_GLV_HIGH10_HI=1`, and `QSB_GLV_ROUND_CC=1`.

This is an isolation experiment following the separate combined chain-pipe-plus-GLV submission `64cf37f1-e704-45e9-ae02-3e3fe34413ec` (B). B was officially verified at **906,504,249 candidates per second** with **129,857 verified hits** over **1,201.6706 seconds**, but rejected because its 0.17% score increase fell short of the required 1%. The C archive sets `QSB_CHAIN_PIPE=0`, so the next-record overlap branch is excluded from the ranked build. The source retains the disabled helper and the small `qsb_load_glv` code-load refactor from that experiment; it does not claim that the complete `pinning.cu` bytes match the promoted file. `QSB_CODEX_DRAW_20260924_C` is an unreferenced source marker for this archive and is intended to have no runtime effect.

All submitted source changes are under `candidates/pinning/`. The benchmark specification, harness, setup command, problem generator, score rule, verifier, and all other tracks are untouched. The note records the candidate that is actually built, including the disabled scheduling code, so the score is not misdescribed as a byte-identical redraw of the promoted source.

## Mechanism and unchanged table geometry

The promoted source decomposes each of two signed GLV scalars into six fixed-base terms. The twelve table records enter one XYZZ chain with eleven mixed additions. `QSB_BIGTBL=1` and `QSB_FOUR_HOT=1` retain shifts `[0, 18, 37, 55, 73, 100]` and record counts `[262144, 262144, 131072, 131072, 67108864, 85279885]`. Each record is 64 bytes; the first four banks occupy exactly **48 MiB** of the persisting-L2 prefix, and the full allocation is **153,175,181 records**, or **9,803,211,584 bytes**. The two signed components share these physical records. The bounded top center, recoder, builder, and host spot check remain the promoted source's versions.

The GLV coefficient change concerns the rounded-reciprocal scalar split before those table records are selected:

1. `QSB_GLV_LEAN=1` expresses exact 32-by-32-bit products and multiply-adds with `mul.wide.u32` and `mad.wide.u32`. The high-product accumulation uses the add carry flag to count overflow. The original C arithmetic remains behind the off switch.
2. `QSB_GLV_HIGH10_HI=1` uses only the high words needed from the tenth diagonal of the coefficient product. Its omitted low terms have a proved nonnegative bound; the bit-383 rounding guard is widened to preserve the result. Boundary cases still fall back to the full exact coefficient calculation.
3. `QSB_GLV_ROUND_CC=1` folds the final rounding carry with PTX add-carry instructions. Its original exact C form remains available behind the off switch.

These are source-level arithmetic and instruction-selection changes, not a new GLV decomposition, relaxed correctness criterion, table size, or point-add count. The fallback coefficient calculation and the remainder of the scalar split are retained. No approximate arithmetic is used to justify publishing a hit: the inherited host gate re-derives nominations with OpenSSL, and the official verifier independently checks published records. The host gate cannot recover a genuine hit that a faulty GPU path omitted; a fixed-seed hit-set differential was therefore part of local validation.

Although `pinning.cu` still contains `qsb_pointadd_chain_pipe`, its body and loop branch are guarded by `#if QSB_CHAIN_PIPE` and are excluded at the default value zero. The active mixed-add path follows the original loop. The source-level extraction of `qsb_glv_code` and `qsb_load_glv_code` is retained to make the switchable branch buildable. No asynchronous copy, cache hint, new native cubin loader, table re-cut, or occupancy reservation is enabled. The inherited `pinning_sm89.cubin` sidecar is not loaded by this source.

## Exactness checks and local timing

The port compiled with the setup-equivalent `nvcc -O3 -DQSB_ZEROS_N=24` build. The coefficient oracle in `candidates/pinning/test_glv_coeff.py` compared the adapted coefficient screen against full exact products for 300,818 inputs per reciprocal and both high-diagonal configurations, totaling 601,636 value comparisons; all matched. This tests boundary and random scalar cases, but it alone does not exercise the complete GPU recovery and search path.

On one generated fixed problem, the promoted parent, the B combination, and this C GLV-only configuration each produced **exactly the same 4,533 unique `(sequence, locktime, recid)` hits** through the first 30 complete sequences. The table builder's independent OpenSSL point spot check also passed. The fixed-problem comparison covers the active coefficient changes and the surrounding GPU path; it is not a proof over all inputs or a replacement for the official verifier.

The local RTX 4090 direct-run progress meter at the 30-sequence checkpoint gave the following **matched warm-condition observations** around 40–41 °C:

| Configuration | Progress rate at sequence 30 |
|---|---:|
| Promoted FOUR_HOT parent | 911.8 million searched candidates/s |
| B: GLV arithmetic plus chain-pipe | 909.9 million searched candidates/s |
| C: GLV arithmetic, chain-pipe off | 914.4 million searched candidates/s |

C was about **0.3%** faster than the parent in this small matched comparison and about **0.5%** faster than B. A separate cooler C start around 35 °C reached 921.7 million/s at sequence 30. These are direct kernel progress rates over an early interval, affected by temperature and scheduling; they are **not** verified 1,200-second official scores. The observed margin is below the 1% promotion requirement and is too small to establish a robust speedup from these runs alone. This submission tests whether the isolated arithmetic port helps on the official runner without B's scheduling branch.

A second C binary compiled with the ranker's unmodified default `nvcc -O3 -DQSB_ZEROS_N=24` flags, which produce a compute-52 PTX path on this local toolchain, also passed the table check and reproduced the same first-30-sequence hit count. Starting near 40 °C, it printed 913.5 million/s at sequence 30, close to the separately built native sm_89 C binary's 914.4 million/s at the same checkpoint. This is a build-path and correctness cross-check, not a ranked throughput prediction.

The literal `yukon run --track pinning` command was attempted earlier in this checkout after `yukon setup --track pinning`, but the local machine lacks the organizer's bridge executable at `/opt/starkware-challenge/bench-exec.sh`. The local fixed-seed tests used the supported `QSB_GRINDER=cmd:python3 harness/gpu_wrap.py --src candidates/pinning/pinning.cu` path and direct compiled-binary runs. The official bridge-equipped RTX 4090 runner remains authoritative for score and correctness.

## Ranking interpretation and reproduction

The ranker uses an N=24 gate and a fixed window of about 1,200 seconds. Its score is based on independently verified hits and actual elapsed time, not the maximum rate printed by a GPU progress line. The current promoted score at preparation time was 904,971,814; its 100-basis-point promotion floor is approximately **914,021,533** verified candidates per second. B's verified score of 906,504,249 did not change that frontier. Another public submission may change the score and floor while C is in flight. Problem seeds, finite hit counts, runner speed, startup cost, and thermal state vary between draws. A favorable official result would validate this package's score, but a single run would not isolate a subpercent causal effect without paired measurements.

The prior combined B package was submitted with the same three GLV coefficient switches enabled and `QSB_CHAIN_PIPE=1`; C changes the scheduling default to zero while keeping the coefficient layer. Its official result should therefore be read alongside both B and the promoted FOUR_HOT parent. B's rejection does not settle whether the coefficient arithmetic alone is faster, because its chain-pipe path changes scheduling and its one official seed has finite hit-count variation. This C submission isolates that active coefficient layer for an independent official measurement.

To reproduce the submitted source on a bridge-equipped runner, use the pinning track's standard commands:

```sh
yukon setup --track pinning
yukon run --track pinning
```

The setup build uses `nvcc -O3 -DQSB_ZEROS_N=24` and links OpenSSL and libm. Check the live defaults in `GLVScalar.cuh` and `pinning.cu`, then verify all emitted hit records through the unchanged official verifier. The generated executable, build stamp, local benchmark output, and hit files are not intended to be in this submission archive. The public result should report verified score, hit count, elapsed time, and self-reported rate separately, with the official promotion decision as the outcome.
