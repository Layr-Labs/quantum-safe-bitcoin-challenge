# Subset: the promoted `521075fe` GPU side and host loop (`QSB_Y_PAIR` chain, completed-slot snapshot) with our host side, re-tuned on Zen 4 EPYCs (100 co-grinder patterns, run-time batch) and terrapinelf's faster co-grinder arithmetic from `a33e04c3` (16-lane key-hash schedule with SHA-NI rounds, single-accumulator IFMA columns)

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 + i7-13700K host (CUDA 12.8.93). AVX-512 code was executed with Intel SDE 9.48 (`-spr`). Our host has no AVX-512.

## Starting point

Every ranked subset run publishes its verified hits, and each hit's window pattern says whether the GPU (the 128-pattern set) or the co-grinder (the other 158) found it, so the two parts of each score separate exactly.

| ranked run | GPU side | host side | GPU M/s | CPU M/s | score |
|---|---|---|---:|---:|---:|
| `d052bc3d` (promoted) | GLV11 P18 | Meganpark980320 lane, spinning host | 643.32 | 48.31 | 691.63 |
| our `a141df2b` | GLV11 P18 (`d052bc3d`'s image) | r7 lane + 9 windows, v3 producer code on floating threads, blocking waits + host core | 634.26 | **62.17** | 696.43 |
| kshitij-hash `8f99a3e9` | GLV11 + `QSB_Y_PAIR` | r7 engine, record placement, blocking waits | 637.13 | 57.47 | 694.60 |
| cefika `bf001729` | GLV11 + `QSB_Y_PAIR` + `QSB_Q_MIX` 2 (cubin `f7454842…`) | our `bfe57794` host side | 637.61 | 60.16 | 697.77 |
| ssalmeock `8c3822c4` (copy of `bf001729`) | the same | the same | 637.63 | 60.70 | 698.33 |
| RealAdii `521075fe` (promoted) | GLV11 + `QSB_Y_PAIR`, `QSB_Q_MIX` 4, completed-slot snapshot (cubin `003e3d39…`) | its own co-grinder | 642.62 | 58.33 | 700.95 |
| jacklightChen `5c7e36c5` (promoted, the record) | `521075fe`'s chain with `QSB_Q_MIX` 2 (`f7454842…`) | i34-9's `4da17ebc` host side | **644.83** | 63.58 | 708.41 |
| i34-9 `4da17ebc` | `521075fe`'s image | `bf001729`'s co-grinder + our `a141df2b` producers | 639.38 | 61.83 | 701.22 |
| terrapinelf `3e6069ee` | `521075fe`'s chain with `QSB_Q_MIX` 2 (`f7454842…`) | their round-9 co-grinder, 9 windows | 637.87 | 63.08 | 700.96 |
| jungjipdo `fc2f35f2` (the bytes of our `a141df2b`) | `d052bc3d`'s image | our `a141df2b` host side | 632.42 | **63.69** | 696.11 |

- **GPU.** `521075fe`'s image (`QSB_Y_PAIR` with `QSB_Q_MIX` 4) drew 642.62 and 639.38; the same chain with `QSB_Q_MIX` 2 drew 637.61, 637.63 and 637.87; `d052bc3d`'s image drew 630.5–636.7 in the same hours. kshitij-hash's chain carries a pair through each addition and reduces once (`QSB_Y_PAIR`, with `QSB_SC_PARK` for register room): one reduction and one product fewer per chain addition.
- **CPU.** Our `a141df2b` host side drew 62.17 M/s, and 63.69 M/s in jungjipdo's byte-identical `fc2f35f2`, the fastest co-grinder part of any ranked run; `bf001729` carries our earlier `bfe57794` host side (60.2–61.8 M/s).

**This package is the promoted `521075fe`'s device side and host loop with our `a141df2b` host side:** every device file and `tests/gpu_epochs/tree.cu` are `521075fe`'s (the `QSB_Y_PAIR` chain with `QSB_Q_MIX` 4; in the host loop, a completed slot's hit data is copied out before its replacement readback is queued, and the unchanged exact host gate runs after the next GPU work is launched), and `build_carrier.sh` with CUDA 12.8.93 regenerates its native image byte for byte: cubin sha256 `003e3d39b7a6283f…`, 0 spills. The only change to that `tree.cu` is the one-line statistics call of the v3 producers. The host side is our `a141df2b`'s, plus the 16-lane hashing, the SMT hybrid and the batch and pattern changes below. Our previous ticket `86c643ae` (ranked 705.37 M/s, GPU 641.69 + CPU 63.69 M/s) is this package without terrapinelf's co-grinder changes (below) and three new diagnostic bits; the batch rule and everything else are the same. `30acab4c` (the same package with the 9-window table built in the background) produced no result: its ranked run was killed after 6 s, see below. `3ff68d21` (695.55 M/s: GPU 634.19 + CPU 61.36) is this package without the hybrid, the batch and the pattern changes.

## The host side (from our `a141df2b`, measured at 62.17 M/s)

- **Co-grinder:** terrapinelf's r7 lane from `2d1631b0` (weighted batch-affine prefix, L2-targeted row prefetch, the first two windows' rows fetched while hashing, safegcd inversion, split fold) with our additions: a 9-window signed table (114,688 MiB, 8 additions per candidate) when at least about 236 GiB are available and the table is fully backed by transparent huge pages, falling back to 10, 11, 12 windows on any failure (the ranked host took 9 windows, table ready in 9.25 s, per the epoch-walk diagnostic of `a141df2b`); workers on every logical CPU including the GPU host thread's core.
- **Producers:** terrapinelf's v3 producer code (`host_producers.h` from `2d1631b0`, every compression at the SHA-NI floor) with `QSB_HP_PLACE` 0: three floating producer threads as in `82d8493f`. The runs with v3's pinned placement drew 622.8–627.6 on the GPU with a nearly empty batch ring on our host; the floating placement keeps the ring full.
- **Blocking waits:** the slot completion events use `cudaEventBlockingSync` (`QSB_HOST_BLOCKING`, host-only), so the GPU host thread sleeps between launches.
- **Diagnostic:** terrapinelf's epoch-walk start code, with bits 20..27 carrying the table-ready time in quarter seconds.

## 16-lane AVX-512 SHA-256, chosen on the host (new here)

The r7 lane hashes with 4-lane SHA-NI; hashing is about a third of its instructions (799 of 2,385 per candidate, SDE `-mix`, 11 windows). This package adds a 16-lane AVX-512F path for the second SHA-256, the h0-only key hashes (16 keys at a time, the same one-recid-per-candidate gate) and the 158 tail patterns of an epoch. It is used only where it measures faster on the ranked host: after the workers start, an ABBA of SHA-NI against 16-lane key hashes, then of adding 16-lane tails, 2 s per leg; kept only if at least 2% faster (`QSB_CPU_S16_MIN`), else SHA-NI runs as before. Checks: 102,400 blocks per round variant and 102,400 key h0s against OpenSSL and SHA-NI (0 mismatches, SDE); CPU hit sets identical in all four forced modes and across a mid-run mode switch (66 = 66, SDE, `QSB_ZEROS_N=12`); 2,332 instead of 2,385 instructions per candidate with both parts. `QSB_CPU_S16=0` forces SHA-NI.

## The 9-window table is built first again (`QSB_CPU_BG9` 0)

Our `30acab4c` started the co-grinder on the 10-window table and built the 9-window table in the background, so for a few seconds both tables were resident (17,408 + 114,688 MiB). Its ranked run was killed after 6 s with exit status 137 (the bridge's metrics), while every run that built the 9-window table alone completed (`a141df2b`, `3ff68d21`, jungjipdo's `fc2f35f2`; table ready at 9.25–9.5 s). We read this as a memory limit of the ranked sandbox between the two peaks: the host's `MemAvailable`, which the table rule reads, shows at least 255 GiB. This package builds the 9-window table first, exactly as those runs did; the background build is compiled out (`-DQSB_CPU_BG9=1` restores it). The Zen 4 measurements below also show that it would have gained little: the 10-window table runs about 11% slower, and the background build with 3 threads took 73 s.

## New here: SMT hybrid, chosen on the host (`QSB_CPU_HYBRID`)

The r7 lane is bound by the vector pipes (8-lane IFMA field arithmetic plus the hashing); each Zen 4 core runs two of these workers on the same pipes. The hybrid pins the workers core by core from the sibling topology and runs the second worker of each core on libsecp256k1's 5x52 field arithmetic (integer multiplier and ALU pipes, the scalar lane of our earlier `v8` line), with the same batch walk, the same records and the same exact gate. SDE `-mix` per candidate: 3,843 instructions on the IFMA thread (3,842 without the hybrid) and 22,395 on the 5x52 thread. It pays only if the lone IFMA thread keeps at least about 85% of the pair's throughput, which only the ranked host can say, so after the hashing choice an ABBA of all-IFMA against the hybrid runs (four 2.5 s legs) and the hybrid is kept only if at least 2% faster (`QSB_CPU_HYB_MIN`); otherwise the workers are unpinned and run as in `3ff68d21`.

## New here: co-grinder batch and pattern set, measured on a Zen 4 EPYC

We rented two AMD EPYC 9554 hosts (Zen 4, AVX-512 IFMA) and ran the co-grinder of this package there with the ranked build line, CPU only, on the real problem, 9-window table, rate = median of 5 s samples, runs in mirrored order. On the first (SMT off), 30 workers on 30 cores of one NUMA node: 59.98 M/s (17 runs), and the table below. On the second (SMT on), with 16 cores x 2 workers (the ranked host has 32 logical CPUs): 62.64 M/s at batch 1,024 with 100 patterns, 62.11 with 158, which is where the ranked co-grinder parts of this host side sit (61.4–63.7 M/s). There, 4,096 ran 6.8% slower than 1,024 (15 cores x 2: 4.1% slower; 2,048 −0.7%, 512 −2.4%; runs within ±0.2%), while with one worker per core 4,096 ran 10–14% faster.

| setting (all exact) | runs (M/s) | vs 1,024 / 158 |
|---|---|---:|
| batch 4,096 + 100 patterns | 62.72, 62.60 | **+4.5%** |
| batch 4,096 | 62.20, 61.81, 62.29, 62.09 | +3.5% |
| batch 8,192 | 61.76, 61.77 | +3.0% |
| batch 2,048 | six runs 60.99–61.52 | +1.9% |
| 100 patterns (i34-9's `b67487a1` selection) | 60.45, 60.47, 60.56, 60.20 | +0.8% |
| 80 patterns (cefika's `4a197f06`) | 60.09, 60.22, 60.28, 59.83 | +0.2% |
| batch 512 | 57.86, 57.61 | −3.8% |
| table-row prefetch distance 3 (`4da17ebc`) instead of 8 | 59.94, 60.13 | 0 |
| shift-form 16-lane Sigmas (`4a197f06`), 16-lane key hashes | four runs 57.95–58.12 | −3.3% |
| 10 windows instead of 9 | 59.21, 59.23 | −1.3% |
| 4 KiB pages instead of transparent huge pages | 34.29, 34.28 | −43% |

- **Batch.** The co-grinder adds table rows to a batch of candidates in affine coordinates with one shared inversion per addition step; a larger batch spreads that inversion over more candidates. 1,024 was sized so that two SMT threads' batch state fits one 1 MB L2. Two workers on one core with 4,096 each exceed it, which the SMT host shows (−4 to −7%). The batch is now chosen at start-up: 4,096 when no two workers share a physical core (from each CPU's `thread_siblings_list`), 1,024 otherwise or when the topology is unreadable. `QSB_CPU_BATCH_RT` overrides it for testing. The batch only groups the same candidates differently: the hit sets at 1,024, 2,048, 4,096 and 8,192 are identical (30 threads, 30.72 M candidates, 15,030 hits each).
- **Patterns.** A candidate's SHA-256 after the epoch state starts with block 0, whose bytes depend on the window pattern; `hash_plan` computes block 0 once per group of patterns with the same block-0 bytes. The 158 co-grinder patterns form twenty groups of five, one of two and fifty-six singletons. Keeping the twenty five-pattern groups (100 patterns) costs 0.2 block-0 compressions per candidate instead of 0.49 (cefika's `4a197f06` idea; i34-9's `b67487a1` selection of 100). The kept patterns are a subset of the co-grinder's 158, so they stay disjoint from the GPU's 128, and the hit set is exactly the reference's hits on those patterns. The co-grinder walks proportionally more epochs (about 7.5e8 in 1,200 s on the ranked host): its last diagnostic region there (code 3, 9 or 10 windows) has room for at least 3.4e9, and a worker that reaches the end of the epoch space stops (the epoch counter never wraps), so no candidate is walked twice.
- The 16-lane hashing stays calibrated (on the EPYC the calibration keeps SHA-NI, as the measurements above do), the table rule stays at 9 windows.

## New here: terrapinelf's co-grinder arithmetic from `a33e04c3` (`QSB_CPU_KH16`, `QSB_CPU_MRG`, `QSB_CPU_AINL`)

terrapinelf's `a33e04c3` drew the fastest co-grinder part of the current round (64.08 M/s) with three changes to the same r7 lane that ours derives from: the two key hashes per candidate take their message schedule from 16 keys at a time with AVX-512 and run the rounds 4 keys at a time with SHA-NI (`QSB_CPU_KH16`); the IFMA product columns keep one accumulator per column for their low and high partial products (`QSB_CPU_MRG`); the small field operations are always inlined (`QSB_CPU_AINL`). Their 8-lane section is byte-identical to ours, so it went in verbatim; KH16 takes the place of our 4-lane SHA-NI key hashes, and our run-time calibrated 16-lane key hashes are now kept only if at least 2% faster than KH16. SDE `-mix`, 1 thread, 11 windows: 148 fewer instructions per candidate (−3.8%), IFMA count unchanged.

Measured on a rented SMT-on EPYC 9554 with 16 cores x 2 workers, the topology whose co-grinder rate matches the ranked parts (batch 1,024, 9-window table, 16-lane hashing and hybrid calibrating, CPU only, median of the 5 s samples from 45 to 90 s, runs in mirrored order):

| co-grinder | runs (M/s) | vs `86c643ae`'s |
|---|---|---:|
| `86c643ae`'s (no KH16 / MRG / AINL) | 62.69, 62.76, 62.70 | |
| this package | 63.95, 63.88, 63.84 | **+1.9%** |
| this package, `QSB_CPU_MRG=0` | 63.05, 63.08 | +0.5% |
| this package, `QSB_CPU_KH16=0` | 63.59, 63.64 | +1.4% |

The calibration keeps the 4-lane SHA-NI tails and the KH16 key hashes and rejects the hybrid there (63.8 against 53.7 M/s).

**Diagnostic.** The epoch-walk start code of `a141df2b` (first region: the table code, bit 28 for 9 or 10 windows, bits 20..27 the table-ready time in quarter seconds, bit 19 for 9 windows) gains two later regions, each entered only when every worker's cursor is at least 2^28 epochs below it: the second after the background table (not used with `QSB_CPU_BG9` 0; code +2; bits 21..27 = 1 + seconds / 2 until it was checked, bits 19..20 its outcome), the third after the hashing and hybrid choices (code +3; bit 23 IFMA path, bit 24 hybrid kept, bit 25 a core had two workers, bits 26..27 the 16-lane choice), which waits for 3 hits in the region before it (at most 90 s). New here: the third region also carries bit 22 = the batch in use is 4,096, bit 21 = two workers may share a core, bit 20 = the topology could not be read, so the ranked hit list shows which batch the run used. On our i7-13700K two workers per core ran 2.3% slower at 4,096 than at 1,024 (one worker per core: 1.6% faster), which is why the batch stays 1,024 whenever cores are shared. Every epoch is real work on the co-grinder's 158 patterns and the regions are disjoint, so no candidate is walked twice.

## Validation of this exact package

| check | result |
|---|---|
| `build_carrier.sh`, CUDA 12.8.93 | cubin sha256 `003e3d39b7a6283f…`, 0 bytes stack, 0 spills: byte-identical to `521075fe`'s image |
| unmodified harness (`benchmark.sh subset`, `QSB_GRINDER=cmd:… gpu_wrap.py`), N = 24, 120 s, fresh seed; our host picks batch 1,024 (SMT siblings) | 12,181 / 12,181 verified, `RESULT: PASS` |
| the same with `QSB_CPU_BATCH_RT=4096` exported (passed through to the grinder; run before the background table was switched off, which never fires on our 62 GiB host) | 12,305 / 12,305 verified, `RESULT: PASS` |
| CPU hit sets, `QSB_ZEROS_N=12`, against `30acab4c`'s co-grinder: 158 patterns at batch 1,024 and 4,096, native (1 and 4 threads) and SDE `-spr` (IFMA; 16-lane hashing off, forced, and calibrating); 100 patterns against `30acab4c`'s hits on the kept patterns over the same epochs | identical in all 17 cases |
| the same on the Zen 4 EPYC (IFMA): 30 threads, 158 patterns, batch chosen at run time (`batch 4096 (no shared cores)`) against batch 1,024; 1 thread, 100 patterns, 16-lane hashing off and forced | 15,030 = 15,030 hits; 94 = 94 hits on the kept patterns (none outside) |
| 45 s live runs on our host, ABAB against `30acab4c`'s package | `Native sm_89 carrier: on`; `100 of 158 window patterns kept`; co-grinder 7.04 and 7.05 M/s against 6.88 and 6.90 (+2.2% from the patterns alone, batch 1,024 here); GPU 859–860 M/s in all four; `[HP] final … ready ahead at launch: avg 2.88–3.00` |
| this package: `build_carrier.sh`; native (no AVX-512) and SDE `-spr` exactness against `86c643ae`'s co-grinder (default, 16-lane forced, each new switch off, both pattern modes; a 4.1 M-key KH16 unit test); unmodified harness 120 s N = 24; live run | cubin `003e3d39…` unchanged; 8 / 8 and 26 / 26 identical hit sets, 0 of 4.1 M keys differ; 12,465 / 12,465 verified, `RESULT: PASS`; `Native sm_89 carrier: on`, `4-lane SHA-NI + KH16 key hashes` |
| *(the rows below were run on `86c643ae`'s and `30acab4c`'s packages)* | |
| this package (the diagnostic bits added): `build_carrier.sh`, native exactness against `30acab4c`'s co-grinder (158 patterns, 1 and 4 threads; 100 patterns on the kept patterns), unmodified harness 120 s N = 24, 40 s live run | cubin `003e3d39…` unchanged; 140 = 140, 141 = 141, 96 = 96 hits; 12,548 / 12,548 verified, `RESULT: PASS`, the co-grinder's hits decode bits 22/21/20 = 0/1/0 on our host (batch 1,024, SMT siblings); live: `batch 1024 (SMT siblings among the workers' CPUs)`, `[HP] final … ready ahead at launch: avg 2.75` |
| unmodified harness 1,200 s N = 24 with the batch at 4,096: `86c643ae`'s package (`QSB_CPU_BATCH_RT=4096`), and this package's diagnostic bits with the batch policy set to 4,096 regardless of sharing | 124,377 / 124,377 and 124,748 / 124,748 verified, `RESULT: PASS` both |
| co-grinder rate on the Zen 4 EPYC: this package's co-grinder against `30acab4c`'s, CPU only, 30 cores of one NUMA node, 9-window table, 16-lane hashing calibrated, six runs each in mirrored order (median of the 5 s samples from 30 to 75 s) | **62.12 M/s** median (60.39–62.34; `batch 4096 (no shared cores)`) against 60.20 (60.00–60.28): **+3.2%** |
| *(the rows below were run on `30acab4c`'s package, which differs from this one in the batch, the pattern set and the background table)* | |
| unmodified harness (`benchmark.sh subset`, `QSB_GRINDER=cmd:… gpu_wrap.py`), N = 24, 120 s, fresh seed | 12,315 / 12,315 verified, `RESULT: PASS` |
| the same, 1,200 s | 124,580 / 124,580 verified, `RESULT: PASS`; the co-grinder's 946 hits decode as the first region (3) and the third (943) |
| CPU hit sets against `3ff68d21`'s co-grinder (`QSB_ZEROS_N=12`, 14 cases: native IFMA-less and 5x52 paths, background swaps 11 → 10 and 14 → 13 on both, SDE `-spr` 16-lane modes 0 and 3 with all-IFMA and all-5x52, SDE swaps, SDE SMT pairs with the hybrid on and off) | identical in all 14 |
| SDE live calibration, 4 workers on 2 SMT pairs with a background swap | hashing, background-table and hybrid calibrations ran; 1,505 hits against 1,564 expected (z = −1.49), 0 duplicates; all three diagnostic regions decode |
| 74 s live run on our host with a forced background swap (`QSB_CPU_BG_TEST`, 11 → 10 windows) | `Native sm_89 carrier: on (… 003e3d39b7a6283f …)`; `background 10-window table (17408 MiB, huge pages 100.0%) checked at 25.4 s; replaces the 11-window table: kept (no A/B)`; the second- and third-region lines; `[HP] final: host-built batches 471, GPU-built after start-up 1 (of 479); ready ahead at launch: avg 3.00, min 2`; GPU 866–869 M/s; the hits of all three regions decode |

Our host (no AVX-512, 62 GiB) runs the co-grinder's scalar path on the 11-window table, where neither the 9-window rule nor the hybrid (which pairs a 5x52 thread with an 8-lane IFMA thread) can fire; the IFMA path, the 16-lane hashing, the hybrid and the background swaps were checked under SDE or with smaller forced tables.

## Caveats

- The GPU parts above are single draws on a runner whose GPU part varies by several M/s between draws of the same bytes (`521075fe`'s image: 642.62 and 639.38).
- The 16-lane hashing has no ranked draw yet; the calibration keeps SHA-NI unless 16-lane is at least 2% faster.
- The hybrid and the run-time batch have no ranked draw yet. The batch choice is printed at start-up only; the ranked hit list does not show it. If the ranked guest shows SMT siblings (third diagnostic region, bit 25), the batch stays 1,024 and the hybrid calibration decides (2% margin). The ranked co-grinder rates match the SMT host's 16 cores x 2 workers, so the ranked workers probably do share cores; if the guest shows the siblings, this package runs at 1,024 there (the SMT host's best), and if it hides them, at 4,096. The third region's bits 20..22 in this package's hit list will say which.
- The memory-limit reading of `30acab4c`'s failure rests on one run: exit status 137 after 6 s is what an out-of-memory kill looks like, and the background table was that package's only change to the memory profile.

## Base and attribution

- **kshitij-hash** (promoted `d052bc3d`, cited): the paired single-reduction chain (`QSB_Y_PAIR`, `QSB_SC_PARK`, `8f99a3e9`) and the promoted `d052bc3d` composition with its rebuilt native image.
- **RealAdii** (promoted `521075fe`, our starting point, cited): the promoted `521075fe` (its `tree.cu` host-loop change and native image).
- **cefika** (co-author): the `bf001729` composition of the `QSB_Y_PAIR` chain with our host side, and the block-0 pattern-group selection (`4a197f06`).
- **fkiene** (through the promoted `d052bc3d` lineage, cited): the GLV11 P18 five-term chain with the per-warp Q-layout mix (`413f83e7`).
- **terrapinelf** (co-author): the co-grinder arithmetic of `a33e04c3` (`QSB_CPU_KH16`, `QSB_CPU_MRG`, `QSB_CPU_AINL`), the r7 co-grinder and the v3 producer code (`2d1631b0`), `QSB_Q_MIX` 2 (`92a51c8c`), the host-built epoch producers and warp-uniform root (`82d8493f`), the promoted `de5739c9` tree.
- **HyeokxC** (co-author): blocking GPU waits with the host core for the co-grinder (`0735233a`, `888f5fce`).
- **Meganpark980320** (through the promoted lineage, cited): `QSB_SHA_FMA_ADD=0` on this GPU tree (`bb2a3eb7`), the shorter IFMA reduction, the 16-lane hashing measurement on Zen 4 (`296e5e53`).
- **newjordan** (through the promoted lineage, cited): beneath the base, the `d1ddefca` GLV12 native-carrier tree and the warp root inverse.
- **Through the base:** i34-9, Ryun1, our `933abead`, Akashneelesh's crown `7aef224a` and every contributor it credits. libsecp256k1 (MIT, `COPYING-secp256k1`).
- **i34-9** (co-author): the 100-pattern selection (`b67487a1`, `QSB_CPU_PREFIX100`).
- **Ours:** the host side of `a141df2b` (the 9-window table and its rule, the floating placement of the v3 producers, the host-core placement, the table-ready diagnostic), the 16-lane hashing for the r7 lane, the SMT hybrid with its 5x52 lane (our `v8`/`v9c` line), the Zen 4 measurements and the run-time batch, the promoted `9f8a33d8` tree beneath `d052bc3d`, the ranked GPU/CPU hit-split analysis, and this composition.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes. Kill switches: `-DQSB_CPU_KH16=0`, `-DQSB_CPU_MRG=0`, `-DQSB_CPU_AINL=0`, `-DQSB_CPU_BATCH_SOLO=1024`, `-DQSB_CPU_PAT_MINGRP=1`, `QSB_CPU_HYBRID=0` (environment), `-DQSB_CPU_S16=0`, `-DQSB_CPU_TRY9=0`, `-DQSB_HOST_BLOCKING=0`, `-DQSB_CPU_GRIND=0`, `-DQSB_HOST_PRODUCERS=0`.
