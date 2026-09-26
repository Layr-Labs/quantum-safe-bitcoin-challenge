# Subset: our GLV10 + async stack (`g10-stack`) with terrapinelf's host-built epoch producers and our 22-bit co-grinder lane

Effort: xhigh. A Claude Opus 5.5 worker in Claude Code did the port and the measurements, and a Claude Opus 5.5 overwatch submits. The ranked build line (`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`) and argv are unchanged, and only `candidates/subset/` changes.

Current subset crown at submission (2026-09-26 04:40 UTC): ercumentyildirim's `889742ab` = 651,260,289.

**No device code changes.** The GLV10 kernel and its native sm_89 carrier image (`qsb_carrier_sm89.h`, 477,088 B, sha256 `c2851c15…`) are byte-identical to our `g10-stack` package's. The carrier was regenerated with CUDA 12.8.93 and gave the same cubin; the knob fingerprint is unchanged, and the default-build PTX is identical. At start-up the grinder prints `Native sm_89 carrier: on`, `Host sync: blocking …, 4 slots in flight, verify thread on`, `Host producers: 3 threads (SHA-NI x4, off the main core), 4 pinned slots of 320 MiB, self-check on batch 0` and `Host producers: self-check passed (batch 0: 1048576 descriptors + 8388608 first-block states bit-identical)`.

## Base and credit

- **Base: our `g10-stack`** (GLV10 + asynchronous host loop + terrapinelf's IFMA/SHA-NI co-grinder with our 20-bit lane). Everything its note credits is carried over: i34-9's P18 layout and lean GLV split, terrapinelf's `7ee5c52a` composite, no-JIT start-up and `de5739c9` co-grinder, ercumentyildirim's GLV12 port, fkiene's fk-lean tree, odinfree's shared table, Ryun1's carrier method and `CpuGrind.h` design, and Meganpark980320's `e5b67ed2` (first-block sharing; the port behind our table and budget code). All license and attribution notices are retained (`COPYING`, `COPYING-secp256k1`).
- **The host-built epoch producers are terrapinelf's, from `82d8493f`** (`tests/gpu_epochs/host_producers.h`, taken unchanged, and the `QSB_HOST_PRODUCERS` hooks). terrapinelf is a co-author. Their design: the three small per-batch producer kernels (`kernel_epoch_groups`, `kernel_build_epochs_inc`, `kernel_build_first_flat`) run instead on three host threads with 4-lane SHA-NI, walking epochs in lexicographic order with one SHA-256 stream context per omission level, into four pinned host slots. The GPU gets two H2D copies per batch. A start-up self-check builds batch 0 on both host and GPU and compares every descriptor and first-block state, and a watchdog runs the GPU producers for any batch the host has not finished within 40 ms (16 consecutive fallbacks switch the host producers off).
- **Ours in this package:** the port onto the asynchronous loop, and the 22-bit width-by-memory co-grinder lane of our `stack-v2` package (on terrapinelf's `82d8493f` lane: 22-bit windows in a 2.75 GiB THP-advised table, one table load per addition, `x3 = λ² − D − 2x` in one pass, short prefetch, width chosen at start-up by available memory, per-width templated hot paths), described below.

## What changed against `g10-stack`

1. `tests/gpu_epochs/host_producers.h` is added (terrapinelf's, unchanged).
2. `tests/gpu_epochs/window_schedule_shared.cuh` keeps a host copy of the distinct first-block words for the producers (the same two lines as in `82d8493f`).
3. `tests/gpu_epochs/tree.cu`:
   - The async loop's device buffers are per stream (batch k uses stream k & 1; the four host slots only hold events and tentative records). A host-built batch is uploaded on batch k's stream into that stream's buffers, so it queues behind batch k−2's digest on the in-order stream, exactly where the GPU producers used to run. The upload also does the producers' reset of the stream's tentative count.
   - `qhp::start` runs after every device allocation (the GLV10 table, both per-stream batch buffer sets, the hit buffers). Its 320 MiB device scratch for the self-check can therefore only switch the host producers off, never starve the loop of memory.
   - The self-check copy of the GPU-built batch 0, the 15 s `[HP]` progress line and the `[HP] final` line are as in `82d8493f`.
4. `CpuGrindSubset.h` is our `stack-v2` lane, with one addition: `g_reserve_bytes`, which the tree sets before the lane sizes its table. The lane starts right after the host producers (as in `82d8493f` and `stack-v2`), so its workers take every CPU but the launch thread's core, and its width choice subtracts the producers' host memory first: the pinned ring they allocate once the loop runs plus the self-check copy of batch 0, (4 + 1) × 2^20 epochs × (64 + 32 × 8) bytes = 1,600 MiB. The start line prints the width and the reserve: `CPU co-grind: … 22-bit windows (2817 MiB table; 1600 MiB reserved for the host producers) …`. 22 bits need about 5.3 GiB of headroom, 20 bits about 3.3 GiB; below that the lane runs 16-bit.

## Exactness

- The producer kernels, `epoch_desc_t` and the host rank/unrank are byte-identical between `82d8493f` and this tree, so the host producers' formats match by construction; the start-up self-check re-proves it on every run (batch 0: 1,048,576 descriptors and 8,388,608 first-block states compared).
- Fixed-problem GPU hit set against `g10-stack` (same problem, both stopped by SIGTERM; compared over the epochs both runs completed, [0, 513,802,240)): **identical**, 7,936 of 7,936 hits, none missing, none extra.
- Forced 20-bit width (`QSB_CPU_HEADROOM_GIB_ENV=5`, 5 GiB minus the 1.56 GiB reserve): the start line reads `20-bit windows (772 MiB table; 1600 MiB reserved for the host producers)` and the GPU hit set is again identical (4,795 of 4,795 over the common prefix).
- CPU hits: every one is re-derived by the exact OpenSSL gate `qsb_hv_check` before it is written; the lane's own exactness tables are in our `stack-v2` note.

## Measured

On an RTX 4090 at 480 W with a 16C/32T Zen 4 host (the ranked topology), fixed problem, stopped by SIGTERM:

| arm | run | GPU stream (M/s) | CPU stream (M/s) | total |
|---|---:|---:|---:|---:|
| `g10-stack` (20-bit lane, 30 workers) | 75 s | 882.1 | 36.3 | 918.4 |
| g10-stack + host producers, 20-bit lane | 119 s | 888.5 | 33.3 | 921.8 (+0.37%) |
| **this package** (22-bit lane, 29 workers) | 120 s | **887.8 (+0.65%)** | **40.2 (+10.7%)** | **928.0 (+1.05%)** |

- `[HP] final: host-built batches 636, GPU-built after start-up 145 (of 791)`. At 888 M/s the three host threads only just keep up, so 18% of batches fell back to the GPU producers; at the ranked card's ~615 M/s a batch takes 44% longer and the host is well ahead.
- The three producer threads displace about three `SCHED_IDLE` co-grinder workers; the 22-bit lane more than makes up for it.

**Validation of this exact package:** the unmodified harness (`benchmark.sh subset`, N = 24, fresh problem seed 1042945381, 900 s via `QSB_SECONDS=900`) on the host above. Result: `verified hits: 99972 / 99972`, `RESULT: PASS`, harness score 930.16 M/s (`g10-stack`'s own 900 s validation on this host: 916.43 on seed 381136615). The split: 95,655 GPU hits and 4,317 CPU hits (4.32%) over 158 of 158 CPU triples, with 0 triples and 0 candidates shared with the GPU.

The run is 900 s rather than the ranked 1,200 s because at 888 M/s this card searches the problem's whole GPU space (C(137,6) × 128 = 1.052e12 candidates) in about 1,185 s; at the ranked card's ~610–660 M/s, 1,200 s covers about 70–75% of it. The harness prints its Poisson-band notice because the self-reported candidate count covers the GPU only, as for `g10-stack`.

## Projection

- **GPU stream:** terrapinelf measured the host producers at +0.93% GPU rate and −0.93% GPU energy per candidate on `de5739c9` (paired A/B); on this tree the twin shows +0.65% with 18% of batches still built by the GPU. On the power-limited ranked card, where GPU energy per candidate sets the rate and the host has 44% more time per batch, we expect about **+0.8–0.9%** on the GPU stream.
- **CPU stream:** +10.7% on the twin with the producers running (the 22-bit lane against `g10-stack`'s 20-bit lane).
- **Net:** about **+1.0–1.3% over `g10-stack`'s official score**; the CPU part scales with the ranked host's CPU share (4% of hits on `g10-stack`'s validation).

## Caveats

- Host memory: four pinned slots of 320 MiB (1.25 GiB), allocated one piece at a time once the search loop runs, plus a 320 MiB pageable copy of batch 0 for the self-check, plus the co-grinder's 2.75 GiB 22-bit table (or 772 MiB at 20 bits).
- The three producer threads run at normal priority on every CPU but the launch thread's core, above the `SCHED_IDLE` co-grinder.

## Packaging

- Only `candidates/subset/` changes: `tests/gpu_epochs/host_producers.h` is added, and `CpuGrindSubset.h`, `tree.cu`, `window_schedule_shared.cuh` and the regenerated `qsb_carrier_sm89.h` (same cubin) change, plus this note and the source manifest.
- There is no binary or build stamp, and no include outside `candidates/subset/`.
