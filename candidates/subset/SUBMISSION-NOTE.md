# Subset: Meganpark980320's co-grinder (`296e5e53`) on the promoted `9f8a33d8` tree, with the GPU host thread's core given to two more co-grinder workers (blocking-sync host waits) and one scalar safegcd inverse per batch-inversion root

Effort: max. Prepared with Claude Opus 5.5 in Claude Code.

**We have no RTX 4090 and no AVX-512 CPU.** Every speed argument below comes from the public ranked runs. Our 4× RTX A6000 host (Zen 3) and Intel SDE were used for correctness only.

## What this is

The tree is Meganpark980320's `296e5e53`. That is ercumentyildirim's promoted `9f8a33d8` with Meganpark's co-grinder. Its ranked draw was 681.92 M/s: GPU 633.15 + CPU 48.78.

Three host-only changes are added. The device code and the native image are unchanged: `build_carrier.sh` with CUDA 12.8.93 gives cubin sha256 `070afc8c84113a07…`, and only the header's source-hash comment moves.

1. **`QSB_HOST_BLOCKING=1`** (`tree.cu`): the two slot-completion events are created with `cudaEventBlockingSync`. The GPU host thread therefore sleeps in `cudaEventSynchronize` instead of spinning a CPU.
   - While it waits for batch k−2, batch k−1 is already queued on the other stream, so the wake-up is hidden behind batch k−1.
   - The idea is newjordan's `QSB_ASYNC_BLOCKING` (`212237f4`). Their 4090 measured 835.4 vs 835.3 M/s unstarved.
2. **`QSB_CPU_HOST_CORE`** (`CpuGrindSubset.h`, on when `QSB_HOST_BLOCKING` is): the co-grinder no longer reserves the host thread's core. It runs `ncpu` workers (32 instead of 30 on a 32-CPU host), pinned one per CPU.
   - Workers stay `SCHED_IDLE`, so the host thread and the producers always preempt them.
3. **`QSB_CPU_SCALAR_INV=1`** (`CpuGrindSubset.h` + `CpuSafegcd.h`): `fe8_inv4` combines the eight lanes' chain products in one horizontal product and inverts it with one scalar variable-time safegcd.
   - It replaces the 255-squaring Fermat chain on the vector pipes.
   - `fe8_inv4` serves every window step, the final step and the 8-lane table builder.
   - `CpuSafegcd.h` and `fe8_inv_lanes` are jacklightChen's (`3ca8bbb6`), byte for byte. Its scalar core is terrapinelf's `97f347a8` (libsecp256k1 `modinv64_var`). The lane tree keeps zero lanes isolated, as the lane-wise exponentiation did.
   - The scalar 5×52 path (`f52_inv4`) uses the same inverse.
   - The batch stays 4,096. `3ca8bbb6`'s 1,024 batch and its single-read scalar pass are not taken.

## Why: what the ranked runs show

- **The CPU part can be measured precisely.** Every ranked run publishes its verified hits.
  - A hit's window pattern says whether the GPU (the fixed 128-pattern set) or the co-grinder (the other 158) found it.
  - The co-grinder's worker t walks epochs t, t+T, t+2T, …, where T is the worker count. Each CPU hit's epoch is the lexicographic rank of its first six skips.
  - The furthest epoch in each residue class mod T gives that worker's progress. Summed over workers and multiplied by 158 candidates, this gives the lane's rate.
  - The walk estimate repeats to ±0.1–0.3% across runs of identical code; CPU-hit counts carry ±1.3% Poisson noise. Runs of the `97f347a8` lane: 46.29, 46.20, 46.01 and 46.25 M/s. `9f8a33d8`'s lane: 43.11, and 43.10 in `413f83e7`.
- **Lanes, walk-based (M/s):**

  | lane (run) | workers | M/s |
  |---|---:|---:|
  | `9f8a33d8` | 31 | 43.1 |
  | `3ca8bbb6` (`9f8a33d8`'s lane + safegcd roots, 1,024 batch) | 31 | 44.5 |
  | `97f347a8` / `81f1b821` | 30 | 46.0–46.3 |
  | `654841c2` (11 windows) | 30 | 48.50 |
  | `296e5e53` (Meganpark) | 30 | 48.8 |

- **The host core is spare capacity.** In `9f8a33d8`, the extra `SCHED_IDLE` worker beside the spinning host thread progressed 11.53 M epochs, the most of all 31 workers (the other 30: 10.12–11.52, mean 10.55).
  - With the host thread asleep, both of that core's CPUs can take a worker.
  - Counter-evidence for doing this without idle priority: `c1915087` (one more worker, normal priority) walked 1.9% less.
- **GPU part:** the ten draws of this image (`070afc8c…`) are 625.76–633.15, mean 630.2 (sd 2.2).
- **Safegcd roots:** `3ca8bbb6` put them, with 1,024-candidate batches, into `9f8a33d8`'s lane and walked 3.3% further (44.52 vs 43.11, same 31-worker placement).
  - Here the batch stays at 4,096, where the Fermat root is 279 of about 3,350 vector field operations per window step.
- **Expectation:** CPU 48.8 × (1.04 to 1.067 for the host core) × (1.02 to 1.03 for the roots) ≈ 52–54 M/s.
  - Total ≈ 630 + 52–54 ≈ 682–684 M/s against 675.54, i.e. about +1.0% to +1.3%.
  - A single draw's GPU part varies by about ±2.2 M/s.

## Validation of this exact package

| check | result |
|---|---|
| `build_carrier.sh`, CUDA 12.8.93 | cubin sha256 `070afc8c84113a07…` (462,496 B), 0 spills: `9f8a33d8`'s image |
| `./setup.sh subset` | passed |
| GPU hit identity (sm_86 diagnostic build of the same source, GPU only, fixed seed, 150 s) | 5,169 / 5,169 verified; the 4,730 hits in the common range are identical to `9f8a33d8`'s |
| unmodified harness (`run_benchmark.py`, N=24, co-grinder on; Zen 3, so the scalar lane path), 150 s, two seeds | 5,501 / 5,501 and 5,371 / 5,371 verified, `RESULT: PASS` (114 and 120 from the CPU file) |
| co-grinder exactness, one worker, deterministic candidate count, against the same tree with Fermat roots | scalar path (native), N=16, 3,002,368 candidates: 112 = 112 hits, identical<br>16-lane IFMA pipeline under Intel SDE 9.48 `-spr`, N=16, 401,408 candidates: 17 = 17, identical<br>same, N=12, 602,112 candidates: 278 = 278, identical, with the same 8-lane-built table digest (`QSB_CPU_TABLEHASH`) |
| GPU-starvation check on the A6000 (co-grinder on, ABBA against `296e5e53`, 90 s runs; not a 4090 estimate) | GPU rate per MHz +0.5% (0.19667 vs 0.19566): the sleeping host thread does not starve the GPU |
| whole binary under Intel SDE 9.48 `-spr` (the AVX-512 IFMA path), co-grinder on (small 64 MiB table), 240 s | `CPU co-grind: 32 threads (of 32 CPUs), 8-lane IFMA`, `16-lane AVX-512 pipeline`; 8,357 / 8,357 hits verified by `harness/verify.py`, including 2 from the IFMA co-grinder; SIGTERM to exit 1.3 s |

## Caveats

- The ranked host's scheduler behaviour is not published.
  - If a wake-up ever left the GPU idle, the GPU part would drop. The two-slot pipeline keeps a batch queued, and newjordan measured no change on a 4090.
- The safegcd root's gain at a 4,096 batch is a prediction. `3ca8bbb6` measured the related change (+3.3%, with 1,024 batches) on `9f8a33d8`'s lane.
- Kill switches:
  - `-DQSB_HOST_BLOCKING=0` (spin wait; also turns the host-core workers off)
  - `-DQSB_CPU_HOST_CORE=0`
  - `-DQSB_CPU_SCALAR_INV=0` (Fermat roots)
  - and every switch of `296e5e53`

## Base and attribution

- **Meganpark980320** (co-author): the co-grinder (`296e5e53`, the same file as `9745ce9b`): the 16-lane pipeline, the 8-lane table builder, the fused operations, the calibrated next-window prefetch and the 10-lookup cap.
- **newjordan** (co-author): the blocking-sync host wait (`212237f4`).
- **terrapinelf** (co-author): the scalar safegcd core (`97f347a8`); the host-built epoch producers and warp-uniform root (`82d8493f`); and the promoted `de5739c9` beneath everything.
- **jacklightChen** (co-author): `CpuSafegcd.h` and the zero-safe lane tree (`3ca8bbb6`).
- **ercumentyildirim**: the promoted `9f8a33d8` tree and image, the producers' 16-lane path, and the extra-worker-on-the-host-core idea that items 1–2 extend.
- **Through the base:** i34-9, fkiene, Ryun1 and every contributor the base credits. libsecp256k1 (MIT, `COPYING-secp256k1`).
- **Concurrent, still queued when this was prepared:**
  - terrapinelf's `18c9afa8` also sleeps the host thread (`QSB_HP_BLOCKSYNC`) and gives its CPU to a co-grinder worker, on terrapinelf's own co-grinder.
  - fkiene's `c6480c1e` also carries `3ca8bbb6`'s safegcd changes, with 1,024 batches, into this co-grinder.
  - This package combines both on Meganpark's co-grinder at 4,096 batches. It was built independently of both and takes no code from them.
- **Ours:** items 1–2 as implemented here; the port of item 3 into this co-grinder; the walk-based ranked lane measurement; and the checks above.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes, and there are no includes outside it.
