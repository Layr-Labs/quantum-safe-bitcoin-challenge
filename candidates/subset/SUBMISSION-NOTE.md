# Subset: the promoted `d052bc3d` (fkiene's P18 GPU chain with its native image, Meganpark980320's co-grinder) with the GPU host thread's core given to two more co-grinder workers (blocking-sync host waits), one scalar safegcd inverse per batch-inversion root, and a shorter IFMA reduction

Effort: max. Prepared with Claude Opus 5.5 in Claude Code.

**We have no RTX 4090 and no AVX-512 CPU.** Every speed argument below comes from the public ranked runs. Our 4× RTX A6000 host (Zen 3) and Intel SDE 9.48 were used for correctness only.

## What this is

The tree is kshitij-hash's promoted `d052bc3d`:
- the promoted `9f8a33d8` record;
- fkiene's GLV11 P18 GPU chain from `413f83e7`, with a native sm_89 image rebuilt from its source;
- Meganpark980320's co-grinder from `296e5e53`.

Its ranked draw was 691.63 M/s: GPU 643.32 + CPU 48.31.

Four host-only changes are added, all in `CpuGrindSubset.h` (+ the new `CpuSafegcd.h`) and two lines of `tests/gpu_epochs/tree.cu`. The device code and the native image are unchanged: `build_carrier.sh` with CUDA 12.8.93 gives `d052bc3d`'s cubin sha256 `91948fc251250a66…`, and only the header's source-hash comment moves.

1. **`QSB_HOST_BLOCKING=1`** (`tree.cu`): the two slot-completion events are created with `cudaEventBlockingSync`, so the GPU host thread sleeps in `cudaEventSynchronize` instead of spinning a CPU.
   - While it waits for batch k−2, batch k−1 is already queued on the other stream, so the wake-up is hidden behind batch k−1.
   - The idea is newjordan's `QSB_ASYNC_BLOCKING` (`212237f4`), which measured 835.4 vs 835.3 M/s unstarved on a 4090.
2. **`QSB_CPU_HOST_CORE`** (on when `QSB_HOST_BLOCKING` is): the co-grinder no longer reserves the host thread's core. It runs one `SCHED_IDLE` worker per logical CPU (32 instead of 30 on a 32-CPU host), so the host thread and the producers always preempt them.
3. **`QSB_CPU_SCALAR_INV=1`**: `fe8_inv4` combines the eight lanes' chain products in one horizontal product and inverts it with one scalar variable-time safegcd, instead of the 255-squaring Fermat chain on the vector pipes.
   - It serves every window step, the final step and the 8-lane table builder; the scalar 5×52 path uses the same inverse.
   - `CpuSafegcd.h` and the zero-safe lane tree are jacklightChen's (`3ca8bbb6`), byte for byte. The scalar core is terrapinelf's `97f347a8` (libsecp256k1 `modinv64_var`).
   - The batch drops to 1,024 (terrapinelf's L2-sized batches from `97f347a8`, as in `3ca8bbb6` and `cfe9f377`): both SMT threads' batch state then fits the core's 1 MB L2, and with the scalar root a four times more frequent root stays cheap.
4. **`QSB_CPU_FOLD9=1`**: `fe8_red` folds product column 9 (a single IFMA high part, < 2^52) at 2^208 into columns 4 and 5 before column 5 is split. That removes the column-9 split and the second fold: 15 IFMA per reduction instead of 18.
   - The code is ercumentyildirim's (`cfe9f377`), after terrapinelf's `QSB_CPU_FOLD3` (`18c9afa8`).
   - hi(c9·R) < 2^37 added to column 5 (< 9·2^52) keeps its high part below 2^4, so every later bound is unchanged.

## Why: what the ranked runs show

- **The CPU part is measurable precisely.** A hit's window pattern says whether the GPU (the fixed 128-pattern set) or the co-grinder (the other 158) found it.
  - The co-grinder's worker t walks epochs t, t+T, t+2T, …, where T is the worker count, and each CPU hit's epoch is the lexicographic rank of its first six skips.
  - Summing each worker's furthest epoch, times 158 candidates, gives the lane's rate. Identical code repeats to 0.02–0.3% across runs (e.g. `9f8a33d8`'s lane: 43.11, 43.10, 43.05), where CPU-hit counts carry ±1.3% Poisson noise.
- **Lanes, walk-based (M/s, workers):**

  | lane | M/s (workers) |
  |---|---:|
  | Meganpark's (`296e5e53`, `d052bc3d`, `ded2ae38`; 10 lookups on the ranked host per `ded2ae38`'s diagnostic) | 48.8, 48.78, 48.70 (30) |
  | the same + safegcd roots, 1,024 batches, producer-lag yield (fkiene's `c6480c1e`) | 49.95 (30): +2.5% |
  | **the same + our items 1–3 (our `0735233a`)** | **52.95 (32): +8.6%** |
  | `654841c2` | 48.50 (30) |
  | the same + one worker beside the spinning host thread (`990960b3`) | 50.93 (31): **+5.0%** |
  | `9f8a33d8` | 43.11 (31) |
  | the same + safegcd roots and 1,024 batches (`3ca8bbb6`) | 44.52 (31): **+3.3%** |

- **The host core is spare capacity.** In `9f8a33d8` the worker beside the spinning host thread progressed 11.53 M epochs against a mean of 10.55 for the other 30. `990960b3` gained 5.0% from that one worker. With the host thread asleep, both of that core's CPUs can take a worker.
- **Measured: our items 1–3 on this co-grinder.** Our `0735233a` (the same co-grinder changes, on `9f8a33d8`'s GPU image) drew 685.37 M/s: GPU 632.11 + CPU 53.26 hits.
  - Its walk: 52.95 M/s with 32 workers, against 48.75 for the unmodified lane (the mean of its three runs): **+8.6%**.
  - All 32 workers progressed 11.86–13.09 M epochs; the two on the host thread's core were not slower. So the host core added the full +6.7% (32/30), and each worker gained 1.9% from the safegcd roots at batch 4,096.
  - `c6480c1e` (the same lane with `3ca8bbb6`'s safegcd roots at batch 1,024 and a producer-lag yield) gained 2.4% per worker. That is why this package takes the 1,024 batch; the difference is within about two walk-noise widths.
- **GPU part:** this image drew 643.32 in its one ranked run (the previous GLV12 image: 630.2 over 10 runs, sd 2.2).
- **Expectation:** CPU 52.95 (measured with items 1–3) × 1.005 (batch 1,024) × 1.00–1.03 (FOLD9, not yet measured on the ranked host) ≈ 53.2–54.8 M/s, on a GPU part of ≈643 (one draw).
  - Total ≈ 697.3 M/s against the 698.55 bar, with a one-run spread of about ±3.4 M/s: roughly a one-in-three chance to clear it.

## Validation of this exact package

| check | result |
|---|---|
| `build_carrier.sh`, CUDA 12.8.93 | cubin sha256 `91948fc251250a66…` (462,752 B), 0 spills: `d052bc3d`'s image; the header's source hash recomputes from this tree |
| `./setup.sh subset` | passed |
| GPU hit identity against `d052bc3d` (sm_86 diagnostic build of each source, GPU only, fixed seed, 150 s) | 5,847 / 5,847 verified; the 5,835 hits in the common range are identical |
| unmodified harness (`run_benchmark.py`, N=24, co-grinder on; Zen 3, so the scalar lane path), 150 s | 6,162 / 6,162 verified, `RESULT: PASS` (137 from the CPU file) |
| co-grinder exactness against `d052bc3d`'s co-grinder, one worker, deterministic candidate count | scalar path (native), N=16, 3,002,368 candidates: 112 = 112 hits, identical<br>16-lane IFMA pipeline under Intel SDE 9.48 `-spr`, N=12, 602,112 candidates: 278 = 278, identical, with the same 8-lane-built table digest (`QSB_CPU_TABLEHASH`) |
| whole binary under SDE `-spr` (the AVX-512 IFMA path), co-grinder on (64 MiB table), 240 s | `CPU co-grind: 32 threads (of 32 CPUs), 8-lane IFMA`, `16-lane AVX-512 pipeline`; 9,386 / 9,386 hits verified by `harness/verify.py`, including 2 from the IFMA co-grinder; SIGTERM to exit 1.3 s |

## Caveats

- Items 1–3 were measured on the ranked host (`0735233a`, at batch 4,096); FOLD9 and the batch-1,024 difference were not, and nothing here was timed on a 4090 or AVX-512 hardware of our own.
- If a wake-up ever left the GPU idle, the GPU part would drop. The two-slot pipeline keeps a batch queued, and newjordan measured no change on a 4090.
- Kill switches:
  - `-DQSB_HOST_BLOCKING=0` (spin wait; also turns the host-core workers off)
  - `-DQSB_CPU_HOST_CORE=0`
  - `-DQSB_CPU_SCALAR_INV=0`
  - `-DQSB_CPU_FOLD9=0`
  - `-DQSB_CPU_BATCH=4096`
  - and every switch of `d052bc3d`

## Base and attribution

- **kshitij-hash**: the promoted `d052bc3d` (the composition and the rebuilt native image).
- **fkiene**: the GLV11 P18 chain with the per-warp Q-layout mix (`413f83e7`).
- **Meganpark980320** (co-author): the co-grinder (`296e5e53`).
- **newjordan** (co-author): the blocking-sync host wait (`212237f4`).
- **terrapinelf** (co-author): the scalar safegcd core (`97f347a8`), the column-9 fold idea (`QSB_CPU_FOLD3`, `18c9afa8`), the host-built producers and warp-uniform root (`82d8493f`) and `de5739c9` beneath everything.
- **jacklightChen** (co-author): `CpuSafegcd.h` and the zero-safe lane tree (`3ca8bbb6`).
- **ercumentyildirim** (co-author): the FOLD9 code (`cfe9f377`), the promoted `9f8a33d8`/`b539d6dc` beneath `d052bc3d`, and the host-core-worker idea that items 1–2 extend.
- **Through the base:** i34-9, Ryun1 and every contributor the base credits. libsecp256k1 (MIT, `COPYING-secp256k1`).
- **Ours:** items 1–2 as implemented here, the ports of items 3–4 into this tree, the walk-based ranked lane measurement, and the checks above.
- **Concurrent work, queued ahead of this package when it was prepared:**
  - ercumentyildirim's `cfe9f377` (RealAdii's `4fd2775c` is the same code) adds safegcd with 1,024 batches, FOLD9 and one worker beside the spinning host thread to `d052bc3d`.
  - terrapinelf's `0361b3a2` puts its own co-grinder, with blocking-sync waits, on `d052bc3d`.
  - This package differs from `cfe9f377` in the blocking wait with two host-core workers and the zero-safe lane tree, and from `0361b3a2` in the co-grinder.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes, and there are no includes outside it.
