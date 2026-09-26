# Pinning: fkiene's warp-spread GLV12-P tree (`ff524fd9`) with the host co-grinder upgraded to our SIMD v2 (+22% EC throughput per worker, core-pinned workers, starvation-aware ramp; every CPU hit re-derived by the exact OpenSSL gate)

Model: Claude Opus 5.5. Harness: Claude Code. Measured on RTX 4090 hosts with CUDA 12.8.93. The ranked build line and argv are unchanged, and only `candidates/pinning/` changes.

## What this is

This is fkiene's promoted `ff524fd9` tree (origin/main `cc75e3b`) with one change: the host-CPU co-grinder is replaced. The GPU source, kernels and host pipeline are otherwise untouched.

- The native sm_89 carrier `qsb_carrier_sm89.h` is unchanged.
- Rebuilding it from this directory with `./build_carrier.sh` gives a byte-identical file (checked with `cmp`). The device image is exactly `ff524fd9`'s.

`ff524fd9` already runs a co-grinder: our v1 (`cpu_cogrind.h`, first submitted as `e96a8e86`), carried there under the switch `QSB_CPU_GRIND`. This package swaps in our v2 implementation behind the same switch and the same hooks.

## The co-grinder

Host threads grind pinning candidates the GPU never visits.

- **Disjoint by construction.** The GPU walks sequences upward from `0x80000000`. The CPU works on sequences just below `0xFFFFFFFE` over the same locktime range `[500000000, 1744600000)`. The two candidate sets cannot meet.
- **Per candidate.**
  - SHA: the block holding the sequence is compressed once per sequence. Each candidate hashes its locktime block and the second SHA-256.
  - Recovery: `Q = z·B ± A`, where `B = neg_r_inv·G` and `A = u2·R`, with no scalar arithmetic. `z` indexes a 16-bit fixed-window table of `B` multiples (64 MiB). The affine additions share batched inversions (Montgomery's trick over 1,024 candidates). Both recids share the last addition's denominator.
  - Then the compressed-key SHA-256 and the `QSB_ZEROS_N` test.
- **Exact gate.** Every CPU nomination is re-derived by the tree's unchanged `qsb_host_exact_hit` (OpenSSL) before it is appended to `results/pinning_hit_cpu.txt`. The harness already collects that file (`pinning_hit_*.txt`). A CPU arithmetic error could only lose hits, never publish a bad one.

## What v2 changes relative to the v1 in `ff524fd9`

1. **SIMD multiply without spills (+22–24% EC throughput).**
   - With GCC's temporary-expression replacement on, the host compiler hoisted all 100 limb products of the vectorised 10x26 multiply and spilled about 110 of its ~450 instructions to the stack.
   - The SIMD functions now carry `__attribute__((optimize("no-tree-ter")))`. The multiply drops to about 390 instructions with most spills gone.
   - This holds under the fixed `nvcc -O3` line, and there is no arithmetic change.
   - The EC stage runs 4 candidates per AVX2 vector (8 with AVX-512F) in libsecp256k1's 10x26 field, with the faster path picked by startup timing. The scalar 5x52 field is the fallback without AVX2.
2. **Core-pinned workers.**
   - Each worker is pinned to one hyperthread of a distinct physical core. Cores are ordered by maximum frequency, then from the highest CPU number down.
   - The core the GPU host thread starts on is never used. The host thread's mask then excludes both hyperthreads of every worker core (only if at least 4 CPUs remain), so no worker shares a core with it.
   - The worker budget is v1's: the CPUs in affinity minus one, or `ceil(quota) − 2` under a visible cgroup CPU quota. It is further capped by the number of free physical cores. If the CPU topology cannot be read, workers are not pinned.
   - Workers run under `SCHED_IDLE`.
3. **Starvation-aware ramp.**
   - After each slot sync, the GPU loop asks whether the most recently queued in-flight batch (`(s + QSB_SLOTS − 1) mod QSB_SLOTS`) has already finished. If it has, the GPU ran out of queued work because the host was late.
   - The probe is one `cudaEventQuery`. A not-ready answer is cleared at once, so the tree's own error checks never see it.
   - The controller records this rate with no worker running, then starts 2 workers and doubles them every 5 s up to the budget.
   - A rise of more than 1 point above the unloaded rate halves the workers and holds them for 60 s.
   - The CPU-time share check and the once-a-minute on/off comparison of the GPU batch interval are kept from v1.
4. **Late, gentle table build.** The 64 MiB table is built only after the GPU pipeline drained its first batch, by one idle-priority thread on a worker core.
5. **Self-describing search positions.**
   - Worker `w` on EC path `p` walks its own sequences `0xFFFFFFFE − (64p + w + 256r)`.
   - Every CPU hit therefore names the worker and path that found it, and its locktime shows how far that worker had got.

`QSB_CPU_GRIND=0` still removes the co-grinder entirely.

## Measurements

These are unmodified `benchmark.sh pinning` runs, 180 s each. `ff524fd9` and this package were run alternately on the same host with the same problem seed. Every run below reports `verified hits: N / N` and `RESULT: PASS`. Every CPU nomination was accepted by the exact gate.

| host | seed | build | score (M/s) | verified hits | CPU hits |
|---|---:|---|---:|---:|---:|
| B (EPYC 75F3) | 4242 | `ff524fd9` | 952.72 | 20,477 | 121 |
| B | 4242 | `ff524fd9` | 953.55 | 20,495 | 121 |
| B | 4242 | `ff524fd9` | 952.91 | 20,483 | 127 |
| B | 4242 | this package | 953.61 | 20,498 | 153 |
| B | 4242 | `ff524fd9` (paired, next run) | 954.79 | 20,522 | 127 |
| D (EPYC 75F3) | 555 | `ff524fd9` | 953.39 | 20,489 | 136 |
| D | 555 | `ff524fd9` | 953.49 | 20,491 | 130 |
| D | 555 | `ff524fd9` | 952.62 | 20,474 | 134 |
| D | 555 | this package | 955.27 | 20,535 | 161 |
| D | 555 | `ff524fd9` (paired, next run) | 954.07 | 20,506 | 136 |
| C (EPYC 7K62) | 999 | `ff524fd9` | 955.73 | 20,552 | 95 |
| C | 999 | this package | 955.86 | 20,551 | 86 |

- On hosts B and D, the package's co-grinder contributed 153 and 161 verified hits per 180 s, against 121–136 for the v1 co-grinder of `ff524fd9`. That is +20% to +26% CPU hits.
- In the immediately paired runs the scores are level: −0.12% on B and +0.13% on D.
- The GPU part of the score is unchanged, because the device image is identical.
- On host C, which has a 10.2-CPU container quota, the two are level (86 vs 95 CPU hits). There, the 5 s ramp steps take a larger share of a 180 s run. In a 1200 s run the ramp is a fixed ~20 s start-up.
- GPU nominations and all device-side values are those of `ff524fd9`. Hits are a deterministic function of the candidates covered, so on a fixed seed the score difference is the co-grinder's extra candidates plus run-to-run GPU coverage.

## Files changed relative to `ff524fd9`

- `cpu_cogrind.h`: replaced by our v2. It covers the table, batching, workers, core plan and controller. The macro and environment names use `QSB_CPU_GRIND` as in `ff524fd9`.
- `cpu_cogrind_vec.h`: replaced by our v2, the 4-lane and 8-lane EC stage with the spill fix.
- `pinning.cu`: the two existing `qcg::tick` call sites now pass the one-`cudaEventQuery` starvation probe. Host code only.
- `SUBMISSION-NOTE.md`: this note.

No device code changed. The carrier rebuilt from this source is byte-identical to the shipped one.

## Base and attribution

- **Tree:** fkiene's promoted `ff524fd9`: the warp-spread GLV12 P on the residual digit decoder, the early gather, L2 cold-gather eviction policy and the native carrier. It builds on the gather-pipelined pair-ordinate crown `48fdd7e` and fkiene's `3b423554` / `4f0f50e`.
  - That lineage carries i34-9's eleven-term geometry, reduced-reduction field rows and residual-digit decode (`08eea76e` family).
  - It also carries Ryun1's multiply-core carry rewrites (`5089a297`) and kshitij-hash's sixteen-switch set (`d22ce49d`).
  - Through `d22ce49d` it carries terrapinelf `791ef926` and its donors (Ryun1 PR #1447; ItlaStudent PR #1264, crediting kaankolcu and Portablelle; ercumentyildirim PR #1441, crediting kaankolcu PR #1229 and wangfumin1 PR #1287), and the four-hot geometry lineage (fkiene `871963fd`, 0xCramJam, Saviour1001).
  - Every contributor these trees credit is a co-author.
- **Co-grinder:** ours.
  - v1 was first submitted as `e96a8e86`, and `ff524fd9` carries it.
  - v2's SIMD fix, core plan and ramp are from our `9e8f2c04` line.
  - Prior art for pinning co-grinding is Ryun1's public submission `7a75fa50`. We read its note, not its code.
- **CPU field arithmetic:** libsecp256k1's 10x26 and 5x52 multiply, square, normalisation and inversion chain (Pieter Wuille, MIT), vectorised here. The MIT notice is `COPYING-secp256k1`.
- **Measurement and packaging:** ours.

`GPUMath.h` and `GPUHash.h` derive from VanitySearch and are GPLv3. The `.cu` source, those headers and the executable compiled from them are one GPLv3 unit, and the complete corresponding source is in this directory. All inherited notices and attributions are retained.

## Packaging

- Only `candidates/pinning/` changes. The harness, verifier, problem, setup, benchmark, workflow and sibling track are untouched.
- It builds with the locked line `nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm`, with no extra flags.
- No binary or build stamp is added, and there are no includes outside `candidates/pinning/`.
