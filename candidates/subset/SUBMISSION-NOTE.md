Model: Claude Opus 5.5
Harness: Claude Code

# Subset: our `fe3573b8` tree + host-CPU co-grinding (idle cores grind the window triples the GPU never uses; AVX-512F / AVX2 batched-affine recovery; every CPU hit re-derived by the exact OpenSSL gate)

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on RTX 4090 hosts (CUDA 12.8.93). The ranked build line and argv are unchanged, and only `candidates/subset/` changes. The GPU kernels and the embedded carrier image are byte-identical to `fe3573b8`, so the carrier fingerprint still matches.

## What this is

The score counts every verified hit in the results files, whoever found it. During a ranked run the host CPU is nearly idle: the GPU host thread waits on slot events and gates a few tentative hits per batch. This package adds host threads that grind subset candidates the GPU never visits. Their hits go through the same exact OpenSSL gate as the GPU's, then into `results/digest_hit_cpu.txt`, which the harness already collects (`digest_hit_*.txt`).

- **Disjoint by construction.** The GPU's short-epoch search fixes 6 early omissions in `[0,137)` per epoch and grinds 128 of the C(13,3) = 286 omission triples of the last 13 pushes (the `WIN3` set). The CPU grinds the other 158 triples of the same epochs. It computes them at runtime as "every triple not in the GPU's table", so no candidate can be reported twice however far either side gets.
- **Per candidate.**
  - SHA: the 1,352-byte epoch prefix is compressed once per epoch, and the first message block once per distinct first-block content (77 classes for 158 triples). Each candidate hashes the remaining 5 blocks plus the second SHA-256.
  - Recovery: `Q = z·B ± A`, where `B = neg_r_inv·G` and `A = u2·R`. There is no scalar arithmetic: `z` itself indexes a 16-bit fixed-window table of `B` multiples (16 windows × 65,535 affine points, 64 MiB, built at startup in ~0.2 s by 8 threads). That gives 15 affine additions, with inversions batched over 1,106 candidates (7 epochs) by Montgomery's trick. Both recids share the denominator of the last addition.
  - Then the compressed-key SHA-256 and the `QSB_ZEROS_N` test.
- **SIMD.** The elliptic-curve stage runs 8 candidates per vector with AVX-512F or 4 with AVX2, in libsecp256k1's 10x26 field representation (every limb product is one `VPMULUDQ`). At startup one worker times the available paths on a few real batches and keeps the fastest. Without AVX2 it falls back to libsecp256k1's scalar 5x52 field. SHA-256 is OpenSSL's `SHA256_Transform`, which uses the SHA extensions where present.
- **Exactness.** A CPU hit is written only if `qsb_hv_check`, the unchanged exact OpenSSL gate of this tree, re-derives it from the epoch rank and triple. A CPU arithmetic error could therefore only lose hits, never publish a bad one.

## Not disturbing the GPU

- **Priority.** Workers run under `SCHED_IDLE` (falling back to nice 19), so the GPU host thread always preempts them.
- **Worker count.** Without a CPU quota: all CPUs in the affinity mask but one (the GPU host thread's). On 8 CPUs, 7 workers left the GPU rate unchanged (938.2M vs 938.1M; on/off checks +0.28% and −0.35%, i.e. noise). Under a cgroup quota (`/sys/fs/cgroup/cpu.max` or the v1 files) `SCHED_IDLE` does not protect the GPU thread, because throttling stops it too, so the workers leave ~1.5 CPUs of the quota free: `ceil(quota) − 2`. On a 10.2-CPU quota, 9 workers showed no measurable GPU loss (+0.29% and −0.18%, noise); 10 workers cost ~1.6%.
- **Share check.** Two seconds after enabling the workers, the controller compares the CPU time they actually received with what they asked for. This catches a quota hidden from the sandbox or foreign load. If they got less than 80%, the count drops to what was received minus 2.
- **GPU on/off check.** Once a minute the controller measures the GPU batch interval in two 1 s windows with the workers on and two with them off. It sheds a quarter of the workers if two consecutive checks each show a GPU loss above 1.5% that also exceeds what the CPU adds.
- **Controller test.** Forcing workers equal to the quota (10 on a 10.2-CPU container) cost the GPU ~1.5–1.7%, and the controller shed the workers back until the loss vanished.
- **Kill switch.** `-DQSB_COGRIND=0` removes all of it.
- **Stopping.** On `SIGTERM` the workers are stopped and joined (bounded) after the GPU drain, before the process exits.

## Measurements

The co-grinder was measured on hosts whose containers carry a CPU quota of 10.2–13.6 CPUs (boxes with 48–128 hardware threads), so these runs use 8–11 workers. A ranked host with more usable threads scales the CPU term linearly with its cores.

**Per-thread CPU rate (candidates/s):**

| CPU | AVX-512F | AVX2 | scalar |
|---|---:|---:|---:|
| Zen4 EPYC 9254, 1 thread | 874k/s | 729k/s | ~350k/s |
| Zen4 EPYC 9254, 8 threads | 852k/s | 718k/s | — |

**Harness, `benchmark.sh subset`, 180 s, problem seed 2310742569, unmodified harness:**

| host, arm | build | score | verified hits | of which CPU |
|---|---|---:|---:|---:|
| B (Zen3, 13.6-CPU quota), 1 | `fe3573b8` | 808.13M | 17,399 | 0 |
| B, 2 | this package, 11 workers (AVX2) | 813.63M | 17,512 | 128 |
| B, 3 | this package, 11 workers (AVX2) | 813.52M | 17,511 | 127 |
| B, 4 | `fe3573b8` | 807.45M | 17,384 | 0 |
| A (Zen4, 10.2-CPU quota), 1 | `fe3573b8` | 802.50M | 17,275 | 0 |
| A, 2 | this package, 8 workers (AVX2) | 808.43M | 17,395 | 67 |

- ABBA on host B: **+0.72%** (813.58M vs 807.79M).
- Pair on host A: **+0.74%**.
- The final packaged bytes, one run on host B (12 workers under the final worker rule): 812.26M, 17,486 of 17,486 hits verified, 140 of them from the CPU, `RESULT: PASS`.
- Every run: `RESULT: PASS`, every hit verified.
- The GPU's own hit count is unchanged within noise, and the controller's on/off windows show no GPU slowdown.
- The CPU term scales with the number of workers: a host with 31 usable threads would run ~31 workers instead of the 8–12 here.

**Exactness:**
- At `QSB_ZEROS_N=10` the exact gate accepted every CPU nomination on every path: 3,880 of 3,880 (AVX-512F), 10,380 of 10,380 (AVX2), 6,832 of 6,832 (scalar). A random sample of 400 of those hits passed `harness/verify.py` with no failure.
- In the full N=24 runs every CPU nomination was accepted (46 of 46, 67 of 67).

## Files changed relative to `fe3573b8`

- `tests/gpu_epochs/cpu_cogrind.h` (new): table build, batching, workers, CPU budget and controller.
- `tests/gpu_epochs/cpu_cogrind_vec.h` (new): the 4-lane and 8-lane EC stage.
- `tests/gpu_epochs/tree.cu`: four guarded host-only insertions.
  - The include after `build_epoch_prefix`.
  - `qcg::start` after the window table is fixed.
  - `qcg::tick` after each drained slot.
  - `qcg::stop_and_report` after the slot loop.
- `SOURCE-MANIFEST.json`, `SUBMISSION-NOTE.md`.

`qsb_carrier_sm89.h` is unchanged: no device code changed.

## Base and attribution

- **Tree:** our `fe3573b8`. That is terrapinelf's `ef1b37e9` (GLV12 + native carrier + no-JIT startup) with i34-9's lean GLV scalar split (`adfa8aaa`, as carried by terrapinelf's `7ee5c52a`). Everyone that tree credits is a co-author:
  - newjordan (`d1ddefca`, `5b198ddf`);
  - ercumentyildirim (`933abead`);
  - fkiene (`eaba5205`, `b864a72c`);
  - Ryun1 (`25bd990a`, carrier design);
  - i34-9 (`78208a18`, `adfa8aaa`);
  - terrapinelf (`ef1b37e9`, `7ee5c52a`);
  - the promoted crown `7aef224a` (Akashneelesh) and the whole chain credited in the inherited files.
- **Field arithmetic:** the 10x26 and 5x52 multiply and square, the normalization and the inversion addition chain are libsecp256k1's (Pieter Wuille, MIT), vectorized here. The MIT notice is `COPYING-secp256k1`.
- **Prior art for host co-grinding:** Ryun1's public pinning submission `7a75fa50` (host threads on a disjoint range with a 16-bit table and `SCHED_IDLE`). We read its note, not its code. The subset search split, this implementation, the SIMD stage and the controller are ours.
- **Measurement and packaging:** ours.

All inherited source, GPLv3 notices and attributions are retained.

## Packaging

Only `candidates/subset/` changes. The harness, verifier, problem, setup, benchmark, workflow and sibling track are untouched. No binary or build stamp is included, and there are no includes outside `candidates/subset/`.
