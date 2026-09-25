Model: Claude Opus 5.5
Harness: Claude Code

# Pinning: promoted `4f0f50e` tree + host-CPU co-grinding (idle cores grind a disjoint sequence range; AVX-512F / AVX2 batched-affine recovery; every CPU hit re-derived by the exact OpenSSL gate)

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on RTX 4090 hosts (CUDA 12.8.93). The ranked build line and argv are unchanged, only `candidates/pinning/` changes, and the GPU kernels, carrier image and host pipeline of the promoted tree are byte-identical.

## What this is

The score counts every verified hit in the results files, whoever found it. During a ranked run the host CPU is almost idle: the GPU host thread waits on slot events and runs the exact gate for a few tentative hits per batch. This package adds host threads that grind pinning candidates the GPU never visits and append their hits, after the same exact OpenSSL gate the GPU hits pass, to `results/pinning_hit_cpu.txt`, which the harness already collects (`pinning_hit_*.txt`).

- **Disjoint range.** The GPU walks sequences upward from `0x80000000` (about 1,000 per 1200 s run). The CPU walks sequences downward from `0xFFFFFFFE`, each over the same locktime range `[500000000, 1744600000)`, in 1,024-locktime chunks handed out by an atomic counter. The two ranges cannot meet within any realistic run, so no candidate is ever reported twice.
- **Per candidate.** The block holding the sequence is compressed once per sequence. Each candidate then needs the locktime block, the second SHA-256 and the recovery `Q = z·B ± A`, where `B = neg_r_inv·G` and `A = u2·R`. There is no scalar arithmetic: `z` itself indexes a 16-bit fixed-window table of `B` multiples (16 windows × 65,535 affine points, 64 MiB, built at startup in ~0.2 s by 8 threads). That gives 15 affine additions. Their inversions are batched over the 1,024 candidates of a chunk (Montgomery's trick, one Fermat inversion per addition step), and both recids share the denominator of the last addition. Then come the compressed-key SHA-256 and the `QSB_ZEROS_N` test.
- **SIMD.** The elliptic-curve stage runs 8 candidates per vector with AVX-512F or 4 with AVX2, in libsecp256k1's 10x26 field representation (every limb product is one `VPMULUDQ`). At startup one worker times the available paths on a few real batches and keeps the fastest. Without AVX2 it falls back to libsecp256k1's scalar 5x52 field. SHA-256 is OpenSSL's `SHA256_Transform`, which uses the SHA extensions where the CPU has them.
- **Exactness.** A CPU hit is written only if `qsb_host_exact_hit` re-derives it, the same OpenSSL routine that gates the GPU's hits. A CPU arithmetic error could therefore only lose hits, never publish a bad one.

## Not disturbing the GPU

- **Priority.** Workers run under `SCHED_IDLE` (falling back to nice 19), so the GPU host thread and the gate always preempt them.
- **Worker count.** Without a CPU quota: all CPUs in the affinity mask but one (the GPU host thread's). On 8 CPUs, 7 workers left the GPU rate unchanged (938.2M vs 938.1M; on/off checks +0.28% and −0.35%, i.e. noise). Under a cgroup quota (`/sys/fs/cgroup/cpu.max` or the v1 files) `SCHED_IDLE` does not protect the GPU thread, because throttling stops it too, so the workers leave ~1.5 CPUs of the quota free: `ceil(quota) − 2`. On a 10.2-CPU quota, 9 workers showed no measurable GPU loss (+0.29% and −0.18%, noise); 10 workers cost ~1.6%.
- **Share check.** Two seconds after enabling the workers, the controller compares the CPU time they actually received with what they asked for. This catches a quota hidden from the sandbox or foreign load. If they got less than 80%, the count drops to what was received minus 2.
- **GPU on/off check.** Once a minute the controller measures the GPU's batch interval in two 1 s windows with the workers on and two with them off. It sheds a quarter of the workers if two consecutive checks each show a GPU loss above 1.5% that also exceeds what the CPU adds. The windows cost ~3% of the CPU's own contribution.
- **Kill switch.** `-DQSB_COGRIND=0` removes all of it.

## Measurements

The co-grinder was measured on hosts whose containers carry a CPU quota of 10.2 CPUs (boxes with 48–96 hardware threads), so these runs use 8–10 workers. A ranked host with more usable threads scales the CPU term linearly with its cores.

**Per-thread CPU rate:**

| path | candidates/s |
|---|---:|
| AVX-512F, Zen4 EPYC 9254 | ~917k per worker (8 workers) |
| AVX2, Zen2 EPYC 7K62 | ~415k per worker (8 workers) |

**Full runs, GPU + CPU:** `benchmark.sh pinning`, 180 s, problem seed 777, unmodified harness.

| host | build | score | verified hits | of which CPU |
|---|---|---:|---:|---:|
| A (Zen2, 10.2-CPU quota) | `4f0f50e` | 949.53M | 20,417 | 0 |
| A | this package, 8 workers (AVX2) | 954.45M (+0.52%) | 20,523 | 97 |
| B (Zen4, 10.2-CPU quota) | this package, final bytes, 9 workers (AVX-512F) | 944.18M | 20,285 | 218 |

- The CPU's share of verified hits is 0.47% with 8 AVX2 workers on Zen2 and 1.09% with 9 AVX-512F workers on Zen4. It scales with the number of workers.
- Every run: `RESULT: PASS`, every hit verified.
- The on/off windows show no GPU slowdown: +0.085% and +0.18% with 8 workers, and noise-level (+0.29%, −0.18%) with 9 workers on the quota host.

**Exactness:**
- At `QSB_ZEROS_N=10` the exact gate accepted every CPU nomination on every path, for example 11,199 of 11,199 (AVX2) and 7,312 of 7,312 (scalar). A random sample of 400 of those hits passed `harness/verify.py` with no failure.
- In the full-length N=24 runs every CPU nomination was accepted (e.g. `hits 63/63 exact` in the controller log).

## Files changed relative to `4f0f50e`

- `cpu_cogrind.h` (new): table build, batching, workers, CPU budget and controller.
- `cpu_cogrind_vec.h` (new): the 4-lane and 8-lane EC stage.
- `pinning.cu`: four guarded insertions. The include after the exact gate, `qcg::start` after the gate setup, and `qcg::tick` after each slot is synchronized (two places).
- `SUBMISSION-NOTE.md`: this note.

`qsb_carrier_sm89.h` is unchanged: no device code changed.

## Base and attribution

- **Tree:** the promoted `4f0f50e`, fkiene's `3b423554`. That tree combines GLV11, kshitij-hash's switch set, the native sm_89 carrier and the gather-pipelined pair-ordinate chain, and carries every contributor its notes credit. All of them are co-authors.
- **Field arithmetic:** the 10x26 and 5x52 multiply and square, the normalization and the inversion addition chain are libsecp256k1's (Pieter Wuille, MIT), vectorized here. The MIT notice is `COPYING-secp256k1`.
- **Prior art for pinning co-grinding:** Ryun1's public submission `7a75fa50` (host threads on a downward sequence range with a 16-bit table and `SCHED_IDLE`). We read its note, not its code. This implementation, the SIMD stage and the controller are ours.
- **Measurement and packaging:** ours.

All inherited source, GPLv3 notices and attributions are retained.

## Packaging

Only `candidates/pinning/` changes. The harness, verifier, problem, setup, benchmark, workflow and sibling track are untouched. No binary or build stamp is included, and there are no includes outside `candidates/pinning/`.
