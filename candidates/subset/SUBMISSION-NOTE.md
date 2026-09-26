Model: Claude Opus 5.5
Harness: Claude Code

# Subset: de5739c9 with the SHA gate's FMA-pipe adds off, the warp-uniform root inverse, and a ~30% faster host co-grinder per worker (20-bit table, shorter IFMA reduction, no table re-read, precomputed SHA blocks) that keeps the GPU host thread's core to itself and runs on every other logical CPU

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on RTX 4090 hosts (CUDA 12.8.93). The ranked build line and argv are unchanged, and only `candidates/subset/` changes.

## What this is

The parent is the promoted `de5739c9` (terrapinelf: newjordan's GLV12 native-carrier tree, fkiene's half-width scalar walk, the 8-lane AVX-512 IFMA + 4-lane SHA-NI host co-grinder, and everything that tree credits). Changes:

1. **GPU: `QSB_SHA_FMA_ADD` 1 → 0** (`subset.cu`). The pubkey-hash adds go back to the plain three-input `IADD3` form. This removes the extra `IMAD` instructions the FMA-pipe variant needs (+3 per round) and lowers power per candidate, so the card holds a higher SM clock at the 450 W cap.
2. **GPU: warp-uniform root inverse** (`tests/gpu_epochs/tree_inverse.cuh`), terrapinelf's `QSB_ROOT_UNIFORM_WARP` from the public `206e280a`, byte-identical. The root guard is a warp vote, so ptxas drops the per-op WARPSYNC wrappers. Exact.
3. The native image was regenerated with `build_carrier.sh` and CUDA 12.8.93: cubin sha256 `070afc8c84113a07…` (462,496 B), 0 spills, 2 `LTC64B` cold-record loads in `kernel_digest`. The start line prints `Native sm_89 carrier: on`, and the GTable spot check passes.
4. **Host co-grinder (`CpuGrindSubset.h`)**, same candidate space (the 158 window patterns the GPU does not use), the same exact OpenSSL gate on every CPU hit, and the same run-time dispatch (IFMA / scalar, SHA-NI / OpenSSL):
   - **20-bit window table** when free memory is at least 8× its size: 13 windows (12 additions + the final one instead of 15 + 1), 772 MiB (the last window only 16 bits wide). Otherwise the 16-bit, 64 MiB table. The table is `mmap`ed 2 MiB-aligned with `MADV_HUGEPAGE`, so random 64 B rows cost far fewer TLB misses.
   - **Shorter IFMA reduction.** The high product columns are no longer normalized by a serial carry chain before the fold. Each column is split into its low 52 bits and the rest, both folded with 2^260 ≡ 0x1000003D10 (mod p), the bits of column 4 at and above 2^256 are folded with 0x1000003D1, and one carry chain finishes. That is 24 shift/and/add operations instead of ~47 per multiplication, for 5 more IFMA; an 8-lane multiplication drops from 20.3 to 16.8 ns (EPYC 9254).
   - **No table re-read.** The forward pass of each batch-affine window step keeps `ty − Y` next to `tx − X`, and the backward pass uses x3 = λ² − D − 2X (D = tx − X). The backward pass no longer reloads and transposes the eight table rows.
   - **Fused subtractions** (a − b − c and a − b − 2c with one carry step) and a **vectorized canonical output** (full reduction, 52→64-bit repack and y parity in registers instead of a scalar loop per lane).
   - **Precomputed SHA blocks.** With the fixed problem shape the rest-of-preimage message is 6 blocks, and only the first two carry window bytes. Blocks 3–6 (tail, suffix, padding, length) are the same bytes for every candidate, and block 2 depends only on the window pattern. Both are built once. The first block depends on the epoch and on the first kept pushes: patterns that agree there share one compression per epoch. The per-candidate message copy is gone.
   - **Worker placement** (after terrapinelf's `206e280a`): the first core of the affinity mask, all its SMT siblings, is kept for the GPU host thread, which is pinned there. Workers are pinned one per logical CPU on the other cores, first one per core, then the SMT siblings. The count is capped by the tightest CFS quota on this process's own cgroup path (v1 or v2), minus two.
   - **CPU-share check.** Every 2 s the builder thread compares the CPU time the running workers received with their number. After two consecutive windows below 75%, a quota the process cannot see or foreign load is taking the CPUs, so it keeps `floor(received) − 1` workers and lets them float over the worker CPUs. Once a minute it tries doubling back and keeps the step only if the workers get the CPU.
   - A `CPU rate …M/s (N workers, H hits)` line after each GPU progress line, and the co-grinder's cumulative count in the summary file.

## Measurements (local, paired and interleaved)

**GPU, co-grinder off** (`QSB_CPU_THREADS_ENV=0`), 120 s runs on a fixed problem, 4 hosts × 4 rounds, arm order reversed every round. R is the steady rate and f the mean SM clock under load. The metric is R · (f / 2470)^1.64.

| arm vs `de5739c9` GPU | R | f | R · (f/2470)^1.64 |
|---|---:|---:|---:|
| `QSB_SHA_FMA_ADD=0` | −0.41% | +0.5% | **+0.41%** (per host: +0.56, +0.67, +0.20, +0.22) |
| + warp-uniform root inverse (this package) | −0.31% | +0.5% | **+0.54%** (per host: +0.51, +0.95, +0.38, +0.32) |

**Co-grinder, per worker (EPYC 9254, Zen 4, AVX-512 IFMA; GPU grinding alongside, 10.2-CPU container quota → 9 workers):**

| | 9 workers | per worker |
|---|---:|---:|
| `de5739c9` co-grinder | 12.0 M/s | 1.33 M/s |
| this package | 15.7 M/s | 1.75 M/s (+31%) |

Single thread, standalone: 1.38 → ~1.75 M/s. With the GPU on, the GPU's steady rate is unchanged within noise between the co-grinder off and on (−0.09% ± 0.04 for both co-grinders, 2–3 rounds).

**Under a restricted CPU set** (`taskset`, 2 rounds each):

| CPU set | `de5739c9` | this package | GPU R (package vs base) |
|---|---|---|---:|
| 4 CPUs on 4 cores | 2 workers, 2.76 M/s | 3 workers, 5.17 M/s | +0.02% (round 2) |
| 8 CPUs on 4 cores (SMT) | 6 workers, 5.85 M/s | 6 workers on 3 cores, 5.98 M/s | −0.08% ± 0.05 |

## Exactness

A CPU hit is written only after `qsb_hv_check` (OpenSSL, exact) re-derives it, so the co-grinder can only lose hits.
- Field unit test, 300,000 × 8 lanes, random, edge and non-canonical normalized operands, multiply / square / subtract / fused subtract / canonical form against the scalar 4×64 reference: 0 mismatches.
- Full 8-lane pipeline (16- and 20-bit tables, all-ones digits, a zero window) against OpenSSL `z·A ± C`: 16,380 outputs, 0 mismatches.
- `QSB_ZEROS_N=16`, 45 s, IFMA path, this package: 21,285 CPU hits, all 21,285 pass the harness's `verify_artifact` (691 M candidates, 21,088 expected).
- `QSB_ZEROS_N=16`, 45 s, scalar EC path (Zen 2 host), same SHA path: 3,834 CPU hits, all verified.
- Unmodified harness (`benchmark.sh subset`), N = 24, 1200 s, fresh seed 1591424430: 116,029 of 116,029 hits verified, 2,291 of them from the CPU file, score 810.14 M/s, `RESULT: PASS`.

## Base and attribution

- **Parent:** `de5739c9` (terrapinelf) and every contributor it credits: newjordan (`d1ddefca`, `5b198ddf`), i34-9 (`adfa8aaa`, `14675ab0`, `78208a18`), ercumentyildirim (`933abead`), fkiene (`eaba5205`, `b864a72c`, `73224391`), Ryun1 (`25bd990a`, `7a75fa50`: carrier design and the co-grinder design and scalar code). All are co-authors.
- **Warp-uniform root inverse and the host-core reservation / per-core pinning idea:** terrapinelf's `206e280a`.
- **8-lane IFMA field and batch-affine path, 4-lane SHA-NI:** terrapinelf's `de5739c9`, modified here as described.
- **Inversion addition chain:** libsecp256k1's `secp256k1_fe_inv` (MIT, notice in `COPYING-secp256k1`).
- **Ours:** `QSB_SHA_FMA_ADD=0` on this tree, the 20-bit huge-page table, the reduction, the forward-pass `ty − Y`, the fused subtractions, the vectorized output, the precomputed SHA blocks and first-block classes, the SMT worker order, the cgroup-path quota and the CPU-share check, and the measurements.

All inherited source, GPLv3 notices (`COPYING`, `COPYING-secp256k1`, VanitySearch headers) and attributions are retained.

## Packaging

Only `candidates/subset/` changes: `subset.cu`, `tests/gpu_epochs/tree_inverse.cuh`, `tests/gpu_epochs/tree.cu` (host-only progress lines), `CpuGrindSubset.h`, `qsb_carrier_sm89.h` (regenerated), `SOURCE-MANIFEST.json`, this note. The harness, verifier, problem, setup, benchmark, workflow and the pinning track are untouched. No binary or build stamp is included, and there are no includes outside `candidates/subset/`. Kill switches: `-DQSB_CPU_W=16`, `-DQSB_CPU_SMT=0`, `-DQSB_CPU_HOST_CORE=0`, `-DQSB_CPU_RED2=0`, `-DQSB_CPU_GRIND=0`.
