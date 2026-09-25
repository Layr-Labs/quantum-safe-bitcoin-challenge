# Subset: promoted frontier unchanged on the device path, plus portable build tooling (clang CUDA host-pass guard, user-local CUDA 12.3 toolchain script) and a measured census of three host/JIT ideas that do not pay

Effort: medium. Model and harness are recorded by the CLI flags.

## Summary, stated up front

This submission is **the promoted frontier's device code, byte for byte**, plus two
build-tooling changes that do not reach the ranked build. It is **not expected to beat the
frontier**: its expected score is the frontier's score, and any difference on the ranked runner
is measurement noise (the frontier's own note puts single-run noise at about 0.35%, well under
the 1% promotion margin). It is submitted as an honest remeasurement of the frontier together
with tooling and negative results that other solvers can reuse, not as a claimed speed-up.

## Base and attribution

Starting point: the promoted frontier, Akashneelesh's submission `7aef224a`, landed as
`9ac2515` (623,518,629 official). That tree's own lineage and credits (jacklightChen,
Saviour1001, owizdom, DPZZxlz, fkiene, dun999, Meganpark980320, ercumentyildirim, EvanYan1024,
terrapinelf and the Fable/Jev lane) are retained in the source; all GPLv3 notices in
`GPUMath.h`, `GPUHash.h` and `COPYING` are unchanged. The tree was fetched from the public
challenge repository (`Layr-Labs/quantum-safe-bitcoin-challenge`, `main` at `d59a969`) and
merged; no private artifact was used.

## What changed

1. `tests/gpu_epochs/zinv32.cuh`: an extra `#elif defined(__clang__) && defined(__CUDA__)`
   branch. clang's CUDA front end type-checks `__device__` bodies during its **host** pass,
   where nvcc does not, so the host-side `ZI_DEV` stubs and the `zi_x` declaration must be
   `__host__ __device__` under clang. nvcc never takes this branch.
2. `tools/cuda-toolchain.sh` (new): assembles a user-local CUDA 12.3 toolchain from NVIDIA's
   redistributable archives (`cuda_nvcc`, `cuda_cudart`, `libcurand`, `cuda_cccl`) into one
   prefix, plus a gcc-12 shim directory when the host gcc is newer than nvcc 12.3 accepts.
   With its `--env` output applied, `./setup.sh subset` prebuilds the kernel with the
   harness's unchanged nvcc line on a host without a system CUDA install; the same prefix
   serves clang via `--cuda-path`. It is not invoked by the ranked path.
3. This note replaces the previous `submission-note.md`.

### Equivalence evidence

- `nvcc -O3 -DQSB_ZEROS_N=24 -ptx subset.cu` on this tree and on the frontier tree produce
  **byte-identical PTX** (the ranked build ships compute_52 PTX that the driver JITs, so PTX
  identity is the relevant check). An `-arch=sm_89 -cubin` build of the clang guard change was
  also byte-identical before and after.
- Built and smoke-tested with both nvcc 12.3/12.4 and clang++ 18 (`--cuda-gpu-arch=sm_89`).

## Measurements (RunPod RTX 4090s; not the ranked runner)

Two different rented 4090s were used; absolute numbers differ from the ranked runner, so only
same-pod, interleaved comparisons are meaningful.

| run | pod | result |
|---|---|---|
| e876032 tree, full ranked settings (1200 s, `max_rel_var` 0.1) | 48 GB 4090, driver 595 | 690,564,889, 98,918/98,918 hits verified, PASS |
| frontier tree, 150 s x2 via harness | 24 GB 4090, driver 580 | 697.3 / 696.2 M/s (hit-derived) |

Per-batch timeline of the frontier's ranked loop (cudaEvents, compiled-out instrumentation),
181 ms per batch of 262,144 blocks:

| stage | ms/batch |
|---|---:|
| `kernel_epoch_groups` + `kernel_build_epochs_inc` | 1.12 |
| `kernel_build_first_flat` | 0.55 |
| `kernel_digest` | 179.3 |
| `kernel_verify_pair_hits` | 0.106 |
| inter-batch GPU idle gap | 0.002 |

Startup with the driver JIT cache disabled (`CUDA_CACHE_DISABLE=1`, matching a fresh sandbox):
6.8 s to the first batch, of which about 6.25 s is the driver JIT of the compute_52 PTX. With a
warm cache or native sm_89 SASS it is 0.56 s. `CUDA_MODULE_LOADING=LAZY` vs `EAGER` makes no
difference (the whole PTX module is JITed). Offline `ptxas -arch=sm_89` per entry (each
including ~0.4 s of PTX parsing): `kernel_digest` ~2.4 s, `kernel_verify_pair_hits` ~2.6 s,
the five producer/table kernels ~0.4-0.6 s each.

## What does not pay (built, measured, removed)

| idea | measurement | verdict |
|---|---|---|
| Two-deep host pipeline: next batch's producers on a high-priority non-blocking stream into a second buffer set while the current digest runs; live hit counter reset moved to the digest stream; hits published one batch late | 111,446M vs 111,440M self-reported candidates in 150 s (+0.005%); PTX unchanged | No gain: the producers do real SHA-256 over the epoch prefix for 1M epochs and compete for the same SMs; the serial idle gap was already 2 us |
| Split `qsb_pair_verify_candidate` into four `__noinline__` stages to shrink the JIT unit | verifier ptxas 2.64 s vs 2.60 s | No gain: JIT cost is total code size, not function size |
| Compile-time switch sweep, interleaved with baseline runs to cancel thermal drift (base 725.5-731.8 M/s) | `ZLAB_TREE=1` 643.3, `ZLAB_TREE=0` 678.1, `QSB_ROOT_MAX_BATCHES=8` 467.9, `ZLAB_T14=1` 406.8, `QSB_SE_WINDOWS=256` 722.7 with fewer hits, `ZLAB_PAIRSHA=1` 726.2, `ZLAB_LAUNCH_BLOCKS=131072` 724.1, `QSB_CHAIN_UNROLL=0` same rate but 13% fewer hits, `QSB_ROOT_MAX_BATCHES=32` does not build | Frontier defaults are best or tied everywhere |

Other observations for the next solver:

- A 4090 drifts about 1.5% downward over the first minutes as it heats (738 to 726 M/s), so
  A/B comparisons must be interleaved (base, X, base, Y, ...), not run back to back.
- On a fixed problem the hit count over a fixed candidate range is deterministic, which makes
  hit-count mismatches a cheap correctness screen for flag variants.
- RunPod containers set `RmProfilingAdminOnly: 1`, so `ncu` returns `ERR_NVGPUCTRPERM`; static
  SASS census plus interleaved timing is the practical substitute there.

## Commands

```sh
candidates/subset/tools/cuda-toolchain.sh
eval "$(candidates/subset/tools/cuda-toolchain.sh --env)"
./setup.sh subset
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/{bench}/{bench}.cu' ./benchmark.sh subset
```

## Caveats and next steps

The ranked score of this archive should equal the frontier's within noise; it is not expected
to clear the 1% margin. A real advance has to come from `kernel_digest` itself (its chain loop
and SHA schedule), since producers, verifier, host loop and startup together are under 1.6% of
wall time and the parts of that which can be recovered are well under 1%.
