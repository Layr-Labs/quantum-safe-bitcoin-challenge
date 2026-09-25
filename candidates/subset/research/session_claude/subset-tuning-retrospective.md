# Subset tuning retrospective — 2026-09-26

Every entry below is a local RTX 3090 measurement of the *promoted frontier source*
(or one line changed in it), 60 s or 300 s arms through the unmodified harness, with
every emitted hit re-derived by the unchanged verifier. They are recorded so a later
session does not re-run them.

## Measurement traps that cost this session most of its time

1. **The cold-JIT claim was a compile.** A/B arms that used a fresh tag compile a new
   binary; the ~9.4 s that appeared as "harness elapsed" on the first arm of a set was
   `nvcc`, not a driver JIT. Measured directly with a *pre-warmed* binary under
   `timeout 20`: 20.06 s warm vs 20.04 s cold (`CUDA_CACHE_DISABLE=1`) — the module's
   JIT is under 0.05 s for this source here. `cuobjdump --list-elf/--list-ptx`
   confirms the ranked build embeds `sm_52` cubins and PTX, so a JIT does exist; it is
   just cheap locally.
2. **The verified-hit count is quantised by the completed-batch boundary.** The two
   arms of a pair routinely land on identical counts (1855, 1855, 1855) and the
   differences that do appear are one quantum. Lowering `N` does **not** improve
   resolution: the quantum is a batch, whose hit content scales with `N`. At `N=22`
   three of four arms in `OFF/ON/ON/OFF` were identical (6722).
3. **The kernel's own periodic progress rate is the better local metric** — it has
   ~0.3% resolution instead of ~1%, and it is what made the table-footprint
   diagnostic below possible.
4. Session-to-session drift with the *same* binary is ±3%; the first arm of a session
   runs on a cooler card.

## Variants measured, and what they cost

| Variant | Local result |
|---|---|
| `ZLAB_LAUNCH_BLOCKS` 262144 → 32768 | −0.08% (neutral) |
| → 65536 | ~neutral |
| → 16384 / 8192 / 4096 | −1.8% / −1.7% / −3.6% |
| `ZLAB_T14` (14-term, 144 MiB table) | −5.1% |
| `ZLAB_PAIRSHA` (interleaved two-recv SHA) | neutral |
| `__ldcs` + `__stcs` on the first-state buffer | −1.2% |
| `__stcs` only on the first-state writes | −2.1% |
| host `cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 64)` | neutral-to-negative |
| native `-arch=sm_86` vs compute_52 JIT | not resolvable |
| the sm_89 carrier with `ld.global.nc.L2::64B` (`.L2..64B` on all table loads) | not resolvable |
| module JIT cost | under 0.05 s |

## What did show a large, unambiguous effect

Masking the table record offset so the lookups land in a small window (arithmetic
unchanged, values wrong, only the rate read):

| Footprint | Rate at 15 / 30 / 45 s | vs stock |
|---|---|---:|
| 64 MiB (stock) | 253.2 / 253.6 / 252.4 M/s | — |
| 8 MiB (still over the 6 MiB L2) | 264.3 / 265.1 / 264.5 M/s | +4.3% |
| 4 MiB (fits) | 299.0 / 296.9 / 297.2 M/s | **+17.4%** |

The step appears where the footprint crosses the cache capacity, so the stock kernel
is **latency**-bound on table access (253 GB/s against 936 GB/s of DRAM bandwidth).
That, and the fact that the batch's own state is 512 MiB against a 72 MiB L2, is the
whole argument for the submitted `ZLAB_LAUNCH_BLOCKS` change.
