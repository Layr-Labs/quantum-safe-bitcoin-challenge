# v22 — init dead-time elimination on the promoted GLV12 stack

## What this package is

v20 bytes (promoted GLV12 `QSB_BIGTBL` + `GLV_LEAN`/`HIGH10_HI`/`ROUND_CC` +
`SFC2_DROP` on the v19 union stack) plus a pure **initialization-time**
bundle. No kernel math changed; every change is host-side or a no-op hint.

## Why: the 0.976 deficit is dead time, not lost hits

Every package in the field — ours, the promoted frontier, every rejected
draw — shows `official / self_reported ≈ 0.976`. The harness scores
`verified_hits × 2^23 / elapsed_wall`, where `elapsed` covers the entire
subprocess: spawn, CUDA init, **PTX JIT**, table build, allocs, plus the
1200 s `timeout` grind. Cross-checking v20's own record:
`126352 hits × 2^23 = 1.0597e12 candidates ÷ 903.8M/s = 1172.6 s of actual
grinding vs 1201.6 s wall` → **~29 s of zero-candidate dead time inside the
scored window**, worth +2.4% if recovered — the promotion margin.

## Changes (all init-time; the hot kernel is untouched)

1. **Native sm_89 cubin + driver-API launches.** The ranked build
   (`nvcc -O3 -DQSB_ZEROS_N=24`, no `-arch`) embeds sm_52 SASS that cannot
   execute on sm_89; the driver JIT-compiles ~24K lines of PTX inside the
   window (est. 5–15 s). `pinning_sm89.cubin` (built from THIS source,
   `research/build_sm89_cubin.sh`, e_flags sm_89, all six kernels + all
   runtime-uploaded constants verified present) is loaded through
   `cudaGetDriverEntryPoint` → `cuModuleLoadData`/`cuLaunchKernel` — a cudart
   call, no `-lcuda`/`-ldl` needed. Located via `argv[0]` dir →
   `/proc/self/exe` → cwd fallback. Every upload also mirrors into the cubin
   module via `cuMemcpyHtoD` + readback-compare; any failure anywhere flips
   `g_use_cubin` and the launch falls back to `<<<>>>`. The gtable build is
   the first cubin launch and its OpenSSL spot-check doubles as the
   end-to-end self-test of the path.
2. **Two-phase table build.** The 1.465 GiB device build launches on a
   private non-blocking stream immediately after the host ladders upload;
   all other init (constants, slot streams/events, ~1.5 GiB pipeline
   buffers, L2 policy, host gate) proceeds while the GPU works. Sync +
   verify sit just before the search loop.
3. **Sampled readback.** The spot-check no longer mallocs and copies the
   full 1.465 GiB table to verify ~216 records — each sampled record is
   fetched with its own 64-byte `cudaMemcpy` (~14 KiB total). Same (ch,i)
   sample sequence, same OpenSSL comparison, same fallback on failure.
4. **Threaded ladder build.** The six per-chunk L/H ladders are
   independent; each now runs on its own `std::thread` with private OpenSSL
   objects (serial fallback if the runtime cannot spawn threads).
5. **`cudaMallocAsync`** on the slot streams for the ~1.5 GiB pipeline
   buffers (blocking `cudaMalloc` fallback preserved).
6. **SIGTERM drain.** The harness kills the window with `timeout` SIGTERM;
   a handler now drains the ≤2 in-flight batches and publishes their hits
   before exiting (~2 expected hits/run previously lost mid-pipeline).
7. **`QSB_TBL_PREFETCH=0`.** The v21 prefetch burst is retired (measured
   −37%: LSU/L2 queue flood under ~19 warps/SM occupancy).

## Verification

- sm_89 build: `kernel_pinning_pipeline<true,0>` = **106 regs, 0 spills** —
  identical to v20 (all changes are host-side).
- `nvcc -O3 -DQSB_ZEROS_N=24` (exact ranked line): builds clean.
- `-DQSB_BIGTBL=0` (GLV14 baseline): builds clean, same symbol set.
- Cubin self-verification: `build_sm89_cubin.sh` asserts e_flags=sm_89 and
  every kernel symbol + every runtime-uploaded constant is present.
- The GLV12 recode oracle is unchanged (math untouched): 400,061 magnitudes
  ×2 signs exact, bound 10,659,985.
- Fail-safe posture: cubin engagement is all-or-nothing with loud fallback;
  a stale/mismatched cubin fails the gtable spot-check and reverts.

## Expected effect

Recovering ~20–28 s of dead time maps to ~+1.7–2.3% official
(each in-window second is ~0.083%). On the fast worker class (~904M self)
that puts the official estimate at ~897–904M vs the 890.09M floor; on a
slow draw (~870M self) it lands ~885–892M — a genuine contest of the floor
rather than a guaranteed miss. The yield ratio `official/self` on the
artifact doubles as the engagement measurement: ~0.99 means the levers
fired, ~0.976 means they did not.

## Risk analysis and dead-time accounting

**Decomposition of the ~29 s deficit** (bounded by the 0.976 haircut; the
split between JIT and serial init is not measurable without a GPU probe):

| Item | Estimated | Mechanism |
|---|---|---|
| PTX→SASS JIT | ~5–15 s | `nvcc -O3` ships sm_52 cubin + compute_52 PTX; the driver compiles ~24K PTX lines for sm_89 at first launch, inside the window. Eliminated by the shipped cubin. |
| GPU table build | ~1.5–3 s | 22.9M-entry GLV12 table. Now overlaps host init (phase-2 sync). |
| Host ladders | ~0.5–2 s | Six serial OpenSSL ladders. Now six threads. |
| Full-table readback | ~1–2 s | 1.465 GiB `malloc` + page faults + D2H for a 216-record check. Now ~14 KiB sampled fetch. |
| Pipeline allocs | ~0.5–1 s | ~1.5 GiB across 2 slots. Now stream-ordered async. |
| Constant uploads + host EC | ~0.5 s | Small memcpys, 2 `EC_POINT_dbl`, gate setup. Unchanged. |
| Teardown / tail loss | ~1–2 hits | In-flight batches die at SIGTERM. Now drained + published. |

**Why v8 measured neutral (−0.16%):** two readings were recorded — silent
cubin fallback, or dead-time-model error. The arithmetic here is now
definitive: the identical ~0.976 ratio across packages with different
self-rates (871→851M, 904→882M, 902→883M, 867→846M) is a fixed ~29 s
constant, not a throughput-proportional decay. And v9's own variance
measurement (±3–4%) means a +2% init gain is inside single-draw noise —
v8 cannot falsify the model either way. This submission is the clean
instrument: if the cubin engages, the artifact's `official/self` ratio
should visibly lift toward ~0.99.

**Failure modes are loud, not silent:**
- Cubin missing/stale/mismatched → `g_use_cubin=0` → identical v20 runtime
  path (JIT cost returns, no correctness risk).
- Cubin constant upload fails readback-verify → path disabled entirely.
- A genuinely wrong table (any cause) fails the OpenSSL spot-check →
  `compute_gtable` host rebuild → identical correctness guarantee as v20.
- A param-packing defect in `cuLaunchKernel` → zero hits → score ≈0
  (rejected loudly, never a silent degradation; the 23-arg array was
  triple-checked against the mangled signature).
- `std::thread` spawn failure (pre-glibc-2.34 runtimes without `-pthread`)
  → `std::system_error` caught → serial ladder fallback.
- `cudaMallocAsync` unsupported (WDDM/no pool) → per-pointer `cudaMalloc`
  fallback.

**What this package does NOT change:** every device-side instruction that
was measured at 882.10M/903.8M is byte-identical — GLV12 recode, table
geometry, field arithmetic, SHA path, hit filter, gate semantics. The
claimed delta is entirely in the ~29 s between process start and grind.

## Provenance

Base = v20 (`d802907`), itself the promoted `pr1259` GLV12 package plus our
LEAN/SFC2 union. Init machinery revived from our v8 (`b1ed21e`) with the
path-resolution and verification hardening that attempt lacked evidence
for. Sampled readback independently re-derived (field pr1284 reached the
same idea; our `QSB_SPOT_DEV` predates it).
