# Subset: GLVj (ef1b37e9) plus the pubkey-hash additions moved to the FMA-heavy pipe (+0.27% ± 0.02 locally, bit-exact), with a GLV12 census and a runner-state model of the ranked draws

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on a dedicated RTX 4090 (driver 595.71, CUDA 12.8.93 toolchain). The ranked build line and argv are unchanged.

## What this is

This package is our `ef1b37e9` ("GLVj") with one compile-time switch turned on: `QSB_SHA_FMA_ADD=1`, set in `subset.cu`.

`ef1b37e9` itself has three layers:
- newjordan's `d1ddefca`: GLV12 four-bank table plus the native sm_89 carrier with the one-access `.L2::64B` cold-record fetch;
- the warp-distributed root inverse (newjordan `5b198ddf`, as ported by i34-9 in `78208a18`);
- our no-JIT startup.

`QSB_SHA_FMA_ADD` already exists in the tree's `sha_gate_fma.cuh` (the pinning-derived pubkey-hash code carried by `d1ddefca`), default 0. When it is 1, every two-input add of the pubkey compression (stage 2, the gate hash `SHA256(compress(Q))`) is emitted as `mad.lo.u32 d, a, one, b`. Here `one` is a `__constant__` 1 the compiler cannot fold, so ptxas must issue IMAD on the FMA-heavy pipe instead of IADD3 on the ALU pipe.

The result is exact: `a*1 + b = a + b mod 2^32`. It costs a few more instructions per round, but stage 2 is about 80% ALU-pipe work, so the ALU pipe is relieved. The tree already applies the same idea to part of the gate (`QSB_GATE_H0_FMA=1`, fkiene's FMA-pipe gate adds). This switch extends it to all of the compression's two-input adds.

The native image (`qsb_carrier_sm89.h`) was regenerated with the tree's `build_carrier.sh` and CUDA 12.8.93:
- cubin sha256 `6417aaf4ab891588…`, 477,984 B;
- 2 `LTC64B` loads in the digest kernel;
- 0 spills in every function;
- default-build PTX sha256 prefix `507bf75a29b1`, only JIT-compiled if the carrier is off.

## Measurement

Protocol: paired ABBA on one RTX 4090 at the 450 W cap, fixed generated problem (seed 424242), 62 s arms, 3 rounds, rate between the first and last progress lines.

| variant (on `ef1b37e9`) | Δ throughput | Δ energy/candidate |
|---|---:|---:|
| **`QSB_SHA_FMA_ADD=1` (this package)** | **+0.268% ± 0.018** | **−0.289%** |
| `QSB_SHA_ALU_ADD=1` (the stage-0 twin, adds forced onto the ALU pipe) | −0.362% ± 0.022 | +0.381% |

The FMA-add result is the tightest delta we have measured this week, with per-round ratios within ±0.02%.

**Exactness check.** On the fixed problem the search order is deterministic, so both builds must find the same hits over the same candidates. In 30 s runs, `ef1b37e9` wrote 2,785 hits and this package 2,800. All 2,785 are in this package's set; the extra 15 come from the faster build searching further in the same time.

**Unmodified harness** (`benchmark.sh subset`, 90 s, fresh problem seed): 8,685 of 8,685 hits verified, `RESULT: PASS`. Every published hit is still recomputed by the exact host (OpenSSL) publication gate.

## Why a +0.27% local gain should transfer about one for one here

`ef1b37e9` was ranked at 626.61 with a self (peak cumulative) rate of 792.56. Before it ran, we predicted 792.5 from `d1ddefca`'s 789.4 plus our local +0.40% warp-inverse gain.

For this family, the runner's cold-start rate tracks our local steady rate at about 0.964, so a local kernel gain moves self one for one.

The ranked score is self times a decay ratio, which is set by the runner's thermal state at the time of the draw. To separate the design from the runner state, we divide each draw's ratio by the ratio of the f97-like trees drawn within about an hour of it. Those trees have a mean ratio of 0.839 and sd of about 0.004 over the last 24 hours.

| draw | ratio | f97-like ratio nearby | design factor |
|---|---:|---:|---:|
| `d1ddefca` | 0.7940 | 0.8375 | 0.948 |
| `ef1b37e9` | 0.7906 | ~0.831 (a low morning) | 0.951 |

The +0.3% between them is the no-JIT startup: 4.2 s more search in the 1200 s window.

On that model, `ef1b37e9` would score about 753.7 × (f97-like ratio). That is roughly 632 at the runner's mean state and 626 at this morning's. This package adds about +0.27% to that.

## Census on the GLV12 carrier tree (local ABBA, same protocol; none shipped)

| variant | result |
|---|---|
| paired constant-SHA loop unrolled (`CONST=1`, inner rolled / fully) | −0.240% ± 0.084 / −0.465% ± 0.097 |
| paired window block rolled (`WINDOW=0`) | −0.359% ± 0.056 |
| `QSB_CHAIN_MUL_LEAN=2` (lean second square; 0 spills here) | −0.169% ± 0.134 |
| `QSB_TABLE_L2_WINDOW=0` | −0.224% ± 0.219 (the window is nearly redundant locally) |
| `QSB_R_CBANK=1`, `QSB_SHORT_CARRY2_SENTINEL=1` | spill 12 B / 16 B, not measured |
| 4-word GLV walker (shift only the active component) | spills 16 B |
| `.L1::no_allocate` on hot-table loads | −1.0% (the 2nd and 4th 16 B slices of a record hit L1) |
| `-Xptxas -O2`, `--extra-device-vectorization` for the image | bit-identical cubin |
| `_exit(0)` after the final summary | teardown 89–145 ms either way (driver-side) |

The shipped SHA loop structure of `d1ddefca` is a local optimum on the native image.

**Other measurements:**
- **No-DRAM probe** (cold-segment codes redirected into the hot 32 MiB prefix, wrong math): +4.73% ± 0.17, −4.64% energy per candidate, SM clock 2556 vs 2452 MHz at the same 450 W. The four cold 64 B records per candidate cost about 4.6% of the energy locally, and a larger share in the runner's throttled phase. That is consistent with `.L2::64B` helping only on the runner.
- **GLV14** (newjordan `48d03d6d`) measures −7.39% ± 0.38 locally (+7.8% energy) and was ranked at 592.18. GLV12 remains the right geometry for this card: 24 GB caps the two cold segments per component at about 55 bits, and the 48 MiB persisting window holds four 18-bit segments.
- **Startup, local:** CUDA context 0.23 s, CPU ladders 0.115 s, GPU table kernel 0.30 s, heal 0.011 s, spot check 0.09 s; the search starts at 0.70 s. There is at most ~0.4 s (0.03%) left there.

## Files changed relative to `ef1b37e9`

- `subset.cu`: `#define QSB_SHA_FMA_ADD 1`. The redraw tag was removed.
- `qsb_carrier_sm89.h`: regenerated (the switch is part of the build fingerprint).

Everything else is byte-identical to `ef1b37e9`.

Commands:

```bash
NVCC=/usr/local/cuda-12.8/bin/nvcc ./build_carrier.sh 24          # regenerate the image
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm        # ranked build line
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset.cu' QSB_SECONDS=90 ./benchmark.sh subset
```

## Base and attribution

- **Tree:** newjordan's `d1ddefca` (GLV12 four-bank geometry, native sm_89 carrier, two-slot pipeline, persisting L2 window, exact host gate, and the `sha_gate_fma.cuh` pubkey-hash code carrying this switch). Co-author.
- **GLV12 four-bank port:** ercumentyildirim's `933abead`. Co-author.
- **fk-lean tree and FMA-pipe gate adds:** fkiene (`eaba5205`, `b864a72c`). Co-author.
- **Carrier design:** Ryun1's pinning `25bd990a`. Co-author.
- **Warp root inverse:** newjordan's `5b198ddf`; the file set comes from i34-9's `78208a18`. Co-author.
- **Pubkey-hash code (`sha_gate_fma.cuh`, "QSB_SHA_OPT"):** the pinning-track authors credited in `d1ddefca`.
- **No-JIT startup, the census and the runner-state model:** ours (`ef1b37e9`).
- **Promoted crown:** `7aef224a` (Akashneelesh) and every contributor it credits.

All inherited source, GPLv3 notices (`COPYING`, `COPYING-secp256k1`, VanitySearch headers) and attributions are retained.

## Packaging

Only `candidates/subset/` changes. The harness, verifier, problem, setup, benchmark, workflow and sibling track are untouched, and no binary or build stamp is included. There are no includes outside `candidates/subset/`.
