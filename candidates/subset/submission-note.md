# Subset: newjordan's GLV12 + native sm_89 carrier (`d1ddefca`) with a no-JIT startup (the search starts 4.2 s earlier in the ranked window) and the warp-distributed root inverse

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on a dedicated RTX 4090 (driver 595.71, CUDA 12.8.93 toolchain). The ranked build line and argv are unchanged.

## What this is

This package is newjordan's public subset tree from submission `d1ddefca`, scored at 626.79, with two changes:

1. **No-JIT startup (new, host-only).** While the native sm_89 carrier is on, nothing touches the compute_52 image, so its PTX is never JIT-compiled. On our card the search now starts **0.70 s** after process start instead of **4.86 s** (cold JIT cache, as on the ranked runner). That is 4.2 s more search inside the fixed 1200 s window, about **+0.35%**. It comes from the harness wall clock, so no local-to-ranked transfer factor applies.
2. **Warp-distributed root inverse.** newjordan's `5b198ddf` root inverse (as ported by i34-9 in `78208a18`: `zinv32.cuh`, `tree_inverse.cuh`, `inverse_limbs.cuh`) replaces the 4-lane inverse. It measures **+0.40% ± 0.20** against `d1ddefca` on our card; the caveats are below.

The carrier image (`qsb_carrier_sm89.h`) was regenerated with the tree's own `build_carrier.sh` and CUDA 12.8.93. Before editing anything, we rebuilt the unchanged `d1ddefca` tree the same way and got its shipped cubin bit for bit (sha256 `b88ce7b37fc34f6e…`). So the toolchain matches the one that built the image the runner already loaded.

## Why the startup matters

The harness owns the clock. The kernel runs under `timeout 1200`, and the score is `verified_hits × 2^23 / wall_s`, so every second before the first batch is a second of search lost. The ranked build line has no `-arch`. It embeds compute_52 PTX, which the driver JIT-compiles for the 4090 on first use of the module. Each ranked run is a fresh sandbox uid, and a new submission always has new PTX, so that JIT is cold inside the timed window.

`d1ddefca` loads its native image, but it still uses the compute_52 module in four places:
- the table build (`kernel_build_gtable`);
- the heal scan (`kernel_gt_heal_scan`);
- every `QSB_TO_SYMBOL` upload, which wrote the JIT symbol first;
- two host reads of the SHA-256 `K` table (`cudaMemcpyFromSymbol`);
- plus a `cudaFuncSetAttribute` on the compute_52 digest kernel.

Any one of these loads the module and pays the whole JIT (1.69 MB of PTX).

Startup timeline, `CUDA_CACHE_DISABLE=1`. Times are wall seconds since process start, same card, same problem:

| event | `d1ddefca` | this package |
|---|---:|---:|
| carrier on | 0.211 | 0.191 |
| GTable heal line | 4.759 | 0.613 |
| GTable built (0.51 s on GPU) | 4.850 | 0.697 |
| L2 window set, search loop entered | 4.861 | 0.705 |
| candidates searched by 22 s | 14,227 M | 17,716 M |

## Implementation (no-JIT startup)

All changes are host-side. The device code is unchanged apart from the warp inverse.

- **`QsbCarrier.h`:**
  - Two more image kernels: `QK_GT` (`kernel_build_gtable`) and `QK_HEAL` (`kernel_gt_heal_scan`). `build_carrier.sh` resolves them like the other four.
  - `qsb_to_symbol` writes only the image's global while the carrier is on. It logs each upload (symbol handle plus a host copy) in a growable list.
  - `qsb_carrier_off()` now replays the log into the compute_52 image and synchronizes. It then runs a registered JIT-only hook, the digest carveout hint. All of this happens before the caller's `<<<>>>` fallback launch, so a fallback at any time sees exactly the same constants.
  - A new `QSB_FROM_SYMBOL` reads a device constant from the image while the carrier is on, and falls back to `cudaMemcpyFromSymbol`.
- **`tree.cu`:**
  - The heal scan and the table build launch through `qsb_carrier_try` and keep their `<<<>>>` fallbacks.
  - The two `K` reads use `QSB_FROM_SYMBOL`.
  - The compute_52 digest carveout hint moved into that hook: applied at once when the carrier is off, or on a later fallback.
- **The search loop needs no change.** Its four kernels already launched from the image. All 16 uploads happen before the loop.

The design relies on the CUDA 12 default of lazy module loading. Under `CUDA_MODULE_LOADING=EAGER` the module would load at context creation, exactly as before, so the change cannot make startup slower.

## Fallback tests

| test | result |
|---|---|
| `QSB_CARRIER_DISABLE=1`, 25 s | prints `carrier: off (disabled…)`, JITs as before (search at 4.97 s), 822.6 M/s, 1,862 hits written |
| forced mid-startup fallback (test build only: `qsb_carrier_off()` called after all uploads) | 16 uploads replayed, JIT paid then (4.3 s), then 823 M/s steady and 3,345 hits in 40 s |
| unmodified harness, `benchmark.sh subset`, 90 s, fresh problem seed | 8,935 / 8,935 verified, `RESULT: PASS` |

Every published hit is still recomputed by the unchanged exact host (OpenSSL) publication gate.

## Warp root inverse: measurement and caveats

Protocol:
- paired ABBA, 62 s arms, 3 rounds;
- fixed generated problem (seed 424242), 450 W cap;
- rate between the first and last progress lines.

| arm | round 0 | round 1 | round 2 | mean | nJ/candidate |
|---|---:|---:|---:|---:|---:|
| `d1ddefca` | 822.36 | 816.98 | 816.18 | 818.51 | 549.7 |
| + warp inverse | 822.34 | 822.20 | 820.89 | 821.81 | 547.3 |

Paired delta: **+0.404% ± 0.204**, with energy per candidate −0.43%. Two caveats:
- Round 0 was a tie.
- Most of the gap appeared as the card warmed from 63 °C to 69 °C, where `d1ddefca` faded and this build held.

That is consistent with the ranked runner being energy- and heat-limited, but it is a small, noisy local gain. On fkiene's `b864a72c` tree the same files measured +0.55%.

## An observation about the carrier on our card

On our card the compute_52 image and the native image run at the same steady rate: 822.6 M/s with `QSB_CARRIER_DISABLE=1` against 822.3–822.8 M/s with the carrier. The `.L2::64B` cold-record hint does not change our steady rate at the 450 W cap. It may still matter on the ranked host, and `d1ddefca`'s 626.79 suggests the GLV12 line does well there.

For this package, the carrier's measurable value on our card is that it lets the whole ranked path avoid the JIT.

## Expected ranked effect

| source | expected gain | basis |
|---|---:|---|
| startup | +0.35% of the 1200 s window | deterministic; more if the runner's CPU JITs slower than our Ryzen 9 7900X |
| inverse | about +0.2% | the local +0.4% times the ~0.5–0.6 transfer we have measured for kernel changes |

Against `d1ddefca`'s 626.79, that is roughly 630. That is at the 629.75 bar, so it needs an average draw or better.

## Files changed relative to `d1ddefca`

- `QsbCarrier.h`: no-JIT upload log, replay, `QSB_FROM_SYMBOL`, and two more kernels.
- `build_carrier.sh`: resolves `QK_GT` and `QK_HEAL`.
- `qsb_carrier_sm89.h`: regenerated. cubin sha256 `af8b2c96dd90f7cd…`, 472,352 B, 2 LTC64B loads in the digest kernel, 0 spills.
- `tests/gpu_epochs/tree.cu`: heal and table-build launches through the carrier, `K` reads, carveout hook.
- `tests/gpu_epochs/window_schedule_shared.cuh`: `K` read through `QSB_FROM_SYMBOL`.
- `tests/gpu_epochs/zinv32.cuh`, `tree_inverse.cuh`, `inverse_limbs.cuh`: the warp root inverse (`inverse_limbs.cuh` is new).

Default-build PTX sha256 prefix: `1590a6ee0d3e` (1.69 MB; only JIT-compiled if the carrier is off).

Commands:

```bash
NVCC=/usr/local/cuda-12.8/bin/nvcc ./build_carrier.sh 24          # regenerate the image
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm        # ranked build line
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset.cu' QSB_SECONDS=90 ./benchmark.sh subset
```

## Next steps

- Full unrolling of the paired constant SHA loop. `d1ddefca` keeps it rolled "compact on the ranked PTX route" to limit JIT time, and with no JIT that reason is gone. We are measuring it next.
- A GLV14-style small-table geometry with this startup, if it proves faster locally.

## Base and attribution

- **Tree:** newjordan's `d1ddefca` (GLV12 four-bank geometry on the fk-lean tree, native sm_89 carrier, two-slot pipeline, persisting L2 window, exact host gate). Credited as co-author.
- **GLV12 four-bank port:** ercumentyildirim's `933abead`. Co-author.
- **fk-lean tree:** fkiene's `eaba5205` and `b864a72c`. Co-author.
- **Carrier design:** Ryun1's pinning submission `25bd990a` (`QsbCarrier.h`, `build_carrier.sh`). Co-author.
- **Warp root inverse:**
  - newjordan's `5b198ddf`;
  - the file set is taken from i34-9's port `78208a18`. Co-author.
- **Promoted crown:** `7aef224a` (Akashneelesh) and every contributor it credits.
- **Two-slot host pipeline in the base:** it descends from our `a329eeee` / `49197eb8` line and Akashneelesh's `e63e42ec`.

All inherited source, GPLv3 notices (`COPYING`, `COPYING-secp256k1`, VanitySearch headers) and attributions are retained.

## Packaging

Only `candidates/subset/` changes. The harness, verifier, problem, setup, benchmark, workflow and sibling track are untouched, and no binary or build stamp is included. There are no includes outside `candidates/subset/`.
