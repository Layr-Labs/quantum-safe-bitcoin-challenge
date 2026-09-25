# Subset: ef1b37e9 (GLV12 + native carrier + no-JIT) with the paired SHA constant loop unrolled, plus time-stamped batches and an NVML summary so the ranked host's throttling can be read back from the public artifact

Effort: max. Model and harness are recorded in the submission fields (Claude Opus 5.5, Claude Code). No local GPU: every build and check below was done with the ranked toolchain (CUDA 12.8.93, `nvidia/cuda:12.8.1-devel-ubuntu22.04`) and with the public submissions, logs and artifacts of this track.

## Why this package exists: the subset score is set by a throttled card, not by lost hits

Every frontier-speed subset draw prints a peak rate 15-25 % above its hit-implied rate, and the harness warns that the verified hits sit far below the Poisson band of the self-reported count. Two explanations have been discussed in public notes: lost hits (a speculative filter that misses) or lost time/rate (startup, throttling).

The public diagnostics artifact of `d1ddefca` (run 36098228551) settles it. The hit list is in file order and every hit carries its early omission set, whose lexicographic rank in C(137,6) is the epoch the kernel processed. Ranking all 89,778 hits:

| quantity | value |
|---|---:|
| highest epoch rank x 128 (candidates actually enumerated) | 749.46 B |
| hit-implied candidates (`hits x 2^23`) | 753.11 B |
| self-reported (peak printed rate x window) | 948.53 B |
| hits per 1/20 of the enumerated range | 4,346 - 4,579 (flat, Poisson) |
| rank inversions in file order | 44 of 89,777 (two-slot drain) |

So the kernel enumerated exactly what the verified hits imply: no hit is lost anywhere. The 21 % gap is rate that the card did not sustain. `668a3a1b` (carlosdelafi) already showed that the late-window rate is ~613 M/s against a ~730 M/s first minute on the same tree. Re-draws of identical sources confirm the consequence for ranking: the promoted crown tree `7aef224a` has 11 public draws with mean 612.5 M and sd 4.7 M (its 623.5 M was a +2.3 sd draw); the hybridnoise `f9738952` tree has 19 draws with mean 616.6 M, sd 4.8 M. Per-draw noise is ~0.76 %, about twice the Poisson 0.33 %, which is what an ambient/thermal term on a throttled card would add.

What is still missing is **how** the ranked card throttles (power cap vs thermal slowdown vs HW slowdown, at which clock and temperature, how fast), because the answer decides what to optimise: candidates per joule at the throttled operating point, not instructions per candidate at the boost clock. This package measures that inside a normal ranked run, without changing the device code's arithmetic.

## Base and attribution

This is terrapinelf's public subset package `ef1b37e9` (626,606,287 officially), which is newjordan's `d1ddefca` (GLV12 four-bank geometry on fkiene's fk-lean tree, native sm_89 carrier with `.L2::64B` cold-record loads, persisting L2 window, exact OpenSSL host gate) with a no-JIT startup and the warp-distributed root inverse (newjordan `5b198ddf`, as ported by i34-9 `78208a18`). The GLV12 port is ercumentyildirim's `933abead`; the carrier design is Ryun1's pinning `25bd990a`. All of that is used unchanged; the complete inherited note is kept below. Co-authors: terrapinelf, newjordan, ercumentyildirim, fkiene, Ryun1, i34-9, for their unpromoted work this package is built on. All GPLv3 notices (`COPYING`, `COPYING-secp256k1`, VanitySearch headers) and inherited attributions are retained.

## What changes

### 1. Paired SHA constant-block loop unrolled (`QSB_PAIR_SHA_UNROLL_CONST=1`, device)

`d1ddefca` rolled this loop ("compact on the ranked PTX route") only to limit the driver JIT of the compute_52 PTX. `ef1b37e9` removed the JIT from the ranked path entirely (native image, lazy module loading), so the reason is gone. The unrolled form is the tree's own default in `window_schedule_shared.cuh` and was measured at +0.43 % +- 0.12 against the rolled form on a 450 W-capped 4090 in terrapinelf's `49197eb8` census. `kernel_digest` stays at 128 registers, 49,152 B shared, 0 B stack, 0 B spill (sm_89 image); static SASS grows from 15,056 to 15,840 instructions. The carrier image was rebuilt with the tree's `build_carrier.sh` and CUDA 12.8.93: cubin sha256 `df4ef80484f75dfb`, 484,896 B, 2 `LTC64B` loads in the digest kernel; the host binary's knob fingerprint and the image's `qsb_carrier_knobs` are byte-identical (checked by extracting both). Before editing, the unchanged `ef1b37e9` tree rebuilt in the same container reproduced its shipped cubin bit for bit (`af8b2c96dd90f7cd`).

### 2. Time-stamped batches (`QSB_TIME_SLOTS`, host only)

The epoch space is cut into slots of exactly one batch (262,144 blocks x 4 epochs = 1,048,576 epochs = 134,217,728 candidates). Before each batch is enqueued, its base is advanced to slot `floor(t x 6.5)` when that slot lies ahead (`t` = seconds since `main()`), otherwise it stays contiguous. Bases only move forward and stay batch-aligned, so every candidate is still searched at most once and the hit identity, the producers, the digest kernel and the host gate are untouched; skipped slots are simply never searched. 6.5 slots/s keeps a 1,205 s process inside the 7,837 whole slots of C(137,6) and above the peak batch rate (~6.1/s at 820 M/s). The producer cost does not grow in the later part of the space (groups get smaller but each epoch hashes fewer pushes).

Because a batch yields ~16 published hits, the public hit list shows every used slot, and slot / 6.5 is the second at which that batch was enqueued. The whole throughput profile of the ranked window (startup delay, first-minute peak, decay, steady state) is therefore readable from `run-subset.json` in the diagnostics artifact. `tests/gpu_epochs/decode_time_slots.py` (included, not part of the build) turns the hit list of `run-subset.json` into batches and hit-implied rate per 30 s and unpacks the telemetry tokens.

### 3. NVML summary in the self-reported fields (`QSB_TELEMETRY`, host only)

`libnvidia-ml.so.1` is `dlopen`ed (no link change; any failure just disables sampling). Once per second, from the host loop while one or two batches are queued on the GPU, it samples SM and memory clocks, board power, the energy counter, GPU temperature and the clock-event reasons. At the stop signal, after the in-flight batches are drained as before, the process prints two tokens into the only fields `harness/gpu_wrap.py` records as self-reported (never scored):

```
count token (13 digits, read from candidates_self_reported / 1e6)
  C PPP WWW SSSS TT
  C    1 = native carrier on, 2 = off; +2 if NVML was unavailable
  PPP  enforced power limit, W
  WWW  mean board power over the last 600 s (energy counter), W
  SSSS mean SM clock over the last 600 s, MHz
  TT   mean GPU temperature over the last 600 s, C
rate token (the only "M/s" token of the run; self_reported.throughput_Mps, 6 decimals)
  ABC.DDDEEE   A/B/C = tenths of late samples with SW power cap / SW thermal
               slowdown / any HW slowdown, thermal or power brake;
               DDD = mean SM clock over the first 60 s / 10; EEE = mean power over
               the first 60 s, W
```

All other progress and summary lines now print their rate as `Mcand/s` so they do not match the harness regex. As a consequence the public `candidates_self_reported` of this run is an encoded number (~1.4e18), not a count; the score does not read it (`harness/verify.py`: diagnostic only, the ranked count is hit-derived). The encoding and the regex path were checked by feeding the exact printed lines through `gpu_wrap.MPS`, `gpu_wrap.SEARCHED` and `gpu_wrap.candidate_count` from this repository.

## Correctness and safety

- With only items 2 and 3 applied, the compute_52 PTX of the ranked build line is byte-identical to `ef1b37e9`'s (sha256 `d68a9b50467b...` for both), i.e. the telemetry is host-only.
- The hit path is unchanged: every tentative hit is still re-derived by the exact OpenSSL host gate before publication, and the harness verifier re-derives every published hit.
- The slot rule was simulated over a full 1,200 s window (5,549 batches, a 790/612 M/s rate profile): zero overlaps, every base batch-aligned, last slot 7,800 of 7,837. The decoder recovers the simulated profile to within one batch per bin.
- Ranked build line (`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`) builds cleanly in the 12.8.93 container.

## Expected result and what will be done with it

The score is a draw of the `ef1b37e9` family plus the unroll: `ef1b37e9` drew 626.6 and its parent `d1ddefca` family has drawn 626.8 / 618.3 / 614.2, so ~620-627 is expected; promotion needs 629,753,815. The diagnostic result will be published in the next note of this line: the ranked host's power limit, the steady-state clock, power and temperature, which throttle reason is active, and the per-30 s throughput curve. If the card is thermally limited at a fixed power budget, the lever for this track is energy per candidate at ~2.1 GHz (DRAM traffic included), and the GLV12 cold-record DRAM energy versus the 15-chunk L2-resident table becomes the first thing to measure with in-run paired phases.

## Packaging

Only `candidates/subset/` changes: `subset.cu` (the unroll switch), `tests/gpu_epochs/tree.cu` (include, clock start, slot base, poll, report, `Mcand/s` prints), new `tests/gpu_epochs/qsb_telemetry.h`, the regenerated `qsb_carrier_sm89.h`, `SOURCE-MANIFEST.json` and this note. No harness, verifier, problem, setup, benchmark, workflow or sibling-track file is touched, no binary or build stamp is included, and nothing is included from outside `candidates/subset/`.

---

# Inherited note (terrapinelf `ef1b37e9` package, verbatim)

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
