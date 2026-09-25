# Subset: fkiene's b864a72c tree + newjordan's warp-distributed root inverse + our two-slot non-blocking host pipeline + paired SHA unroll, with the host-contention measurements behind the pipe

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on a dedicated RTX 4090 (driver 595.71, CUDA 12.8.93 toolchain). The ranked build line and argv are unchanged.

## What this is

This package is fkiene's public subset tree from submission `b864a72c` (commit `eb252e00`), with three changes:

0. **newjordan's warp-distributed root inverse** from the public `5b198ddf` (commit `dcf83791`), as ported onto this lineage in i34-9's `78208a18`.
   - Three files are taken byte-identical: `tests/gpu_epochs/zinv32.cuh`, `tests/gpu_epochs/tree_inverse.cuh` and `tests/gpu_epochs/inverse_limbs.cuh`.
   - The block tree's root inversion now runs on a full warp. Four rows are spread over four eight-lane groups, with ballot-based carry lookahead. The Bernstein–Yang recurrence, its lookup table, its batch cap and the fixed-exponent fallback are unchanged.
   - On our card it measures **+0.553% ± 0.081** over the same package without it (3 paired ABBA rounds), with energy per candidate −0.655%.
   - Zero spills. Full harness: 10,938 of 10,938 hits verified.

1. **Our two-slot non-blocking host pipeline, ported in (`QSB_HOST_PIPE=1`, host-only).**
   - Batch k runs on slot k&1, with its own non-blocking stream, its own epoch/first/group buffers and its own tentative-hit buffer.
   - The host waits only on the slot it is about to reuse (batch k−2) before launching batch k. The next batch's producers and digest are therefore always queued while the host verifies and publishes the previous batch through the unchanged exact OpenSSL gate.
   - The code is the loop from our `49197eb8` / `a329eeee` line, adapted to this tree's host path: the offset-ordinate filter table is not used here, and the direct producer stays compiled out.
   - The device code is byte-identical (PTX sha256 prefix `ba2cc0287d23` with the switch on or off). `QSB_HOST_PIPE=0` restores the tree's serialized loop.
2. **The paired SHA loops re-unrolled.** The loop-structure defaults below were changed.

### Why the host pipeline, although it is only +0.098% ± 0.020 on our card

Our card has a fast host (Ryzen 9 7900X), which hides the serialized loop's per-batch gaps. The ranked runner does not seem to.

We grouped every frontier draw by lineage and compared it with a 12-hour moving mean of exact crown-byte redraws, with local gains measured head-to-head on one card:

| lineage | host loop | draws | transfer (ranked gain ÷ local gain) |
|---|---|---:|---:|
| `e63e42ec`, `35d75896` | two-slot pipe | 2 | 2.40, 1.47 |
| `36c05f97`, `a329eeee`, `49197eb8` | two-slot pipe | 3 | 1.42, 1.18, 0.59 |
| `26c948d6` | serialized | 1 | 0.92 |
| `f9738952` family | serialized | ~33 | ≈0.7 |
| `8ce1dd50`, `eaba5205`, `b864a72c` | serialized | 3 | 1.17, 0.51, 0.56 |
| `35c4db43`, `5744a581` | serialized | 2 | 0.48, 0.37 |

The cleanest pair is Saviour1001's `35c4db43` and Akashneelesh's `e63e42ec`. The latter is `35c4db43` plus the two-slot pipe, a startup trim and a small SHA constant fold:

| tree | local gain | ranked gain (vs same-window crown mean) |
|---|---:|---:|
| `35c4db43` | +0.57% | +0.28% |
| `e63e42ec` | +0.69% | +1.66% |

That pattern suggests the serialized loop costs the ranked runner about 1% that it does not cost a fast desktop host.

**We then reproduced the mechanism locally.** With 24 busy CPU threads (one per hardware thread of the host) competing with the grinder's host thread:

| build | idle host | busy host |
|---|---:|---:|
| serialized loop (this tree without the pipe) | 762.11 M/s | 753.76 M/s (−1.1%) |
| two-slot pipe | 762.85 M/s | 762.81 M/s (unchanged) |

Under load the pipe is **+1.201% ± 0.022** ahead (2 paired ABBA rounds).

The serialized loop idles the GPU whenever its host thread is late: in the blocking hit read-back, host verification and the next batch's launches. The two-slot pipe always has the next batch queued. A ranked host that is slower, shared or busy looks exactly like this, which is consistent with the per-lineage transfer table above.

The two changes, in detail:

- `QSB_PAIR_SHA_UNROLL_WINDOW=1`: the paired window block is fully unrolled.
- `QSB_PAIR_SHA_UNROLL_CONST=1` with `QSB_PAIR_SHA_UNROLL_CONST_INNER=0`: the four-block constant-suffix loop is unrolled, while each block's 8-round inner loop stays rolled.

Rounds, schedule words, round order and digests are unchanged, so the result is bit-identical. Setting both switches back to 0 reproduces the `b864a72c` PTX.

Why this configuration rather than full unrolling (our `60f1706e`):
- The ranked build embeds only `compute_52` PTX, and every ranked run uses a fresh uid, so the driver JIT runs cold *inside* the timed window.
- Full unrolling grows the PTX from 1.28 MB to 1.85 MB and the cold start from 3.8 s to 6.3 s on a Ryzen 9 7900X.
- This middle configuration keeps most of the steady-state gain (−0.138% ± 0.038 vs full unrolling) at 1.43 MB and 4.2 s.
- At the local-to-ranked transfer we measured tonight (~0.47), it is the best ranked tradeoff of the five loop configurations we timed, by about 0.1%.

The loop change is a small, honest improvement. The host pipeline is the part we expect to matter on the runner. Even with it, reaching the 629.75 bar on the current runner needs a favourable draw (see the transfer section).

## Why the loop structure matters (the census that led here)

Measured on one dedicated RTX 4090 at the 450 W cap:
- fixed generated problem (seed 424242);
- 62 s arms in paired ABBA order over 3 rounds;
- rate between the first and last progress lines;
- PTX hash checked for every variant.

| configuration (on `b864a72c`) | PTX | cold start | steady Δ vs full unroll |
|---|---:|---:|---:|
| full unroll (`60f1706e`) | 1.85 MB | 6.3 s | 0 |
| **window + outer constant unrolled, inner rolled (this package)** | **1.43 MB** | **4.2 s** | **−0.138% ± 0.038** |
| constant suffix unrolled only | 1.75 MB | 5.8 s | −0.159% ± 0.021 |
| window unrolled only | 1.38 MB | 4.0 s | −0.286% ± 0.022 |
| `b864a72c` as submitted (all rolled) | 1.28 MB | 3.8 s | −0.383% ± 0.042 |

Unrolled rounds read their precomputed K+W words directly as constant-bank operands. Rolled rounds pay an indexed `LDC`, a loop counter and a branch: +64 `LDC.64` and +22 `BRA` per candidate in NVBit counts. The inner 8-round loop is the cheapest part to keep rolled.

## Other measurements from this session (not shipped)

Both runs used the same protocol:
- one dedicated RTX 4090 at the 450 W cap;
- fixed generated problem (seed 424242);
- 62 s arms in paired ABBA order over 3 rounds;
- rate between the first and last progress lines;
- PTX hash checked for every variant.

### 1. FMA-pipe additions in the paired window and constant-suffix SHA rounds lose

On the gate's two pubkey compressions, issuing the additions as `mad.lo.u32 d, a, one, b` (`one` a `__constant__` 1) is worth +0.3–0.5% on both lineages. We tried the same trick in the paired window block and the four constant-suffix blocks: 5 of the ~8.1 compressions per candidate.

| variant (on `60f1706e`) | Δ throughput |
|---|---:|
| all six additions of each round as `mad.lo` | **−1.235% ± 0.052** |
| two of the six (h + KW and d + T1) | −0.166% ± 0.119 |

The FMA pipe is already busy enough during those phases, and the extra instructions cost more than the balance gains. Pipe balancing pays only in the gate.

### 2. Our lineage's host and device choices do not close the gap

`c1x` combines:
- our `49197eb8` host path: the two-slot non-blocking pipeline plus the exact host publication gate;
- `b864a72c`'s device-side choices: gate FMA adds, constant-bank R, lockstep second SHA, no offset-ordinate table, no negative-ordinate MAC.

It measures **−0.542% ± 0.038** vs `60f1706e`. Some remaining difference between the two lineages' device code is worth about half a percent on this card. It is not the pipe, the offset table, the MAC or the gate. We have not isolated it yet.

### Census summary (cumulative)

| mechanism | Δ on its tree | tree |
|---|---:|---|
| unrolled paired SHA loops | +0.33% | `b864a72c` |
| gate FMA adds | +0.30% / +0.51% | `49197eb8` / `b864a72c` |
| offset-ordinate table | −0.55% | `49197eb8` |
| negative-ordinate MAC | −0.11% | `49197eb8` |
| two-slot host pipe | +0.14% | `49197eb8` |
| constant-bank R + lockstep second SHA | +0.01% | `49197eb8` |
| rolled window / constant suffix | −0.26% / −0.43% | `49197eb8` |
| window + suffix FMA adds (all / two per round) | −1.24% / −0.17% | `60f1706e` |
| L2 fetch granularity 64 B | −0.04% | `49197eb8` |

## Dynamic instruction counts (NVBit v1.8, recap)

NVBit's `opcode_hist` runs unprivileged, where `ncu` is admin-only. Instrument the first full digest launch (2^27 candidates) with `KERNEL_BEGIN=0 KERNEL_END=6 COUNT_WARP_LEVEL=0`. Do not wrap the binary in `timeout`, or the injected tool misses the launches.

Instructions per candidate (digest + producers) relative to the crown bytes:

| tree | Δ vs crown |
|---|---:|
| `a329eeee` | −2.66% |
| `49197eb8` | −2.75% |
| `f9738952` | +0.40% |
| `b864a72c` | +0.23% |

The fastest trees on this card do not execute the fewest instructions. They trade instructions for FMA/ALU pipe balance.

**Measured transfer, same evening, same runner state:**

| tree | local vs crown bytes | ranked |
|---|---:|---:|
| crown bytes (`65188461`) | 0 | 607.19 |
| our `49197eb8` | +1.85% | 612.31 |
| fkiene `b864a72c` | +2.94% | 615.56 |
| `f9738952` family (5 draws) | +1.93% | 612.6–619.3, mean 615.6 |

The local-to-ranked transfer comes out at **~0.47** three ways:
- `49197eb8` vs crown: 0.45;
- `b864a72c` vs crown: 0.47;
- `b864a72c` vs `49197eb8`: 0.49.

That evening all three single draws transferred at about 0.45–0.49 of their local gain. The wider per-lineage table above (12-hour baselines, 45 draws) suggests the two-slot host pipe lifts transfer to about 1.0 or more. Without the pipe, this tree family would be expected near 617 on tonight's runner. If the pipe is worth about 1% there, as `35c4db43` → `e63e42ec` suggests, it would be expected near 622–623.

## Correctness

The device code differs from `b864a72c` only in loop structure and in the root inverse's warp distribution (see above). Both are exact. The host loop is our two-slot pipeline, which drains every batch exactly once and in batch order before its slot is reused; a timeout can only drop batches that were never drained. With the unmodified harness (`benchmark.sh`, 120 s, fresh problem seed), this exact package verified 10,938 of 10,938 hits: `RESULT: PASS`. The package without the warp inverse also ran 1,200 s locally: 762.0 M/s, yield 0.999. Every published hit is still recomputed by the unchanged exact host (OpenSSL) publication gate. The speculative filter can only lose a hit, never publish a wrong one.

## Base and attribution

- **Tree:** fkiene's `b864a72c`, credited as co-author. Its lineage:
  - jrcarlos2000's `cc35b5d5` redraw of hybridnoise's `f9738952` (K32 limb-0 corrections and the fused R² + PPP − 2Q reduction) on our PR1088 composite;
  - fkiene's base-A recoding, constant-bank R, lockstep second SHA, FMA-pipe gate adds, L2 fetch hint and filter-only z2 carry cut.
- **Promoted crown** `7aef224a` (Akashneelesh) and every contributor it credits: dun999, fkiene, Meganpark980320, ercumentyildirim, EvanYan1024, terrapinelf, jacklightChen, Saviour1001, owizdom, DPZZxlz, DrCleverHans, Babbaragga, mitchuski, kayu052, Portablelle.
- **Warp-distributed root inverse:** newjordan (`5b198ddf`), credited as co-author. Its port onto the b864 lineage is by i34-9 (`78208a18`), also credited.
- **Two-slot host pipeline:** our `a329eeee` / `49197eb8` line. It was modelled on the pinning frontier's loop (Saviour1001, terrapinelf, i34-9) and first brought to subset by Akashneelesh's `e63e42ec`, whose ranked draws motivated this port.
- **Runner data point:** carlosdelafi's late-window probe `668a3a1b` (self 612.9 vs score 619.27) showed the runner's late plateau.

All inherited source, GPLv3 notices (`COPYING`, VanitySearch headers) and attributions are retained.

## Packaging

Only `candidates/subset/` changes. The harness, verifier, problem, setup, benchmark, workflow and sibling track are untouched, and no binary or build stamp is included.
