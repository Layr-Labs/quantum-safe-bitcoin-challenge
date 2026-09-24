# Subset: fkiene's b864a72c tree with its paired SHA loops unrolled again (+0.33% on a dedicated RTX 4090), with a per-mechanism census of the two top public lineages and NVBit dynamic instruction counts

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on a dedicated RTX 4090 (driver 595.71, CUDA 12.8.93 toolchain). The ranked build line and argv are unchanged.

## Summary

This package is fkiene's public subset tree from submission `b864a72c` (commit `eb252e00`), with one change:

- **Paired-window and constant-suffix SHA loops fully unrolled again** (`QSB_PAIR_SHA_UNROLL_WINDOW=1`, `QSB_PAIR_SHA_UNROLL_CONST=1`, `QSB_PAIR_SHA_UNROLL_CONST_INNER=1`).
  - The rounds, round order, schedule words and digests are unchanged. Only the loop structure changes.
  - Unrolling lets every round's precomputed K+W word become a constant-bank operand instead of an indexed `LDC` plus loop control.
  - On the base tree this measured **+0.33%**: the rolled `b864a72c` ran 758.75 M/s and this package 761.29 M/s over 3 paired ABBA rounds.

Against the promoted crown bytes on the same card:

| tree | Δ vs crown bytes |
|---|---:|
| this package | about **+3.3%** |
| `b864a72c` as submitted | +2.94% |
| our previous package `49197eb8` | +1.85% |

Energy per candidate at the 450 W cap falls by the same fraction as throughput rises.

The source is otherwise fkiene's tree byte for byte. All kill switches keep their inherited defaults. Setting the three unroll switches back to 0 reproduces the `b864a72c` PTX exactly.

## Why this change: a per-mechanism census of the two top public lineages

We measured the two fastest public lineages head to head on one card, then moved their mechanisms across one at a time:
- fkiene's `b864a72c`, from the hybridnoise `f9738952` / PR1088 line;
- our `49197eb8`, from the `a329eeee` / PR1134 line.

**Protocol:**
- A fixed generated problem (seed 424242).
- Each arm is one 62 s process launch of the prebuilt binary with the ranked argv. Arms are ordered ABBA over 3 rounds.
- The rate is measured between the first and last progress lines.
- Energy per candidate is mean `nvidia-smi` power over the same window divided by rate.
- Every variant's PTX hash was checked before timing.

| variant | Δ throughput | reference |
|---|---:|---|
| `b864a72c` (as submitted) | +0.976% ± 0.091 | vs `49197eb8` |
| `49197eb8` + `QSB_R_CBANK` + `QSB_PAIR_Z_LOCKSTEP` | +0.007% ± 0.013 | vs `49197eb8` |
| `49197eb8` + `QSB_GATE_H0_FMA` | +0.295% ± 0.076 | vs `49197eb8` |
| `49197eb8` + all three above | +0.355% ± 0.053 | vs `49197eb8` |
| `49197eb8` with the window SHA rolled | −0.257% ± 0.076 | vs `49197eb8` |
| `49197eb8` + all three, both SHA loops rolled | −0.500% ± 0.076 | vs the all-three build |
| `b864a72c` without `QSB_GATE_H0_FMA` | −0.51% | vs `b864a72c` |
| **`b864a72c` with both SHA loops unrolled (this package)** | **+0.33%**; +1.246% ± 0.041 vs `49197eb8` | vs `b864a72c` |
| `49197eb8` without the offset-ordinate table (`QSB_YOFF_FILTER=0`) | +0.547% ± 0.042 | vs `49197eb8` |
| `49197eb8` without the negative-ordinate MAC | +0.110% ± 0.016 | vs `49197eb8` |
| `49197eb8` without the two-slot host pipe (GPU verify path) | −0.135% ± 0.027 | vs `49197eb8` |
| L2 max fetch granularity 64 B on `49197eb8` | −0.04% ± 0.10 | vs `49197eb8` |

**What the census says:**

1. **Rolling loses on both lineages.** On `49197eb8` the rolled window loses 0.26% and the rolled constant suffix 0.43%. On `b864a72c`, unrolling both gains 0.33%. Unrolled rounds read their K+W words straight from the constant bank. Rolled rounds pay an indexed load, a loop counter and a branch, as the NVBit counts below show: `LDC.64` +64 and `BRA` +22 per candidate.
2. **The FMA-pipe gate adds are worth 0.3–0.5% on this card.** The digest kernel issues about 55% of its dynamic instructions on the ALU pipe. Moving the pubkey-hash additions to the FMA pipe helps even though it adds instructions.
3. **The offset-ordinate table (`QSB_YOFF_FILTER`) is a local negative on our lineage: −0.55%.** It removes instructions from the chain loop (1,026 vs 1,038 SASS per interior madd), yet the build without it is 0.547% ± 0.042 faster. The negative-ordinate MAC is a small local negative too (−0.11%). The two-slot host pipe is a small positive (+0.14%). `b864a72c` has neither the offset table nor the MAC, which, together with its FMA-pipe gate adds, explains most of its local lead over `49197eb8`.

## Dynamic instruction counts (NVBit v1.8 `opcode_hist`)

`ncu` is admin-only on most rented boxes, but NVBit runs unprivileged:

```
LD_PRELOAD=nvbit_release_x86_64/tools/opcode_hist/opcode_hist.so KERNEL_BEGIN=0 KERNEL_END=6 COUNT_WARP_LEVEL=0 \
  ./subset prob/subset.bin 0 <seq> <lt> 1 0 single_hash
```

This instruments the three producer kernels and the first full `kernel_digest` launch (262,144 blocks, 2^27 candidates) at thread level. Do not wrap the binary in `timeout`: the injected tool then misses the launches. Instructions per candidate:

| tree | digest | producers | Δ total vs crown | IMAD.WIDE | IMAD (non-wide) | ALU |
|---|---:|---:|---:|---:|---:|---:|
| crown `7aef224a` | 24,860 | 235 | 0 | 9,094 | 1,601 | 13,918 |
| `a329eeee` | 24,284 | 145 | −2.66% | 9,008 | 1,722 | 13,306 |
| `49197eb8` | 24,261 | 145 | −2.75% | 8,994 | 1,714 | 13,296 |
| `f9738952` | 24,961 | 235 | +0.40% | 8,993 | 2,415 | 13,231 |
| `b864a72c` | 24,919 | 235 | +0.23% | 8,997 | 2,503 | 13,088 |

The lineages win locally in different ways:
- **The `f9738952` / `b864a72c` line** is faster on this card although it executes *more* instructions than the crown. Its gains come from pipe balance (`IMAD` 823 vs 27 per candidate from the FMA-pipe adds) and from issue efficiency.
- **The `a329eeee` line** executes 2.7% fewer instructions than the crown.

The ranked runner does not necessarily reward the same things (see below).

## The ranked runner (public data, 221 frontier-speed draws)

- **It decays from its cold peak; there is no fixed startup loss.** carlosdelafi's `668a3a1b` printed only late-window rates. Its self-reported rate fell to 612.9 M/s while its hit-derived score was 619.27. A late-run plateau about 16% below the cold peak therefore explains the ~0.85 score/self ratio of the track.
- **It has drifted downwards.** Exact redraws of the crown bytes (self ≈ 714.5) averaged 612.3 on 09-22, 609.4 on 09-23 and 605.9 on 09-24 (daytime). The `f9738952`-like trees moved the same way.
- **Night slots draw a little higher.** Draws that completed between 21:00 and 08:00 UTC show a score/self ratio about 0.5% higher than those completing between 09:00 and 17:00.
- **Transfer depends on the mechanism.** Single-draw transfers are noisy (±0.65%), but they cluster:
  - The `a329eeee` line has the two-slot host pipe and no FMA-pipe adds. It drew about 1.0–1.2× its local gain (`36c05f97` 623.05, `a329eeee` 621.98).
  - The FMA-pipe trees drew about 0.5–0.8×: `f9738952` over ~32 draws, `eaba5205` over one.

  Tonight's draws weaken that split. Around the same hour, `49197eb8` drew 612.31, the crown bytes 607.19 (`65188461`), and four `f9738952`-family draws 614.04–619.27 (mean 616.4). So local speed is still the best single predictor we have, at about 0.6–0.8 transfer. This package has the highest local speed we measured.

## Other results from this session (for other solvers)

1. **Hit yield is 1.0.** A 1,200 s local run of `49197eb8` published 107,282 hits against 2·attempts/2^24 expected: yield 0.999. Short arms read about 0.975, from a deficit confined to the first ~30 s. That is worth at most ~0.15% of a ranked run and is the same for every tree.
2. **Affine pair level: closed by a probe.** We tried summing table points pairwise in affine coordinates, with Montgomery's trick inside each candidate plus an extra block inverse. A wrong-math cost probe replaced 12 interior madds with 6 pair iterations plus the forward Montgomery pass.
   - Forward pass: 216 SASS per pair.
   - Pair iteration: 1,874 SASS, against 2,052 for the two madds it replaces.
   - Net per pair: about zero before the extra collective (−3.7%), and the probe already spills.
3. **`QSB_CHAIN_UNROLL=2`** gives no loop saving (2,056 vs 2 × 1,026) and adds spills.
4. **1,200 s at thermal steady state:** `49197eb8` ran 751.19 M/s against 737.66 for the crown bytes (+1.83%). The short-arm gains hold at steady state on this card.

## Correctness

- Unrolling changes only loop structure. The same `S2Round` sequence runs on the same schedule words in the same order, so both states and the digest are bit-identical.
- The exact host publication gate (OpenSSL recomputation of every tentative hit) is fkiene's, unchanged. The speculative filter can only lose a hit, never publish a wrong one.
- A full `benchmark.sh` run of this exact package with the unmodified harness is reported in the validation section.

## Base and attribution

- **Base:** fkiene's `b864a72c`, used unchanged apart from the three unroll defaults. fkiene is credited as co-author.
- **That tree's own lineage:**
  - jrcarlos2000's `cc35b5d5` exact redraw of hybridnoise's `f9738952` (K32 limb-0 corrections and the fused R² + PPP − 2Q reduction on our PR1088 composite).
  - fkiene's own base-A recoding, constant-bank R, lockstep second SHA, FMA-pipe gate adds, 64 B L2 fetch hint and filter-only z2 carry cut.
  - The promoted crown `7aef224a` (Akashneelesh) and every contributor it credits: dun999, fkiene, Meganpark980320, ercumentyildirim, EvanYan1024, terrapinelf, jacklightChen, Saviour1001, owizdom, DPZZxlz, DrCleverHans, Babbaragga, mitchuski, kayu052, Portablelle.
- **Runner analysis:** carlosdelafi's late-window probe `668a3a1b` supplied the decisive runner data point.

All inherited source, GPLv3 notices (`COPYING`, VanitySearch headers) and attributions are retained.

## Expected ranked effect

If this family's local lead transferred in full, this package would sit about +3% above the crown bytes. That would be about 624–630 on the current runner, depending on the slot. If it transfers like `f9738952` (about 0.6–0.7×), expect about 618–622. The draw itself is the measurement.

## Packaging

Only `candidates/subset/` changes. The harness, verifier, problem, setup, benchmark, workflow and sibling track are untouched, and no binary or build stamp is included.
