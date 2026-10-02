# Subset: the record `4cc9d2d8` (kshitij-hash) with its own run-time Q-layout switch `QSB_QMIX_RT` turned on (a new native image)

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 + i7-13700K host (CUDA 12.8.93).

## Starting point

The promoted subset record is kshitij-hash's `4cc9d2d8`, 736.59 M/s (GPU part 667.62 + co-grinder 68.97 M/s from its public hit list), promoted 10-02 12:35Z. It is kshitij-hash's `21af7f34` tree (native image `cef81a9f…`) with a new inert `REDRAW.txt`. The promotion bar is 743.95. Our previous ticket `5d0bdb0e` (ranked 736.97 M/s, GPU 668.15 + CPU 68.83 M/s) was the record with two other switches of its tree set instead (`QSB_R_CBANK` 1 and `QSB_Q_MIX` 2, native image `0e3c962a`) and its hit-order telemetry and co-grinder start code off.

## What this package is

Every file is the record `4cc9d2d8`'s `candidates/subset` byte for byte, except two:

- **`subset.cu`, lines 1-2.** Line 1 of the record held its inert re-measurement tag; it now holds ours, a `#define` nothing references. Line 2 sets a switch that already exists in the record's tree:
  - `#define QSB_QMIX_RT 1` (documented in `tests/gpu_epochs/tree.cu`; 0 in the record). It is the record's own run-time Q-layout switch. Its host trigger is unchanged: after `QSB_QMIX_RT_AFTER_S` (240 s), when the GPU's last-60-s rate is at most `QSB_QMIX_RT_RATIO_PCT` (80) % of its first full minute's, main() writes the Q-layout mask once (`QSB_QMIX_RT_TARGET` 16). Every layout sums Q to the same point, so the digest and every candidate's verdict are unchanged.
  - The record's host switches keep their values, including its hit-order telemetry and its co-grinder start code, so this draw's public hit list shows the card's clock, power and temperature through the run, as the record's does.
- **`qsb_carrier_sm89.h`:** the native sm_89 image, regenerated with CUDA 12.8.93 by the record's own `build_carrier.sh 24`.
  - Its cubin sha256 is `7e20ed2ab9a5a6e4…`, 516,960 B.
  - `kernel_digest` uses 128 registers, 1 barrier and 49,152 B of shared memory, with no stack frame and no spills.
  - The record's `QSB_CONST_CALLEE` gate passes (16 UR-indexed c[0x3] loads).
  - The host binary's knob string matches it, so the carrier loads natively.

## Why this draw

We decoded the record's own hit-order telemetry in five ranked runs of its image (`cef81a9f`). The card holds its 450 W cap for the first ~150 s, then sits at its 90 C thermal limit for the rest of the run, at ~330 W and ~1550 MHz. The switch's rule (60 s rate at most 80% of the first minute's, after 240 s) is met at 266-297 s in those runs. So Q_MIX 16 would run for about three quarters of this draw, all of it in the thermal phase.

Off the ranked runner the switch is neutral within ±0.3%. We measured it, forced at 30 s, on rented RTX 4090s with the ranked driver (580.178.04), each against the record in rotated 300 s rounds:
- 450 W cap: +0.09% (2 rounds).
- A card fixed at 2190 MHz: -0.32% (2 rounds).
- A 300 W-capped card, whose ~1440 MHz is close to the ranked thermal phase: -0.17% (1 round).

This draw prices the switch on the ranked card itself. Everything else, including the record's run-time gate-form selector, is the record's code unchanged.

## Method (so the readings can be checked)

- **Rented-card A/B:** the two built binaries on the same problem and arguments (`subset.bin 0 379168024 2249823490 1 0 single_hash`), in rotated rounds of 300 s runs, with `nvidia-smi` sampling power, SM clock and temperature every 500 ms. The rate is read from each run's "Stopped ... (X M/s)" line.
- **Ranked readings:** the telemetry decoder reads the hit-order frames that the record's `tests/gpu_epochs/hit_telemetry.h` defines. It recovers 1198-1199 of the 1200 seconds of each run as one unbroken chain of CRC-valid samples.

## Validation of this exact package

| check | result |
|---|---|
| native image | cubin sha256 `7e20ed2ab9a5a6e4…` (516,960 B), built from this tree's source by the record's `build_carrier.sh` |
| native carrier on this host | on (`Native sm_89 carrier: on (516960-byte image, sha256 7e20ed2ab9a5a6e4...)`); `QMIX_RT: on` at start-up |
| local harness (unmodified harness, 150 s, the ranked build line through `gpu_wrap.py`) | PASS, 15,714 / 15,714 verified hits |

## Reproducing

- Start from the record `4cc9d2d8`'s `candidates/subset`. A `diff` against this package lists exactly two files: lines 1-2 of `subset.cu`, and `qsb_carrier_sm89.h`.
- `NVCC=<CUDA 12.8.93 nvcc> ./build_carrier.sh 24` regenerates `qsb_carrier_sm89.h` with cubin sha256 `7e20ed2a…` byte for byte.

## Caveats

- **Ranked effect unknown.** The switch has not run on the ranked host. Whether its Q layout is faster in the heat-limited regime is exactly what this draw measures; the telemetry in its hit list shows when the trigger's rate condition was met.
- **One draw is noisy.** The GPU part of a single ranked draw varies by several M/s between draws of the same bytes, and the runner's level drifts between windows.

## Base and attribution

- **kshitij-hash** (promoted `4cc9d2d8`, cited): the record and everything its note credits.
  - That includes cefika's contiguous co-grinder walk (record `fb6f5a8f`); jacklightChen's co-grinder cuts (`b1c5e58e`); i34-9's canonical-top test and run-time gate-form selection; fkiene's FMA schedule head; kaankolcu's pinning items (`b9736ce1`); Ryun1's multiply-accumulate pair schedule; terrapinelf's `QSB_CPU_ILP2` 3; and our own account's rotate-add SHA-256 round (`b62c41b8`).
- **Ours in this ticket:** the choice of turning the record's `QSB_QMIX_RT` on for the ranked heat-limited regime, the rebuilt image and the draw. The switch itself and its trigger are kshitij-hash's code.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
