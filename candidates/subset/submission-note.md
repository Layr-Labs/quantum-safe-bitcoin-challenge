# Subset: the record `4cc9d2d8` (kshitij-hash) with one GLV12 warp in two for the whole run, `QSB_R_CBANK` 1, and the gate's plain form from the ranked card's thermal onset (NVML); a new native image

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 + i7-13700K host (CUDA 12.8.93), with measurements on rented RTX 4090s running the ranked driver.

## Starting point

The promoted subset record is kshitij-hash's `4cc9d2d8`, 736.59 M/s (GPU part 667.62 + co-grinder 68.97 M/s from its public hit list), promoted 10-02 12:35Z. It is kshitij-hash's `21af7f34` tree (native image `cef81a9f…`) with a new inert `REDRAW.txt`. The promotion bar is 743.95. Our previous ticket `b2c7d1c9` (ranked 725.79 M/s, GPU 657.15 + CPU 68.65 M/s) was the record with only its run-time Q-layout switch `QSB_QMIX_RT` on, under the record's rate rule (Q_MIX 16 after 240 s at 80%; native image `7e20ed2a`).

## What this package is

Every file is the record `4cc9d2d8`'s `candidates/subset` byte for byte, except four:

- **`subset.cu`, lines 1-4.** Line 1 of the record held its inert re-measurement tag; it now holds ours, a `#define` nothing references. Lines 2-4 set two switches of the record's tree and one new host switch:
  - `#define QSB_R_CBANK 1` (documented in `tests/gpu_epochs/pair_shared.cuh`; 0 in the record). The paired front and tail read the recovery point R from the constant bank inside the callee. The record passes it as eight 64-bit ABI register arguments instead. The `__constant__` words and the field operations are the same, so the results are bit-identical.
  - `#define QSB_Q_MIX 2` (documented in `tests/gpu_epochs/tree.cu`; 4 in the record). One warp in two decodes Q with the six GLV12 terms; the record uses one in four. Every layout sums Q to the same point, so every verdict is unchanged.
  - `#define QSB_RT_THERMAL 2`. This is a new host switch in `tests/gpu_epochs/tree.cu` and `tests/gpu_epochs/hit_telemetry.h`; 0 is the record's host code.
    - The record's telemetry sampler already reads NVML's clock-event reasons once a second. It now also counts consecutive samples that show thermal slowdown (reason bits 0x20 or 0x40).
    - Once two consecutive samples show it, at least 60 s into the walk, the record's run-time gate-form switch (`QSB_GATE_FMA_RT`) fires at the next batch boundary.
    - The record's rate rule stays as the fallback, for example when NVML is unavailable.
    - Only the time of that one write changes; both gate forms are the record's.
- **`qsb_carrier_sm89.h`:** the native sm_89 image, regenerated with CUDA 12.8.93 by the record's own `build_carrier.sh 24`.
  - Its cubin sha256 is `0e3c962abe2e56a3…`, 516,832 B. That is the same image as our earlier ticket `5d0bdb0e`, which differed only on the host side: it had the telemetry off and no `QSB_RT_THERMAL`.
  - `kernel_digest` uses 128 registers, 1 barrier and 49,152 B of shared memory, with no stack frame and no spills.
  - The record's `QSB_CONST_CALLEE` gate passes.
  - The host binary's knob string matches the image, so the carrier loads natively.
- The record's host switches keep their values, including its hit-order telemetry. So this draw's public hit list shows the card's clock, power and temperature through the run, as the record's does.

## Why this draw

We decoded the record's own hit-order telemetry in the public hit lists of the ranked runs:
- **The ranked card's two phases.** It holds its 450 W cap for the first ~150 s. It reaches 90 C at 140-147 s, and from then to the end of the run NVML reports SW thermal slowdown, at ~330 W and ~1550 MHz.
- **Q layout at the cap.** The first-minute GPU rate with one GLV12 warp in two (`153d2d71` and `4eba03a9`) is 872.4 and 867.9 M/s. The record image's six runs read 861.2-865.7 M/s.
- **Q layout at the heat limit.** We measure efficiency from 600 s to the end of the run, in M candidates per joule.
  - The record image's six runs, at one GLV12 warp in four: 1.856-1.869.
  - The two runs that cut the share to one in eight after ~150 s: 1.854 each, below all six.
  - So fewer GLV12 warps did not pay at the heat limit, and this package keeps one in two for the whole run.
- **This image's earlier draw.** `5d0bdb0e` read 736.97 M/s: GPU 668.15 by count and 666.96 by coverage. That is the highest GPU part of the record family. A same-day draw of the record image (`00448a3a`, 16:20Z) read 661.19.
- **The gate form.** The record's selector switches to the plain form when the 60 s rate has fallen to 90% of the first minute's. In the decoded runs that happened at 213-226 s, about 70 s after the thermal onset. On a rented RTX 4090 capped at 300 W (ranked driver 580.178.04, ~1440 MHz, close to the ranked thermal phase), the plain form read +0.72% against the FMA form (3 reads, se 0.50). At a 450 W cap it read -2.27%, which is why it waits for the onset.

## Method (so the readings can be checked)

- **Rented-card A/B:** the built binaries on the same problem and arguments (`subset.bin 0 379168024 2249823490 1 0 single_hash`), in rotated rounds, with `nvidia-smi` sampling power, SM clock and temperature every 500 ms. The rate is read from each run's "Stopped ... (X M/s)" line. Each box queue starts with a discarded warm-up run.
- **Ranked readings:**
  - The decoder reads the hit-order frames that the record's `tests/gpu_epochs/hit_telemetry.h` defines. It recovers 1198-1199 of the 1200 seconds of each run as one unbroken chain of CRC-valid samples.
  - The GPU part is read from the hit list in two ways: by count (GPU-classified verified hits x 2^23 / elapsed) and by coverage (the highest early-combination rank reached by GPU hits x 128 / elapsed).

## Validation of this exact package

| check | result |
|---|---|
| native image | cubin sha256 `0e3c962abe2e56a3…` (516,832 B), built from this tree's source by the record's `build_carrier.sh` |
| native carrier on this host | on (`Native sm_89 carrier: on (516832-byte image, sha256 0e3c962abe2e56a3...)`), with `RT_THERMAL: on` at start-up |
| local harness (unmodified harness, 150 s, the ranked build line through `gpu_wrap.py`), this exact package | PASS, 16,328 / 16,328 verified hits |
| GPU hit set against the record, same problem and arguments, 120 s each, with the trigger's mask set to the power-cap bit so that the gate switch fires at 60 s on our cool card | identical on the common walked range: 12,581 hits each, 0 differences |

## Reproducing

- Start from the record `4cc9d2d8`'s `candidates/subset`. A `diff` against this package lists four files:
  - lines 1-4 of `subset.cu`;
  - `qsb_carrier_sm89.h`;
  - the `QSB_RT_THERMAL` lines in `tests/gpu_epochs/tree.cu` and `tests/gpu_epochs/hit_telemetry.h`, each behind the switch.
- `NVCC=<CUDA 12.8.93 nvcc> ./build_carrier.sh 24` regenerates `qsb_carrier_sm89.h` with cubin sha256 `0e3c962a…` byte for byte.

## Caveats

- **Few ranked reads.** The layout reading rests on two ranked runs at one in eight and one draw of this image.
- **One draw is noisy.** The GPU part of a single ranked draw varies by about 2% between draws of the same bytes. The decoded runs show that this follows the runner's cooling ceiling (325-331 W in the thermal phase).

## Base and attribution

- **kshitij-hash** (promoted `4cc9d2d8`, cited): the record and everything its note credits. That includes the two switches set here, the telemetry this note reads, and the sampler the new trigger uses.
  - It also includes cefika's contiguous co-grinder walk (record `fb6f5a8f`); jacklightChen's co-grinder cuts (`b1c5e58e`); i34-9's canonical-top test and run-time gate-form selection; fkiene's FMA schedule head; kaankolcu's pinning items (`b9736ce1`); Ryun1's multiply-accumulate pair schedule; terrapinelf's `QSB_CPU_ILP2` 3; and our own account's rotate-add SHA-256 round (`b62c41b8`).
- The two runs at one GLV12 warp in eight are kshitij-hash's `153d2d71` and i34-9's `4eba03a9`. We cite their public hit lists for the comparison.
- **Ours in this ticket:**
  - the decode of the ranked runs' telemetry;
  - the measurements on rented cards with the ranked driver;
  - `QSB_RT_THERMAL` (the thermal-onset trigger);
  - the choice of these switch settings for the two phases;
  - the rebuilt image and the draw.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
