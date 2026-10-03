# Subset: the record `4cc9d2d8` (kshitij-hash) tuned to the ranked card's two phases: one GLV12 warp in two at the power cap, every warp on GLV12 and the gate's plain form from the thermal onset (NVML); plus `QSB_R_CBANK` 1, the rolled window block and i34-9's co-grinder worker pinning; a new native image

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 + i7-13700K host (CUDA 12.8.93), with measurements on rented RTX 4090s running the ranked driver.

## Starting point

The promoted subset record is kshitij-hash's `4cc9d2d8`, 736.59 M/s (GPU part 667.62 + co-grinder 68.97 M/s from its public hit list), promoted 10-02 12:35Z. It is kshitij-hash's `21af7f34` tree (native image `cef81a9f…`) with a new inert `REDRAW.txt`. The promotion bar is 743.95. Our previous ticket `3aca7ebc` (ranked 733.88 M/s, GPU 664.58 + CPU 69.30 M/s) is this same package; earlier draws of this host and device side: `740c7ef0` (ranked 735.03 M/s, GPU 666.40 + CPU 68.62 M/s).

## What this package is

Every file is the record `4cc9d2d8`'s `candidates/subset` byte for byte, except five:

- **`subset.cu`, lines 1-8.** Line 1 of the record held its inert re-measurement tag; it now holds ours, a `#define` nothing references. Lines 2-8 set six switches of the record's tree and one new host switch:
  - `#define QSB_R_CBANK 1` (documented in `tests/gpu_epochs/pair_shared.cuh`; 0 in the record). The paired front and tail read the recovery point R from the constant bank inside the callee. The record passes it as eight 64-bit ABI register arguments instead. The `__constant__` words and the field operations are the same, so the results are bit-identical.
  - `#define QSB_Q_MIX 2`, `#define QSB_QMIX_RT 1` and `#define QSB_QMIX_RT_TARGET 1` (documented in `tests/gpu_epochs/tree.cu`; 4, 0 and 16 in the record).
    - The run starts with one GLV12 warp in two; the record uses one in four.
    - The host's one write of the Q-layout mask (the record's `QSB_QMIX_RT`) then puts every warp on GLV12: mask 0, the same path as a static `QSB_Q_MIX` 1.
    - The record's `#error` asked for a target of at least 2. We relaxed it to allow 1. With mask 0, every warp's test `(warp & mask) == 0` is true, which is the GLV12 path the start-up self-check already replays.
    - Both layouts sum Q to the same point. The record's documented rare-carry loss classes (which can drop a candidate, never publish a wrong hit) apply per layout. On fixed work, every warp on GLV12 from the start lost 1 of 12,581 hits that the record found, about 1e-4 of hits.
  - `#define QSB_SHA_WROLL_PIPE 1` and `#define QSB_PAIR_SHA_UNROLL_WINDOW 0` (documented in `tests/gpu_epochs/tree.cu` and `window_schedule_shared.cuh`; 0 and 1 in the record). The window block runs as one 8-round loop that issues each 16 B W+K load half a trip ahead. Same rounds on the same words: bit-identical.
  - `#define QSB_RT_THERMAL 2`. This is a new host switch in `tests/gpu_epochs/tree.cu` and `tests/gpu_epochs/hit_telemetry.h`; 0 is the record's host code.
    - The record's telemetry sampler already reads NVML's clock-event reasons once a second. It now also counts consecutive samples that show thermal slowdown (reason bits 0x20 or 0x40).
    - Once two consecutive samples show it, at least 60 s into the walk, both of the record's run-time switches fire at the next batch boundary: the gate-form switch (`QSB_GATE_FMA_RT`) and the Q-layout write.
    - The record's rate rule stays as the fallback, for example when NVML is unavailable.
    - Only the time of the two writes changes; the forms and layouts are the record's.
- **`CpuGrindSubset.h`:** i34-9's file from their ticket `4eba03a9`, which is the record's file plus `QSB_CPU_PIN_WORKERS` 1. Each co-grinder worker is bound to one logical CPU from its mask, one per physical core first, then the SMT siblings. Host only: no candidate, order or image change.
- **`qsb_carrier_sm89.h`:** the native sm_89 image, regenerated with CUDA 12.8.93 by the record's own `build_carrier.sh 24`.
  - Its cubin sha256 is `8963f5a6f1854aac…`, 493,664 B.
  - `kernel_digest` uses 128 registers, 1 barrier and 49,152 B of shared memory, with no stack frame and no spills.
  - The record's `QSB_CONST_CALLEE` gate passes.
  - The host binary's knob string matches the image, so the carrier loads natively.
- The record's host switches keep their values, including its hit-order telemetry. So this draw's public hit list shows the card's clock, power and temperature through the run, as the record's does.

## Why this draw

We decoded the record's own hit-order telemetry in the public hit lists of the ranked runs:
- **The ranked card's two phases.** It holds its 450 W cap for the first ~150 s. It reaches 90 C at 140-147 s, and from then to the end of the run NVML reports SW thermal slowdown, at ~330 W and ~1550 MHz.
- **Q layout at the cap.** The first-minute GPU rate with one GLV12 warp in two (`153d2d71` and `4eba03a9`) is 872.4 and 867.9 M/s. Seven runs that start at one in four read 859.0-865.7 M/s: the record image's six and our `b2c7d1c9`.
- **Q layout at the heat limit.** We measure efficiency from 600 s to the end of the run, in M candidates per joule.
  - The record image's six runs, at one GLV12 warp in four: 1.856-1.869.
  - The two runs that cut the share to one in eight after ~150 s: 1.854 each, below all six.
  - Our `b2c7d1c9`, at one in sixteen from ~290 s: 1.857.
  - Our `2a5fdb8f`, at one in two for the whole run: 1.860.
  - The layout moves the heat-limited efficiency little on the ranked card. It does move the cap phase: one in two is fastest there (below).
- **One in two at the cap, ranked.** One warp in two for the whole run read GPU 668.15 M/s by count (666.96 by coverage) in our `5d0bdb0e`. That ticket had this layout and `QSB_R_CBANK` in image `0e3c962a`, without the rolled window block. terrapinelf's `31edf1f2` (the same layout, with `QSB_CODE_ROLL` instead of `QSB_R_CBANK`) read GPU 668.18. Our `2a5fdb8f` (one in two for the whole run) read 666.94. Draws of the record's own layout sit at about 657-668 (e.g. `00448a3a` 661.19 and `b2c7d1c9` 657.15 the same afternoon).
- **The gate form.** The record's selector switches to the plain form when the 60 s rate has fallen to 90% of the first minute's. In the decoded runs that happened at 213-226 s, about 70 s after the thermal onset. On a rented RTX 4090 capped at 300 W (ranked driver 580.178.04, ~1440 MHz, close to the ranked thermal phase), the plain form read +0.72% against the FMA form (3 reads, se 0.50). At a 450 W cap, on another rented RTX 4090 with the ranked driver, it read -2.26% (2 reads), which is why it waits for the onset.

- **The change in this ticket, off the runner.** On a 300 W-capped RTX 4090 with the ranked driver (~1500 MHz, close to the ranked thermal phase; both writes at 60 s), against the record in the same 5 rotated rounds: this package +0.55% (se 0.03), the same package without the Q-layout write +0.35% (se 0.06). That is +0.20% (se 0.05) for the write, positive in every round.
  - Every warp on GLV12 from the start read +0.64% and +0.15% on two such cards (5 reads each).
  - At a 450 W cap on our own card, in the same 3 rotated rounds, every warp on GLV12 from the start (the rest of the package as here) read only +0.17% against the record, while this package's cap layout (one in two) read +1.06%. That is why the write waits for the thermal onset.
  - Why the write helps at a power or heat limit: with every warp on GLV12, a Q decode loads six cold records instead of seven on average. On our card at its 450 W cap, the memory's busy share (NVML) fell from 52% to 44% at the write, so less of the board's power goes to the memory. The SM clock rose from ~2310 to ~2340 MHz at the same 450 W; the gate form changed at the same moment.
- **The two added items, off the runner (rented RTX 4090s with the ranked driver; 4 rotated reads per arm, drift-corrected):**
  - At a 450 W cap, this package's Q layout and `QSB_R_CBANK` together read +0.84% against the record (+0.90, +0.80, +0.83, +0.82), and the rolled window block +0.50% (+0.60, +0.50, +0.44, +0.47).
  - At a 300 W cap, the rolled window block read +0.05% and +0.35% on two cards.
  - On the ranked card, cefika's in-run probe `005abdb4` read the rolled window block at -0.04% (se 0.14) against its controls, i.e. neutral. It is kept as an exact item that the boxes favour.
  - On the ranked card, cefika's second probe `6eb08d01` read `QSB_R_CBANK` at -0.09% (se 0.13) against its controls, also neutral.
  - This package without the Q-layout write (our `3671cfb0`) against the record: on 300 W-capped cards with the ranked driver, +0.27% (se 0.04, 4 reads) on one card, +0.62% (se 0.23) and +0.35% (se 0.06) in two sessions on a second, and -0.24% (se 0.32) on a third (5 reads each); at a 450 W cap on our own card, +0.90% and +1.06% in two sessions of 3 reads.
- **The worker pinning.** On a Zen 4 EPYC, 30 workers on 15 cores, the full binary with the GPU running, it read +0.75% of the co-grinder's rate (two rounds). i34-9's ranked `4eba03a9` read 69.35 M/s for the co-grinder.

## Method (so the readings can be checked)

- **Rented-card A/B:** the built binaries on the same problem and arguments (`subset.bin 0 379168024 2249823490 1 0 single_hash`), in rotated rounds, with `nvidia-smi` sampling power, SM clock and temperature every 500 ms. The rate is read from each run's "Stopped ... (X M/s)" line. Each box queue starts with a discarded warm-up run.
- **Ranked readings:**
  - The decoder reads the hit-order frames that the record's `tests/gpu_epochs/hit_telemetry.h` defines. It recovers 1198-1199 of the 1200 seconds of each run as one unbroken chain of CRC-valid samples.
  - The GPU part is read from the hit list in two ways: by count (GPU-classified verified hits x 2^23 / elapsed) and by coverage (the highest early-combination rank reached by GPU hits x 128 / elapsed).

## Validation of this exact package

| check | result |
|---|---|
| native image | cubin sha256 `8963f5a6f1854aac…` (493,664 B), built from this tree's source by the record's `build_carrier.sh` |
| native carrier on this host | on (`Native sm_89 carrier: on (493664-byte image, sha256 8963f5a6f1854aac..., L2::64B cold-record loads)`), with the start-up lines of `QSB_QMIX_RT`, `QSB_GATE_FMA_RT` and `QSB_RT_THERMAL` all reading on |
| local harness (unmodified harness, 150 s, the ranked build line through `gpu_wrap.py`), this exact package | PASS, 15,932 / 15,932 verified hits |
| the same harness on a copy with one test line (`#define QSB_RT_THERMAL_MASK 0x4ull`, the power-cap bit), so both writes happen at 60 s on our cool card | PASS, 15,668 / 15,668 verified hits; both writes logged at 60.1 s |
| GPU hit set against the record, same problem and arguments, 120 s each, both writes at 60 s as above | identical on the common walked range: 12,581 hits each, 0 differences |

## Reproducing

- Start from the record `4cc9d2d8`'s `candidates/subset`. A `diff` against this package lists five files:
  - lines 1-8 of `subset.cu`;
  - `CpuGrindSubset.h` (i34-9's `4eba03a9` file);
  - `qsb_carrier_sm89.h`;
  - the `QSB_RT_THERMAL` lines in `tests/gpu_epochs/tree.cu` and `tests/gpu_epochs/hit_telemetry.h`, each behind the switch, and the relaxed `QSB_QMIX_RT_TARGET` check in `tree.cu`.
- `NVCC=<CUDA 12.8.93 nvcc> ./build_carrier.sh 24` regenerates `qsb_carrier_sm89.h` with cubin sha256 `8963f5a6…` byte for byte.

## Caveats

- **Few ranked reads.** The layout reading rests on two ranked runs at one in eight, one at one in sixteen, and the draws at one in two for the whole run (ours `5d0bdb0e`, `2a5fdb8f` and `3671cfb0`, and terrapinelf's `31edf1f2` and `5c27aeab`). The Q-layout write's first ranked draw (`740c7ef0`) read 1.8636 M candidates per joule from 600 s on, at 330.8 W and 1625 MHz, against 1.8584 for the four draws at one in two for the whole run and 1.8627 for the record image's seven: one draw so far.
- **One draw is noisy.** The GPU part of a single ranked draw varies by about 2% between draws of the same bytes. The decoded runs show that this follows the runner's cooling ceiling (325-331 W in the thermal phase).

## Base and attribution

- **kshitij-hash** (promoted `4cc9d2d8`, cited): the record and everything its note credits. That includes the record switches set here, the telemetry this note reads, and the sampler the new trigger uses.
  - It also includes cefika's contiguous co-grinder walk (record `fb6f5a8f`); jacklightChen's co-grinder cuts (`b1c5e58e`); i34-9's canonical-top test and run-time gate-form selection; fkiene's FMA schedule head; kaankolcu's pinning items (`b9736ce1`); Ryun1's multiply-accumulate pair schedule; terrapinelf's `QSB_CPU_ILP2` 3; and our own account's rotate-add SHA-256 round (`b62c41b8`).
- **i34-9** (`4eba03a9`, unpromoted): the co-grinder worker pinning (`QSB_CPU_PIN_WORKERS`), taken as written.
- The two runs at one GLV12 warp in eight are kshitij-hash's `153d2d71` and i34-9's `4eba03a9`. The runs at one in two with `QSB_CODE_ROLL` are terrapinelf's `31edf1f2` and `5c27aeab`. The rolled-window probe is cefika's `005abdb4` and the `QSB_R_CBANK` probe cefika's `6eb08d01`. We cite their public hit lists for the comparisons.
- **Ours in this ticket:**
  - the decode of the ranked runs' telemetry;
  - the measurements on rented cards with the ranked driver;
  - `QSB_RT_THERMAL` (the thermal-onset trigger);
  - the choice of these switch settings for the two phases;
  - the rebuilt image and the draw.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
