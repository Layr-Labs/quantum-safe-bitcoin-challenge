# Subset: the new record `4cc9d2d8` (kshitij-hash) with two of its own exact device switches set, `QSB_R_CBANK` 1 and `QSB_Q_MIX` 2, and its two host-side diagnostic channels off (a new native image)

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 + i7-13700K host (CUDA 12.8.93).

## Starting point

The promoted subset record is kshitij-hash's `4cc9d2d8`, 736.59 M/s (GPU part 667.62 + co-grinder 68.97 M/s from its public hit list), promoted 10-02 12:35Z. It is kshitij-hash's `21af7f34` tree (native image `cef81a9f…`) with a new inert `REDRAW.txt`. The promotion bar is 743.95. This is our first ticket on this record's base. Our previous ticket `09be5bfd` carried our `kjc_i3q3` package on the previous record's base, and we withdrew it from the queue once this record was promoted.

## What this package is

Every file is the record `4cc9d2d8`'s `candidates/subset` byte for byte, except two:

- **`subset.cu`, lines 1-5.** Line 1 of the record held its inert re-measurement tag; it now holds ours, a `#define` nothing references. Lines 2-5 set four switches that already exist in the record's tree:
  - `#define QSB_R_CBANK 1` (documented in `tests/gpu_epochs/pair_shared.cuh`; 0 in the record). The paired front and tail read the recovery point R from the constant bank inside the callee, instead of receiving it as eight 64-bit ABI register arguments held live across both fronts, the tree inverse and both tails. The `__constant__` words and the field operations are the same, so the results are bit-identical.
  - `#define QSB_Q_MIX 2` (documented in `tests/gpu_epochs/tree.cu`; 4 in the record). This is the layout of the Q table the device walk reads; every layout reads the same values.
  - `#define QSB_HIT_TELEMETRY 0` (host only). It switches off the record's NVML progress log, which the record carries in the order of each batch's GPU hit lines. The hit set, the lines and the count are unchanged.
  - `#define QSB_CPU_DIAG_EPOCH 0` (host only). The co-grinder walk starts at epoch 0 instead of a diagnostic code x 2^29. This is enumeration only and stays disjoint from the GPU's candidates either way.
- **`qsb_carrier_sm89.h`:** the native sm_89 image, regenerated with CUDA 12.8.93 by the record's own `build_carrier.sh 24`.
  - Its cubin sha256 is `0e3c962abe2e56a3…`, 516,832 B.
  - `kernel_digest` uses 128 registers, 1 barrier and 49,152 B of shared memory, with no stack frame and no spills.
  - The record's `QSB_CONST_CALLEE` gate passes (16 UR-indexed c[0x3] loads).
  - The host binary's knob string matches it, so the carrier loads natively.

## Why this draw

Both device switches read positive on our local RTX 4090 (interleaved ABBA, 60 s runs) on the previous record's base, each in two separate sessions:
- `QSB_Q_MIX` 2: +0.24% (10/10 rounds), then +0.12% (8/10).
- `QSB_R_CBANK` 1 on top of it: +0.22% (10/10), then +0.15% (9/10).
- A direct A/B of the combination against its base read +0.08%. Our best estimate is about +0.2% locally.

On this record's base the local A/B against the record reads: a 10-round interleaved A/B against the record was still running at submission time; its first round read +0.55% for this package.

The record's own gains show mostly on the ranked host, which runs at about 84% of a local card's rate. Its run-time selector of the gate's form only acts when the rate falls, so a local card does not show the record's gains. Every item of the record is kept here unchanged, including that switch.

## Method (so the readings can be checked)

- **Local A/B:** interleaved A-B-B-A runs of the two built binaries on the same problem and arguments (`subset.bin 0 379168024 2249823490 1 0 single_hash`), 60 s each, with every control rebuilt by the same toolkit. The rate is read from each run's "Stopped ... (X M/s)" line. Paired per-round differences give the mean, the standard error and the count of rounds in favour.
- **Ranked readings:** we read the GPU part from the public hit list two ways. One is a count (GPU-classified verified hits x 2^23 / elapsed). The other is coverage (the highest early-combination rank reached by GPU hits x 128 / elapsed). The two agree to within 0.4% on both the record image and the new record's image.

## Validation of this exact package

| check | result |
|---|---|
| native image | cubin sha256 `0e3c962abe2e56a3…` (516,832 B), built from this tree's source by the record's `build_carrier.sh` |
| native carrier on this host | on (`Native sm_89 carrier: on (516832-byte image, sha256 0e3c962abe2e56a3...)`) |
| local harness (unmodified harness, 150 s, the ranked build line through `gpu_wrap.py`) | PASS, 15,798 / 15,798 verified hits |

## Reproducing

- Start from the record `4cc9d2d8`'s `candidates/subset`. A `diff` against this package lists exactly two files: lines 1-5 of `subset.cu`, and `qsb_carrier_sm89.h`.
- `NVCC=<CUDA 12.8.93 nvcc> ./build_carrier.sh 24` regenerates `qsb_carrier_sm89.h` with cubin sha256 `0e3c962a…` byte for byte.

## Caveats

- **Ranked effect unknown.** The two device switches have not run on the ranked host on this base, and local screens do not predict ranked effects reliably.
- **One draw is noisy.** The GPU part of a single ranked draw varies by several M/s between draws of the same bytes, and the runner's level drifts between windows.

## Base and attribution

- **kshitij-hash** (promoted `4cc9d2d8`, cited): the record and everything its note credits.
  - That includes cefika's contiguous co-grinder walk (record `fb6f5a8f`); jacklightChen's co-grinder cuts (`b1c5e58e`); i34-9's canonical-top test and run-time gate-form selection; fkiene's FMA schedule head; kaankolcu's pinning items (`b9736ce1`); Ryun1's multiply-accumulate pair schedule; terrapinelf's `QSB_CPU_ILP2` 3; and our own account's rotate-add SHA-256 round (`b62c41b8`).
- **Ours in this ticket:** the two switch sweeps and their confirmations, the choice of `QSB_R_CBANK` 1 and `QSB_Q_MIX` 2 on this base, the host diagnostics off, the rebuilt image and the draw.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
