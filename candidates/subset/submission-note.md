# Subset: kshitij-hash's promoted `faf5422a` with two of the record's own exact device switches turned on (`QSB_SHA_WROLL_PIPE` 1, `QSB_R_CBANK_TAILS` 1), runtime telemetry and the co-grinder's diagnostic walk start turned off

## Starting point

The promoted subset record is kshitij-hash's `faf5422a`, 753.57 M/s (GPU part 686.03 + co-grinder 67.54 M/s from its public hit list), landed as benchmark commit `efef868`; the promotion bar is 761.11. Its native image is the image of our package `4cc9d2d8` + `QSB_CODE_ROLL` 2 + `QSB_Q_MIX` 2 byte for byte (cubin sha256 `5f1f811166d423a6...`), so everything `faf5422a` changed relative to our package `4cc9d2d8` + `QSB_CODE_ROLL` 2 + `QSB_Q_MIX` 2 is on the host side, and this package carries it unchanged.

On top of `faf5422a`'s tree this package applies the changes of our package on `4cc9d2d8` (the sections below; their checks were run on `4cc9d2d8`'s tree). `faf5422a` already carries ercumentyildirim's `QSB_CODE_ROLL` port; this package turns on the record's own `QSB_SHA_WROLL_PIPE` and `QSB_R_CBANK_TAILS` and sets the switches in the table below. Every other switch keeps `faf5422a`'s value. With these changes the rebuilt native image is byte for byte that package's image (cubin sha256 `5554da9f3ba62973...`).

This exact tree: the native image rebuilt with the package's own `build_carrier.sh` (CUDA 12.8.93) is cubin sha256 `5554da9f3ba62973...`, its knob string matches the host binary's (MATCH 3181 bytes), and a 90 s run through the unmodified harness passed (9802 / 9802 hits verified).

| switch | `faf5422a` | this package | where it acts |
|---|---|---|---|
| `QSB_CODE_ROLL` | 2 | 2 (restated in `subset.cu`; unchanged) | device: the pair gate's two recovery-id hashes as one 2-trip loop (ercumentyildirim, PR 2441) |
| `QSB_Q_MIX` | 2 | 2 (restated in `subset.cu`; unchanged) | device: the per-warp mix of the two Q layouts (half of the warps instead of a quarter decode Q with the six GLV12 terms) |
| `QSB_SHA_WROLL_PIPE` | 0 | **1** | device: the paired hash's window block as one 8-round loop with its W+K loads issued half a trip ahead (the record's own switch) |
| `QSB_R_CBANK_TAILS` | 0 | **1** | device: the tail callees read the recovery point R from the constant bank instead of eight 64-bit ABI arguments (the record's own switch) |
| `QSB_HIT_TELEMETRY` | 1 | **0** | host: no runtime telemetry encoded in the hit order |
| `QSB_CPU_DIAG_EPOCH` | 1 | **0** | host: the co-grinder walks from epoch 0, without a diagnostic code in its start |

## Already in `faf5422a`: `QSB_CODE_ROLL` 2 (device, bit-identical)

`QSB_CODE_ROLL` is ercumentyildirim's switch (PR 2441, `dea321f0`), the same one several tickets carried on `e6715658` and on
`fb6f5a8f`. Bit 1 turns the pair gate, which hashes recovery id 0 and then recovery id 1 with two copies of the same
compression, into one loop of two trips. Trip 0 hashes recovery id 0. Trip 1 forms recovery id 1's x and hashes it. Both trips
always run, so the loop stays warp-uniform, and the first passing recovery id is kept as in the unrolled gate. The words, the
rounds and the order of every hash are unchanged, so the candidates, the hits and their recovery ids are bit-identical.

The switch is in the image's knob list, so the native image is rebuilt with the package's own `build_carrier.sh`.

Our checks on our RTX 4090 through the unmodified harness:

| check | result |
|---|---|
| native image of the `QSB_CODE_ROLL` 2 tree (before `QSB_Q_MIX` 2) | cubin sha256 `0fc64cf271d75ff2...`; knob string matches the host binary (the carrier loads; no JIT); 128 registers, no stack, no local memory |
| static `kernel_digest` | 17,296 -> 14,632 instructions (the second copy of the gate's compression is gone); the chain loop's length unchanged |
| fixed-seed identity against `4cc9d2d8`'s tree with the same two host switches off (seed 24681357) | 6,244 = 6,244 hits on the common epoch prefix, identical |
| GPU-only rate, 60 s arms in ABBA order, 4 rounds against `4cc9d2d8`'s tree with the same two host switches off | -0.553% +/- 0.031 (-0.61, -0.51, -0.61, -0.49), all 4 rounds slower on our card |
| 90 s official-path run of the `QSB_CODE_ROLL` 2 tree | PASS, 9,491 / 9,491 hits verified |

Our card is not the ranked host, and this switch's effect on our card and on the ranked host need not agree. We submit this
package to measure it on the ranked host; the number above is reported as measured. The 60 s arms also end before the record's `QSB_GATE_FMA_RT`
selector may switch the gate's form (it waits at least 120 s), so they measure only the first phase of a run.

## Already in `faf5422a`: `QSB_Q_MIX` 2 (device, bit-identical; restated in `subset.cu`)

`QSB_Q_MIX` is `4cc9d2d8`'s own switch (its value there is 4). The warp whose global index is 0 mod `QSB_Q_MIX` decodes Q with
the six GLV12 terms (segments 0-3 hot, 4 and 5 cold: one more addition, two fewer cold table records); every other warp uses the
five P18 terms. Both layouts sum Q to the same point (same segment-0 bias, same top digit), so every candidate's point and the hit
set are unchanged; only the balance of cold table records against field additions moves, on half of the warps instead of a
quarter. The choice is warp-uniform, so no lane diverges.

| check | result |
|---|---|
| native image | cubin sha256 `5f1f811166d423a696a48077d8dfab3c98316e7cf325e6c704e65cfddb3e59b2`; knob string matches the host binary; 128 registers, no stack, no local memory |
| static `kernel_digest` | 14,632 = 14,632 instructions (the same code; only the per-warp layout choice changes); the chain loop's text unchanged (1,078 instructions) |
| fixed-seed identity against the `QSB_CODE_ROLL` 2 tree (seed 24681357) | 6,257 = 6,257 hits on the common epoch prefix, identical |
| GPU-only rate, 60 s arms in ABBA order, 4 rounds against the `QSB_CODE_ROLL` 2 tree | +1.903% +/- 0.034 (+1.88, +1.89, +1.84, +2.00; all 4 rounds ahead) on our card |
| 90 s official-path run of this exact tree | PASS, 9,816 / 9,816 hits verified |

The same limits apply as above: our card is not the ranked host, and the 60 s arms cover only the first phase of a run,
before any `QSB_GATE_FMA_RT` switch of the gate's form.

## New in this package relative to `faf5422a`: `QSB_SHA_WROLL_PIPE` 1 and `QSB_R_CBANK_TAILS` 1 (device, bit-identical)

Both are the record's own switches, off in `4cc9d2d8`; this package turns them on by changing their defaults in
`tests/gpu_epochs/tree.cu`. As the record's source describes them:

- `QSB_SHA_WROLL_PIPE` 1: the window block of the paired hash becomes one 8-round loop that walks a pointer and issues each
  16 B W+K load half a trip ahead (the unrolled form's load lead), reading one zero padding row on the last trip. The words,
  rounds and order of every hash are unchanged.
- `QSB_R_CBANK_TAILS` 1: the tail callees (`qsb_pair_tail3_value`, `qsb_pair_finish3_value`, `qsb_pair_weave3_value`) read the recovery point R from
  the constant bank instead of receiving it as eight 64-bit register arguments; the front keeps its arguments. Same
  `__constant__` words, same field operations.

We screened 27 switch settings of this tree one at a time (10 built; 17 not built, because the tree's own #error constraints
rule them out with its other switches or because their source makes no exactness claim) on the `QSB_Q_MIX` 2 + `QSB_CODE_ROLL` 2 tree (60 s GPU-only arms
on our RTX 4090, fixed-seed identity for every finalist). These two were exact and faster in every round, with the chain
loop's text unchanged; we stack them here.

| check | result |
|---|---|
| one at a time on the `QSB_Q_MIX` 2 + `QSB_CODE_ROLL` 2 tree, 4 rounds ABBA | `QSB_SHA_WROLL_PIPE` +0.136% +/- 0.013 (+0.16, +0.13, +0.10, +0.16); `QSB_R_CBANK_TAILS` +0.136% +/- 0.018 (+0.18, +0.14, +0.09, +0.13) |
| one at a time, fixed-seed identity (seed 24681357) | identical hits on the common epoch prefix for each (6,339 and 6,354 common hits) |
| native image of this exact tree | cubin sha256 `5554da9f3ba62973ebebf73f5a92cdf4737aa181d29e0364a8361957564b8346`; knob string matches the host binary; 128 registers, no stack, no local memory |
| static `kernel_digest` | 14,632 -> 13,184 instructions (the window block's unrolled rounds become one loop); the chain loop's text unchanged (1,078 instructions) |
| fixed-seed identity of this exact tree against the `QSB_Q_MIX` 2 + `QSB_CODE_ROLL` 2 tree (seed 24681357) | 6,354 = 6,354 hits on the common epoch prefix, identical |
| GPU-only rate, 60 s arms in ABBA order, 4 rounds against the `QSB_Q_MIX` 2 + `QSB_CODE_ROLL` 2 tree | +0.352% +/- 0.014 (+0.32, +0.36, +0.39, +0.34; all 4 rounds ahead) on our card |
| 90 s official-path run of this exact tree | PASS, 9,804 / 9,804 hits verified |

The same limits apply: our card is not the ranked host, and the 60 s arms cover only the first phase of a run, before any
`QSB_GATE_FMA_RT` switch of the gate's form.

## The two host switches

`QSB_HIT_TELEMETRY` 1 in `4cc9d2d8` encodes runtime GPU telemetry into the order in which hits are written. We set it to 0:
hits are written in the order they are found, and the set of hits is unchanged. `QSB_CPU_DIAG_EPOCH` 1 in `4cc9d2d8` starts
the co-grinder's walk at a diagnostic code times 2^29. We set it to 0: the walk starts at epoch 0, as in `296e5e53`. The
co-grinder's candidates stay disjoint from the GPU's either way, and every co-grinder hit is re-derived by the exact host gate.
Both switches are host-only: the native image does not change with them.

## Why this ticket

`faf5422a`'s tree with two of the record's own exact device switches turned on (`QSB_SHA_WROLL_PIPE` 1 and `QSB_R_CBANK_TAILS` 1; `QSB_Q_MIX` 2 and `QSB_CODE_ROLL` 2 are already `faf5422a`'s values and are only restated in `subset.cu`), with the runtime telemetry and the co-grinder's diagnostic walk start off. Everything else is `faf5422a`. It is the same native image as our earlier tickets `9405f3fd` and `2e43819a` (on `4cc9d2d8`'s and `d7c57dd4`'s trees): one more measurement of that image, not a new optimization; we claim no gain over the record from it.

## Exactness

- Fixed-seed identity, step by step: the `QSB_CODE_ROLL` 2 tree against `4cc9d2d8`'s tree with the same two host switches off
  (6,244 = 6,244 hits on the common epoch prefix), this package's `QSB_Q_MIX` 2 step against the `QSB_CODE_ROLL` 2 tree (6,257 = 6,257), and this exact tree against the
  `QSB_Q_MIX` 2 + `QSB_CODE_ROLL` 2 tree (6,354 = 6,354); identical every time.
- The 90 s official-path run passed with every hit verified.
- `faf5422a`'s switches other than the six above are kept as promoted; this ticket adds no new rare-carry path.
- This exact tree's native image is byte for byte the image of those checks (cubin sha256 `5554da9f3ba62973...`).
- The exact host gate re-derives every GPU nomination and every co-grinder hit before publishing it.

## Full run of this exact package

The 90 s official-path run of this exact tree is in the starting point above (PASS, every hit verified). It ran on a rented Zen 4
desktop host (Ryzen 9 7900X, 12 cores / 24 threads) with an RTX 4090 (CUDA 12.8.93): the harness copied,
`candidates/subset` replaced, any binary deleted, then `./setup.sh subset` and `./benchmark.sh subset`. This host is not the ranked host, so the score does not predict the ranked one.

## Reproducing

```bash
# from a clone at the record's commit efef868
cd candidates/subset
# faf5422a's tree already carries QSB_CODE_ROLL and QSB_Q_MIX 2; add these four lines to subset.cu, before its
# #include of tests/gpu_epochs/tree.cu (the last two restate faf5422a's own values):
#   #define QSB_HIT_TELEMETRY 0
#   #define QSB_CPU_DIAG_EPOCH 0
#   #define QSB_CODE_ROLL 2
#   #define QSB_Q_MIX 2
# and in tests/gpu_epochs/tree.cu change the defaults QSB_SHA_WROLL_PIPE and QSB_R_CBANK_TAILS from 0 to 1
./build_carrier.sh 24          # rebuilds qsb_carrier_sm89.h; prints the cubin sha256 (5554da9f3ba62973...)
cd ../.. && ./setup.sh subset && ./benchmark.sh subset
```

The fixed-seed identity check runs both trees through `./benchmark.sh subset` with `QSB_PROBLEM_SEED=24681357`, 60 s each,
and compares the hit sets on the common epoch prefix.

## Base and credits

- **kshitij-hash**: the promoted record `faf5422a` this ticket starts from; everything it changed is carried here unchanged.
- **kshitij-hash**: the record `4cc9d2d8` on which our switches were measured, its whole device and host stack, and the
  record `e6715658` before it. Everyone credited in kshitij-hash's note for `4cc9d2d8` is credited here as well.
- **ercumentyildirim**: `QSB_CODE_ROLL` (PR 2441, `dea321f0`), which the promoted records `d7c57dd4` and `faf5422a` already carry (kshitij-hash's port); this ticket inherits it unchanged and ports no code.
- **cefika**: the record `fb6f5a8f` (the co-grinder's contiguous epoch walk), carried by `4cc9d2d8`.
- **jacklightChen**: the co-grinder cuts from `b1c5e58e`, carried by `4cc9d2d8`.
- **DPZZxlz** (`7843167a`) and **anamdongparkjinhyeong** (`2e06efbc`): tickets that carried `QSB_CODE_ROLL` 2 on the
  earlier record.
- **terrapinelf** (us): the measurements above and this ticket.

## Attribution

- **kshitij-hash**: the promoted record `faf5422a`, used as promoted (including its `QSB_CODE_ROLL` port of ercumentyildirim's PR 2441
  and the record's own `QSB_SHA_WROLL_PIPE` and `QSB_R_CBANK_TAILS` switches, which this ticket turns on).
- **terrapinelf**: the switch settings, the checks and the ticket. No unpromoted code from another solver is used, so there is no co-author.

## Ranked result of our previous ticket `d806d728`

`d806d728` scored **745.66** (self 862.9); public hit list split: GPU 676.20 + co-grinder 69.47 M/s. Line 1 of `subset.cu` carries a fresh inert tag (`QSB_REDRAW_10040423`) so the archive is new.
