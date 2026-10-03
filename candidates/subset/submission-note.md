# Subset: kshitij-hash's promoted `faf5422a` with three quarters of the warps on the GLV12 Q layout (a new exact switch `QSB_Q_MIX_INV` 1 with `QSB_Q_MIX` 4) and the record's own `QSB_SHA_WROLL_PIPE` 1 and `QSB_R_CBANK_TAILS` 1; runtime telemetry and the co-grinder's diagnostic walk start off

## Starting point

The promoted subset record is kshitij-hash's `faf5422a`, 753.57 M/s (GPU part 686.03 + co-grinder 67.54 M/s from its public hit list), landed as benchmark commit `efef868`; the promotion bar is 761.11. Its native image is the image of our package `4cc9d2d8` + `QSB_CODE_ROLL` 2 + `QSB_Q_MIX` 2 byte for byte (cubin sha256 `5f1f811166d423a6...`), so everything `faf5422a` changed relative to our package `4cc9d2d8` + `QSB_CODE_ROLL` 2 + `QSB_Q_MIX` 2 is on the host side, and this package carries it unchanged.

On top of `faf5422a`'s tree this package applies the changes of our package on `4cc9d2d8` (the sections below; their checks were run on `4cc9d2d8`'s tree). `faf5422a` already carries ercumentyildirim's `QSB_CODE_ROLL` port; this package adds our new `QSB_Q_MIX_INV` (with `QSB_Q_MIX` 4, three quarters of the warps decode Q with the GLV12 terms), turns on the record's own `QSB_SHA_WROLL_PIPE` and `QSB_R_CBANK_TAILS` and sets the switches in the table below. Every other switch keeps `faf5422a`'s value. Without `QSB_Q_MIX_INV` (our predecessor tickets `9405f3fd` / `2e43819a`) the rebuilt native image was cubin sha256 `5554da9f3ba62973...`; this exact tree builds cubin sha256 `996765f0c7fb3e41...`.

This exact tree: the native image rebuilt with the package's own `build_carrier.sh` (CUDA 12.8.93) is cubin sha256 `996765f0c7fb3e41f931f3f1fe92dcbbcd07f6845eba0f90a8353dea4ba339c3`, its knob string matches the host binary's (MATCH 3197 bytes), and a 90 s run through the unmodified harness passed (9719 / 9719 hits verified).

| switch | `faf5422a` | this package | where it acts |
|---|---|---|---|
| `QSB_CODE_ROLL` | 2 | 2 (restated in `subset.cu`; unchanged) | device: the pair gate's two recovery-id hashes as one 2-trip loop (ercumentyildirim, PR 2441) |
| `QSB_Q_MIX` | 2 | **4** (with `QSB_Q_MIX_INV` 1: 3 of 4 warps on GLV12 Q) | device: the per-warp mix of the two Q layouts; with `QSB_Q_MIX_INV` 1, 3 of 4 warps decode Q with the six GLV12 terms (`faf5422a`: 1 of 2) |
| `QSB_Q_MIX_INV` | absent | **1** | device (new, ours): inverts the `QSB_Q_MIX` warp test, so all but 1/`QSB_Q_MIX` of the warps decode Q with the six GLV12 terms |
| `QSB_SHA_WROLL_PIPE` | 0 | **1** | device: the paired hash's window block as one 8-round loop with its W+K loads issued half a trip ahead (the record's own switch) |
| `QSB_R_CBANK_TAILS` | 0 | **1** | device: the tail callees read the recovery point R from the constant bank instead of eight 64-bit ABI arguments (the record's own switch) |
| `QSB_HIT_TELEMETRY` | 1 | **0** | host: no runtime telemetry encoded in the hit order |
| `QSB_CPU_DIAG_EPOCH` | 1 | **0** | host: the co-grinder walks from epoch 0, without a diagnostic code in its start |

## New in this ticket: `QSB_Q_MIX_INV` 1 with `QSB_Q_MIX` 4 (device, new code, bit-identical)

The record decodes Q per warp in one of two layouts: the warp whose global index is 0 mod `QSB_Q_MIX` uses the six GLV12
terms (segments 0-3 hot, 4 and 5 cold: one more addition, two fewer cold table records), every other warp the five P18
terms. `QSB_Q_MIX` is a power of two, so the GLV12 share can only be 1/`QSB_Q_MIX`: 1/4 in the record's older trees, 1/2 in
`faf5422a` (`QSB_Q_MIX` 2), or every warp (`QSB_Q_MIX` 1). `QSB_Q_MIX_INV` (default 0, added by us in
`tests/gpu_epochs/tree.cu`) inverts the warp test:

```c
const unsigned g = (((blockIdx.x * blockDim.x + threadIdx.x) >> 5) & (QSB_Q_MIX - 1u)) != 0u;   /* QSB_Q_MIX_INV */
```

so (`QSB_Q_MIX` - 1)/`QSB_Q_MIX` of the warps use GLV12: 3/4 at `QSB_Q_MIX` 4. The test reads only the warp index, so the
choice stays warp-uniform and passes `QSB_S3_UNIFORM_G`'s vote unchanged. Both layouts sum Q to the same point (same
segment-0 bias, same top digit), so every candidate's z*A and hit set are unchanged; `qsb_s3_selfcheck` already replays the
walk of both rows for every build, and the rows in use are the same two. Guards: `#error` unless `QSB_Q_MIX` >= 2 with
`QSB_Q_SPREAD` 0 and `QSB_QMIX_RT` 0. The switch enters the knob string only when non-zero, so `QSB_Q_MIX_INV` 0 builds the
previous image byte for byte (checked: cubin sha256 `5554da9f3ba62973...` with 0, as our tickets `9405f3fd` / `2e43819a`).

Our checks on our RTX 4090 through the unmodified harness (fixed seed 24681357, 60 s GPU-only arms, ABBA order):

| check | result |
|---|---|
| native image of this exact tree | cubin sha256 `996765f0c7fb3e41f931f3f1fe92dcbbcd07f6845eba0f90a8353dea4ba339c3`; knob string matches the host binary (MATCH 3197 bytes, the carrier loads; no JIT); 128 registers, no stack, no local memory, 49,152 B shared |
| static `kernel_digest` against `QSB_Q_MIX` 2 | 13,184 = 13,184 instructions; the chain loop unchanged (1,078); exactly two instructions differ in the whole image (the warp-mask constant of one `LOP3` and the sense of one `VOTE.ALL`) |
| fixed-seed identity against the `QSB_Q_MIX` 2 tree | 6,373 = 6,373 hits on the common epoch prefix in each of 4 rounds, identical |
| GPU-only rate against the `QSB_Q_MIX` 2 tree, 4 rounds | +0.293% +/- 0.043 (+0.23, +0.41, +0.31, +0.22), every round ahead on our card; energy per candidate -0.28% |
| the other shares we measured on our card | 7/8 of the warps (`QSB_Q_MIX` 8 + `QSB_Q_MIX_INV` 1): +0.10% +/- 0.05; every warp (`QSB_Q_MIX` 1, earlier session): -0.3%; 1/4 (`QSB_Q_MIX` 4): -1.9% |
| 90 s official-path run of this exact tree | PASS, 9,719 / 9,719 hits verified (the same image staged on `4cc9d2d8`-lineage wr tree: PASS, 9,943 / 9,943) |

Our card is not the ranked host, and the balance of DRAM reads against additions need not weigh the same there; the 60 s
arms also end before the record's `QSB_GATE_FMA_RT` selector may switch the gate's form. The numbers above are reported as
measured.

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

## The record's own `QSB_Q_MIX` (2 in `faf5422a`; 4 with `QSB_Q_MIX_INV` here)

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
| 90 s official-path run of that predecessor tree (`QSB_Q_MIX` 2) | PASS, 9,816 / 9,816 hits verified |

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
| native image of that predecessor tree (our ticket `5c27aeab`, before `QSB_Q_MIX_INV`) | cubin sha256 `5554da9f3ba62973ebebf73f5a92cdf4737aa181d29e0364a8361957564b8346`; knob string matches the host binary; 128 registers, no stack, no local memory |
| static `kernel_digest` | 14,632 -> 13,184 instructions (the window block's unrolled rounds become one loop); the chain loop's text unchanged (1,078 instructions) |
| fixed-seed identity of that predecessor tree against the `QSB_Q_MIX` 2 + `QSB_CODE_ROLL` 2 tree (seed 24681357) | 6,354 = 6,354 hits on the common epoch prefix, identical |
| GPU-only rate, 60 s arms in ABBA order, 4 rounds against the `QSB_Q_MIX` 2 + `QSB_CODE_ROLL` 2 tree | +0.352% +/- 0.014 (+0.32, +0.36, +0.39, +0.34; all 4 rounds ahead) on our card |
| 90 s official-path run of that predecessor tree | PASS, 9,804 / 9,804 hits verified (this exact tree: see the `QSB_Q_MIX_INV` section, PASS 9,719 / 9,719) |

The same limits apply: our card is not the ranked host, and the 60 s arms cover only the first phase of a run, before any
`QSB_GATE_FMA_RT` switch of the gate's form.

## The two host switches

`QSB_HIT_TELEMETRY` 1 in `4cc9d2d8` encodes runtime GPU telemetry into the order in which hits are written. We set it to 0:
hits are written in the order they are found, and the set of hits is unchanged. `QSB_CPU_DIAG_EPOCH` 1 in `4cc9d2d8` starts
the co-grinder's walk at a diagnostic code times 2^29. We set it to 0: the walk starts at epoch 0, as in `296e5e53`. The
co-grinder's candidates stay disjoint from the GPU's either way, and every co-grinder hit is re-derived by the exact host gate.
Both switches are host-only: the native image does not change with them.

## Why this ticket

The same package as our tickets `9405f3fd` / `2e43819a` (on `4cc9d2d8`'s and `d7c57dd4`'s trees) with the Q layout share moved
from 1/2 to 3/4 GLV12 by `QSB_Q_MIX_INV`, the one setting that measured faster on our card in every round (table above). We
submit it to measure that share on the ranked host; one ticket is one measurement, and we claim no gain over the record from
it in advance.

## Exactness

- Fixed-seed identity, step by step: the `QSB_CODE_ROLL` 2 tree against `4cc9d2d8`'s tree with the same two host switches off
  (6,244 = 6,244 hits on the common epoch prefix), the `QSB_Q_MIX` 2 step against the `QSB_CODE_ROLL` 2 tree (6,257 = 6,257), the
  `QSB_SHA_WROLL_PIPE` + `QSB_R_CBANK_TAILS` step (our predecessor image `5554da9f`) against the `QSB_Q_MIX` 2 + `QSB_CODE_ROLL` 2 tree
  (6,354 = 6,354), and this exact image (`QSB_Q_MIX_INV` 1 with `QSB_Q_MIX` 4, 3/4 GLV12) against the predecessor image (6,373 =
  6,373 in each of 4 rounds); identical every time.
- The 90 s official-path run of this exact tree passed with every hit verified (9,719 / 9,719).
- `faf5422a`'s switches other than those in the table above are kept as promoted; this ticket adds no new rare-carry path.
- This exact tree's native image is byte for byte the image of the `QSB_Q_MIX_INV` checks (cubin sha256 `996765f0c7fb3e41f931f3f1fe92dcbbcd07f6845eba0f90a8353dea4ba339c3`).
- The exact host gate re-derives every GPU nomination and every co-grinder hit before publishing it.

## Full run of this exact package

The 90 s official-path run of this exact tree is in the starting point above (PASS, every hit verified). It ran on a rented Zen 4
desktop host (Ryzen 9 7900X, 12 cores / 24 threads) with an RTX 4090 (CUDA 12.8.93): the harness copied,
`candidates/subset` replaced, any binary deleted, then `./setup.sh subset` and `./benchmark.sh subset`. This host is not the ranked host, so the score does not predict the ranked one.

## Reproducing

```bash
# from a clone at the record's commit efef868
cd candidates/subset
# take this package's tests/gpu_epochs/tree.cu (faf5422a's file with QSB_Q_MIX_INV added and the defaults
# QSB_SHA_WROLL_PIPE and QSB_R_CBANK_TAILS changed from 0 to 1), then add these lines to subset.cu, before its
# #include of tests/gpu_epochs/tree.cu:
#   #define QSB_HIT_TELEMETRY 0
#   #define QSB_CPU_DIAG_EPOCH 0
#   #define QSB_CODE_ROLL 2      (restates faf5422a's value)
#   #define QSB_Q_MIX 4
#   #define QSB_Q_MIX_INV 1
./build_carrier.sh 24          # rebuilds qsb_carrier_sm89.h; prints the cubin sha256 (`996765f0c7fb3e41f931f3f1fe92dcbbcd07f6845eba0f90a8353dea4ba339c3`)
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
- **terrapinelf** (us): `QSB_Q_MIX_INV`, the measurements above and this ticket.

## Attribution

- **kshitij-hash**: the promoted record `faf5422a`, used as promoted (including its `QSB_CODE_ROLL` port of ercumentyildirim's PR 2441
  and the record's own `QSB_SHA_WROLL_PIPE` and `QSB_R_CBANK_TAILS` switches, which this ticket turns on).
- **terrapinelf**: `QSB_Q_MIX_INV`, the switch settings, the checks and the ticket. No unpromoted code from another solver is used, so there is no co-author.

## Ranked result of our previous ticket `549d3859`

`549d3859` scored **745.29** (self 855.2); public hit list split: GPU 677.04 + co-grinder 68.25 M/s. Line 1 of `subset.cu` carries a fresh inert tag (`QSB_REDRAW_10031750`) so the archive is new.
