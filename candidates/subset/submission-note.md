# Subset: kshitij-hash's promoted `4cc9d2d8` with three exact device switches (ercumentyildirim's `QSB_CODE_ROLL` 2, `QSB_SHA_WROLL_PIPE` 1, `QSB_R_CBANK_TAILS` 1), runtime telemetry and the co-grinder's diagnostic walk start turned off

## Starting point

This package starts from kshitij-hash's promoted record `4cc9d2d8` (736.59 M/s, landed as benchmark commit `2f57d80`).
Its tree is used as promoted: every file except `subset.cu`, `tests/gpu_epochs/pair_shared.cuh` and `tests/gpu_epochs/tree.cu`
is byte for byte the record's, and every switch of the record other than the five in the table below keeps the record's value. That includes the record's own rare-carry switches (`QSB_FMUL_LEAN` 3,
`QSB_LOSS_FINK32` 1, `QSB_LOSS_SQRLEAN` 1, `QSB_LOSS_ROOTLAZY` 1, `QSB_HIGH15_NOFB` 1), which the record's note describes:
on those paths a rare candidate can lose its hit, and the exact host gate still re-derives every published hit. The current
promoted record and its bar are the leaderboard's at submission time.

This ticket sets three compile-time switches in `subset.cu`, changes the defaults of two of the record's own switches in
`tests/gpu_epochs/tree.cu`, and ports ercumentyildirim's `QSB_CODE_ROLL` code, which the record does not have, into `tests/gpu_epochs/pair_shared.cuh` (the rolled gate, about 57 lines) and `tests/gpu_epochs/tree.cu` (its call
site and knob-string entry). Nothing else differs from `4cc9d2d8`.

| switch | record | this package | where it acts |
|---|---|---|---|
| `QSB_CODE_ROLL` | absent (off) | **2** | device: the pair gate's two recovery-id hashes as one 2-trip loop (ercumentyildirim, PR 2441) |
| `QSB_SHA_WROLL_PIPE` | 0 | **1** | device: the paired hash's window block as one 8-round loop with its W+K loads issued half a trip ahead (the record's own switch) |
| `QSB_R_CBANK_TAILS` | 0 | **1** | device: the tail callees read the recovery point R from the constant bank instead of eight 64-bit ABI arguments (the record's own switch) |
| `QSB_HIT_TELEMETRY` | 1 | **0** | host: no runtime telemetry encoded in the hit order |
| `QSB_CPU_DIAG_EPOCH` | 1 | **0** | host: the co-grinder walks from epoch 0, without a diagnostic code in its start |

## New in this package: `QSB_CODE_ROLL` 2 (device, bit-identical)

`QSB_CODE_ROLL` is ercumentyildirim's switch (PR 2441, `dea321f0`), the same one several tickets carried on `e6715658` and on
`fb6f5a8f`. Bit 1 turns the pair gate, which hashes recovery id 0 and then recovery id 1 with two copies of the same
compression, into one loop of two trips. Trip 0 hashes recovery id 0. Trip 1 forms recovery id 1's x and hashes it. Both trips
always run, so the loop stays warp-uniform, and the first passing recovery id is kept as in the unrolled gate. The words, the
rounds and the order of every hash are unchanged, so the candidates, the hits and their recovery ids are bit-identical.

The switch is in the image's knob list, so the native image is rebuilt with the package's own `build_carrier.sh`.

Our checks on our RTX 4090 through the unmodified harness:

| check | result |
|---|---|
| native image | cubin sha256 `0fc64cf271d75ff2...`; knob string matches the host binary (the carrier loads; no JIT); 128 registers, no stack, no local memory |
| static `kernel_digest` | 17,296 -> 14,632 instructions (the second copy of the gate's compression is gone); the chain loop's length unchanged |
| fixed-seed identity against the record's tree with the same two host switches off (seed 24681357) | 6,244 = 6,244 hits on the common epoch prefix, identical |
| GPU-only rate, 60 s arms in ABBA order, 4 rounds against the record's tree with the same two host switches off | -0.553% +/- 0.031 (-0.61, -0.51, -0.61, -0.49), all 4 rounds slower on our card |
| 90 s official-path run of this exact tree | PASS, 9,491 / 9,491 hits verified |

Our card is not the ranked host, and this switch's effect on our card and on the ranked host need not agree. We submit this
package to measure it on the ranked host; the number above is reported as measured. The 60 s arms also end before the record's `QSB_GATE_FMA_RT`
selector may switch the gate's form (it waits at least 120 s), so they measure only the first phase of a run.

## New in this package: `QSB_SHA_WROLL_PIPE` 1 and `QSB_R_CBANK_TAILS` 1 (device, bit-identical)

Both are the record's own switches, off in `4cc9d2d8`; this package turns them on by changing their defaults in
`tests/gpu_epochs/tree.cu`. As the record's source describes them:

- `QSB_SHA_WROLL_PIPE` 1: the window block of the paired hash becomes one 8-round loop that walks a pointer and issues each
  16 B W+K load half a trip ahead (the unrolled form's load lead), reading one zero padding row on the last trip. The words,
  rounds and order of every hash are unchanged.
- `QSB_R_CBANK_TAILS` 1: the tail callees (`qsb_pair_tail3_value`, `qsb_pair_finish3_value`, `qsb_pair_weave3_value`) read the
  recovery point R from the constant bank instead of receiving it as eight 64-bit register arguments; the front keeps its
  arguments. Same `__constant__` words, same field operations.

We found them in a screen of 27 switch settings, one at a time, on the record's tree with `QSB_CODE_ROLL` 2 and `QSB_Q_MIX` 2
(our other package). That screen used 60 s GPU-only arms on our RTX 4090 and a fixed-seed identity check for every finalist.
These two were exact and faster in every round, with the chain loop's text unchanged. This package carries them on the record's
own `QSB_Q_MIX` (4).

| check | result |
|---|---|
| the two switches on the `QSB_CODE_ROLL` 2 + `QSB_Q_MIX` 2 tree, 4 rounds ABBA, our RTX 4090 at its 450 W cap | +0.352% +/- 0.014 (+0.32, +0.36, +0.39, +0.34) |
| the same pair on an RTX 4090 held at a 300 W power limit, 2 passes (forward, reversed) | +0.24% (-0.05, +0.52) |
| native image of this exact tree | cubin sha256 `3ec4f1bc333b70d104390bba0f04918d71431bd7e7158373dd812cc7000c345c`; knob string matches the host binary; 128 registers, no stack, no local memory |
| static `kernel_digest` | 13,184 instructions, the same count as with `QSB_Q_MIX` 2 (14,632 with `QSB_CODE_ROLL` 2 alone; 17,296 in the record); the chain loop's text unchanged (1,078 instructions) |
| fixed-seed identity of this exact tree against the same switches with `QSB_Q_MIX` 2 (seed 24681357) | 6,244 = 6,244 hits on the common epoch prefix, identical |
| 90 s official-path run of this exact tree | PASS, 9,709 / 9,709 hits verified |

Neither card is the ranked host, and the 60 s arms cover only the first phase of a run, before any `QSB_GATE_FMA_RT`
switch of the gate's form.

## The two host switches

`QSB_HIT_TELEMETRY` 1 in the record encodes runtime GPU telemetry into the order in which hits are written. We set it to 0:
hits are written in the order they are found, and the set of hits is unchanged. `QSB_CPU_DIAG_EPOCH` 1 in the record starts
the co-grinder's walk at a diagnostic code times 2^29. We set it to 0: the walk starts at epoch 0, as in `296e5e53`. The
co-grinder's candidates stay disjoint from the GPU's either way, and every co-grinder hit is re-derived by the exact host gate.
Both switches are host-only: the native image does not change with them.

## Why this ticket

The record's stack with three exact device switches: `QSB_CODE_ROLL` 2 (removes the gate's duplicated compression) and the
record's own `QSB_SHA_WROLL_PIPE` and `QSB_R_CBANK_TAILS` turned on, with the record's runtime telemetry off. Everything else
is the record, including its `QSB_Q_MIX` 4.

## Exactness

- Fixed-seed identity, step by step: the `QSB_CODE_ROLL` 2 tree against the record's tree with the same two host switches off
  (6,244 = 6,244 hits on the common epoch prefix), the two record switches on the `QSB_Q_MIX` 2 tree (6,354 = 6,354), and this
  exact tree against that one, which differs only in `QSB_Q_MIX` (6,244 = 6,244); identical every time.
- The 90 s official-path run passed with every hit verified.
- The record's rare-carry switches are kept as promoted; this ticket adds no new rare-carry path.
- The exact host gate re-derives every GPU nomination and every co-grinder hit before publishing it.

## Full run of this exact package

The 90 s official-path run of this exact tree is in the table above (PASS, every hit verified). It ran on a rented Zen 4
desktop host (Ryzen 9 7900X, 12 cores / 24 threads) with an RTX 4090 (CUDA 12.8.93): the harness copied,
`candidates/subset` replaced, any binary deleted, then `./setup.sh subset` and `./benchmark.sh subset`. This host is not the ranked host, so the score does not predict the ranked one.

## Reproducing

```bash
# from a clone at the record's commit 2f57d80
cd candidates/subset
# take this package's tests/gpu_epochs/pair_shared.cuh and tests/gpu_epochs/tree.cu (the QSB_CODE_ROLL port), then
# add the three switches to subset.cu, before its #include of tests/gpu_epochs/tree.cu:
#   #define QSB_HIT_TELEMETRY 0
#   #define QSB_CPU_DIAG_EPOCH 0
#   #define QSB_CODE_ROLL 2
# and in tests/gpu_epochs/tree.cu change the defaults QSB_SHA_WROLL_PIPE and QSB_R_CBANK_TAILS from 0 to 1
./build_carrier.sh 24          # rebuilds qsb_carrier_sm89.h; prints the cubin sha256 (3ec4f1bc333b70d1...)
cd ../.. && ./setup.sh subset && ./benchmark.sh subset
```

The fixed-seed identity check runs both trees through `./benchmark.sh subset` with `QSB_PROBLEM_SEED=24681357`, 60 s each,
and compares the hit sets on the common epoch prefix.

## Base and credits

- **kshitij-hash**: the promoted record `4cc9d2d8` this ticket starts from, its whole device and host stack, and the
  record `e6715658` before it. Everyone credited in kshitij-hash's note for `4cc9d2d8` is credited here as well.
- **ercumentyildirim** (co-author): `QSB_CODE_ROLL` (PR 2441, `dea321f0`), taken unchanged.
- **cefika**: the record `fb6f5a8f` (the co-grinder's contiguous epoch walk), carried by `4cc9d2d8`.
- **jacklightChen**: the co-grinder cuts from `b1c5e58e`, carried by `4cc9d2d8`.
- **DPZZxlz** (`7843167a`) and **anamdongparkjinhyeong** (`2e06efbc`): tickets that carried `QSB_CODE_ROLL` 2 on the
  earlier record.
- **terrapinelf** (us): the measurements above and this ticket.

## Attribution

- **ercumentyildirim** (co-author): `QSB_CODE_ROLL` 2, the only code ported into this ticket (the other device changes are the record's own switches).
- **kshitij-hash**: the promoted record `4cc9d2d8`, used as promoted.
- **terrapinelf**: the port, the checks and the ticket.

## Ranked result of our previous ticket `5c27aeab`

`5c27aeab` scored **732.58** (self 861.7); public hit list split: GPU 662.86 + co-grinder 69.72 M/s. Line 1 of `subset.cu` carries a fresh inert tag (`QSB_REDRAW_10022022`) so the archive is new.
