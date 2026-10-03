# Subset: kshitij-hash's promoted `4cc9d2d8` with a run-time chain-layout switch (a new exact switch `QSB_LAYOUT_RT` 1: our `QSB_Q_MIX` 2 layout until the record's own gate-form rate rule fires, then every GLV term on the GLV12 layout), plus ercumentyildirim's `QSB_CODE_ROLL` 2, `QSB_SHA_WROLL_PIPE` 1, `QSB_R_CBANK_TAILS` 1; runtime telemetry and the co-grinder's diagnostic walk start off

## Starting point

This package starts from kshitij-hash's promoted record `4cc9d2d8` (736.59 M/s, landed as benchmark commit `2f57d80`).
Its tree is used as promoted: every file except `subset.cu`, `tests/gpu_epochs/pair_shared.cuh` and `tests/gpu_epochs/tree.cu`
is byte for byte the record's, and every switch of the record other than the ones in the table below keeps the record's value.
That includes the record's own rare-carry switches (`QSB_FMUL_LEAN` 3, `QSB_LOSS_FINK32` 1, `QSB_LOSS_SQRLEAN` 1,
`QSB_LOSS_ROOTLAZY` 1, `QSB_HIGH15_NOFB` 1), which the record's note describes: on those paths a rare candidate can lose its
hit, and the exact host gate still re-derives every published hit. The current promoted record and its bar are the
leaderboard's at submission time.

It is the package of our tickets `9405f3fd` (the `QSB_Q_MIX` 2 layout for the whole run) and `02df1bed` / `e6cf8eea` (every GLV
term on the GLV12 layout for the whole run), with one new switch that runs the first layout until the record's
`QSB_GATE_FMA_RT` rate rule fires and the second from the next launch on.

| switch | record | this package | where it acts |
|---|---|---|---|
| `QSB_CODE_ROLL` | absent (off) | **2** | device: the pair gate's two recovery-id hashes as one 2-trip loop (ercumentyildirim, PR 2441) |
| `QSB_Q_MIX` | 4 | **2** | device: the warp whose global index is even decodes Q with the six GLV12 terms (the start layout) |
| `QSB_P_MIX` | absent | **1** | device (ours, from our tickets `0110ce71` / `02df1bed`): the GLV12 P tail rows in the walk table |
| `QSB_LAYOUT_RT` | absent | **1** | device and host (new, ours): the layout is a per-launch argument; the start layout until the rate rule fires, then every warp on the GLV12 Q and P terms |
| `QSB_SHA_WROLL_PIPE` | 0 | **1** | device: the paired hash's window block as one 8-round loop with its W+K loads issued half a trip ahead (the record's own switch) |
| `QSB_R_CBANK_TAILS` | 0 | **1** | device: the tail callees read the recovery point R from the constant bank instead of eight 64-bit ABI arguments (the record's own switch) |
| `QSB_HIT_TELEMETRY` | 1 | **0** | host: no runtime telemetry encoded in the hit order |
| `QSB_CPU_DIAG_EPOCH` | 1 | **0** | host: the co-grinder walks from epoch 0, without a diagnostic code in its start |

## New in this ticket: `QSB_LAYOUT_RT` 1 (device and host, new code, bit-identical)

### What it does

`QSB_Q_MIX` (the record's switch) and `QSB_P_MIX` (ours) choose, per warp, how the chain decodes Q and P: with the five P18
terms (segment 0, cold segments 6, 7, 4, 5) or with the six GLV12 terms (segment 0, hot segments 1-3, cold segments 4 and 5:
one more addition, two fewer cold table records). Both layouts sum every candidate to the same point (same segment-0 bias, same
top digit; any run of signed segments that tiles bits 18..99 telescopes to the same constant), so the hit set does not depend
on the layout. Per candidate:

| layout | cold table records | field additions | our ticket with it for the whole run |
|---|---|---|---|
| record (`QSB_Q_MIX` 4, P18 P) | 7.5 | 9.25 | `4cc9d2d8` |
| start layout of this ticket (`QSB_Q_MIX` 2, P18 P) | 7 | 9.5 | `9405f3fd` |
| after the switch (`QSB_Q_MIX` 1 + `QSB_P_MIX` 1) | 4 | 11 | `02df1bed`, `e6cf8eea` |

In those tickets the layout is fixed at compile time: the P tail's first row and the loop bound are immediates, and the Q mask
is `QSB_Q_MIX - 1`. `QSB_LAYOUT_RT` 1 makes all three a function of one `unsigned` selector that `kernel_digest` receives by
value and hands down (`QSB_LAY_PARAM` / `QSB_LAY_PASS`) through `qsb_pair_front*_value` and `qsb_k2s_front*` to
`qsb_filter_chain_trial`:

- selector 0: psi target = the P18 tail (row 11), loop bound 15, Q mask `QSB_Q_MIX - 1` (= 1);
- selector 1: psi target = the GLV12 P tail (row 16), loop bound 21, Q mask 0 (every warp on the GLV12 Q terms).

The stops are in the loop's own units (descriptor bytes under `QSB_S3_DOFF`). `main()` passes 0 to every launch until the
record's `QSB_GATE_FMA_RT` rule fires (the same rate samples, the same `AFTER_S` 120 and `RATIO_PCT` 90 by default: the
trailing 60 s GPU rate at or below 90% of the first minute's, no earlier than 120 s), and 1 to every launch from the next
one on. The change is one host variable; there is no device write.

### Why a launch argument and not a `__constant__` write

The slot pipeline's streams are `cudaStreamNonBlocking`, so a `cudaMemcpyToSymbol` on the legacy stream is not ordered
against launches already in flight. For the record's run-time gate-form choice that is harmless (any mix of the two gate
forms is exact). For a layout it is not: a launch that read the new psi target with the old bound (row 16 with bound 15)
would skip the P terms, and the old target with the new bound (row 11 with bound 21) would walk into row 16, a second
psi entry. A launch's arguments are fixed when it is enqueued, so with the selector as an argument every candidate of a
launch walks one layout by construction. Launches enqueued before the switch keep the start layout and may still run beside
the first ones after it; both are exact.

### What it costs in the chain

We compiled four encodings of the selection (same 128 registers, no stack, no spill for all four):

| encoding | chain-loop instructions (static) |
|---|---|
| immediates (`QSB_LAYOUT_RT` 0; our tickets) | 1,078 |
| one `unsigned` selector by value (this ticket) | 1,080 |
| `{target, bound, mask}` struct by value, or three scalars by value | 1,093 |
| `__constant__` struct (would also need an ordered write) | 1,088 |

With the selector, the target and the bound are computed into uniform registers before the loop (the same count as the
immediates); the two extra loop instructions are uniform-datapath reloads.

### Checks

`qsb_s3_selfcheck` pins what the chain derives from each selector (selector 0: P18 tail stops and the `QSB_Q_MIX` mask;
selector 1: GLV12 tail stops and mask 0) and replays the ZDEC walks of both layouts against the table codes. A host-only unit
test of the self-check (built with the harness's compile line) returns 1 for the shipped values and 0 for each of four
deliberately wrong ones (target 16 with bound 15 for selector 0, bound 20 for selector 1, mask 1 for selector 1, and a wrong
bound with the pinning removed, which the ZDEC replay alone catches).

At `QSB_LAYOUT_RT` 0 the tree compiles to the same native images as our earlier tickets, byte for byte: `QSB_Q_MIX` 2 /
`QSB_P_MIX` 0 to `5554da9f...` (`9405f3fd`), `QSB_Q_MIX` 2 / `QSB_P_MIX` 1 to `11c1c543...` (`0110ce71`), `QSB_Q_MIX` 1 /
`QSB_P_MIX` 1 to `51a6fca5...` (`02df1bed`). `QSB_LAYOUT_RT` enters the knob string only when non-zero; its trigger macros
are host-only.

Our checks on our RTX 4090 through the unmodified harness (60 s arms; the rule waits at least 120 s, so the production
build runs the start layout for a whole arm; host-only test builds of the same image fix the switch time instead, or make
the production rule fire at its first check):

| check | result |
|---|---|
| native image of this exact tree | cubin sha256 `24f6539c88dcf80084e50706ec7cdc0740e78b388b5eb200dfc57807e9117a46`; knob string matches the host binary (the carrier loads; no JIT); 128 registers, no stack, no local memory, 49,152 B shared |
| static `kernel_digest` | 13,192 instructions (13,184 for `9405f3fd`'s image, 13,176 for `02df1bed`'s); chain loop 1,080 (1,078) |
| fixed-seed identity, never switching, against `9405f3fd`'s tree (seed 24681357) | 6,373 = 6,373 hits on the common epoch prefix in both rounds (6,416 = 6,416 in total), identical |
| fixed-seed identity, switching at 1 s (live boundary: two start-layout launches still in flight), against `9405f3fd`'s tree | 6,203 = 6,203 hits on the common epoch prefix in both rounds, identical (the switch came at batch 8 and batch 9, at 1.013 s and 1.051 s, with one start-layout launch still in flight on each of the two slot streams); against `02df1bed`'s tree also 6,203 = 6,203, identical |
| GPU-only rate, 60 s arms, never switching, against `9405f3fd`'s tree | +0.017% +/- 0.028 (-0.01, +0.04): no measurable cost of the selector on our card |
| GPU-only rate, 60 s arms, switching at 1 s, against `02df1bed`'s tree | +0.23% (881.80 against 879.80 M/s; against `9405f3fd`'s tree the two are -2.72% and -2.94%), i.e. no cost in this layout either |
| 90 s official-path runs of the same image: production trigger / never / switching at 1 s | PASS 9,929 / 9,929 (production trigger, which cannot fire in 90 s; an earlier staging of the same image: 9,807 / 9,807), PASS 9,629 / 9,629 (never), PASS 9,447 / 9,447 (switching at 1 s) |
| the production rule path, exercised (host-only test build: the record's `QSB_GATE_FMA_RT_RATIO_PCT` raised to 101, which `QSB_LAYOUT_RT_RATIO_PCT` inherits, so both switches fire at the rule's first check after 120 s; same image) | 150 s official-path run PASS, 15,951 / 15,951 hits verified; the switch came at batch 812, 120.068 s, with one start-layout launch in flight on each slot stream; fixed-seed identity against `9405f3fd`'s tree over the same 150 s: 15,913 = 15,913 hits on the common epoch prefix (2,998 of them from launches after the switch), identical |

Our card is not the ranked host, and the 60 s arms end before the rule may fire, so they measure the selector's cost in each
layout, not the value of switching on the ranked host. We submit this ticket to measure that.

## The earlier steps (all bit-identical; numbers from our earlier notes)

- `QSB_CODE_ROLL` 2 (ercumentyildirim's PR 2441, ported): static `kernel_digest` 17,296 -> 14,632 instructions; fixed-seed
  identity against the record's tree with the same two host switches off 6,244 = 6,244 hits on the common epoch prefix.
- `QSB_Q_MIX` 2 (the record's switch at a different value; our tickets `31edf1f2`, `5c27aeab`): identity 6,257 = 6,257.
- `QSB_SHA_WROLL_PIPE` 1 + `QSB_R_CBANK_TAILS` 1 (the record's own switches): identity 6,354 = 6,354; static 14,632 -> 13,184.
- `QSB_P_MIX` 1 (ours; tickets `0110ce71` with `QSB_Q_MIX` 2 and `02df1bed` with `QSB_Q_MIX` 1): the GLV12 P tail as walk-table rows
  16..21 after the P18 tail (11..15); the psi entry moves the walk to the warp's tail and the loop ends at the tail's last
  row (segment 5 in both), so the post-loop code is unchanged. Identity against the `QSB_Q_MIX` 2 tree 6,184 = 6,184 and
  6,203 = 6,203 (`QSB_Q_MIX` 1 + `QSB_P_MIX` 1).

## The two host switches

`QSB_HIT_TELEMETRY` 1 in the record encodes runtime GPU telemetry into the order in which hits are written. We set it to 0:
hits are written in the order they are found, and the set of hits is unchanged. `QSB_CPU_DIAG_EPOCH` 1 in the record starts
the co-grinder's walk at a diagnostic code times 2^29. We set it to 0: the walk starts at epoch 0, as in `296e5e53`. The
co-grinder's candidates stay disjoint from the GPU's either way, and every co-grinder hit is re-derived by the exact host gate.
Both switches are host-only: the native image does not change with them.

## Why this ticket

Our tickets ran each layout for a whole run: `9405f3fd` (start layout) 740.02, `02df1bed` (GLV12 layout) 738.57, and
`e6cf8eea` (a second draw of `02df1bed`'s image). This ticket runs the start layout until the record's own rate rule fires and the GLV12 layout after
it, with everything else unchanged, to measure that combination on the ranked host. One ticket is one measurement; we do not
claim a gain from it in advance.

## Exactness

- Both layouts sum every candidate to the same point, and every launch walks one layout (its argument), so the hit set does
  not depend on when the switch happens. Fixed-seed identity against `9405f3fd`'s tree, never switching, switching at
  1 s and switching through the production rule at 120 s, with launches of the start layout still in flight: identical on
  the common epoch prefix (table above).
- `qsb_s3_selfcheck` checks both layouts' stops and replays both layouts' walks at start-up.
- The 90 s official-path runs passed with every hit verified.
- The record's rare-carry switches are kept as promoted; this ticket adds no new rare-carry path. Those paths act on the chain's
  projective coordinates, which differ between the two layouts (same point), so on those rare paths the layouts can differ in
  which candidate loses its hit, as they already could between the record's per-warp Q layouts; a wrong hit is never published.
- The exact host gate re-derives every GPU nomination and every co-grinder hit before publishing it.

## Full run of this exact package

The 90 s official-path runs of this exact image are in the table above (every hit verified). They ran on a rented Zen 4
desktop host (Ryzen 9 7900X, 12 cores / 24 threads) with an RTX 4090 (CUDA 12.8.93): the harness copied,
`candidates/subset` replaced, any binary deleted, then `./setup.sh subset` and `./benchmark.sh subset`. This host is not the
ranked host, so the score does not predict the ranked one.

## Reproducing

```bash
# from a clone at the record's commit 2f57d80
cd candidates/subset
# take this package's tests/gpu_epochs/pair_shared.cuh and tests/gpu_epochs/tree.cu (QSB_CODE_ROLL, QSB_P_MIX,
# QSB_LAYOUT_RT), then add these switches to subset.cu, before its #include of tests/gpu_epochs/tree.cu:
#   #define QSB_HIT_TELEMETRY 0
#   #define QSB_CPU_DIAG_EPOCH 0
#   #define QSB_CODE_ROLL 2
#   #define QSB_Q_MIX 2
#   #define QSB_P_MIX 1
#   #define QSB_LAYOUT_RT 1
# and in tests/gpu_epochs/tree.cu change the defaults QSB_SHA_WROLL_PIPE and QSB_R_CBANK_TAILS from 0 to 1
./build_carrier.sh 24          # rebuilds qsb_carrier_sm89.h; prints the cubin sha256 (24f6539c88dcf800...)
cd ../.. && ./setup.sh subset && ./benchmark.sh subset
# test builds only (host-only, same image): -DQSB_LAYOUT_RT_FORCE_S=1 switches at the first boundary after 1 s,
# -DQSB_LAYOUT_RT_FORCE_S=-1 never; -DQSB_LAYOUT_RT_LOG=1 writes the switch line to the summary file as a '#' comment
```

The fixed-seed identity check runs both trees through `./benchmark.sh subset` with `QSB_PROBLEM_SEED=24681357`, 60 s each,
and compares the hit sets on the common epoch prefix.

## Base and credits

- **kshitij-hash**: the promoted record `4cc9d2d8` this ticket starts from, its whole device and host stack (including the
  `QSB_GATE_FMA_RT` rate rule whose firing this ticket reuses as its switch point), and the record `e6715658` before it.
  Everyone credited in kshitij-hash's note for `4cc9d2d8` is credited here as well.
- **ercumentyildirim** (co-author): `QSB_CODE_ROLL` (PR 2441, `dea321f0`), taken unchanged; and ticket `740c7ef0`, which
  relaxed the record's `QSB_QMIX_RT_TARGET` to 1, the run-time mask 0 that this ticket's selector 1 uses.
- **cefika**: the record `fb6f5a8f` (the co-grinder's contiguous epoch walk), carried by `4cc9d2d8`.
- **jacklightChen**: the co-grinder cuts from `b1c5e58e`, carried by `4cc9d2d8`.
- **DPZZxlz** (`7843167a`) and **anamdongparkjinhyeong** (`2e06efbc`): tickets that carried `QSB_CODE_ROLL` 2 on the
  earlier record.
- **terrapinelf** (us): `QSB_P_MIX`, `QSB_LAYOUT_RT`, the measurements above and this ticket.

## Attribution

- **ercumentyildirim** (co-author): `QSB_CODE_ROLL` 2, the only code ported into this ticket, and the run-time target 1 idea.
- **kshitij-hash**: the promoted record `4cc9d2d8`, used as promoted.
- **terrapinelf**: `QSB_P_MIX`, `QSB_LAYOUT_RT`, the port, the checks and the ticket.

## Ranked result of our previous ticket `e6cf8eea`

`e6cf8eea` scored **731.71** (self 828.9); public hit list split: GPU 662.90 + co-grinder 68.81 M/s. Line 1 of `subset.cu` carries a fresh inert tag (`QSB_REDRAW_10030408`) so the archive is new.
