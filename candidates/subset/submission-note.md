# Subset: kshitij-hash's promoted `faf5422a` with five exact instruction cuts (-162 instructions per candidate together): ours in the constant-block SHA callee (`QSB_CC_GLUE` 1), the window loop (`QSB_C3_SHA_WIN` 3), the block-inverse tree and root (`QSB_C3_TREE_GLUE` 381) and the tail order (`QSB_C3_TAIL_ORDER` 1), and the record's own `QSB_TREE_UNROLL` 1, two of the record's own exact device switches (`QSB_SHA_WROLL_PIPE` 1, `QSB_R_CBANK_TAILS` 1), runtime telemetry and the co-grinder's diagnostic walk start turned off, plus cadamcat's co-grinder worker pinning as navrajin-alt ported it to the record's co-grinder (`QSB_CPU_PIN_WORKERS` 1, host-only), a 2048 batch on each core's second SMT sibling (`QSB_CPU_BATCH_SIB`, host-only, after cefika's `QSB_CPU_BATCH_ODD`) and key-hash fusion into the co-grinder's final pass (`QSB_CPU_KHFUSE` 1, host-only)

## Starting point

The promoted subset record is kshitij-hash's `faf5422a`, 753.57 M/s (GPU part 686.03 + co-grinder 67.54 M/s from its public hit list), landed as benchmark commit `efef868`; the promotion bar is 761.11. Its native image is the image of our package `4cc9d2d8` + `QSB_CODE_ROLL` 2 + `QSB_Q_MIX` 2 byte for byte (cubin sha256 `5f1f811166d423a6...`), so everything `faf5422a` changed relative to our package `4cc9d2d8` + `QSB_CODE_ROLL` 2 + `QSB_Q_MIX` 2 is on the host side, and this package carries it unchanged.

On top of `faf5422a`'s tree this package applies the changes of our package on `4cc9d2d8` (the sections below; their checks were run on `4cc9d2d8`'s tree). `faf5422a` already carries ercumentyildirim's `QSB_CODE_ROLL` port; this package turns on the record's own `QSB_SHA_WROLL_PIPE` and `QSB_R_CBANK_TAILS` and sets the switches in the table below. Every other switch keeps `faf5422a`'s value. This package also carries four switches of ours (`QSB_CC_GLUE` 1, `QSB_C3_SHA_WIN` 3, `QSB_C3_TREE_GLUE` 381, `QSB_C3_TAIL_ORDER` 1) and the record's own `QSB_TREE_UNROLL` turned on (below), so its native image differs from the record's (cubin sha256 `dadec456af927c91...`, the image of our previous tickets); with all five at 0 the image is byte for byte `5554da9f3ba62973...`.

This exact tree: the native image rebuilt with the package's own `build_carrier.sh` (CUDA 12.8.93) is cubin sha256 `dadec456af927c91...`, its knob string matches the host binary's (MATCH 3253 bytes), and a 90 s run of this exact tree through the unmodified harness passed (9,644 / 9,644 hits verified).

| switch | `faf5422a` | this package | where it acts |
|---|---|---|---|
| `QSB_CODE_ROLL` | 2 | 2 (restated in `subset.cu`; unchanged) | device: the pair gate's two recovery-id hashes as one 2-trip loop (ercumentyildirim, PR 2441) |
| `QSB_Q_MIX` | 2 | 2 (restated in `subset.cu`; unchanged) | device: the per-warp mix of the two Q layouts (half of the warps instead of a quarter decode Q with the six GLV12 terms) |
| `QSB_SHA_WROLL_PIPE` | 0 | **1** | device: the paired hash's window block as one 8-round loop with its W+K loads issued half a trip ahead (the record's own switch) |
| `QSB_CC_GLUE` | absent | **1** | device: the constant-block SHA callee's block loop rolled with each block's rounds 0..15 peeled (new, ours; below) |
| `QSB_C3_SHA_WIN` | absent | **3** | device: the window loop's row pointer stepped in place, the first two trips peeled, the loop ends on the pointer (new, ours; below) |
| `QSB_C3_TREE_GLUE` | absent | **381** | device: glue cuts in the block-inverse tree and its warp-0 root (bits 0, 2-6 and 8; new, ours; below) |
| `QSB_C3_TAIL_ORDER` | absent | **1** | device: the two tails called in the order B, A (new, ours; below) |
| `QSB_TREE_UNROLL` | 0 | **1** | device: the block inverse's fixed 8-level tree written out for the 256-thread block (the record's own switch) |
| `QSB_R_CBANK_TAILS` | 0 | **1** | device: the tail callees read the recovery point R from the constant bank instead of eight 64-bit ABI arguments (the record's own switch) |
| `QSB_CPU_PFD1` | 8 | 8 (unchanged; the record's value) | host: the co-grinder's first-window loop prefetches its table rows 8 groups ahead (our previous ticket used 3; below) |
| `QSB_CPU_BATCH_SIB` | (absent) | **2048** | host: workers on each core's second SMT sibling batch 2048 candidates, the first-sibling workers keep 1024 (after cefika's `QSB_CPU_BATCH_ODD`; below) |
| `QSB_CPU_TOUCH_FUSE` | 0 | 0 (unchanged; the record's value) | host start-up: the full first-touch pass over a 9- or 10-window co-grinder table runs as in the record (our previous ticket used 1; below) |
| `QSB_CPU_PIN_WORKERS` | absent | **1** | host: each co-grinder worker is bound to one logical CPU of the process mask, one per physical core first, then the SMT siblings (cadamcat's switch as navrajin-alt ported it to the record's co-grinder, unchanged; below) |
| `QSB_CPU_KHFUSE` | absent | **1** | host: the co-grinder's key hashes run inside its final batch-affine pass instead of in a separate loop after it (ours; below) |
| `QSB_HIT_TELEMETRY` | 1 | **0** | host: no runtime telemetry encoded in the hit order |
| `QSB_CPU_DIAG_EPOCH` | 1 | **0** | host: the co-grinder walks from epoch 0, without a diagnostic code in its start |

## The co-grinder's first-window prefetch distance and first-touch pass at the record's values (host-only; since our earlier ticket)

- **What changed:** two constants in `CpuGrindSubset.h`, against our previous ticket's file (git blob `c17640ec`):
  `QSB_CPU_PFD1` 3 -> 8 (line 249) and `QSB_CPU_TOUCH_FUSE` 1 -> 0 (line 626). Both are now the record's values, as in
  `faf5422a`'s file (git blob `1d6a6e06`). Nothing else changes except the inert tag on line 1 of `subset.cu`: the device
  sources are byte for byte our previous ticket's, the native image is the same, and the new co-grinder file is git blob
  `57bd8d33e518`.
- **Exactness:** both are timing or start-up settings only. `QSB_CPU_PFD1` is how many groups of candidates ahead the
  first-window loop (the window-0 table loads) issues its row prefetches; a prefetch is a cache hint, so the same rows are
  loaded. `QSB_CPU_TOUCH_FUSE` 0 runs the full first-touch pass over a 9- or 10-window table, as the record does, instead of
  leaving the remaining page faults to the table build's own writes; the table entries are the same (the `table_check` gate
  is unchanged). The same candidates are walked in the same order, and the records are the same.
- **Why:** our ranked reads put both values inside noise: `QSB_CPU_PFD1` 3 about +0.1 +/- 0.1 % of the co-grinder's rate,
  `QSB_CPU_TOUCH_FUSE` 1 +0.02 %. Our previous tickets used 3 and 1, values from dukemawex's and petarkostov's tickets; this
  ticket uses the record's 8 and 0, so neither of those values is in this package. Against `faf5422a`'s file the co-grinder
  file now differs only by `QSB_CPU_PIN_WORKERS`, `QSB_CPU_BATCH_SIB` and `QSB_CPU_KHFUSE` (2 of the record's lines
  replaced, 156 lines added).
- **Checks of this exact tree:**
  - CPU-only lane harness (no CUDA), 120 s at 20 zero bits on 4 cores with both siblings each (8 workers):
    `RESULT VALID`, 2,594 / 2,594 hits verified.
  - Staged package through the unmodified harness, 90 s at 24 zero bits: `RESULT: PASS`, 9,644 / 9,644 hits verified.
    Fingerprint MATCH 3253; cubin `dadec456af927c91` and PTX `0f6532a3b355` unchanged.

## Also in this ticket: `QSB_CPU_PIN_WORKERS` 1 (host, co-grinder; the native image is unchanged)

The worker pinning is cadamcat's `QSB_CPU_PIN_WORKERS` (first in `1eaebd49`; PR #3113 is `7b919feb`) as navrajin-alt ported
it to the record's co-grinder file in `24f8fc43` (2026-10-02, 03:11 UTC): the record's file plus 53 lines, git blob
`f09f0a0a`. cadamcat's note credits the earlier per-worker CPU binding of HyeokxC's `0735233a`. The same file is in i34-9's
`4eba03a9` (about eleven hours later) and in ercumentyildirim's tickets (for example `65206f94`, which took it from
`4eba03a9`); it reached us through `65206f94`. Our earlier tickets credited it to i34-9, following `65206f94`'s note; this
ticket credits it as the public record shows.

At start the co-grinder reads the process CPU mask, orders its CPUs one per physical core first and then the SMT siblings (from
`/sys/devices/system/cpu/cpuN/topology/thread_siblings_list`), and each worker binds itself to one CPU of that list before it
lowers itself to `SCHED_IDLE`. Without the pinning the scheduler moves the idle-class workers between CPUs; with it each worker
keeps its core's caches and TLB. It changes no candidate, no walk order and no gate: the same epochs and window patterns are
walked by the same number of workers. `QSB_CPU_PIN_WORKERS 0` or the environment variable `QSB_CPU_PIN_WORKERS_ENV=0` restores
unbound workers. This ticket keeps those 53 lines unchanged; our `QSB_CPU_BATCH_SIB` adds two lines beside them (the number of
physical cores in the list, which it reads).

## `QSB_CC_GLUE` 1 (device, bit-identical; new switch of ours)

`qsb_pair_const4` (`tests/gpu_epochs/window_schedule_shared.cuh`, used with the record's `QSB_CONST_CALLEE` 1) hashes the
four constant SHA-256 blocks of a pair. In the record each block runs its 64 rounds as a rolled 8-round loop that starts from
working registers, so every block begins with 16 register copies of the saved state, a loop-counter reset and a reset of the
K+W row base. With `QSB_CC_GLUE` 1 the block loop itself is rolled with one pointer walk over the K+W rows across all four
blocks (the uniform `c[0x3]` row loads stay uniform), each block's rounds 0..15 are peeled so they read the saved state words
and write the working registers directly, and rounds 16..63 run as three 16-round trips (half the loop control per round).
The words, the rounds and their order are unchanged, so every digest, candidate and hit is bit-identical. With
`QSB_CC_GLUE` 0 (default) the callee and the native image are byte for byte the record's.

| check | result |
|---|---|
| static `kernel_digest` | 13,184 -> 13,056 instructions; the constant callee 1,002 -> 879 (64 working-state copies gone); 128 registers, no stack, no local memory, no spills; the chain loop's text unchanged (1,078) |
| dynamic instructions per candidate (NVBit, `kernel_digest`) | 19,204.58 -> 19,145.08 (-59.5, -0.31 %): IADD3 -36, IMAD.MOV.U32 -31.5, BRA -10.5, UIADD3 -8, ISETP -8; IMAD.IADD +28 and CALL +8 (ptxas's forms) |
| fixed-seed identity against the same tree with `QSB_CC_GLUE` 0 (seed 24681357) | 6,354 = 6,354 hits on the common epoch prefix, identical |
| GPU-only rate on our card, 60 s arms, 4 rounds ABBA | -0.221 % +/- 0.035 (-0.22, -0.32, -0.18, -0.17) |

Our card is not the ranked host. The cut removes instructions but loses some operand reuse in the callee, so on our card it reads
slower; we submit it to measure the instruction cut on the ranked host and report the local number as measured.

## `QSB_C3_SHA_WIN` 3, `QSB_C3_TREE_GLUE` 381 and `QSB_C3_TAIL_ORDER` 1 (device, bit-identical; new switches of ours)

Three new switches remove loop control, register copies and redundant lane work around the arithmetic. Each is 0 by default
(the base byte for byte) and joins the image's knob string only when non-zero.

**`QSB_C3_SHA_WIN` 3** (`tests/gpu_epochs/tree.cu`, `window_schedule_shared.cuh`): the `QSB_SHA_WROLL_PIPE` window loop with its
16-byte row pointer kept as one 64-bit register that inline PTX steps in place (`add.cc`/`addc`) and reads with
`ld.global.v4.u32` (the base's own 128-bit load), the first two 8-round trips peeled (they read the loaded first-state words, so
the 16 working-state copies before the loop go; their rows are read at immediate offsets and the loop's pointer lags them), and
the loop ending on the pointer's low word instead of a trip counter. The same 64 rounds on the same rows in the same order (each
row read once, plus the one discarded read of padding row 16 the base also makes) and the same feed-forward. The exit test
compares the low 32 bits of the pointer: it advances 2 rows per trip over at most 8 trips (16 rows, far below 2^32 bytes), so
the low word takes a distinct value at every step and equals the end value exactly when the 64-bit pointer does.

**`QSB_C3_TREE_GLUE` 381** (bits 0, 2, 3, 4, 5, 6 and 8; `tree_inverse.cuh`, `inverse_limbs.cuh`):
- bit 0: no batch cap in the warp-0 root's divstep loop. The loop is Bernstein-Yang's divstep on f = p and g = the lazy root
  (0 <= g < 2^256). By Bernstein-Yang (2019, Theorem 11.2) g is 0 after at most floor((49*256+80)/17) = 742 divsteps (724 for
  256-bit inputs), within 25 batches of 30, and the loop leaves at the first batch whose g is zero. The cap of 32 batches is
  never reached, and the fallback behind it never runs (0 executions in 262,144 blocks under NVBit). Removing the counter, its
  test and the fallback changes no value.
- bit 2: `acc -= factor*m` as `acc += (-factor)*m` with -factor (0, -1 or -977) formed once outside the loop. m < 2^30 and
  |factor| <= 977, so the signed product is exact in 64 bits and the sum does not wrap (|acc| < 2^62).
- bit 3: the 30-step decision on lanes 0 and 8 only (instead of 0, 8, 16 and 24). The two columns of the transition matrix are
  independent and delta's update does not read them, so lane 0 (column 0) and lane 8 (column 1) form the same integers that
  lanes 0/8 and 16/24 form in the base, and the matrix entries the shuffles deliver are the same.
- bit 4: the root's canonical form computed on lane 0 only; its only caller reads lane 0's result alone.
- bit 5: tree level branches whose writer count is a multiple of 32 test `tid < count` through a full-warp vote; the value is
  warp-uniform, so the same threads run the same products in the same order, without the reconvergence bracket.
- bit 6: each 32-byte tree node stored as four 64-bit stores to the same bytes (instead of two 16-byte stores that need copies
  into aligned register quads); the loads stay 16-byte; same words, addresses and barriers.
- bit 8 (needs bit 2): three forms in the root loop's full-warp body. (a) The carry bias is added when the accumulator starts:
  biased = Kb + a*x + b*y + (-factor)*m mod 2^64 with the loop-invariant per-lane Kb = 2^63 - (digit ? 2^31 : 0), the same 64-bit
  word as the base's acc + 2^63 - (digit ? 2^31 : 0); m is shuffled from the row's digit-0 lane, whose Kb has a zero low word, so it
  is computed from the same low word as before. (b) The correction is one PTX mad.wide.s32 (-factor, m < 2^30: exact int64).
  (c) next = digit == 7 ? high : next0 as one lop3 with a loop-invariant lane mask. (d) The top limb's -2^31 (the bias telescoping
  into high) is added with m: high = a*xt + b*yt + (int32)(m | 2^31), and (int32)(m | 2^31) = m - 2^31 because m < 2^30; the same
  int64 high after hi7 and the carry are added (no wrap: |high| < 2^62). Same values throughout.
Bits 3 and 4 keep the same warp instructions with fewer active lanes; the other bits remove whole instructions.

**`QSB_C3_TAIL_ORDER` 1**: tail(B) is called before tail(A). Each tail is the same call on the same words and returns the same
verdict; only the order of a thread's two possible hit records changes. Hit records are already written in an inter-warp order
set by atomicAdd, and the record set (one record per passing candidate, far below the per-launch cap) is the same.

| check | result |
|---|---|
| static `kernel_digest` | 13,648 -> 13,296 instructions (smaller); 128 registers, no stack, no local memory, no spills; the chain loop 1,078 with the same opcode histogram |
| dynamic instructions per candidate (NVBit, `kernel_digest`) | 19,113.46 -> 19,042.40 (-71.1, -0.37 %) on top of the two cuts below; 19,204.58 -> 19,042.40 (-162.2, -0.84 %) for all five |
| fixed-seed identity against the tree without these three switches (seed 24681357) | 6,354 = 6,354 hits on the common epoch prefix, identical |
| GPU-only rate on our card, 60 s arms, 2 rounds ABBA | +0.011 % +/- 0.144 (+0.16, -0.13) on our card |

## `QSB_TREE_UNROLL` 1 (device, bit-identical; the record's own switch)

The record's `QSB_TREE_UNROLL` (`tests/gpu_epochs/tree.cu`, `tree_inverse.cuh`) writes the block inverse's product tree out for
the 256-thread `kernel_digest` block (a static assert checks the block size), so the per-level loop counters, branches and
address arithmetic go and the row offsets become immediates. Same products of the same operands in the same order:
bit-identical.

| check | result |
|---|---|
| dynamic instructions per candidate (NVBit, `kernel_digest`), on the `QSB_CC_GLUE` 0 tree | 19,204.58 -> 19,172.96 (-31.6, -0.16 %) |
| fixed-seed identity of the same switch on the wr tree (seed 24681357) | 6,373 = 6,373 hits on the common epoch prefix, identical |
| GPU-only rate on our card (wr tree, 4 rounds ABBA) | +0.074 % +/- 0.018 (+0.07, +0.12, +0.03, +0.08) |

## `QSB_CC_GLUE` 1 and `QSB_TREE_UNROLL` 1 together (our earlier ticket's image `efd38418`)

| check | result |
|---|---|
| static `kernel_digest` | 13,184 -> 13,648 instructions; 128 registers, no stack, no local memory, no spills; the chain loop 1,078 with the same opcode histogram |
| dynamic instructions per candidate (NVBit) | 19,204.58 -> 19,113.46 (-91.1, -0.47 %); the two cuts add up (-59.5 and -31.6) |
| fixed-seed identity against the tree without both cuts (seed 24681357) | 6,373 = 6,373 hits on the common epoch prefix, identical |
| GPU-only rate on our card, 60 s arms, 2 rounds ABBA | -0.033 % +/- 0.033 (-0.07, +0.00) on our card |

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

This ticket is our previous package with the two co-grinder constants above back at the record's values. The native image (cubin `dadec456af927c91...`) and the device sources are unchanged; the co-grinder file is `faf5422a`'s plus `QSB_CPU_PIN_WORKERS`, `QSB_CPU_BATCH_SIB` and `QSB_CPU_KHFUSE`. Line 1 of `subset.cu` carries a fresh inert tag (an unreferenced `QSB_REDRAW_*` define). The device cuts described above were measured on the ranked host in our earlier tickets. We claim no gain beyond what the ranked run shows.

## Exactness

- Fixed-seed identity, step by step: the `QSB_CODE_ROLL` 2 tree against `4cc9d2d8`'s tree with the same two host switches off
  (6,244 = 6,244 hits on the common epoch prefix), this package's `QSB_Q_MIX` 2 step against the `QSB_CODE_ROLL` 2 tree (6,257 = 6,257), and this exact tree against the
  `QSB_Q_MIX` 2 + `QSB_CODE_ROLL` 2 tree (6,354 = 6,354); identical every time.
- The 90 s official-path run of this exact tree passed with every hit verified (9,644 / 9,644).
- Fixed-seed identity of the `QSB_CC_GLUE` 1 tree against the same tree with it at 0: 6,354 = 6,354 hits on the common epoch prefix, identical.
- `faf5422a`'s switches other than the ones in the table above are kept as promoted; this ticket adds no new rare-carry path.
- This tree's native image is cubin sha256 `dadec456af927c91...`, the image of our previous tickets. Its fixed-seed identity against the same device tree without the three `QSB_C3_*` switches: 6,354 = 6,354 hits on the common epoch prefix, identical (table above). The device checks above were run when each switch was introduced; the device files are unchanged since.
- The co-grinder file is `faf5422a`'s plus three host-only switches. They change where a worker runs, how it groups its own candidates into batches and when its key hashes are computed; each keeps the candidates, their order and the records (sections above and below).
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
# and in tests/gpu_epochs/tree.cu change the defaults QSB_SHA_WROLL_PIPE and QSB_R_CBANK_TAILS from 0 to 1,
# and apply this ticket's QSB_CC_GLUE change (tests/gpu_epochs/tree.cu and window_schedule_shared.cuh; default 1),
# and change the default QSB_TREE_UNROLL from 0 to 1 in tests/gpu_epochs/tree.cu,
# and apply this ticket's QSB_C3_SHA_WIN / QSB_C3_TREE_GLUE / QSB_C3_TAIL_ORDER changes (tree.cu, window_schedule_shared.cuh,
# tree_inverse.cuh, inverse_limbs.cuh; defaults 3 / 381 / 1)
# and replace CpuGrindSubset.h with this ticket's own file (git blob 57bd8d33e518): faf5422a's file (git blob 1d6a6e06)
# plus QSB_CPU_PIN_WORKERS (as in navrajin-alt's 24f8fc43, git blob f09f0a0a), QSB_CPU_BATCH_SIB and QSB_CPU_KHFUSE
# (the host-only co-grinder switches described above and below)
./build_carrier.sh 24          # rebuilds qsb_carrier_sm89.h; prints the cubin sha256 (dadec456af927c91...)
cd ../.. && ./setup.sh subset && ./benchmark.sh subset
```

The fixed-seed identity check runs both trees through `./benchmark.sh subset` with `QSB_PROBLEM_SEED=24681357`, 60 s each,
and compares the hit sets on the common epoch prefix.

## Key-hash fusion into the co-grinder's final pass (`QSB_CPU_KHFUSE` 1, host-only, already in our earlier ticket `d4f58600`; the native image is unchanged)

- **What changed:** the co-grinder's last batch-affine pass (`ec8_final_cf_khf`, `ec8_final_cf_kh`'s EC steps line for line) now runs `kh16_pass` on each pair of groups right after it stores their key-hash message words. Before, it stored the words for the whole batch (72 B per candidate) and hashed them in a separate loop afterwards. The message-word buffer shrinks from 576 B per group to 1,152 B per worker; each group's 16-bit prefilter mask goes into a per-batch array, which also removes one phase transition per batch. The batches stay 1024 / 2048 per SMT sibling pair.
- **Exactness:** the same function runs on the same 144 message words, so the key hashes and prefilter masks are the same. The publication loop after the batch is unchanged except that it reads the stored mask. Groups, candidates and recovery ids therefore reach the exact gate in the same order, and the hit file holds the same records in the same order. `QSB_CPU_KHFUSE` 0 builds the previous code byte for byte.
- **Checks when it was introduced** (on our previous co-grinder file, git blob `c17640ec`, which differs from this one only in the two constants above):
  - CPU-only lane harness, 90 s on 4 cores with both siblings each: `RESULT VALID`, 144 / 144 hits re-derived by `harness/verify.py`.
  - Against the previous lane at 24, 20 and 16 zero bits: identical hit sets (0 only in either; 126, 785, 2,901 and 26,915 hits) and identical per-worker publication sequences, including ~1,350 cases of two hits in one batch.
  - A unit test gives bit-identical masks, message words and gate-call sequences, and it fails on a deliberately swapped mask.
  - Staged package: `RESULT: PASS`, 9,670 / 9,670 hits verified. Fingerprint MATCH; cubin `dadec456af927c91` and PTX `0f6532a3b355` unchanged.

## Co-grinder batch per SMT sibling (`QSB_CPU_BATCH_SIB` 2048, host-only; the native image is unchanged)

- **What changed:** one host-only change in `CpuGrindSubset.h` (since our ticket `44d7206d`). When the shared-core batch (`QSB_CPU_BATCH`, 1024) is in use and the
  workers are pinned (`QSB_CPU_PIN_WORKERS`: one CPU per physical core first, then the SMT siblings), each worker on a core's second
  sibling uses a batch of 2048. The first-sibling workers keep 1024, so one core holds 0.25 + 0.5 MB of EC state in its 1 MB L2.
  Unpinned lanes fall back to odd workers, as in cefika's switch. The device image is unchanged (cubin `dadec456af927c91`).
- **Origin:** cefika's `QSB_CPU_BATCH_ODD` (the odd-numbered workers take a second batch size), in cefika's ranked tickets on this
  record (co-grinder file git blob `e6825eb2`, first in `171d06c5`). With 32 workers on 16 two-thread cores (the ranked host) the
  pinning puts worker t and worker t + 16 on the two siblings of one core, so the larger batch goes to one sibling per core
  instead of to odd t.
- **Exactness:** each worker owns its batch buffers and walks its own contiguous epoch range in order. Only the grouping of its own
  candidates into batches changes, so the candidates, their order and the records are the same. 2048 is a multiple of 32 within
  32..8192 (static_assert), the same range `QSB_CPU_BATCH_RT` accepts.
- **Checks when it was introduced** (our ticket `44d7206d`, whose co-grinder file also had `QSB_CPU_PFD1` 3):
  - CPU-only lane harness (no CUDA), 150 s on 4 cores with both siblings each (8 workers, batch rule 1024 and 2048):
    `RESULT VALID`, 197 / 197 hits re-derived by `harness/verify.py`. The base lane in the same harness: 205 / 205.
  - Staged package: `RESULT: PASS`, 9940 / 9940 verified hits. Fingerprint MATCH 3253; cubin and default PTX unchanged
    (`dadec456af927c91`, `0f6532a3b355`).

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
- **cadamcat** (co-author): `QSB_CPU_PIN_WORKERS`, the co-grinder worker pinning (`1eaebd49`; PR #3113, `7b919feb`): the switch,
  its run-time off switch and the one-CPU-per-core-first order that navrajin-alt's port follows (unpromoted work, hence the
  co-author credit).
- **navrajin-alt** (co-author): the port of that switch to the record's co-grinder file (`24f8fc43`, git blob `f09f0a0a`), whose
  53 lines this ticket carries unchanged (unpromoted work, hence the co-author credit).
- **HyeokxC**: the earlier per-worker CPU binding in `0735233a`, which cadamcat's note credits. The record's own all-CPU worker
  footprint (`QSB_CPU_ALLCPU`) is credited in its source to the same ticket, and HyeokxC is among the record's co-authors.
- **i34-9** and **ercumentyildirim**: tickets that carried the same co-grinder file on this record (`4eba03a9`; `65206f94` and
  others), through which it reached us.
- **terrapinelf**: the `QSB_CC_GLUE`, `QSB_C3_SHA_WIN`, `QSB_C3_TREE_GLUE` and `QSB_C3_TAIL_ORDER` cuts, `QSB_CPU_BATCH_SIB` and
  `QSB_CPU_KHFUSE`, the switch settings (including the record's own `QSB_TREE_UNROLL`), the checks and the ticket.
- **cefika** (co-author): the per-worker second batch size (`QSB_CPU_BATCH_ODD`; co-grinder file git blob `e6825eb2`, first in
  `171d06c5`) that `QSB_CPU_BATCH_SIB` ports to pinned workers (unpromoted work, hence the co-author credit).
- **tanhaien**: an earlier ranked ticket with the record's `QSB_TREE_UNROLL` 1 on the record's own image (`756b1fcc`), which we read as one more data point.

**Attribution request.** Several recent subset tickets reuse our native image `dadec456` (our exact cuts `QSB_CC_GLUE`, `QSB_C3_SHA_WIN`, `QSB_C3_TREE_GLUE` and `QSB_C3_TAIL_ORDER` on kshitij-hash's `faf5422a`) and our co-grinder changes (`QSB_CPU_BATCH_SIB` per-sibling batches, `QSB_CPU_KHFUSE` key-hash fusion). That is what open submissions are for, so thanks for building on them. Following the challenge's co-author norm, please list @terrapinelf as a co-author when your ticket carries them, as kshitij-hash already does. We credit cadamcat and navrajin-alt (worker pinning) and cefika (the per-sibling batch idea) the same way.

## Ranked result of our previous ticket `d7b07788`

`d7b07788` scored **745.29** (self 870.1); public hit list split: GPU 675.08 + co-grinder 70.21 M/s. Line 1 of `subset.cu` carries a fresh inert tag (`QSB_REDRAW_10060616`) so the archive is new.
