# Subset: our `92c398de` with fkiene's shared-PP fold ported into the chain point add (`QSB_PP_FOLD`), -68 IMAD.WIDE per candidate

Effort: max. Written with Claude Opus 5.5 in Claude Code.

## Summary

This ticket is our package `92c398de` (756.59 M/s on its ranked draw on Oct 6) with one device change in the chain point add,
which by itself is 38% of the kernel's executed instructions: fkiene's shared-PP fold from his unpromoted pinning ticket
`16bf3002` (PR #3784, `_ModRot128` and `_ModMultHR`), ported into the subset point-add asm, plus two small exact companions.
The host code is `92c398de`'s byte for byte (the co-grinder lane included; `QSB_CPU_BATCH` stays at `92c398de`'s 1,024 / 2,048 split).

| switch | file | change |
|---|---|---|
| `QSB_PP_FOLD` 1 | `hit_filter_field_sc.cuh`, `point_add_f8_zz3.cuh` | the three products by PP in the add (ZZ3 = ZZ1·PP, PPP = P·PP, Q = X1·PP) share one PP·2^128 multiplier: each forms a 385-bit sum that is folded once (`_ModMultHR`), with `_ModRot128` for the rotation. fkiene's two-carry captures are rewritten in the form ptxas 12.8 fuses, and the products are issued in the order ZZ, PPP, Q (the only order without spills) |
| `QSB_ADD_F9X15` 1 | `hit_filter_field_sc.cuh` | drops the x15 carry capture in the R^2 square: y14 <= 2^32-2, so the carry out of x14 is provably 0 for every input |
| `QSB_FOLD_FFIRST` 32 | `tree.cu` | reorders two independent fold chains (bit 5 = f13); only the register allocation changes |

Each switch defaults to 0 in the shared headers; with all three at 0 the tree rebuilds `92c398de`'s image `dadec456` byte for byte.
The new native image (`build_carrier.sh`, CUDA 12.8.93) is cubin sha256 `ebd66ca50e8e82320ce2bcbca535890f14b412a833e5e5da9e4bc55687d977d3`
(451,808 bytes); `kernel_digest` keeps 128 registers, no stack frame, no spills, and the host binary's knob string matches the image's
(`Native sm_89 carrier: on`).

## Exactness

`QSB_ADD_F9X15` and `QSB_FOLD_FFIRST` are exact: bit-identical outputs on 2 million mixed and edge inputs.

`QSB_PP_FOLD` is not bit-exact for every canonical input. The port drops a carry in exactly three places, each reachable
from canonical inputs. For each, an interpreter of the asm and a big-integer predicate that follows its dataflow agree on
random cases and on constructed triggers and near-misses. Notation: in `_ModMultHR(a, PP, PPR)`, T = aL·PP + aH·PPR < 2^385
(aL, aH the low and high 128 bits of a) has 32-bit words x0..x12, L192 = T mod 2^192 and Thi = T >> 256.

| class | where | the carry is dropped when | per call, uniform words |
|---|---|---|---|
| R | `_ModRot128` (PPR = PP·2^128 mod p), once per trip | the carry into word 6 is 1 and word 2 of PP is 0xffffffff (for example PP = 2^256-2^128-1) | 2^-65 |
| A | fold word f2 of each `_ModMultHR` (ZZ3, PPP, Q) | L192 + 977·(x8 + x10·2^64 + x12·2^128) >= 2^192 | 2^-57.9 |
| B | carry out of word 5 of each `_ModMultHR` | F + G·2^32 >= 2^192, with F = (L192 + 977·(x8 + x10·2^64 + x12·2^128)) mod 2^192 and G = Thi + 977·(x9 + x11·2^64) | 2^-33 |

A chain trip makes one Rot128 call and three `_ModMultHR` calls, and a candidate runs 7 or 8 trips (7.5 on average).
Assuming the words of the multiplicands are uniform and independent (they come from SHA-derived scalars and table points; no
one chooses them), the added loss is about 2.6e-9 of candidates (2.8e-9 at 8 trips), almost all of it class B. That is an
estimate under that assumption, not a worst-case bound: on adversarial intermediates every call could drop a carry. The
`QSB_ADD_F9X15` cut is exact for every input (y14 = hi32(a6·a7 + c) <= 2^32-2, so the carry out of x14 is always 0), and
`QSB_FOLD_FFIRST` only reorders two independent carry chains.

As a cross-check against an independent big-integer recompute: on 4 million uniformly random point-add inputs the ported add
matched the previous code bit for bit except 2 rows, and in both of those rows the previous code had dropped a carry and the
new one was correct.

Each candidate's point is its own fixed-base chain (built from that candidate's scalar digits), so a dropped carry affects that
one candidate only. The GPU filter can then miss it, or nominate a false hit, which the host recomputes with OpenSSL and drops
before anything is written (the harness also re-derives every hit). The record already carries loss classes of the same kind
(`QSB_LOSS_FINK32`, `QSB_LOSS_SQRLEAN`, about 1.15e-6 of candidates).

GPU hit identity on fixed work (same problem, GPU only, 60 s each, the binary built from this tree): all 6,457 hits that
`92c398de`'s binary found in its window are among the 6,489 that this tree's binary found in the same time (it walked slightly
further along the same candidate order). This checks the shared prefix; it is not a full-coverage proof.

## Instruction counts and local measurement

Dynamic per-candidate counts from exact per-PC instrumentation of `kernel_digest` (NVBit, 3 launches clamped to 2,048 blocks):

| | `92c398de` | this ticket |
|---|---:|---:|
| thread-instructions per candidate | 19,042.4 | 18,896.9 (-0.76%) |
| IMAD.WIDE per candidate | | -68.0 |
| IADD3 / SEL per candidate | | -51 / -25.5 |
| chain-add block (static) | 963 | 947 (IMAD.WIDE 593 -> 585) |
| last add (static) | 1,134 | 1,112 |

On an RTX 4090 at its 450 W limit, GPU only, gate in its plain form (`-DQSB_GATE_FMA_RT_FORCE_S=1` on the host line only), 120 s per
arm in rotating order with the SM clock and board power logged once a second: in the two rounds at matched temperature (83 C) this
image ran at the same power (449.6 W) at a 0.96% higher SM clock and did 1.0% less work per clock, so the rate was even (-0.07%).
It spends less energy per cycle. The ranked card runs most of a draw at its thermal limit, where throughput follows the clock that
the power budget allows. This image's first ranked draw (`df43eca5`, slot 15:20-15:47 UTC) read 735.49 M/s: GPU 664.46 M/s and
co-grinder 71.04 M/s, split from the run's public hit list (the GPU walks 128 window patterns, the co-grinder the other 100).
The neighbouring slot on the same runner ran `92c398de`'s device image and read GPU 646.07 M/s. That is a same-period reference,
not a paired comparison: the card's thermal state drifts through the day. This ticket is a second draw of the same image.

## Reproducing

```
./setup.sh subset
./benchmark.sh subset
```

Off the official runners:

```
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/{bench}/{bench}.cu --no-build' ./benchmark.sh subset
```

To rebuild the image: `NVCC=/path/to/cuda-12.8/bin/nvcc ./build_carrier.sh` in `candidates/subset`.

## What the rest of the package is

Base: kshitij-hash's promoted record `faf5422a` (benchmark commit `efef868`). On top of it, `92c398de` (ours) has the exact device cuts
`QSB_CC_GLUE` 1, `QSB_C3_SHA_WIN` 3, `QSB_C3_TREE_GLUE` 381, `QSB_C3_TAIL_ORDER` 1, the record's own `QSB_TREE_UNROLL` 1,
`QSB_SHA_WROLL_PIPE` 1 and `QSB_R_CBANK_TAILS` 1, `QSB_Q_MIX` 2, `QSB_CODE_ROLL` 2, host-only `QSB_HIT_TELEMETRY` 0 and
`QSB_CPU_DIAG_EPOCH` 0, and the co-grinder lane `QSB_CPU_PIN_WORKERS` 1, `QSB_CPU_PFD1` 3, `QSB_CPU_BATCH_SIB` 2048,
`QSB_CPU_TOUCH_FUSE` 1, `QSB_CPU_KHFUSE` 1. Line 1 of `subset.cu` carries an inert tag (`QSB_REDRAW_10071549`), unreferenced and outside every knob
string.

## Base and credits

- **fkiene** (co-author): the shared-PP fold (`_ModRot128`, `_ModMultHR`), from his unpromoted
  pinning ticket `16bf3002` / PR #3784, ported here into the subset point add.
- Ours: the port into the subset add asm (fused carry forms, product order, `QSB_ADD_F9X15`, `QSB_FOLD_FFIRST`), the exactness and
  hit-identity checks, the instruction counts and the measurements; and `92c398de` underneath.
- **i34-9** (co-author): `QSB_CPU_PIN_WORKERS`, as `92c398de` credits it.
- **dukemawex** (co-author): `QSB_CPU_TOUCH_FUSE` and the `QSB_CPU_PFD1` value, as `92c398de` credits them.
- **petarkostov** (co-author): the `QSB_CPU_PFD1` 3 lane, as `92c398de` credits it.
- **cefika** (co-author): `QSB_CPU_BATCH_ODD`, which `QSB_CPU_BATCH_SIB` ports, as `92c398de` credits it.
- **kshitij-hash**: the promoted record `faf5422a` under all of it; **ercumentyildirim**: `QSB_CODE_ROLL` (PR 2441); and everyone those
  packages credit.
- All inherited source, GPLv3 notices (`COPYING`, `COPYING-secp256k1`) and attributions are kept.
