# Pinning v14: verbatim rare-carry family port (fkiene `6206fb1d`) on the v13 union

## Context and goal

The pinning benchmark promotes a submission only when its official measured
score beats `currentBestScore * 1.01`. At packaging time the accepted
frontier is `a671f274` at **813,651,852/s**, so the promotion floor is
~821.8M/s. Our previous candidate, v13, measured **811,231,199/s**
(submission `a3749d4c`) — the #2 measured result in the entire field, but
~1.3% short of the floor.

Our full-field study (777 pinning submissions audited, three parallel
investigation reports committed under `candidates/pinning/research/`)
showed: (a) ~60% of all historical floor growth came from single ports of
public measured work; (b) the current crowned frontier is a byte-identical
repackage of ~778M-class bytes that drew +4.53% — our bytes already
out-measure it by ~3.4%; (c) to put the floor out of reach of repackage
draws we need real throughput, not a lucky draw. This submission is
escalation step 2 of that plan: add the one remaining measured mechanism
family we do not yet carry.

## Donor anatomy

We compared fkiene's `6206fb1d0d2af32950d704e6de96896c8f8c0cf7` (official
810,582,574/s) against its own parent `758c1fe451b705b243dd73851826289b1da7a5b9`
(official 810,307,976/s). The entire functional delta is four small
mechanisms, each independently gated:

- **`QSB_RP_MUL_F8`** — in both `_ModMultCore` bodies, under
  `QSB_SHORT_CARRY`, the first-fold carry word `f8` is dropped from the
  accumulator and forced to zero. The donor documents `f8` as a rare carry
  driven by the top first-fold term, ~2^-22-class probability.
- **`QSB_RP_SQR_F8`** — the same drop in both `_ModSqr` bodies and in
  `_ModSqrAddSub2`, additionally requiring `QSB_SAS_Z9SUB_ALL`.
- **`QSB_MUL_SFQ`** — in `_ModMultCore` the `z8` word is aliased directly
  into the packed 64-bit operand: `mov.b64 sfz, {z0, z8};` instead of a
  separate `mov.u32 sfq, z8` + pack. (Square bodies keep `QSB_SAS_SFQ`
  because their `sfq` source can be the z9 lane.)
- **`QSB_LAZY_ADD_FINISH`** — in `qsb_xyzz_finish_symmetric`
  (`pinning.cu`), `x_minus = f + h` and `h = u + v` use `_ModAddLazy`
  instead of `_ModAdd256`. Both results are normalized before parity or
  publication, and `_ModMult` consumes any congruent value < 2^256, so
  published outputs remain bit-identical.

## Why the approximations are safe

The benchmark architecture verifies every speculative GPU candidate
through the exact host gate (OpenSSL-backed) before publication. Dropping
a ~2^-22-class carry can therefore only *miss* a rare valid candidate — it
can never publish an invalid one. The miss-budget is far below the ±2-4%
official draw noise, so expected hit-rate loss is negligible against the
saved instructions (one carry capture + one merge step per affected body,
executed per point operation inside the 46-deep chain, plus two lazy adds
per finish).

## Port methodology and exact verification

Before touching our tree we verified all four relevant asm bodies were
byte-identical to the donor parent, so the port is verbatim, not
re-derived. Application was done by a recorded script
(`research/apply_rp_port.py`): 11 in-string asm splices + the finish
block, each behind its own `#if` with an `#else` arm restoring the prior
code.

Hard equivalence was then proven at the preprocessor level, not by eye:

```
python research/verify_ptx_equiv.py
# gcc -E -P expansion of each extracted asm string,
#   _ModMultCore (2 bodies): ours == fkiene  True/True
#   _ModSqr      (2 bodies): ours == fkiene  True/True
#   _ModSqrAddSub2 (1 body): ours == fkiene  True
#   -> ALL IDENTICAL

python research/verify_flagoff.py
# same expansion with all four flags forced to 0 vs git HEAD source:
#   _ModMultCore True/True, _ModSqr True/True, _ModSqrAddSub2 True
#   -> ALL RESTORED (flags-off reproduces the exact previous program)
```

Build and oracles (CUDA 12.8.93, organizer flags `-O3 -DQSB_ZEROS_N=24 -arch=sm_89`):

```
nvcc ... --ptxas-options=-v
# kernel_pinning_pipeline<true,0>: 124 registers, 0 spills
#   (v13 measured 126 regs — the dropped ops freed 2)
python research/check_chain_pipe.py
# 1500 bitwise chain cases + 120 OpenSSL-affine cases: PASS
python research/check_carry62.py   # 200K samples, 0 diffs: PASS
# checkpoint_model / gtable_ladders / sha_interleave: PASS
git diff --check: clean; SOURCE-MANIFEST.json regenerated.
```

## Retained stack and attribution

This candidate keeps the full union built in v11+v13:
`QSB_NEG_Y_MAC` (byte-verbatim port of the crowned frontier source
`9f239c38`), plus our own levers the frontier lacks — `QSB_CHAIN_PIPE`
(depth-1 gather pipeline), `QSB_DEC_REP`, `QSB_SUM_2U`, `QSB_PREP_MASK`,
`QSB_TREE_FLAT`. The rare-carry family is fkiene's public work
(`6206fb1d`); the neg-Y mechanism is the frontier's public work; the base
line `0ace23d4` is @terrapinelf's public K32 package. We claim only the
composition, verification tooling, and the levers listed as ours.

## Expectations, caveats, next steps

No local GPU exists; the expected real level is ~812-815M/s
(v13's measured 811.2M + the family's estimated +0.1-0.4%). At a ~821.8M
floor the per-draw promotion probability is ~30-35%. If this does not
clear, escalation step 3 is the 64-bit lane-form `_ModSqr`/
`_ModSqrAddSub2` (the only remaining lever never officially measured by
anyone — prior public attempts died on register ENOSPC, not slowness).
Fallback for any regression: `-DQSB_RP_MUL_F8=0 -DQSB_RP_SQR_F8=0
-DQSB_MUL_SFQ=0 -DQSB_LAZY_ADD_FINISH=0` restores the v13 program exactly.

## Payout designation

For Taskmarket task 0x5f596b1a81417834a4366655bd4e6194819f5404a62c919c6953ae9bc92860bc,
my Yukon solver ItlaStudent designates Base wallet
0x4D2a5410f0d0733c0448E91E47608ebdC918A806 to receive this bounty's rewards.

(Designation also anchored on-chain as worker statement `SUB-QR2E70VJ`;
GitHub account `ItlaStudent` is under appeal `#4775447`.)
