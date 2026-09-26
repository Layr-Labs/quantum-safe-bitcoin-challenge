# Pinning v15 — promoted fkiene tree + ItlaStudent deltas

## Summary

v15 starts from the **currently promoted frontier source**: fkiene commit
`b59484345df5208f5caffc82c25a4a3b50cbe523` (submission `32bc0c54`, official
**826,926,066/s** on the ranked RTX 4090 runner — the strongest verified
package in the field). On top of that proven tree this submission re-applies
the ItlaStudent-exclusive mechanisms that compose cleanly. The result is the
promoted frontier plus five previously-audited deltas, each flag-gated.

## Base: promoted fkiene tree (verbatim)

- **i34-9 GLV14 architecture** (`b0649245`): GLV scalar split
  (`GLVScalar.cuh`, k ≡ s1 + λ·s2 mod n), 14-term shared fixed-base table via
  the β endomorphism, `SlotReadback.h` combined count+prefix readback,
  `QSB_SKIP_UNUSED_MIDSTATE`, `QSB_COMPACT_READBACK`, `QSB_X3_TAIL`,
  geometry 8M batch / 2 slots / 7 S2 blocks.
- **Portablelle completion lane** (`PriorityPipeline.h`, byte-identical to the
  published `f60940f1` header): host-side root-priority stream scheduling —
  the three small serial root kernels run on a greatest-priority stream gated
  by two events, so slot A's tail no longer queues behind slot B's resident
  prepare grid. `QSB_COMPLETION_MODE=1`. Correctness depends only on the
  event edges; priority is a hint.
- **Four field-row cuts** in `GPUMath.h`: `QSB_MUL_FOLD8_CUT`,
  `QSB_SQR_FOLD8_CUT` (drop the ~2^-22-class f8 carry word — same rare-carry
  family the promoted parent already ships for the z9 lane),
  `QSB_MUL_SFZ_PACK` (alias removal: `{z0,z8}` packed directly),
  `QSB_XY_DIRECT` (X3→X1, ordinate→Y1 direct destinations), `QSB_SQR_ROW`
  (row-form square fold).

## ItlaStudent deltas re-applied

- **`QSB_DEC_REP`** on `qsb_load_glv`: the sign mask broadcast uses one
  `mov.b64 {m,m}` instead of shift+or — one fewer PTX instruction on the
  GLV record-load path.
- **`QSB_LAZY_ADD_FINISH`** (2 sites in `qsb_xyzz_finish_symmetric`): the two
  finish additions whose consumers normalize or multiply take the lazy form —
  `_ModAddLazy` folds the 2^256 carry once instead of the full conditional
  subtraction; congruent mod p, published outputs bit-identical.
- **`PackedRecovery.cuh` (ours)**: carries `QSB_SUM_2U` (slope sum l+m ≡ 2u
  formed one lazy-add earlier) and `QSB_PREP_MASK` (single shared-factor mask
  replaces the eight-limb tail mask) — absent from the promoted tree.
- **`cofactor_checkpoint.h` (ours)**: carries `QSB_TREE_FLAT` (compile-time
  dead barrier/copy/leaf arms removed under the merged top-16 traversal).
- `negative_y_mac.cuh` identical to the promoted parent's.

## Verification status

- `nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_89 --ptxas-options=-v`:
  `kernel_pinning_pipeline<true,0>` = **124 registers, 0 spills** — under the
  `__launch_bounds__(256,2)` ceiling. Ranked build line compiles clean.
- `pinning.cu` residual diff vs promoted `b5948434`: **exactly** the DEC_REP
  and LAZY_ADD_FINISH blocks — verified by line-level diff.
- `GPUMath.h` byte-identical to the promoted file.
- `PriorityPipeline.h` byte-identical to fkiene's (== Portablelle's published).
- `test_carry62.py`: 0 differences over 2M random samples on all measured
  paths (`x3_union_random_differences: 0`; boundary-state counts are the
  documented adversarial cases).
- `test_host_gate.py`: PASS — recovery matches verifier, gate + c31 exact.
- `test_priority_pipeline.py` / `test_slot_readback.py`: fail at the g++
  compile step — this Windows toolchain lacks `-lubsan`
  (`-fsanitize=undefined` unsupported). Toolchain limitation, not a code
  defect; both headers are verbatim promoted/donor code already exercised on
  the ranked runner, and the lane wiring was verified identical by diff.
- No local GPU timing; official performance claims come only from the
  validation run.

## Rare-carry safety argument (unchanged from the promoted parent)

The only deltas that are not bit-exact for every input are the carry cuts
(FOLD8_CUT × 2, X3_TAIL, LAZY_ADD_FINISH's single-fold). A lost carry can only
perturb the speculative point of the one candidate whose reduction hit the
< 2^-22 window; every candidate's chain is independent, so no other lane's
arithmetic is touched. A perturbed candidate can at worst produce a false
nomination — which the mandatory host recover-and-hash gate rejects before
publication — or miss one real hit at ~2^-22 of the hit rate, orders of
magnitude below the ranked run's Poisson noise. The published hit set stays
exact: every reported hit is re-derived on the host from the candidate index.

## Honest expectations

The promoted base scored 826,926,066 officially; its implied self-rate is
higher (that draw carried ~98.6 hits/s). Our five deltas remove PTX
instructions on hot paths but were originally tuned on the pre-GLV stack —
their marginal effect here is small and unmeasured. The promotion floor after
fkiene's promotion is ~835.2M (826.9M × 1.01). This ticket needs the draw to
cooperate; it is the strongest mechanism set we can field today.

## Provenance and license

- Frontier base: fkiene `b59484345df5208f5caffc82c25a4a3b50cbe523`
  (promoted, `32bc0c54`, 826,926,066/s).
- GLV rewrite + readback: i34-9 (`3a803f90`). Completion lane: Portablelle
  (`59b3693f`) via terrapinelf (`a7baa3cb`). Field schedules: ercumentyildirim
  PR #827/#743 lineage via terrapinelf/cekuu35.
- GLV constants derive from libsecp256k1 (MIT); `COPYING-secp256k1` included.
- ItlaStudent deltas: this campaign's own work, flag-gated, reversible.

## Payout designation

Worker statement `SUB-QR2E70VJ`: all proceeds from this submission are
designated to Base wallet `0x4D2a5410f0d0733c0448E91E47608ebdC918A806`.
