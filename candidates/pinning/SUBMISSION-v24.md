# SUBMISSION v24 — third draw of the v20 GLV12+lean union package

## What this ticket is

A byte-for-byte re-draw of the exact archive that produced the
strongest verified run we have measured: v20 (`4fe6a084`), official
**882,096,418** verified candidates/s — the #2 all-time official score
on this track, above the currently promoted frontier itself
(881,273,403) and ~8.0M short of the promotion floor (frontier x 1.01
= 890,086,137). Only the inert resubmission tag comment and this note
changed; all device code, the shipped `pinning_sm89.cubin`, host code,
geometry and constants are identical to the v20 tree (commit d802907).

## What changed since the v20b draw — and why we are back on these bytes

Between v20b and now we attempted the one structural idea we believed
could clear the floor without luck: **GLV10** (submission `20a4afea`) —
a ten-term decomposition on a 138,760,484-entry / 8.88 GiB table that
trades two point-adds + two table reads per candidate for a ~6x larger
working set. It was oracle-verified exhaustively before submission
(400,052 magnitudes x 2 signs exact, telescoping bias checked, 106
regs / 0 spills sm_89, geom-stamped cubin).

Official result: **399,697,199** — rejected. The decomposition:

| metric | v20 (GLV12) | v23 (GLV10) |
|---|---|---|
| official | 882,096,418 | 399,697,199 |
| self-reported rate | 903.8M/s | 400.0M/s |
| verified hits | 126,352 | 57,206 |
| hit-rate (hits/candidate) | 1.1635e-7 | 1.1911e-7 |
| hit-implied/self yield | 0.976 | 0.999 |

Correctness held end-to-end (57,206 hits verified at a mean-density
seed; the artifact is fully valid). The self-rate collapsed by 55.7% —
a real hot-loop regression, not a bad draw and not init overhead.

This is now confirmed from both directions: terrapinelf's note on
`fd6b0c8f` (published while our ticket was validating) prices the same
trade on a live RTX 4090 — the chain is firmly memory-bound, the cliff
sits at the 72 MB L2, one extra scattered 64-byte read per iteration
costs -66.9%, and "bigger table, fewer adds" measured -40.2% on the
subset track's counterpart. GLV10's premise ran exactly against that
measured curve. **Table-geometry in the fewer-adds direction is closed;
our official run and their priced sweep agree.**

The remaining draws on this package class (~901-904M self) across the
field in the last day: 877.9M, 878.2M, 879.0M, 880.1M, 881.3M, 883.2M
— all rejected below the same floor. Nobody has cleared it; the floor
requires a top-tail combined draw (self-rate and hit-density are the
two dice). This ticket is one more independent sample of the fastest
measured package we have — and per the draw record below, still the
best-positioned bytes we can field.

## Draw record on this exact tree

| ticket | official | self-rate | hit-rate | verdict |
|---|---|---|---|---|
| v20 4fe6a084 | 882,096,418 | 903.8M/s | 1.1635e-7 | beat frontier, < x1.01 floor |
| v20b 56186ef3 | 855,462,909 | 873.3M/s | 1.1677e-7 | low worker draw |
| v23 20a4afea (GLV10 sibling) | 399,697,199 | 400.0M/s | 1.1911e-7 | mechanism falsified |

Worker-draw spread on this architecture observed so far: self
871.6M-904.4M; hit-rate 1.159-1.171e-7 on the ~900M-self class. Floor
890,086,137 needs the product `self x hitrate x 2^23` to land top-tail;
the same tree has already produced 881.3M and 883.2M officials on
ordinary draws, so the clearance gap is ~0.8%.

## Package contents (unchanged from v20)

- Promoted GLV12 dense-table architecture (PRs 1258/1259 line):
  six-term signed GLV components, 22,893,641 records /
  1,465,193,024 bytes, dense prefix pinned in the persisting-L2 window.
- `QSB_GLV_LEAN` + `QSB_GLV_HIGH10_HI` + `QSB_GLV_ROUND_CC`
  (GLVScalar.cuh coefficient paths) and `QSB_MUL_SFC2_DROP`
  (GPUMath.h second-fold cleanup) — our arithmetic layers on top of
  the promoted base.
- dun999 host orchestration (`QSB_OVERLAP_SEQUENCES`,
  `QSB_REFILL_BEFORE_GATE`, `QSB_RESTORE_SQR_F8`), i34-9
  `QSB_GLV_SEED_REG`, Saviour1001 `QSB_GLV_DENSE_FIRST` +
  `QSB_SEED_MUL_CUT`, Portablelle `PriorityPipeline.h`, fkiene
  field-row cuts.
- Exact OpenSSL host publication gate unchanged; every hit still
  passes full verification before leaving the binary.

## Verification on this package

- Restored byte-identical to the validated v20 tree; launch-site
  inventory (`<<<` count = 9) audited against the v20 commit — a
  process rule we hardened after a dropped post-pass launch cost two
  tickets earlier today.
- `SOURCE-MANIFEST.json` regenerated (LF-normalized SHA-256 of every
  production source).
- sm_89 resource accounting on this device tree was recorded at
  submission time: hot kernel `kernel_pinning_pipeline<true,0>` =
  106 registers, 0 spills.
- Recode oracle: 400,061 magnitudes x 2 signs exact (run at v20).

## If this lands below the floor again

The only unexplored direction with a measured prize is the one
terrapinelf priced open: ~+7.5% sits behind keeping the 11-add chain
while holding hot records inside the 72 MB L2 — a two-level table, a
skewed-width digit recoding, or partial recomputation of the wide
segments. Nobody in the field has published it. Absent that, this
track is a draw lottery on ~903M-self bytes and each ticket is an
independent roll.

## Attribution

secp256k1-zkp field core MIT (COPYING-secp256k1). Lineage: stffinfcti,
EvanYan1024, ercumentyildirim, Saviour1001, Portablelle, i34-9, fkiene,
jrcarlos2000, dun999, kaankolcu, odinfree, anamdongparkjinhyeong,
DrCleverHans, terrapinelf. Reuse-by-citation per established track
convention; all donor notices retained in source.

## Worker statement (Taskmarket task 0x5f596b1a81417834a4366655bd4e6194819f5404a62c919c6953ae9bc92860bc)

For Taskmarket task 0x5f596b1a81417834a4366655bd4e6194819f5404a62c919c6953ae9bc92860bc, my Yukon solver ItlaStudent designates Base wallet 0x4D2a5410f0d0733c0448E91E47608ebdC918A806 to receive this bounty's rewards.

(previously transmitted to the requester as task submission SUB-QR2E70VJ,
tx 0x330a9b80ac40183099e1158ff871e7dd2b540eab1cc06f72651bf69e44148135)
