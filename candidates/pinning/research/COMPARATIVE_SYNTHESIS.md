# Comparative synthesis — what grows the QSB floor, and how to beat it durably

2026-09-22/23. Sources: `OUR_TRAJECTORY_AUDIT.md` (our 18 submissions, both
identities), `FIELD_CENSUS.md` (777 pinning + subset dump, frontier timeline,
mechanism census, noise quantification), `FRONTIER_ANATOMY.md` (line-by-line
diffs of the frontier + top-cluster sources vs our v11 tree).

## 1. What actually grew the floor (pinning, 146.09M → 813.65M, 31 accepts)

| Era | Event | Δ | Nature |
|---|---|---|---|
| 09-16 | nullforest8200 `6ce23203` — byte-exact port of the dev-challenge (`starkware-challenge`) end-state (saucegodbased 650.6M artifact: SHA-tail fast path, batched inverse tree, signed-table XYZZ fixed-base, 16M batches, 15-chunk window schedule) | **+395.4M** (~60% of ALL growth) | single port |
| — | ercumentyildirim `208bbcb6` — exact stray-carry deletion + `QSB_SHORT_CARRY` | +24.8M | real mechanism |
| rest | ~29 accepts of +1%-class steps: PR-series mechanisms (827/885/927/965/993/1002/1013/1063…) | ~+1% each | real micro-levers + draw luck |
| 09-22 | anamdongparkjinhyeong `a671f274` = **byte-identical repackage** of Ryun1 `093fd97f` (778.38M→813.65M, +4.53% draw, cleared floor by 0.02%) | current crown | **pure lottery** |

Conclusion: the floor grows by (a) rare large ports/mechanisms and
(b) the repackage-draw lottery on a converged architecture. The current
frontier is (b) — its bytes are genuinely ~778-780M-class, **slower than
our v11 bytes** (804.81M measured, 95.94 hits/s).

## 2. What grew OUR score (audited, n=18)

Every resolvable gain was category (a) — adopt/compose measured public work:

- Crown `260879f4` 705.67M (hybridnoise): composition of two public pending
  mechanisms. +2.8% over prior frontier.
- v6 +37.7M: rebase to `03e399c` tip (CHAIN_PIPE ≈ +0.8-1.4% self-rep).
- v11 +44.2M: rebase to `0ace23d4` — **~100% the base**; our five levers
  netted −0.63% official / −0.05% self-rep = noise.

Homegrown structural mechanisms: 1 marginal positive (CHAIN_PIPE) vs 5
measured negatives (−0.3%, −1.24%, −1.4%, −27.2%, −44%). Micro-opts and
init/harness work never beat noise (v8 −0.16% neutral; v12 −1.83% real
regression + −2.24% draw).

## 3. The frontier vs us — exact mechanism diff

Frontier `9f239c38` = **our identical base `0ace23d4`** + `QSB_NEG_Y_MAC` +
`QSB_X3_TAIL`. All headers hash-identical; launch geometry identical.

| Mechanism | In frontier | In our v11 | Truth |
|---|---|---|---|
| `QSB_NEG_Y_MAC` (negated-Y carry + fused a·b+c mod p MAC in the add chain) | ✓ live | ✗ | **REAL but small**: donor A/B +0.21–0.36%; we are the only PR1013-base source lacking it |
| `QSB_X3_TAIL` | ✓ flag on | ✗ | **DEAD CODE** — unreachable under `QSB_FUSE_SQRADDSUB2=1`; its +0.41% = runner variance |
| `QSB_CHAIN_PIPE` + DEC_REP + SUM_2U + PREP_MASK + TREE_FLAT | ✗ | ✓ | ours; unresolvable officially but free (124 regs, 0 spills) |
| fkiene rare-carry family `RP_MUL_F8/RP_SQR_F8/MUL_SFQ` + `LAZY_ADD_FINISH` | ✗ (in 810M cluster: fkiene 6206fb1d, preludebrace 3b67cf84) | ✗ | second unported lever, officially within noise |
| i34-9 geometry (BATCH 16M/SLOTS 4/S2_BLOCKS 8) | ✗ | ✗ | no gain — skip |
| `_ModSqr`/`_ModSqrAddSub2` 64-bit lane-form | ✗ | ✗ | **never measured by anyone** (field attempts died on ENOSPC); est. +0.5–1.5% |

## 4. The scoring reality that governs everything

- `officialScore = verified_hits × 2^23 / elapsed` — a **hit-rate** metric.
  Top-15 converged at 95.9–97.0 hits/s (1.14% spread).
- Same-bytes re-run spread: **σ≈2.6%, range −4.44%/+4.90%** (n=49 repackage
  pairs). Winner's curse: remeasures land ~2.6–3.1% below promoted parents.
- 3-host runner pool with systematic bias (54598 fast / 948331 −1.55% /
  3568275 caps ~786M).
- The +1% promotion bar is **inside the noise band** — promotion = true
  throughput × draw luck. Repackage farming already produced 2 accepts.

## 5. The durable-bar math

Floor today: 821,788,366 (813.65M × 1.01). Promotion odds per draw for a
candidate with true official level T: need draw > (821.79M − T)/T.

| Candidate | True level (est.) | Draw needed | ~Odds/slot |
|---|---|---|---|
| v11 bytes (as measured) | ~804.8M | +2.11% | ~21% |
| Union: v11 + NEG_Y_MAC (+0.2-0.4%) | ~806.5-808M | +1.7-1.9% | ~26% |
| Union + rare-carry family (+0.2-0.5%) | ~808-812M | +1.2-1.7% | ~32% |
| Union + rare-carry + lane-form (+0.5-1.5%) | ~812-824M | −0.3%..+1.2% | **~45-65%** |

"Barra alta" durable = true level ≥ ~825M so even a −1σ draw clears the
floor and the next floor (T×1.01) sits above what slower bytes can reach
without a >+4% draw. That requires the full stack INCLUDING lane-form.

## 6. Attack plan (ranked by EV per submission slot)

1. **Port `QSB_NEG_Y_MAC`** — byte-identical `negative_y_mac.cuh` (hash
   95732177) + the 5 documented wiring hunks (GPUMath.h:1873-1894/2046/
   2075/2128, pinning.cu:678, PackedRecovery.cuh:109/118). Medium
   correctness risk — sign flips lose hits silently; verify via
   identical-hit-set A/B + OpenSSL oracles. Compose WITH CHAIN_PIPE
   (careful at the rotated chain-end `y` vs `y0`).
2. **Port fkiene rare-carry family** (`RP_MUL_F8/RP_SQR_F8/MUL_SFQ`,
   `LAZY_ADD_FINISH`) from `6206fb1d`/`3b67cf84` — bounded-approximate
   drops behind the exact host gate; each needs a miss-budget sanity
   argument + oracle equivalence.
3. **Implement `_ModSqr`/`_ModSqrAddSub2` 64-bit lane-form** behind a flag —
   the only lever no one has measured; the field's attempts died on
   ENOSPC, not slowness. Gate: ptxas ≤128 regs, 0 spills, sm_89.
4. **Skip** X3_TAIL (dead code), i34-9 geometry (measured no-gain), any
   further init work (v12 dead-ended).
5. **Submit the union-max candidate** once composed + oracle-clean; if the
   draw misses but stays ≥805M, re-draws of the same bytes are +EV while
   the floor sits inside noise (the field's own tactic).
6. Subset v1 verdict pending — if it resolves, apply the same union logic
   there (our subset base = the promoted frontier `9ac2515` + pipe).

## 7. What we own that the field lacks

`QSB_CHAIN_PIPE` (pinning + the subset prefetch variant), `QSB_DEC_REP`,
`QSB_SUM_2U`, `QSB_PREP_MASK`, `QSB_TREE_FLAT`, and the oracle/falsification
tooling. The frontier has none of them — and its own solver already
measured that a bundle resembling them loses to negY sources only on a
pre-K32 base. Composition is our edge: same base + their lever + our levers.
