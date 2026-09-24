# Frontier anatomy — what separates 9f239c38 (813,651,852) from our v11 tree (804,814,918)

Audit date: 2026-09-22. Method: `git fetch origin <sha>` for each competitor commit, then
`git diff -w` (whitespace-insensitive — every submission commit rewrites the whole directory with
different line endings, so raw `git diff` is unusable). All SHAs verified with `git cat-file -t`.

## 0. Executive summary

**The new frontier is our own adopted base plus one real mechanism plus one dead flag.**

`9f239c38` = `0ace23d4` (terrapinelf PR1013-class source, the exact public base of our v11,
official 809,952,202) **+ `QSB_NEG_Y_MAC`** (negative deferred ordinate + fused seeded
multiply-add, `negative_y_mac.cuh`, from public PR #1060/#1063) **+ `QSB_X3_TAIL`** — which is
**dead code under the shipped flags** (see §3). Everything else — SHA path, launch geometry,
table layout, checkpoint structure — is byte-identical or flag-identical to our tree.

The official gap 810,314,192 (758c1fe4 = base+negY) → 813,651,852 (frontier = base+negY+dead-X3)
is **+0.41% on a functionally identical kernel** — i.e. runner/draw variance, not mechanism.
The only real code lever we lack is `QSB_NEG_Y_MAC` (+ fkiene's optional `RP_*_F8`/`MUL_SFQ`
rare-carry drops, present in the 810.58M rejected source but not in the frontier).

Our tree additionally carries mechanisms the frontier does NOT have (`QSB_CHAIN_PIPE`,
`QSB_DEC_REP`, `QSB_SUM_2U`, `QSB_PREP_MASK`, `QSB_TREE_FLAT`, all live) plus dead v12 init-cut
scaffolding (all flags =0, cosmetic).

## 1. Lineage proof

```
94abdd0 (797.45M, promoted) ── +PR993 RAW finish +PR999 pkg +PR1002 K32 +TOP16 +narrow parity
        = 0ace23d4 "PR1013" (809,952,202)   <── our v11 base
0ace23d4 + QSB_NEG_Y_MAC  = 758c1fe4 "PR1063" (810,314,192)   [rejected #2]
758c1fe4 + QSB_X3_TAIL    = 9f239c38 (813,651,852)            [new frontier]
```

Verified: `git diff -w 758c1fe4 9f239c38 -- candidates/pinning/` = **only** GPUMath.h (+14:
X3 flag + `QSB_X3_FOLD` splice inside `_ModX3Fused`) and pinning.cu (+7: flag, `#error` coupling,
one startup printf at :2998). `git diff -w 0ace23d4 9f239c38` = negative_y_mac.cuh (+243),
GPUMath.h (+68), PackedRecovery.cuh (+14), pinning.cu (+12). That is the complete delta vs our
adopted base.

## 2. Mechanism M1 — `QSB_NEG_Y_MAC` (the only real separator)

**What it is.** The XYZZ chain's deferred ordinate is stored **negated** (−Y_def instead of
Y_def) end-to-end: `_PointAddXYZZ_mm` seeds it via a swapped subtraction, all 13
`_PointAddXYZZT<true>` iterations preserve the sign convention, the checkpoint/finish carry −Y,
and the packed finish compensates by swapping the slope operands.

The payoff: the per-add step `R = S2·ZZZ1 − Y1` becomes `R = S2·ZZZ1 + (−Y1)` — an **add** of the
stored negative ordinate, foldable into the multiply itself. `qsb_muladd_seed(r,a,b,c)` computes
`r = a·b + c mod p` in a single asm block: `_ModMultCore`'s exact 32-bit schoolbook (64
`mul.wide.u32`, even/odd column split) with c's four limbs injected as bias into accumulator
words e0..e3 *before* reduction, then the identical h·977 double fold.

**Evidence (frontier):**
- `negative_y_mac.cuh` (new file, 243 lines): `qsb_muladd_seed` asm — `mov.u64 bias,%12;
  add.cc.u64 e0,e0,bias;` ×4 limbs + `addc.u64 bias_carry,0,0` inside the product accumulation.
- `GPUMath.h:1873-1894`: flag + `#include "negative_y_mac.cuh"` + `qsb_negate_residue` (normalizes
  the [p,2^256) exceptional case then `_ModNeg256`; only used in the dead non-defer arm).
- `GPUMath.h:2045-46` (`_PointAddXYZZT`): `qsb_muladd_seed(R,S2,ZZZ1,Y1)` replaces
  `_ModMult(S2,ZZZ1); _ModSub256(R,S2,Y1)`.
- `GPUMath.h:2074-75`: `_ModSub256(Q,T,Q)` (X3−V, negated) replaces `_ModSub256(Q,Q,T)` — the
  subsequent `_ModMult(Q,R)` then yields −R·(V−X3) = −Y3def, preserving the invariant.
- `GPUMath.h:2127-28` (`_PointAddXYZZ_mm`): same swap seeds −Y at the mm seed.
- `pinning.cu:678`: chain end `qsb_muladd_seed(Y,y0,V,Y)` replaces our
  `_ModMult(x1,y0,V); _ModSub256(Y,Y,x1)` (our pinning.cu:814) — one MAC resolves the anchor and
  emits −Y_actual.
- `PackedRecovery.cuh:108-110` (+117-118 non-lazy arm): `_ModAddLazy(l,u,v); _ModSub256(m,u,v)`
  instead of `_ModSub256(l,u,v); _ModAddLazy(m,u,v)` — stored v is negated, so l=u+v_stored and
  m=u−v_stored recover the original slopes at zero cost.

**Us:** none of this exists; our chain does `_ModMult(S2,ZZZ1); _ModSub256(R,S2,Y1)`
(GPUMath.h:2045-46 region / pinning.cu:728-730 in the pipe body) and `Q−X3` ordering
(GPUMath.h:2128 equivalent, PackedRecovery.cuh:127).

**Size of effect.** Per `_PointAddXYZZT` call: removes one `_ModSub256` (~10-11 SASS under
K32_SUB: 4×sub.cc + mask + 2×and + mov.b64 + 32-bit K fold) and adds ~9 bias instructions inside
the MAC (4 mov + 5 add); net count ≈ −1..−3 instrs, but the sub's **serial latency disappears
from the mult→R dependency** on all 13 chain adds plus the chain-end MAC (14 sites/candidate).
Donor's matched A/B: **+0.214% (8 seq) / +0.362% (16 seq)**, identical sorted hit sets
(1,201 / 2,384 hits); official isolated delta +0.04% (809.95→810.31, inside draw noise).
Field evidence: every negY-carrying source sits ≥806.9M; none below.

**Port difficulty: moderate.** Byte-identical header copies cleanly; the invasive part is
applying the sign convention inside our `_PointAddXYZZT_pipe` (pinning.cu:698-770) and the
rotated chain end (our pinning.cu:801 uses `y`, not `y0` — the anchor variable differs under
CHAIN_PIPE). PackedRecovery swap composes with our `QSB_SUM_2U` (`sum = u+u` still valid since
l+m = 2u either way — keep it; the frontier reverted to `l+m` only because it lacks SUM_2U).

**Correctness risk: medium.** A single missed sign flip silently negates Y → wrong
pubkeys/parities → real hits lost (host gate only filters false positives, cannot recover misses).
The donor package is proven (byte-identical hunks, identical hit sets on two seeds); port must
reproduce the convention exactly, incl. the `qsb_negate_residue` boundary rule (product reps in
[0,2^256), `_ModNeg256` requires ≤p). Verify with the debug-mode hit-set A/B before any submit.

## 3. Mechanism M2 — `QSB_X3_TAIL` (frontier-only, but DEAD CODE)

Frontier `pinning.cu:33-38` + `GPUMath.h:602-619`: in `_ModX3Fused`, the h·K fold's
4-instruction carry chain (`add.cc t0,t0,k; addc.cc t1; addc.cc t2; addc t3`) becomes a single
`add.u64 t0,t0,k`, accepting a 2^-33/op divergence (host-gate filtered).

**Critical caveat the field missed: `_ModX3Fused` is unreachable under the shipped config.**
With `QSB_FUSE_SQRADDSUB2=1` (both trees, GPUMath.h:144-146), every live `R²+PPP−2V` goes through
`_ModSqrAddSub2` (frontier GPUMath.h:1993-95, 2057-59). `_ModX3Fused`'s only call sites are the
`#else` arms (GPUMath.h:1999, 2063) and `_PointAddXYZZ_early` under `QSB_EARLY_LOAD=0`
(pinning.cu:546). A `__forceinline__` device function with no live caller emits no SASS — the
frontier's hot loop is instruction-identical to 758c1fe4's. The +0.41% official delta is
runner/hit-draw variance on the same binary (their own SUBMISSION.md documents a 3-host runner
pool: host 54598 vs 948331 differs by −1.55%, and a byte-identical A/A drew 810.31M vs 797.69M).

Classification: **cosmetic/dead**. Do not port for speed. (If ever run with
`-DQSB_FUSE_SQRADDSUB2=0`, it is a real ~3 instrs × 13/candidate cut at 2^-33/op risk.)

## 4. Mechanisms we have that the frontier lacks

| Flag | Our evidence | Status |
|---|---|---|
| `QSB_CHAIN_PIPE=1` | pinning.cu:680-801 — `_PointAddXYZZT_pipe`, 3-buffer (x,y,o) rotating prefetch of chunk c+1/c+2 into dead regs | Live. Officially unproven: v11 (804.81M) sits inside the base's own draw spread (809.95M / 809.25M resubmit / 775.78M redraw). Frontier solver's own 070d51b2 (808.72M) carried the identical mechanism on a weaker pre-K32 base and lost to negY sources. |
| `QSB_DEC_REP=1` | pinning.cu:660-677,725-731 — `mov.b64 {m,m}` sign-mask dup | Live, ~1 instr/load ×15/candidate. Frontier uses plain `(m32<<32)|m32` (their pinning.cu:649). Also in 070d51b2. |
| `QSB_SUM_2U=1` | PackedRecovery.cuh:59-63,128-130 — `sum=u+u` one lazy-add earlier | Live; frontier reverted to `sum=l+m` (their PackedRecovery.cuh:111). |
| `QSB_PREP_MASK=1` | PackedRecovery.cuh:65-70,86-94 — mask `hc` once vs 8-limb tail mask | Live; frontier masks vbar/tbar post-multiply (their :72-74). |
| `QSB_TREE_FLAT=1` | cofactor_checkpoint.h:111-119,152,210,228 — dead barrier/copy arms removed | Live; frontier's cofactor_checkpoint.h is byte-identical to base 0ace23d4 (hash ca3adacf). |
| v12 scaffolding | pinning.cu:21-38, `QSB_JIT_WARM/LAD_THREADS/SPOT_DEV` all **=0** | Dead host code + `<atomic>/<thread>` includes. Cosmetic — but costs nothing at runtime. |

## 5. Rejected-cluster pass (what they have that we don't)

| Source | Score | vs base 0ace23d4 | Has | Lacks |
|---|---|---|---|---|
| fkiene `6206fb1d` | 810.58M | GPUMath.h +129, negY +243, pinning +29 | negY (byte-identical 95732177) + `QSB_RP_MUL_F8`/`QSB_RP_SQR_F8` (drop the f8 first-fold carry word in `_ModMultCore`, both `_ModSqr`, `_ModSqrAddSub2` — rare-carry class) + `QSB_MUL_SFQ` (alias z8, drop `mov.u32 sfq,z8`) + `QSB_LAZY_ADD_FINISH` (2 lazy adds in stage-2 finish, pinning.cu `QSB_FINISH_ADD`) | X3_TAIL, CHAIN_PIPE, our extras |
| terrapinelf `758c1fe4` | 810.31M | GPUMath.h +54, negY +243, pinning +5 | negY only (this IS PR1063) | everything else |
| i34-9 `d889efa5` | 810.05M | negY + geometry | negY + `QSB_BATCH 16M`, `QSB_SLOTS 4`, `QSB_S2_BLOCKS 8` ("small register spill") — **no arithmetic changes**; also committed a build artifact | — |
| preludebrace `3b67cf84` | 809.53M | = 6206fb1d + i34-9 geometry | negY + RP_F8 bundle + LAZY_FINISH + BATCH16M/SLOTS4/S2_8 | — |
| terrapinelf `ca461625` | 809.25M | ~0 | **byte-identical PR1013 resubmit** (= our base, no new code) | everything |
| anamdong `070d51b2` | 808.72M | pre-K32 base + CHAIN_PIPE + DEC_REP + PREP_MASK + SUM_2U + TREE_FLAT | **our exact v11 bundle minus K32** — the frontier solver's own earlier attempt | negY, K32, X3 |
| terrapinelf `1e8f3b02` | 808.04M | ~PR999 (no K32) | — | K32, negY |
| fkiene `2f504874` | 806.94M | negY-enhanced + RP_F8 | negY variant (284-line header: +`QSB_MAC_BIAS_DIRECT` −4 staging movs, +`QSB_MAC_SFQ` −1 mov) + RP_F8/SFQ + LAZY_FINISH | — |
| preludebrace `16e6c9e8` | 805.53M | older base | — | K32, negY |

Read: negY sources occupy 5 of the top-6 slots; the fkiene RP_F8+SFQ bundle and the i34-9
geometry are both officially indistinguishable from plain negY (810.58/810.05 vs 810.31) and
preludebrace's bundle+geometry scored *below* the bundle alone — geometry is likely neutral or
slightly negative at these margins.

## 6. Ranked levers separating 813.65M from our 804.81M

1. **`QSB_NEG_Y_MAC` — real, port it.** Fused `a·b+c mod p` MAC + negated-ordinate convention.
   ~13 fused sites + chain end + slope swap. Modeled ~+0.2-0.4% (donor A/B); officially the
   mechanism behind every ≥806.9M source. Port = copy `negative_y_mac.cuh` verbatim + 5 hunks
   (GPUMath.h flag/`qsb_negate_residue`/dispatcher/`_PointAddXYZZT`/`_PointAddXYZZ_mm`;
   pinning.cu chain end; PackedRecovery slope swap — keep `QSB_SUM_2U`). Medium risk (sign
   convention must be exact through the checkpoint); verifiable via identical hit sets.
2. **Runner/draw variance — not code.** Frontier's own docs: byte-identical sources swing
   775.78M↔810.31M across the 3-host pool (only host 54598 reaches 800M+). Our 804.81M vs the
   base's 809.95M is inside that band — the v11 CHAIN_PIPE bundle is *not* falsified by the
   official number, and the frontier's edge over PR1063 is *not* mechanism.
3. **fkiene rare-carry bundle (`RP_MUL_F8`/`RP_SQR_F8`/`MUL_SFQ`, +`LAZY_ADD_FINISH`)** — present
   in 3 of top-10 (810.58/809.53/806.94). ~1 instr per multiply/square site (~7 sites/chain-add →
   ~90 instrs/candidate modeled ~+0.2%); officially within noise of plain negY. Easy port (macro
   splices already written); each drops a ≤2^-31-class correction — same risk family as our
   existing SHORT_CARRY/CARRY62/C31, host-gate protected.
4. **`QSB_X3_TAIL` — dead code, zero effect.** §3. Skip (or port only for source parity).
5. **i34-9 geometry (`BATCH 16M`, `SLOTS 4`, `S2_BLOCKS 8`)** — no official gain observed;
   S2_BLOCKS=8 admits a register spill. Skip.
6. **fkiene enhanced MAC (`MAC_BIAS_DIRECT`, `MAC_SFQ`)** — 284-line variant, −5 instrs/MAC call;
   only in the 806.94M draw. Optional follow-on once base negY lands.

## 7. Field consensus — what the top-10 share that we lack

- **`negative_y_mac.cuh`**: present in 6 of 10 top sources (5 byte-identical, hash 95732177…;
  fkiene's 2f504874 has the extended variant). **We are the only source on the PR1013 base
  without it.** The 4 top-10 sources lacking negY are: our HEAD, the pure-base resubmit
  (ca461625), anamdong's pre-K32 CHAIN_PIPE attempt (070d51b2 — i.e. our own mechanism
  portfolio on an older base), and pre-K32 line holdovers (1e8f3b02, 16e6c9e8).
- **Shared base**: every ≥808M source except 070d51b2 is on the PR1013 K32+RAW+TOP16+narrow
  base — which we already have byte-for-byte (GPUHash.h, sha_pinsha.cuh,
  sha_schedule_interleaved.cuh, LeafRecovery.cuh, ParityWindow.cuh, RecoveryConstant.h are
  hash-identical to the frontier; launch geometry identical: BATCH 8M, SLOTS 2, S2_BLOCKS 7,
  TREE_N 128, GT_CHUNKS 15, 64 MiB signed table).
- **No SHA-schedule divergence anywhere**: the entire top-10 field shares identical
  sha_pinsha.cuh/sha_schedule_interleaved.cuh — no interleave lever exists there.
- **The frontier's winning margin is not a mechanism**: it is PR1063 (negY) + a dead flag + a
  favorable draw. The actionable gap between us and the field is exactly one port:
  `QSB_NEG_Y_MAC`, optionally followed by fkiene's `RP_*_F8`/`MUL_SFQ` splices.
