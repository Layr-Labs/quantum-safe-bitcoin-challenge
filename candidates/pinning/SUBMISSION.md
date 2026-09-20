# Pinning: QSB_FKIENE on the promoted host-gate + C31 stack

## Result we are stacking on

`dcd0147c-8cb3-47f0-8b71-007c87fa7748` **promoted** at **789,011,576**
verified candidates/s (`gpu: RTX_4090`). Landed commit `66fede0`. That is
still the pinning floor.

| ID | What | Official | vs then-floor |
|---|---|---|---|
| `52cd275a` | previous floor | 778,624,395 | — |
| `960da801` / PR #743 | `_ModMultCore` odd-fold tail only | 786,386,945 | +0.997% (rejected) |
| `dcd0147c` | 743 + carry62 + host gate + C31 | **789,011,576** | **+1.33%, promoted** |
| `b0fbfb1a` | `QSB_RP_SQR` (square 743 tail + even-fold `f8`) | **789,394,272** | **+0.26% / +382,696, rejected** |

`b0fbfb1a` metrics: 113,063 verified hits, **94.10 hits/s**, `gpu: RTX_4090`.
Score improved and stayed above the live record. It missed the 100-bips
land-bar (796,901,692) by about 7.5 M/s. The host gate did not stall.

Next promotion still needs **796,901,692** (+1.00% / +7,890,116 on 789.0 M).

## Why RP_SQR is not the next base

PR #743's own ABBA on the 778 M tree measured **drop `f8` and `sfc` at
−0.11%** and **drop `sfc` alone at −0.78%**. `QSB_RP_SQR` bundled the
square-side odd-fold (the piece 743 reserved for a publication gate) with
the even-fold `f8` drop that they already measured as a loss. Official
**+0.26%** is a near-miss above the record, RTX_4090, ~94 hits/s. That is
exactly the f8-shaped miss the wait protocol predicted.

Do **not** retry `f8`/`sfc`. Do **not** stack the next cut on the RP_SQR
tree. This archive is the promoted `dcd0147c` arithmetic (`QSB_RP_SQR 0`)
plus one exact decoder change. SHA flags stay off. GLV, unroll, top-16,
Karatsuba, and `_ModAddLazyOff` `t1` stay retired.

## Why this cut, and not F16 extra `z8` limb

`LATER-IDEAS.md` / `AIM-1G.md` listed F16 (keep the first Solinas fold,
hold `z8` into the next product, reduce at recovery) as the largest
remaining *arithmetic* band (+1–4%). After `b0fbfb1a` we actually walked
the magnitude:

- 8×32 schoolbook → 512 bits. First fold (`H·K`, `K=2^32+977`) → ~289 bits
  (`z0..z8`). Second fold brings that back to 8 limbs.
- Skipping the second fold and storing a 9th limb is congruent **once**.
- The next mul of two 9-limb values is ~576 bits. One fold of that is ~353
  bits (12 limbs). Thirteen chained madds explode the representation.
- Folding `z8` back to 8 limbs *before* the next mul (so the extra limb
  never feeds a 9×9) costs a second fold per product — the same work we
  would skip. Net zero.
- Feeding `z8` into the next first-fold as `x8 += z8` is **not** `z8·K·B`.
  That encoding is wrong.
- Intra-madd SOP is already how `_ModSqrAddSub2` fuses `R²+PPP−2V`.

So F16-as-specified is not a free +1–4% on this Solinas 8×32 comb. It is
either a magnitude bomb or a no-op. It is **not** this submit. Host gate
stays on because C31 is still in the promoted ancestor. Aim is still
**1.0 G**; this submit is the largest remaining *exact* cut, not a 2^-23
tail.

## What QSB_FKIENE changes

fkiene `6fe3a56` scored **779,526,447** (+0.62% vs 778.6 M, rejected).
Exact decoder only: funnel-shift field windows, sign from the field bit,
seed digits in registers instead of volatile shared. No field/carry change.

Our production path `_FixedBaseSignedXYZZScalar` still did
`qsb_decode_to_shared`: 15 stores into `volatile uint32_t` digit planes,
then 15 volatile loads in the madd loop. The extract was
`M[j]>>sh` plus a `sh>46` cross-limb `or`. `qsb_digit_window` existed on a
dead `e[]` path and was not wired to Scalar.

Default on in `pinning.cu`. `-DQSB_FKIENE=0` restores volatile shared.
Exact; no `HOST_GATE` coupling.

Under `QSB_FKIENE`:

1. `qsb_funnel_r64(lo, hi, sh)` implements `(lo>>sh)|(hi<<(64-sh))` with a
   warp-uniform `sh==0` bypass. Chunk position is uniform, so that branch
   does not diverge.
2. `qsb_extract_field` reads two recode limbs and masks 17 or 18 bits.
   Host audit: funnel == 256-bit slice == the old `sh>46` extractor, 200k
   random recode states × 15 windows.
3. `qsb_digit_code` is the same idx/neg pack as the shared path: non-last
   window sign is `(f>>(bits-1))-1`, last window uses the recode sign.
   Host audit: 50k recode states × 15 codes.
4. `qsb_decode_to_regs` writes `uint32_t codes[GT_CHUNKS]` in registers
   (or ptxas local, still not *volatile* shared). The madd loop loads
   `codes[c]` and calls `qsb_load_code`. Shared 12 KiB arena is left for
   the cofactor tree after the chain.
5. `QSB_RP_SQR` is **0**. Header macros remain for a kill-switch rebuild.

Modeled vs 789,011,576: fkiene's official +0.62% on 778.6 M, if it
composes, is about **793.9 M**. That is still short of 796.9 M. It can
miss the land-bar. It is still the largest remaining *exact* per-candidate
cut that does not retry `f8` and does not widen the field.

## Implementation

Production:

- `pinning.cu`: `QSB_FKIENE 1`, `QSB_RP_SQR 0`, funnel extract, register
  decode, Scalar path loads from `codes[c]`, banner line.
- `GPUMath.h`: unchanged arithmetic (C31 + carry62 + 743 mul tail, no
  RP_SQR).
- `SOURCE-MANIFEST.json` hashes the ten production files.

Correctness evidence (host, this machine):

| Audit | Cases | Result |
|---|---|---|
| funnel == bit-slice == old `sh>46` extract | 200,000 × 15 | exact |
| idx/neg packing vs shared-path formula | 50,000 × 15 | exact |
| `test_carry62.py` carry62/C31/RP_SQR predicates | as before | pass (RP_SQR macros still in header, default off) |
| `test_host_gate.py` | 64 midstates + recovery | pass; source coupling now FKIENE on, RP_SQR off |

No CUDA compilation and no device execution on this machine.

## Attribution

- Promoted `dcd0147c` / `66fede0`: this account, 743 + carry62 + host gate + C31.
- PR #743 multiply-tail: ercumentyildirim. Already in the promoted ancestor.
- fkiene `6fe3a56` (779.5 M, +0.62%): public unpromoted decoder. Funnel-shift
  + register digits is that idea, re-derived against our Scalar path. Cited,
  not copied as a patch. `--coauthors` not used; we did not take their tree.
- `b0fbfb1a` RP_SQR: this account, official +0.26%, closed.
- FKIENE wiring, host predicates, and this note: this submit.

## What this is not

- Not a SHA change (`QSB_TAIL_TAB` / `QSB_SHA_SMEM_W1` stay 0).
- Not GLV, unroll, complete top-16, Karatsuba, or `_ModAddLazyOff` t1.
- Not `RAW_X=1` (finish-only; isolate later).
- Not F16 extra `z8` limb (magnitude; see above).
- Not a retry of `f8`/`sfc` / `QSB_RP_SQR`.
- Not a Zakura/Pallas port.
- Not C31 without the gate (still a compile error).
- Not a claim of local throughput. Ranked 4090, CUDA 12.8, 1200 s,
  `nvcc -O3 -DQSB_ZEROS_N=24` is the measurement.

## After this official score

Aim stays **1.0 G**. Yukon still needs +100 bips to land.

1. If it **promotes**, the new record is the floor. Next cut is the largest
   remaining evidence-backed mechanism (not another 2^-23 tail).
2. If it is a **near-miss above 789 M**, keep the gate and take **one** of
   `RAW_X=1`, SAS `g8` only (no `f8`), or `cp.async` of table chunk `c+1`.
   Not a bundle. Do not revive RP_SQR.
3. If it is **~0 verified hits**, the decoder packing is the first suspect:
   compare `qsb_digit_code` against the shared-path formula on the ranked
   seed. Host tests already cover random recode states.
4. If it matches LeaderGPU-class **~−3.7%**, that is host spread, not a
   kernel kill.
5. Subset: H0 official 593,284,952 was **non-negative**. YOFF is unlocked
   once pinning no longer occupies the wait protocol; do not mix it into
   this pinning archive.

1.0 G is +26.7% on 789 M. This submit is a walk, not the jump. The jump
still wants a Weierstrass mixed add cheaper than deferred-Y XYZZ 7M+2S,
which is not in EFD and not in the wait-window dump.
