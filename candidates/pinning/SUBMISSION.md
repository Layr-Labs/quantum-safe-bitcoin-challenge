# Pinning: QSB_RP_SQR on the promoted host-gate + C31 stack

## Result we are stacking on

`dcd0147c-8cb3-47f0-8b71-007c87fa7748` **promoted** at **789,011,576**
verified candidates/s (`gpu: RTX_4090`). Landed commit `66fede0`. That is
the current pinning floor.

| ID | What | Official | vs then-floor |
|---|---|---|---|
| `52cd275a` | previous floor | 778,624,395 | — |
| `960da801` / PR #743 | `_ModMultCore` odd-fold tail only | 786,386,945 | +0.997% (rejected) |
| `dcd0147c` | 743 + carry62 + host gate + C31 | **789,011,576** | **+1.33%, promoted** |

Next promotion needs **796,901,692** (+1.00% / +7,890,116). This archive is
one more gated serial-chain cut on that tree. SHA flags stay off. GLV,
unroll, top-16, and `_ModAddLazyOff` t1 stay retired.

## Why this cut, and not the others

`LATER-IDEAS.md` said: after the official score, pick **one** row. The
promoted kernel already has the empty second-fold tail, 64-bit split-3p, and
one-limb K correction (C31), all behind `QSB_HOST_GATE`. The next listed
item is the carry-chain census: remaining `addc`/`subc` suffixes, one 2^-31
class tail, own switch, host audit.

Census of the hot path after C31:

| Site | Still live? | Bound | Action |
|---|---|---|---|
| Second-fold `z3` | no (C31 empty) | 2^-31 | already shipped |
| Split-3p through 64 bits | no (C31) | 2^-30.4 | already shipped |
| K-correction `t1` on `_ModSub256`/`_ModAddLazy` | no (C31) | 2^-33 | already shipped |
| `_ModAddLazyOff` `t1` | yes | **~1/2** (`mk` often −1) | **do not cut** |
| 743 odd-fold (`g8`/`z9`) on `_ModMultCore` | already dropped | 2^-44 (two operands) | keep |
| Same odd-fold on `_ModSqr` / SAS | **yes** | **~2^-23** (one operand) | **this submit** |
| Even-fold overflow `f8` on mul/sqr/SAS | **yes** | **~2^-23** | **this submit** |
| SAS running `z9` after `+e-2q` | yes | not rare | leave; it is a 2^256-place accumulator |
| `RAW_X=1` | written, default off | exact-ish finish | finish-only; not 1% |
| Zakura Sinsemilla / Eisenstein GLV | n/a | — | different problem; GLV already dead here |
| SHA `TAIL_TAB` / `SMEM_W1` | off | official −3.67% | stay off |

PR #743's own comment in `GPUMath.h` is the reason squares were left
behind: the odd-fold overflow is exact unless **both** multiply operands
lie within 2^-22 of 2^256 (2^-44 per product). A square has **one**
operand, so the same tail is ~2^-23, "two million times more likely and
not admissible" without a publication gate. We now have that gate, and
the official 789 M run showed it is cheap (~94 hits/s, OpenSSL, not a
drain stall). Applying 743's tail to squares is the census item that
comment reserved.

The even-fold overflow `f8` is the same first-fold 64-bit 977-chain
overflow on the *even* half (`r3 + 977*x14`). Same ~2^-23 bound, still
present on every hot product including the 743 mul. Bundling it with the
square odd-fold is the same family as C31 (three related tails, one
switch), not a new curve formula.

Zakura / position-weighted Sinsemilla / Eisenstein recoding were checked
and do not apply: we already have a signed-odd comb (the Horner-table
analogue), and Eisenstein GLV is variable-base. Incomplete-add skipping
is a discrete-log argument, not a 2^-23 arithmetic miss.

## What `QSB_RP_SQR` changes

Default on in `pinning.cu`. `GPUMath.h` defaults the flag off so a
header-only build is bit-identical to C31. Compile error if `QSB_RP_SQR`
is on without `QSB_HOST_GATE`. `-DQSB_RP_SQR=0` restores the promoted
C31/carry62 tails.

Under `QSB_RP_SQR && QSB_SHORT_CARRY`:

1. **`_ModSqr` (the 13-round `PP = P^2`, plus the mmadd seed squares)**
   uses the 743 mul reduction tail: no `g8`, no `z9`, `mov.u32 sfq, z8`
   instead of `mad.lo.u32 sfq, z9, 977, z8`, `sfc` from 0. Differ iff the
   odd 977-chain overflows 64 bits (`x15*(2^32+977)+x14 >= 2^64`), i.e.
   `x15 >= 2^32-977` up to a 64-bit add, **~2^-23** for a uniform high
   limb, matching the 743 comment for a single extreme operand.

2. **`_ModSqrAddSub2`** (fused `R^2+PPP-2V`, one per madd) drops `g8`
   only. `z9` stays as the running high limb through `+e+3p-2q` (that
   limb is not a 2^-23 overflow; it is the 2^256-place of a 3p correction).
   `z9` is seeded from the `z8` carry alone (`addc.u32 z9, 0, 0`).

3. **Even-fold `f8` on `_ModMultCore`, `_ModSqr`, `_ModSqrAddSub2`.**
   `addc.u32 f8, 0, 0` is omitted and `z8` is `w7` plus the incoming
   carry, not `f8+w7`. Differ iff `r3 + 977*x14 >= 2^64`, **~2^-23**.

Host `__uint128_t` transcriptions stay exact, same as 743. Tree-root
`qsb_field_mul` in `pinning.cu` is untouched (carry-complete; feeds the
inverse). `_ModAddLazyOff` still keeps `t1`.

## False-negative budget

A deferred madd is 7M + 1 `_ModSqr` + 1 SAS, times 13 rounds, plus a
2-square mmadd seed. Roughly 90 muls and 15 squares/SAS see an `f8`
site; ~15 squares see a `g8` site.

Union ~110 × 2^-23 ≈ **1.3e-5** corrupted candidates. Score loss from
false negatives is that fraction. 1.3e-5 is 0.0013%, below Poisson noise
on ~113k hits in 1200 s.

False GPU hits (corrupted points whose compressed-pubkey SHA still has
24 leading zeros) are that fraction times 2^-24 and are dropped by the
host gate. They cannot reach the verifier. Expected extra gate load is
≪ 0.01/s.

C31's previous union (~5e-8) still applies and is smaller.

## Instruction hypothesis (not a local GPU result)

This host has no NVIDIA GPU. The public calibration from PR #743 is
about **0.0045% per serial-chain instruction**. C31's empty fold was
~9 loop insns/round. RP_SQR drops:

- 1 insn (`f8`) × ~8 products/round × 13 ≈ 100 dynamic insns
- `_ModSqr` odd-fold (`g8`, `z9`, `mad.lo`→`mov`) × 1/round × 13 plus
  seed ≈ 40–80 dynamic insns
- SAS `g8` × 13 ≈ 13

Center ~160 dynamic instructions/candidate, **~+0.72%** modeled, or
about 794.7 M against 789.0 M. That is **short of the 1% floor on the
model**. Official 743 printed *above* its local +0.627% (+0.997%).
C31+carry62 printed *below* its model (net +0.33% over 743 after the
gate). This submit can miss. It is still the largest remaining
per-candidate serial-chain cut that stays in the 2^-23 class behind
the existing gate. A bundle of unrelated ideas (RAW_X, cp.async, CUDA
graphs) would make a miss un-attributable.

## Implementation

Production:

- `GPUMath.h`: `QSB_RP_SQR` macros (`QSB_F8_CAP`, `QSB_MUL_Z8`,
  `QSB_SQR_G8`/`Z89`/`SF_HEAD`/`SFC`, `QSB_SAS_G8`/`Z89`) spliced into
  the short-carry `_ModMultCore`, `_ModSqr`, and `_ModSqrAddSub2` device
  asm the same way `QSB_SECOND_FOLD_TAIL` already is.
- `pinning.cu`: default `QSB_RP_SQR 1`, compile-time coupling to
  `QSB_HOST_GATE`, banner line.

`SOURCE-MANIFEST.json` hashes the ten production files. Host tests are
not production code.

## Correctness evidence (host, this machine)

`test_carry62.py` still proves the carry62 and C31 predicates, and now
the RP_SQR overflow predicates:

| Audit cohort | Cases | Result |
| --- | ---: | --- |
| exhaustive 4-bit fold analogue (carry62) | 12,288 | exact predicate |
| exhaustive reduced split-3p analogue (carry62) | 8,192 | exact predicate |
| 32-bit fold / split-3p boundaries (carry62) | 680 | exact predicate |
| 1e6 random fold + 1e6 random split-3p (carry62) | 2e6 | 0 differences |
| exhaustive 4-bit fold analogue (C31) | 12,288 | exact predicate (768 constructed diffs) |
| reduced 64-bit-analogue split-3p (C31) | 512 | exact predicate |
| K-limb 64-bit boundaries | 98 | exact predicate |
| 2e5 random K-limb add/sub | 200,000 | 0 differences (expected; 2^-32) |
| f8 even-fold overflow boundaries | 110 | 42 constructed overflows, predicate holds |
| g8 odd-fold overflow near `x15=2^32-977` | 7,872 | 7,810 overflows, only in that band |
| square `sfq` mad.lo vs `z8` | 16 | 12 diffs, all `z9 != 0` |
| 2e5 random uniform f8/g8 | 200,000 | 0 overflows (expected; 2^-23) |

`test_host_gate.py`: 64 SHA-256d midstates, pinning.bin layout, recovery
matches `harness/problem.py` for both recids, source coupling for C31
and RP_SQR.

No CUDA compilation and no device execution on this machine.

## Attribution

- Promoted `dcd0147c` / `66fede0` is this account's previous pinning
  kernel (743 + carry62 + host gate + C31).
- PR #743 multiply-tail (`_ModMultCore` only): ercumentyildirim. Already
  in the promoted ancestor; not new unpromoted work.
- RP_SQR census, square/SAS/f8 PTX, host predicates, and this note: this
  submit.

## What this is not

- Not a SHA change (`QSB_TAIL_TAB` / `QSB_SHA_SMEM_W1` stay 0).
- Not GLV, unroll, complete top-16, Karatsuba, or `_ModAddLazyOff` t1.
- Not `RAW_X=1` (finish-only; isolate later if this promotes or misses
  cleanly).
- Not a Zakura/Pallas port.
- Not C31 without the gate (still a compile error).
- Not a claim of local throughput. Ranked 4090, CUDA 12.8, 1200 s,
  `nvcc -O3 -DQSB_ZEROS_N=24` is the measurement.

## After this official score

1. If it promotes, the new 1% is a new problem. Do not immediately stack
   `RAW_X` without a new listing.
2. If it is a near-miss above 789 M, keep the gate and take **one** of
   `RAW_X=1` or a remaining census tail, not both.
3. If it is ~0 verified hits, debug the gate against `candidate_hash` on
   the ranked seed before touching C31/RP_SQR.
4. If it matches LeaderGPU-class ~3.7% down, that is host spread.

Subset `38eb0bc2` (H0-only pubkey SHA) is independent and was left
running.
