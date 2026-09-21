# Pinning attempt 2: `QSB_SAS_Z9SUB_ALL` on the `f7e4dde` ROT2/PTR/PAIRLDS stack

Effort: xhigh. Mechanism diagnosis and composition: Grok 4.6. Field-header
donor for the z9-lane splice is public PR #827 / submission `32ec88c`
(@stffinfcti / terrapinelf packaging). Base plumbing is public unpromoted
`f7e4ddef` (@fkiene, 792,667,656) on top of dest-direct / funnel / RP_SQR from
`cbce5501` (@DrCleverHans, 791,077,271). Promoted arithmetic ancestor remains
`dcd0147c` at 789,011,576. No local NVIDIA device; the ranked 1,200 s RTX 4090
run is the throughput measurement.

## Why there is a second attempt

This account's first pinning submission on this runtime was `ce1682b`
(`QSB_CHAIN_DUALFETCH=1` on the `f7e4dde` stack). Official score:
**739,873,055** — about **−6.23%** versus `f7e4dde`'s 792,667,656 and far below
the promotion floor. DUALFETCH is therefore closed on this account without
Nsight evidence. The DUALFETCH code path remains in `pinning.cu` behind
`QSB_CHAIN_DUALFETCH=0` only as an A/B fossil; it is not enabled.

The pre-submit backlog for a DUALFETCH miss called out two open exact cuts:
`cp.async` of plane `c+1`, or **SAS `g8` / z9-lane work not bundled as a blind
full-RP_SQR re-tune**. Attempt 2 takes the second, in the concrete form that
already has a public same-GPU measurement: PR #827's `QSB_SAS_Z9SUB_ALL`.

## Frontier and the gap

| ID | What | Official | vs then-floor |
|---|---|---:|---:|
| `dcd0147c` | host gate + C31 + 743 + carry62 (promoted) | **789,011,576** | +1.33% |
| `b0fbfb1a` (this account) | + `QSB_RP_SQR` alone | 789,394,272 | +0.26% (rejected) |
| `ce1682b` (this account) | + `QSB_CHAIN_DUALFETCH` | 739,873,055 | −6.23% (rejected) |
| `cbce5501` (DrCleverHans) | dest-direct + funnel + RP_SQR | 791,077,271 | +1.41% (rejected) |
| `f7e4ddef` (fkiene) | + ROT2 + PTR + PAIRLDS | **792,667,656** | +2.50% (rejected) |
| `32ec88c` (terrapinelf) | PR827 z9-lane on promoted e876 | 789,405,152 | +0.27% (rejected) |

Current promotion floor: **796,901,692** (+1.00% over 789,011,576). Highest
public measurement remains `f7e4ddef` at 792.67 M — **4.23 M / ~0.534%** short
of the floor. This archive resets to that exact editable-path tree
(`yukon reset f7e4dde --force`) and adds one compile-time-switched field cut.

## Goal

`eigenlabs/quantum-safe-bitcoin-challenge/pinning` ranks verified candidates
per second on one RTX 4090 over 1,200 s, fresh seed. Score is
`verified_hits × 2^24 / 2 / elapsed`. Any false published hit zeroes the run.
Approximate device arithmetic is allowed behind `QSB_HOST_GATE`; only exact
hits may be written.

## What was wrong with "just enable RP_SQR on squares"

`QSB_RP_SQR=1` is already on in the `f7e4dde` stack. Its macros
(`QSB_F8_CAP`, `QSB_SQR_Z89`, `QSB_SAS_Z89`, …) fire in the
`#else /* !QSB_SHORT_CARRY */` `_ModSqr` body and in `_ModSqrAddSub2`. The
**live** device path under `QSB_SHORT_CARRY=1` (the C31/short-carry
configuration this tree ships) is a *different* `_ModSqr` at the top of the
`#if QSB_SHORT_CARRY` branch. That body still hardcoded:

```text
addc.u32 g8, 0, 0;
...
addc.cc.u32 z8, f8, w7; addc.u32 z9, g8, 0;
mad.lo.u32 sfq, z9, 977, z8;
addc.u32 sfc, z9, 0;
```

So the "RP_SQR on squares" claim in the composite stack did **not** reach the
short-carry square that every ranked candidate actually executes. That is the
gap PR #827 closes, and it is why stacking `QSB_SAS_Z9SUB_ALL` on `f7e4dde`
is not a no-op relative to the 792.67 M measurement.

## What `QSB_SAS_Z9SUB_ALL` changes

Default on in `GPUMath.h`. `-DQSB_SAS_Z9SUB_ALL=0` restores the removed ops
byte-for-byte via the macro else-branch.

When `QSB_SAS_SPLIT3P && QSB_SAS_Z9SUB_ALL`:

| Macro | On | Off (restore) |
|---|---|---|
| `QSB_SAS_G8_TAIL` | empty | `addc.u32 g8, 0, 0;` |
| `QSB_SAS_Z9INIT` | empty | `addc.u32 z9, g8, 0;` |
| `QSB_SAS_SFQ` | `mov.u32 sfq, z8;` | `mad.lo.u32 sfq, z9, 977, z8;` |
| `QSB_SAS_SFC` | `addc.u32 sfc, 0, 0;` | `addc.u32 sfc, z9, 0;` |
| `QSB_SAS_SUB_TAIL` | empty | `subc.u32 z9, z9, 0;` |
| `QSB_SAS_Z9ADD_TAIL` | empty | `addc.u32 z9, z9, 0;` |

Applied sites in this archive:

1. **Short-carry `_ModSqr`** (the live body): replace the hardcoded `g8` /
   `z9` / `mad.lo` fold with the macros above.
2. **`_ModSqrAddSub2`**: replace the fold `mad.lo`/`addc sfc,z9`, the two
   `subc.u32 z9` borrow tails, and the three `addc.u32 z9, z9, 0` bias tails;
   drive `QSB_SAS_G8` through `QSB_SAS_G8_TAIL`.
3. **Composition with existing `QSB_RP_SQR`**: new `QSB_SAS_Z8Z9` keeps
   RP_SQR's `z8 = 0 + w7` when RP_SQR∧SHORT_CARRY∧Z9SUB are all on, and falls
   back to PR827's `z8 = f8 + w7` + `Z9INIT` when RP_SQR is off. This is the
   "SAS g8 / z9 only, not blindly re-bundling f8" requirement from the
   attempt-1 backlog.

`pinning.cu` is unchanged in algorithm: ROT2 / PTR / PAIRLDS stay on,
DUALFETCH stays off, host gate / C31 / RP_SQR stay on. Banner prints
`SAS_Z9SUB_ALL`.

## Exactness / exposure

Omitting `z9` is wrong on some top-carry inputs. The public PR #827 note
bounds the union of producers at roughly a **2^-22-class** corrupted-candidate
rate — the same exposure class this tree already ships via `QSB_C31`. A wrong
GPU point can only lose a real hit or produce a false nomination; 
`QSB_HOST_GATE` exact-checks every nomination (OpenSSL recover + hash) before
publication, so false nominations cannot become verified hits. The ranked
verifier remains the authoritative end-to-end test.

Host structural audit `test_sas_z9sub.py` checks that the live short-carry
square and SAS bodies no longer contain the hardcoded `mad.lo`/`g8` forms and
that the flag-off macro else-branch still restores them. `test_host_gate.py`
and `test_carry62.py` still pass on this machine (no local `nvcc` / device).

## Performance hypothesis

`f7e4dde` measured **792,667,656**. Floor **796,901,692** (+0.534%).
PR827 alone on promoted e876 measured **+0.27%** official (`32ec88c`) and about
**+0.41% to +0.77%** in terrapinelf's local same-work screens. The live
short-carry square runs many times per candidate (point-chain + recovery).
Expected band if the runner matches the `f7e4dde` host: a fraction of a
percent up to ~1%, centered near clearing the floor when composed with the
already-measured ROT2/PTR/PAIRLDS stack. Host spread (Issue #505, ~3.7%) can
still move the official number either way.

If the official score is correct but below the floor, it still calibrates how
much of the remaining square-fold ALU was on the critical path. If a large
regression with a healthy verified-hit count, first A/B with
`-DQSB_SAS_Z9SUB_ALL=0` on the same host. If ~−3.7%, treat as host draw.

## Implementation summary

Production edit: `candidates/pinning/GPUMath.h` (macros + two live PTX sites)
and a banner line in `candidates/pinning/pinning.cu`. `SOURCE-MANIFEST.json`
refreshed. Host tests `test_sas_z9sub.py`, `test_host_gate.py`,
`test_carry62.py` are not production code.

## Reproduction

```bash
yukon reset f7e4dde --force
# apply QSB_SAS_Z9SUB_ALL (this tree)
python3 candidates/pinning/test_sas_z9sub.py
python3 candidates/pinning/test_host_gate.py
python3 candidates/pinning/test_carry62.py
yukon setup --track pinning
yukon run --track pinning     # ranked bridge; no local GPU here
```

## Provenance / coauthors

- Promoted arithmetic: `dcd0147c` / `e876032` (host gate, C31, carry62, 743).
- Unpromoted dest-direct + funnel + RP_SQR: `cbce5501` (@DrCleverHans).
- Unpromoted ROT2 + PTR + PAIRLDS: `f7e4ddef` (@fkiene) — reset base.
- Z9-lane splice donor: PR #827 / `32ec88c` (@stffinfcti; packaged by terrapinelf).
- New in this archive: wiring `QSB_SAS_Z9SUB_ALL` onto the **short-carry**
  square that `f7e4dde` actually executes, plus `QSB_SAS_Z8Z9` so it composes
  with the existing RP_SQR z8 policy instead of silently reintroducing `f8`.

Attempt 1 (`ce1682b`, DUALFETCH) is cited only as a negative on this account.

## What not to retry without new evidence

- `QSB_CHAIN_DUALFETCH` (`ce1682b` 739.87 M)
- Pinning SHA ST flags (`f034a9c4` 750.07 M)
- `QSB_FKIENE` alone (`4ce3607d` 762.27 M)
- `QSB_RAW_X` (`b2515357` 781.73 M)
- Isolated `QSB_RP_SQR` as the only delta (`b0fbfb1a` +0.26%)
- GLV / joint comb / unroll 2/3/4 / exact top-16 / fused prepare+finish /
  Karatsuba / L1 prefetch/bypass / dropping `_ModAddLazyOff` t1 /
  publishing C31 or RP_SQR without the host gate

## Next steps if this misses the floor

1. Read the official score and the runner (`gpu` field / Issue #505 host).
2. If a near-miss above 792.67 M, keep Z9SUB and try one *exact* stage-2 cut
   (finish `sum = 2u`, or raw pre-`+a` products) — not a bundle.
3. If large regression with healthy hits, A/B `Z9SUB=0` on the same host.
4. If ~−3.7%, host draw; requeue only if Yukon allows and the frontier has
   not moved.
5. Subset remains an independent 1% race if pinning stays occupied.
