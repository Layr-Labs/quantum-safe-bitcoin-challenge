# Pinning dead ends (this account, 2026-09-20)

Official and public negatives that must not be retried without new evidence.

## Official, this account

| ID | Change | Score | vs 778,624,395 | Lesson |
|---|---|---:|---:|---|
| `f034a9c4` | `QSB_TAIL_TAB=1` + `QSB_SHA_SMEM_W1=1` on 7b0a15b | 750,065,705 | −3.67% | 107,375 verified hits, 89.41 hits/s. Relative drop matches LeaderGPU vs intel-r5 (~3.7%). Does **not** isolate the ST path as slower, and does **not** justify composing SHA into the next arithmetic submit. Keep both flags **0**. Stop pinning SHA work. |
| `0227bc3` | older weaker lineage (2026-09-17) | 679,373,443 | n/a then | Built off the wrong base. Always start from the live promoted source. |
| `2c85ba63` | PR #743 + `QSB_CARRY62` only | cancelled | n/a | Cancelled while validating so the host-gate + C31 bundle could take the slot against the 778 M floor. Do not requeue carry62-only unless this larger bundle is a large regression. |

## Official, public, used as evidence

| ID | Change | Score | Lesson |
|---|---|---:|---|
| `960da801` PR #743 | `_ModMultCore` tail truncation only (no squares) | **786,386,945** (+0.997%) | Real. Missed 1% floor 786,410,639 by **23,694/s**. Keep this change. |
| `eb6d9871` | exact four-wave complete top-16 (63 products vs 43) | 769,172,989 (−1.21%) | Dead. Extra products beat the two saved waves. Do not compose. |
| `334b1839` / `5ffaef34` / `0c846cf0` / `4c77f9bd` | grouped / joint GLV | 674.7 / 722.8 / 714.3 / 523.5 M | GLV is closed. 32→64 MiB random-load 1184→610 GiB/s. |
| PR #700 | short-carry top-16 reassociation | local +0.83%, then unsafe | PR #705 counterexample. `QSB_TREE_TOP2` already merged last two levels with original operand order. |
| unroll 2/3/4 | chain-loop unroll | official losses | Fewer dynamic ops, worse I-cache/regs. `QSB_UNROLL=1`. |
| L1 prefetch / PR #714 header / product order | on PR #706 | −1.0 / −2.4 / −0.2% | Do not retry on this runtime. |
| fused prepare+finish | one grid | −9.2% | Keep two-kernel pipeline. |
| Karatsuba | field mul | −6.1% subset | |

## Still open (not dead)

- Exact host publication gate + C31 (fold / 64-bit split-3p / one-limb K) on top of 743+carry62 — **promoted** `dcd0147c` at 789,011,576.
- `QSB_RP_SQR` (743 odd-fold on squares + SAS g8 + even-fold f8) behind the same gate — this submit.
- Further *per-candidate* tails (`RAW_X=1`, remaining census) only after this official score.
- CUDA graphs / further JIT trim as bundle fillers only.
- Do **not** drop `_ModAddLazyOff` t1 (`mk` is frequently −1).
