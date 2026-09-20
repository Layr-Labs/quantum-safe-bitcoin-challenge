# Pinning dead ends (this account, 2026-09-20)

Official and public negatives that must not be retried without new evidence.

## Official, this account

| ID | Change | Score | vs 778,624,395 | Lesson |
|---|---|---:|---:|---|
| `f034a9c4` | `QSB_TAIL_TAB=1` + `QSB_SHA_SMEM_W1=1` on 7b0a15b | 750,065,705 | −3.67% | 107,375 verified hits, 89.41 hits/s. Relative drop matches LeaderGPU vs intel-r5 (~3.7%). Does **not** isolate the ST path as slower, and does **not** justify composing SHA into the next arithmetic submit. Keep both flags **0**. Stop pinning SHA work. |
| `0227bc3` | older weaker lineage (2026-09-17) | 679,373,443 | n/a then | Built off the wrong base. Always start from the live promoted source. |
| `2c85ba63` | PR #743 + `QSB_CARRY62` only | cancelled | n/a | Cancelled while validating so the host-gate + C31 bundle could take the slot against the 778 M floor. Do not requeue carry62-only unless this larger bundle is a large regression. |
| `b0fbfb1a` | `QSB_RP_SQR` (square 743 tail + even-fold `f8`) | **789,394,272** | **+0.26%** | 113,063 hits, 94.10 hits/s, `gpu: RTX_4090`. Above the record, short of 796,901,692. 743 ABBA already measured `f8`/`sfc` as losses. **Do not retry `f8`/`sfc`. Do not stack on this tree.** Next base is promoted `dcd0147c`. |
| `4ce3607` | `QSB_FKIENE` funnel-shift digits in registers | **762,268,179** | **−3.39%** vs 789.0 M | 109,172 hits, 90.87 hits/s, RTX 4090. Not 0 hits (packing worked). Modeled +0.62% from fkiene `6fe3a56` did not compose; throughput dropped. **Do not retry FKIENE.** Keep `QSB_FKIENE=0`. |

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

## Research-closed (wait window 2026-09-20, no official score — do not port)

Full writeup: `AIM-1G.md`. Papers in `qsb-research/`. These are not Yukon rejects; they are methods that cannot buy 1.0 G on *this* kernel.

| Method | Why closed |
|---|---|
| VanitySearch / CUDACyclone / SatoshiPool 50 GH/s / Ultrafast “1B sequential” | sequential `+G` or X-set / 6×GLV-X, not recover+SHA |
| gECC batch affine + Montgomery trick (arXiv:2501.03245) | 14.4 M FPMUL/s on A100; would spill the 15-add register chain |
| gECC Montgomery mul + SM2 IADD3 | we already Solinas 73 IMAD; SM2 is a different prime |
| PipeMSM / ICICLE / many-base MSM | wrong problem (many bases) |
| Co-Z 5M+2S | needs shared Z; table is affine |
| Joint GLV / fewer L2 1-D windows | official 674–723 M; zero-doubling comb; L2 math |
| Renes complete mixed 11M | more muls than 7M+2S |
| Longa mbNAF / `dP+Q` | needs doublings |
| RNS, 5×52 radix, CGBN, tensor, Karatsuba | worse mapping on Ada 8×32, or already official-neg |
| safegcd on the hot chain | we do not invert in the 15-add loop |
| Edwards / FourQ / x-only Montgomery | isogeny or wrong curve |

## Still open (not dead)

- Exact host publication gate + C31 (fold / 64-bit split-3p / one-limb K) on top of 743+carry62 — **promoted** `dcd0147c` at 789,011,576.
- `QSB_RP_SQR` — official `b0fbfb1a` **+0.26%, rejected**. Closed.
- `QSB_FKIENE` (funnel-shift digits in registers) — official `4ce3607` **762,268,179 (−3.39%)**. Closed. Do not retry.
- Further *per-candidate* tails: next is `RAW_X=1` on the promoted decoder (`PARITY_SUM` must use the raw x products or RAW_X is a no-op). Then SAS `g8` only, then `cp.async`.
- CUDA graphs / further JIT trim as bundle fillers only.
- Do **not** drop `_ModAddLazyOff` t1 (`mk` is frequently −1).
