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
| `ecf55f7d`/`344bfc5` | fkiene first-fold ×977 interleave into dead e/o regs inside `qsb_field_mul` | **−2.7%** official | Longer live ranges cost more than hidden latency on sm_89. |
| `dcaa914b`/`036538d` | odd-product first-fold interleave (same family) | **−0.77%** official | Whole issue-order-interleave family falsified. |
| `ca3c9f45` / `d8c76a02` | recovery-path dual-issue interleave | −2.0% / −3.9% | Same disease — never propose scheduling-interleaves that extend register lifetimes. |
| fkiene RAW_DIFF bundle (`614c5398`, sub `b7b11e5`) | `a−b mod 2^256` as `_ModSub256` + dropped `_ModMult(Q,R)` | **failed** (no score) | Double bug: raw sub is off-by-K per borrow (2^256 ≡ K mod p — mathematically wrong, oracle fails 100%); missing `_ModMult` leaves Y3 unscaled. Verified dead, never port. |

## Still open (not dead)

- Exact host publication gate + C31 (fold / 64-bit split-3p / one-limb K) on top of 743+carry62 — this submit.
- Further *per-candidate* 2^-31-class tails from the carry-chain census, only behind the gate, only after this official score.
- CUDA graphs / further JIT trim as bundle fillers only.

## RAW_DIFF — mathematically dead (falsified 2026-09-22, v11 session)

Replacing `_ModSub256` with raw subtraction mod 2^256 (fkiene pending-bundle
idea, submission `b7b11e5`/`614c5398` → **FAILED**): with `B=2^256`,
`p=B-K`, a borrow wraps by `B` and `B == K (mod p)` — the result is off by
exactly `K` per borrow, NOT congruent. Multiplies/squares/parity windows
cannot repair a wrong representative. CPU oracle anchored to OpenSSL fails
every case under the raw sub and passes under canonical `_ModSub256`. The
pending bundle also deleted `_ModMult(Q,R)` (the `R*(V-X3)` term) in
`_PointAddXYZZT` — independently fatal; its validation indeed failed. A
corrected borrow fold (`borrow -> -K`, ~3 instrs) was evaluated and rejected
as not worth it vs the masked `+p`. **Do not revive.** Recorded in
`GPUMath.h` comment so the idea is not re-attempted.
- Do **not** drop `_ModAddLazyOff` t1 (`mk` is frequently −1).

## Init-cut bundle — measured regression (v12, bdc7e080, 2026-09-22)

`QSB_JIT_WARM` + `QSB_LAD_THREADS` + `QSB_SPOT_DEV` as a bundle: official
**772,365,297** = **-4.06%** vs the identical-kernel v11 (804,814,918).
927.5G candidates vs 966.9G over the same 1200s window. Corrected
decomposition (trajectory audit): -1.83% real self-rep throughput loss +
-2.24% hit-draw luck on seed 1666001017; real loss ~22s-equivalent, not
~51s (raw candidates-vs-score math double-counts the seed draw).

Prime suspect: `CUDA_MODULE_LOADING=EAGER`. Lazy per-function loading only
JIT-compiles kernels actually launched; EAGER compiles the WHOLE module —
and the first real launch still blocks until module load finishes, so the
side-thread warmup cannot hide it. Parallel OpenSSL ladders and 64B device
spot-reads were not isolated (each isolation costs a ~2h submission slot).

Rule of thumb learned: init work is only worth cutting if it is NOT on the
first-launch critical path, and JIT-hiding only works when the hidden
compile is smaller than the host work it overlaps. **Do not re-enable as a
bundle.** Individual revival only with a fresh hypothesis + cheap evidence.
