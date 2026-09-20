# Aim: 1.0 G verified cand/s (research log)

North star for every pinning submit after this file exists: **1,000,000,000** verified candidates/s on one RTX 4090.

Yukon still only needs +100 bips to *land* a new record. We do not lower the aim to 1%. Each submit is **one** largest honest remaining cut toward 1.0 G. Small exact leftovers are fillers, not the plan.

**Do not implement while `b0fbfb1a` is validating.** This file is the wait-window log so we do not re-research dead methods between submits.

Live floor (2026-09-20): **789,011,576** (`dcd0147c`, commit `66fede0`).  
1.0 G is **+26.7%** / +210,988,424/s. Next Yukon land-bar: **796,901,692**.

Full papers (read-only, do not execute):  
`/Users/harshghodkar/Documents/batcave/qsb-research/`  
Wave-1 note `notes/1B-PATH.md`, wave-2 `notes/1B-PATH-2.md`.

---

## How we get closer

The kernel is already the right *kind* of algorithm: signed-odd 15-chunk comb, zero doublings, 64 MiB table in 72 MiB L2, deferred-Y XYZZ **7M+2S**, Solinas 8×32 at **73 IMAD.WIDE** per mul, host gate on hits.

Public GH/s numbers are a **different problem** (sequential `+G`, X-set membership, 6×GLV-X). They do not transfer. 1.0 G here is ~32% cheaper EC if the chain is ~80% of time: a mixed add around **5M+2S**, or ~11 windows that still fit L2. Neither is on the shelf.

So the grind is:

1. Submit the biggest *on-chain* cut we actually have (right now: **F16 extra `z8` limb**, catalog +1–4%).
2. Log the official score. Keep what worked. Close what did not.
3. Spend the wait window on papers/math, not kernel edits.
4. Repeat. Aim stays 1.0 G even when a given submit only moves tens of millions.

Honest stacked ceiling from the papers we have: **~0.85–0.95 G**. That is closer. 1.0 G is the aim anyway — it keeps us from settling for another 2^-23 fold tail. If a new Weierstrass mixed add below 7M+2S appears, it becomes the next switch immediately.

---

## Where we are (kernel facts)

| Piece | Production |
|---|---|
| Scalar | `Q = z·A ± U2R`, `z = SHA256d(preimage)`, r,s pinned |
| Comb | 15 signed-odd windows, 64 B/point, 64 MiB, L2-resident |
| Add | EFD XYZZ `madd-2008-s` 8M+2S, Y deferred → **7M+2S** |
| Field | 8×32 schoolbook + Solinas `2^256 ≡ 2^32+977`. Mul **73 IMAD**, sqr **45 IMAD** (CE nvcc 12.9 sm_89) |
| Gate | `QSB_HOST_GATE` OpenSSL recover+SHA before publish (~94 hits/s on the promoted and RP_SQR runs) |
| Decoder | `QSB_FKIENE` funnel-shift + register codes (this submit). `QSB_RP_SQR` off. |

743 ruler: a loop insn is ~0.0045% **only on the reduction’s dependent chain**. Off-chain rare-carry tails can lose (`f8` −0.11%, `sfc` −0.78%). Do not retry those.

---

## Research closed (do not port / do not retry)

Logged 2026-09-20. No kernel work. Sources in `qsb-research/`.

### Wrong problem (GH/s that is not recover+SHA)

| Source | Claimed rate | Why it is not us |
|---|---|---|
| VanitySearch-Bitcrack README | 6.9 Gkeys/s 4090 | sequential `+G`, batch inverse. Locktime+1 is a random-oracle step in z. |
| CUDACyclone README | 6.1–6.5 Gkeys/s 4090 | same class |
| SatoshiPool 253–256 scanner | 50 GH/s “effective” | 1 affine add → 6 X via `{id,−id,λ,−λ,λ²,−λ²}`, **compare X to a set**, no SHA. `SHA(βX)` uncorrelated with `SHA(X)` |
| UltrafastSecp256k1 | 4.88 M ECDSA sign/s; “1B sequential” | sign ≠ recover+SHA; sequential scan |

Do not clone or run those trees.

### Wrong EC layer (looks +30%, would drop us to tens of M)

| Method | Paper | Blackboard | Why closed here |
|---|---|---|---|
| Batch affine + Montgomery trick (gECC) | arXiv:2501.03245, TACO 2025. PDF `papers/1b-path2/gec-arxiv-2501.03245.pdf`. README only `github/gecc-readme/` | affine add ~6M amortized vs Jacobian 11–16M | **gECC FPMUL = 14.4 M/s on A100** vs RapidEC ~3.9 M. We are **789 M** because 15 adds stay in **registers**. GAS invert needs 15 memory round-trips of every accumulator + 96 MB intermediates per 2^20 adds. 743 says we are IMAD-chain bound, not memory bound. |
| RapidEC Jacobian/NAF | SC'22, `github/rapidec-readme/` | GPU ECDSA on SM2 | Their baseline, not a comb. Do not build. |
| PipeMSM / ICICLE / ZPrize MSM | eprint 2022/999 | `Σ a_i P_i` many bases | We have many scalars, **one** base A. Pippenger replaces mixed adds with more full adds. |
| Co-Z ZADD 5M+2S | eprint 2010/309, 2011/430 | shared Z | Table points are affine (Z=1); scaling them up costs the muls you saved. Euclidean chains have more adds than 15 windows. |
| Joint GLV / 6-keys-per-add | official 674–723 M | split 256→128+128 | Add-count win of GLV is halving **doublings**. This chain has **zero doublings**. 8-bit joint digits → 16 windows, not fewer than 15. 2D gather 1184→610 GiB/s. |
| Fewer 1-D windows in L2 | table-size math | 11×23-bit ≈ 2.8 GB | 72 MiB L2 / 64 B ≈ 1.13 M points. 15×2^16 odd = 64 MiB, covers 256 bits. `w=20` already lost ~20%. |
| Edwards / Hisil / FourQ / x-only Montgomery | eprint 2007/414, P24–P26 | faster add on another curve | Isogeny in and out per candidate, or a curve with 2-torsion (secp256k1 has prime order). |
| Sequential `+G` / Gray-code locktime | vanity | 1 add / candidate | z = SHA256d. No differential walk. |
| Skip SHA / CRC / H0 as a 27% well | subset H0 official **593,284,952** (+0.77%) | 24-bit SHA word 0 of compress(Q) | Wrong bit of X randomizes the hash. H0 is exact 24-bit; it is not 27%. Unlocks YOFF, does not unlock 1.0 G. |

### Wrong field / formula (more muls, or already ours)

| Method | Paper | Why closed |
|---|---|---|
| gECC Montgomery mul + SM2 IADD3 reduce | same gECC paper, Table 3 | Theoretical Montgomery 136 IMAD, CGBN-1 measured 185. We already do **Solinas 73 IMAD**. Even/odd carry split already in `_ModMultCore`. SM2 `q_inv=1` IADD3 is a **different prime** (`2^256−2^224−2^96+2^64−1` vs `2^256−2^32−977`). |
| libsecp256k1 5×52 / 10×26 radix | bitcoin-core field | Steal **magnitude**, not radix. 5×52 is 25×64-bit muls on a 32-bit GPU. Worse than 8×32. |
| Renes–Costello–Batina complete mixed, a=0 | eprint 2016/177 | **11M + 2·m_{3b}**. Exception-free. We are not in the P=±Q case (2^{-256}). Worse. |
| Longa–Miri `dP+Q` / mbNAF | eprint 2008/051, 2008/052 | Composite ops and multibase need **doublings**. Comb has zero. |
| RNS ECC on GPU | Antão 2011; eprint 2025/1068 | Wins at 512+ bits. 256-bit 8-limb positional + Solinas is the right mapping. Conversion in/out per candidate eats any win. |
| Tensor / INT8 / 4×64 / CGBN / Karatsuba | official Karatsuba −6.1%; CGBN 4 threads | Ada IMAD.WIDE *is* the 8×32 lattice. Tensor is FP16/INT8. CGBN SHFL ~5 cy. |
| safegcd on the hot chain | eprint 2019/266; Zakura Pasta port | 4–5× vs Fermat for **one inverse**. Hot chain does not invert. Finish / host-gate only. Gate already 94 hits/s, not stalling. |
| Gueron–Krasnov P-256 | eprint 2013/816 | CPU 4×64 Solinas analog. We already Solinas 8×32. |
| SHA TAB / SMEM | `f034a9c4` 750.1 M (−3.67%) | Pinning SHA retired. |

---

## Research open (ladder toward 1.0 G)

One switch per submit. Host gate stays on for any 2^{-31}-class arithmetic.

| Order | Cut | Expected vs 789 M | Notes |
|---|---|---|---|
| **0 (this submit)** | **FKIENE decoder** | ~+0.62% official on 778 M → ~793.9 M modeled | Funnel-shift field windows, 15 codes in registers, no volatile shared. Exact. `QSB_RP_SQR=0`. May miss the 1% land-bar; largest remaining *exact* cut. |
| **1** | **F16 extra `z8` limb** | **not the +1–4% we hoped** | 8×32 first fold → 9 limbs (~289 bits). Next 9×9 first-fold → ~12 limbs. Thirteen madds explode. Folding before the next mul costs the second fold we would skip. `x8 += z8` is not `z8·K·B`. SOP is already `_ModSqrAddSub2`. **Do not ship extra-limb F16 until a stable 9-limb schedule exists.** Longa/Scott/libsecp *magnitude* is still the right family to watch. |
| 2 | Isolated `RAW_X=1` | 0–1% | Only if F16 is the wrong shape. Finish-only. Exact-arithmetic source (gate on, C31 off) so a miss is attributable. |
| 3 | gECC-style IMAD **bank/predicate hygiene** on the *existing* 73-IMAD mul | 0–3% | `cuobjdump` vs the CE listing first. Do **not** rewrite as Montgomery. 743 forbids adding off-chain “helpful” tails. |
| 4 | fkiene decoder | ~+0.6% official on 778 M | Exact. Funnel-shift + seed digits in regs. Compose after a real arithmetic promote. |
| 5 | SAS `g8` only | ~+0.07% | Last 743-admitted square tail. |
| 6 | `cp.async` next table chunk | 0–2% | Exact. Not L1 prefetch (that lost). |
| — | Kernel fission SHA ‖ chain | −5 to +8% | Occupancy experiment. 743 says chain-bound. |
| **watch** | New Weierstrass mixed add **< 7M+2S** with affine Q | **+10–25% → this is the 1.0 G cut** | Not in EFD, not in this dump. Do not invent one. Becomes #1 the day a paper lists it. |
| subset | YOFF after pinning is terminal | H0 already **+0.77% official, non-negative** | Cross-track. Does not move pinning 1.0 G. |

Do **not** stack F16 with a census tail in one submit.

---

## Papers on disk (read-only)

Wave 1 `qsb-research/papers/1b-path/`: EFD XYZZ, Bernstein–Lange 2007/414, co-Z 2010/309 and 2011/430, SatoshiPool HTML, vanity READMEs.

Wave 2 `qsb-research/papers/1b-path2/`:

| File | Use |
|---|---|
| `gec-arxiv-2501.03245.pdf` | gECC. Closed as EC layer. Open as IMAD-hygiene compose. |
| `batch-affine-vs-mixed.html` | 6M Weierstrass vs 14M Edwards. |
| `pipemsm-2022-999.pdf` | MSM, wrong problem. |
| `longa-2010-335-incomplete-reduction.pdf` | F16 family. |
| `longa-2022-367-sum-of-products.pdf` | F16 family (AB±CD). |
| `scott-2017-437-slothful.pdf` | F16 family (bounded extra limb, not “never reduce”). |
| `longa-miri-2008-051-composite.pdf` | Closed (doublings). |
| `longa-2008-052-multibase.pdf` | Closed (doublings). |
| `renes-2016-177-complete.pdf` | Closed (11M mixed). |
| `gueron-krasnov-2013-816.pdf` | CPU analog, already have. |
| `bernstein-yang-2019-266-safegcd.pdf` | Finish invert only. |
| `solinas-1999-corr99-39.pdf` | GMN; we already implement secp256k1. |
| `zakura-safegcd.html` | Finish invert only. |

GitHub READMEs only (DO-NOT-EXECUTE): `qsb-research/github/{gecc-readme,rapidec-readme,vanitysearch-bitcrack-readme,cudacyclone-readme}/`.

Not cloned: full gECC, RapidEC, UltrafastSecp256k1, VanitySearch source, CUDACyclone source, SatoshiPool scanner, Quantus miner, Zakura node.

---

## After `b0fbfb1a` is terminal

1. Record official score + `gpu` (Issue #505).
2. If ~0 hits: debug gate vs `candidate_hash`. Do not publish C31 ungated.
3. If ~−3.7%: LeaderGPU draw, not a kernel kill.
4. If f8/sfc shaped loss: **do not retry f8/sfc**. Proceed to F16.
5. Implement **one** switch: F16 extra `z8` limb, host-audit, submit ≥5 KiB note.
6. Return to wait. Aim is still 1.0 G.

Subset YOFF only after pinning is terminal (H0 already non-negative).
