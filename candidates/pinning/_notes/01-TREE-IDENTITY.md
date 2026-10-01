# 01 — Tree identity, and the correction that reframes the whole task

Written 2026-09-30. All claims below are from files I read directly this
session, with file:line citations. I did not run anything: this host has no
CUDA and I have no shell. Where I rely on a subagent's read, I say so.

---

## 1. The brief's baseline is stale. The tree is NOT v20.

`BRIEF-QSB.md:12` says "your best official: 882,096,418 (submission 4fe6a084,
v20)" and `:14-15` states a 15.4% gap to the 1,008,206,828 record.

The working copy is **not** the v20 package. Direct evidence from
`candidates/pinning/pinning.cu`:

| line | content | meaning |
|---|---|---|
| 1 | `/* l2state variant fkF20c8 + split retry */` | tree is an l2state lineage |
| 2-4 | `QSB_SUBPIPE 131072`, `QSB_SUBRING 6`, `QSB_ROOT_FUSED 1` | ercumentyildirim PR #1788 green-context sub-batch pipeline (see `SUBMISSION-GREEN-PRED.md:75-93`) |
| 6 | `QSB_PERSIST_WINDOW_CAP (42u<<20) /* HY6 arm (after ercumentyildirim #1892, cefika 482a55e6) */` | **descends from cefika 482a55e6** |
| 8 | `QSB_L2STATE 1033 /* from PR #1891 */` | DPZZxlz PR #1891 |
| 9-10 | `QSB_GREEN 20`, `QSB_GREEN_SHARED 8` | 116-SM prepare / 20-SM finish split |
| 161 | `QSB_GLV11 1` | eleven-term geometry, 21.1 GiB table |
| 44 | `QSB_FEED_BLOCK 2 /* HY24 ... */` | later HY round |
| 5398 | `QSB_CG_HIGHFOLD /* HY7 port of jacklightChen fb1105b1, 27 Sep */` | **27 September** — post-dates the record |

The directory also carries `cpu_cogrind3*.h`, `cg_*.h`, `qsb_carrier_sm89.h`
and `build_carrier.sh` — the CPU v2 co-grinder and the native sm_89 carrier.

## 2. What that costs the brief's arithmetic

The live leaderboard (I fetched `https://www.yukon.org/qsb` this session) is
unmoved since **28 Sep 04:17 UTC**:

```
01 kaankolcu                Codex   1,008,206,828  +12,877,351 (+1.29%)
02 cefika                   Opus 5    995,329,477  +16,106,745 (+1.64%)
03 terrapinelf              GPT       979,222,732  +18,392,607 (+1.91%)
04 fkiene (2 runs)          Opus 5    960,830,125  +26,379,737 (+2.82%)
05 kshitij-hash             Opus 5    934,450,388  +19,605,344 (+2.14%)
06 terrapinelf              GPT       914,845,044  + 9,873,230 (+1.09%)
07 fkiene                   Opus 5    904,971,814  +23,698,411 (+2.69%)
08 anamdongparkjinhyeong   Astra     881,273,403  +30,400,702 (+3.57%)
```

`pinning.cu:6` names **cefika 482a55e6** as an ancestor. That is leaderboard
row 02 = **995,329,477**. The tree in hand therefore sits in a lineage that
has already been measured at 995.3M, i.e. **1.29% below the 1.008B record**,
not 15.4% below it.

Consequence, stated plainly: **the 15.4% figure measures a two-generations-old
package against today's field.** The engineering gap in front of me is on the
order of **+1% to +2.3%**, and the promotion rule is `record x 1.01`.

- If the in-hand bytes are already the 1.008B record's bytes, then clearing the
  floor is a **re-roll**, which `BRIEF-QSB.md:36-40` forbids and which I will
  not do.
- If they are the 995.3M bytes, clearing needs **+2.31%** of real throughput.

I cannot resolve which from the tree alone: `pinning.cu:1` says "l2state
variant fkF20c8", the record commit is `8d07d3e` (kaankolcu), and I have no
`git` in this session. This is the single most important open fact and it is
recorded as OPEN in the ledger at the end of this file.

## 3. The brief's "still open" list is dead or already banked

`BRIEF-QSB.md:54-60` and `DEAD-ENDS.md:26-31` list as open:
host gate + C31 on PR #743 + carry62; 16M batch + root-priority completion
lane; further 2^-31 tails.

Measured against this tree:

- `QSB_HOST_GATE 1` (line 41) and `QSB_C31 1` (line 53) — **present and
  default-on**. Scored long ago: the C31+gate package was the 786.4M
  generation (`DEAD-ENDS.md:17`).
- `QSB_COMPLETION_MODE 1` (line 429), `QSB_SLOTPIPE 1` (410), `PriorityPipeline.h`
  present — the root-priority completion lane is **in**. Measured
  `PRIORITY-SUBMISSION.md:39` at **+0.081918%** pooled, below its own
  predeclared +0.15% gate; that note says it "should not be repeated after
  one scored result".
- Chain-loop unroll (2/3/4), Karatsuba, grouped/joint GLV, exact top-16,
  L1 prefetch, fused prepare+finish, PR #714 header, product order: all in
  `DEAD-ENDS.md` with official losses. `QSB_UNROLL 1` (line 299) is the
  surviving default.

`DEAD-ENDS.md` and `NEXT-OPTIMIZATIONS.md` are written against the **778M**
frontier (`NEXT-OPTIMIZATIONS.md:4`). They are two months stale. Reading
`DEAD-ENDS.md` as the current agenda would produce a seventh re-roll.

## 4. The "+7.5% prize" is the baseline, not a lever

`SUBMISSION-v27.md:111-119` (and v24/v25/v26, and `ITERATIONS.md:46-49`) name
terrapinelf's open prize: ~+7.5% for "keeping the 11-add chain while holding hot
records inside the 72 MB L2 — a two-level table, a skewed-width digit recoding,
or partial recomputation of the wide segments."

This tree already **is** that geometry: `QSB_GLV11 1` (line 161), eleven-term,
`pinning.cu:670-673` static_asserts the segment ladder, and line 666 puts the
table at 22,688,113,472 B with 786,432 hot records (`QSB_HOT_RECS 786432u`,
line 1416) pinned by the persisting-L2 window. `SUBMISSION-GLV12-V2.md`
(line 51) records the eleven-term geometry promoted at **948,943,797**.

Anyone reading only `DEAD-ENDS.md` would propose that as new work. It is not.

## Ledger

| # | item | status | evidence |
|---|---|---|---|
| 1 | brief baseline (882M / 15.4%) is stale | **DONE** | `pinning.cu:1-10,44,161,5398`; live leaderboard fetch |
| 2 | tree descends from cefika 482a55e6 = 995.3M | **DONE** | `pinning.cu:6`; leaderboard row 02 |
| 3 | tree ≥ 26 Sep (HY7, jacklightChen 27 Sep) | **DONE** | `pinning.cu:5398` |
| 4 | brief's "still open" items already in tree | **DONE** | `pinning.cu:41,53,410,429`; `PRIORITY-SUBMISSION.md:39` |
| 5 | terrapinelf +7.5% prize IS the baseline | **DONE** | `pinning.cu:161,666,670-673,1416` |
| 6 | **is the tree identical to, or behind, the 1.008B record?** | **OPEN — blocking** | needs `git diff 8d07d3e -- candidates/pinning/pinning.cu`; no `git` in this session |
| 7 | read stage-0 hot loop for a structural lever | OPEN | |
