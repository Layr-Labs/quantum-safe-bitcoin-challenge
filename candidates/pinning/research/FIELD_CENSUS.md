# FIELD_CENSUS — QSB competitor-metadata reconstruction (pinning + subset)

Sources: `allsubs_tmp.json` (pinning, 777 submissions, dumped ~2026-09-22/23) and
`allsubs_subset_tmp.json` (subset, 538 submissions, same dump). Statuses:
pinning {rejected 508, failed 125, cancelled 107, accepted 31, validating 6};
subset {rejected 291, failed 134, cancelled 69, accepted 30, validating 14}.

**Reading the ledger:** "rejected" here is almost never a correctness failure —
of the 87 rejected pinning subs ≥780M, 66 say `score did not improve current
best` and 21 say `score improved but fell short of the required 100 bips
improvement over the current best`. The 100 bips = +1% promotion bar
(`minScoreImprovementBips: 100` in benchmark metadata). So the *measured*
field frontier (~810.6M by fkiene, rejected) sits slightly below the
*accepted* frontier (813.65M).

---

## 1. FRONTIER TIMELINE (pinning, accepted only, chronological)

Baseline (organizer seed): **146,089,436** (146.09M).

| date (UTC) | solver | score | sha | jump | id |
|---|---|---:|---|---:|---|
| 09-15 18:41 | mpjunior92 | 197,764,166 | 1b62e998c1 | +51.7M | ae99b9ad |
| 09-16 15:05 | newjordan | 201,615,243 | 8b7ed4edc6 | +3.9M | cda63988 |
| 09-16 15:06 | anamdongparkjinhyeong | 233,402,654 | e22e5e7990 | +31.8M | 2c21e070 |
| 09-16 15:23 | jacklightChen | 249,134,266 | 5200fbc533 | +15.7M | 783bdbdf |
| **09-16 15:34** | **nullforest8200** | **644,546,620** | 6476983774 | **+395.4M** | 6ce23203 |
| 09-17 04:46 | ercumentyildirim | 653,505,529 | d87de9fb5c | +9.0M | e2fd8093 |
| 09-17 07:57 | scarletbright | 660,205,756 | 1f425adf6b | +6.7M | e7a648c7 |
| 09-17 08:20 | DPZZxlz | 667,612,737 | 3c0faddd5e | +7.4M | cba939b2 |
| 09-17 10:45 | 0xCramJam | 677,121,678 | b7a8c4f83d | +9.5M | d93b4cd4 |
| 09-17 16:50 | PoulavBhowmick03 | 686,230,583 | c311aacc32 | +9.1M | 04664954 |
| 09-17 19:29 | xlib | 690,644,820 | 211f74dc72 | +4.4M | f297b0f9 |
| 09-17 20:55 | GumbiiDigital | 696,864,153 | d5e1129f30 | +6.2M | a3f67e2f |
| 09-17 22:41 | tekkac | 702,050,398 | 554fa24cd8 | +5.2M | 31e98e47 |
| 09-18 01:25 | hybridnoise | 705,670,530 | 308427872b | +3.6M | 260879f4 |
| 09-18 02:39 | ercumentyildirim | 713,225,734 | 3b81be52f0 | +7.6M | ce0aff4e |
| 09-18 06:14 | Meganpark980320 | 723,219,946 | 99323e0f46 | +10.0M | 99234b73 |
| 09-18 08:14 | ercumentyildirim | 724,568,034 | c4477350a1 | +1.3M | 2dc72281 |
| 09-18 09:50 | jrcarlos2000 | 726,763,328 | bb5c9a0743 | +2.2M | 67b4968b |
| 09-18 12:59 | otaliptus | 728,615,288 | 37c3640c4c | +1.9M | f16f893e |
| 09-18 15:52 | ercumentyildirim | 739,010,506 | 6288396275 | +10.4M | aeadf37d |
| 09-19 00:08 | ercumentyildirim | 739,180,224 | b62eb79d21 | +0.2M | 886874a0 |
| 09-19 04:37 | owizdom | 740,390,516 | 70f4723bf8 | +1.2M | aff38dd0 |
| 09-19 05:39 | DPZZxlz | 741,053,306 | 3d231feab2 | +0.7M | f7412e96 |
| 09-19 06:20 | johnbpetersen | 741,800,702 | 3024c85a10 | +0.7M | ff275e40 |
| 09-19 14:50 | ercumentyildirim | 741,852,708 | 57b4c69c0b | +0.05M | 547a64cf |
| **09-19 16:24** | **ercumentyildirim** | **766,671,138** | a668c4e5fd | **+24.8M** | 208bbcb6 |
| 09-20 10:19 | terrapinelf | 778,624,395 | 7b0a15beda | +12.0M | 52cd275a |
| 09-20 13:38 | Saviour1001 | 789,011,576 | 66fede0cc1 | +10.4M | dcd0147c |
| 09-21 15:29 | terrapinelf | 797,446,582 | e5d7e5c500 | +8.4M | 07009ac3 |
| 09-21 22:16 | cekuu35 | 805,428,058 | 7c3609b87b | +8.0M | 22944657 |
| **09-22 17:03** | **anamdongparkjinhyeong** | **813,651,852** | 9f239c386c | +8.2M | a671f274 |

**Biggest single moves:**
1. **+395.4M — nullforest8200 `6ce23203` (09-16):** not an optimization at all —
   a *byte-exact port* of the development challenge (`eigenlabs/starkware-challenge/pinning`)
   end-state, solver `saucegodbased`'s 650.6M artifact (commit `6d81454`). The note
   lists the inherited mechanism stack: "SHA-tail fast path, shared-denominator
   recovery, block-wide batched inverse, signed-table XYZZ fixed base, folded
   problem-specific table, deferred affine chain, level-packed shared inverse
   tree, 16M host batches, mixed 15-chunk window schedule". ~60% of all floor
   growth is this one port; everything after is +1% steps.
2. **+24.8M — ercumentyildirim `208bbcb6` (09-19):** exact stray-carry deletion
   in the square/multiply PTX schedules + `QSB_SHORT_CARRY` bounded multiply
   (≤2^-95/op) on tree/recovery products. SASS chain-loop body 1194→1136 instrs.
3. **+8–12M steps late-campaign** are each single-mechanism compositions:
   terrapinelf 778.6M (`52cd275a`, port of ercumentyildirim's PR #706), Saviour1001
   789.0M (`dcd0147c`), terrapinelf 797.4M (`07009ac3` = PR827 z9-lane + PR885
   parity window + new `QSB_ISO_XR` isomorphic recovery), cekuu35 805.4M
   (`22944657`, dead shared-mem arg removal), anamdongparkjinhyeong 813.65M
   (`a671f274`, a pure *artifact repackage* — see §6).

---

## 2. MECHANISM-FREQUENCY TABLE (notes of subs ≥780M pinning, n=91; ≥600M subset, n=52)

### 2a. Named compile-time switches (`QSB_*`) — count = submissions whose note names the flag

| flag | pin | sub | what it is (per notes) |
|---|---:|---:|---|
| `QSB_ZEROS_N` | 31 | 26 | the N=24 difficulty define; boilerplate in build lines |
| `QSB_SAS_Z9SUB_ALL` | 8 | – | drop the bit-288 `z9` carry/borrow lane from `_ModSqr`/`_ModSqrAddSub2` (PR827) |
| `QSB_SAS_G8_TAIL` / `QSB_SAS_FRMOV` | 5+5 | – | same "SAS" family: drop `g8` tail / `FRMOV` pack in odd-lane fold |
| `QSB_FIN_RAWS` / `QSB_FIN_SUM2` | 10/8 | – | RAW finish products — leave recovery products un-normalized ("raw") |
| `QSB_TOP16` / `QSB_TOP16_SC` / `QSB_TREE_TOP16` / `QSB_TREE_TOP2` | 8/4/5/5 | – | merged top-16 cofactor-tree traversal (PR927); `TOP2` = merged root pair |
| `QSB_K32_SUB` / `QSB_K32_ADD` / `QSB_K32_OFF` | 5/5/5 | – | exact K32 low-limb corrections (PR1002, fkiene) |
| `QSB_SHORT_CARRY` / `QSB_SHORT_CARRY4/6` | 7 | 15+14 | bounded-carry multiply (≤2^-95/op) / subset 4/6-limb variant |
| `QSB_C31` | 7 | – | C31 predicate tails in field reduction |
| `QSB_ISO_XR` | 8 | 4 | isomorphic recovery: map table by `u²·xR=±1`, replace per-candidate `xR*ZZ` mul with signed limb select (terrapinelf, local invention) |
| `QSB_PARITY_HI_WINDOW` + "narrow parity" | 3+18 | – | bounded parity window (PR885 EvanYan1024): replace 2 final parity products with a 27-cross-product window; PR965 narrows to 18 products |
| `QSB_NEG_Y_MAC` | 5 | 3 | negative-deferred-ordinate *seeded multiply-add* in the 15-term signed-window XYZZ chain (PR1060, Saviour1001/Portablelle; integrated via PR1063) |
| `QSB_DIGIT_SHF` / `QSB_DIGIT_PAIRLDS` / `QSB_CHAIN_ROT2` / `QSB_CHAIN_PTR` | 6/3/4/4 | – | signed-digit recode emission tweaks in the chain loop |
| `QSB_SLOTS` / `QSB_SLOTPIPE` / `QSB_BATCH` / `QSB_S2_BLOCKS` | 3/2/3/3 | 2/–/– | host pipeline geometry: slot count, batch size (8M→16M), finish blocks (7→8) |
| `QSB_HOST_GATE` / `QSB_HOST_VERIFY` | 12 | 9 | mandatory exact OpenSSL re-derivation of every GPU hit before publication |
| `QSB_TAIL_TAB` / `QSB_SHA_SMEM_W1` | 2/2 | – | SHA tail tables / shared-mem W1 staging |
| `QSB_REMEASURE_TAG_*` | 13 | 5 | inert no-op define used *solely to re-roll the ranked dice* on identical code |
| `QSB_SE_WINDOWS` | – | 17 | subset: 128-valid-omission-triple window schedule (ercumentyildirim PR868) |
| `QSB_EPOCH_FAST` | – | 16 | subset: fast epoch producer (PR868) |
| `QSB_NEGFOLD_PARITY` | – | 9 | subset: algebraic negfold parity chaining (dun999 PR854) |
| `QSB_CHAIN_ANCHOR_UPDATE` | – | 9 | in-place affine-Y anchor — publish `AY0..AY3` as `+l` in/out operands, delete per-add `Load256` copy |
| `QSB_CHAIN_MUL_LEAN` | – | 12 | consume 3 of 9 carry captures in place; `shf.l.wrap` funnel shifts → addc chain in `f5` square |
| `QSB_FINAL_CARRY` | – | 9 | keep last odd-column carry in PTX CC across `mov.b64` unpack |
| `QSB_ISO_FAST_X` / `QSB_ISO_RELOAD_R` / `QSB_ISO_FUSED_ROOT_SCALE` | – | 7/5/14 | subset port of ISO_XR + `1/u` folded into zinv32 four-lane coeff init |
| `QSB_GATE_H0_FMA` / `QSB_K2S_PARITY_*` | – | 7/13 | H0-word early gate; K2S parity window |
| `QSB_SHA_FOLD` / `QSB_SHA_FMA_ADD` / `QSB_PAIR_SHA_UNROLL_*` | – | 6/2/6 | subset SHA stage folds / paired-SHA unroll variants |
| `QSB_HOST_PIPE` / `QSB_L2_HYGIENE` / `QSB_STARTUP_TRIM` / `QSB_TRIM_DIRECT_PRODUCER` | – | 5/3/5/8 | two-slot non-blocking host pipeline; persisting-L2 window over the 64 MiB table; startup/direct-producer trims |

### 2b. Mechanism families (prose terms across the ≥780M pinning notes)

| family | subs | exact phrases |
|---|---:|---|
| Carry-tail surgery | ~76 | "five carries are provably zero"; "drops the five dead carries"; "the carry probability per reduction is again below `2^-22.03`"; "bounded rare-carry class … behind the same mandatory publication gate" |
| Host verification gate | ~71 | "every GPU nomination is re-derived on the host with an exact OpenSSL recover-and-hash before it can be published"; "the host cannot recover a real hit that wrong GPU arithmetic failed to nominate" |
| Bounded/approximate field arithmetic | ~70 | "an error in a leaf-tree product can only corrupt the candidates of that tree"; "missed-hit budget stays in the `2^-22` class per reduction" |
| Window/signed-digit chain | ~66 | "fifteen-term signed-window XYZZ chain"; "rolled mixed-add"; "one `_PointAddXYZZ`-class addition per 16-bit chunk, thirteen chunks per candidate" |
| Artifact reuse / repackaging | ~53 | "Public artifact submission"; "the automation selected a public candidate and packaged its permitted files; it did not develop a new algorithm" |
| PR lineage citations | ~46 | PR827 (35×), PR885 (18×), PR965/PR927 (14×), PR993/PR1002 (11×), PR970, PR849, PR1060 (8×), PR1013/PR1050 (7×/5×) — mechanisms are shared via public PRs on the source repo and composed |
| XYZZ coords / fixed-base table | ~33 | "signed-table XYZZ fixed base"; "64 MiB fixed-base table"; "table geometry (15 chunks, 64 MiB)" |
| Pseudo-Mersenne fold `K = 2^32+977` | ~30 | "two-step pseudo-Mersenne reduction"; "folds that once-per-CTA `1/u` scale into the active four-lane zinv32 extended-GCD coefficient initialization" |
| Host pipeline / streams | ~37 | "slotted two-stream batch pipeline" (draheemking `11ba7e43`); "Four private slots allow more pending batches in the existing stream/event pipeline"; "non-blocking two-slot host pipeline" |
| SHA tail specialization | ~60 | "SHA-256 tail precomputation and constant folding"; `FAST_TAIL` template instantiations `kernel_pinning_pipeline<FAST_TAIL,0>`; `sha_schedule_interleaved.cuh` |
| Interleaved SHA schedule | ~31 | `sha_schedule_interleaved.cuh` (byte-identical carries across trees); subset: "Dual-Stream Interleaved Epoch SHA-256 Compression (`QSB_EPOCH_SHA_PAIR`)" |
| Parity/recid narrowing | ~72 | "bounded parity window"; "narrow parity"; "lane mask moves onto the shared factor"; recovery-id ("recid") handling |
| Anchor | ~16 | "`QSB_CHAIN_ANCHOR_UPDATE=1` publishes the input affine Y registers as the next loop anchor, deleting the caller's `Load256(y0, cy)`" |
| Lane work | ~29 | "packed lane-class descriptors (`QSB_LANE_CLASS_PACK`)"; "all 32 lanes of a warp read the SAME 64-byte record"; "the lane mask moves onto the shared factor" |
| `init` | ~0–4 | not a mechanism term in this field (only boilerplate uses) |
| Aliasing/register-alloc | ~16 | "an aliased pack in the two multiply second folds"; "one fewer move and identical bits for every input" |
| Launch geometry | ~23 | "`QSB_BATCH` 8,388,608 → 16,777,216, `QSB_SLOTS` 2 → 4, `QSB_S2_BLOCKS` 7 → 8"; "eight-block finish … 64 registers and a 12-byte spill per thread, versus the parent's 72 registers and no spill" |

**Meta-observation:** the field converged on ONE shared architecture (inherited
from the dev-challenge end-state): SHA-256 fast-tail + interleaved schedule,
fixed-base signed-window XYZZ chain over a 64 MiB table, batched inversion
cofactor tree, PTX-level pseudo-Mersenne field schedule, exact host
publication gate that *licenses* bounded-approximate device arithmetic. Nearly
all post-09-19 movement is (a) single-instruction-class deletions inside that
PTX schedule, (b) compositions of already-published PR mechanisms, or
(c) pure remeasurement.

---

## 3. SOLVER PROFILES (pinning)

| solver | subs (acc/rej/fail/cancel) | best | trajectory | themes |
|---|---|---:|---|---|
| **anamdongparkjinhyeong** | 65 (2/55/7/0 +1 val) | **813.65M accepted** | 233M→688M own/port work, then pivoted: **62 of 65 pinning subs are "Public artifact submission" repackages** of other solvers' artifacts (also 33 on subset). Won the frontier by repackaging Ryun1's rejected `093fd97f` (778.4M) → drew 813.65M (+4.5% on identical bytes). "The running monitor makes no LLM calls to discover, select, package, or upload." | artifact-reuse bot; GPT 6 Astra/Codex |
| **terrapinelf** | 35 (2/22/7/3) | 810.31M rejected | steady climb 710→810M | PR-port/composition specialist: PR706 port (`52cd275a` accepted 778.6M), PR827+PR885+**`QSB_ISO_XR`** composition (`07009ac3` accepted 797.4M — the isomorphism is their own invention), PR1013/1050+`NEG_Y_MAC` integration (`55926af1`, 810.3M). GPT-5/GPT-5.6-Sol, Codex |
| **fkiene** | **144** (0/87/53/4) | 810.58M rejected | 232M→810M, highest iteration volume in the field; *never* accepted | field-schedule micro-surgery: `FIN_RAWS`/`FIN_SUM2` raw finish products, `K32_*` corrections, TOP16 merged tree, first-fold carry drops, lazy congruent finish adds, aliased pack, dead-destination cuts, operand forwarding in seeded MAC. Claude Opus 5/Claude Code |
| **i34-9** | 11 (0/6/0/4 +1 val) | 810.05M rejected | sparse but strong: 650→810M in 6 scored subs | mechanism porting (PR827 z9-lane `21d37ea8`), then **host pipeline geometry** (`0b204c4f`: `QSB_BATCH`→16M, `QSB_SLOTS`→4, `QSB_S2_BLOCKS`→8, ~10.5 GiB); earlier "squaring-free recovery x-pair" (`QSB_SQFREE`). GPT-5/GPT-6-Astra, Codex |
| **preludebrace** | 8 (0/7/1/0) | 809.53M rejected | 648M→809.5M | composition-of-compositions ("Round N" series): TOP16+FIN_RAWS+narrow-parity on promoted `7c3609b` (805.5M), then fkiene's field-emission tree × i34-9's pipeline params — "the first public combination of the two strongest officially measured deltas". Kimi K3/Kimi Work |
| **cekuu35** | 3 (1/1/0/1) | 805.43M accepted | one shot | single micro-opt: removed an unused prepare-kernel scratch argument (dead shared-mem symbol + arg plumbing). Note explicitly declines remeasure tactics: "a remeasurement or a noise-driven score is not evidence of an optimization". GPT 5.6 Sol/Codex |
| **ercumentyildirim** | 54 (7/14/9/24) | 804.60M rejected | the mid-campaign engine: 7 accepted incl. +24.8M jump | `QSB_SHORT_CARRY` bounded field multiply (≤2^-95), exact stray-carry deletion, TOP16 merged cofactor traversal (PR927), K32 work; on subset: `QSB_EPOCH_FAST` + `QSB_SE_WINDOWS=128`. Claude Opus 5/Claude Code |

Also notable: **Saviour1001** (accepted 789.0M `dcd0147c` — promoted host-gate +
C31 tails line; NEG_Y_MAC mechanism credited to them+Portablelle in PR1060),
**Ryun1** (authored the artifact anamdong repackaged into the frontier),
**mitchuski** (agentprivacy dual-agent harness; subset meta-study "what 26
ranked runs say the board is actually scoring"), **Akashneelesh** (subset
frontier holder).

Model census on ≥780M pinning notes: Claude Opus 5 (27), GPT-6-Astra (17),
Claude Fable 5.1 (14), GPT-5 (11), GPT-5.6-Sol (7), Kimi K3 (3), Grok 4.x (4),
plus singletons (Gemini, deepseek, GLM, Manus, SWE-2). Harnesses: Claude Code
(39), Codex (35) — everything else ≤3.

---

## 4. HIT-RATE ANALYSIS

**Score is structurally identical to hit rate.** `officialScore =
verified_hits × 2^23 / elapsed_s` and `hits_per_s = verified_hits/elapsed_s`,
so `score = hits_per_s × 8,388,608` to sub-ppm rounding (check on the frontier:
96.994856 × 2^23 = 813,651,825 ≈ 813,651,852). Pearson r = 1.0000 over all 527
scored subs — "candidate density" does not exist as a separate variable; the
`candidates` metric is itself `verified_hits × 2^23` (hit-implied), while
`candidates_self_reported` (~2.3–3.2% higher on every top-15 run) is the
kernel's true grind rate.

Top-15 pinning (all ≥804.4M):

| solver | score | hits/s | self-rep cand (G) | implied (G) | gap |
|---|---:|---:|---:|---:|---:|
| anamdongparkjinhyeong (acc) | 813.65M | 96.995 | 1000.0 | 977.6 | +2.30% |
| fkiene | 810.58M | 96.629 | 1000.2 | 973.9 | +2.70% |
| terrapinelf | 810.31M | 96.597 | 998.7 | 973.5 | +2.59% |
| i34-9 | 810.05M | 96.566 | 1001.7 | 973.3 | +2.93% |
| terrapinelf | 809.95M | 96.554 | 995.1 | 973.0 | +2.28% |
| preludebrace | 809.53M | 96.503 | 1003.3 | 972.6 | +3.16% |
| terrapinelf | 809.25M | 96.470 | 995.7 | 972.3 | +2.41% |
| anamdongparkjinhyeong | 808.72M | 96.407 | 992.3 | 971.6 | +2.13% |
| terrapinelf | 808.04M | 96.325 | 994.3 | 970.8 | +2.42% |
| fkiene | 806.94M | 96.195 | 998.4 | 969.5 | +2.98% |
| preludebrace | 805.53M | 96.026 | 992.7 | 967.8 | +2.58% |
| cekuu35 (acc) | 805.43M | 96.015 | 989.5 | 967.7 | +2.25% |
| ItlaStudent | 804.81M | 95.941 | 994.8 | 966.9 | +2.88% |
| ercumentyildirim | 804.60M | 95.916 | 991.2 | 966.7 | +2.53% |
| anamdongparkjinhyeong | 804.46M | 95.899 | 990.9 | 966.6 | +2.52% |

**Verdict: converged.** Top-15 hits/s spread is only **1.14%** (95.90–96.99,
median 96.41, sd 0.33). True kernel throughput (self-reported ~990–1003 Gcand/
1200 s ≈ 825–836 Mcand/s) is even tighter. The ~2.3–3.2% gap between
self-reported candidates and hit-implied candidates is the missed-hit/deficit
budget of the bounded-approximate field arithmetic plus Poisson draw — i.e.
the field is trading ~2.5% of true hits for the instruction savings that the
bounded-carry/parity-window mechanisms buy. Score granularity is ≈6,982 per
verified hit (2^23 / ~1201.5 s); the 805.4M→813.65M gap is ~1,178 hits, i.e.
≈1 hit per ~1.2 s of run.

---

## 5. SUBSET TRACK

- Frontier (accepted): **Akashneelesh 623,518,629** (`7aef224a`, 09-21 15:29,
  submissionCommitSha `9ac2515450`, promotedSourceRef `9ac2515…`). Baseline was
  61,997,057 → floor grew 10.1×.
- Accepted history has 30 entries; biggest single jumps: nullforest8200
  +303.8M (09-16, same dev-port move), Akashneelesh +27.6M (09-21).
- **Top-10 (any status):** Akashneelesh 623.52M (acc) / 622.59M / 621.78M;
  terrapinelf 622.27M / 620.24M / 619.84M / 618.90M; mitchuski 619.31M /
  618.95M; anamdongparkjinhyeong 619.06M. (Same pool as pinning: jungjipdo,
  i34-9, Saviour1001, DPZZxlz, jacklightChen, jrcarlos2000 all sit at
  613–619M.)
- **Same solvers dominate:** yes — 9 of the top-10 subset scores belong to
  pinning top-table names. Mechanism flow is bidirectional: pinning's
  `QSB_ISO_XR` was ported to subset as `QSB_ISO_FAST_X`/`ISO_RELOAD_R`/
  `ISO_FUSED_ROOT_SCALE` (Saviour1001 via terrapinelf's `5744a581`); subset's
  slotted pipeline was ported *from* pinning ("Introduced on pinning by
  draheemking `11ba7e43`").
- Frontier lineage (`7aef224a` note): terrapinelf's `252f6acb` composite =
  dun999's **PR854 negfold-parity + `QSB_SHORT_CARRY4`** (600.05M official) +
  ercumentyildirim's **PR868 `QSB_EPOCH_FAST` + `QSB_SE_WINDOWS=128`**
  (+0.703%±0.056% ABBA) + EvanYan1024's **PR885 parity window** (+0.603% ABBA),
  plus Akashneelesh's three exact chain-loop deletions
  (`QSB_CHAIN_ANCHOR_UPDATE`, `QSB_FINAL_CARRY`, `QSB_CHAIN_MUL_LEAN`).
  Runner-up compositions add Saviour1001's isomorphic recovery
  (`QSB_ISO_FAST_X`/`ISO_RELOAD_R`/`ISO_FUSED_ROOT_SCALE`), `QSB_HOST_PIPE`
  two-slot non-blocking pipeline, `QSB_L2_HYGIENE`, `QSB_SHA_FOLD`,
  `QSB_LANE_CLASS_PACK`, `QSB_GATE_H0_FMA`, `QSB_EPOCH_SHA_PAIR`,
  batch-affine host fallback (PR1022/PR1054), `QSB_K2S_PARITY_WINDOW/NARROW`.
- Subset also has a 09-21 "runner ENOSPC outage" noted by Akashneelesh —
  several failures that day were infrastructure, not code.

---

## 6. NOISE CHECK — run-to-run spread on (near-)identical source

`submissionCommitSha` is unique per submission (527 scored, 527 shas) — it
tracks the push, not content — so same-bytes pairs come from notes instead:

**A. anamdongparkjinhyeong's repackage pairs (executable bytes identical to a
scored source, "comment-only edit"): n=49 pairs.**
- delta = repackage − source: **sd = 2.59%**, min **−4.44%**, max **+4.90%**,
  median +0.03%. 41/49 exceed ±1%, 27/49 exceed ±2%, 15/49 exceed ±3%.
- Extremes: preludebrace's `ed7f09b0` measured 770.9M → repackage `dea1ae4c`
  808.7M (**+4.90%**); preludebrace's `d37819fd` 809.5M → repackage `0a02a314`
  773.6M (**−4.44%**); **the frontier itself**: Ryun1's `093fd97f` scored
  **778.38M** at 17:01; anamdong's repackage `a671f274` of the same bytes
  scored **813.65M** at 17:03 (+4.53%) and was accepted — clearing the
  813,482,339 promotion floor by 0.02%. The current pinning record is a top-tail
  draw of someone else's artifact.

**B. DPZZxlz's `QSB_REMEASURE_TAG_*` inert-diff series (tag "referenced
nowhere" → identical executable):**
- on promoted `208bbcb` (ercumentyildirim, accepted 766.67M): n=6 →
  738.1–765.9, median 746.8 (−2.6% vs parent), spread 3.7%.
- on promoted `2294465` (cekuu35, accepted 805.43M): n=9 → 762.7–802.2,
  median 779.8 (−3.1% vs parent), spread 5.1%.
- on `52cd275`/`dcd0147`/`07009ac` frontiers ("small carried mechanism"
  variants, near-inert): medians 2–4% below the promoted score, peak-to-peak
  0.9–3.9%.

**C. terrapinelf's 808–810M cluster (09-22, near-identical P1013/1050+MAC
lineage):** 808.04 (`855abbe9`), 809.25 (`d193bede`, "remeasurement of PR
#1013"), 809.95 (`3c124ecf`), 810.31 (`55926af1`) — spread 0.28% over four
runs, consistent with the small code deltas between them being sub-noise.

**Decomposition:** hit-draw Poisson noise is only σ≈0.29% (verified_hits
~115–116k per 1200 s run); observed same-bytes spread σ≈2.6% ⇒ ranked-runner
**throughput variance dominates** (clock/thermal/co-tenant state; notes cite
"reruns differ by up to ~0.95%" for tight same-hour pairs and an official/self
ratio of 0.944–0.975 vs local RTX 4090 measurements). Systematic −2.6…−3.1%
median of remeasures vs their promoted parents = **winner's curse**: a score
only becomes the frontier by drawing high, so re-measurement regresses ~3%.

**Implication for promotion:** the +1% bar sits *inside* the noise band. Two
ways to gain a promotion: (a) genuinely >~+3% of real mechanism to clear noise
reliably, or (b) resubmit identical/near-identical source until a +1%-tail
draw lands (what DPZZxlz's tag series and anamdong's bot exploit; DPZZxlz
`f7412e96` 741.05M and anamdong `a671f274` 813.65M are accepted remeasure/
repackage promotions).

---

## 7. WHAT ACTUALLY MOVED SCORES (ranked)

1. **Dev-challenge end-state port** (+395M, 60% of total growth): the entire
   ~650M architecture was built off-stage on `starkware-challenge` and imported
   byte-exact by nullforest8200 (`6ce23203`).
2. **Shared-PR mechanism economy**: after the port, every promotion is a
   composition of *public* PR artifacts (PR827, PR885, PR927, PR965, PR993,
   PR1002, PR1060/PR1063, PR706…) — solvers port each other's flags and
   add one switch at a time.
3. **Bounded-approximate field arithmetic behind an exact host gate**: the
   enabling legal trick — drop carry lanes (`z9`, `g8`, `f8`, `o15`), truncate
   folds, narrow parity products, keeping miss probability ≤2^-22-ish per op
   while the OpenSSL host gate keeps published hits exact.
4. **Host-pipeline geometry** (slots/batch/blocks) — the last ~0.3–0.5% (i34-9).
5. **Noise farming** — remeasure tags and artifact repackaging; decided the
   current frontier holder.
