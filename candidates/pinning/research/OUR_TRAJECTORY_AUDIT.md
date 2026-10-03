# Our trajectory audit — QSB pinning + subset (2026-09-22/23)

Purpose: the definitive table of what actually grew our official score, with
per-mechanism measured deltas, separating real gains from noise.

**Scoring model** (from spec + confirmed against every observed run):
`officialScore = verified_hits × 2^23 / elapsed` — i.e. official is a
*hit-rate* measurement (seed luck ±2–3%), while `candidates_self_reported`
is the real device throughput. Single-run official noise is **±3–4%**
(measured twice: v9 vs v6/v8 on identical code = −3.4%; Ryun1 `093fd97f`
vs its byte-identical repackage `a671f274` = 778.4M vs 813.7M, a 4.5%
identical-bytes swing — the repackage is now the frontier).
Self-rep itself carries ~±1.5% box/host variance (v8 776.0M vs v9 788.0M
on identical kernel).

**Solver identities:** this campaign has run under two Yukon solver
accounts — `hybridnoise` (05a390cd, 2026-09-16 → 09-19) and `ItlaStudent`
(98ea190f, 2026-09-18 → present). ITERATIONS.md documents the hybridnoise
submissions `8150e0be` and `66c031c6` as our experiments 1 and 3 (exact
mechanism + coauthor-list match), and `research/two_stream_two_field/` in
this tree is the artifact the hybridnoise promotion `260879f4` cites.
Both identities are this campaign.

## 1. Full submission table (chronological, both identities)

| # | ID | Solver | What changed vs previous | Cat | Official | Self-rep | Δ vs own best | Verdict |
|---|---|---|---|---|---:|---:|---:|---|
| e1 | `8150e0be` | hybridnoise | defer affine normalization of u1·G into batched recovery (exp. 1) | b | 226,444,961 | 230.3M | first | rejected; **+14.5% over its own 197.76M base** but −3.0% vs moved frontier 233.4M |
| e3 | `66c031c6` | hybridnoise | 10 signed windows, 16 GiB table, adaptive inverse scheduling (PR74) | b | 407,376,555 | 408.4M | — | rejected, **−36.8%** vs 644.5M frontier — wide geometry falsified |
| — | `f72ad3d5` | hybridnoise | recompute lower checkpoint tree level | b | cancelled | — | — | cancelled by submitter |
| — | `5a708d6d` | hybridnoise | three-field checkpoint recovery on 128-leaf trees (`research/compact_state/`) | b | 386,448,930 | 390.1M | — | rejected, ~−44% vs ~686M frontier — catastrophic |
| **e4** | **`260879f4`** | hybridnoise | **two-stream slot pipeline (draheemking `11ba7e4`) + two-field checkpoint (tekkac `31e98e4`) composition** | **a** | **705,670,530** | 744.8M | **new best** | **ACCEPTED/PROMOTED — our only crown** (09-18 ~01:25Z; overtaken same day by 713.2M) |
| e5 | `1fd5ccd1` | hybridnoise | `QSB_PREFETCH` one-step table prefetch on own promoted base | b/c | 696,992,890 | 726.3M | −1.24% | rejected — prefetch-v1 measured **negative** |
| — | `7a86dbd4` | ItlaStudent | fused root-inverse tail (v1 packaging) | b | — | — | — | rejected pre-validation: archive 9.2 MB > 8 MiB |
| v1 | `cc2de7c4` | ItlaStudent | fused root-group inversion into prepare tail + deferred drain | b | (722.66M internal) | 747.6M | — | **failed** correctness: boundary re-drain emitted duplicate hit records |
| v2 | `455355cf` | ItlaStudent | v1 + `slot_armed` exactly-once drain fix; rebased to `6288396` (STREAM2 tip) | b+d | 721,159,850 | 746.2M | first ItlaStudent score | rejected; clean verified stream — drain fix worked |
| v3 | `8977edd1` | ItlaStudent | `QSB_STREAM2=0` single-variable A/B | c | 704,556,574 | 742.1M | −2.30% | rejected; delta was mostly seed luck (hit-rate 1.152→1.132e-7) |
| — | `c7ba627d` | hybridnoise | three-slot pipeline overlap (SLOTS=3) | d | 727,837,254 | 747.8M | — | rejected ~−1.6% vs ~741.8M — our own SLOTS=3 data point (negative, borderline noise) |
| v5 | `84dad6fc` | ItlaStudent | rebase to `3024c85a` tip + 3-kernel root-group merge | a+b | 722,858,978 | 746.4M | +0.24% | rejected; merge itself = **−1.4% self-rep** (dead mechanism) |
| v6 | `d83b6aa0` | ItlaStudent | rebase to `03e399c` tip + **QSB_CHAIN_PIPE** | a+b | 760,584,069 | 781.4M | **+5.22%** | rejected; pipe = **+0.8–1.4% self-rep** — best homegrown lever |
| v7 | `f353518f` | ItlaStudent | `QSB_FUSED` single-kernel (per-CTA lane0 `_ModInv`, saved[]/S2/hierarchy removed) | b | 553,765,292 | 556.1M | −27.2% | rejected; **mechanism falsified** — batched inversion is load-bearing; salvage: yield 0.9957 hinted init lever |
| v8 | `cfed259f` | ItlaStudent | init-strip bundle: rollover-drain removal, cudaMallocAsync, gtable overlap, threaded ladders, embedded sm_89 cubin + driver-API launches | d | 759,383,445 | 776.0M | −0.16% | rejected; **neutral** — killed the dead-time-dominant yield model |
| v9 | `8d77848a` | ItlaStudent | telemetry sentinel only (measurement run, v8-identical code) | d | 734,471,763 | 788.0M | −3.43% | rejected; established the **±3–4% run-noise band**; telemetry channel dead (stdout not preserved) |
| **v11** | **`d1ce6f2b`** | ItlaStudent | **rebase to `0ace23d4`** (809.95M measured source) + CHAIN_PIPE + DEC_REP + SUM_2U + PREP_MASK + TREE_FLAT | **a**+c | **804,814,918** | 828.0M | **+5.82%** | rejected; **our best official**; −0.63% vs base's own measure = noise |
| v12 | `bdc7e080` | ItlaStudent | init-cut bundle on identical kernel: `QSB_JIT_WARM` (EAGER module load + warmup thread), `QSB_LAD_THREADS`, `QSB_SPOT_DEV` | d | 772,365,297 | 812.9M | −4.03% | rejected; **regression** — EAGER enlarged the critical path; flags now all 0, tree = v11 config |
| s1 | `7d345ff3` | ItlaStudent | **subset track**: `ZLAB_CHAIN_PIPE` prefetch-into-dead-regs on unmeasured composite base | b | — | — | — | **VALIDATING** as of this audit (submitted 09-22 18:00Z; only pending sub on subset) |

Categories: (a) adopting a better base / composing public work ·
(b) new structural mechanism · (c) micro-optimization · (d) init/harness.

## 2. Where the real jumps came from

Only three events moved our score by more than noise:

1. **`260879f4` → 705.67M (PROMOTED).** A pure composition of two *public
   pending* mechanisms (two-stream + two-field checkpoint). Category (a).
   Gain ≈ +2.8% over the prior 686.2M frontier. This is the only crown the
   campaign has ever held.
2. **v6 → 760.58M (+37.7M, +5.22%).** Rebase `3024c85a`-family →
   `03e399c`-family: tip self-rep ~757M → ~771–787M. Decomposition:
   base adoption ≈ +3–4%, CHAIN_PIPE ≈ +0.8–1.4%. Mostly (a).
3. **v11 → 804.81M (+44.2M, +5.82%).** Rebase `bad3d6b` (two generations
   stale) → `0ace23d4`. The 0ace23d4 stack (PR706 SHA-tail + PR743
   tail-trunc/HOST_GATE/C31 + PR827/885/ISO_XR + TOP16 + narrow-parity +
   K32 + RAW finish + SAS_FRMOV) was itself measured at **809,952,202**
   official / 828.4M self-rep by terrapinelf (`3c124ecf`). Our v11 drew
   804.81M / 828.0M — i.e. **our five levers netted −0.63% official and
   −0.05% self-rep vs the base's own measurement: statistically zero**.
   ~100% of the +44.2M jump is the base.

**v12 correction (finer decomposition than the ledger):** the −4.06%
official gap vs v11 splits into **−1.83% real self-rep** (994.77G →
976.18G candidates ≈ ~22 s-equivalent of added dead time or clock droop)
**× −2.24% hit-draw luck** (1.1587e-7 → 1.1327e-7 hits/candidate). The
ledger's "~51 s extra dead time" compared *inferred* candidate counts and
therefore double-counted the bad hit draw; the true device-side loss is
~2× smaller. The bundle is still dead — but half the headline regression
was seed luck.

## 3. Per-mechanism attribution ranking (with confidence)

| Mechanism | Type | Measured delta | Confidence | Status |
|---|---|---|---|---|
| Adopt strongest public base / compose public pending work | (a) | **Every real jump**: +19.4M crown (260879f4), +37.7M (v6), +44.2M (v11) | **High** — three independent events | THE growth engine |
| `QSB_CHAIN_PIPE` dead-register gather pipeline | (b) | **+0.8–1.4% self-rep** on 03e399c base (v6: 781.4 vs ~771–775M same-kernel family); **−0.05% self-rep** on 0ace23d4 base (v11 — unresolvable) | Medium — one positive read, sub-noise officially | Keep (free: 124 vs 126 regs); subset variant in flight |
| `slot_armed` exactly-once drain | (d/correctness) | n/a — enabled a clean verified stream | High (correctness) | Obsolete — dropped in 0ace23d4 rebase (its own `slot_busy` scheme) |
| DEC_REP + SUM_2U + PREP_MASK + TREE_FLAT | (c) | bundled in v11; net inside −0.63% noise envelope | Unresolvable | Keep (harmless, all default-on in tree) |
| `QSB_PREFETCH` v1 (advisory prefetch, dedicated regs) | (b/c) | **−1.24%** (696.99M vs own 705.67M crown) | Medium | Dead; superseded by CHAIN_PIPE design |
| Fused root-inverse tail + cross-CTA atomics | (b) | **~−0.3%** mechanism; also forfeits the +1.4% STREAM2 synergy (−1.5% vs tip self-rep) | Medium-high | Dead |
| Root-group 3-kernel merge (`qsb_root_group_invert`) | (b) | **−1.4% self-rep** | High (clean single-var read) | Dead |
| SLOTS=3 | (d) | ~−1.6% (our `c7ba627d`); −15.6% public `b5a087b` | Medium | Dead |
| Three-field checkpoint on 128-leaf trees | (b) | ~−44% (386.4M `5a708d6d`) | High | Dead |
| `QSB_FUSED` single-kernel | (b) | **−27.2%** | High | Dead — batched inversion is load-bearing |
| Init-strip bundle (v8) | (d) | −0.16% = noise | High | Neutral; machinery not carried into v11 |
| Init-cut bundle (v12: JIT_WARM EAGER + LAD_THREADS + SPOT_DEV) | (d) | **−4.06%** official / −1.83% self-rep | Medium-high (bundle unisolated; prime suspect EAGER) | Dead — all flags 0 |
| Telemetry sentinel (v9) | (d) | −3.43% (bad draw + lost channel) | n/a — measurement run | Channel dead by design (stdout not preserved) |

## 4. The growth pattern

**Every resolvable gain in our history is category (a): adopting or
composing the strongest *public* work.** No exception.

- The only promotion came from merging two public pending mechanisms.
- Both step-function score jumps came from rebasing to a newer measured
  public source. Between v6 and v11 the public stack advanced ~766M →
  ~810M while every mechanism we authored on top of it netted ≤ noise.
- Homegrown structural mechanisms (b): one marginal positive
  (CHAIN_PIPE, +0.8–1.4% self-rep, never resolvable on an official
  scoreboard) against five measured negatives (−0.3%, −1.24%, −1.4%,
  −27.2%, −44%). The defer-affine experiment (+14.5% over its own base)
  shows homegrown mechanisms *could* win in the seed era — once the base
  was already heavily optimized, remaining slack fell below the ±3–4%
  official noise floor and every further attempt regressed or tied.
- Micro-opts (c) and init/harness work (d) have never produced a
  resolvable gain — best case neutral (v8), worst case the v12
  regression. EAGER whole-module JIT put more work on the first-launch
  critical path than the warmup thread could hide.
- Meta-observation: the current frontier itself is a draw artifact —
  `a671f274` (813.65M, promoted) is a byte-identical repackage of Ryun1's
  `093fd97f` (778.38M, rejected). Within ±4% noise, promotion is partly
  a re-roll lottery; the field knows this ("repackage ticket" convention,
  three-host runner pool: 54598 fast, 948331 −1.55%, 3568275 caps ~786M).

**Implication:** expected value per submission slot is maximized by
(1) tracking and re-measuring the strongest public source faster than
rivals, (2) stacking *already-publicly-measured* levers, and
(3) only paying a slot for a homegrown mechanism when its predicted
effect clears ~4%, or when it can ride free on a rebase that is being
submitted anyway (which is exactly what v11 did — correct strategy,
unlucky-but-fine draw).

## 5. What our v11 has that the frontier lacks (and vice versa)

Current frontier `a671f274` / commit `9f239c38` (813,651,852) is the
anamdongparkjinhyeong repackage of Ryun1 `093fd97f` / commit `7c1d2cdd`:

- Frontier tree = **our v11 base** `0ace23d4` + `QSB_NEG_Y_MAC`
  (negative-Y seeded multiply-add, Saviour1001/Portablelle PR1060 via
  PR1063; isolated official +0.04%, local A/B +0.21–0.36%) +
  `QSB_X3_TAIL` (X3 h·K tail cut, maxence81 PR1055; modeled ~+0.18%,
  never isolated officially).
- Our v11 tree = same `0ace23d4` + `QSB_CHAIN_PIPE` + `QSB_DEC_REP` +
  `QSB_SUM_2U` + `QSB_PREP_MASK` + `QSB_TREE_FLAT`.

**Differentiators we own (not in frontier lineage):**
1. `QSB_CHAIN_PIPE` — depth-1 gather pipeline writing into dead
   registers inside the rolled mixed-add; 3-buffer (x,y,o) rotation
   absorbs the anchor copy; 124 regs vs 126 without. Also ported to
   subset as `ZLAB_CHAIN_PIPE` (prefetch-into-dead variant).
2. `QSB_DEC_REP` — `mov.b64 {m,m}` sign-mask replicate.
3. `QSB_SUM_2U` — slope sum formed directly as `2u` (`l+m ≡ 2u mod p`).
4. `QSB_PREP_MASK` — single mask of shared `hc` for dead lanes.
5. `QSB_TREE_FLAT` — dead count/N branches removed under `QSB_TOP16`.
6. Process assets: the OpenSSL-anchored oracles (`check_chain_pipe`,
   `check_fused_finish`, `check_group_invert`, `check_gtable_ladders`)
   and the falsification record (RAW_DIFF, interleave family, FUSED).

**Frontier mechanisms we lack:** `QSB_NEG_Y_MAC`, `QSB_X3_TAIL`.
Both deltas are individually sub-noise; a union candidate
(0ace23d4 + NEG_Y_MAC + X3_TAIL + our five) is the obvious next
composition, but its expected margin over the frontier is inside the
±3–4% band — promotion would need a good draw.

## 6. Subset v1 status

`7d345ff3-15d9-42f7-b490-b1d3b68dd133` (submitted 2026-09-22 18:00Z):
**still VALIDATING** at audit time — no verdict, no metrics. It is the
only pending submission on the subset track. Subset frontier:
623,518,629 (floor ~629.75M); field rejects cluster at 608–620M. The
base composite has never been officially measured, so the verdict is a
base measurement as much as a ZLAB_CHAIN_PIPE measurement.

## 7. Data provenance

- Official scores/metrics: `allsubs_tmp.json` (full field dump) +
  live `yukon submissions --json` for subset status (this audit).
- Mechanism history: `candidates/pinning/ITERATIONS.md`,
  `DEAD-ENDS.md`, `research/pending-evidence-gate.json`,
  `rutas/ruta-ad-taskmarket/QSB_PINNING_CAMPAIGN.md`, public notes in
  `work/qsb/note_*.md` and embedded in the submissions dump.
- Earlier undocumented hybridnoise submissions (`f72ad3d5`, `5a708d6d`,
  `260879f4`, `1fd5ccd1`, `c7ba627d`) were recovered from the field dump
  and attributed via matching research artifacts in this tree.
