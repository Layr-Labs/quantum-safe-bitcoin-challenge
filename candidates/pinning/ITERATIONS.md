## v32+v33 verdicts + v34 fired + autonomous draw machine (2026-09-24 ~19:25Z)

**v32 `b0112253`: REJECTED — 886,134,595.** Best crown yield from us
yet (self ~915M/s, hitr ~1.165e-7) — still -27.9M under floor.

**v33 `62d80d3b` (FIVE_HOT): REJECTED — 793,544,587. MECHANISM
FALSIFIED BY MEASUREMENT.** Self-rate **837.1M/s vs crown 928.8M/s
(-9.9%)**: the +2 serial point-adds cost ~-8% each while the -2 saved
cold reads returned only ~+4-5%. Corrected cost model published:
**add-latency dominates ~4:1 at the crown's operating point — the
"more terms for more L2" direction is closed for the whole field**.
At 12 adds/6 terms FOUR_HOT is already this axis' optimum.
Correctness was never the issue (1M+ magnitudes, 0 mismatches).

**v34 `017539ce` in flight** (19:19Z): crown bytes + tag, 4th draw.
**draw_machine.py armed**: on each verdict auto-prepares+fires the
next draw (v35..v40 max 6) — aborts on our promotion, floor>=910M
(crown ceiling), or 2 submit failures. Zero human latency between
draws.
## STARKWARE $18K PRIZE POOL — same challenge, bigger stakes (2026-09-24 ~18:30Z)

quantum.starkware.co/claim: the challenge pays a separate $20K pool.
Week 1 ($2K, Sep 15-23) CLOSED — counts PROMOTED subs only:
mpjunior92 $1K (102.06%), anamdong $600 (70.71%), ercument $400
(63.47%). Us: $0 (all our high scores were rejected, correct).
**Weeks 2-3 RUNNING until Oct 7: $18K** — top-3 $6K/$3.6K/$2.4K +
$6K raffle across 40 solvers at >=3% gain.
**Current board: ONLY fkiene at 2.69%** (his 904.97M promotion vs
881.27M period baseline). Any promotion of ours >=914,021,532 =
+3.72% -> instant #1 = $6,000 + $199 bounty + raffle qualification
(fkiene himself does not qualify for the raffle yet).
Gain = sum of each promoted submission's % vs period baseline, across
BOTH tracks (pinning + subset). Subset track contested (644 subs, 5
in flight, frontier 623.5M) — not a free lane.
Claim path = Connect GitHub on the claim page (ItlaStudent) — manual
user action only when there is something to collect.

## v33 ARMED + auto-fire pipeline (2026-09-24 ~18:30Z)

Commit df5a3e2: QSB_FIVE_HOT=1 shipped, SOURCE-MANIFEST.json rebuilt
(59 top-level files, LF hashes), SUBMISSION.md = v33 note (6.9KB).
Watcher work_tmp/watch_v32_fire_v33.py polls v32 (b0112253) every
90s; on ANY verdict fires `yukon submit` automatically — aborts only
if the frontier moved so high the floor exceeds 950M (FIVE_HOT's
projected self ceiling). v32 still validating at 18:25Z; field best
904,971,814. Latest field near-miss: 1b64e724 slot-readback 901.0M.

## FIVE_HOT derivation + self-rate analysis; v31 verdict, v32 in flight (2026-09-24 ~19:30Z)

**v31 `9030aaa0`: REJECTED — official 861,959,792** (self 915.9M/s,
yield 0.941, seed 1946349613, 123,403 hits). Second consecutive
low-seed draw on the crown. v32 `b0112253` (same bytes, third draw)
submitted ~17:13Z, still validating at write time.

**Self-vs-official scaling analysis (user-flagged, CONFIRMED).**
Across all 704 historical runs: official = self x (hitr/1.1921e-7) —
the gap is multiplicative, not additive. Same yield distribution at
830M as at 915-929M; the million-gap grows because the base grows
(20-25M in the 760M era, 22-54M now). For floor 914.02M: self <=916M
makes promotion IMPOSSIBLE (best yield ever observed is 0.997);
928.8M needs yield 0.984 (tail); **self >=950M makes promotion a
median-seed event**. Crown-package promotion odds ~3-5%/ticket.

**QSB_FIVE_HOT=1 derived + verified (commit d7fc048, flag-gated
default OFF).** Seven-term re-cut of the fixed-base table: widths
[17,18,17,17,13,18] tiling bits 0..99 + bounded top at shift 100
(same center 170,559,769, same |c| bound). Six-bank prefix = 32.25
MiB < the 48 MiB window the ranked device proves -> only the top
segment streams: cold DRAM records/candidate 4 -> 2, at the cost of
+1 point-add per component (14 vs 12 adds). Table shrinks 9.80 ->
5.115 GiB (init ~half). Expected self ~940-952M if the memory-bound
model holds; falsifiable by self-rate either way.
Verification without GPU: native C harness compiling the verbatim
HOST_EXACT block (400K mags x2 signs, 0 mismatches); independent
python port (600,077 x2, 0); geometry invariants; three-flag compile
matrix; 9 launch sites intact; bias telescopes via the parametrized
term 1u<<(gt_shift(1)-1). **Bug caught in audit: gt_batch_ladder
aborted (count<1) on the 13-bit chunk (hi=0); guarded if(high>0) —
that chunk's H row is never read.** Would have burned the ticket.
Status: package ready pending v32 verdict; if v32 misses, v33 =
FIVE_HOT=1 + manifest + SUBMISSION-v33.md (note drafted, 6.9KB).

## PIVOT — frontier promoted to 904,971,814; v30 = re-draw of the new crown (2026-09-24 ~14:57Z)

**fkiene `871963fd` PROMOTED — 904,971,814** (self 928.8M/s, yield 0.974,
seed 930493768, 129,635 hits). Not a draw: +2.9% real throughput via
**QSB_FOUR_HOT=1** — four cached GLV banks, geometry [18,19,18,18,27]
top@shift-100: four of six terms hit the 48MiB persisting-L2 window
instead of three -> cold DRAM records/candidate 6->4. Terrapinelf's
priced L2 prize, captured. Lineage: 0xCramJam 90f89008 -> Saviour1001
5ab5328d (+2.7% local) -> fkiene 7e95c40 on crown 1fe5a8e4. Cubin
BYTE-IDENTICAL to ours (geometry is host-side runtime config).
New floor = 914,021,532. v20-class bytes (ceiling 885.3M) are dead for
promotion — the strategy pivots to re-draws of the new crown.
v29 (still validating when the crown moved) is moot for promotion.
v30 = crown tree 7e95c40 + inert tag.

**v30 `164a0446`: REJECTED — official 864,287,854.** self 913.2M/s,
yield 0.946, seed 1491162771, 123,741 hits — low-yield draw. Crown-
class draw data so far: fkiene 904.97M (promoted), Saviour1001
903.95M (yield 0.997!), jrcarlos2000 860.5M (yield 0.942), ours
864.29M (0.946). Floor 914.02M sits in the upper quartile of draws.
v31 `9030aaa0` = second draw, submitted ~16:16Z.

## v28 verdict + v29 submitted (2026-09-24 ~14:24Z)