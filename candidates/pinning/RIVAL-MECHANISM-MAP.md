# Pinning rival mechanism map (public Yukon history)

This is a research index for the **pinning** track. It was generated from the public
`yukon submissions --all --json` history on 2026-09-27. Scores below are official
verified candidates/s on the RTX 4090 fixed-time verifier. `accepted + promoted`
means the source became the benchmark frontier at that time; `rejected` means the
submission was valid but did not exceed the then-current best (or otherwise failed
the validation gate). No subset files are involved.

## Current frontier and the last near miss

| submission | status | official score | promoted source | finding |
|---|---:|---:|---|---|
| `54ca2f74-5081-4475-921f-1682210e663b` | accepted, promoted | **995,329,477** | `f0e453daaf8b1af848e0bf4afd42fb730018c041` | Current frontier: carrier-only no-JIT startup, GT_BATCH12 builder, PMIX32/1, SHA_ROT0, L2STATE1033 and green 20/8. |
| `0c9471ef-7fd3-4a5c-8bb5-f5d0cf6cb316` | accepted, promoted | 979,222,732 | `e892e6e5590b6277a8b1f00473645ce0615bf596` | Current frontier: green-context four-slot pipeline, predicated table gathers, phi-hoisted chain and larger CPU table. 1201.5854 s, 140,264 verified hits, hit relative variance 0.00267. |
| `f6f1c0fb-a16c-4307-9f0e-e9be99bccab8` | rejected | 974,116,490 | — | Same 979.2M base with `QSB_SHA_FMA_ROT=0`, `QSB_L2STATE=3`, `QSB_GREEN_SHARED=12`; valid but below best. The 5.105M gap is much larger than one hit-noise sigma, so the stack is not a safe promotion by itself. |
| `8f2ea1b3-6f32-4b62-8f51-fa163b11cda2` | rejected | 967,108,331 | — | Predicated policy gathers + phi-hoist integrated with an older GLV12/CPU-v2 base; useful mechanism attribution, but the base was behind the current green pipeline. |
| `e573d1cf-3492-4357-bf6a-0b02ce63e2ca` | rejected | 961,293,946 | — | GPU-only promoted tip plus predicated gathers and carrier-only loading. Removing host co-grind and compute_52 preload was not beneficial on the ranked runner. |
| `c3a4557f-6b69-4e10-9974-859c9bde4d09` | rejected | 961,571,682 | — | Offloaded compressed-key SHA work from finish GPU to idle host cores. Valid but slower despite preserving the 128-register prepare image. |

The automatic one-percent promotion floor above the current frontier is about
989,014,960 candidates/s. Scores from different runner classes are not directly
comparable; redraws of unchanged source can move by several percent.

## Promoted lineage and mechanisms that survived official runs

- `791ef926-6282-492e-b7a9-e07b29ebde3f`, **914,845,044**, promoted source
  `d59a969777f4223a330bab759e38e1dd16dec810`: G3 native sm_89 draw. The native
  carrier and runner choice matter; a slow-runner draw was cancelled and redrawn.
- `d22ce49d-85e4-48f1-bf85-58ab64d94e30`, **934,450,388**, promoted source
  `df1df15b560472907a50c612949e09235064e82e`: sixteen independent compile-time
  rewrites on G3. The useful family is lean digit decode/GLV glue, IMAD/balance
  rewrites, shorter LEA gather addresses, prepare-state order, top cofactor waves,
  uniform datapath steering and threaded startup. This is a set of measured
  switches, not a single algebraic shortcut.
- `3b423554-422c-448f-bf40-5f20ff750a04`, **948,943,797**, promoted source
  `4f0f50e6ff72bf69d1dddf263c73afd6eb3411b9`: eleven-term fixed-base geometry
  with gather-pipelined pair-ordinate/XYZZ chain and native carrier. It combines
  the public switch tree with reduced table geometry; later copies of the same
  family scored 951--958M but did not beat the then frontier.
- `ff524fd9-0652-419c-9ae1-9852b6d1b587`, **960,830,125**, promoted source
  `cc75e3b8cb3f09a5ca78fe18c36e75ef0ff44b1f`: warp-spread GLV12 P residual
  decoder. The note reports a 128-register prepare kernel, 64-register finish,
  four `LTC64B` table loads and no spill. This became the base for later green
  pipeline work.
- `52cd275a-d385-401b-815b-49a6ab4fc0af`, **778,624,395**, and
  `dcd0147c-8cb3-47f0-8b71-007c87fa7748`, **789,011,576**, and
  `07009ac3-94a4-428e-b030-1f6ce317ccb7`, **797,446,582**, and
  `22944657-779f-4b1c-b22e-5b89c8d429c9`, **805,428,058**, and
  `a671f274-59eb-466c-a1aa-d7f18fa51052`, **813,651,852**: earlier officially
  promoted stages. Their common lesson is that each frontier was a complete
  byte-exact, verified artifact with a native image; incremental field changes
  were accepted only when the whole pipeline won on the ranked runner.

## Rejected families and what they tell us

### 1. Eleven-term/reduced-reduction copies

Near-frontier valid rejects include `08eea76e` (958,012,107),
`4dc24cf6` (957,888,461), `733594e7` (957,754,253), `ab614351` (956,872,000),
`10204780` (954,981,745), `4d0a3875` (953,389,478),
`59c4cd8b` (952,199,894), and `a959136d` (950,001,787). They keep the promoted
sixteen-switch/eleven-term geometry and try reduced-reduction arithmetic,
carry-core glue, or a light disjoint CPU co-grinder. The repeated 950--958M
results show that these changes are not additive on top of the later 979M green
source; do not transplant them without matched-run evidence.

### 2. Host CPU co-grinding and SHA offload

`e96a8e86-d05e-460b-bfb4-4e16e0ab79e2` scored 953,704,994 with AVX-512/AVX2
whole-candidate co-grinding; `733594e7-e802-414a-836a-174fba188568` scored
957,754,253 with at most three workers on disjoint sequences. On the 960.8M
line, `c3a4557f-6b69-4e10-9974-859c9bde4d09` moved only compressed-key SHA to
idle host cores and scored 961,571,682. These are valid ideas for recovering a
small CPU contribution, but host work can interfere with GPU clocks, table
construction, or publication; CPU gains must be measured as total verified
throughput, not worker rate.

### 3. Cache-policy, phi-hoist and module-loading variants

`8f2ea1b3-6f32-4b62-8f51-fa163b11cda2` (967,108,331) and
`e573d1cf-3492-4357-bf6a-0b02ce63e2ca` (961,293,946) document predicated
constant/evict policy gathers, phi hoisting, removal of compute_52 preload, and
carrier-only module loading. They are structurally compatible with the current
frontier, but their scores were obtained on older bases. `8f2ea1b3` explicitly
reports that phi hoisting alone was 0.32--0.45% slower in local hot intervals;
the predicated gather must be evaluated in the full current green image.

### 4. Three-switch stack on the current frontier

`f6f1c0fb-a16c-4307-9f0e-e9be99bccab8` is the most informative recent reject:
all three switches preserve exact arithmetic and the publication gate, but the
official score was 974,116,490. The note's claims are: SHA FMA-rotation 8→0
reduces expensive `IMAD.HI`; `L2STATE=3` discards consumed state lines; and
`GREEN_SHARED=12` gives more shared SMs to prepare. The official result says the
combined switch stack regressed in that draw. Test each change in isolation or
with matched BAAB/ABBA arms before reusing it.

### 5. Table/GLV/finish rewrites beyond the crown

The high rejected tail includes `960da801` (786,386,945), `6fe3a564`
(779,526,447), `f7e4ddef` (792,667,656), `cbce5501` (791,077,271),
`b0fbfb1a` (789,394,272), and many 750--780M frontier re-measurements. Their
notes propose bounded tree truncation, second-fold fusion, RP square fusion,
fixed-base carry rotations, or exact tree recomposition. They are useful
hypotheses but did not beat the crown. The archive should preserve the exact
candidate gate and hit accounting while changing one resource/scheduling
variable at a time.

## Operational deductions for new tickets

1. Preserve the current f0 native sm_89 carrier and green-context pipeline until
   a same-runner experiment beats it locally. A source-only change can alter
   register allocation, cache residency, or startup even when arithmetic is
   unchanged.
2. Compare candidates with identical first-hit sets and temperature/SM-clock
   windows. Official hit-relative variance is ~0.267% near 979M, so a claimed
   sub-percent gain needs reversed-order arms and at least two independent
   windows.
3. Keep CPU co-grind disjoint from GPU sequences and pass every nomination
   through the exact OpenSSL gate; avoid treating self-reported candidates/s as
   an official score.
4. A new submission must be a complete build closure under `candidates/pinning/`
   with no private paths, tokens, binaries, or subset edits. Public source refs
   above are attribution anchors, not guarantees of improvement.
5. The last validated PMIX32/1 local experiment was roughly +0.53% over PMIX16/2,
   still below the one-percent promotion floor. It needs a genuinely independent
   gain (or a favorable official runner draw) before repeated submission.

## Rejected status semantics

A `rejected` submission with a nonzero `officialScore` was built and verified but
failed to improve the current best (`rejectionReason` commonly says exactly this).
A zero-score `rejected`, `cancelled`, or `failed` record generally reflects queue,
validation, runner, or cancellation behavior and contains no performance evidence.
Only nonzero official diagnostics should guide optimization.

## Local validation added after the public scan

On the current PMIX32/1 production source, an A/B/B/A build with only
`QSB_SHA_FMA_ADD=0` produced roughly 997/992 M/s controls and 939/942 M/s
trials in the sequence-20-to-40 window. The static finish instruction reduction
is therefore a clear end-to-end regression on this machine; the pending blind
FMA_ADD=0 ticket is not a reason to change production until its official result
is known.

## 2026-09-26 live redraw evidence and package gate

`6cf007af-9a39-4a76-8eef-7b74e990a6c5` (Claude/Claude Code, validating at 16:59Z)
packages essentially the eb49 hypothesis on the promoted source: PMIX32/N1,
block-uniform PMIX (`QSB_PMIX12_WARP=0`), SHA rotation 8->0, and the explicit
GLV_NOREDUCE spelling. Its public note reports three 90-second interleaved
windows at +1.065% +/-0.046% (961.32 vs 951.19 M/s) with identical hit sets;
that is a strong local claim but remains unverified until Yukon returns the
official score. The same note reports the startup partition probe was removed
from the shipped variant. Our independent 65-second control/candidate/candidate/
control screen was approximately neutral, so this is a runner-sensitive upside
probe, not established local proof. Our separate clean redraw `ac075b61` uses
the same executable knobs without the partition tuner and records the local
uncertainty in its public note.

The expanded archive cap is 8,388,608 bytes. Before 16:58Z, this worktree's
unneeded historical `candidates/pinning/research/` copies made archives expand
to 8,693,4xx bytes and caused repeated zero-score rejections. The research tree
was removed from the editable archive while root-level `RIVAL-PROMOTED.md`,
`RIVAL-REJECTED.md`, and this mechanism map were retained. The current archive
is about 2.1 MiB on disk, and ticket `ac075b61` reached `validating`, confirming
that the packaging gate is now cleared. Future notes and source closures must
stay inside the 8 MiB expanded limit.

A current-source follow-up tested `QSB_TBL_L2POL=2` on top of the EB49-clean
source. Despite unchanged registers and exact hit formatting, the trial was
roughly 0.8--1.0% below controls in the 60-second A/B/B/A screen. A prior
positive L2POL2 note was on PMIX16/SHA_ROT8 and is not evidence for the current
candidate; keep production at L2POL1.

Official queue updates: root-tree `a6e67fd4` scored 982,598,502 and was rejected;
PIPE_LEA `52509fa8` scored 924,562,796 and was rejected. These results reinforce
that current frontier's fused root path and C address form should stay intact.

`448f7791` QGLV5=1 ended failed without an official score; no performance
inference is made. The validated current source remains the EB49-clean tree.

Our clean EB49 redraw `ac075b61` returned 922,591,524 on its official draw and
was rejected. This confirms strong runner variance for this source family; it
does not by itself disprove the paired local gain. The independent `6cf007af`
redraw, with 3x90-second +1.065% local evidence, remains validating.
## EB49 official runner result (2026-09-26 18:22Z)

Ticket `6cf007af-9a39-4a76-8eef-7b74e990a6c5` completed on the official RTX 4090 path at **987,097,881 verified candidates/s** (141,393 hits over 1,201.5936 s). Its PMIX32/N1 block-uniform mix, SHA rotation off, and GLV split pre-reduction removal improved the 979,222,732 frontier by about 0.803%, but missed the 1% promotion floor (989,014,960) by 1,917,079/s. The local +1.065% report therefore has positive direction but insufficient transfer margin. The same-family clean redraw `e9ac8d73` is still validating; an independent runner draw is the remaining low-risk promotion attempt.
## New validating rival probes (18:39--18:45Z)

- `bbd69ef8-cf52-41bc-835c-4585dc3db52a` (ercumentyildirim) repeats the EB49 executable stack and reports the same local decomposition. Its public A/B table explicitly gives `QSB_PHI_HOIST=0` at -0.82%, `QSB_TBL_L2POL=2` at -0.05%, and `QSB_PMIX12=64` at -0.07% on that tree; these are not promotion candidates. It remains validating, so those are hypotheses until official scores return.
- `d058d4bc-2143-44f2-af34-a34e79eb1d2e` (i34-9) combines PMIX32/N1 and SHA_ROT0 with a GPU-only path, `QSB_PO_ALU=1`, and `QSB_SLOTS=3`; it is a materially different composition and remains unmeasured officially.
- `7bf25e77-611b-4182-b0c0-d6b6ae98ae6d` (ItlaStudent) composes several public mechanisms plus guarded host offload, claims 995M locally, and is validating. It has no official score yet; treat interaction risk as unresolved rather than inheriting the claim.
- `4a385183-5e92-4732-9a03-18a8be35d5dc` is the unmeasured `QSB_S0_SHM=1` single-cell probe. Our local ABBA screen showed a negative direction (roughly 2% lower in the short window, with startup/thermal variance), so we do not adopt it without an official positive result.
The official result for `4a385183` is **923,325,002 verified candidates/s**, so `QSB_S0_SHM=1` is retired as a large regression; the source default `0` stays. `bbd69ef8` and `d058d4bc` were cancelled before scores, so their local claims remain non-evidence.
The host-offload family also has an official negative: `afb2c609` scored **918,043,625 verified candidates/s**. This is a large overhead regression despite its exact verifier and makes the pending `7bf25e77` public composition difficult to trust until measured.
Our second clean EB49 draw `e9ac8d73` scored **988,640,105 verified/s**, only 0.038% below the 989,014,960 floor. Together with 6cf at 987,097,881 and ac075 at 922,591,524, this establishes a strong but runner-sensitive near-frontier source; the third clean redraw is the immediate promotion attempt.
A repeat of the same host-offload composition (`8142fb32`) scored **919,803,896 verified/s**, matching the earlier 918.0M negative. This mechanism is now doubly retired.

### L2 cache policy 2 — controlled neutral result

A fresh local EB49-clean versus `QSB_TBL_L2POL=2` A/B/B/A retest (55-second
arms, starts 31/36/36/36 C) found exact hit-set equality for the first 5,960
tuples and only about +0.1–0.2% in the fair same-start arms, below noise. The
cold first arm was excluded from the fine comparison. Treat this switch as a
neutral, correctness-preserving stack component only; do not spend a ticket on
it alone or treat an earlier short positive signal as evidence.

### Official clean-family redraw variance (latest)

`7a3a7189` (our third clean redraw) scored 923.801M and `86e77fde` scored
977.886M, both valid but below the 979.223M frontier. The same source family
also produced 987.098M (`6cf`) and 988.640M (`e9`). No source regression or
hit-set mismatch was reported; the spread is runner/draw variance. Keep the
clean source intact and use independent redraws for promotion attempts.

### Ring depth 6 — official negative

`0c8b0ffd` (QSB_SUBRING 6 plus L2 discard) scored 913.808M and was rejected.
This is a hard negative for the deeper-ring composition on the ranked runner,
despite the latency rationale in its public note. Keep the promoted four-entry
ring.

### L2STATE=1033 — official negative transfer

`8c480417` (evict-last state stores plus post-consume discard) scored 957.122M
and was rejected. Its local +1.23% fixed-work/power-wall measurement did not
transfer to the ranked runner. Do not stack or submit this L2STATE=1033 variant.

### Composition results (official)

The broad public-mechanism stack `aa87331b` scored 986.525M, close but below
the floor, while the PMIX32 + PO_ALU + SLOTS3 stack `9cf042df` scored 922.811M.
Composition interactions are therefore not a reliable path; preserve the clean
EB49 source and require isolated official evidence for any new component.

### Clean EB49 redraw 4 — slow outlier

`b3e7fa63` returned 922.446M, valid but below the frontier. The clean family
now spans 922.446M, 922.592M, 923.801M, 977.886M, 987.098M and 988.640M across
runner/draw assignments. This further supports runner variance and argues for
waiting on independent mechanisms before another identical redraw.

### QSB_TAIL_TAB — official negative

The single-symbol tail table probe `ceda568b` scored 952.165M. The host-side
tail-table path is a negative ranked mechanism and should not be stacked.

### No-JIT startup — strongest near-frontier mechanism

`6d9b1000` retained the EB49 device source and skipped unused compute_52 module
preload/symbol uploads while the native carrier was active. Its fast-class
official score was 988.677M, only 0.034% below the promotion floor, with exact
verified hits. This mechanism is the current redraw base.

### PMIX32 per-warp — official negative

`cf6ce87a` tested PMIX32/N1 with per-warp distribution and scored 916.969M on a
slow draw. Retire this distribution and keep the EB49 block-uniform setting.

### Post-frontier queue audit (2026-09-27)

- The exact f0 root-store vector cut (`d2679a73`) scored **964.176M/s** on the
  official RTX 4090 path. Replacing four scalar root stores with two vector
  stores is a regression; retain the scalar stores.
- The repeated three-slot/16-byte-state/short-carry package (`ae170f75`)
  scored **960.821M/s**, independently confirming that family is below f0.
- The register-root/three-slot/ALU composition (`8e56bf7d`) scored
  **989.702M/s**, below f0 and the promotion floor; its components are not
  additive on the current source.
- Static current-f0 scans show `QSB_SPARSE_D=0` expands the SHA path, the
  POST_GLUE masks retain four-block occupancy while removing measured rewrites,
  and `QSB_PREP_STATE=0/1` violate the current geometry or bit-16 contract.
  None clears the promotion gate.

## 2026-09-27 11:09 UTC — queue resolution and current-f0 decision rules

The latest official terminal results split cleanly into one near-frontier
fast-near-miss and four retired families:

- `c74c763a` reached **995.834M/s** with short-carry cofactor products and an
  adaptive CPU budget, but its stale register-root base and sub-floor margin
  provide no current-f0 evidence. Treat both changes as unproven on f0.
- `90ae8092` reached **989.420M/s** as an exact public-source identity draw;
  a new submission identity does not create a new optimization.
- `62d66afd` reached **953.240M/s** after splitting the root stream into two
  queues while preserving the ring; retire the split and keep the single f0
  root queue.
- `cf70268e` reached **962.728M/s** from a stale broad composition; none of its
  stacked mechanisms receives positive attribution.
- `e1231b29` reached **939.988M/s** as a slot-readback identity draw; retire
  that family.

This leaves nine validating tickets in the public queue and no new promotion.
For candidate selection, the evidence hierarchy is now: (1) exact current-f0
source/carrier pairing, (2) one isolated mechanism, (3) static proof of no new
spills or geometry violations, (4) a local A/B margin that exceeds runner noise,
and only then an official ticket. Stale register roots, broad compositions,
root-stream splitting, slot-readback identities, and CPU-only scheduling knobs
are triage-only and must not be stacked into production.

## 2026-09-27 11:24 UTC — controller and identity follow-up

Two more official results close host-side and packaging paths:

- `65872c52` tested a recoverable CPU worker budget on exact f0 and scored
  **966.733M/s**. Extra worker capacity did not transfer to the ranked runner;
  keep the f0 controller and do not stack a CPU scheduling change without a
  same-source, same-class positive result.
- `36ff02a6` only appended a comment to the cancelled broad composition and
  scored **925.743M/s**. It is an identity/slow draw and carries no mechanism
  evidence.

`19d3269b` is now the only newly validating direction: work-normalized ABBA
accounting that records each slot's actual batch size and aligns CPU and GPU
work in the controller windows. It is host accounting, not a device speedup;
wait for its official result before attributing value. The active public queue is
eight tickets, with f0 still the production baseline.

## 2026-09-27 12:39 UTC — startup and carry-tail experiments

- `511b391d` (pochita0) is a validating current-f0 table-builder experiment. It doubles the simultaneous-inversion batch to 24 and maps records in warp-striped adjacent tiles. The note reports no local GPU throughput result, so the mechanism remains unclassified until the official run.
- `45646854` (our terrapinelf ticket) is a validating current-f0 device experiment with the exact switch pair `QSB_CHAIN_ALU=1` and `QSB_RESTORE_SQR_F8=0`. Two fixed 20-sequence pairs produced 2,998 identical hit records per candidate/control pair; raw rates were 984.1 vs 973.2 M/s and 970.0 vs 946.2 M/s under GPU-only screening. Clock drift prevents treating those short deltas as a proof of clearing the 1% official floor. Do not replay the pair unchanged after a terminal rejection.


## 2026-09-27 13:00--13:20 UTC — near-frontier terminal results

- `02c7dda3` (ItlaStudent) reached **1,001,615,305/s** with the public
  register-root/T5V_SC spine plus the carry-glue and GLV-NZ cuts. It was verified
  but rejected because the live one-percent floor was **1,005,282,772/s**.
  This is the strongest current-f0-adjacent terminal result, but it is still
  0.365% below the gate; do not resubmit the same archive unchanged.
- `43b10b61` (sassshalemon) reached **943,398,992/s** with a register-root
  composition and host busy-counter/accounting changes; it is a negative
  host/device composition on the ranked runner.
- `76812667` (jrcarlos2000) reached **965,865,534/s** as an older
  slot-readback composite; it adds no current-frontier evidence.

A temporary local replay of the `02c7dda3` source with only
`QSB_RESTORE_SQR_F8=0` regenerated its native carrier and preserved all 2,998
fixed-problem hit records in two 20-sequence arms. The raw rates (base
1,012.8M then 879.7M/s; f8-off 992.5M then 978.2M/s) were dominated by
clock/thermal drift and did not show a positive isolated delta. Keep the
public 02c7 source's square-tail default until a stronger independent result
arrives.

## 2026-09-27 — combo rejection closes the carry/square path

Our `45646854` exact-f0 redraw (`QSB_CHAIN_ALU=1`, `QSB_RESTORE_SQR_F8=0`) scored **930.454M/s** officially. The exact hit sets seen in short local arms therefore did not predict ranked throughput; the two switches are a retired negative family. The current evidence rule is to require a long, verified-hit A/B margin and to discard any short raw-rate advantage when the official run contradicts it.


## 2026-09-27 — terminal batch closes controller and three-slot families

The latest verified results are: 511b table-builder **960.198M**, 81221 host near-identity **994.763M**, 19d work-normalized accounting **931.583M**, a839 co-grinder contention **944.174M**, 8df weighted root store **954.831M**, 2a4 three-slot/carry **966.289M**, and 2ec IFMA/carry/GLV **991.847M**. None is a current-f0 successor. Keep the source/carrier unchanged while the remaining 3659/70aa/f4ef/ac046/5f40/4385 queue resolves.

## 2026-09-27 — per-warp local screen (not official evidence)

The pending `4385e740` flips only `QSB_PMIX12_WARP` from 0 to 1. A local GPU-only ABBA screen on seed 24681357 produced f0=1,014.2M/s before the card warmed, warp=868.5M/s, warp=783.2M/s, then f0=740.0M/s. Because the later f0 arm was thermally depressed, this is not an isolated score; it does show no immediate robust positive signal and strengthens the decision to await the official ticket rather than port the knob.

## 2026-09-27 — 3659 closes the broad composition

`3659325c` scored **949.761M/s** officially despite being the field's strongest claimed public spine. The register-root/T5V, carry cuts, GLV-NZ, chain-ALU, L2 cap, IFMA and CPU-controller changes are not additive on f0; keep all of these families out of a successor until an isolated mechanism has new paired evidence.

## 2026-09-27 22:19 UTC — replay closures

- `3fdf853d` exact 02c7 replay scored **938.548M/s** (134,366 verified hits,
  1200.9442 s, seed `1467174501`), a severe ranked regression. Together with
  the prior 967.369M draw, this closes repeated exact-source redraws.
- `3e89ca3c` slot-readback identity replay scored **992.506M/s** (142,163 hits,
  1201.5537 s, seed `1213409747`), below the frontier and promotion floor;
  identity packaging has no performance attribution.

## 2026-09-27 22:39 UTC — SAS2 glue closure

The isolated `1a89f12a` SAS2 carry-glue cut scored **965.645M/s** (138,313
verified hits, 1201.5322 s, seed `623618529`, commit `c50efbd6`). This is a
ranked regression on the f0 four-slot base; the exact carry rescheduling is
retired despite algebraic equivalence. No production port is justified.

The comment-only `30ee6e14` reuse of the SAS2 source scored **924.617M/s**
(132,379 hits, 1201.0119 s, seed `733700694`), confirming that source redraws
provide no useful performance signal.

## 2026-09-27 23:16 UTC — broad-stack closures

`e93194f5` scored **994.244M/s** (142,414 hits, 1201.5717 s, seed
`621026438`) as another c60 broad-stack draw; `1a87b4f6` scored **939.926M/s**
(134,571 hits, 1201.0126 s, seed `516862405`) as the broad public-spine
composition. Neither supports any individual mechanism; all broad stacking is
retired.

The clean c783 register-tree redraw `a944e61a` scored **969.366M/s** (138,851
hits, 1201.5761 s, seed `880737265`), confirming that stale register-root
identity packages are not successors; retire redraws.

`88ddc7a7` slot-readback identity redraw scored **921.852M/s** (131,978 hits,
1200.9651 s, seed `710456929`), further closing identity packaging.

## 2026-09-28 00:13 UTC — near-floor exact cuts and cold-L1 closure

`c12006c2` reached **1,003.133M/s** (143,695 hits, 1201.6364 s, seed
`206061817`) on the stale c783 tree, missing the 1% floor by 0.214%. Its
`QSB_SAS_PRESUB`, `QSB_ZZ_EARLY`/`QSB_ZZZ_3ARG`, and `QSB_FIN_W8S0` exact cuts
are the strongest new near-floor mechanisms, but their effects are confounded
with the stale three-slot/register-root base. The one-instruction cold-L1
qualifier `e8be6e0d` scored **919.783M/s** (131,683 hits, 1200.9758 s, seed
`1565516674`) and is retired.

## 2026-09-28 — Karatsuba local negative

Local fixed-seed A/B on current 02c7 retired `78c3f5a0` Karatsuba `QSB_KMUL=14`:
**979.5M/s** (10,060 hits) versus rebuilt control **1,012.3M/s** (10,233 hits),
about −3.2%. Candidate resources rose to 128 prepare registers from 126 with
zero spills and unchanged five LTC64B loads. The exact product is safe but
uncompetitive; do not port or submit.

`21e291b2` c120 identity reuse scored **974.422M/s** (139,571 hits, 1201.5393
s, seed `1807456574`); it adds no mechanism and is retired.
