# Pinning rival mechanism map (public Yukon history)

This is a research index for the **pinning** track. It was generated from the public
`yukon submissions --all --json` history on 2026-09-26. Scores below are official
verified candidates/s on the RTX 4090 fixed-time verifier. `accepted + promoted`
means the source became the benchmark frontier at that time; `rejected` means the
submission was valid but did not exceed the then-current best (or otherwise failed
the validation gate). No subset files are involved.

## Current frontier and the last near miss

| submission | status | official score | promoted source | finding |
|---|---:|---:|---|---|
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

1. Preserve the current native sm_89 carrier and the green-context pipeline until
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
