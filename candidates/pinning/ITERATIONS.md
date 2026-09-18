# Pinning experiment ledger

## Karatsuba native screen, after subsetPR86 upload

Both own entries remain pending: pinningPR74 and subsetPR86. No visible GPU
progress distinguishes them; earlier pinning is the provisional priority.
The isolated research/karatsuba candidate replaces only device multiply with
one-level difference Karatsuba, preserving the corrected reduction and square.
21,465 actual-PTX semantic full-product/reduction cases pass; native production
sm89/default builds pass. Control source hashes match submittedPR74 exactly.
However ranked prepare grows7817→9221staticinstructions and122→128registers,
with new4/8byte spills. Finish and fused also grow and gain spills. This
specific schedule is not selected; raw64→48products did not translate to a
better full-kernel resource profile. No measured slowdown or universal
Karatsuba rejection is claimed. Pending production remains unchanged.

Process correction: exact baseline include hashes are now required before
native comparisons. The earlier adaptive report had stale host-loop hashes;
production-native-results.json is the verified control for this experiment.

## Research priority

The user requested large improvements only. Future research should target
structural gains, using roughly 25%+ as a working prioritization target rather
than chasing small increases above the 1% promotion floor. Predicted gains need
a concrete cost argument and remain unconfirmed until official GPU evaluation.
Current priority follows submission availability: free track first; with both
pending, prepare the likely first finisher. Progress is stronger evidence than
submission age, which is only a tentative ordering heuristic. The recurring
follow-up now permits research and preparation, but no upload or cancellation.

## Initial setup — 2026-09-16

- Work directory: repository root returned by `yukon clone`.
- Yukon: `v2026.09.12-1`; schema v2; selected track `pinning`.
- Base commit: `1776cde0ffbc0c3b6ddb8b4748708cdc2017c8e2`.
- Frontier: `ae99b9ad-82b9-49fe-ac8d-4d3bd68896d5`, `1b62e99`,
  197,764,166 verified candidates/s on RTX 4090.
- Setup: passed CPU verifier smoke; CUDA unavailable locally.
- Ranked baseline: attempted, unavailable because official bridge is absent.
- CPU diagnostic (seed 0, N=6, fixed_hits=3): 134 candidates; 3/3 verified;
  11.8 s; score 8 is CPU diagnostic only, not a claimed GPU score.

## Experiment 1 — defer affine normalization

- Change: carry homogeneous `u1*G` into recovery; normalize the two recovered
  points together. Saves one inversion and two multiplications per candidate.
- Retain `(256, 2)` launch bounds and 1,048,576-candidate batches.
- Local check: 547 scalars, 1,094 recovered keys, affine debug wrapper all
  match OpenSSL; one inversion per production recovery.
- CUDA compilation, PTX arithmetic and speed require the official GPU runner.
- Submission: `8150e0be-d5f7-4a2c-bcb7-bec3d0a4cc64`; **rejected**;
  no GPU score claimed. Exact attribution: GPT 6 Astra xhigh / Codex.
- Submitted `pinning.cu` SHA-256:
  `797440c9e3c72c54bbe2cfee246472731d737879aa04c5ee612d81e3a6afdd97`.
- Process improvement: source-extracted mathematical checks and explicit
  separation of CPU diagnostics from candidate GPU results.
- Official result: **226,444,961 verified candidates/s**, +14.5% over its
  original 197,764,166 baseline but 2.98% below the current 233,402,654 frontier.
  This completed evaluation frees the pinning slot. Subset PR27 remains pending,
  making pinning the current research/submission priority.
- The original results-only schedule was superseded by the user's subsequent
  research and two-track prioritization instructions. The existing 20-minute
  heartbeat now checks both tracks and prepares successors under that policy.

## Experiment 2 — strongest pending pipeline, under review

- Strongest inspected base: PR24 `6e76a74fed8e6e5b8439e64ec20f586085f37d52`,
  alvaroborras's specialization of nullforest8200 PR17. PR38 independently ports
  the same development frontier; its author reports 645,625,292 verified/s on
  a 90-second RTX 4090 run. This is public author evidence, not our GPU result.
- PR41's 24-bit/10.7-GB table reports 363,988,688 verified/s, below the stronger
  pipeline. Other inspected entries largely duplicate mechanisms already in it.
- The best pipeline already uses shared direct XYZZ recovery and hierarchical
  inverse trees; do not count these as new improvements. Inspect checkpoint
  compression and wider mixed windows for incremental gains, and retain the
  known final-carry correction before selecting an arithmetic implementation.

## Ongoing goal research — field base and affine screen

- Refreshed both tracks: subset PR27 still validating with no later scored
  subset entry observed. Pinning PR11 promoted at 249,134,266; its mechanisms
  remain weaker than the strongest inspected pending pipeline. Newjordan's
  911f664 note reviewed; no new mechanism beyond that stronger base.
- Preserved PR24 source with provenance and produced an isolated corrected
  header. Both host and actual-inline-PTX semantic models reproduce the old
  carry defect and pass after repair: 100,900 host calls and 4,360 PTX-model
  calls. A targeted mutation also catches missing `.cc` on the final addition.
  Source details and limitations: `research/FIELD_BASE.md`.
- Main candidate source and prepared subset source remain unchanged. No CUDA
  result or performance claim. Only Apple Clang is installed; its listed
  targets exclude NVPTX, and no local container runtime is present.
- Screened affine batching as a larger architecture, with a nominal point-chain
  reduction from 95M+28S to70M+14S before other costs. Naive repeated checkpoints
  are too costly to justify selection; a fused schedule needs an explicit
  traffic/lifetime proof. See `research/AFFINE_PIPELINE.md`. Goal remains active.

## Experiment3 — ten-window candidate submitted

- Submission:66c031c6-16e4-484e-ad4d-0e7e0ef3f38f, validating, PR74,
  head154b2bd97a9c918ac3d992c31a6cf7901aeab1a0. Frontier644,546,620.
- Exact source fingerprint:ed600d1c31cc168b0010419da1dfafc78d1be147b816e78dbab52cd37f2e5e54.
  Entry SHA:a30d5228ed06d01807d3197275b3998f0b4cd3f91ad103c82fbf1df2d39faf03.
- Ten signed windows reduce point chain95M+28S to60M+18S;16GiB runtime table,
  bounded chunk builder/batched host ladders, corrected PR64 bundle. Runtime
  comparison selects external or CTA-local inverse on distinct real ranges.
  Six trial groups capped at4 batches; all hits retained. No claimed score.
- Exact production sm89 and default CUDA12.8.93 builds PASS; split122/78regs
  without stack/spills; fused126regs,120B stack,zero register spills.
  CPU12769recodings/414curves/822keys; builder101506address/4518entry cases;
  selector11UBSan cases; host36schedules/1092launches/9756records allPASS.
- Public notes:submission-wide-windows.md; source/check evidence under
  research/wide_windows/. Prior rejected source preserved separately.
- Attribution:GPT6Astra xhigh/Codex; coauthors alvaroborras,Saviour1001,newjordan,
  bndbww7w6w-cmyk,MakiRH4,jacklightChen,Meganpark980320.
- Submission is an expected substantial attempt, not a measured win.16GiB
  gather behavior and startup remain risks; runtime schedule choice does not
  prove wide geometry beats the cache-resident base. Grok reviewed source,
  warned about uncertain speed and prompted the trial cap; Gemini503 draft
  was not used as evidence.
- Workflow improvement: immutable native include-closure checks, publication
  scan excluding generated bytecode, actual host scheduler/selector audits,
  and corrected full checkpoint traffic accounting. No trusted harness edits.
- Both tracks now validate. Next priority subsetPR60 by age as a tentative
  heuristic; no GPU progress/FIFO guarantee exposed. Never cancel either.

## Post-PR74 research — arithmetic screens and Muse feedback

- Both own PR74/PR86 remain validating. Pinning takes research priority by
  submission age only; no GPU progress currently establishes finish order.
- Isolated difference-Karatsuba passed21,465actualPTX semantic product/reduction
  checks and native compilation, but exact-control ranked kernels grew and
  acquired spills. Not selected. See research/karatsuba/README.md.
- Muse now has five completed zero-reported-cost research rounds, all reviewed.
  Corrected its4x64product/register comparison and lazy-reduction claims.
- Added research/lazy_field/screen.py: source-extracted conditional magnitude
  analysis finds three contract violations in a literal lazy conversion. One
  weak normalization at the seed satisfies the rest of the ten-window chain.
  This is not an implemented GPU field backend or a speed result.
- Conservative5x52/10x26layouts add products/storage;9x29overflows32-bit limbs
  under the selected magnitude contract. No substantial net gain established.
  Deferred pending a concrete efficient primitive schedule and native evidence.
- Public PR88 combines already-known small schedule/constant-coordinate ideas;
  PR89 changes host hit readback only. Notes reviewed, source unreviewed.
  Neither supplies a new large mechanism. PR68subset scored452720843 and was
  rejected; PR85was cancelled by its owner. Both were earlier than ourPR86.
- All9recorded PR74production files still match the submitted hashes.

## Post-PR74 research — two concrete carry schedules screened

- Built isolated radix29column and narrow-MAD versions of the current field
  multiply. Production, specialized square and corrected reduction are intact.
- Actual generated PTX semantic tests: radix29=23,025fullproducts/residues,
  narrowMAD=20,961; carry/packing mutations rejected. Reports now bind the
  semantic-model and checker hashes as well as device source.
- Both fullsm89/defaultCUDA12.8.93 builds pass. Exact PR74controlinclude
  closure matched. Radix29prepare grows7817to11661staticinstructions; narrow
  MADprepare7817to7819, with a new8/8Bfinishspill. Neither selected.
  No GPUtiming or universalfamilyrejection follows from these static screens.
- Gemini completed a source-only review but incorrectly describedmadc.hi;
  officialPTXsemantics and an explicit regression check correct it. Grok's
  bounded240secondrequest timedout with nofinalresponse. See the experiment
  READMEs and narrow_mac/external-review.json.
- CompilerVM stopped after builds. Bothsubmittedclosures remain unchanged.
- Public pinningPR91combines existingexternalinverse and two sparseSHAs;
  priorcomponent scores647007541/646395221remain belowpromotionthreshold.
  SubsetPR90cutsdiagnosticI/O; PR92restoresoldblockinverse; PR93combines
  alreadyknownrankedtemplates/constantpoint. No newlargearchitecture found.
  PR92's claimednaive1023multiplies is contradictedbyexactPR62helper:
  255upsweep+510downsweep=765. SourceofnewPR92stillunreviewed.


### 2026-09-16: compact/wide and inverse interaction prototype

Prepared `research/adaptive_geometry/candidate` beside unchanged pendingPR74.
Four complete path variants use distinct real warmup/timed batches and retain
all hits. Compact mandatory, wide optional after actual search allocations.
Each geometry passed12,769 recodings,414OpenSSLcurve chains and822recoveredkeys;
policy and allocation failures/lifetime passedUBSan stubs. NativeCUDA12.8.93
sm89/default builds passed. Exactwide rankedresource/count summariesmatchPR74,
all0spill; compactprepare128regs. ExistingcompactbuilderCPUevidence source-bound.
VMstopped. NoGPUtiming or execution, no production change or submission.
This prepares a fairer comparison/fallback but adds no new arithmetic gain over
our pendingwidecandidate. Awaititsresult andseek furtherdominantwork reduction.
Latestqueue: bothownvalidating; earlierb8781c8rejected605686827. No later scored
completionobserved. Muse8completedfree andreviewed; nonew substantialfinding.


### 2026-09-16: three-field checkpoint architectures

Implemented full-tree rebuild, half-tree rebuild, and pre-scaled half-tree
variants under `research/compact_state/`. Algebra saves one of four fields by
changing the collective denominator. CPU source kernels and independent affine
oracles pass:5000 scaled recoveries,2048 tree leaves,1856 pipeline candidates
including singular/tail lanes. RealCUDA12.8.93sm89/defaultbuilds pass.
Full/half/scaledhalf finish introduces8/32/16byte spill stores and loads,
respectively; staticfinish instructions5325/5346/5045 versus4598control.
No variant selected for a new submission; logicaltraffic reductions alone
do not support a substantialperformancelead. Field/point arithmeticcontrol
unchanged; compilerVMstopped. Gemini's defectclaims contradicted caller/native
checks; Grok180secondreviewtimedout. Preserveboth pending evaluations.


## 2026-09-17 — PR98 checkpoint and warp synchronization transfer

Preserved all nine production files from pending PR74. Implemented three isolated
variants in `research/warp_checkpoint/`: warp-only full checkpoint, upper62 compact
checkpoint, and compact plus fused-tree warp synchronization. Exact public head
2d5eba297f8d9823ca0f5a552d3d1788fbe3031c; credit Meganpark980320 for unpromoted reuse.
Muse round9 also recommended this already-observed public mechanism.

All three pass native CUDA12.8.93 sm89/default builds and CPU pipeline checks.
The compact and combined variants pass C++ ThreadSanitizer; two deliberately
missing cross-warp barriers are detected. Combined prepare122/finish79/fused126
registers, zero spills; finish5032 static instructions versus4598 control.
External checkpoints save48B/candidate for0.75 extra field multiplies; block
barriers16 to8 externally and16 to7 in the fused inverse. No GPU execution or
throughput measurement. Retain as a promising successor component, not an
established large gain. No new submission while both tracks remain pending.
See `research/warp_checkpoint/README.md` and source-bound `comparison.json`.


## 2026-09-17 — PR74 rejected after valid GPU evaluation

Official score407376555 verified candidates/s, versus644546620 frontier
(-36.7964%). Source154b2bd97a9c918ac3d992c31a6cf7901aeab1a0, unchanged
production fingerprinted600d1c31cc168b0010419da1dfafc78d1be147b816e78dbab52cd37f2e5e54.
58318 verified hits over1200.8714s onRTX4090, seed274283709. Source executed
correctly for the official run but the wide composite substantially regressed.
Arithmetic/lookup reductions alone were insufficient; no profile attributes
the loss to a particular bottleneck. Public diagnostic artifact contains no
fused/external selection or table startup log. Full result in
research/wide_windows/official-result.json.

The new warp_checkpoint component is not established to recover this deficit;
retain it as a possible supporting change on a stronger base. Do not submit it
alone as an expected large win. adaptive_geometry still offers a compact
comparison path, but has no official throughput. Subset PR86 failed without
a score, so neither own track is pending. Prefer the strongest substantial
successor while retaining both terminal source snapshots for diagnosis.


## 2026-09-17 — Bend two-chain partition prerequisite

Refreshed pinning frontier: promoted `04664954`, commit `067302c`, official
score 686,230,583 verified candidates/s. The source still uses one 15-point
deferred-XYZZ chain; its public note identifies a balanced two-chain split as
the largest unmeasured ILP direction. Relevant pending notes were inspected;
none establishes this split as solved or faster.

Added `research/bend/two_chain_partition/` against exact promoted source hashes.
Bend 2.0.5 proves a reusable append/sum law and the exact 8+7 partition of the
frontier's 15 signed shifted terms. `bend PROOF.bend` prints `All terms check.`.
A deliberate dropped-chunk-8 mutation in `BROKEN.bend` is rejected. The theorem
covers term and weight preservation only; it does not cover secp256k1 exceptional
addition, XYZZ/deferred-anchor invariants, CUDA behavior, or speed.

Cost gate: the split removes no field operation and adds a second seed/resolve
boundary, one projective merge, and another live XYZZ accumulator. Do not submit
or integrate it without a source-bound sm_89 no-spill/occupancy result and an
NVIDIA A/B timing that demonstrates enough latency overlap to pay those costs.
No production CUDA source changed and no submission was made.


## 2026-09-17 — paired public-key SHA compiler calibration

Staged exact promoted `067302c` control and a one-line `QSB_PK_UNROLL=1`
candidate under `research/pk_unroll/`. Both pass real CUDA 12.8.93 `sm_89` and
default-flag builds. Full finish remains at 80 registers while its 24-byte stack
and 20/20-byte spills disappear; static SASS grows 4,696 to 7,336 slots.
FastTail finish falls 80 to 76 registers, also removes the small spill, and
grows 3,312 to 4,600 slots. Prepare kernels are unchanged.

This establishes finish-path register headroom but not useful SHA interleaving
or speed. The sharp code-size growth and lack of NVIDIA timing make the switch
unqualified as a standalone submission. It also cannot establish feasibility
for a second live XYZZ accumulator in the already-128-register prepare kernel.
Preserve it for a future timed composition; continue seeking work-removing
prepare changes or a materially lower-state two-chain construction.


## 2026-09-17 — naïve two-XYZZ chain rejected by resource lower bound

Combined the checked Bend 8+7 schedule law with the exact-frontier native CUDA
report. The promoted prepare specialization already consumes 128 registers per
thread with zero spills. A second live XYZZ accumulator requires at least 32
additional 32-bit register-equivalents; its affine-Y anchor adds eight before
merge temporaries. Shared placement instead adds 160 bytes/thread, or 20 KiB
per 128-thread block, on top of the existing 8 KiB and introduces repeated
chain traffic. Serial register reuse removes the hypothesized ILP benefit.

Reject the direct simultaneous two-state implementation. This is a scoped
architecture rejection, not a rejection of every partitioned multiplication.
Revisit only with a compact second-chain representation, cross-thread state
ownership, or a work-removing merge that changes the bound. This prevents an
expensive CUDA implementation/submission of the obvious split while pending
fused-reduction and two-field-checkpoint candidates provide official evidence.


## 2026-09-17 — Bend check of pending two-field denominator

Inspected pending `31e98e4`. Its revised two-field checkpoint removes the
candidate-tree checkpoint and reduces saved state from96B to64B/candidate,
claiming roughly2GiB less logical state/checkpoint traffic per2^24 batch. It
trades extra cofactor products for that traffic reduction and reports a local
counterbalanced+1.7063% RTX4090 result; this remains unofficial while validating.

Added `research/bend/two_field_denominator/`. Bend2.0.5 proves the polynomial
prerequisite for its denominator substitution: the XYZZ invariant V^2=A^3
implies V^2*d=A^3*d, which is the cross product behind
A/(V*d)=V/(A^2*d). A deliberate A^2 exponent mutation is rejected. The proof
does not establish nonzero field denominators, canonical representatives, CUDA
tree behavior, or speed. If the pending candidate promotes, this artifact
supports inheriting its algebra; if it fails, inspect official resource/runtime
feedback before deciding whether the representation or only its schedule lost.


## 2026-09-17 — independent native build of pending fused reductions

Fetched public immutable PR219 base `f7f588c7` and pending PR225 candidate
`211f74dc` (`f297b0f`). Both pass exact-source CUDA12.8.93 sm89/default builds.
The full prepare path changes126->128registers,0spills in both, and
7400->7424SASS slots; FastTail prepare changes by the same+2registers/+24slots.
Finish kernels are identical. Full-path IMAD.WIDE.U32 changes655->653,
IMAD.WIDE.U32.X stays1224, while IADD3.X rises1108->1150.

The combined fusions alter dependency shape but show no obvious static work
reduction and consume the last two prepare registers. Await the official score;
do not duplicate the validating submission. If it promotes, inherit it. If it
loses, single-fusion follow-up is justified only by feedback indicating a
scheduling interaction rather than a general arithmetic loss.


## 2026-09-17 — explicit stronger-than-pending submission gate

Quantified all12current pending notes in `research/pending-evidence-gate.json`.
The largest paired same-card effect is `11ba7e4`'s two-stream pipeline at roughly
+7% on a rented RTX4090, though its separate local scored run is687402879 and
does not clear the current official+1% floor693092889. The most rigorous
current-frontier arithmetic/state evidence is `31e98e4`: repeated+1.7063% on
RTX4090 with every adjacent pair above1%. `337d803` reports+1.26% on RTX4090;
`1044715` reports+1.81% hit-derived on RTX3070 with a contradictory progress
rate. Other entries have weaker, estimated, or no local timing.

No current local candidate is stronger than the full pending field, so immediate
submission is not yet authorized by the evidence gate. The clearest path is a
validated composition of the orthogonal two-stream and two-field mechanisms,
or a repeatable same-card result exceeding the roughly7% paired two-stream
effect. This gate must be refreshed immediately before any upload; negative
official results can lower it as pending candidates resolve.


## 2026-09-18 — fused root-group inversion + deferred drain (candidate, pre-submission)

Baseline: promoted `bad91ac` / f16f893 / otaliptus @ 728,615,288 cand/s
(frontier moved from bb5c9a0/726,763,328 while implementing; candidate rebased,
only pinning.cu differs — all other production sources LF-hash-match the tip).

Change (pinning.cu only):
- New `qsb_fused_root_inverse<128>`: folds the root-group inversion chain into
  the prepare kernel tail. Each prepare CTA publishes `roots[blockIdx.x]`,
  then thread0 does `__threadfence()` + `atomicAdd` on a per-group arrival
  counter (group = 128 consecutive CTAs). The last-arriving CTA rebuilds the
  packed product tree in the dead `qsb_digit_arena()` shared area (12 KB),
  runs one `_ModInv` on the group product, expands/normalizes leaf inverses,
  and writes `roots[i]=1/r_i`, `roots[count+i]=(1/r_i)*u2ry` — the exact
  `qsb_root_group_finish` contract.
- Removed launches: `qsb_root_group_prepare`, `qsb_invert_super_roots`,
  `qsb_root_group_finish` (functions kept in file, now dead). Removed allocs:
  `d_super_roots`, `d_root_checkpoint`; `tree` arg dropped from launches.
- Arrival counters co-allocated at `d_hit_cnt[s]+1` (word0 stays the hit
  counter); existing per-batch memset covers them — zero extra stream ops.
- Host: `slot_done[s][parity]` events; enqueue batch k then drain parity 1-p
  (batch k-1); per-parity `slot_seq`/`slot_lt` attribution; final drain of
  each slot's latest parity. `QSB_SLOTS` still 2.
- `QSB_ROOT_GROUP_N` (default `QSB_TREE_N`) parameterizes the group width;
  compile-time check requires it == prepare block width.

Tests (local, no GPU — NOT a score):
- clang CUDA device+host passes compile clean for sm_89.
- ptxas -v: S0<false,0> 122 regs / 0 spills / 12292B smem (tip baseline was
  128 regs / 8B spill before our tail — the fused-squaring tip freed
  registers); S0<true,0> 128/0; finish 72/0. Stack +80B (ModInv frame).
- research/check_fused_inverse.py: real extracted helper run as a simulated
  128-thread collective (real barriers) vs OpenSSL — PASS, full 128-root and
  partial 73-root groups, exact inverse + weighted-inverse outputs.
- git diff --check clean.
- check_projective.py: still incompatible with frontier source structure
  (expects 399cf3b-era symbols `_PointMultiSecp256k1Projective`,
  `kernel_pinning_real`) — documented as stale, not a failure of the change;
  our change does not touch projective recovery algebra.

Negative results recorded this session:
- .cs evict-first on production saved[0..3] planes: DEAD — official 6bf7195
  measured 724,075,734 (-0.37% vs then-frontier). Subagent claim "untried"
  was wrong (refactor recreated the call-site gap; mechanism already tested).
- volatile-drop on digit staging: PTX shows MORE shared ops without volatile
  (109 vs 92); reverted — volatile forces the intended staging round-trip.

Pending-field check at submit time (7 validating): none implement root-chain
fusion. 37a0122 (scarletbright) warp-scopes barriers INSIDE the existing root
kernels + SHA round-0 fold (orthogonal). 8e76620 early-load/prefetch flags;
dcaa914 multiply interleave on bb5c9a0; b5a087b SLOTS=3 (known-regressor
territory); aeadf37 QSB_STREAM callsite restore (local -0.02%); 20de2f4
byte-identical resubmit; 01deb03 smaller batches.

Predicted effect (unmeasured): structural latency removal — 3 launches +
serialized 2-inversion phase per batch + host drain gap. Estimated +0.3-0.9%
(subagent cost model); NOT a measured result. Submitted under the
user-authorized gate: qualified candidate, mechanism unique vs pending field,
verified algebra + clean resources.

Submission: 7a86dbd4-70fb-4e3f-b38d-2e9056332ad9 — queued/validating (job 845de33f, submitted ~16:28Z 09-18).
Candidate pinning.cu LF sha256: 3f9b8024fbdcbcecb6a6d6f0ac3539fd80a2dd855a8c80cd80d64dbc8b42f1ce
