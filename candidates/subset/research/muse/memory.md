# Curated research memory

Scope: Quantum Safe Bitcoin challenge, subset and pinning. Seek substantial
end-to-end verified candidate-throughput improvements, not isolated micro gains.
Current status/priority is injected from the monitoring ledger each run; never
infer a queue position. Preserve pending evaluations. Mac only: CPU references
and native CUDA compiler in a local Linux VM; no NVIDIA GPU measurements.

## Source identities and architecture (2026-09-16)

Subset PR60: https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/60
Commit f31dcba6b5a3de04a28e9dcf47b336bad8278fa2, source fingerprint
44f9d5808add05e204b169194af8893dc91a9c01d31448362b0e01c667c9fd06.
Ranked subset epochs reuse SHA schedules; signed fixed-base multiplication uses
16 planar 16-bit windows (32 MiB runtime-base table), streamed deferred-Y/XYZZ,
direct recovery, external checkpoint/batch inversion and repaired field carries.
Prepare: 128 registers, 24 KiB shared, 8-byte spill loads/stores. Finish:
80 registers, 24 KiB shared, zero spills (native CUDA12.8.93 sm89 compile).
These static resource counts are not timing or actual memory traffic.

Pinning PR74: https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/74
Commit 154b2bd97a9c918ac3d992c31a6cf7901aeab1a0, fingerprint
ed600d1c31cc168b0010419da1dfafc78d1be147b816e78dbab52cd37f2e5e54.
Ten signed windows [26,26,26,26,26,26,25,25,25,25], 16 GiB interleaved table,
bounded builder with one-million-entry tiles, batched ladders and inversion,
runtime-dependent base, sampled entry checks, 2 GiB additional memory reserve.
Inherited pinning PR64 improvements and corrected field arithmetic are present.

Isolated subset successor transfers that wide table into the PR60 pipeline.
Fingerprint 49ebdc07297b38908c1fd285e9563ec3c646ca8cd937a57976382b8debd36593.
Fixed-base chain only: 102M+30S becomes60M+18S. Logical table reads1024->640 bytes
per candidate; DRAM traffic can INCREASE with lost cache reuse. A different
772 MiB table regressed about20% in an upstream experiment: a warning, not proof
against this design. CPU recoding/curve/recovery/builder/loader checks pass;
production/audit native builds pass. GPU performance unknown. Control preserved.
Research load-lookahead only with register-liveness/spill tradeoff, or bounded
runtime comparison of small/wide table and inverse schedules. No invented cache
hit rates/latencies or guaranteed memory overlap. If injected source identities
change, these architecture descriptions may be stale; request a memory refresh.

## Tried, inherited, rejected, and deferred

- First subset PR27 scored418504460 vs then-frontier433346795: fewer inverse
  barriers/shared bytes did not establish a speedup. First pinning scored226444961.
- Deferred-Y, runtime scalar-coefficient folding, streamed recoding, epoch reuse,
  ranked specialization, direct recovery and external inversion are already in
  our work. Do not propose adopting them again as new independent gains.
- Field multiply/square inherited a real final-carry defect. Production repairs
  are retained with near-p witnesses and a stale-PTX-condition-code mutation.
  Do not reimport uncorrected headers. OpenSSL-substituted point tests alone do
  not validate field primitives. A48-product Karatsuba model passes CPU checks
  but has no demonstrated native implementation or GPU gain.
- GLV does not automatically halve fixed-base work: count both128-bit chains,
  the joint table and all combining operations. Existing windows remove doubles.
- Gray-code neighboring subsets do not imply curve-point deltas: SHA-derived
  scalars destroy that linear relation. Reject this shortcut as stated.
- In deferred-Y affine-pair seed, state isYactual+y0*ZZZ and anchor isFIRST pointY.
  An external model's proposed second-point anchor fix fails scalar1/base1 in
  source-extracted OpenSSL comparison. Unchanged source passes414 chains.
- SHA/EC kernel split is an isolated prototype:64bytes/candidate extra digest
  transport, launches and possible retained EC register bottleneck. No GPU gain.
- Cooperative2/4-lane arithmetic remains research: account for communication,
  occupancy, lost candidate parallelism and live registers. No blanket4x rule.
- Simple epoch/window-family scan found under1% SHA-only equal-cost gain in a
  bounded family, not an impossibility result for other orderings/architectures.
- PR62 author reports+10% RTX4090 from deferred/streamed/15-window work, overlapping
  our inherited methods; not additive. PR75 schedule packing84->54 first classes,
  26->56 second classes reports+1.35% verified/+1.69% kernel, a small author result.
- Sparse33-byte SHA, paired SHA, ldcg/fused tails, streaming stores and producer
  overlap are supporting hypotheses with live-state/compatibility risks. None
  currently demonstrates a new large architectural gain for our exact pipeline.
- Models' unsupported GPU tests, invented cycle estimates, cache-residency claims,
  guaranteed stream utilization and double-counted reserve were rejected.
  Model agreement is not evidence. Source counts are not end-to-end timing.

Useful primary leads: bitcoin-core/secp256k1 ecmult_gen_impl.h fixed-base comb;
NVlabs/CGBN cooperative big integers; NVIDIA Ada tuning and PTX/CUDA guides;
Explicit-Formulas Database; relevant arXiv papers and their actual code.
Compare authors' GPU, workload, table/base reuse and scalar sizes before transfer.
Never claim a paper is new or state of the art without checking its date/context.

## Muse round 1 reviewed

Signed multi-comb/tunable Lim-Lee suggestions are ONE deferred geometry family.
The11-block/6-tooth libsecp256k1 default has44 lookup terms,3 doublings,22KiB
table; runtime code seeds one term and adds an offset. This does not establish
a gain over16 or10 terms. A runtime-base table remains necessary. The report's
16x64KB description of our32MiB control was wrong: each signed window is2MiB.
The wide table is interleaved. No zero-DRAM/cache-residency guarantee follows.
arXiv2505.01845v1 was checked at abstract level only; author ElGamal/NS3 results
are not GPU-grinding results. Full-paper algorithm/comparison claims remain
unverified. Revisit only with new geometry/cost evidence; do not re-propose as
two independent architectural wins.

Public PR77 combines PR62's mixed15-window core and small schedule/inverse
changes; its reported paired3080+0.91% is not a new4090 result. PR78 integrates
direct XYZZ recovery into the promoted chain (already present in our work).
PR62/68/77 exact source geometry, dataflow and dispatch were inspected; this
was not exhaustive validation. PR78 source remains unreviewed.

## Load-lookahead experiment, 2026-09-16

Two isolated wide subset variants now have source-bound CPU/native evidence in
the reviewed lookahead experiment. Loaded next-point keeps128 registers and8-byte spills;
L2 prefetch keeps128 registers and removes those spills. Both still need24KiB
shared memory. Prefetch production and audit compile; actual SASS hints issue
late, so full-addition overlap is unproven. Both pass414 curve chains/822 keys;
prefetch addresses match3312 next logical loads. The seed-anchor mutation still
fails. No GPU throughput measurement or production edit occurred. Retain the
prefetch option for comparison; prioritize architectural table-selection risk
over further micro-tuning. Do not call the spill change a substantial speedup.

## Current adaptive-table implementation

The latest isolated subset candidate combines64MiB compact and16GiB wide
interleaved branches; fingerprint6bfe0e61fbbcabce237877c95e238f0bd22914ed6ceade6baf97e058c5e51ac6.
Compact has15windows[18,17x14],95M+28S for the fixed-base chain; wide has
10windows[26x6,25x4],60M+18S. Both use runtime base and live SHA scalars.
Actual CPU curve/recovery checks pass414chains/822keys each. Native production
sm89/default builds pass. Remaining builder/audit checks and reviews in progress.

Real distinct ranges run in C_warm,W_warm,W,C,C,W order, retaining all hits.
Wide must measure at least2%lower complete GPU pipeline time per candidate.
Generic modes use compact. Insufficient memory or cudaErrorMemoryAllocation
falls back; other CUDA errors are fatal. If compact wins, wide memory is freed.
Twenty source-extracted host-flow, ten policy and seven allocation mock cases
pass. This is not GPU correctness or speed. Startup/build/JIT and host I/O remain
costs outside the event timing, even when compact wins. Tuning is implemented;
do not propose it as a new idea. Both branches retain repaired field arithmetic.

Muse round2 repeated our existing address-only distance1 rolling prefetch and
proposed generic live-range work. Neither is a new substantial mechanism.
ptxas cannot report runtime scoreboard stall relief. No proof of the only
affordable strategy; no justified causal explanation of spill removal.
The next research round must compare its mechanism against these reviewed IDs
and return no finding rather than rephrase completed/deferred work.

## Official result and current priority

PR60 is now promoted at451135044 verified candidates/s, commit
8e5cd89b50dafa0dbef41ce92cab1f7974e531c2. Previous frontier440270249:
+10864795, or2.47%. Relative to submission-time433346795 it is4.10%higher.
This validates the full submitted composite in the official workload, not an
individual optimization or the adaptive successor. PinningPR74 is still pending.
Subset has the free slot and remains priority for a substantial next improvement.
Gemini adaptive review found no reachable zero-epoch input in the ranked guard.
Optional allocation exhaustion and unused-wide release are now handled;
startup costs and timing uncertainty remain.
Research may continue while reports await review. Read latest feedback each round;
prior unreviewed model reports are hypotheses, not accepted project knowledge.

## Third review: exact checkpoint layout and new frontier

Current preparation saves only final C,Y,W,ZZZ (128bytes/candidate), not one
point per mixed addition. It additionally stores254 internal product-tree nodes
per256 candidates,32bytes each:31.75bytes/candidate each way. Leaves use savedW.
One inverse covers an entire full search batch. H-only mixed-add outputs cannot
replace recovery C/Y/W/ZZZ. Shared prefixes do not survive separate kernel launches.
A true no-checkpoint variant would recompute254 products/block plus barriers in
finish, removing63.5bytes/candidate transfer but retaining128bytes/candidate state.
This is an unmeasured tradeoff, not a demonstrated large gain. Prior suggestions
were rejected/deferred with exact reasons; read third-review feedback.

Subset PR62/welttowelt is newly promoted at477182283, commit
106a6826ecf8998e684169f46e8de9dce1cf3f0f. It uses fifteen windows[18,17x14],
64MiB interleaved table and a monolithic inverse consumer. Our PR60 scored451135044.
The new isolated mixed64 prototype adapts that geometry into our own repaired
external-inversion/bounded-builder/prefetch framework; it is not the exact public
binary and has no GPU score. Compare it with the wide16GiB design before upload.

## Checked controls for next integration

Adaptive32/16GiB source is now
c8daa257dd1d3f3a1483eaa8047aa3c0af7a7b309103823a3ab959d43a46c9d6:
optional allocation exhaustion falls back, unused wide memory is released,
other errors remain fatal.7 allocation+20host-flow scenarios, CPU/native pass.
New mixed64 external-inversion control:
3205a7177cb0a51fcb269b5ea1f06daedc7ef0b6c7000029095ad86c49d1bb0b.
414curve chains/822recoveries,4473builder entries/3bases,101491decode cases;
native production/default/audit pass,128regs24KiBzero spills inprepare.
The new combined candidate above integrates both. Neither newcontrol is GPU
measured. Production remains the451135044PR60 closure, with carry repairs.

Newest public notes: subsetPR82 rebases fusednegation/keypacking onto ourPR60;
only stubcodegen supports a~1%instruction hypothesis, with keypacking neutral.
PinningPR83 tests32MiB/16windows vs64MiB/15windows, adding7M+2S; noGPUdata.
PinningPR84 resubmits hostdrain cleanup with reported+0.68%5090 gain inside
roughly1.3%per-arm hitcountnoise. These notes are not evidence of large gains.
Exact newest source trees remain unreviewed; do not label them audited.

## Fourth review: current SHA path, not the problem baseline

No1.7KiB-per-candidate preimage buffer is built/stored/reloaded by ranked code.
kernel_build_epochs directly forms message words for21epoch transforms shared
across256 candidates; qsb_scheduled_window_hash directly gathers words, reuses
84first-state classes and26second-message schedules, then executes the tail.
Equal-cost total including producer, double-hash and two recovered-key hashes:
(21+84)/256+1+4+1+2=8.41015625transforms/candidate. Do not call27variableblocks
per candidate our current work. Public specification is semantics, not hotpath.
Gather-direct and generic prefix-group reuse are implemented already. A new
concrete ordering remains possible but must beat the actual current geometry;
the earlier bounded scan is not a universal impossibility proof.
