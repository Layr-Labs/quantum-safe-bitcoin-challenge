# Iteration 0 — 2026-10-02

## Standing orders and frontier

Read `.angelY/ORDERS.md`, README and the manifest; board subset only. Rechecked
all submissions and promoted notes/trees. Current promoted best remains cefika
fb6f5a8, 728,337,167 on RTX4090. Local RTX3090 uses equal dev GLV12 tables for
both arms; no absolute 3090-to-4090 comparison is claimed.

## Rejected first lever

Requested OUTER_LITK + staggering + PRE3_ROOT stack passed exact verification:
338.397645 M/s versus promoted-control smoke 360.127130 M/s (-6.03%). Existing
stag arm was not the promoted control (stagger1/park0 instead of stagger0/park1).
Added explicit promoted wrapper to correct the reference. Stack-all smoke also
passed at 362.658747 M/s; that isolated draw is not promotion evidence.

## New ground: four smaller independent digest CTAs

Set block128, launch bounds(128,4), shrink park buffers AND file-scope active
product arena plus inverses. First incomplete allocation version retained the
active product arena, used32KiB shared and scored282.926067M/s, PASS. Full shrink
uses24KiB shared,128 registers,no spills. Both build and ranked sm89 cubin report
that resource footprint. Retained capacity832words of disabled shared LUT.
Tree levels, epoch pair multiplier, late hit indexing all traced; unchanged
candidate/window set and exact verifier.

`bash candidates/subset/stack_ab.sh block128 3 promoted 120`:

| Pair | promoted M/s | block128 M/s | order |
|---|---:|---:|---|
|1|317.379041|344.324586|A B|
|2|310.195276|335.292389|B A|
|3|315.001493|327.267847|A B|

Means314.191937 vs335.628274, **+6.822689%**, all six PASS. New stack_ab preserves
each log+JSON under a unique repo-local directory, checks exit status and score
verified flag, copies score while holding GPU lock. No stale file parsing.
Exact artifacts in lab/ab-promoted-block128-20261001T222116-1533876.

Regenerated carrier with pre-existing build_carrier.sh, CUDA12.8.93. Installed
missing CUDA12.8 cuobjdump/nvdisasm tools (development only). Cubin473376bytes,
SHA d938cf8a0d9f28dfe9af5ebf9565d0723009fb3a17a169a6520ac72455832f80,
four LTC64B loads. Ranked fixed compile succeeded. Moved44 tracked local ELF
binaries and large lab board snapshots into development cache, leaving3.22MB
submission source+notes+small measurement logs. No runtime cache reliance.

Commit259b64d: verified improvement +6.82%. Submission
**f45d9172-5753-425b-b07b-40a2fc159b4e**, validation job
f498da9e-eb06-42de-b432-1b7f7b154399, queued receipt in
lab/submission-block128-receipt.json. Public note SUBMISSION-BLOCK128.md.
Attribution flags modelGPT-6.1-Sol,harnessangelX; effortxhigh.
Watcher owns validation; no polling wait.

## Next experiment, while submission is in flight

Double ZLAB_LAUNCH_BLOCKS to524288 with otherwise identical block128 source.
The smaller block halved paired candidates per launch; this knob restores the
previous launch-fill count without increasing resident CTA size. Expect less
host overhead, possible larger descriptor/producer allocations. Reuse existing
benchmark gate, then interleaved comparison against block128 if it improves.
Do not submit an unverified or sub-4% improvement. Preserve best in-flight source.

# Iteration 1 — 2026-10-02: launch geometry and occupancy boundary

Read ORDERS; rechecked board and top promoted notes/public trees. Frontier remained
cefika fb6f5a8, 728337167. No trusted harness/probem/score code was changed.

## Measured winner, submission blocked by the occupied slot

Doubled blocks/launch on 128-thread CTA: 524288 instead of262144, restoring the
old per-launch paired work. Incremental interleaved3pairs versus block128:
A371.172705,364.251203,365.742676 (mean367.055528); B366.919389,376.156954,
378.607465 (mean373.894603), +1.863%, all6verifierPASS. Complete3pairs vs exact
promoted-source control: A358.033403,361.614635,358.540270 (mean359.396103);
B385.524155,386.422367,388.825921 (mean386.924148), +7.660%, all6PASS.
Artifacts: lab/ab-block128-block128_fill2-20261001T224044-1655336 and
lab/ab-promoted-block128_fill2-20261001T225741-1758741. AllN24,120s,seed1789110211;
stack_ab.sh serial lock and copied scores. Comparisons use equal devGLV12 geometry,
NOT ranked4090absolute performance. Firstpair incrementalnegative: modestlaunchgain.

Native sm89 CUDA12.8.93 carrier rebuilt:128regs,24576shared,zero stack/spills,
4LTC64Bloads, cubinsha e370d476d6ebfd4b51157257ce0f9d4a94d1d8068f953244ce20e9ff446eff18.
First build failed to locate installednvdisasm; fixedtoolPATH, finalbuildPASS.
Winner commit2cea234; noteSUBMISSION-FILL2.md exceeds5KiB. Immediate submission
attempt after board recheck returned conflict: account alreadyhas1submission in
flight(limit1). NOnewreceipt/id. Priorf45d9172-5753-425b-b07b-40a2fc159b4e occupies
slot; watcher owns it. On notify retry currentbest, do not poll-wait or cancel it.

## Reopened tree unrolling: no adoption signal

128leafconstexprn specialization requires removing256-onlyassertion; arithmetic,
indices andbarriers unchanged. Compiled128regs,24KiB,zerospills; scored existing
benchmark1pair:383.933575 vs382.593006,+0.350%, bothPASS. Disabledinproduction,
patchretainedlab/iter1-disabled-probes.patch. Notqualifiedbenefit.

## Register headroom loses decisively

Launch_bounds(128,3),139regs vs128, same24KiB,zerospills. Exact benchmark1pair:
327.718132 vs388.347232,-15.612%, bothPASS. DEAD: independent resident CTAs/warps
are worth more than 11extra regs here. Do not repeat without a materialmemory or
latency mechanism. This reinforces thefourCTA configuration, not a proof that
any lower registercap willhelp.

## Next direction

Try in-place downward tree inversion: overwrite consumed internal product levels
with their inverse; retain all leafproducts. Can remove4KiB downwardinversearena,
20KiBtotal. At64-node downlevel partnerreadcrosses warps: addblockbarrier AFTER
allinputloads BEFORE overwrites, inadditionto the existing publishingbarrier.
32-node level isonewarp. Rootwave16 finishesalloriginalproductreads before
writing inverses. This new aliasing/occupancy direction is distinct fromunrolling.
Firstcompile/runtimeverify20KiBversion atsamefourCTA bound; ifusefulthen pair
withfiveCTA bound totrade registercap for20warps. Do not blindlyalias LUT832words.

## In-place inverse arena: first verified signal, confirmation running

Exactexistingbenchmark1pair: controlfill2 368.425570, inplace383.169787M/s,
**+4.002%**, bothPASS. Compile128regs,20480shared,zerospills. This is onlyonepair;
3pairconfirmation proc1326 queued, noadoptionyet. 5CTAvariant compiles96regs,
56Bstack,4Bspillstores64Bspillloads; scoredpair proc1325 queued. Source/ELF frozen
outsidecandidate plus lab/iter1-inplace.patch and source-elfhashmanifest. Production
device source restoredto measuredfill2 beforehandoff; no stale-carrier mismatch.
Experimental cachedELFs retainedforpendingrun; rebuildinginplacewrapper requires
applyingthelabpatch or usingtheisolatedsource. Do NOTuse unpatchedwrapper as
proof ofoptimization. Detailedinvariant audit lab/iter1-inplace-design.md.
Thebestcandidate hasnotbeenaccepted:CLIconflict,no newreceipt. Onwatchernotify
submitcurrentbest immediately(afterremovingdevelopmentELFs), noteSUBMISSION-FILL2.md.

In-place native-sm89 resource diagnostic also confirms128regs20480shared,
0stack/spills (CUDA12.8.93 cubin), so20KiBshrink isnotdevgeometryonly. ProbeELF
strings explicitlycontainQSB_SE_BLOCK=128,ZLAB_LAUNCH_BLOCKS=524288 and
QSB_TREE_INPLACE=1; controlELFcontainscorrectfirsttwo. Aready-next experimental
nativecarrier build runsproc1327 outsideproductiontree while3pairgate proceeds.
Do notinstallthatimage until experimentwinsgate. Slotremainsownedbywatcher.

In-place5CTA native-sm89compile is substantiallyworseforspills than dev:
96regs20KiB,120Bstack,36Bspillstores164Bspillloads (dev56/4/64). Awaitruntime
screen; do nottransferdevspeeduncritically. Matching4CTAinplacecarrierfinished
outsideproduction:473504Bcubinsha3d0dda78009d4a9ea06ae95ad97a1b3cef20a69f5d0cf43345e857a91385e2ac,
4LTC64Bloads,128regs20KiB0spills. Stillnotinstalled; gatemustwinfirst.

FiveCTAinplace finalscreen **261.247036vs382.686202M/s,-31.733%**, bothPASS.
Kill thisstack; higheroccupancywithspilling disastrous. Scoredartifact
lab/ab-fill2-inplace5-20261001T233146-1938512. FourCTAinplace3pairconfirmation
continuesaloneGPUserial(proc1326). No productionchanges.

## Next materially-distinct root scheduling probe

Isolatedsource enablesexistingtemplatedtree with ROOT_WARP3 and NO LUT/pre3;
warp3 runsthe samewave-top, warp-localrootinverse and down32. Up/down<=32
nodeownershipmoveswithroot; level64 remainslane0..63. Logicalaudit: oldbarrier
half64 publishes sharednodes crosswarp; previousphasehalf32 useswarp3sync;
root/down32samewarp; finalblockbarrierpublishestosameindicesfor64writers.
NoLUTcapacityorpre3spills added. Compile128regs24KiB0spills. Exactscoredpair
proc1335queuedseriallybehindconfirm1326; patchlab/iter1-rootwarp3.patch,
frozensourcepointer /work/tmp/qsb-iter1-rootwarp3-dir. No productionedit.
This differsfromprevious ROOT_WARP2+LUT/pre3stack; don'tinferdeadfromthatstack.

Further launch-fill probe1048576blocks isolatedsourcecompile128regs24KiB0spills,
scoredpairproc1337 queued. Logicalcapacity audit: paired2*1048576epochs fitsint;
blockpositions134217728fitint; tag2^28fits30bitsbelowrecidtopbits; descriptors
128MiB/slot and8firstslots512MiB/slot fitdevsmalltablememorybudget. Allallocations
stillderivedfromlaunchcount; tailsdrainandstopunchanged. No devicearithchange.
Devicecodeis sameasfill2 apartfromonehostlaunchcountknob. Usefulifhostoverhead
continuestopay; growthmayworsenstopdrainandmemory. Notqualifieduntilverifier.
Current3pairinplaceconfirmthrough2pairs:control383.984112/379.568171 vsinplace
380.686111/377.243409. Bothpairwisenegative(-0.86/-0.61%), first+4%screenlikely
noise/drift; keepbestfill2regardlesswhilefinalpairlands. Noadoptionononescreen.

Reproducibilityrepair: explicitlypin oldlaunchcount262144 in n24L_block128.cu
andn24L_promoted.cu nowthatproductiondefault524288. Measurementsusedcached
pre-changeELFs, sooriginalnumbersvalid; a freshcontrolrebuildmustnotinherit
newdefault. Runningconfirmationusesunchangedfill2wrapper. Publicnoteupdated
withactualCLIconflict andcomplete139reg/fiveCTAdeadendnumbers.

Launch4 nativecarrierpreparedoutsideproduction:473376Bcubin,deba60856498dce3...
128regs24KiB0spills4LTC64B. Gate screenstillqueued/runningbehindotherarms.
No productionedit. The CLI conflict receipt persistedlab/submission-fill2-conflict.json
withnullnewIDandoldf45d9172watcherslot. Userrequiredrapidoutputisnotpermission
forcancellingpriorvalidationorpoll-waiting. Retrybestonslotnotify.

Rootwarp3standalone finalexistingbenchmarkpair374.052082vs379.337741M/s,
**-1.393%**, bothPASS; native128regs24KiB0spills. Nosignal, don'tsweepallwarps.
Patchretained; productionunchanged. FourCTAinplacefinalpairstillrunning;
launch4screenqueued. Existingbestwaitingwatcherslot; no newreceipt.

Anothernewtarget: first-stateallocationstride16slots although128windows have
exactly8firstscheduleclasses. Isolatedfirstslots8probecompiling1341; sourceguard
makesQSB_FIRST_SLOTS overridableand addsit tocarrierknobs. Allocation/producer/
digest/verify-stridesallusethe macro, hostuploadusesruntimefirst_strideargument.
qsb_prepare checkfirst_distinct<=slots BEFOREtransposed writes protects8capacity;
first_unique temporary remains128capacity. Fullclasses/fields unchanged. Atfill2
shrinksperstream firststates512MiB->256MiB; digestepochreads become256Binstead
of512Bspan. PotentialL2traffic/TLB andhostupload savings, notarithmeticgain.
No productionedit, exact existingbenchmarkscreenmustpassfirst. Relevant source
window_schedule_shared.cuh not generateddata. Neverassumeotherwindowfamiliesfit8.

InplacefourCTA3pairfinalmean379.329058vs381.357653M/s **-0.532%**,all6PASS,
all3pairnegative. Initial+4.002%falsepositive; rejectfullshrinkadoption. Native
matchingcarrierpreparedonly, notinstalled. Artifactsconfirmation233244retained.
Continuefirstslots8layoutdirection(proc1343),largerlaunch4screen1337.

Firstslots8nativeimagebuiltoutsideproductionCUDA12.8.93,f9fd0d1d..473376B,
128regs24KiB0spills4LTC64B. Preflightnotyetcomplete1343. Structuralfamilyaudit
counts128triples8firstclasses distributions3,15x6,35; notnewtestproofofcrypto,
exactbenchmarkverifierwilldecide. Standardbuffer firstslotstride16*8*4=512B
vs8*8*4=256B; fill2epochs1048576 =>512MiB->256MiB eachof2GPUstreambuffers.
Earlierquicknotereadlaunch4memoryas512MiBincorrect: with16slotsand2097152epochs
it's1024MiB perstream. StillfitsGLV12devrigavailablememory, butmemory/drainrisk
iswhyfixedtimescreenmatters. No candidate productionedit.

Launch4screenfinal384.291331vs380.558346M/s, **+0.981%**,bothPASS. Kernelarith
unchangedbutlarger1GiBfirstbufferperstream anddrain121.6s. Onlysinglepair, not
newproductionbest. Firstslots8screen1343running; ifpositive, confirmpackingand
thenstacklargerlaunchagainstmatchedcontrol. Scoredartifactab-fill2-launch4-20261001T234728-2035330.

Nativefirstslots8SASSaddressaudit: digesthasexactly2instructionchanges, IMAD
stride0x80->0x40 andpairedfirststateORoffset0x80->0x40 (wordunits); allremaining
nativeinstructionlinesidentical. Thisbindingprovesstrideleverlivewithoutpoint
arithmeticcodegendrift. Diffretainedlab/iter1-firstslots8-digest-sass.diff.
Hostproducer usescudaMemcpy2DAsync withexplicitfi_pitch andusedwidth8*32,
packinghalvesGPUallocbutNOhostproducerrowwidthchange; preservesbatch0selfcheck.
3pairconfirmationproc1347queued whileinitialscreen finishes1343, notadopted.

Pack8+launch4stackcompile clean128regs24KiB0spills; scoredexistingpairproc1351
queued. With8firstslots thislargerbatchuses512MiBperstream (sameascurrentfill2
with16slots) whiledoublingworkperlaunch. Combinedmechanismnot assumedadditive.
Firstslots8confirmationproc1347 continues; screensserialinterleaving mayrunother
armsbetweenpairmembersbutallcompareexistingbest andunique copiedscores.

Pack8launch4matchingnativecarrierreadyoutsideproduction eb00e214147a1482...
128regs24KiB0spills4LTC64B. Screen1351queued;1343firstpackedscreeninprogress,
1347confirmationalsopending. No needtoholdverifiedwinnerforimagebuildifgatewins;
imagesalreadyready. OnnextwatchernotifyretrycurrentBEST afterboardrecheck;
ifpackingnotqualifiedyetkeep2cea234fill2production. All scorelogswriteuniquedirs.

Firstslots8initialexactscreen376.008392vs385.851135M/s, **-2.551%**,bothPASS.
Losslessbufferpackingworksbutfirstspeedreadnegative. Existing3pairconfirmation
alreadyqueued1347willdistinguishnoise; do notpromote. Combinedpack8launch4screen
1351alsopending. Footprint andtwoSASSaddresschangesevidenceconcretebutnotwin.
Readyproductionfill2 unchanged; submissionretryonlyafterwatchernotifyslotrelease.


# Iteration 2 — 2026-10-02: root exchange, split CTA, compact decision LUT

Re-read ORDERS/README/manifest and board. Top promoted still cefika fb6f5a8
728337167, ff27a2b; its public note and device tree re-read; e671 note extracted.
Our f45d9172 remains validating at iteration start; watcher owns it, no polling
loop and no resubmission through conflict. Ready verified fill2 remains production.

## Concrete completed scores (N24, seed1789110211,120s, serial GPU lock)

- slotstag: 373.934565 vs 371.572845 M/s, +0.636%, both verified. Evidence `ab-fill2-slotstag-20261002T001800-2241255`.
- pack8launch4: 362.726365 vs 372.931444 M/s, -2.736%, both verified. Evidence `ab-fill2-pack8launch4-20261002T000549-2161900`.
- topshfl: 376.095140 vs 387.523734 M/s, -2.949%, both verified. Evidence `ab-block128_fill2-topshfl-20261002T003251-2335473`.
- split64: 315.913805 vs 388.134841 M/s, -18.607%, both verified. Evidence `ab-block128_fill2-split64-20261002T004021-2380630`.

Firstslots8 3 alternating pairs now complete, all6PASS: A379.959697,
367.281765,369.896983; B373.106826,370.071546,387.251840. Mean +1.1898%,
but firstpair -1.804%, second +0.760%, third +4.691%. Contradicts the first
negative screen without becoming a reliable winner. No adoption.

## New implementations

- Register-shuffle tree top: P8/P4/P2 go register-to-register instead of shared
  store/barrier/load; operands and cofactor algebra preserved. Exact pre-existing
  tree_audit passes all tested block sizes and partial counts. Both local/native
  128regs24KiB0spills. First scored pair -2.949%, rejected adoption.
- Split64: SAME128 window set, split two64-laneCTAs per epoch pair; SAME CPU
  complement. Host doubles nblk, GPU lane=tid+64*(blockIdx&1), epoch owner
  blockIdx/2, late hit tag same. 128regs12KiB0spills local+native, eight resident
  blocks feasible; exact gate PASS but -18.607%. Doubling root overhead beats
  any benefit of independent resident blocks. Not a win.
- Compact32 rootLUT: ALL832 entries losslessly encode in31 bits (coefficient
  parity exploited), 3328B vs6656B. Constant lookup and shared lookup screens
  running. Shared compact table fits128-leaf inverse arena (4KiB), unlike old
  full64bit shared LUT needing8KiB; this preserves fourCTA occupancy.
  Existing exact tree_audit with ROOT_LUT32=1,ROOT_LUT_SMEM=1 passes all sizes,
  sparse tails, direct8192root outputs and bounded fallback. Separate frozen
  source and native image prepared; production defaults allOFF, fingerprint
  unchanged. See lab/iter2-compact32-review.md.

## Audit lesson, not candidate failure

Production SHORT_CARRY3=1 is speculative: tree_audit's arbitrary edge operands
are not valid exact-tree assertions for it. Both changed/unchanged controls
returned byte-identical logs (sha256 a5fbb1119aecf331...). With exact multiply
(SHORT_CARRY3=0) and unused narrow parity specialization OFF plus plain-root
ISO scaling OFF, same pre-existing test passes. Do not re-label these identical
control failures as a regression, or count exact test as production scorer;
production always separately runs benchmark.sh's exact verified hit gate.

Commands are retained in lab/frozen_ab.sh, source frozen under temporary
experiment trees. Build with nvcc-O3-DQSB_ZEROS_N=24; native build_carrier.sh
needs installed cuobjdump/nvdisasm onPATH. Captured reports in lab.

Default-off native rebuild of the full new source experimentally confirmed
identical CUBIN and knob bytes to verified fill2 (e370d476...); only generated
header's source-hash comment differs. Production header NOT changed.
Compact32 constant screen380.052120vs392.313841M/s **-3.1255%**, bothPASS.
Shared32 screen in flight; eightCTA split and shuffle losers left OFF.

Shared compact32 scored376.253106vs386.117278M/s **-2.5547%**, bothPASS.
Constant31andshared31packing bothloseinitialscreens. Next diagnostic: split40
(matrix32+flags8) preservingoriginalPRMTcoefficient decode,4160Btable and
4160Binversearena. Testswhether31bitdecodecostratherthanlookupmemoryhidesgain.

Split40 exactPASS but329.513712vs381.581868M/s **-13.645%**. Driveroccupancy
confirms64Bsharedincrementcrosseshardboundary:24576B4CTAs,24640B3CTAs.
Root64->split40decodepreservesPRMTbutoccupancycostdominates. Duplicateouter
rowsidentifiedbyall64entriesbitcomparison(-6/-5,+5/+6); SHORTswitch704entries
3520Bfitsoriginal4KiB; native128regs24KiB0spillsready1b2f24451ae5805b...,
exactauditthenproductionpairqueued1414. Threepairlaunch4confirmation1407
stillrunning;firstpair384.531824vs380.421569M/s+1.080%,bothPASS.

LUT40SHORTauditpreexistingtree_auditall32/64/128/256sizesPASS;128regs24576B
0spillslocal+native. Run1414scorepairinflight,control391.389281M/sPASS,
candidatequeuedunderGPUlock. FinalcodeaddscompiletimeguardsSHORTrequires40,
sizeofZiLut40exactandarenasizecoversLUT; entry1418repeatexactauditqueued.
Labwrapperinclude pathsfixed../subset.cu(reproduciblefromlab). Currentbest
verifiedproductionstillfill2; nocandidatequalifies4%gate,newsubmissionnone.

LUT40SHORT corrected24KiBgeometryscore380.746996vs391.389281M/s **-2.7191%**,
bothPASS. Thisremovesoccupancycliffbutkillstablepackingasadoptiondirection.
Finalassertioncompileandpreexistingtree_audit1418PASSagain.Root8192exact
andall32/64/128/256partialszeroswrong.No productiondefaultchanges.
HEADd538abeplusresultsnotescommitfollowing. Launch4thirdpairBstillpending;
A364.324560M/s(first2pairsA380.421569,382.642154;B384.531824,385.494486).

Launch4confirmationcomplete3alternatingpairsALL6PASS: A380.421569,
382.642154,364.324560(mean375.796094); B384.531824,385.494486,382.611130
(mean384.212480),**+2.239615%**overfill2. Thirdcontroldrawdownlarge; first2
pairs+1.080%,+0.745%,soevidencepositivebutnoise/driftstillimportant.
Launch4nowcurrentnext-bestforqualification; readyfrozennativealreadyexists.
Started1428directpromoted3pairgateBASE=promotedfrozen_ab.sh launch4promoted,
S1789110211,T120. Thisisstanding-ordergateversuspromoted,notversusfill2.
If>=4%verifiedsubmitafterwatchernotifyslotavailability. No currentnewreceipt.
CurrentHEAD6e9589aproductionfill2unchanged; laboratorysource/scorecommitsonly.

# Iteration 3 — 2026-10-02: launch gate resolved, producer and outer-SHA probes

Standing orders re-read; board top unchanged cefika fb6f5a8,728337167. Existing
f45d9172 still validating in the single board snapshot; watcher owns slot.
Launch4 gate1428 COMPLETE againstfill2, all6 exact verifierPASS:
A403.651443,385.069314,381.791675; B386.267158,379.483676,376.914628M/s.
Mean A390.170811,B380.888487, delta **-2.379041%**. Allthree pairs negative.
This contradicts the prior +2.24% fill2 comparison. It does NOT settle the
standing-order-promoted-source gate (see control routing correction below).
Full scored artifacts ab-block128_fill2-launch4promoted-20261002T012744-2687279
(its directory label correctly reflects the actual fill2 control).
Summary retained iter3-launch4-promoted-summary.json; historical filename
is misleading. Launch4 is not qualified for submission.

New materially distinct producer-only QSB_FIRST_FLAT_FAST probe: specialize
runtime mapping for8classes (shift/mask; fallback otherwise), two uint4 stores
instead of8 scalar stores, same state layout/digest. Logical review retained
iter3-firstflat-review.md. Offbydefault, enabled fingerprint prevents stale
carrier. Frozen dev+native build1443EXIT0, native cbf3fb1ed433f8dd...,
128regs24KiB0spills unchangeddigest, producer48regs(vs47base). Native SASS
confirms STG.E8 -> STG.E.1282, emitted1464instructions both (added guarded
fast path trades saved stores). Scored pre-existing verifier pair1444 running;
no correctness or speed claim yet. Old first_stage_audit.cu references removed
kernel_build_first, not live kernel_build_first_flat; do not count it as proof.

New SHA256d outer-pair probe QSB_OUTER_PAIR uses the inherited gate's generic
round-interleaved paired transform for the two outer32-byte hashes. Earlier
OUTER_LITK was solo literalK and ZLAB_PAIRSHA is startup-only; this is distinct
per-candidate dependency scheduling. Logical review iter3-outerpair-review.md.
Knoboff; identity tracks enabledvalue; compile guard requires gatepair and
no literalK. Frozen build1446 running, scored pair to follow resources.
Two transient disassembly misses were tooling only (deleted build CUBIN and
nvdisasm path); fixed by decoding generatedheader and addingtoolPATH. Exact
proof still comes from unchanged benchmark/verifier, never from compile exit.

## Control routing correction — source-emitter audit, not a performance theory

Auditing the actual script uncovered `base=${4:-block128_fill2}` ignored
BASE=promoted from prior handoff's job1428 launch command. Thus1428 was another
fill2 comparison, NOT directpromoted. Allscores/verifications real, labeling
and adoption interpretation retracted above. The new script accepts the
explicit positionalcontrol first, thenBASE fallback, thenfill2 default. Every
new run records exact source paths/text/hash, cachedELFhash and matching stamp
before GPUuse. Existing cached promotedELF had been cleaned; first gatedlaunch
1449 nowcorrectly FAILSbeforeGPU at its missingELF/stale stamp assertion.
Fresh promotedwrapper build1450 running, then explicit positional promoted
threepair gate (same128windowGLV12 geometry as all prior local experiments).
This concrete audit missed by prior directions prevents false gate decisions.

Firstflat3 screen1444 completed exactPASS both, A379.091389 vs B368.458047M/s,
**-2.804955%**, no adoption. Native instructiontextdigest byteforexact identical
betweenproduction andfirstflat; producer's twoSTG128 are the only functional
storeform changes. Probeleftoff. Outerpair1446 buildEXIT0, native71b0f79d...,
128regs24KiB0spills. Digest14528instructions versus14512baseline(+16),14calls
41branchesboth; scored screen1447 in flight. No new submission yet: same
watcher-ownedf45d9172slot; bestqualified remainsfill2.

Outerpair screen1447 completed exactPASS both: A383.766384 versus
B370.702353M/s **-3.404162%**. Runtime outerhash stream scheduling is not a win;
leaveknoboff. Existing ranked128register/24KiB/zero-spill geometry preserved
but increasing ILP/pairedstate does not automatically improve compiled code.

Corrected actualpromotedgate1450 live with manifest: promoted256threads/262144
launch versus frozenlaunch4 128threads/1048576 launch. Firstpair exactPASS:
A352.821018 versus B377.038829M/s **+6.863%**. This dramatically contrasts
with the wrongly labeled1428fill2repeat; directpromotedsource wins cannot be
inferred from a control nickname. Must finish allthreealternatingpairs; no
adoption/submission conclusion from this firstpair. See run manifest hashes.

Actualpromotedgate1450 COMPLETE allsixPASS: A352.821018,349.601807,346.031719
(mean349.484848), B377.038829,375.803530,366.950792(mean373.264384).
**+6.804168%** verified across3alternatingpairs; allpairspositive. Meets4%bar.
Finalcurrent-source nativebuilt1459; entireCUBINbyteidentical to frozenimage
measuredinthose3pairs, SHA256deba60856498dce3bc6a18d153e30f913bd515f2c6f6033decda8092067c4d9c.
128regs24KiB0spill. Productionnowlaunch1048576andthatimage; experimentsallOFF.
Freshfinaldevbuild/preflight1461running. Presubmitboardsnapshotf45stillvalidating,
cefikatop728337167unchanged. Submitbestafterfinalpreflight (conflict may persist,
neverinventreceipt/newID). PublicnoteSUBMISSION-LAUNCH4pendingwritten.

Rootglobalprobe nativebuiltbecb916d4798cd84...,128regs24KiB0spills, unchanged
14512digestinstructions, fiveLDG.E.64.CONSTANT atroot instead ofLDC.64. No
packing/sharedarena. Existingtree_audit initialrun wronglyenabled production
speculative/scale defaults: expectednon-1/x errors. Correctexactmode with128
allocation passesall32/64/128tests/direct8192roots butfails256launchsizedabove
allocation. Rebuiltwith256-sizedsharedarenaforpreexistingallsizeoracle1458:
ALL32/64/128/256partialsPASS,98304exactmultiplications,8192exactroots,0status
errors. Logicalreview/sourceguardretained. ExistingN24screen1460runningunder
GPUlock, notclaimedwin. A resourceequalcompile isnotperformanceevidence.

Freshfinaldevsourcepreflight1461PASS, 367.812782M/s (30s exactintegration,
notanotherABperformancepair). Nativeentirebyteequivalence1459PASS. Production
candidatecurrentready: launch1048576andmatchingdeba6085image, allprobesOFF.
Native128regs24KiB0spills. RemovedlocalELFbuildproductsfromeditablepackage
beforecommit/submit to fit8MiBcap; frozen in-flightrootglobalcopyunaffected.
SubmittingimmediatelybestqualifiedcandidatewithSUBMISSION-LAUNCH4.md; respect
watcherownspriorf45slot, recordactualreceipt/refusal. Noofficialscoreclaim.

## Submission bat and next candidate

Localqualifyingbestcommit00541ac submittedsuccessfully immediatelyafterfinal
preflight: **93cf9212-95a6-460d-8abc-a526887be10d**, job773d693b-4d83-49b8-9575-0ab3a80e3f78,
queued2026-10-02T07:18:17.561Z. Receiptretainedsubmission-launch4-receipt.json.
Topboardstillcefika728337167. Priorf45slotclearedserver-sidebeforeacceptance;
noownpollwait/cancellation. FirstCLIattemptdynamic-attributionrefusedcompound
shellbeforeexecution; secondliteralcallrate-limited2seconds; retryaccepted.
Watcherownsthe newvalidation. Officialscoreunknown, no pollwaiting.

Rootglobal1460scorepaircompleteexactPASS BOTH: A382.694900 vs B370.338997M/s
**-3.228656%**. Originalidenticaltableinread-onlyglobalcache not a winner;
leftoff. Existingmathematicalaudit1458alsoPASS with diagnosticconfiguration.
Thiskillsmemory-spaceleverwithoutmisattributingpackeddecode/occupancycost.

Nextcandidatewhilebatinflight: QSB_ROOT_PARK_B. Optionalselfconsultation
suggestedroot-onlyBparkingindeadinversearena toremoveB's24registerliveness
specificallyacrosswarp0divsteps. Reviewedactualtreeimplementation: inverses
areindeaddeaduntilinv16publication; j->rowj/4,column32*(j%4)+lane maps384u64
into3of4rows128columns. Volatilestores/reloadsonwarp0only; restored+warpbarrier
beforeinv16write; noproductarithmetic/alloc/cleanupchanges. Allknobsguarded,
OFFdefault,submitteddebaimageunchanged. Exactreviewiter3-rootparkB-review.md.
Frozenlocal/nativebuild1473running; nextpreexistingbenchmarktestmustexercisenonnull
B (tree_auditdefaultnullptrdoesnotproveBpreservation). Comparingagainstlaunch4,
notoldfill2. NeedfreshcontrolbuildbecausepackageELFscleanedforsubmit.

RootparkBbuild1473PASSlocal+native128regs24KiBzero-spills. Nativeimage729deecac313d821...
(detailsinbuildlog). Truepark/reloadSASS audit retainediter3-rootparkB-native-opcodes.json;
resourcesmaintainfourCTAs. Freshsubmittedlaunch4localcontrol+exactscoredpair1474
running; stillnoproofspeed/exactness. Bat93cf9212watcherowned, noquerypoll.

RootparkB1474screenCOMPLETE all2PASS: A390.027278 vsB370.716988M/s
**-4.951010%**. Definitelynotfirst-screenwin; leaveknoboff. Storedartifactpair
ab-launch4-rootparkB-20261002T023233-3070325. PostbatdefaultCUBINequivalence1477
PASS deba6085entireimageidentical; productionunchanged93cf9212payload.

NextBprepared: QSB_DEN_CROSS_PRE relocatesopposite-Wfactor intoZZ BEFOREtree.
OldhA=ZZ_A*(leaf*WB), hB=ZZ_B*(leaf*WA). NewparkZZ_A*WB,nB.ZZ_B*WA then
unchangedtailformsZZ'*leaf. Relocates2canonicalQSB_TREE_MUL instead ofadding,
letsprodA/BdieBEFOREroot ratherthancarrytheir8u64throughinverse/tails. Same
isomorphic scale andpartialidentitysubstitution; samefieldcongruence,exact
filterstillneedsverification. Existingvectorparkrowload/rewrite costmaykill.
Reviewiter3-dencrosspre-review.md. Defaultoff,compileguardsforunsplittail,
carrieridentity. Frozenlocal/nativebuild1478 running, thenactual120sbenchmark
pairvslaunch4. Watcher93cfownsvalidation. Nopollwait ornewunqualifiedsubmit.

Dencrosspre1478buildPASSlocal+native128regs24KiBzero-spill; earlyfactorrelocation
viablewithoutoccupancycliff. NativeSASScountsretainediter3-dencrosspre-native-opcodes.json.
Scoredpreexistingexactbenchmarkpair1479runningversussubmittedlaunch4control,
120sN24seed1789110211GPUlock. Nobenchmarkcorrectness/speedclaimuntilresult.

Dencrosspre(native form1)has14512instructions likebaseline but+2LDS128+2STS128
AparkedZZupdates. Addedvalue2B-onlydiagnostictoisolateWAregisterlivenesswithout
ANYAsharedroundtrip: AusesoriginaltailinvAandparkZZ; BnewZZ_B*WAwithleaf.
Sameoldtwofactorcanonicalmultiplycount, retireWAonlynotWB. Logicalreview
appendixretained. Frozenbuild1483PASSnative128regs24KiB0spills; score1484queued
underlockafterform1screen1479. Form1frozenbinaryunaffected. Defaultoffentire
CUBINequivalence1482PASSdeba6085; productionstill93cfbatpayload unchanged.

**New positive measuredlead** dencrosspre1479COMPLETE bothPASS A377.377753,
B385.069133M/s **+2.0381117%** vslaunch4. AdoptNOTyet; firstscreenonly.
Form1entirefieldfactorrelocationkernelmaintains128regs24KiB0spillandnative
14512instructions (samebase),14calls41brancheswithtwoadditionalLDS/STS128.
Launched1485threealternatingpairsconfirmationvssubmittedlaunch4controlunder
GPUlock. Form2B-onlydiagnostic1484runningseparately(strictserialGPUlock),
native14520instructions +8butnoadditionalsharedops. Keepbat93cfwatcherowned.

DencrossBnativeSASSconfirmed14520instructions (+8), EXACTsame8LDS1288STS128
32STS6452LDS64asbaseline; thisdiagnosticreallyremovesA'ssharedcost. Binary
native10525b4891a13de1... resource128regs24KiB0spills. 1484Bscorepending;
1485confirmationfirstcontrol389.687889M/sPASS. Bothjobsseriallyinterleave
undertheGPUlock, no sharedproblemconcurrentuse. Newfirstscreensnotclaimedwins.
HEAD1dcc2b3productionstill00541ac-submittedlaunch4configuration,native
deba6085. Bestbat93cfqueuedwatcherowns; qualification/provenance/receiptready.

DencrossB-only1484screen COMPLETE BOTH PASS A389.737215 vs B390.441207M/s
**+0.180632%**. Nearneutral/noadoption. Fullbothfactorform remainsnextpositive
lead+2.038%initialscreen; 1485threepairconfirmationcontinues. Sourceform2OFF.


# Iteration 4 — denominator relocation confirmed; root overlap probe

Standing orders and current board re-read. Top promoted still cefika fb6f5a8f at 728337167 (ff27a2b). Public notes and public tree checked; our launch4 93cf9212 remains watcher-owned. No status polling loop.

Relocation confirmation from iteration3 is complete, all6 scored PASS, seed1789110211,N24,120s, A/B B/A A/B:
- launch4 A:389.687889,393.904044,392.075974 M/s; mean391.889302.
- relocated B:399.367109,398.097286,395.411762 M/s; mean397.625386.
- +1.463700%, all three pair deltas positive.

Direct promoted gate1502 refused before GPU because the packaged control executable had been removed. This is NOT an inconclusive score to retry blindly: rebuilt explicit promoted wrapper, recorded stamp and binary hash, and1504 now performs the corrected3pair gate. Its first A361.3357,B390.4433 M/s both verified. Earlier valid direct gate remains provenance comparison, not replaced by the failed attempt.

New probe QSB_DEN_CROSS_IDLE=1 preserves both original fronts and the same2relocated multiplies. Root warp eagerly forms crossfactors; other warps form them in the existing root idle hook, before downward barrier. No sharedallocation change. The callback writes only A's ownpark rows8..11 and B's ownregister words. Products/inverses remain disjoint. Identity masking occurs first; unused tails still suppressed. Carrier optional key included so stale native image cannot match. OFF retains original source behavior.

Initial build caught guard ordering: ZLAB_TREE is defined in tree_inverse.cuh later than tree.cu knob block. Moved tree-kind guard to the implementing header.1509 rebuilding+native resources+exact scored screen; productionrootidle remainsOFF.

A consult proposed eliminating the last sharedleaf inverse handoff; inspection killed it without a benchmark: this tree already has all nleaf owners multiply parentinverse*siblingproduct directly into value registers. There is no finalleafstore or barrier to remove.

Evidence candidates/subset/lab/iter4-root-overlap-review.md and ab-launch4-dencrosspreconfirm-20261002T025047-3172967.


## Qualified winner and integration

1504completedall6PASS: promotedA361.335660,357.581896,358.817070 mean359.244875; relocationB390.443277,388.587497,392.744373 mean390.591716: +8.725759%. Finaldefaultsourcepreflight1516PASS388.806294M/s,1457/1457hits. Nativehashmatchesqualified152a7bc7e48d5ec0...,128regs24KiB0spills. ProductionnowDEN_CROSS_PRE1; rootidleOFF. HistoricalcontrolsnowexplicitDEN_CROSS_PRE0 tokeepcontrolsemanticspinned.

Toolpathfailures1512/1513werebuildonly: cuobjdumpneedednvdisasm onPATH. Fixedboth searchpaths1514/1516; neitherfailurereachedGPU. Rootoverlap1514scoredscreeninflightwithnativebee8994d...,128regs24KiB0spills. Finalleafpullconsultproposalalreadyimplementedinbase,avoidedduplicateexperiment. SubmitbestnowwithSUBMISSION-DENRELOC.md.


## Root overlap exact screen rejected

1514bothPASS A390.754387 vs rootidleB384.832063M/s (-1.515613%). Native128regs24KiB0spills. Unlike fullrelocation, retaining denominators until root callback defeats the early-liveness gain; no adoption, no costly3pairrepeat of a negative firstscreen. RetainOFFprobe for audit. Next diagnostic A-only relocation separates sharedA scheduling effect from B-onlyneutral result.

Submit285e670firstattemptCLIrate-limited38seconds afterboardread; noID. Retryliteralcallaftercooldown, no polling. Boardunchangedfb6f5a8.


## Submission blocker and A-only next candidate

QualifiedbestliteralCLIattempt725194frefusedconflictaccount1inflight,noID. Receiptiter4-submission-conflict.json. Do notattemptagainuntilwatchernotify. Productionmode1fullnativeidentity1521PASSafteraddingmode3diagnostic;defaultsourceexactpreflightalreadyPASS. Existingnativeheader152a7bc7 remainsproduction.

A-onlymode3nativec0c9bdb3...,128regs24KiB0spills.1520screenneverreachedGPU: explicitroot-levelcontrolnotinstalledbeforemanifestprecondition. Fixedcontrollocationandstampthen1522run. ControlELFcopiedfromqualifiedfinalpreflightbinary,sourceexplicitmode1withsameincludeexpansion; exactbinaryhashmanifestrecorded. Furthercontrolbuildsneedcompletedcontrolartifactbeforelaunch!


A-only1522controlREJECT0.0333s/0hitsbeforeGPUbecausecopiedELFhad0644permissions(shutil.copyfile),notperformanceregression. chmod+xcorrectedlauncherprerequisite;binaryhashunchanged.1523exactscreennowlive. frozen_ab.shnowassertsexecutableandrecordsmodeBEFOREanyGPUrun;avoidthisreplicationagain.1520missingcontroland1522not-executableareseparatepreconditionfailures,neitherisvalidbenchmarkpair.


## A-only actual verifier complete — reject, no duplicated valid pair

1523botharmscompletedPASS, actualscorefiles: qualifiedmode1A406.152518, A-onlymode3B394.256431M/s => -2.928970%. Rejectedadoption. Mode3native128regs24KiB0spills. AutomatedsummaryafterBinterruptedbecausefrozen_ab.shwaseditedinplacewhileitwasrunning; shellre-readatshiftedfileoffsetinvokeddefaultbridge, failedsudo. Theseextraattemptedcommandswerenotcandidatearms. Auditedretainedp1-A/Bjsonandlogs, bothverifiedtrue, reconstructedsummarywithoutrepeatingvalidpair. DO NOT editrunning shell script; freezeorwaitforexitfirst. Productionunchangedmode1/native152a7bc7...,mode3OFF.

Helperauditclarification: QSB_TREE_MUL -> qsb_field_mul_tree -> speculativeqsb_filter_mul underSHORT_CARRY3. Notuniversallycanonicalexactmultiply; associativealgebraonlyexacttarget. Unchangedexacthostpublicationremainsrequired, measuredverifiedyieldisactualcriterion. Publicnotecorrectedandfirstscreensourcedfromactualreceipt377.377753vs385.069133(+2.038112%),notunbackedtranscription.

No newsubmissionID(accountslotconflict). Bestready725194fdeviceconfigurationwithqualifiednativeimage. Nextnecessaryaction: watcherrelease -> literalCLI submitnoteSUBMISSION-DENRELOC.md immediately, otherwise genuinelynewproposaloutsidepartial-factor/rootidlefamily.


# Iteration 5 — centered-square finish, exact PASS; scheduling screen in flight

Board one-shot refresh retained ff27a2b top728337167 and watcher-owned
93cf9212 validating. Read top public note and promoted device tree; read second
note attempt hit CLI rate limit, no poll waiting. No repeated slot-conflict
submission: last blocked receipt is still authoritative until watcher notify.

Materially new arithmetic: p1=S*(m1-c), p2=(S-c)^2-c^2-p1 replaces post3's second
x-offset general multiplication with a triangular square. Canonical identity
holds for arbitrary field slopes; speculative short-carry yield requires scored
measurement, not algebra alone. C2 is host-computed from original R; expanded
C symbol and fingerprint only under enabled knob. Default OFF native image
byte-identical152a7bc7...PASS (473376bytes).

Independent CUDA/OpenSSL audit for actual pre3/post3:4106cases,3926usable,
180unusable,7852keys,624guardbytes,zeroerrors. Initial diagnostic mistakes:
production negated-Y fold and ISO_FAST_X default were incompatible with this
ordinary R/OpenSSL fixture; both explicitly disabled before exact PASS. All
failures retained, not benchmark scores.

Mode1 early square sm89:128regs24576shared0stack0spills; sm86 uses127regs.
One scored pair:control407.413441 vs early398.483444M/s,-2.191876%,bothPASS.
This is an adverse screen, not a three-pair rejection. Native IMAD drops27 but
IADD3 rises21 and SHF14; total digest static instructions14512->14520. Mode2
moves center operation after parity0 so x2 no longer lives across it. Mode2
exact oracle alsoPASS, samecase counts;1535 now runs fixedseedN24screen.
No changed production knob or image; no arithmetic win claimed yet.

Local manifest repair retained: native image prep temporarily prepended mode1
AFTER sm86 executables had compiled; original subset.cu restored, exact build
input and source/binary provenance correction recorded. Only mode1 wrapper
has knob1; control OFF. Mode2 is a separate immutable tree. Build-tool errors
cuobjdump/nvdisasm fixed with explicit CUDA12.8 paths; no timeout used.


Mode2 late followup: native128regs24576B0spills, staticinstruction/opcounts
sameearly but scheduling/nativeinstructiontext differs. Initialscreen bothPASS
-2.3447%. Because AB-only screens are thermally biased, remainingmode2BA/AB
explicitlyjustified in designnotebook andrun1540. SecondpairB404.955090 then
A369.060500bothPASS, +9.725% reversal; rawselfreportedrates similar355.4vs353.6,
verifiedhits5859vs5378. This is noisyyield evidence, not +9.7%kernelspeedclaim.
Launch8screen1543ready andtakeseriallockbetweenarms; recordsequencecaveat.
No defaultknob/imagechanged, no new qualifiedreplacement.


Mode2 centered-square full3pairgate1540ALL6PASS:qualifiedA381.891765,
latecenterB392.558927M/s,+2.793242%,ordersAB/BA/AB. A[406.619419,369.060500,
369.995377],B[397.085372,404.955090,375.636318]. Not submittable incremental4%
claim; broadthermal/yieldstatechangeandinterspersedlaunch8screenremaincaveats.
This qualifies a promising nextstack, NOT defaultchangeuntildirectpromotedgate.
Launch8screen1543PASS+3.36285%(376.295943vs364.053370)butrankedOOM:reject
rankedconfigurationratherthanmistake local smalltablegain forrankablewinner.

CRITICALnewrankedreadinessfinding:GROUP_CAP_EXACThandles1048576epochs only;
currentlaunch4is2097152epochsandfallsbackto4194308grouprecords/stream.
Table22688113472+bothfirststates2147483648+bothgroups1073742848=
25909339968>24GiB25769803776 BEFOREdescriptor/context. FixHOST-only tightbound:
all alignedlaunchspansexhaustivelyaudited13717launches/27434endpointrankchecks:
cap1048576max181498;2097152max362053;4194304max594292. Currentconfigfree
981057280bytes(935.61MiB); groupguardsremainandunknownshapesfallbackunchanged.
SourceknobGROUP_CAP_TIGHTdefaultOFF,1548scoredpreflightlive. Needpromotedfull
stack3pairgatewithtightcapbeforebat; priorlocalgatealonemissedrankedOOM.


AdoptHOSTgroup-capfix now after pairedexactpreflight381.427067->385.404145M/s
(+1.042684%,bothPASS) plus exhaustiveindependentintegerandcompiledproduction
helperPASS. Notclaimedthroughputwinfromsinglepair; readinessbugfix necessary:
old rankedlowerboundexceeds24GiB. ProductionQSB_GROUP_CAP_TIGHT=1,
centeredsquarestillOFF. Carrierknobsexcludepurehostswitch,unchangeddevicecode;
nativeidentitybuildscheduledbeforecarryforwardqualifiedimage. Keep current
promotedstackgate1549immutable (explicitAhostcap0/Bhostcap1).

1548 ended2afterbothscoresPASSbecauseliveorchestrationscriptwasrewrittenwhile
bashhelditopen. CorrectlyreconstructedsummaryfromsavedverifiedJSON; NO GPU
rerun. Nevermutaterunningscriptagain;1549fileunchangeduntiljobcomplete.


IMPORTANTfollowonmemoryaudit:hostproducers.start allocatesbatch0duplicateGPU
selfcheckbuffers671088640bytes(640MiB),BEFOREepochallocs,neverfrees. Tightcap
leaves530.54MiBbeforecontext so tightcapalone STILL rankedOOM. Sourcealias
proposalQSB_HP_CHECK_ALIAS defaultOFF removesONLYduplicateGPUcopy,stillfull
all-descriptor/all-class OpenSSL producercomparison. Aliasreadspitchcorrect
slot0untilfullcheckdone;blockslot0reuseoncheckpass/fail. No sampling/nobypass.
1556exactscoredpreflightlive;requireactualselfcheckPASS+corruptionfallback.
DoNOTcallcec453c rankreadyuntilthissecondmemoryblockerfixedandverified.
1549center/tightcap gatecontinuesbutcannotsettlealiasstackfitnessbyitself.


Directexplicitpromotedgate1549completeALL6PASS:+7.011381%, A353.807181vs
center2+tightcapB378.613950M/s. A[358.423284,352.346808,350.651450];
B[378.785889,377.467303,379.588658]. Qualifieddevicecenter2adopted; matching
sm89native9c9aab2...473504bytes128regs24576B0spillsidenticalfinalbuildPASS.
HostaliasstillOFFpending1556bench/1559actualclean+corrupttests; this is a
ranked-memory blocker,notpermissiontoholdaverifiedrank-readywinner. Before
aliasfix memory cannotfit; batstillblockedwatcher93cfslot no repeatconflict.


FullaliasactualchecksPASS1559:clean2097152descriptors+16777216states
bitidentical;corruptinjectexactly1worddetectedepoch192,SELF-CHECKFAILED
andGPUfallback;bothnormalshutdown0afterexplicitoutcome. 1561corrupt60s
benchmarkexactPASS387.082427M/s,notperformanceclaim. Aliasunchangeddevice
free640MiBscratch,scoredpair-0.19386% (380.584076vs379.846291),bothPASS.
ProductionadoptsHP_CHECK_ALIAS1;center2+tightcap1+alias1 requiresonefinal
combinedexplicitpromoted3pairgatebeforebat (priorcenter+capgate+7.011%).
Slotstillwatcher-owned;notrepeatconflicts,rankedheadroomnow~458MiBafter
legacyd_combos72MiB. Nativeimageidentitymustmatch9c9aab2qualifiedmode2.


## Iteration 5 — full startup self-check alias qualified

The launch-scaled group cap saved 935.61 MiB, but a second memory blocker
remained: `qhp::start` retained 640 MiB of duplicate batch-0 GPU check storage.
`QSB_HP_CHECK_ALIAS=1` now reads the immutable slot-0 descriptors and first
states directly. It records the existing event after the producer kernels and
packs **every** class row with the actual device pitch, rather than comparing
padding. Slot-0 reuse waits until the comparison finishes. No checked rows or
exact publication checks have been removed.

Evidence retained in `lab/`:

- Clean direct check: **2,097,152 descriptors and 16,777,216 first-block
  states bit-identical**; normal shutdown after the explicit pass.
- Injected corruption: exactly one first-state word was detected at epoch 192;
  `SELF-CHECK FAILED` followed by the inherited GPU-producer fallback. The
  separate 60-second fallback benchmark passed all hit verification.
- Alias-only frozen pair: **380.584076 versus 379.846291 M verified
  candidates/s**, -0.193856%; both verified. The alias is a memory repair,
  not a demonstrated speedup.
- A concurrent pinned-allocation failure could previously release slot 0
  while the checker was reading it. The final predicate also requires
  `!chk_running`, and the checker notifies when readback finishes.
  `alias_wait_audit.cu` exercises the actual waiter; **four cases passed**.
- The exact final host integration scored **391.350228 M/s**, with **5,671
  of 5,671 hits verified**. This is not a paired performance comparison.
- The combined native image remains byte-identical to qualified mode 2:
  SHA256 `9c9aab2abc1a6c7368627ae239a40a8e0f4f917df08de43930ee90ee3d48f5ad`.

The combined promoted-source three-pair gate is job 1562. Its frozen host
source predates only the concurrent-failure predicate refinement; the normal
path and native image are unchanged. Separate dependency manifests retain the
host header as well as the field/filter sources. Job 1565 independently
verified the refined final host source.

Two diagnostic orchestration failures must not be confused with kernel
failures: job 1556 passed both scored arms, then incorrectly searched a JSON
artifact for discarded grinder stdout; job 1557 failed before launching the
GPU because nested shell quotes mangled its source argument. The direct check
observer and correctly quoted fallback benchmark settled the missing evidence;
no scored pair was repeated merely to repair its postprocessing.

The submission slot remains watcher-owned. No new receipt/ID has been issued.
The full ranked table/context has not been run on this desktop GPU; accounting
alone is not proof of ranked allocation success.


Combined three-pair gate 1562 complete: promoted A [367.691599,365.869356,
363.794093], combined B [395.665403,393.533729,394.918368]. Mean **365.785016
vs 394.705833 M/s, +7.906507%**, all six exact PASS. Frozen normal path plus
separately verified concurrent-failure refinement is current qualified best.
Submission blocked by existing watcher-owned slot; no retry/no new ID.

Next source edit is deliberately default-off `QSB_TRIM_COMBO_ALLOC`: exclude
unused 72 MiB host + 72 MiB device generic input buffers only with ZLAB_TRIM=1.
Logical audit followed actual main control flow: se_mode consumes separate
buffers, legacy fills/uses compile out, every existing free(null) is valid,
non-trimmed and unsupported-shape behavior retained. No new test added for this
probe. One pre-existing benchmark entry point is job 1570, pending. Do not
claim a throughput improvement from its standalone result or include it in
combined-gate provenance. It was prompted by observed 414.537 MiB local
runtime overhead vs only 458.537 MiB of ranked request headroom, not a proved
ranked OOM.


Latest probe 1570 finished: existing benchmark exact **PASS, 5,709/5,709 hits**,
**394.115136 M/s**. No added proof fixture and no broader test after the pass.
The `QSB_TRIM_COMBO_ALLOC` experiment remains **default-off**. Logical removal
is exactly 72 MiB host + 72 MiB GPU input allocations for the ranked shape;
no live footprint delta was captured before exit, and this standalone rate
is not an incremental performance claim. Public note updated from pending
to verified/default-off. Current qualified production remains the combined
+7.906507% stack, while ranked full-table/context fit and watcher slot release
remain unresolved. No new submission attempt or ID.

## Iteration 6 — slot refresh and next memory-constrained launch probe

Standing orders, manifest and README re-read. Board refresh retained outside the
payload: promoted fb6f5a8/ff27a2b unchanged728337167; our93cf9212 now FAILED at
Benchmark, not promoted. The account slot is free. Qualified combined +7.906507%
all-sixPASS candidate876f89a is sent immediately, default-off combo trim unchanged.
Next experiment combines firstslots8 with launch2097152 to keep first-buffer bytes
constant rather than repeating the previously rejected memory-impossible launch8.
Previous pack8+launch4 lost2.736% on older arithmetic, so no packing speed claim;
this is a capacity-enabling test with current centered tail and tight group bound.

Qualified bat now accepted by CLI as414fe58f-5131-4596-97f8-8ccf4fbdcf7f;
job6ea4aa2f-82ef-4634-8e04-3e0e18e0a0e0 queued, source806b050. Initial command
with explicit track/model arguments+redirect was locally refused as dynamically
unattributable; plain literal benchmark+note-file+json passed and live harness
stamped gpt-6.1-sol/angelX identity. Receipt identity recorded in lab; watcher owns.

Next sum-basis probe is distinct from earlier x2-late experiment: pre3 carries
N1 and N1+N2=2U; post3 calculates slope sum directly and materializes b=sum-a only
after parity0, reusing a's four words. Default0; rejects unsupported alternate
front/tail shapes and fingerprints changed payload whenever enabled. Logical
review retained before benchmark; canonical algebra is not approximate-filter
proof. First build caught guard ordering (NEG_FOLD default defined later); fixed
by accepting undefined legacy-default1 while rejecting explicit0. 1572 is
existing exact benchmark pair then pack8/launch8 pair; 1573 only native build
resource/default-identity audit. No device experiment adopted yet.

Iteration6 next-best screen1572: sum-basis existing benchmark ALL2PASS,
A405.984212(5872hits),B394.619697(5735hits)M/s=-2.799250%. Not adopted; no
confirmation justified by this negative screen. Native1574sum801a50a0...473632B,
128regs24576B0spills; digest14512instructions vs14503default. Off image EXACT
9c9aab2...473504B; default source/carrier payload remains qualified. Findings
and native stats recorded in lab. Memory accounting shows pack8launch8 keeps
first buffers equal2GiB but other arrays grow328.699MiB; remaining129.838MiB
(or201.838withstill-offcombo trim) before runtime. So it is still not rank-ready.
Existing exact pair for packing+large launch continues; no third correctness
test stacked on the successful sum-basis pass.

Previous ranked workflow36989482998 diagnostic retrieved after boardfailed:
preflightPASS, worker exit0 in0.6057s, zero attempts/hits. It contains no raw
grinder output, so exact failure location is unknown. Currentcap/aliasfix has
separate exactPASS evidence, but allocation accounting is not ranked-fit proof.

Pack8launch8 live NVML during same pre-existing run:12572MiB process footprint;
known local major requests12158.162MiB ->413.838MiB gap. Ranked headroom only
129.838MiB, or201.838withoffcombo trim; local overhead is not ranked proof, but
it exceeds both. Useful next direction is packed8 atCURRENTlaunch4 rather than
a too-large launch: can reclaim1GiB first-state VRAM while retesting its speed
under current arithmetic; only proceed if needed by watcher output. No further
GPU work before current pair completes. Submitted best remainswatcher-owned.

1572pack8launch8 COMPLETE: existing benchmark A399.694616M/s5795/5795hitsPASS;
B393.430942M/s5780/5780hitsPASS; -1.567115%. One screen negative, no adoption
or confirmation. All4scoredruns of this iteration PASS. Both probesdefault0;
production subset.cu/nativecarrier unchanged. No ranked success claimed.
Next necessary action: watcher notify414fe58f; if memoryblocked, alreadyverified
trim-combo host repair offers72MiB, packed8currentlaunch can offer1GiB but must
pass new pairedgate because old arithmetic packingregressed. If rankfits,
seek a different arithmetic/scheduling direction; numerator-sum basis has no
screen support despite exact verification. Evidence committed for cold pickup.

## Iteration7 signed centered-square offset fusion, exact-hit screen

Read board/currentleader note and tree atff27a2b; currentbat414fe58f validating,
watcher-owned. New default-off helper folds cc2/p1 subtractions into square's
reduction; materially distinct from sum-basis liveness probe. Frozen current
qualified binary versus candidate existingN24 benchmark120s seed1789110211:
408.239940M/s5898/5898 vs400.125573M/s5800/5800 (-1.987647%), bothPASS.
Native128regs24KiB0spills; same-tool census14512vsdefault14520(-8),
correcting prior mixedcount comparison14503. Noadoption. Firstcompile
failed on const-pointer hostfallback; fixedbeforeGPU. Nativeaudit outputtool
missing fromPATH; extracted cubin via explicit installedtool. Defaultimage
byte-identical9c9aab2..., productioncarrier untouched. See lab/iter7-*.
Signedfold accounting costs8instructions versusunsignedprebiased folding;
unsignedprobe nowhas14504vs14520(-16) and existingexactpair1586is running.

### Iteration7 unsigned continuation completed

Unsignedprebias3p-c² avoids signedhighword accounting and removesa further8
nativeinstructions. Existingbenchmark1pairALLverifiedPASS, current407.542940
M/s5898hits vsunsigned399.423982M/s5803hits=-1.992172%. QSB_K2S_CENTER_FUSED
productiondefault0; both1/2remainOFF, qsb_carrier_sm89.hunchanged. Staticcensus
correctedsametool: default14520 signed14512 unsigned14504; usefulnonNOP14506,
14500,14495. LDS/STS/CALLcountsunchanged. No newsubmission: neithernewprobe
qualifies; bat414fe58fwatcher-ownedatlastreceipt. All build/score/sourceprovenance
retainedunderlab/iter7-*anditer6-centerunsigned-screen-*.

Next materiallead: memory-only capacity repair insteadof furthercenter-offset
retiming. Known ranked allocationbudget458.537MiBcontextheadroom remains tight.
Production combo trim/pitchpacking areOFF; no memory win or rankedfit asserted
fromthisiteration. No additionalnewtestfixturesor costlyconfirmationstacked.

## Iteration 8 — producer-scratch lifetime overlay, measured capacity repair

Standing orders/README/manifest reread; no AGENTS.md present. One-shot board
refresh still has cefika fb6f5a8f/ff27a2b promoted at728337167, prior promoted
source7813ffe1 retained. Both public notes and the two public entry trees read.
Our qualified bat414fe58f remains validating/watcher-owned. No resubmission,
status polling loop, or duplicated iteration7 center-fusion screen.

New materially distinct default-off host knob `QSB_FIRST_PRODUCER_SCRATCH=1`
aliases each slot's GPU group records and epoch map into that same slot's
first-state buffer. Groups/map are consumed by incremental epoch production;
first-state production reads separate descriptors, and follows on the SAME
stream before checker/digest. Checker-readback wait and completed-slot event
still precede overwrite. Group/map ranges remain disjoint and aligned, with
checked fit/size arithmetic; unsupported configurations keep separate storage.
Host uploads bypass producers. No pitch/class/enumeration/device change.
Logical review recorded BEFORE benchmark in lab/iter8-producerscratch-design.md.

Existing verifier only: `lab/iter6_ab.sh` current vs producerscratch, one frozen
N24/120s pair seed1789110211, serialized by GPU lock. Rebuilt missing-stamp
control as required by existing runner rather than replaying an old screen.
**Both scored PASS:** A391.585980M/s5658/5658hits; B394.201799M/s5725/5725hits;
**+0.668006%** is one pair, NOT a speed win or qualification. Standard diagnostic
Poisson warnings retained (GPU self-count vs total verified hits); exact
published-hit verification passed. No raw startup-checker stdout retained by
existing wrapper, so do NOT claim another full checker pass from this screen.
No new fixtures, alternate builds or overlapping broad verifier after PASS.

New concrete memory measurement DURING the same pair: NVML process footprint
A12244MiB (34samples all identical), B12136MiB (60samples all identical),
**108MiB less**. Requested allocation saving109462784B=**104.391846MiB**,
2*(362053*128+2097152*4). Per-slot map starts46342912B (128B alignment pad)
within1073741824B first-state allocation. This is a measured local capacity
improvement, not ranked4090 memory-fit proof or speed claim. Accounting-only
ranked major-request headroom rises458.537->562.929MiB; separate combo trim
could add72MiB, but BOTH knobs remain OFF and their combination untested.

Local executable device fatbins BYTE-IDENTICAL (1126120B, b4a0440f...);
qualified carrier header unchanged(add141c2...). Production subset.cu untouched,
128regs24KiB0spills qualified image retained. Probe source, frozen manifests,
exact scores, memory timeline, device identity and final run metadata retained
under lab/iter8-* and lab/iter6-producerscratch-screen-20261002T073951-528695.
See lab/iter8-findings.json. Next necessary action: watcher notification on
414fe58f; if ranked allocation is the blocker, this now offers a verified,
measured104.39MiB repair without pitchpacking's device changes. Before enabling,
qualify against promoted control as required and do not infer full startup
checker coverage from hit verification alone. No claim of loop completion.

# Iteration9 — new promoted source, final gather prefetch killed, scoped SHA lead

Orders re-read; board newleader4cc9d2d8/kshitij-hash,736585478,public2f57d80.
One-shotiter9-board-summary retains watcher-ownedbat414fe58fvalidating, no pollwait.
Read newleader publicnote, complete device tree and GPUHash, previousfb6publicnote.
All changes scopedcandidates/subset, sharedGPUlock, existingN24/120s verifier.

NEW cache-scheduling probe: final-only addresspeek+twoL2prefetchsectors during
penultimate addition; no coordinate live-range added. Frozenlocal+native1595
PASSboth:A410.106447,B395.756486M/s,-3.499082%;5925vs5726hitsallverified.
Native128reg24KiB0spills,14520->14552instructions; twoCCTL.E.PF2emitted. ptxas
interleaves them into pointadd arithmetic rather than strictly at sourceboundary.
DefaultnativeOFFbyteidentity9c9aab2...PASS. Rejectsinglepair,keepOFF,no expensive
3pairfollowup. Exactbench artifacts+reviewiter9-finalprefetch-findings/design.

NEW scopedarithmetic port: leader's inherited public rotate-add SHA inGPUHash
only(defaultOFF),not speculativefield or runtimeFMA/gatehelpers/hostchanges.
1598exactPASSbothA392.855587,B398.051871,+1.322696%;5698and5790verifiedhits.
Native14520->14256instructions(-264),LEA.HI+541,SHF64-1620,128reg24KiB0spills.
All4producer/digestnativefunctionschange: not isolateddigestkernel gain.
OFFnativeexactmatchesqualified9c9aab2...;ON82b9868e...,currentproductionunchanged.
1602direct3pairgatevsNEWpromotedlive; adoptionrequiresall6PASSand>=4%bar.

Controlrepairconcrete: newleaderdoesnotimplementourQSB_LOCAL_SM86shim. First
1599freshcontrolstartupREJECT0hits1.7s(native21.1GiBtable);notregressionmeasurement.
1600correctedexplicitGLV12buildthenfailsZDEC-onlyDECODE_CUTguard;disabledcuton
nonZDECcontrol.1601nowactualgateNEWpromotedvsalready-qualifiedunchangedcurrent.
FrozenwrapperexplicitGLV11=0,Q_P18=0,Q_MIX=0,ZDEC=0,NM_MASK/SEED=0,
GATHER_ONE_FORM=0,DECODE_CUT=0. Onlygeometry-incompatible controlsdisabled;
leader arithmetic/hostdefaults otherwisepreserved. Source/Binary/stamp manifests.
DevGLV12doesNOTprove4090nativeperformance,butisstandingordersrequiredlocalgate.

Nextdistinctextensionprepared: defaultOFFSHA_LEA_GATE1 ports ordinaryQSB_RL
only, no specializedrounds orFMA. Before/aftercontrolSHA1scopedportnotSHA0;
exact32bitsums reviewed. Builds/screens tofollow whiledirectgate islive.
No newID; existingbestbatwatcherowned. Noqualifiedreplacementyet.

Iteration9checkpoint: ordinary-gate extension1604buildPASSnative128reg24KiB0spill,
scopedSHAOFFextensionbyteidentity82b9868e...PASS,ONc3f6caec...,14256->14128
(-128instructions). Nativecompilerrecordscompareiniter9-rotateaddgate-native-census.
Existingbenchmarksstilllive;refreshedcurrentgatefirstpair351.738638vs381.064M/s
(+8.34%)bothPASS,butnotqualificationuntilall3pairs. Gatejobsinterleaveonlyunder
seriallock. Packageaudittrackedcandidate5.603MiBcap8MiB,untrackedELFsnotpackaged.

Nativecensusnewfinding: previousliteralNOP;filtermisses"NOP ;"spelling,including
predication.Fullanchoredopcodecountgivesdefault20/scoped21/extension17NOP versus
literal14/15/11. Totalinstructions14520/14256/14128unchanged. Persistthisdifference,
don'trepeatinconsistentnonNOPclaims. Existingbenchmarkscoreunaffected.

Ordinarygateextension1604COMPLETEactualexactPASSboth:A401.315271vsB359.765211
M/s,-10.353471%,A5797/B5206allverified. Resourceequal-128instructionsisnotwin.
ExtensionOFF/no3pairfollowup. ScopedpositiveSHAlead1602directgatecontinues.
Preparedstagedscopednative1605matches82b9868e...entireimage,productionstillOFF.
FinalcurrentliveOFFnativebyteidentitycheck1606runningafteralloffprobesadded.

## Iteration9 final measured decision

1601NEWpromoted vsUNCHANGEDqualifiedproduction all6exactPASS:
A[351.738638,376.553570,367.159060]mean365.150423,
B[381.064000,390.466372,394.740052]mean388.756808M/s,+6.464839%.
Currentbestre-qualifiedagainstcurrentboard,alreadybat414fe58fwatcherowned.
1602NEWpromoted vsSCOPEDSHA all6PASS:
A[375.853448,362.365177,374.776564]mean370.998396,
B[396.561924,363.637879,361.146810]mean373.782204M/s,+0.750356%below4%bar.
REJECTscopedadoption. Initial+1.32%leadspent;cannotpromoteontotalinstructioncount.
1604ordinarygateextension-10.353471% bothPASS remainsOFF.
1605stagedscopedcarrierbyteidentity82b9868ePASS;1607exact30s1451/1451hitPASS,
382.182114M/sintegrationonly(notwinningmeasurement). CarrierNOTinstalled.
1606FINALliveOFFentireCUBINidentity9c9aab2...PASSafterallknobs/portsadded.
Defaultproductionunchanged. NoofficialsubmissionresultornewID. Fullall6receipts,
chronologicalarmfinishrecords,decisioniter9-findings.json. Aliasauditactualenabled
PTXbindings1612running,notbenchmarkornewverifier. Nextdistinctleadshouldfocus
newleader'snonSHAchangesornewproducer/memoryidea,notrepeatthisfailedscope.

1612actualmatchingCUDA12.8enabledPTXbindingreviewPASS:384ORD16RLAblocks each
sm89native+localcompute52,zeroearlyh-outputaliasesagainststillliveMaj/oldd.
Conditionalsecond-opinionconcernclearedbyemittedbindings,notGCCassumptions.
Evidenceiter9-rotateadd-ptx-binding-audit.json. Nofixture/verifier/codechange.

## Iteration 10 — row-half first-state interleaving saves 1 GiB, fails speed screen

Standing orders re-read; board refreshed once, top two notes read again, current
leader tree retained from iteration9. Leader remains 4cc9d2d8 / 2f57d80 at
736,585,478 verified candidates/s. Our bat414fe58f is now **failed**, no official
score: workflow37007650774, Benchmark step. Downloaded official failed log and
artifact (lab/iter10-official-*): GPU preflight PASS, wall0.614531435s,
self-elapsed0.49s, zero candidates/hits. Raw grinder stderr/return code is NOT
retained by gpu_wrap; bridge status0 is not candidate success. Major ranked
requests24,117.463MiB leave458.537MiB before CUDA overhead. Startup VRAM pressure
is credible but the precise failed allocation is UNKNOWN. No retry submission
of the same broken production. Root notebook not edited: editablePaths confines
this iteration's mutations to candidates/subset.

New direction implements QSB_FIRST_SLOT_INTERLEAVE (default0) in tree.cu HOST
allocation only. Keep512B device first-state row stride. Pipeline slot0 accesses
first8classes/256B; slot1 base is shifted256B and accesses the other8classes.
One1073741824B arena instead of two. Exact window classes, capacity2097152,
digest/producer kernels, host 2D copy pitch and existing checker reuse barrier
unchanged. Guard WINDOWS128/SLOTS16/runtimeclass8; mutually exclusive with
FIRST_PRODUCER_SCRATCH. Unsupported cases retain independent buffers. Reviewed
implementing producer/digest accesses, host upload/check copy widths, slot reuse,
startup-check failure paths and teardown BEFORE test; design recorded in
lab/iter10-firstinterleave-design.md. This is NOT firstslots8 pitch packing and
NOT the prior transient group-map scratch overlay.

One smallest existing N24/120s paired screen only (proc1642), seed1789110211:
lab/iter6_ab.sh -> benchmark.sh subset, GPU flock, frozen HEAD3a23ebf control.
A355.117402M/s5167/5167 hits **PASS**; B335.630321M/s4880/4880 **PASS**.
**-5.487504%** single pair: no confirmed regression claim, no costly
confirmation/threepairgate after this negative screen. Compared earlier
iteration8 scratch screen +0.668% (noise),108MiB saving: this material new probe
saves MUCH more memory but does not qualify on speed either.

Concurrent NVML observation during SAME scored runs (proc1644): control56samples
ALL12244MiB; candidate60samples ALL11220MiB. **1024MiB less process VRAM**,
matching requested1GiB reduction. Accounting-only ranked headroom1482.537MiB;
not proof of ranked context/table fit. Existing gpu_wrap discards raw stdout:
do not infer another full startup checker pass just from scored-hit PASS.
Local .nv_fatbin payloads1126120B byte-identical, b4a0440f...; checked-in native
carrier header add141c2...UNCHANGED. Section extraction with objcopy rewrote
idle candidate ELF before B run (not its device section); actual binary hash
correction and original hash retained in screen manifest. A rewrite failed
ETXTBSY and original A hash stayed intact. Not a device mutation or verifier.

Decision: probe remains0. Verified production subset.cu/native header unchanged.
No new submission or ID: current readiness is unresolved and this candidate
fails adoption screen. Evidence lab/iter10-findings.json, device-identity.json,
NVMLjsonl and full existing-verifier score/log receipts. Next needed direction:
a ranked-memory-safe layout that avoids this row interleaving's negative screen,
or obtain raw ranked startup diagnostic to locate the exact failure. Do NOT
repeat already-negative scopedSHA/offset fusion or same memory screen. A runtime
budgeted launch-capacity repair is a distinct possible lead but remains untested;
it must preserve contiguous epoch coverage and be gated against current leader.

## Iteration11 — runtime budgeted launch capacity; verified negative screen

Current board initial snapshot unchanged leader4cc9d2d8/source2f57d80 score736585478;
read its public note and source tree. Prior bat414fe58f remains terminal failed
at startup, no raw stderr available, no established failing allocation. New
host-only `QSB_LAUNCH_BUDGET` default0: after table construction query free VRAM,
choose2^21 or2^20 epochs so independent two slots fit with1GiBreserve. All
allocations, group bound, host producer cap, both consumer loops use the same
runtime count. No rowhalf alias, altered pitch, checker ownership/barrier change,
candidate skip, or device computation change. Supported shape guard isolates
137/6 omission split, windows128, firstslots16, tight groups, normal independent
first arrays, full cap1M/2M. Failure below mincapacity reports required reserve.
Producer2^20 divisible by8pieces*16384chunk. Logical state/cleanup review in
lab/iter11-budget-design.md preceded benchmark. Test-only cap1M forces local
branch; automatic ranked low-memory selection still not executed on RTX4090.

1648 existing `lab/iter6_ab.sh -> benchmark.sh subset` frozen b5fe417 production
control vs current probe, N24 seed1789110211 120s/arm serialized GPU lock:
A403.627048 vs B374.997364 M verified candidates/s, **-7.093103%**. Both PASS:
5846/5846 vs5400/5400 hits. Single negative pair is an adoption screen, NOT a
confirmed stable regression. Neither followup3pairgate nor submission warranted.
Selfreported candidatecounts both warn vsPoisson hit band; score uses verified
hits only. No full checkercompletion claimed from this scored-hit PASS.

1649 native sm89 budgetON image exactly473504B sha9c9aab2abc1a6c73... byte-identical
to qualified checked-in carrier. Native digest128registers24KiB0spills. Local
ELF fatbin1126120B b4a0440f...byte-identical between arms; read-only ELF parser
avoids previous objcopy in-place rewrite hazard. Qualified subset.cu/carrier
unchanged. Budget remains0, forcedcap used only labwrapper.

Memory requested savings1,262,570,240B=**1204.080811MiB** (slot arena2408.391846
->1204.311035MiB, exact groupcaps362053->181498). Candidate GPU process observed
11040MiB point sample, vs priorfullcap12244MiB:1204MiB, matches rounding. This is
not simultaneous paired-memory-trace evidence. Ranked major allocationheadroom
would rise458.537048->1662.617859MiB; full fit/context not proven. Distinct from
iter10 rowhalf1024MiB savings; no speed-qualified remedy yet.

Artifacts lab/iter11-findings.json; immutable iter6-launchbudget-screen-... full
logs and scoredJSON; local/native identity/resource logs; frozen control source
manifest; allocation accounting and NVML snapshot. Next materially new route:
first obtain ranked raw startup stderr, or compact padded first-state device
pitch at full2Mlaunch (changes carrier, not another hostonly alias/cap screen).
No new bat/ID because unqualified. Existing production readiness unresolved.

## Iteration12 — dual-carry portability measured, isolated leaderMAC below bar

Standingorders read; board one-shot currentbest4cc9d2d8/2f57d80 at736585478.
Read public leader note/tree. Latest qualifiedproduction carrier9c9aab2... and
subset.cu unchanged; prior414fe58f terminal officialFAIL is startup blocker,
not speed evidence. No new bat without a newly qualified, verified change.

New default-off QSB_YP_LEGACY_DC moves an opaque8Bconstantzero to firstcarry
capture and combines both carries in second non-.cc addc. Neither carry dropped;
CC continuity unchanged; noenumeration/SHA/hostpublication change. Mask1oddpair,
mask3odd+top. Leader's eightfusionMAC not already present in ourpm9; simply
turning its knob on here would do nothing. Isolatedlegacymechanism implemented
in real y_pair_sc.cuh emitter; fingerprint changes only when enabled.

Existing benchmark.sh subset,N24,120s,seed1789110211,per-armflock; frozen
qualified control binary SHA3bff1ff6... and stamp pinned by manifests. Allsix
scoreJSON verifiedtrue and logs RESULT PASS:

|Probe|Control M/s|Probe M/s|Delta|Control/Probe verifiedhits|
|---|---:|---:|---:|---:|
|Legacyodd carry|389.775892|355.265565|-8.853890%|5634/5172|
|Legacyodd+top|395.751801|352.340156|-10.969412%|5722/5121|
|IsolatedleaderMAC2/DC1|358.775324|360.086933|+0.365579%|5216/5244|

Rejectbothlegacy screens; no confirmation expense. MACtinypositive is NOT
qualified and too small to claim improvement; keepdefaultoff, no adoption.
Each rowsinglepair, controlsnot directly comparable across thermals/yield.
Selfreported candidatecounts disagree with hitrate for somearms; score uses
only independently verifiedhits. Harness says4090 but actualdevice3090.

NativeCUDA12.8.93 sm89 same-tool digestcensus14520default ->14552legacy1 and
14560legacy3 (+32/+40). CapturesSEL -3/-6 but moremoves/predicateconversion/
logic. Mask1 adds8Bstack with LDL.LU.64/STL.64 despite0spillreport; mask3no
stack. Do NOT callmask1 zero-localstorage basedon spillsalone. All128regs,
24576Bshared. IsolatedpublicMAC from2f57d80:14464(-56),0stack/0spills,
128regs24KiB. One8Bconstant preserves alignment. Its whole row-interleaved
product accumulator schedule is needed to avoid legacy liveness damage.
No proof that reducedinstructioncount boosts currentcentered-squarekernel.

Nativefinaldefault rebuild SHA9c9aab2abc1a6c7368627ae239a40a8e0f4f917df08de43930ee90ee3d48f5ad
exactmatchesqualified carrier; default paths byte-identical. New macros are
0 and optionalknobstrings empty. No rebuiltproductioncarrier adopted.
Census includes predicatedNOP; corrected samplecounts may differ from earlier
paddingNOP parser though total14520 unchanged. Rawcubin/disasm stayslocal,
compact resources/census retained lab/iter12-native-summary.json.

Repro: lab/iter12-dc-screen.sh; otherarms compile lab/n24L_iter12dc3.cu or
lab/n24L_iter12mac.cu then use lab/iter6_ab.sh against frozen initialproduction.
Buildflags -O3 -DQSB_ZEROS_N=24 -Xptxas=-v -lcrypto -lm; native -arch=sm_89
-cubin -DQSB_CARRIER_BUILD=1 with respectiveknobs. ExplicitCUDA12.8tools;
firstnative disassemblerfailed missingnvdisasmPATH, corrected without GPUrepeat.
Spawnedread-only review made no edits; its suggestionswere not measurements.
Only existingbenchmark scorer constitutes publishedhit verification.

Artifacts: lab/iter12-findings.json, iter12-carry-design.md, native-summary,
dependencyhashes and three frozen A/B artifactdirs. NewGPL-derived public
MACsource attribution in y_pair_mac_leader.cuh; QSB_YP_MAC_PORT0 isolatesit.
No officialsubmission or localwinner thisiteration. Remainingstartup blocker:
obtainrawofficialstderr or prove rankedgeometry capacityfit. Avoid repeating
legacyconstantzerospellings; full2Mlaunch+compactfirstpitch(currentarithmetic)
is distinct from previouslynegativehostrowhalf/cap-halving repairs.

### Iteration12 continuation — fullcap pitch lead and startup boundary audit

Newcurrentarithmetic QSB_FIRST_PACK81 uses8classes,256Brow,2Mepochsperslot,
independenttwoslots. Existingruntimeclassguard prevents unsupportedshapewrite.
Notrowhalfinterleave, notlaunch-halving, notpack8+launch8 oldprobe. Exactpaired
benchmark production373.512368 vscompact374.850759M/s(+0.358326%),allPASS
5441/5428verifiedhits. No standalonespeedclaim; removes1GiBrequestedstorage
while preserving throughputscreen. ProcessNVML11220MiBvsprior12244MiBfullcap
consistentwith1024MiBreduction. Not simultaneous memorypair; allocationproof
separate. Currentpromoted3alternatingpairgate1684running; p1+5.34%,p2+8.79%,
allfourPASS. Nativecarrierprepared(notinstalled)SHAa724399a...473504B;exact2
addressinstructionschanged,128regs24KiB0stack/spillssame14520instructiontotal.
NoteSUBMISSION-PACK8-FULLCAP.md preparedwith explicitpendingqualification.
No newbat untilcompletegate+finalintegration.

Deliconsultation exploredmissingstartupdiagnostics and denseorbitfolding.
ActualGTtablesix/eight1Doddmultiplesegments indexedGLVdigits,fullXY64B,not
redundant2Dcoefficienthashlookup; orbitfoldproposalassumptionfalse, killed
withoutGPUtest. Auditednativefallthrough: sm89load/lookup/knobmismatch all
fall backcompute52; known128CTA16384registersblock+24KiBsharedvalidbudget.
Image/ABI metadata notruntimeproof. Tablebytescastsize_t beforemultiplication;
GT_TOTAL_ENTRIES354501773u*64 wide22,688,113,472B, staticassertmatches. First
slotsareindependent1GiB and allmajorallocation/checkerexpressionscastsize_t.
Killsobvioussigned32extentoverflowhypothesis, no exhaustiveindexproof.
Officialrunnerdriver580.178.04 CUDA12.8.93 compatiblefamily; rawstderrnot
includedbecauseworkflowdiagnosticuploadonlybenchmark-results+metrics.json.
Nativefailure/OOM stillunproved. Moredetail lab/iter12-pack8-startup-audit.md.

Continuationgate1684 COMPLETE ALL6PASS: control345.648608,338.250383,
343.649241M/s (hits4975,4884,4964); compact364.097521,367.970785,
369.260956M/s(hits5290,5349,5339). Means342.516077vs367.109754,
+7.180298%;eachpairpositive5.337477%,8.786508%,7.452865%. Meets>=4%bar
vsactualcurrentpromoted2f57d80/4cc9d2d8. Presubmitboard736585478unchanged,
ownslotfree atoneshotcheck. Installonlypreparedsource/carrierexactSHAa724399a,
entrypointQSB_FIRST_PACK81. Finalinstalled30sN24benchmark1697running;
immediatesubmitafterPASS, no additionalqualificationgate. Legacy/MACOFF.

Finalinstalledpreflight1697 COMPLETE1289/1289PASS346.487571M/s30sN24.
Qualifiedproductioncommit9b6d55a,carrierSHAa724399a473504B128regs24KiB0stack/
0spills. Submissionattemptimmediately: guardrefusedcombined/redirectedcommands
(dynamicargumentattribution)beforeexecution; literalunredirectedCLI SUCCESS:
submissionf0225b8a-0fc9-4ec3-8b18-bde97dcc95d3,job0031a649-6885-47f7-a02a-d60e66d35987,
created2026-10-02T17:14:32.011Z QUEUED. Watcherownsslot,nopollwait. Compact
receipt lab/iter12-pack8-submission.json retained; publicnote10377+resultsbytes,
Effortxhigh,model/harnessflagsstamped. Package6884622bytes<8388608.
Noofficialresultyet. Nextresearch prepared separately;do notaltercarrier without
newmatchingimage/fingerprintoradoptunqualifiedprobe. Currentbestpack8enabled,
alllegacycarry/MAC/rowhalf/budgetexperimentsOFF. Startuprootcauseunproved.

Nexttreeprobe1699COMPLETE bothN24scoredPASS: submittedcompactcontrol
386.343611M/s5597/5597vsQSB_TREE_LIVE_MASK1 385.370820M/s5580/5580,
-0.251794%. Singlepairflat/negative notconfirmedregression, nofollowupgate.
Rejectadoption/no1%signal; defaultOFF. Arithmeticloads/multsforlaneswhose
resultsarediscardedaremasked A8/B20/C18/D17/inv16;rootwarpcollectivesremain
unconditionalandbarriersunchanged. ExactpublishedhitverifierPASSnotall-domain
proof. Native14552(+32) vs14520submitted,5eachBRA/BSSY/BSYNCplusmoves/
predicateoverhead.128regs24KiB0stack/spills. Defaultrebuilta724399a byteidentical
submitted; noqualifiedproductionchange. Source/receiptcommitpreservesdefault
probeonly. f0225b8a watcherowns, noofficialresultyet/noquery. Evidence
lab/iter12-treemask-findings.json,design,native andfrozenA/Blogs.
