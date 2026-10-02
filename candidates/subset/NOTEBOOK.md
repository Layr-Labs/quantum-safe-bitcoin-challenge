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
