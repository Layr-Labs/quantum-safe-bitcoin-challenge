# Next probe after submitted compact-pitch winner: live-lane tree waves

`QSB_TREE_LIVE_MASK` defaults to0. The mechanism comes from current public
leader2f57d80's QSB_TREE_LANEMASK, but this patch preserves our existing
four-limb shared rows, root normalization, relocated denominators, and128CTA.
It does not import leaderrootlazy/row128/rootcombine or otherknobs.

## Logical review before measurement

All lanes of the selected root warp continue to execute every syncwarp,
root broadcast shuffle and zi_inverse_limbs collective. Masks apply only to
field multiply/load blocks with no collective instructions. Sharedwave mode
required; enabledshuffleexperiment is rejected at compiletime.

WaveA: only0..7 products are stored; othercopies are dead.
WaveB: lanes0..15 form c[i],16..19 formP4. These20results are all consumers
needed by nextwave; overwrittenr of otherlanes is initialized0.
WaveC:0..15 form d[i],16..17 formP2. Prior requiredr resides on samelanes.
WaveD:0..15 formE16,16formsroot. Broadcast lane16 reaches everyrootwarplane.
Finalinv16: only0..15 results stored, otherlaneproducts discarded.

Masks cannot remove data needed by anotherlane: onlystored columns are read
in the nextwave; same read-after-write syncwarp remainsoutsidecondition.
Rootinverse usesbroadcastroot, notdeadlane r. Finaldown-sweep unchanged.
Inactiveepochs stillsupplyidentity leaves and exacthostpublication unchanged.

Fullwarp instructionissue may remain largelyidentical evenwith feweractive
lanes; benefit mustcome fromsharedbank/accessand threadwork, not assumed
32-lane instruction savings. Predicate overhead can hurt. Native census and
resource comparison distinct from speed evidence.

## Measurement

Compile lab/n24L_iter12treemask.cu withlocalgeometry,N24 andprebuildstamp.
Use existing lab/iter6_ab.sh: A frozenexactsubmittedcompactbinary
lab/n24L_iter12final.cu, B newmask,one120spair,per-armflock.
No judge edits. Native sm89 bothmask andfinaldefault cubins measured separately;
default must match submitteda724399a... image. Rejectifnegative; apositive
screenwould requirethreepairs andcurrentleadergatebeforeadoption.

Atnotecreation build/screen1699 andnativecensus1700running. NoPASSorwinclaimed.
Submissionf0225b8a watcherowns; nextprobe notpartofsubmittedarchive.

Nativebuild1700done: maskSHA37ef1bb5...474144B;128registers,24576shared,
0stack/spills. Digest14552(+32), +5BRA/+5BSSY/+5BSYNC, extraCS2R4/moves6/
ISETP5/LOP3five, NOP-3. Nativeconditionalcodeexists; noassumedspeedwin.
DefaultnativeSHAa724399a... exactlysubmitted473504B. Dependencyhashaudit
shows onlytree.cu/tree_inverse.cuh nextprobe edits;productionentrypoint/native
carrierandotherdevicefilesunchanged. Candidatebenchmark1699 stillpending.
