# Iteration5: full startup self-check alias, host-only default OFF

Further ranked memory blocker: host_producers_v3.start retains duplicate GPU
batch0 scratch:2097152*(64+ncls8*32)=671088640bytes (640MiB). It allocates before
epoch buffers and never frees; even tightcap leaves only530.54MiB. Tightcap alone
NOT rank-ready. New alias keeps ALL original self-check descriptors and class
words; NO sampling or bypass. Batch0 GPU output slot0 is immutable until checked.
Event recorded same location after producer kernels; worker cudaMemcpy2D reads
allclassrows with original16-slot sourcepitch andpacked8-classdestinationpitch.
Before any reuse of matching d_ep, main waits on check outcome underh->m/cv_ready.
Worker completesfullreadback beforecheckpass/fail andnotifies; mismatch falls
back toGPUproducers unchanged. Alias removes fullscratch allocations/D2D copy.
Need live verified preflight with actualselfcheckPASS, corruptioninjection test
mustreportSELF-CHECKFAILEDfallback, no deadlock. Native image unchanged hostonly.
One-time slot0reusewait may affect timing, not kernel claim. Do not adopt until
preflight+fallback pass. Combinedcenterpromotedgate1549 still noalias; its
throughput cannot provealiasstackfitness/rankedfit. Requirefurtherwholegate.

Diagnosticartifactroutingmistake: standardgpu_wrap discardsgrinderstdout and
run-subset.jsonlivesinsharedbridgeoutnotrepo. 1556/1557post-run markerassertions
cannotproveselfcheckfromartifact (andwillfailafterverifiedruns); retain scores,
no rerunofscoredpairs. New dedicateddirectbinaryprobe observesactualfullcheck
stdout, requests normalSIGTERM onlyAFTER explicitPASSorcorruptFAILfallback.
No generic timeout; checkerresult,nottimeorprocess exit,is acceptance.
EachfullcheckprobeholdsGPUflock; same frozenaliasbinary. Separateofficial
benchmarkhits still verifier authority forpublication/throughput.

AliasfrozenpairALL2scoredPASS380.584076controlvs379.846291alias(-0.193856%).
ActualdirectcleanfullcheckPASS2097152descriptors+16777216first-blockstates
bitidentical,normalshutdownafterexplicitoutcome. Notasamplingcheck. Corrupt
actualprobequeuedafter60sscore fallbacktest. 1557shellnestedquotingfailedBEFORE
GPU (sourceargbecamePythonfragment);fixed1561usesenvwithoutnestedquotes.
1556bothscoredPASSbutitsmarkerassertionfailedafterartifactdoesnotcarrystdout;
noGPUrepeat. Actualchecker1559distinctrequiredgate,notbenchmarkretry.

CorruptactualprobePASS:exactly1first-stateworddetectedat epoch192,zero
bad-descriptors;SELF-CHECKFAILEDthenoff(self-checkmismatch)->GPU fallback.
Normalshutdown0onlyAFTERexplicitfailureoutcome. 60scorruptbenchmarkexact
verifierPASS387.082427M/s(no speedclaimfromcorrupt/differentruntime).
Adoptalias1productionafterALLdistinctgatespass. Removeretained640MiB GPU
scratch, notcheckeddata; fullcomparisonbitidentical cleanPASS,correctedpitch,
slotreuseguardactual. Finalcenter2+alias1+tightcap1wholepromoted3pair gate
nextbecausepreviousqualificationdidnotincludehostaliasreadbacktiming.

Code-reviewfailure-racefix:ifpinnedringallocationfailuresstopsworkerswhile
anotherworkerisstillrun_checkreadingslot0, originalwaitdeadpredicatereleased
slottooearly. Refinepredicate(outcome||dead||stop)&&!chk_running, notifyallwhen
run_checkreadbackfinishesbeforefailureprocessing. Successfulpathunchanged.
Addcompiledactualwaiterconcurrencyfixture: concurrentdead/pendingcheckmust
NOTreturnuntilreaderdone, checkpasswithreaderrunninglikewise;failfinishedand
unrelatedslotreturn. DiagnosticfixneedsactualhelperPASSplusintegrationtest;
finalfullgatefrozenprecedingaliasnormalpathunchanged,manifestexplicit.


### Current status

Production enables the alias after the clean, corrupt, concurrency and exact
integration gates above. The full comparison is unchanged. The combined native
image is identical to qualified mode 2. Combined throughput gate 1562 is still
running; no submission is attempted until it qualifies and the watcher releases
the existing slot. Ranked context headroom remains a caveat.
