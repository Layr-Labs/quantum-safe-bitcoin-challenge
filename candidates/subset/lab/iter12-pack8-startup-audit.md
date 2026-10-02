# Iteration12 continuation — compact pitch and startup audit

## Full-capacity compact-pitch invariant

QSB_FIRST_PACK8=1 changes only the physical first-state row from512B to256B
for128windows. There are8 distinct first-block classes. Preparation computes
firstclasses into temporary bounded QSB_SE_PER_EPOCH arrays, checks
first_distinct <= QSB_FIRST_SLOTS before copying/transposing into8slot storage,
and returnsfailure for unsupported classcounts ratherthan writing outofbounds.
All producer, upload/checker pitches and consumer rowaddresses use the same
QSB_FIRST_SLOTS. Host firstwidth remains ncls*32=256B. Both existingpipeline
slots retain2,097,152epochs; descriptors/groups/taglimits unchanged. Existing
startup alias+checker-readback reuse barriers unchanged. Packedlayout cannot
use row-half sharing, whose enablepredicate requires16slots, so no overlap.
Totalfirstmemory1024MiB vs2048MiB, savingexact1GiB. No candidate-domain cuts.

CUDA12.8 sm89 census: same14520digestinstructions,128regs24576Bshared,
0stack/0spills. Exactlytwoinstructiondifferences from qualifieddefault:
row IMAD.WIDE multiplier0x80->0x40; paired-epoch LOP3 offset0x80->0x40.
Matchingcarrier built in isolatedreadytree, SHAa724399a04db515f...473504B.
Notinstalled until leaderthreepairgatefinishes. Currentproduction unchanged.

## Native startup non-OOM audit

Consultation recommended inspecting native load/launch ABI before assumingOOM.
Actual carrier checks sm89; libraryload/functionlookup/imageknobmismatch all
turncarrierOFF and fallbackto compute52. Kernelargcontract uses commontyped
qsb_carrier_try templatepacking, mangledsymbols regeneratedwithbuildtool.
Samequalifiednativeimage reproducible,128threaddigest (128*128=16384regsCTA)
and24KiBstaticshared; producer/tablebuilder256CTA useindependentresources.
No changedkernelparameters in pitchprobe. Failedlibraryloadalone cannot be
claimedcauseofzero-workbecausefallbackexists. Real sm89 runtime still not
locallytested on3090. Nativeimage metadata/audits are NOT runtimecompatibility
proof.

Inspected largeextenttypehypothesis: GT_TOTAL_ENTRIES354501773u <2^31;
gt_sz=(size_t)GT_TOTAL_ENTRIES*64 yields22,688,113,472B correctly. Existing
static_assert matcheswideallocation. Each priorfirstslotis1GiB (not one2GiB
signedexpression); allocationuses launch_epochs size_t timesQSB_FIRST_SLOTS
and8*sizeof(uint32_t). Hostchecker similarly casts n0 tosize_t beforeproducts.
This kills the specific obvious first/tableextent signed32overflow hypothesis;
doesnotexhausteveryindexexpression or prove startupallocation succeeds.

Officialrunnerartifact37007650774-1 identifiesRTX4090,driver580.178.04,
CUDA12.8.93,supportedcarrierdriverfamily. Outercommandexit0/wall0.614531435s
but bridgeworkerreturnstatus/rawstderr absent; cannotinferkernel ran or faulted.
WorkflowUploaddiagnostics explicitly retainsbenchmark-results plusmetrics.json,
NOT workerstderr. The evidence_dir points to runnerlocalGPUhealthstorage, not
anartifact we can access. Therefore rawstartupcause remainsblocked by missing
runnerdiagnostics, not a failedlocalverifier. Do notassertOOMestablished.

## Consultation orbit-table proposal rejected by actual implementation

Consultation proposed endomorphism-orbit folding under assumptiondense2D
coefficient-squaretable with recordfingerprintlookup/checkercollisionring.
ActualGTlayout is six/eight disjoint one-dimensional signed odd-multiple
segments: GT_CHUNKS6/8, gt_entries/gt_offset/gt_shift, indicesrecodeGLVdigits;
recordstoresfullXY64B; no hashlookup collisionring. GLVendomorphism applied
in chain after indexedgather. Tabledoesnotstore redundantdense(a,b)square.
Thus suggestedfactor2orbitfolding formula doesnotapplyto thisimplementation;
rejectbeforedeviceedits. This concretecodeboundary saves speculativework.

## Evidence pending at creation

Existingbenchmark production vs compactpitch screenPASS:374.850759 versus
373.512368M/s(+0.358326%); speednotproof. Currentpromotedleader3pairgate
running throughbenchmark.sh subset/N24/120s/flock, immutablecachedbinaries;
firstpairALLPASS345.648608->364.097521(+5.34%). Needall3beforeadoption.
Leaderpublictree2f57d80 exact; localGLV12geometryoverrides explicitlyaudited,
leader256CTA/launch262144 preserved. Sourceandbinary/stamps inrunmanifest.
