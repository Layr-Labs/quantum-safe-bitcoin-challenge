# Iter50 consumer-local first-state logical review (screen negative)

Distinct hypothesis: a paired128 CTA consumes two eight-class epoch firststate
rows exactly once. Build16SHAcompressions (one warp with16active lanes) into
512Bof the existing PARK12824KiB sharedarena,then read each state's32Bacross
itswindowclass consumers. All128threads crossa producerpublishbarrier and a
readerretirementbarrier before parking overwrites thatarena. No extra shared.

Invariants: exact existing epoch->mid/remW,14classwords,_SHA256Transform;
writer(member=tid>>3,cls=tid&7) maps128distinct32bitwords. eA=2*CTA index,
partialB aliases descriptorA (inactiveB retains existing masks); overlaunchA
aliases valid descriptor0. Emptybatch never launches. Existing descriptor
rank/window candidate enumeration unchanged. Originalfirstcount==8guard remains.

Host upload only omitsfirststates; descriptor uploads unchanged,slotcopiedevent
stillrecords AFTERallcopies andproducerreuse state unchanged. Host producers
stillcompute full firststates for this isolatedscreen; no pinned/VRAM allocation
saving claimed. GPUproducer fallback stillbuildsdescriptors; firstkernel keeps
full batch0 independent GPUvsCPUchecker,skipslater unusedglobalfirststates.
After checkerfailure hostroute goesGPUdescriptors+localfirst, not legacy digest;
checker poll/reuse safetyunchanged. Alllocalconstruction exactsame GPUtransform.

Optional defaultoff carrierfingerprint fl1 prevents embeddedproductionmix.
Compileguards paired128 packed8 PARK128 slotpipeline hostverifiedv3;
v3guard movedafter itsdefininginclude (initialcompilefailed beforeGPU).
Defaultoffcubins sm89/sm86 byteexact f26ae31; enabled resources spillfree128regs,
24KiBshared; local instructions14576->16000,+1424; barriers3->5; fourLDG128
becomefourLDS128. Source doesnotmodify SHAorfieldarithmetic internals.

Screen preregistered controlthenlocal,120seach,N24seed1789110211,
benchmark.sh existingexactverifier viaiter49-grinder.py no timeoutprocess,
GPUflock. Nonpositive ends direction; positiveonlyqualifies actualleader>=3
alternatingpairs,+4%localgate plus positive productionincrementality,exactPASS.
No qualificationclaim fromresource/copyworkcensus. Boardcurrentfaf5422a/efef868
753571538; latest higher redrawscoresnotpromoted. Finaloutcome follows.

Outcome: exactPASS10935hits,388.675150->365.742002M/s,-5.900338%. Nonpositive preregisteredstop; no redraw/noqualification/nosubmission. Distinct denseprepass next.
