# Iteration 9 next candidate: scoped public rotate-add SHA port

While the final-prefetch benchmark runs, prepare a materially different arithmetic
probe based on the new promoted 2f57d80 source. Borrowed GPL code and upstream
notices/attribution are retained. Rotate-add originally credits ercumentyildirim
public pinning b62c41b8; split form credits skeptic39. Port only GPUHash.h's round
macros and SHA256_RND outer calls, NOT the leader's speculative field arithmetic,
runtime FMA switch, gate special rounds, co-grinder or host changes.

Default-off QSB_SHA_LEA=1, LEA_ORD536, LEA_PARTS0 (no specialized gate integration).
The same original S1/S0 sums are factored: S1=ROR6(e^ROR5(e)^ROR19(e)),
S0=ROR2(a^ROR11(a)^ROR20(a)). XOR commutes with rotation. Put Maj into T for
ALU scheduling, then subtract it for d'. All operations modulo2^32. Assembly
preserves add/sub from NVVM cancellation, allowing LEA.HI rotate-add. Changes
paired window, constant SHA and generic outer SHA only; gate macros stay old.

No scalar/field/probability/enumeration change. Native first kernel can also change
because shared round macro runs there; all four search-loop native functions must
be inspected. Default0 empty token should reproduce qualified entire CUBIN.
Probe1 enabled fingerprint disallows old carrier; compile frozen local+native,
then existing N24 120s pair versus current, GPUlock. If positive, 3pairs versus
new promoted local source are required. Prefetch stays0 in this probe, isolating
arithmetic from cache scheduling. No performance inference from instruction count.

Build passed with matched workspace CUDA12.8 wrapper. Exact OFF native image
matches9c9aab2...; scoped rotate-add ON image82b9868e... retains128reg24KiB0spills.
The new leader control compiled from archived source, ELFfc6ec201... frozen.
Existing benchmark gate versus current and refreshed-promoted gate use separate
jobs but every arm serializes the same lock. Their chronological sequence must
be disclosed; interleaving extra pairs can introduce thermal drift.

## Actual unchanged-verifier screen

1598 completed exactPASS both: current A392.855587M/s5698hits vs scoped B398.051871
M/s5790hits, +1.322696%. Native digest14520->14256(-264), LEA.HI7->548(+541),
SHF.R.U643071->1451(-1620),128reg24KiB0spills. All four native search kernels
change, first flat1464->1400, epochs1896->1832, groups3464->3344. The producer
benefit is NOT isolated from digest performance; claim total pipeline only.
Still a singlepair lead, not established improvement.1602direct3pair gate versus
newleaderGLV12 queued/running;1601unchangedproduction refreshgate separate.

While qualificationruns, prepare selected scopedsource+carrier in a separate
stagingtree; measuredsource snapshot is reused, only header+explicitSHA1wrapper
changes. build_carrier.sh must reproduce82b9868e...entireimage beforestagingcan
beeligible. Do not overwriteproduction orsubmitpendinggate. The ordinarygate
extension remainsseparate untilits incrementalexactscreenisfinished.

## Direct current-promoted gate complete — reject adoption

1602all6PASS: refreshedleaderA[375.853448,362.365177,374.776564],
scopedSHA B[396.561924,363.637879,361.146810]M/s.
MeanA370.998396,B373.782204,+0.750356%. OrdersAB/BA/AB. HitsA[5426,5230,5395],
B[5760,5278,5239]allverified. Belowstanding4%bar,notadopted. Finalcandidate
stagingand1451-hit30scheckalsoPASS,butintegrationdoesnotrepairfailedspeedgate.
Preparedcarrierremainsseparate;productionnative9c9aab2...unchangedallprobesOFF.

Unchangedproduction1601refreshednewleaderqualificationall6PASS:
A[351.738638,376.553570,367.159060]mean365.150423,
B[381.064000,390.466372,394.740052]mean388.756808M/s,+6.464839%.
HitsA[5063,5421,5296],B[5508,5656,5706]. Thisisalreadywatcherownedbestbat414fe58f,
notanewcandidateornewsubmissionID. Don'tsubmitless-qualifiedSHAprobejustbecause
itspublicmacroisfewerinstructionsoritsfirstscreenwaspositive.

Jobs1601/1602andextension1604alternatedseriallock;source/binary/stampfixed.
Observed broadthermal/yielddriftmeansnotaconfirmedscopedregressionvscurrent.
Concreteconclusiononly: scopedemitterfailsnewleaderqualification,noadoption.

## Actual enabled PTX operand review (not another verifier)

A second-opinion review raised a conditional NVCC operand-alias concern: RLA's
newhoutput appears beforeMaj and oldd's lastreads. Host-GCCearlyclobber rules
alone do not establish this CUDA lowering is wrong. Actual matching CUDA12.8
emitted PTX at sm89native and localcompute52 has384ORD16RLAblocks each; EVERY
newhoutput is a distinct virtual register from still-liveMaj and oldd. Zero
substitution hazards. AuditJSON records samples and exact PTXSHA256. No source,
fixture,expected-output,verifier orcounter change needed. This clears that exact
frontend-binding concern, not an all-domain compilercorrectness proof. It does
NOT reverse the scopedcandidate's failedperformancequalification.
