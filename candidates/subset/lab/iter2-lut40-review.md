# Root LUT split into word + flag byte

Compact31 verified but decode costs may offset the saved lookup latency. Test
split40 (matrix word + flag byte). The four matrix coefficients preserve their
original signed byte representation; the signed even delta adjustment uses the
byte's low8 bits, with spare bit0 holding delta-negation. Decode needs the same
PRMT byte operations as the original64 table. 4160B instead of6656B. Inverses
arena can be [4][130] =4160B for a128-leaf production block (alignment rounds the
kernel shared budget). This is still less than25KiB and should allow fourCTAs.

Load both arrays into inverse arena before front. uint32 matrices first3328B,
flag832B next. cp.async's final partial word is safe: total4160 multiple16.
Root reads offset832words + indexbyte. After root's existing barrier, inverse
rows overwrite the table, retaining same tree algebra. CPU matrix variant
keeps original64table. Kill switches defaultOFF, carrier fingerprint only
includes enabled switch. Exact pre-existing tree_audit then benchmark.sh gate.

State/error audit: all832 entry roundtrip assertion; table index0..831; each
thread copies chunks stride blockDim; committed then waited and block-barrier
before root. Decode lowbit as sign, clear lowbit before PRMT signed adj.
Scale/fallback unchanged. Direct-root test kernels issue the same loader.

## Results
Exact tree_auditPASS allsizes,partialcounts,8192rootsandboundedfallback.
Local+native128regs24640Bshared0spills. Firstpair329.513712vs381.581868M/s
**-13.645%**,bothPASS.Only64additional shared bytes versusbaseline correlate
with large loss; real CUDA driver occupancy query queued, not assumed.
Native digest14536instructionsvsbaseline14512andLUT3214576;rootlookups5LDS.U8
+5LDSinstead of5LDC.64,original41PRMTretainedvsLUT3216.
Nextsplit40reductiondrops duplicateouterclamp rows: -6equals-5,+6equals+5.
Use11rows704entries3520B, fitsoriginal4096B arena, preservePRMTdecode.

## CUDA driver occupancy proof

Real sm86 kernels compiled from same frozen source; driver
cuOccupancyMaxActiveBlocksPerMultiprocessor after maximum-shared carveout:
fill2 24576B128regs ->4blocks/SM,split40 24640B128regs ->3blocks/SM.
This is a measured cliff, not a rough bytes/register arithmetic estimate.
Driver report: lab/iter2-occupancy-driver-report.json.

Corrective SHORT switch drops identical -6/-5 and +5/+6 rows. All original
832indicesmaptoidenticalrecordsunderclamp[-5,+5] andindex(dc+5)*64+ratio.
704entries3520B,220copiesvs260; flagsatbyte2816; inversearena4x128x8=4096B.
Copyandreadbarriersunchanged. Expected24KiBoccupancytestedby scoredbenchmark.
NativeSHORT128regs24576Bshared0spills;readyfrozenimagenotinstalled.

Finaltestassertionsarenaboundsaddedandcompiling;SHORTswitchscoredcandidate
currentlyrunningratherthanalreadyqualified.No speedclaimfromresourcebuilds.

## Final SHORT outcome

Scored380.746996vs391.389281M/s **-2.7191%**,bothPASS. Shorteningremoves
theoccupancycliffbutdoesnotimprovethis128CTAfrontier. Root-LUTpacking
(31bitconst/shared,40bitword+flagbyteoriginal/clamp-short) stops paying.
Finalcompiletimearenaguardstestreturnedexistingtree_auditPASSagain,zero
wrongforallblockssparsetails,roots8192exactandboundedfallbackclean.
No production switches enabled, no image replacement, no new submission.
