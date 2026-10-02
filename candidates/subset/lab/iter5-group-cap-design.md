# Iteration5: launch-scaled group cap (host only, default OFF)

Production1048576blocks consumes2097152epochs. Existing GROUP_CAP_EXACT only
handles1048576epochs, so production allocates2*2097152+4groups per stream.
Ranked table22688113472bytes + first states2147483648bytes + group buffers
1073742848bytes =25909339968bytes already exceeds24GiB25769803776bytes,
BEFORE descriptor/index/hit/context memory. Qualified local sm86 gate does not
prove ranked VRAM fit. Tight guarded bound resolves hidden fallback for launch4.

QSB_GROUP_CAP_TIGHT=1 uses exact maxima of rank(first5(last))-rank(first5(first))+1
for ALL aligned capacity launches over C(137,6)=8218472724epochs. Maxgroups:
1048576:181498;2097152:362053;4194304:594292. Finalpartialincluded; epochsstart0,
incrementcapacities as unchanged hostloop. Arbitraryshapes fallback unchanged.
Current launch4 groupbytes92685568vs1073742848 saves981057280bytes=935.61MiB.
Runtime perlaunchgroup span guard remains; no device changes, fingerprints or
carrier image changes. Hostproducer uses independent qhp Slot bounded chunks;
inspect before adoption. Native table + launch8 firststates alone26983080768>
24GiB: doubling launch is not rankable even with group-cap fix. Local launch8
screen is only diagnostic, never submit based on smalltable results alone.

Independent integer audit PASS across13717aligned launches, including rank /
unrank roundtrip on every endpoint. Conservative/finalspanchecks allcorrect.
Explicithost-only knob now in targetsource; scoredfrozenpairedpreflight next.

Ranked corrected memory minimum includingboth64-byteepochdescriptors and
bothu32epoch-groupindices:25213495360bytes (23.48GiB), leaving556308416bytes
(530.54MiB) to context,hits,ladders,otherbuffers. Exactdevicefitstillrequires
rankedrunner; no claimlocalGLV12provescontext headroom. Existing table
buildtemporarybuffersfreedbeforefirstallocations. Preflight exactprobe1548
runs; independentcompiledproductionhelper audit1549beforepromotedgate.

Scoredhosttightcap preflight bothPASS:385.404145 vs381.427067M/s,+1.042684%.
NOTthroughputqualification; allocationfix verifiedbyindependent+actualcompiled
helper and unchangedbenchmark. Script153?1548ended2AFTERbothPASSbecause live
script editedwhilebashretainedreadfileoffset. SummarizedimmutableJSONartifacts
without costlyGPUrerun; nextscriptsmustexecutefrozencopy,nevereditinflight.
Currentpromotedgate1549 launchedaftereditanditsrunningfilewillnotchange.
