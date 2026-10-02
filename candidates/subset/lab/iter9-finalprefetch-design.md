# Iteration 9: final-record address-only lookahead (before benchmark)

Effort: xhigh

New leader: 4cc9d2d8, official 736585478, public source 2f57d80. Entry, full
live device tree and GPUHash read from fetched public git object. The source
has current-record LTC64B demand hints, not destination-less future-record
lookahead. Older fb6f5a8f public note and tree were also read. Previous centered
square fusion and root parking directions are not being repeated.

Default-off QSB_S3_FINAL_PREFETCH=1 peeks P's final descriptor immediately after
the existing penultimate gather, before its deferred addition. The next final
record is decoded with the same mask, centre and offset as the original peeled
load; walker is const and never shifted. It requests each 32-byte half of the
64-byte read-only record via prefetch.global.L2, retaining no coordinates. The
actual final decoder/load, signs, arithmetic, launch, tree and publication stay
unchanged. The penultimate condition uses the descriptor byte offset AFTER any
Q/P transition; both mixed schedules reach descriptor14 before descriptor15.
The prefetch cannot cross the Q/P swap because it is final-only. No table writes
occur during digest. A cache hint has no logical output and cannot change the
hit set. Potential costs: uniform branch, duplicate decode, line pollution,
register scheduling. Successful verification does not itself prove speed.

Guard requires rolled mixed ZDEC, NM masks, DOFF, no ping-pong. Fingerprint emits
an enabled-only key; disabled image and string should remain byte-identical.
Frozen local+native builds precede the existing N24 120s paired benchmark
(seed1789110211) using lab/iter6_ab.sh with full GPU serialization. No new
verifier, fixture, alternate N or mathematical test is needed for a pure cache
hint. Native disassembly must show actual CCTL requests before the penultimate
add. Production remains untouched unless 3 alternating pairs clear >=4% versus
the NEW promoted source control and pass the existing exact verifier.

Local-geometry audit caught that QSB_LOCAL_SM86 selects nonmixed/non-ZDEC
GLV12: the native-only guard would not test a local change. Added parallel
final-only GLV12 path: copy the walker, use ORIGINAL final decoder and descriptor,
and mask out the sign bit. The copy has no external side effects; unused shifts
should disappear under compilation. Native mixed form remains the exact original
ZDEC immediate peek. Guard only excludes ping-pong and incompatible mixed cases.
This dev-vs-ranked difference is a caveat: native performance cannot be inferred
solely from local GLV12. Both variants retain ordinary demand loads.

First build failed before GPU: prepending the raw CUDA directory replaced the
workspace nvcc wrapper and exposed GCC16 incompatibility. Corrected build script
to absolute workspace nvcc wrapper (its GCC-compatible compiler setup). This is
a build-tool precondition failure, not a benchmark or verifier result.

New promoted local control is being prepared from the complete public commit
2f57d80, immutable archive and source manifest, not a nickname or old carrier.
It preserves its own emitter defaults, with only the established GLV12 local
geometry override. This control must be used for any new >=4% submission gate;
current-vs-probe screens are incremental hypothesis tests only.

## Actual existing-benchmark evidence

Screen completed 1595: A410.106447 vs B395.756486 M/s, -3.499082%, A5925/5925
and B5726/5726 verified. Reject adoption; one adverse screen is not a confirmed
regression. No three-pair qualification expenditure warranted. Native has
2 actual predicated CCTL.E.PF2 instructions covering both sectors, retains
128 registers/24576 shared/0 stack/0 spills. Qualified OFF image rebuilt exact
SHA2569c9aab2abc1a6c7368627ae239a40a8e0f4f917df08de43930ee90ee3d48f5ad.

Important falsification: volatile PTX prefetch did NOT guarantee SASS placement
before the preceding point addition. ptxas interleaves decode and CCTL into that
addition's arithmetic; native source location is not a load-issue guarantee.
Disassembly shows CCTL near32e10 while arithmetic is still active, hence some
lead time but less than source-level intended. Correctness is unaffected; no
speed gain established. Any retry needs a distinct, enforceable earlier
scheduling mechanism, not another unchanged prefetch screen.

Census uses an anchored instruction-line regex including predication. It reports
OFF14520/ON14552 total, OFF14500/ON14535 non-NOP, default20vsprobe17NOP. These
are one same-tool code-line census; previous notebooks' differently-filtered NOP
counts are not mixed into this comparison. Byte identity is strongest OFF proof.
