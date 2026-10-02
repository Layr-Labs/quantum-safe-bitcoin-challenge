# Iteration 5: centered-square recovery finish (default OFF)

Distinct from denominator placement: post3 replaces p2=S*(m2-c) with
p2=(S-c)^2-c^2-p1, p1=S*(m1-c), S=m1+m2. Arbitrary-field identity,
not an additional curve assumption. Reuses sum scratch; no shared-memory or
front ABI changes. Constant C extends only under the enabled knob, and the
carrier fingerprint includes that knob. Host computes canonical c^2 from
original recovery R (not ISO_R). Exact publication replay is unchanged.
Nominal 5M -> 4M+1S per candidate plus two subtractions; no speed claim.
Speculative short-carry reassociation does not prove identical tentative yield.
Require independent exact OpenSSL post3 audit, resources, and scored pairs.
Prior repeated-submit setback addressed: watcher-owned 93cf9212 still validating
in a one-shot board refresh. No repeat submission until slot release notify.

Initial exact post3 audit failed (3958 comparisons): diagnostic mistakenly
passed ordinary Y to a pre3 configured for the production negated-Y front.
Fix diagnostic QSB_YNEG_FOLD=0; do not call this a tested algebra failure yet.
Native build failed after compilation because cuobjdump was absent from PATH;
re-run with explicit existing CUDA12.8 cuobjdump path, no tool timeout.
Diagnostic Y correction alone was insufficient: same 3958 comparisons failed.
Found second inherited audit incompatibility: prepare defaults to ISO_FAST_X=1,
but this audit supplies ordinary OpenSSL R and ordinary projective coordinates.
Set ISO_FAST_X=0 explicitly. Record both failed diagnostics, not score claims.

Exact non-ISO/non-negated-Y GPU/OpenSSL audit PASS: 4106 cases, 3926 usable,
180 unusable, 7852 keys, 624 sentinel guard bytes, zero errors. Native sm89
center finish compiles 128 registers, 24576 shared bytes, zero stack/spills;
sm86 dev compiler likewise 127 regs versus qualified control128. Arithmetic
saving is not fitness until scored verifier results complete. First screen1533
runs inherited unchanged benchmark and verifier with fixed seed1789110211,N24,
120s arms. Root best source and carrier remain unchanged; probe is default OFF.

Resource correction: native sm89 uses128 regs (sm86 uses127), not127 as
first narration said. Native shared24576, zero stack/spills still hold.
Local screen manifest corrected: native prep added knob to isolated subset.cu
AFTER local binaries compiled, so manifest initially showed wrong A source.
Restore original subset.cu and retain correction plus exact build input text.
No executable/stamp changes; B knob is in its wrapper, A remains OFF. All further
local gates use frozen dependency copies, no postbuild native edits to that tree.

Early centered-square screen1533 completed both exact PASS: qualified control
407.413441M/s, center398.483444M/s, -2.191876%. NOT adopted. Native static
instructions rise8 (14512->14520) despite IMAD -27; IADD3 +21, SHF+14. Saving
partial products isn't instruction-count reduction. New scheduling mode2 moves
center square/subtractions AFTER parity0, eliminating premature x2 liveness.
Mode1 retained for provenance; mode2 needs its own exact audit and verified
screen. Immutable previous screen tree is not changed.

Late mode2 screen1535 completed both PASS:406.619419 control vs397.085372M/s,
-2.344710%. Both independent probes screen negative but always A first; known
thermal/order drift means neither is a conclusive family rejection. Schedule
remaining mode2pairs (BA then AB) instead of retrying identicalAB, preserving
firstpair artifact. This replication resolves the specific biased-screen issue,
not blind repeated costly action. Local controls/binaries/dependencies frozen.
Mode2 native resource same128regs24576B0spills; static opcounts exactly equal
mode1 (code schedules differ). Both shave IMAD27 yet add total8instructions.

All local scores above use N24 verified-hit score, NOT self-reported count.
Harness metadata labels GPU RTX_4090 but these are sm86 wrappers on local3090.
Both initial probes warn self-report falls outside expected Poisson band; this
is inherited count/yield behavior, not a new verifier error. Candidate fitness
is exact-verified hits/time. No official or ranked throughput claimed.
While mode2 gate1540 continues, launch8 probe1543 can acquire the same serial
GPU lock between arms. This changes thermal history; record global ordering
and treat close results as noisy rather than silently claiming strict six-run
adjacency. Both source/binary arms unchanged and AB/BA/AB order retained.

Mode2 confirmation secondpair: B404.955090, A369.060500M/s bothPASS. Huge
+9.725% reversal from initial-2.345% demonstrates why AB-only screens do not
settle the direction. Control reported GPU+CPU raw353.6M/s vs B355.4 (similar),
but verified-hitcount5378vs5859. Interpret hit-yield sampling/host variability,
not nine-percent kernel acceleration. Gate still incomplete and no adoption.

Complete mode2 confirmation all6PASS: A381.891765mean, B392.558927mean,
+2.793242%. A[406.619419,369.060500,369.995377];
B[397.085372,404.955090,375.636318]. Pair3+1.5246%; pair2+9.725%, pair1-2.345%.
Promisingstacknotyetqualified; directexplicitpromoted3pairgate1549launches
center2+hosttightcap1 vscontrolcenter0/tightcap0/denreloc0/256CTA262144launch.
Native stack alreadymode2resource128regs24KiB0spills; groupcap host-only.
