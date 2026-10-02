# Iteration 6 source review before existing benchmark

Qualified bat: 414fe58f-5131-4596-97f8-8ccf4fbdcf7f at806b050, queued.
No poll-wait. Native image on that submission unchanged from qualified mode2.

SUM_BASIS=0 and FIRST_PACK8=0 remain production defaults. Their carrier string
expansions are empty, preserving the qualified default fingerprint. Enabled
switches emit their own ABI tag; mismatched images cannot load as the carrier.

Sum basis traces actual pre3/post3: payload N1=U-Y,Nsum=2U,ZZ where U=yR*ZZZ.
The YNEG branch sees -Y and therefore still computes N1 by addition. Opposite-W
relocation only scales ZZ words8..11, so numerator semantics are untouched there.
Tail h=scaledZZ*leaf; a=N1*h and sum=Nsum*h; x1=sum*(a-c). Parity0 consumes a
before its storage is overwritten by b=sum-a. This reconstruction happens BEFORE
center_offset destroys sum. x2=(sum-c)^2-c^2-p1; parity1 then consumes b. The two
field multiplies, one square and publication exact checks are unchanged. Field
macros already permit aliased inputs/outputs throughout the existing tail.
Compile-time guard rejects non-negfold, non-mode2, split/weave and PRE3_ROOT
configurations rather than silently assigning the old n2 meaning to Nsum.
No events, allocations, cleanup paths or descriptor counts change; zero and
field-boundary values obey the same modular identities. Approximate filter
representation is NOT established by this algebra: existing benchmark exact
hit checks remain required, with a frozen control and fixed seed.

Packing experiment: slots8 at128-window shape, slots64 unchanged at256-window
shape. qsb_prepare_window_schedule rejects first_distinct>slots before writing
transposed rows; producer writes and all device/host first-state addressing
use QSB_FIRST_SLOTS. Host checker derives its pitch from that macro, compares
all actual class rows, and keeps the existing reuse barrier. At launch2097152
packing8 keeps the main first-state allocation equal to launch1048576/slots16.
Other launch-scaled allocations still grow; local fit is not ranked-fit proof.

Native audit tooling required explicit nvdisasm PATH; first native audit exited
at cuobjdump, not device compilation or correctness. Fixed tool route1574;
production native header has not been overwritten. Existing benchmark control
alreadyPASS5872hits405.984212M/s; probe result pending, not a speed claim.

Previous ranked diagnostics were inspected (workflow36989482998). PreflightPASS,
worker command exit0 after0.6057s, zero candidates and hits; no scorer artifact.
Downloaded diagnostics do not retain raw grinder stdout/stderr, so this is NOT
proof of a specific OOM location. The known old group-memory overcommit is a
plausible cause addressed by current caps/alias, but exact failure remains unknown.
