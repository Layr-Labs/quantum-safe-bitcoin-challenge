# Subset scalar producer and target-device comparison

Prepared with GPT 6 Astra, xhigh effort, in Codex, 2026-09-22.
The source baseline is shared main 7c3609b. No local NVIDIA GPU is available.

The promoted kernel computes two epochs' remaining SHA work and both curve
chains inside a 256-thread block constrained to 128 registers and 48 KiB shared
memory. This experiment moves the existing paired SHA calculation into a
separate producer. The paired producer compiles to 64 registers. A second
producer assigns one epoch/window hash per 128-thread-block lane and compiles
to 39 registers with launch_bounds(128,12). Both have no shared memory or spills. The curve consumer retains 128 registers and 48 KiB, with no spills.
A launch_bounds(128,8) screening build used 47 registers at the same static
instruction count; the 12-block bound reduces register allocation to 39.
This changes resource pressure during SHA; it does not remove SHA rounds or
change the search space. No architecture count is a measured speedup.

The producer stores eight planes of 256 64-bit words per consumer block.
Planes 0..3 contain A's scalar, 4..7 contain B's. Address word index is
`(block*8 + word)*256 + lane`. Adjacent threads use adjacent addresses.
The default 262,144 blocks require 4,294,967,296 bytes. There are 32 bytes
written and subsequently read per candidate. Scalar stores and loads use
PTX .cs, confirmed as STG.E.EF and LDG.E.EF in native and reassembled SASS.
This hints that transient scalars should be evicted before reused curve-table
data; it cannot eliminate all cache interference. Allocation failure retains the
fused path. The current sequential host loop and its exact verification stay
in place; the rejected host-pipeline composition is not reapplied.

The first implementation spilled 20 bytes of stores and 16 bytes of loads
in the curve kernel. Loading B later did not by itself solve that. Inlining
the whole front also failed. The successful change reloads the inexpensive
thread/block identity after the long field calls and reconstructs nomination
identity only on the cold hit path. Volatile special-register reads prevent
hoisting; no execution identifiers, enumeration ranges or outputs change.
The candidate never copies B through shared memory: its immutable scalar
remains in the producer buffer until needed. The original fused route retains
its existing shared parking.

The runtime reference uses the current promoted point seed, point add, final
resolve and chain bodies, copied byte-for-byte apart from function-name
suffixes into baseline_filter_control.cuh. A source test checks those bodies
against 7c3609b. Fused reference and produced candidate call the same existing
finish and exact hit-verification paths. The candidate uses the separately
audited negative-Y MAC; the reference uses promoted positive deferred Y.

The selector runs two separate candidate/reference trials: paired versus fused,
then single versus fused. Each trial warms two batches per route, then runs
8 four-batch cohorts F,S,S,F,S,F,F,S. All 72 batches search new candidates;
completed hits are verified and written normally. CLOCK_MONOTONIC covers
production, consumption, exact verification, blocking copies and host output.
Actual candidate counts normalize partial batches. Each trial forms four
adjacent candidate/reference ratios. Eligibility requires geometric mean
>=1.025, at least three ratios >1.02, and no ratio <0.995. Invalid or
inconsistent measurements are ineligible. Among eligible producers, choose
the larger aggregate gain. Ties retain the paired producer. Neither passing
selects the promoted fused implementation. There is no upper cap on gain.
The decision is fixed after these trials; cohort timing stops. This is a
runtime safeguard, not an offline score or guarantee of promotion.

Checks completed before packaging:

* 107,520 independent OpenSSL comparisons for the extracted paired SHA,
  actual producer stores and actual consumer loads, across 128- and 256-window
  geometries. Empty, odd, block-edge and multi-block tails are covered.
  Single and paired producer buffers are identical in all 28 batch shapes.
  Guard words remain intact and active identities cover each epoch/lane once.
* Per producer-disable setting: 11 synthetic timing scenarios and 7 route
  selection cases; 37/73 end-of-search boundaries for one/two trials. Invalid
  timing fallbacks, identical dispatch arguments, and source checks place
  the decision after verification and all hit writes.
* 2,048 extracted negative/positive source chains agree, with 1,024 independent
  OpenSSL curve sums. The raw PTX and curve checks in RESEARCH-NEG-Y.md also pass.
* Native sm_89 and compute_52 PTX reassembled to sm_89 default kernels:
  fused reference 128 registers, produced curve 128, paired producer 64,
  single producer 39; all four have zero stack and zero spills.

Remaining validation must use the actual ranked run. Compilation, CPU tests
and synthetic policy tests cannot establish the measured candidate rate.

Primary documentation: [Ada resource limits](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#occupancy)
and [PTX cache operators](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html#cache-operators).
The 39-register producer fits the documented 48-warp ceiling by the static
register/block/shared-memory limits; actual occupancy and throughput require
a device run. Streaming cache hints preserve memory semantics.
