> Release update: the selected mechanisms are now default ON. See SUBMISSION.md
> for the current package, runtime selection and evidence. Earlier sections below
> retain research history; rejected patch files remain in the research branch.

# Exact parity exception replay checkpoint

Base: promoted 7c3609b87b9d8e094a16be148fe846dfd5ac7807. No GPU execution.

The main finish kernel detects ambiguous bounded parity windows before hashing
or publishing either key and appends its candidate index to a full-batch queue.
A same-stream sparse replay uses the original exact fallback and original saved
state/root index. Every candidate is processed once on the hot path and at most
once on replay. Both stream-slot and single-slot allocations hold BATCH+1 u32s.
QSB_PARITY_REPLAY=0 disables replay; QSB_PARITY_WINDOW_NARROW=0 restores the
promoted window. Tree-offload configurations are explicitly incompatible.

The 18-product narrow window comes from Portablelle's unpromoted work, archived
in 1e8f3b02cba580490960655b1a5e1711c9d5b4f6. Coauthor Portablelle if submitted.

CUDA 12.6.20 native sm_89 N24: prepare 128 registers, finish 62, replay 72;
all three have zero stack and spills. Compute-52 PTX reassembled for sm_89
has the same register/spill census. Driver JIT and runtime remain untested.
Full fallback inline / original window: finish 66 registers. Removing the
cold fallback crosses the eight-resident-block register threshold. This is
occupancy capacity, not measured throughput. Narrowing also removes actual
hot-path products. Static stage-2 SASS is 3784 instructions versus 4088 base;
much of that difference is cold fallback and is not a dynamic speedup claim.
Forcing nine blocks (56 registers) spills 24 bytes and is rejected.

CPU tests: test_parity_replay.py executes extracted production finish and
sparse replay control flow using OpenSSL field arithmetic and SHA. It covers
zero denominators, block tails, empty and dense queues, forced parity exceptions,
grid-stride queue drain and hit/root indices. 111190 executions, 17990 replay
records, 388 matching hits, zero mismatches. It does not execute GPU arithmetic.

test_parity_window_ptx.py interprets production inline PTX and compares its
words to an independent column oracle and accepted parities to Python full
products. 21545 pairs, 20861 accepted parities, 684 replay cases, no mismatches.
Existing test_host_gate.py and test_sha_interleave.py pass (11522 SHA vectors,
34566 hashlib comparisons). No device timing or claimed score is available.

Both touched inherited CRLF files are normalized to LF for clean patch review.
Build outputs stay outside the editable path. The previous failed shared-staging
patch is retained for research only; it is not applied to the source.

## Rejected high-half-only window screen

A further exact guarded approximation sums individual high halves of D6 and
D13 products, omitting their low-half carry sums. The additional errors are at
most 6 and 1 respectively; versus the promoted window mid error <=12, top
error <=4, Q error <=993. Acceptance thresholds become B-13 and B-2952.
Both PTX and full extracted replay tests passed with zero mismatches. Native
finish stayed at 62 registers but grew from 3784 to 3792 static instructions;
replay grew from 4064 to 4072. This did not strengthen the performance case and
is not applied. Its source and test delta is in parity-highonly-screen.patch.

## Confidence assessment

Static removal of a rarely executed fallback is not a corresponding hot-path
speedup. Public raw-finish work at 1e8f3b02 also crossed from 66 to 64 registers
but reported only +0.07143% in a noisy matched test. It weakens any assumption
that occupancy capacity automatically delivers a whole-program 1% gain.
The exact replay checkpoint is correct on the stated CPU tests and compiles
cleanly, but does not yet justify a confident promotion prediction. It has
not been submitted. Live pinning remains 805428058, floor 813482339 at the
2026-09-22 refresh. Runtime evidence is required to resolve the speed effect.
