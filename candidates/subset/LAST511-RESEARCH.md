# PR511 composition: last511

Port our 59c1c6e filter-only final mixed addition onto odinfree PR511. Only three calls in qsb_filter_chain_trial change. The speculative filter uses its existing raw point formula through the final addition; exceptional or rare carry inputs may lose hits. This is deliberately not a complete point implementation. The exact output checker, complete last-add helper and replay arithmetic remain unchanged. Only independently verified hit throughput counts.

Base evaluated commit: 3f660b709bacee1cddcb34a69d7afd48463a5567. Parent: 3f660b709bacee1cddcb34a69d7afd48463a5567. All inherited authorship, GPL notices and license files are retained. The public trusted harness, input specification and benchmark are unchanged. This source is an experiment pending source-bound device verification and matched default-toolchain throughput measurements.
