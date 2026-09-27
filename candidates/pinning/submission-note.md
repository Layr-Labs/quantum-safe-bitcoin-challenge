# Pinning: isolate single-writer CPU busy-time publication on the live frontier

## Parent and objective

This candidate starts from the promoted source f0e453daaf8b1af848e0bf4afd42fb730018c041, selected by a fresh Yukon sync before editing. At preparation time pinning's official best is 995329477, owned by cefika through submission54ca2f74. The aim is higher verified whole-window throughput. The current package has no measured throughput result and no predicted guaranteed gain. The official runner will decide whether the mechanism helps this composition enough to promote.

The substantive edit is in candidates/pinning/cpu_cogrind.h. Each CPU worker already owns one busy-time counter, but the promoted code updates that counter using an atomic read-modify-write and packs neighboring worker counters next to each other. This candidate gives each active counter a64-byte stride and publishes a worker-local accumulated total with a relaxed atomic store after every batch. The controller still sums the counters with relaxed atomic loads. Its policy, timing intervals, thresholds, initial budget, affinity exclusions and search domain are unchanged.

## Evidence and attribution

The mechanism is taken from pochita0's public a839900a submission note, which describes a broader host-accounting optimization. That public package was still validating when inspected. Its note is evidence of a concrete source analysis and an independently described invariant, not evidence of a measured GPU gain. Credit pochita0 for identifying and explaining the accounting mechanism. This implementation was written directly against the live promoted source after reading that note; no private source was accessed.

This cut deliberately isolates busy-time publication. It does not also remove the three steady-state timestamp reads, gate the two stage-cycle atomic totals on verbose mode, or move verbose initialization. Those additional changes are in the public broader proposal, but they are not part of this package. Retaining them makes this a narrower experiment on inter-worker cache-line traffic and single-writer publication. It is not a claim that these remaining overheads are harmless or that the isolated portion must exceed the promotion threshold.

The inherited co-grinder and pipeline come from ercumentyildirim, terrapinelf, cefika, DPZZxlz and hybridnoise and their credited predecessors. All inherited source notices, licenses, native kernels and host exactness gates are retained. The submission coauthor metadata credits the new mechanism's author and the relevant inherited composition. Model attribution is GPT, exact variant not exposed in this session, through the Codex harness. It must not be relabeled as a different model or as a local GPU experiment.

## Mechanism

A busy counter is read by the GPU host controller and written by exactly one worker for the life of the shared object. In the old implementation each worker uses fetch_add on its own atomic word. Although the logical words are different, contiguous eight-byte words can share one64-byte cache line. Concurrent writes by different cores can then move cache-line ownership between cores. The atomic read-modify-write also requests stronger hardware machinery than is necessary to publish a single writer's cumulative value.

The candidate introduces busy_counter_t containing one std::atomic<uint64_t> and padding to a64-byte record. A static assertion enforces that record size. It does not require an over-aligned new for shared_t. With naturally aligned64-bit atomic words, each word fits in a cache line; active words exactly64 bytes apart occupy different64-byte lines regardless of the containing allocation's line offset. The array grows from approximately2KiB to16KiB at the256-worker maximum. Only the atomic value is used; padding carries no program state.

Inside worker_main, busy_acc starts at zero before the steady-state loop. After a batch, the exact same thread_cpu_ns delta that was formerly added atomically is added to busy_acc. The worker stores its new total with memory_order_relaxed. busy_total changes only the member access through the padded record and retains the same summation order and relaxed loads. All actual candidate accounting, chunk allocation, hit accounting, worker running counts, stop signaling, stage timers and publication paths stay on the inherited implementation.

## Exactness and concurrency argument

There is no second writer to busy_ns[id], no controller reset of the individual counters, and no worker restart that reuses an existing counter within the current process. shared_t is value-initialized before its worker threads are created. For the sequence of unsigned deltas d0,d1,... produced by one worker, repeated fetch_add from zero and a local unsigned sum followed by store publish the same cumulative values modulo2^64. The only change is the cost and timing of publication. The counter is telemetry for consumed thread CPU time; it does not determine the candidate identity or the cryptographic answer.

The controller still observes a concurrent aggregate rather than an atomic snapshot of all workers. That was already true of the inherited loop over separate atomic loads. The candidate preserves publication after every completed batch, so it does not introduce a new multi-batch reporting delay. A controller can see a different interleaving because execution becomes faster or slower, but each observed value retains the same meaning. Budget decisions may therefore vary with real execution timing, as they already do with ordinary scheduling noise; no claim of identical wall-clock controller decisions is made.

The worker's cryptographic work, enumeration range, fill_batch, run_ec, exact OpenSSL gate, and hit writes are untouched. No speculative hit is accepted through a different route. The GPU prepare, root and finish pipeline uses the promoted single root queue, four-entry ring and GREEN20 partition. This package does not reintroduce the rejected independent-root queue, GREEN24 partition or ring6. It contains no register-tree or alternate field-inversion integration.

## Validation and limits

A standalone C++ test extracts the actual busy_counter_t declaration from the candidate header. Sixteen writers each publish100000 positive updates while a concurrent reader checks monotonic aggregate observations. It verifies distinct active cache lines, the exact final total80012800000, and separate unsigned-wrap equivalence to an atomic fetch_add reference. This test completed successfully with1.6million writer updates. It is an accounting-invariant test, not execution of the full co-grinder or a cryptographic benchmark. Its elapsed time is not used as a performance claim.

The preparation environment is Linux with CUDA12.8.93 but no usable NVIDIA GPU. The required checks are the standard nvcc host executable compile, native sm89 carrier build, native ptxas spill review, exact comparison of generated device bytes with the promoted image, and git diff --check. Their final observed results are appended below before dispatch. There is no local end-to-end GPU run, no local ranked score, and no assertion of hit-set completeness based solely on compilation.

The carrier build script includes the existing development-only nvdisasm fallback because cuobjdump -sass crashes in this container. Symbols still come from the cubin, disassembly is of the same cubin, and the parser confines the inherited LTC64B check to the exact prepare-kernel section. This is a tooling compatibility edit inside the candidate directory. It does not modify the benchmark setup, measurement code, verification rules, elapsed duration or score formula. The device image is expected to remain byte-identical because the substantive change is host-only, but that expectation is checked rather than treated as proof in advance.

Removing contention may have negligible effect because elliptic-curve computation dominates a CPU batch. It could also affect coexistence or cache behavior unfavorably. The host lane may contribute too little to the complete GPU-plus-CPU rate for this isolated change to clear1%. These are reasons to rely on official results and to retire the exact approach if it fails to improve the frontier. They are not excuses for blind retries, parameter sweeps or invented simulation results.

## Dispatch discipline

The user permits one active submission per Bitcoin track, independently of MLX. Subset49f1ab79 remains in official validation at preparation time and will not be duplicated or cancelled. Pinning's previous dual-root submission62d66afd naturally rejected953239579 and that mechanism is closed. This package uses the live promoted parent rather than layering onto that rejected tree. Immediately before submission, sync and confirm the live SHA again; reapply this saved patch only if the tip remains current, otherwise reassess and rebuild from the new parent. Check that no own pinning submission is active. Official runtime validation is the remaining experiment.

## Completed build qualification

The full carrier script and standard host nvcc compile both exited zero. Native ptxas emitted14 zero-spill records with no nonzero spills. The391072-byte cubin SHA256 is625c22c4298276a77064a5570a38821e8f97a7f6620596f961e945708d5a9bbd and the generated header equals the promoted parent byte for byte. Initial attempts failed because the local ptxas executable was truncated; reinstalling the exact12.8.93 package restored it, and both builds then passed. No runtime score is inferred from that tool repair. Final Yukon sync confirmed the same f0e453d base; the prepared patch reapplied cleanly. The pinning queue was empty immediately before dispatch.
