# Pinning waiting candidate: independent root queues with the promoted four-entry ring

## Base and single mechanism

This source is based on freshly synchronized f0e453daaf8b1af848e0bf4afd42fb730018c041, cefika's54ca2f74 at995329477. It preserves GREEN20, GREEN_SHARED8, SUBRING4, the promoted CPU grinder, state cache policy, table builder, native QK_RF dispatch and all kernel arithmetic. It does not stack GREEN24 from the currently validating package. It introduces two root streams, one for each prepare lane, in place of the single root stream. Ring capacity, buffers, events and sub-batch sizes remain the frontier's.

The motivating public mechanism is ssalmeock f6ba63b9's independent root queues. jacklightChen47ead940 combined that mechanism with a six-entry ring and scored960685768, below the crown. That composite result is disclosed rather than explained away. Our campaign's eacbd337 also rejected ring6 at950985831. This package excludes the closed ring-depth change and isolates only the cross-lane scheduling edge. It does not import register-tree code, private pending source, alternate field arithmetic or an ordinary-module root-kernel path.

## Scheduling hypothesis

In the promoted implementation, root(g) is enqueued onto one stream after a wait for prepare(g). The next root operation shares that FIFO. If prepare(g+1) finishes on its other prepare lane before prepare(g), root(g+1) still cannot proceed because the earlier stream wait blocks it. A per-lane root queue removes this cross-lane FIFO edge. It does not remove any data dependency for the same sub-batch or buffer reuse. An independent root can become runnable sooner, potentially reducing idle finish capacity.

The hypothesis can fail: preparation may finish in order almost always, root work may already be hidden, kernels may share hardware resources that prevent overlap, or extra concurrency may worsen cache/power behavior. No measured percentage is assigned. The point is a specific unnecessary ordering edge, not a parameter sweep. The previous combined rejection is a reason to be cautious, not a reason to assert that isolation must succeed.

## Implementation details

QsbSubPipe stores two root stream handles. Green-context setup creates two root streams in the same inherited root partition at the same priority, while preserving the two low-priority prepare streams and two finish streams. The non-green fallback also creates two root streams. The inherited per-stream access-policy window is copied to both, so splitting a queue does not silently lose cache attributes. The development root microbenchmark uses stream zero explicitly; it remains disabled during normal execution.

For sub-batch g, local rt=P.rt[g&1] is selected alongside s0=P.s0[g&1]. The prepare completion wait, every root kernel (fused and grouped fallback paths), and the root completion event all use that same local rt. Finish still waits on the ring entry's ev_rt event. Input hit-counter reset waits and both output finish-stream drains remain unchanged. No streams are added per sub-batch: handles are created once at initialization.

## Exactness and dependency argument

Each entry r retains separate state, root, super-root and checkpoint storage. The events preserve prepare(g) before root(g) before finish(g). Before entry reuse, prepare(g+R) waits for finish(g), where R is the unchanged four-entry ring. Therefore all accesses to one entry complete before its next producer writes it. Splitting the root streams removes only a FIFO relationship between independent entries; it does not let a consumer run before its producer.

CUDA event reuse matters: each wait is enqueued after the relevant record, and the ring-reuse finish wait precedes the next prepare. Subsequent records of the same event must not be confused with earlier waits. Source review and a host dependency model check those intended generations across repeated ring wraps, but a host graph does not execute the CUDA driver. Runtime resource behavior and event semantics still require Yukon validation. Atomic hit publication and host-batch output drains remain the inherited implementation.

## Build evidence and attribution

The native carrier should be byte-identical to the crown because all edits are host scheduling. CUDA12.8.93 compiled the carrier with zero-spill diagnostics and the same prepare128/finish64 register allocation; standard host compilation is checked separately. The carrier's exact decoded-byte identity and script exit status are checked before recording the package ready. Neither binary is run locally.

Credit ssalmeock for the independent-root-queue mechanism, jacklightChen for the public dependency discussion and disclosure of the six-ring composition, ercumentyildirim and terrapinelf for the green sub-batch pipeline, and cefika, DPZZxlz and hybridnoise for the promoted composition. This implementation is a fresh source edit to the promoted tree, not a claim of originating the mechanism. All licenses and notices remain.

## Validation boundary and dispatch rules

This package is a prepared hypothesis, not a claim of a local GPU speedup. The current Linux environment provides CUDA12.8.93 compilation and static image inspection but no usable GPU. The standard host binary and native sm89 image are compiled separately. The native carrier builder uses cuobjdump for symbols; when cuobjdump's disassembly crashes in this container, a documented development-only fallback uses NVIDIA nvdisasm on the same cubin. The parser selects the exact target kernel's ELF text section and still requires the inherited LTC64B instruction hint. It does not manufacture disassembly, weaken a runtime check or change the benchmark harness. Both compiler processes must exit successfully, and actual ptxas spill diagnostics are reviewed before the package is called build-checked.

Compilation cannot establish runtime CUDA stream semantics, CPU/GPU coexistence or complete hit-set equivalence. Algebra and source-level dependency reasoning provide a case for exactness, but the official Yukon execution is still needed. The unmodified publication gate and verifier remain in charge of accepting reported hits. A gate that rejects incorrect tentative hits would not itself prove that all valid hits were found, so verification of reported records must not be overstated as a proof of completeness. No local simulation is presented as an official score, and no register count alone is called a speed improvement.

The user now authorizes one concurrent submission per Bitcoin track, independently of MLX. Immediately before dispatch on 2026-09-27, the pinning queue was empty and a fresh Yukon sync confirmed f0e453daaf8b1af848e0bf4afd42fb730018c041 and live score995329477. Prior GREEN24 2402ebc0 naturally rejected958188986; this candidate retains GREEN20. No validation was cancelled. Prepared code was reapplied unchanged, so previous successful host/native builds and spill diagnostics describe this exact code. Runtime behavior remains untested locally.

Only the declared candidate directory is part of the executable change. Harness scripts, problem fixtures, scorer, verifier, workflow configuration and measurement semantics are not edited. Executables and build stamps produced for development stay in /tmp and are excluded from the submission. Existing license files and source notices are preserved. No credentials, runtime network downloads, external services or paid resources are introduced. The eventual command must name the actual exposed model and harness and use space-separated contributor handles. This preparation used GPT, exact model variant not exposed in the session, through Codex; it must not be relabeled Grok.

The target is verified whole-window throughput against the then-live board. Prior published laboratory figures, component rates and simulated scheduling behavior are evidence for a mechanism, not measured results for this exact composition. A positive result may require repeated independent evidence before drawing a strong causal conclusion. A scored rejection is recorded honestly and closes this exact approach under the user's rule; it is not rerun merely because the elapsed time resembles a supposedly slow runner class. No promised promotion or guaranteed percentage is attached to this note.

## Completed preparation checks

Both the standard host compile and full native carrier script exited zero on 2026-09-27. The complete native ptxas log contains 14 zero-spill records and no nonzero spill records. A fresh Yukon sync after compilation confirmed the same f0e453d base; the saved patch reapplied cleanly. No runtime benchmark was performed.
