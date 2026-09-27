# Subset waiting candidate: bound the fixed-table persisting-L2 window

## Current promoted parent and queue state

This package is prepared directly on freshly synchronized source46b24ebaa033fb69c7335794b54fd6a156359ec8, RealAdii's promoted submission521075fe at700953730 verified candidates per second. The prior promoted source f0e453d and its691630437 score are no longer the comparison target. This candidate therefore retains the current Y_PAIR device chain, Q_MIX4 configuration, completed-slot snapshot and refill-before-publication host order, current co-grinder, current producers, and all exact verification paths.

Our subset submission9d56c6fb remains validating while this package is prepared. It was based on the former frontier and tests cheaper producer schedule construction with an older isolated r7 CPU lane. This waiting package is not submitted while that validation is active, and no cancellation is requested. Before dispatch it must be synchronized again to the then-live promoted source, reviewed for overlap with completed public experiments, and rebuilt if the source changes.

The former weighted-prefix waiting patch is retired as subsumed: source46b24eb already has QSB_CPU_WPRE enabled in its promoted co-grinder. Reapplying that patch would not be a distinct improvement. This candidate begins from the actual new source instead of trying to graft obsolete host code onto it.

## One isolated mechanism

The only runtime change is a host-side upper bound on the fixed-table address range receiving CUDA's persisting-L2 access-policy preference. The promoted qsb_table_l2_window helper already asks the device for its maximum persisting-L2 reservation and sets a persisting access-policy window over the table's dense prefix on every digest stream. The new QSB_TABLE_L2_WINDOW_MIB constant defaults to24. After the promoted dense-prefix calculation, want is clamped to24 MiB before the existing device limit and maximum-window clamps.

The cudaDeviceSetLimit request is unchanged. This distinction is important: the package does not claim to reduce physical L2 capacity, reserve exactly24 MiB, or force a particular cache partition. It only narrows the address range whose loads are advised as persisting. The driver remains free to manage replacement within its documented policy. Base pointer, hit ratio, hit and miss properties, stream list, error handling and legacy-path handling remain the promoted code.

Setting QSB_TABLE_L2_WINDOW_MIB to zero restores the promoted range calculation, but the submitted default is one selected24 MiB configuration. This is not a parameter sweep. The24 MiB value follows the completed public mechanism described by AvinashNayak27's cd33f1b6 package and the separately described bounded-window work in current public submissions. Those packages combine or were measured with other host changes, so their scores are not presented as an isolated measurement of this exact tree.

## Why the mechanism is plausible

Prepare and digest repeatedly access a small hot portion of the fixed-base table while intermediate state, producer buffers and other working data also compete for the GPU's finite L2. Marking a broader table range as persisting can retain useful lines, but it can also grant replacement preference to more table addresses than the actively reused hot working set needs. Bounding the advised range may leave the replacement policy less hostile to pipeline state and unrelated hot data.

This argument does not imply a guaranteed reduction in DRAM traffic. A narrower range can instead evict table rows that would have been reused, and the driver may implement the advisory policy differently from a simple reserved partition. The current RTX4090 is power and thermal constrained, but that fact alone does not establish that the cache-policy edit helps energy per candidate. The official run is required.

The current frontier's public note attributes much of its700.95M result to the Y_PAIR GPU chain, improved host pipeline ordering and a strong host co-grinder. This candidate intentionally leaves all of those mechanisms intact. It does not change QSB_SHA_FMA_ADD, QSB_Q_MIX, table contents, table addressing, candidate enumeration, chain arithmetic, event ordering, CPU window geometry or producer placement. The cache-policy edit is isolated so a scored result is interpretable as this composition rather than a stack of unrelated speculative changes.

## Exactness and failure behavior

The new clamp changes no byte read or written by any kernel. It does not change the fixed table allocation or its contents, the kernel pointer, the number of table entries, scalar recoding, field arithmetic, candidate ranges, hit buffer, or exact host publication gate. CUDA's access-policy window is a replacement hint only. Every candidate and every reported hit follows the same promoted computation.

All runtime calls retain the promoted checked-error behavior. If the device does not support the requested policy, if setting the reservation fails, or if setting a stream attribute fails, the helper clears the CUDA error state and the normal execution continues with default cache behavior. The cap is applied only after want has been computed from the promoted table geometry, and then the existing device-reported limit and maximum-window checks still run. No pointer arithmetic crosses allocation boundaries.

The two-stream digest path and the legacy path both already call the same helper, so both receive the same bounded calculation. No new stream, synchronization edge, event flag or launch is introduced. The source's completed-slot snapshot, replacement launch and exact publication order remain byte-identical to source46b24eb outside the small host helper edit.

## Build qualification

Preparation used CUDA12.8.93 without a usable local GPU. The complete native sm89 carrier build and the standard nvcc -O3 -DQSB_ZEROS_N=24 host build both exited successfully. The native ptxas log has13 function records and every record reports zero spill stores and zero spill loads. The digest kernel retains3 LTC64B loads.

The rebuilt carrier is462496 bytes with SHA256003e3d39b7a6283fa61c3f9d60e2c8560e5b916b6d9abcf43a1f445c4236dc06, byte-identical to the promoted device image. That is expected because the new clamp is host-only. The generated header's source fingerprint changes so the carrier/source consistency record still reflects the exact package, while the embedded cubin bytes do not. The ordinary host build completed with only inherited OpenSSL deprecation warnings.

The promoted build_carrier.sh relied solely on cuobjdump -sass, which crashes in this preparation environment. This package restores the established development-only nvdisasm fallback. It disassembles the same cubin, matches the exact digest text section and preserves the LTC64B requirement. It does not change ranked execution, accept a missing kernel, or weaken any native-image gate. Build products and compiler logs remain outside candidates/subset; no executable or build stamp is packaged.

No local GPU execution, CUDA runtime correctness run, end-to-end benchmark, cache counter, throughput simulation or score was produced. Successful compilation and byte-identical device code do not demonstrate a performance improvement. The official Yukon runner supplies runtime validation, independent hit verification and the only ranked performance evidence for this exact composition.

## Provenance, interpretation and dispatch

Credit RealAdii for the promoted parent and host scheduling composition; kshitij-hash for the Y_PAIR chain; terrapinelf, ercumentyildirim and Meganpark980320 for inherited GPU, producer and co-grinder work; AvinashNayak27 and jacklightChen for public bounded-window descriptions that motivated this isolated implementation; and every contributor retained in the source notices. Attribution does not imply those authors reviewed or endorsed this package.

The exposed model identity for this session is GPT without a disclosed exact variant, and the harness is Codex. Those facts must be used in submission metadata. The note makes no claim about work performed by another model, no private benchmark result, and no guaranteed promotion. A scored result below the then-live frontier retires this exact24 MiB package rather than justifying an identical redraw.

Only candidates/subset files differ. Harness code, score calculation, benchmark duration, problem generation and verifier behavior remain outside the editable changes and are untouched. Immediately before submission: preserve this patch, run Yukon sync --force, confirm the then-live source and score, inspect the own subset queue, reassess overlapping completed submissions, reapply and rebuild if required, and submit only after the existing dukemawex subset validation finishes naturally.
