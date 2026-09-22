# Subset: tiled scalar production with a promoted fused baseline

Effort: xhigh. This package was developed with GPT 6 Sol through Codex. It changes only `candidates/subset/` and starts from promoted shared source `7c3609b87b9d8e094a16be148fe846dfd5ac7807`. Its subset implementation was accepted as submission `7aef224a-e3ff-43f9-9877-50cdbda3f653` at 623,518,629 verified candidates/s. The promotion rule requires at least 629,753,816/s from that recorded leader. The frontier must be refreshed at validation time.

The authoring machine has no NVIDIA GPU. All rates in this note are official rates for earlier submissions or mathematical counts. This source has no locally measured RTX 4090 throughput. The official run is its first target-device performance measurement. No claimed score is attached.

## Prior run and reason for a clean port

The previous split-SHA and adaptive scalar candidate, submission `d555fd1e-065e-49f1-8942-bb26bfc849f3`, verified 87,293 hits but scored 609,681,321/s and was rejected for performance. It combined scalar production with a negative-Y field path and allocated a full 4 GiB scalar buffer. The result cannot isolate the effect of either change. Another earlier host/SHA candidate, `bf55587b-adb4-4024-9b7f-186269d9a354`, scored 616,008,463/s, also below the 623,518,629/s leader. Those failures are evidence against projecting a speedup from static compiler counts.

This package ports only the tiled producer and runtime route comparison from the earlier local `31b14eb` successor onto the promoted arithmetic. It removes the predecessor's negative-Y field change and its duplicate control implementation. Both fused and produced consumer routes call the same promoted point-add and recovery functions. The fused consumer retains the actual promoted device instruction body, as checked by an sm_89 SASS census. If the producer is not consistently faster during completed search batches, the selector uses that fused route. Trial batches and startup work still count in the official wall clock, so this fallback does not guarantee a score at or above the promoted run.

## Mechanism

The ranked subset shape enumerates early omission sets as epochs and 128 late omission sets per epoch. The promoted consumer computes SHA-256d scalars and ECDSA recovery in one large kernel. Its scalar path is substantial code inside the same kernel as the point chain. The alternative here runs the same SHA first, writes each pair of 256-bit scalars to a temporary GPU buffer, and then runs the promoted curve arithmetic in a second kernel. It has paired and single-scalar producer routes. No scalar or candidate is omitted: inactive lanes retain their original bounds, the producer covers the same epoch descriptors, and the final partial batch is clipped to the exact remaining count.

The key new organization is a 512-consumer-block tile by default. A full-buffer split producer would reserve about 4 GiB for the ranked launch shape. A tile reserves about 8 MiB and is consumed before the same buffer is overwritten. The producer and consumer for each tile are enqueued in one CUDA stream, preserving the scalar dependency and candidate order. A CUDA graph caches the paired or single full-batch route after its first capture. Partial batches use direct launches. Graph capture and launch failures are handled distinctly: a failed capture disables that cache route and falls back to direct launches; a graph launch error returns an error without retrying work that might already have begun.

Epoch descriptors and first-block SHA states remain full-batch allocations and are produced before the tile stream. Each tentative hit tag adds the tile's first epoch before the unchanged exact replay kernel reads it. The replay kernel and the existing hit publication path run after every tile in a batch is complete. The host waits for the verified hit buffer, writes complete records in the inherited format, and only then includes the batch in the runtime comparison. GPU nominations cannot bypass exact replay or be published from a partial tile. The final batch still drains its results.

There is one compile-time switch for each part of this experiment: `QSB_SUBSET_SHA_PRODUCER`, `QSB_SUBSET_TILED`, `QSB_SUBSET_TILE_GRAPHS`, and `QSB_SUBSET_SHA_SINGLE`. Setting the producer switch to zero removes the producer route and leaves the promoted fused path. The tile and graph switches permit separate correctness and compiler checks. Default selection trials use real, completed normal batches, warming both routes and comparing four adjacent pairs in an ABBA/BAAB schedule. A producer is selected only when its geometric mean rate exceeds fused by at least 2.5%, at least three pairs exceed 2%, and no pair falls below 0.995. This is a safeguard against committing to a slower route, not evidence that either producer has passed the gate on an RTX 4090.

## Correctness checks

`python3 -B candidates/subset/test_scalar_producer.py` extracted the production SHA routines and compared 107,520 produced scalar hashes with independent OpenSSL results over ranked and alternate window counts, final and partial batches, and full-buffer versus tile layouts. It found zero mismatches. `python3 -B candidates/subset/test_subset_tiles.py` checked 180 planner cases, including one-block and odd tails, complete coverage, no duplicate tiles, and global hit-tag identities. `python3 -B candidates/subset/test_subset_tile_graphs.py` checked route-specific graph reuse, direct-tail fallback, capture and launch failures, cleanup, and the no-retry rule. `python3 -B candidates/subset/test_subset_sha_tuning.py` checked selector ratios, completed-batch accounting and final boundaries; its source checks verify that the ranked arithmetic functions are the same as promoted `7c3609b` and that both dispatch routes share their non-tile arguments.

These tests exercise extracted production code and host control flow. They do not execute CUDA on an RTX 4090, check actual graph scheduling under its driver, or compare GPU-produced hit sets. The unchanged verifier and scorer will make those checks during the official run. The candidate keeps every reported tentative hit subject to the exact replay and publication gate.

## Compiler census and limits

Compilation and linking passed with CUDA 12.6.20 in the local `qsb-build:latest` arm64 image using both native `-arch=sm_89` and the organizer's default `nvcc -O3 -DQSB_ZEROS_N=24` command shape. The default compute_52 PTX also reassembled for sm_89. The official runner previously used CUDA 12.8.93 and driver 580.178.04, so this is a build and resource check, not its exact JIT result.

Native sm_89 census from this source:

| Kernel | Static SASS instructions | Registers | Shared memory | Stack/spills |
| --- | ---: | ---: | ---: | --- |
| Promoted fused consumer | 21,376 | 128 | 49,152 B | zero |
| This fused consumer | 21,376 | 128 | 49,152 B | zero |
| Produced curve consumer | 9,672 | 128 | 49,152 B | 8 B frame, 12 B stores, 8 B loads |
| Paired scalar producer | 11,824 | 64 | 0 | zero |
| Single scalar producer | 5,160 | 39 | 0 | zero |

The default compiler command reported zero spills for both consumers; reassembling default PTX for sm_89 reproduced the native produced-consumer spill. Keeping a one-block launch bound removes that spill but requires 176 registers and halves resident consumer blocks, so this package retains the two-block bound. Static instruction totals are not additive throughput predictions: producer and consumer occupancy, extra launch work, memory traffic, graph overhead, clock behavior, and the official compiler can dominate the result. The official `d555fd1e` rejection shows why that distinction matters.

The following independent builds also linked: `-DQSB_SUBSET_SHA_PRODUCER=0`, `-DQSB_SUBSET_TILED=0`, and `-DQSB_SUBSET_TILE_GRAPHS=0`. All binaries, PTX, cubins, SASS and logs were kept outside the editable directory. Only `candidates/subset/` is packaged, and the inherited GPLv3 notices and `COPYING` are preserved.

## Interpretation

The hypothesis is that an 8 MiB tile lets the scalar buffer stay close to the consumer and that splitting the large fused kernel improves scheduling enough to offset producer launches and memory traffic. The paired producer has a similar combined static body size to fused, while the single route trades more launches for a smaller per-kernel instruction body. Neither is a proven speedup on the target GPU. The selector is designed to keep the promoted fused route when completed batches do not show a clear gain. The official score and verified hit set decide whether this change is useful.

If validation underperforms, compile with `-DQSB_SUBSET_SHA_PRODUCER=0` to restore the promoted fused route while retaining the source for analysis. The CUDA graph and tile switches separately isolate launch overhead from buffer locality. The earlier negative-Y field experiment is absent, so a new result can be attributed to this producer and tiling mechanism rather than to a changed fallback arithmetic path.
