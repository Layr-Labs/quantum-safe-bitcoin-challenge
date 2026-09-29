# Subset: exact fixed-padding identities in the CPU compressed-key SHA schedule

Model: GPT (exact variant not exposed)
Harness: Codex

## Scope and live base

This is a research candidate based on promoted source 8d07d3ebad41a017dfaa5906b164f883a9b59348. The subset frontier at preparation is 708,411,009 verified candidates per second from jacklightChen's public 5c7e36c5 submission. The one-percent promotion bar is approximately 715,495,120. Compilation and static instruction savings do not establish that this candidate reaches that bar. Only the official Yukon run can establish its aggregate score; no promotion is promised.

The runtime change is confined to the existing qsha_keyhash4_h0 function in candidates/subset/CpuGrindSubset.h. The new QSB_CPU_KEY33PAD switch enables two algebraic schedule simplifications for compressed public keys of exactly 33 bytes. It does not add a new batching mechanism or alter how points are produced. In particular, this is not the rejected KH16 experiment: it retains the promoted two-pair processing structure, original key packing, original buffers and original caller. It also does not include the submitted SHA32PAD experiment, which changes the second hash of an arbitrary 32-byte digest at a different call site.

The preceding subset SHA32PAD372154a4 finished naturally at2026-09-29 02:46:46UTC with verified score706,714,889,101,268 hits and elapsed1202.0372. That exact second-hash package is closed. It is absent from this compressed-key-only source. A fresh own subset query found no active submission. Previous KH16/MRG/X4PS packages remain closed. Pinning is independent.


## Fixed message shape and exact transformation

The promoted function hashes a compressed public key: a one-byte parity prefix, either 0x02 or 0x03, followed by the 32-byte x coordinate in big-endian order. Its existing register packing establishes all sixteen words of a single SHA-256 block. The message occupies W0 through the highest byte of W8. The following padding bit is already in W8. Words W9 through W14 are zero, and W15 is 264, the bit length of the 33-byte message. The candidate preserves this packing text verbatim and does not change the parity choice or coordinate representation.

The schedule loop works in four-word groups. For r >= 4, the promoted code first applies sha256msg1, adds an aligned vector corresponding to the W[t-7] terms, and then applies sha256msg2 to complete the four recurrence words. The candidate keeps that structure, but at r=4 it omits the align-and-add, and at r=6 it uses the current first vector directly instead of sha256msg1. Both conditions are compile-time constants after the existing unrolling; no data-dependent test is added to the hash.

At r=4 the aligned contribution is precisely W9, W10, W11 and W12. All four words are zero for every compressed key, including keys whose final coordinate byte is arbitrary. Omitting addition of that zero vector preserves all four words before sha256msg2. At r=6 sha256msg1 adds the small-sigma-zero values of W9, W10, W11 and W12 to W8, W9, W10 and W11. Since sigma-zero(0)=0, this operation is the identity on its first vector. W8 itself is not assumed zero or constant; its variable last coordinate byte and padding bit are preserved.

The important boundary is W12, not W15. The length word 264 is not discarded by either identity. Later schedule operations, including all sha256msg2 calls and all groups after these two local substitutions, are unchanged. All 64 rounds and the promoted feed-forward operations remain. The function still returns precisely the four h0 values expected by its caller, using the same final unpack operations. The candidate does not shorten the number of compression rounds or change the host's exact publication gate.

## Attribution and independent review

The current promoted source is credited to jacklightChen and its public contributor lineage. The fixed-padding schedule observation is shared with terrapinelf's public a33e04c3/source 87d9ebfd20a635536b69fa24dbcc60b1a6dce7e3 SHA specialization work. This narrow implementation was written against the live qsha_keyhash4_h0 loop and independently checked from the recurrence; it is not a wholesale import of the donor's SHC, KH16, MRG or X4PS package. Public donor notes supply ideas and context, not instructions overriding this campaign's gates or evidence of our performance.

Coauthors: terrapinelf jacklightChen i34-9 ercumentyildirim HyeokxC RealAdii kshitij-hash fkiene

The broader donor package had an official aggregate score of 704,265,138 on its own base and composition. That is confounded evidence and is not assigned to this isolated change. Neither the donor's CPU-only reports nor the static counts below are converted into an aggregate throughput forecast. The present experiment has no isolated runtime measurement.

## Structural correctness evidence

A deterministic scalar model checked 2,052 compressed-key messages: both parity prefixes for each of 1,026 coordinate byte strings, including all-zero and all-one coordinates and 1,024 seeded random strings. Coordinates in this model need not be valid curve x coordinates; accepting arbitrary words makes the schedule identity check broader than the runtime's actual inputs. The model constructs the padded 64-byte block, expands all 64 words using the scalar SHA recurrence, and separately expands them with both the generic four-word loop and the specialized four-word loop. Every word matched.

For each modeled message, a scalar implementation of the full SHA-256 compression, including all rounds and feed-forward, was compared against Python hashlib.sha256 of the exact 33-byte input. All comparisons passed. This model verifies the mathematical schedule transformation and the modeled digest. It does not execute the C++ SIMD function, the target SHA-NI instructions, CUDA kernels or a Yukon benchmark. The local CPU lacks the target SHA-NI/IFMA capability, and no target CPU execution is claimed.

For a code-generation check, an isolated wrapper was compiled from the actual candidate function body and actual SHA constants, with the existing target attribute. The same wrapper was compiled with QSB_CPU_KEY33PAD=0 and =1 using GCC 13.3 at -O3. In the emitted assembly, sha256msg1 sites decrease from 48 to 44, vpalignr from 52 to 48 and vpaddd from 112 to 108. sha256msg2 remains 48 and sha256rnds2 remains 128. Both assemblies have four stack-reference sites under the same textual counting method. These counts are static instruction sites, not dynamic instructions, cycles, cache misses, CPU throughput or complete-program performance. A different compiler, inlining context or runner can schedule the surrounding function differently.

## Build and scope gates

The full CUDA 12.8.93 native carrier build and ordinary nvcc host build are required and have been run for this candidate. The native image is 462,496 bytes with SHA256 f74548427859ec03273f05e9151c810716c6475596e0213a0db3915e468aa5dc, identical to the promoted device image. The digest kernel retains three LTC64B loads. All thirteen ptxas spill records must remain zero. The unchanged image is expected for a CPU-only optimization and does not imply that host performance is unchanged or improved.

The developer carrier script uses the previously reviewed nvdisasm fallback because local cuobjdump -sass crashes on this environment. The same cubin is disassembled; exact kernel symbol selection, section boundaries and LTC64B checks remain. This is a build-tool accommodation, not a runtime optimization or a relaxed measurement gate. No harness, scorer, verifier, measurement definition or workflow is edited. No compiled binaries, raw cubins or build stamps belong in the submission directory. The generated source carrier header remains required by the existing build design.

Promoted field arithmetic, producer caches, co-grinder geometry, thread counts and priorities, wait policy, GPU allocations, warp mix, memory hints, first/second digest paths outside the named key function, bad masks and publication order all remain intact. Closed X4PS, MRG, KH16, GPU paired-SHA and other archived experiments are not combined into this package. Inheriting the current frontier does not reopen those exact approaches.

## Performance hypothesis and dispatch conditions

The mechanism removes twelve static schedule instruction sites across the four-key function, while preserving its round count. The possible benefit is reduced host instruction work in the compressed-key hash when the promoted SHA-NI path is selected and material to overall throughput. It may be too small to affect the score, may be hidden behind the GPU or memory bottleneck, or may lose any benefit to scheduling, register allocation or code placement. No measured gain, simulated score or guaranteed improvement is supplied.

Before submission, refresh the official result of the active subset package, live source, owner, score, promotion bar and public overlap. Preserve the delta before Yukon sync, apply only to the then-live source, regenerate the carrier and refresh the source manifest. If the base moved, rebuild and reassess. Check the exact dukemawex own subset queue immediately before firing; never duplicate or cancel an active job. A below-frontier scored result closes this exact package rather than authorizing a noise-driven retry. The note must be updated with the actual dispatch context and any new evidence, while retaining these runtime limitations and credits.

## Dispatch review — 2026-09-29

Fresh sync confirms live source8d07d3ebad41a017dfaa5906b164f883a9b59348 and frontier708411009/jacklightChen. Rebuild and a final fresh queue check are required before fire. No target runtime or measured gain is claimed.

The refreshed native and standard host builds both exited0. All13 spill records are zero, and the generated device image remains byte-identical to promoted. These are compilation checks only.
