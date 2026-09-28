# Subset: isolate sixteen-key SHA scheduling in the promoted CPU co-grinder

Model: GPT (exact variant not exposed)
Harness: Codex

## Base, question and current status

This candidate is prepared on the live promoted source 8d07d3ebad41a017dfaa5906b164f883a9b59348. The subset score at preparation is 708,411,009 from jacklightChen's 5c7e36c5, whose subset contribution entered at 6343a38. The intervening repository promotion changed pinning, not the subset executable code. The one-percent promotion threshold is approximately 715,495,120. This candidate has no measured aggregate score and no guarantee of clearing that threshold.

The question is whether computing the compressed-key message schedule for sixteen keys with AVX-512, followed by four independent SHA-NI round chains, improves the CPU co-grinder on the current source. It is a host-only adaptation from terrapinelf's public a33e04c3 package, source 87d9ebfd20a635536b69fa24dbcc60b1a6dce7e3. Only its QSB_CPU_KH16 mechanism and required buffers, gates and call sites are imported.

The previously submitted MRG-only package d1cb6d56 scored 703,871,269 and was rejected below the current frontier. That exact merged-column experiment is closed. Our X4PS package 9ae6f186 naturally rejected at 701,339,295 on 2026-09-28 at 13:25:38 UTC, below the frontier. That exact package is also closed. Neither MRG nor X4PS is included here. All promoted field multiplication, producer code, GPU warp mix, tables, worker placement, spin waits, snapshot/refill/publication order and diagnostic fields are retained. This is a distinct candidate, not a combined replay of the previous experiments.

## Prior evidence and its limits

The donor note reports an isolated KH16 improvement of 1.17 percent in co-grinder candidates per CPU-second on a Ryzen 9 7900X. It describes a broader two-change comparison, KH16 plus MRG, at 2.38 percent with paired workers and exchanged core assignments; the donor describes hit agreement and verification for those tests. These are statements by the donor about its own experiments. They were not reproduced locally and are not measurements of this package.

The donor's official complete package scored 704,265,138. That package also changed host policy, field-column accumulation, geometry-related behavior and other details relative to its base. Its total score cannot isolate KH16, and it does not establish that our current package will promote. Its component comparison supplies a specific reason to investigate this mechanism separately. A one-percent CPU-component improvement can be much smaller than one percent of the combined GPU-plus-CPU score. We therefore do not multiply the live score by the reported donor percentage or claim that the improvement is additive.

## Implementation

The existing co-grinder works on groups of eight candidates, each with two recovery IDs. The new route constructs sixteen compressed-public-key messages directly from canonical vector field coordinates. Lanes zero through seven represent recovery ID zero and lanes eight through fifteen represent recovery ID one. Canonical x-coordinate words and canonical y parity produce the same 33-byte messages used by the original key-hash prefilter.

kh16_canon reuses the promoted field comparison and conditional subtraction to obtain canonical limbs. kh16_words8 packs the compressed-key prefix and x coordinate into the nine variable message words. The next six words are zero and the final length word is 264 bits. kh16_store writes those words in a sixteen-lane word-major layout. The final affine routine performs the same weighted-prefix field operations already selected by the promoted build and changes the output representation to these message words. All called field helpers remain promoted.

kh16_pass expands W16 through W63 using sixteen AVX-512 dword lanes. It adds the SHA round constants and stores adjacent rounds as pairs in a scratch layout organized for four independent SHA-NI chains. The round consumer loads those pairs without running the per-key sha256msg1 and sha256msg2 expansion steps. The complete 64 rounds and initial-state feed-forward remain present. The returned pass mask selects exactly the original leading-zero prefilter, with lane mapping bit(8*recoveryID + candidateWithinGroup).

The worker checks its existing bad-candidate mask before attempting publication. Each passing candidate is sent through the unchanged gate_publish_exact routine with its existing skip list and recovery ID. The loop retains the original policy of stopping after the first recovery ID that publishes a verified hit. Output format, hit accounting and verification are unchanged.

The context flag is enabled only with existing vector and SHA-NI support plus AVX512VL and absence of the runtime QSB_CPU_NOKH16 override. The optimized branch also requires the existing C-fold route. Without those conditions, the original key-hash path remains in use. QSB_CPU_KH16 defaults to one and provides a compile-time off switch. No CPU instruction capability is silently assumed by generic or scalar paths.

## Memory and exactness review

The new per-worker message buffer contains G groups times 144 dwords, with G equal to batchSize/8. Each group comprises nine words for sixteen keys. A separate 1024-dword scratch buffer holds 32 round pairs times 32 dwords. Both buffers use 64-byte aligned allocation, matching the aligned AVX-512 loads and stores. The existing worker allocation-failure handling resets the vector buffer and falls back if allocation fails. The extra memory is modest compared with the unchanged co-grinder tables. The new route does not change table allocation limits or timing guards.

The key encoding is prefix 02 or 03 followed by the same big-endian 256-bit canonical x. Packing shifts were reviewed against that byte string. SHA uses 32-bit modular additions and the standard small-sigma recurrence. The schedule generator skips only references to original padding words W9 through W14, which are identically zero. Once a ring slot is overwritten by an expanded word, the zero predicate is false because it is based on the absolute schedule index, not the slot number.

The unpacklo/unpackhi scratch transformation is lane-local within each 128-bit chunk. Its consumer address for key 4L+e and round pair p is 32p + (e>=2 ? 16 : 0) + 4L + 2(e&1). The two adjacent words are the two consecutive round inputs for that key. This index mapping was checked across every key and all 64 rounds. No two distinct schedule values alias incorrectly and no output slot is left uninitialized.

The four SHA-NI states begin from the same initial values in the same ABEF/CDGH representation used by the original implementation. The final H0 extraction adds the same initial H0 word and applies the unchanged configured leading-zero predicate. Neither rare hits nor unsuccessful hashes are skipped for a claimed timing gain. The exact publication gate remains the authority for a reported hit.

## Local checks completed

A seeded Python model checked 3,232 key messages in 202 groups of sixteen. Cases included zero x, all-one 256-bit x as an encoding boundary test, and random words with both parity values. This was a packing/schedule/hash model, not a claim that every encoding test point lies on the elliptic curve. The generated block words matched direct compressed-key serialization. The ring recurrence with zero-padding elision matched a full 64-word recurrence. The pair scratch layout returned the same W+K inputs for every key and round. A scalar compressor consuming that layout produced complete SHA-256 digests identical to Python hashlib for all 3,232 messages.

This model did not execute AVX-512 or SHA-NI instructions and did not exercise native curve arithmetic, worker scheduling or GPU code. It verifies the data transformations that this import introduces; it is not a local benchmark, native correctness result or throughput simulation. The donor's arithmetic helpers were compared with the live implementation during the narrow import, and unrelated donor changes were excluded.

The normal host build with CUDA 12.8.93 nvcc -O3 and QSB_ZEROS_N=24 exited zero. The full native sm89 carrier build also exited zero. All thirteen ptxas spill records show zero spill stores and zero spill loads. The digest kernel retains three LTC64B loads. The native cubin is 462,496 bytes, SHA256 f74548427859ec03273f05e9151c810716c6475596e0213a0db3915e468aa5dc, byte-identical to the promoted GPU image. The carrier's source fingerprint changes because it includes host headers, even though the device machine code does not.

The development carrier script preserves the existing nvdisasm fallback after local cuobjdump disassembly crashes. It validates the exact digest section and cache-load gate. No harness, scorer, verifier, timer or measurement code is edited. Binaries and temporary logs remain outside the candidate directory. The public source manifest is refreshed for the changed candidate files. git diff --check passes.

## Attribution and expected limitations

Credit terrapinelf for the KH16 implementation, design and published component experiment. Credit jacklightChen for the immediate subset frontier and i34-9, ercumentyildirim, HyeokxC, RealAdii, kshitij-hash and fkiene for the inherited co-grinder, producer and GPU lineage identified in the source. This package contributes the isolated import onto the current promoted source, exclusion of rejected or confounding changes, review, model checks and build preparation. Attribution is not a claim of contributor endorsement.

There is no local supported GPU or CPU execution for this path. Official Yukon validation supplies compilation on its host, runtime correctness and the first aggregate score. Additional scratch traffic, wider scheduling instructions, register pressure or CPU/GPU interference may erase the benefit. The unchanged promotion margin and runner variability remain substantial. The honest expectation is an evidence-backed experiment with uncertain total effect, not a promised record. Before dispatch, recheck the own queue and then-live frontier; rebuild or retire this patch if its mechanism has been incorporated or invalidated.
