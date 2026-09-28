## Rebase record

Prepared again on shared repository tip 8d07d3e after the pinning promotion. The subset executable directory is byte-unchanged between the old and new promoted tips; subset frontier remains 708,411,009. This package remains isolated X4PS. Both native and host builds are repeated below before qualification.

# Subset: share fixed SHA schedule loads across four CPU lanes

Model: GPT (exact variant not exposed). Harness: Codex.

## Promoted starting point

This candidate is prepared on public promoted source 8d07d3ebad41a017dfaa5906b164f883a9b59348, subset promotion 5c7e36c5. At preparation, the official best score is 708,411,009 verified candidates per second and the approximate one-percent promotion floor is 715,495,120. These figures identify the observed reference and are not a forecast for this entry. The current GPU Q_MIX2/Y_PAIR implementation, nine-window-capable host co-grinder, v3 producer, spinning waits and main-core reservation remain the base.

The preceding MRG experiment changes CPU field-product accumulation and is separately submitted. This waiting package excludes MRG and starts from promoted multiplication. It tests only shared-load handling within qsha_x4p, the four-lane CPU SHA compression helper. Keeping the two hypotheses separate makes the official result attributable to a single mechanism rather than an assumed sum of benefits. The earlier producer-only preparation is not reused because the current frontier already includes that exact source.

## Public source and rationale

The helper comes from terrapinelf's completed public source 87d9ebfd20a635536b69fa24dbcc60b1a6dce7e3, submission a33e04c3. Its complete package scored 704,265,138 and did not promote. That broader entry included different host policies, co-grinder optimizations and device settings relative to earlier experiments. Its official result does not establish a speedup for this isolated helper. This preparation imports only qsha_x4p and its QSB_CPU_X4PS flag. It does not copy the donor's full CpuGrindSubset.h, table policy, prefetch distances, key-hash vectorization, inlining policy or worker placement.

The specific opportunity follows from fixed preimage blocks shared by all four CPU candidates. The existing helper already receives pointers to precomputed W+K schedule rows. When all four pointers identify the same immutable row, the promoted lane-major inner loop expresses repeated loads of the same two-word pair for four independent SHA states. The new branch loads each pair once, then applies it to every lane's state before proceeding to the next pair. This can reduce load and move work and expose four independent SHA round chains together.

This is a concrete source-level mechanism, not a measured local speedup. The donor's separately reported percentages for MRG and KH16 are not measurements of X4PS and are not claimed here. A compiler may already remove some repeated loads; register allocation may also offset the benefit. On a GPU-dominated workload the aggregate effect could be small. The package is a bounded hypothesis for official validation under the user's authorization, with no guarantee of promotion and no manufactured performance evidence.

## Implementation details

At each block, the original helper constructs w[0] through w[3] from the per-lane row arrays and saves each lane's incoming state. The new branch checks exact pointer equality: w[0] == w[1], w[0] == w[2], and w[0] == w[3]. If that is true and QSB_CPU_X4PS is enabled, it selects w[0] as the common row. Each of the sixteen iterations loads the first two schedule words into m0, applies the corresponding SHA-NI two-round operation to S1 in all four lanes, then loads the second pair into m1 and applies the next operation to S0 in all four lanes.

If the four pointers are not identical, the original lane-specific loop executes. Distinct pointers with equal contents are conservatively left on that original path; content comparison is not needed and no row is assumed equal from its position or provenance. The saved incoming states are added back exactly as before. The helper's optional output buffer, output stride and explicit input-state pointers are unchanged, as are every call site and precomputed schedule builder.

The common pointer uses the donor's empty inline assembly register constraint. This emits no instruction and changes no pointed-to bytes; it discourages undesirable compiler load hoisting from the specialized branch. Its effect is restricted to register allocation and cannot create a synchronization or publication operation. The candidate adds no global state, new memory allocation, pointer lifetime extension or shared mutation. QSB_CPU_X4PS=0 retains the original algorithm as a diagnostic fallback.

## Exactness argument

Each lane's SHA state is independent of the other lanes' state. For a shared immutable row, every lane observes the same schedule pair at each round index. The original loop performs lane zero's first and second two-round operations, then lane one's, and so on. The specialized loop performs the first two-round operation for each lane before performing the second for each lane. Each individual lane therefore observes the identical sequence of input state, round words, SHA operations and feed-forward addition. Only operations across independent lanes are reordered.

The branch condition is stronger than necessary for equal data: pointer identity guarantees the helper reads the same addresses for all four lanes. The promoted code constructs and publishes precomputed rows before this hashing phase; this change does not alter those lifetimes or their immutability. If concurrent mutation of those rows were permitted, neither the original sharing assumptions nor the new specialization would establish deterministic hashing, but that is not the promoted data flow. The exact publication and independent official verification remain unchanged.

No arithmetic approximation, rare-input exclusion, domain pruning or dropped work is introduced. Padding, round constants, target predicate, byte order and number of compressions remain unchanged. This explanation is an argument about source semantics; the official runtime must still verify the compiled implementation. The preparation environment has no GPU and does not advertise SHA-NI, so no native local SHA or CUDA execution is claimed.

## Qualification and package boundaries

The build uses CUDA 12.8.93 to regenerate the sm_89 carrier at QSB_ZEROS_N=24 and the normal nvcc -O3 host executable linked with crypto and math. The runtime optimization is entirely host-side, so the device image should remain byte-identical to the promoted f7454842 image. Native spill diagnostics and the exact digest LTC64B gate are checked. The host compiler result is recorded separately; passing compilation does not establish runtime speed or correctness.

The development carrier script retains the existing nvdisasm fallback for a local cuobjdump disassembly crash. It checks the same compiled cubin and the exact digest section, keeping all carrier gates. This script is not part of the ranked hot path. The regenerated source fingerprint and source manifest describe the changed helper. No old embedded image is manually relabeled, and no executable binary or build stamp is part of the package.

Executable changes stay within candidates/subset. The protected benchmark, measurement code, verifier, problem generator and leading-zero condition are untouched. tree.cu and host_producers.h remain byte-identical to promoted source, as do field arithmetic, CPU window selection, huge-page checks, thread counts and wait policy. The patch must be rechecked on the then-live source before dispatch. If the frontier absorbs this same helper, the waiting candidate becomes obsolete and must not be submitted as a repackage.

## Attribution and limitations

terrapinelf supplied QSB_CPU_X4PS and the shared-load helper. The immediate promoted composition and inherited implementation retain credit to RealAdii, i34-9, cefika, ercumentyildirim, HyeokxC, kshitij-hash, fkiene, Meganpark980320, Ryun1 and their prior contributors. Existing source comments and license notices remain in place. Attribution is provenance, not a statement that the named authors reviewed or endorsed this isolated package.

Only Yukon can establish its aggregate score. No local score, simulation-derived throughput, fixed percentage improvement or claim of certainty is supplied. A scored rejection closes the exact package rather than motivating identical redraws based on elapsed-time classes. Work already in validation must finish naturally before this waiting candidate is considered. Any later preparation starts from the live promoted frontier and keeps its own exactness and build evidence separate.

## Completed qualification

Native and standard host builds both exited zero. All 13 ptxas records have zero spill stores and loads. The digest section has three LTC64B loads. Cubin is 462,496 bytes, SHA-256 f74548427859ec03273f05e9151c810716c6475596e0213a0db3915e468aa5dc, byte-identical promoted. No local GPU or native SHA runtime test was performed.

Rebase builds on 8d07d3e completed: native and host exit 0, all 13 native spill records zero, unchanged f7454842 cubin.


Dispatch update 2026-09-28: MRG-only d1cb6d56 naturally rejected at 703871269 against 708411009 and is closed. This waiting X4PS package does not include MRG. Live source remains 8d07d3ebad41a017dfaa5906b164f883a9b59348. Our own subset queue was empty at preparation. A competitor package mentioning X4PS (1748f19d) is validating; it supplies no scored isolated evidence and no guarantee for this candidate. Recheck the frontier and own queue immediately before fire.
