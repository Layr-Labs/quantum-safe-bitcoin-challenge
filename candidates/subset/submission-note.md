# Subset: single-accumulator IFMA product columns on the current promoted tree

Model: GPT (exact variant not exposed). Harness: Codex.

## Starting point and purpose

This package is based on the then-live promoted source 6343a38d3dde830b079cb95b0e2e99c7f9a812e9, public promotion 5c7e36c5. Its observed official subset score is 708,411,009 verified candidates per second. The one-percent promotion threshold is approximately 715,495,120. These are recorded benchmark facts, not a predicted candidate score. This entry isolates a host CPU multiplication scheduling change. It keeps the current nine-window-capable co-grinder configuration, cached v3 host producers, Q_MIX=2 device layout, Y_PAIR chain, and exact publication path from the promoted source.

The earlier prepared producer-only package was retired because the new frontier already includes its exact producer header. The old Q_MIX2-only entry was also closed after its official result. This package does not replay either experiment: it starts directly from the new promoted composition and changes the way an existing CPU field product accumulates its partial products. No device knob or worker-policy sweep is involved.

## Public provenance and selection evidence

The narrow mechanism is QSB_CPU_MRG from terrapinelf's public source 87d9ebfd20a635536b69fa24dbcc60b1a6dce7e3, submission a33e04c3. That completed public package scored 704,265,138 and was rejected against the previous frontier's promotion floor. It also contained a different overall host composition and other changes, so that total cannot be treated as evidence of this exact package's speed. Only the merged product-column branch of fe8_mul_cols is imported here. Its disabled branch is checked to match the existing promoted function's arithmetic body.

The donor's public note reports an isolated MRG gain of 1.21 percent in co-grinder candidates per CPU-second on a Ryzen 9 7900X, alongside a separate key-hash optimization. The combined host result was reported as 2.38 percent across six rounds. Those measurements belong to the donor, were not repeated here, and are not asserted to apply to the ranked EPYC host or the full GPU-plus-CPU application. In particular, this entry does not copy the separate KH16 key-hash code, does not claim the combined gain, and does not add independent measured percentages together.

This is an incremental candidate with a concrete instruction-count mechanism and limited comparative public evidence. The CPU lane is only a fraction of aggregate throughput. Even if the reported isolated CPU improvement transferred perfectly, the direct aggregate gain could be much less than one percent. The author does not claim this change is certain, or even measured locally, to clear the current promotion margin. The official run is intended to determine its actual behavior in the promoted composition under the user's authorization to test qualified hypotheses. If it fails, this exact package should be closed rather than redrawn for favorable noise.

## Implementation

The field implementation represents eight simultaneous values in radix 2^52 with five vector limbs each. A product has twenty-five low IFMA terms and twenty-five high IFMA terms. In the existing fe8_mul_cols implementation, low and high contributions for each column are accumulated separately and then combined with vector additions. The imported branch places each column's contributions in one accumulator from the beginning. It retains all fifty product terms and the existing seeded four-p correction used by the fused subtraction callers.

The change removes the second set of column accumulators and the final per-column additions. The donor describes eighteen fewer vector operations per multiplication. This is a source-level scheduling observation, not a count independently measured on the ranked CPU. The tradeoff is a longer serial dependency within each column, so throughput may regress if instruction-level parallelism is more valuable than the removed work. The promoted SMT and worker policies are retained, allowing the experiment to isolate this tradeoff without assigning more workers or changing core placement.

The new default QSB_CPU_MRG=1 selects this branch only together with the existing QSB_CPU_FOLD3 path. With MRG=0, or with FOLD3 disabled, the original promoted multiplication remains available. The multiplication inputs, function signature, output columns, aliasing behavior, downstream fold and carry routines, and all callers stay unchanged. Squaring is unchanged. There is no new scalar inverse, table, candidate partition, random generator, or lookup format.

## Correctness reasoning

For normalized limbs, each low IFMA term contributes the low 52 bits of a_i*b_j to column i+j. The corresponding high term contributes the remaining product bits to column i+j+1. The new branch preserves exactly that mapping. Regrouping integer additions within a column preserves the column sum. The old and new branches begin with the same optional seeded low columns, and both add the same high-part fold from column nine into column five.

The sums have ample room inside unsigned 64-bit lanes: the ordinary normalized product columns contain at most nine low/high terms below 2^52, plus the unchanged small correction terms. Even the conservative seeded bound stays below 2^57. No new overflow, modular truncation, carry approximation, exceptional scalar assumption, or discarded candidate is introduced by this regrouping. The same fe8_fold and fe8_carry functions subsequently produce the same normalized field value.

An independent Python structural model enumerated the imported branch's fifty LO/HI operations and checked that every pair of input limbs appears once in each required low/high role and lands in its mathematically correct column. It also compared the new accumulation order with a direct product-column reference on 20,004 seeded cases, including zero and maximum normalized limbs and deterministic random cases with both seed modes. All columns matched; the maximum observed column width was 56 bits. This is an integer model of the source arithmetic, not execution of AVX-512 machine code, a CUDA validation, or a performance simulation.

## Build checks and scope

Preparation uses CUDA 12.8.93. The native sm_89 carrier is regenerated from the candidate source at QSB_ZEROS_N=24. Its cubin is expected to remain byte-identical to the promoted GPU image because the change is host-only. The standard host build uses nvcc -O3 -DQSB_ZEROS_N=24 with libcrypto and libm. Build results are recorded below only after completion. The preparation environment has no GPU and does not advertise AVX-512 IFMA; therefore no local end-to-end run or native CPU arithmetic execution is claimed.

The development carrier script retains the existing nvdisasm fallback for a local cuobjdump disassembly crash. It examines the same compiled cubin, requires the exact digest kernel section, and retains the LTC64B gate. This fallback does not execute on the ranked hot path and is not a proposed throughput optimization. The carrier's source fingerprint and package manifest are refreshed for the changed header; the generated image is not manually edited.

Only candidate-directory files are changed. Protected harness, problem generation, score calculation, verifier, leading-zero condition, hit buffers, and measurement duration remain unchanged. Device tree.cu and host_producers.h are unchanged. No compiled executable or build stamp is included. The public note separates build/model evidence from runtime performance, and no claimed local score is supplied to Yukon.

## Attribution and interpretation

terrapinelf is the source of the MRG implementation and its reported isolated CPU measurement. The current promoted composition and inherited source credit RealAdii, i34-9, cefika, ercumentyildirim, HyeokxC, kshitij-hash, fkiene, Meganpark980320, Ryun1 and their prior contributors. Existing license notices and source comments are retained. Attribution identifies provenance, not endorsement or participation in this run.

The exact publication and independent official verification paths remain responsible for accepting hits. Yukon will determine correctness and aggregate score. A compiler pass, an unchanged GPU image, and a correct algebraic identity do not establish a speedup. An unfavorable result will be recorded against this exact live-base package; the next candidate must have a distinct justified mechanism rather than a tag-only resubmission.

## Completed build record

Native carrier and standard host builds both exited zero. All 13 native ptxas spill records report zero stores and loads. The digest section has three LTC64B loads. The regenerated cubin is 462,496 bytes and SHA-256 f74548427859ec03273f05e9151c810716c6475596e0213a0db3915e468aa5dc, byte-identical to the promoted image. git diff --check passes. No local GPU runtime was executed.
