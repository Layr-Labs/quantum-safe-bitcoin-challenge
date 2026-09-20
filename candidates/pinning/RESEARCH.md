# Pinning: original-association register cofactor top and source-equivalent scalar decoding

Effort: medium. Developed with GPT 6 Astra using Codex. Base: the latest promoted Pinning source, `66fede0cc10d36ad15041861d3eddaad97481ac6`, submission prefix `dcd0147`, score **789,011,576**. The promoted runtime is inherited, including its host publication gate. Only `candidates/pinning` changes. There was no local C++/CUDA compilation or GPU benchmark; this is a remote performance experiment after source and Python/PTX semantic audits, not a measured speedup claim.

## Why this is a different experiment

Our preceding PR745 (`eb6d9871`) verified but scored **769,172,989**, or -1.21386% against its then-current 778,624,395 baseline. That composition reduced the tree top to four warp multiplication waves, increased scalar work from43 to63 products, and used a separate complete multiplier to support reassociation. The result did not justify carrying it forward unchanged. Its exact arithmetic helper, four-wave schedule, final canonicalizations and changed association are all absent here.

This candidate returns to the current promoted tree's **six multiplication waves and original expression tree**. It targets the intermediate shared-memory traffic and synchronization instead. Some upward products are duplicated into idle lanes to make one operand local in every wave; the other comes from a warp shuffle. The arithmetic helper is the frontier's unchanged `qsb_field_mul_sc`, with no new carry omissions or new multiplication association. A separate decoder improvement bypasses two volatile seed-code store/load pairs and expresses the same field windows directly with32-bit funnel shifts.

The published PR700 scheduling work by EvanYan1024 motivated the earlier top-region investigation. Its four-wave reassociation is not used in this version, but that unpromoted research substantially helped identify and audit this region, so the author is credited. fkiene's public `6fe3a56` description supplies the independent decoder mechanism; it officially scored779,526,447 on the prior base, only about+0.116%, and was rejected. That small observation is not treated as a proven gain on this newer source or submitted alone. Both receive coauthor credit. The current promoted runtime and its original authors are credited as the base, including Saviour1001's publication gate and the ercumentyildirim arithmetic lineage.

## New frontier means a new raw arithmetic contract

The latest base differs from the previous one: it has C31/carry62 reductions and bounded multiply-tail truncation, with a host exact-hit publication filter. Those are inherited unchanged. This submission neither certifies the inherited approximations as exact field arithmetic nor adds another one. The audit executes the actual current PTX, rather than silently reusing our old carry model.

For the current multiply, write B=2^256, K=2^32+977, T=a*b and F=low256(T)+K*high256(T). Let R=low256(F) and h=low32(high256(F)). The current C31 multiply preserves R above bit95 and replaces its low96 bits with `low96(R+K*h)`. This describes the inherited truncated raw result, not canonical reduction modulo p. The PTX audit compares that description with the actual current inline assembly, including near-B inputs that distinguish it from an exact multiplier.

Crucially this raw result is a deterministic function of the full product a*b. Swapping a and b therefore preserves every output bit, even in the inherited error cases. The new schedule permits only these operand swaps and duplicate evaluations; it does not reassociate any product. The canonicalized symbolic expression-tree audit allows swaps within a multiply node only, never changes parentheses. Thus errors in an inherited intermediate propagate in the same way, and the final raw root and exclusions agree with the promoted schedule, rather than merely agreeing modulo p on typical inputs.

The host gate, both output writers, candidate enumeration, point chain, table construction, SHA paths, recovery algebra and pipeline are unchanged. Source closure checks remove only the decoder functions and their seed consumers from old/new `pinning.cu`, then require the remaining bytes to match exactly. `GPUMath.h` and every other inherited runtime header besides the cofactor header are byte-identical.

## Register layout and six-wave dataflow

The first warp owns the top region. Initially lanes0..31 load two copies of the sixteen existing subtree roots. Each lane holds one four-limb `node`; there is no extra shared arena or global checkpoint allocation.

| Wave | Active lanes | Role |
| --- | --- | --- |
| 0 | 16..31 | duplicate the eight pair products |
| 1 | 24..31 | duplicate the four quarter products |
| 2 | 28..31 | duplicate the two half products |
| 3 | 24..27 and31 | four exclusions and one root |
| 4 | 16..23 | eight exclusions |
| 5 | 0..15 | sixteen exclusions |

An upward lane multiplies its own node by the node selected by XOR8,4 or2. Once an immutable product reaches its last use, its lane becomes an exclusion accumulator. In the combined wave the four exclusions use the quarter lane as the local operand and the opposite half as the shuffled operand. Root lane31 uses the two duplicate halves in30 and31. The eight-output wave places exclusion i in lane16+(i xor4); the sixteen-output wave places exclusion i in lane i xor8. Those permutations allow every wave to exchange only one256-bit operand.

The root is retained in lane31 until final publication. The sixteen exclusions are stored at the same `excluded[N-32+i]` offsets consumed by the unchanged lower descent. The up-sweep stops at count16; descent resumes at count32 with offset `2*N-64`. Production widths128 and256 and the smaller audit widths32 and64 are covered. Tail and unusable leaves retain the existing identity contract.

All32 lanes execute each `__shfl_sync` with the full mask before the arithmetic work predicate. Source lane indices always lie in0..31; every node is initialized, including inactive arithmetic lanes. The compile-time wave branches are uniform. No shared memory is communicated internally between waves, so register-transfer synchronization suffices there. A final `__syncwarp` publishes the shared exclusions, and the original lower-tree cross-warp barriers remain. This reasoning is checked at source level; it is not a native CUDA race-detector result.

## Honest work and cost accounting

The original top performs43 scalar multiplications in six warp waves. This version performs57 scalar multiplications, also in six waves:14 duplicate upward products occupy otherwise idle lanes. Unlike PR745, all use the unchanged frontier helper and there are no additional normalizations. Extra active-lane arithmetic can still consume power; it is not free merely because warp instruction count is unchanged.

The source exchange envelope is:

| Region | Original shared top | Register top |
| --- | ---: | ---: |
| 64-bit shared-load instructions | 48 | 4 |
| intermediate64-bit shared-store instructions | 20 | 0 |
| 32-bit shuffle instructions | 0 | 48 |
| warp publication/synchronization calls | 6 | 1 |
| scalar intermediate shared reads | 344 | 128 |
| scalar intermediate shared writes | 104 | 0 |

The128 initial scalar word reads reference64 unique addresses because of duplication; hardware broadcast behavior is not measured. Final output stores are excluded from both sides and remain equivalent. These are source-level counts, not SASS or latency measurements. They support testing the removal of repeated shared store/barrier/load stages, not a numeric throughput prediction.

The helper's unrolled six waves may expand more code than the original rolled traversal. Register liveness, instruction-cache pressure, shuffle cost and extra active-lane power are material risks. No register, occupancy, spill or percentage claim is made without a native build. The point-chain loop's existing unroll factor is unchanged.

## Decoder integration

The unchanged signed-recode setup produces M and its negative flag. Each of the15 compile-time field windows is extracted from a pair of32-bit words with `__funnelshift_r`, then masked to its original width. The top window uses zero for the nonexistent ninth word, and the unused array index is masked into bounds even before constant folding. An exhaustive field/sign check validates the sign-bit-to-mask transformation. The last digit retains the separate negative flag rather than using a nonexistent top field bit.

The first two packed codes return as `uint2` and feed the existing table loader mask/index convention. Their two volatile shared stores and two volatile shared loads disappear (16bytes per candidate). The other13 codes stay at the original plane offsets. No table record, sign, chunk order or scalar domain changes. The surrounding shared arena handoff remains protected by the existing block barrier.

## Validation

`python3 -B candidates/pinning/research/audit.py` runs the packaged source-bound audit without compiling CUDA or C++. The final result is recorded in `research/audit-result.json`:

- 2,221 current-frontier PTX operand pairs, each tested in both orders and against the current raw integer model; includes high-carry structured inputs.
- 46,080 full-tree leaf bit comparisons against a direct model of the promoted merged-top2 traversal, across32/64/128/256 leaves and partially active tails.
- Ten full trees executing the actual current PTX in both baseline and candidate traversal, covering1,920 leaves at production widths128 and256, identity inputs, p/B boundaries and tails.
- 524,288 exhaustive decoder field/sign cases,8,008 raw scalars /120,120 digit comparisons, plus516 basis-vector/sign cases for the extraction mapping.
- Two negative controls distinguish inherited truncation from exact field math and detect a wrong exclusion permutation.

The interpreter rejects unknown instructions and uninitialized registers. The current source's inline PTX comments are removed lexically and its C31 empty-tail macro is explicitly expanded; no CUDA arithmetic is replaced by an ideal exact-field oracle. The audit also binds the integrated helper's indices/predicates, its placement after the field helper definitions, the lower-tree continuation and all decoder changes. Full source hashes accompany the result.

These tests do not replace the official native build, independent hit verification or ranked timing. The remote run is the first performance evaluation of this exact composition. If it loses, keep that result as evidence against this register-exchange arrangement rather than re-uploading inert variants.

## Public screening and scope

The newest public descriptions were screened before packaging. `3fcda0f` reports only about+0.342% locally for the older four-wave reassociation on the PR706 base; it is not evidence for this different register schedule and is not imported. `5293208` and `8e54223` describe micro/inert remeasurements; their reset/redraw instructions are not followed. No other solver's unpromoted implementation or binary was downloaded. The baseline source was fetched because it is the newly promoted frontier.

The earlier SHA ST switches, wider tables, hybrid inversion stages and losing complete four-wave top are not part of this candidate. The new gate and arithmetic defaults are inherited solely through the current promoted base, byte-preserved, with no further approximation added. The benchmark manifest,100-basis-point promotion rule, harness and score path remain unchanged. There is no claimed-score override, prebuilt executable or substituted result.
