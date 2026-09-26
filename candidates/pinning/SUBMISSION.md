# Pinning: signed 13-window host recovery with folded endpoints and the completed predicated-gather GPU path

Effort: medium. Prepared with GPT 6 Astra in Codex. This is a new integration based on the latest promoted Pinning source, with a substantial host algorithm change and selected device changes from a completed positive-result submission. No local C++/CUDA compilation, native SIMD execution or GPU performance measurement has been performed. The official remote evaluator must establish the build, correctness and throughput of this exact combination. No claimed score is provided.

## Starting point and provenance

The promoted Pinning baseline is fkiene's `ff524fd9-0652-419c-9ae1-9852b6d1b587`, landed as `cc75e3b8cb3f09a5ca78fe18c36e75ef0ff44b1f`, with 960,830,125 verified candidates/s. The protected harness snapshot is shared tip `a137e289b236c3622eba80f1ad5e9a0c8a91eb67`; its Pinning executable source was byte-identical to that promotion. Only `candidates/pinning` is editable or uploaded. The benchmark manifest, scoring, problem generation, exact verifier and commands are unchanged.

This work continues the independent host integration prepared in `c6c4bd64-0cd7-4268-88a9-d3bdd0bc4359`. That earlier tree is frozen and is not modified by this candidate. Its official evaluation has now finished: verified correctness, 934,938,743 candidates/s, 133,858 verified hits in 1201.0223 seconds on RTX 4090 (seed 1472433187), rejected, source fb7a053a5511455d4657aaef3f2bd704c8979ffa. That is about 2.695% below the promoted score. It does not establish a positive contribution from the earlier CPU changes, and the public metrics do not identify the selected CPU backend or isolate a cause. This successor adds both a lower-work host algorithm and the completed positive GPU source, with a revised contribution-aware CPU controller described below. Its retained primitives are the promoted Subset IFMA52 field arithmetic and SHA-NI x4 compressor, a Pinning-specific adapter with two alternating Montgomery chains, and the completed Subset safegcd inverse core. The mathematical change in this candidate is not a repeat of that earlier archive.

The cross-track arithmetic sources are promoted Subset `9f8a33d8-0033-4e84-9ca4-04e8508d0088`, source `a137e289b236c3622eba80f1ad5e9a0c8a91eb67`, by ercumentyildirim, and completed Subset `97f347a8-5224-454b-bd99-dc3190c4bc40`, source `aef1aef96eb6644fcc5ca86146dca27658b48aa6`, by terrapinelf. The former supplies the field, fused operations, transpose and SHA primitives; the latter supplies the selected 62-divstep scalar inverse. Their libsecp256k1/Pieter Wuille and Meganpark980320 lineage is credited and the existing license files are retained.

Signed variable-width host windows and precomputation of a recovery constant into the final table are informed by the promoted Subset implementation. Current public descriptions, including fkiene's Subset `413f83e7` and terrapinelf's canceled Pinning `c05f58ec`, support exploring memory-sized tables, but their broader code and claimed performance are not imported. No pending donor source was read. Those descriptions also expose tradeoffs: a larger table can improve CPU candidates per second while raising memory traffic and startup costs. The implementation here is an independent Pinning adapter with complete exceptional-point handling.

For the device path, the selected source is the completed, officially verified Pinning `8f2ea1b3-6f32-4b62-8f51-fa163b11cda2`, by terrapinelf, commit `afe81b02e68d3c2aeeef05617fd195b3e6674601`. Its official score was 967,108,331/s, approximately 0.653% above the promoted baseline, but it was rejected because promotion requires 1%. Its note attributes the complementary predicated cache-policy gathers and phi-hoisted loop to dukemawex. The complete matching native carrier is retained with those device-source changes. This is a measured whole-source result from another submission, not an isolated proof of a device gain or a predicted additive gain for this integration.

## Mathematical host change

The host recovers `Q0 = z*B + A` and `Q1 = z*B - A`. The previous host path expresses the 256-bit scalar as sixteen unsigned 16-bit digits, performs fifteen ordinary additions after initialization, and then performs two final additions sharing one denominator. The new optional IFMA path expresses the same integer exactly with thirteen digits covering 257 bits: ten windows of 20 bits followed by three windows of 19 bits. The final digit is unsigned and accommodates the last carry.

For each lower window, let the extracted value plus incoming carry be `d` and the radix be `R = 2^w`. When `d > R/2`, the digit is `d-R` and the next carry is one; otherwise the digit is `d` and the next carry is zero. A negative zero is normalized to zero. The last field ends at bit 257, so the recoding exactly reconstructs every 256-bit input, including values at or above the curve order. This route does not omit order reduction on a probabilistic assumption: group operations naturally give the same `z*B` for the full input integer.

Lower tables contain magnitudes from one through half the radix, with an explicit infinity row at index zero. The final two banks contain `d*B_last + A` and `d*B_last - A`, including both values at digit zero. They replace the ordinary final window and the later standalone additions of the recovery constant. The table occupies 402,654,080 logical bytes, approximately 384 MiB plus its explicit zero rows. The original 64 MiB table remains available for old backends.

For a dense candidate, counting source-level field operations and excluding the amortized horizontal root machinery:

| Item | Prior unsigned IFMA path | Signed folded path |
|---|---:|---:|
| Mixed affine point additions across both outputs | 17 | 13 |
| Field multiplications | 82 | 65 |
| Field squares | 17 | 13 |
| Batch-root stages | 16 | 13 |
| Logical table rows per candidate | 16 | 14 |

The two recovery outputs are processed sequentially and reuse the same compact state. Concatenating their denominator arrays would save one scalar root stage but expand the per-worker field state. This implementation chooses thirteen root stages to retain the smaller state. The initial twelve-stage research cost model described the concatenated alternative and is not the final implementation count.

A lower-window forward pass combines the digit sign with the ordinate subtraction, so it avoids materializing a negated table ordinate for normal additions. Only initialization or an infinity-to-point load needs an explicit conditional negation. The backward pass retains the fused formulas `x3 = lambda^2 - D - 2X` and `y3 = lambda*(X-x3) - Y`, and prefetches the next window's rows in consumption order. The underlying field primitive bodies are unchanged from the previous integration.

## Exceptional points and complete retry

Zero digits and inactive padding lanes contribute the multiplicative identity to the denominator product. They preserve the point-at-infinity state. Equal-x additions are detected through the already computed canonical zero mask of the combined inverse roots. Over the prime field, a zero product means an active denominator was zero; the new backend retries the complete batch through OpenSSL before publishing any endpoint from that stage.

The final folded tables can themselves contain infinity. Their builder handles equal-x inputs with a complete doubling-or-infinity case and records whether each bank has any infinity rows. Normal tables therefore do not add eight scalar coordinate rereads per lookup; the explicit infinity-row test is needed only for a flagged bank. An infinite accumulator or folded row also selects complete recovery. Both recovery branches use the original scalar and recovery point. If recid zero has already completed and recid one later needs retry, only recid one is retried. This prevents duplicate publication.

The complete path computes the full OpenSSL group operation and hashes every finite compressed point, with the same difficulty predicate. Every tentative hit still passes the unchanged exact host publication gate before it reaches the existing hit file. No exceptional candidate is silently dropped by the new backend, and no carry, hash condition, point check or verification rule has been relaxed.

## Table construction and runtime choice

The existing scalar batched table builder is extended with an explicit row count. It includes the half-radix magnitude, including the final partial block, rather than applying the old power-of-two-exclusive upper bound. The ordinary 16-bit call retains its prior row count. Signed window bases are generated from the same problem-specific `neg_r_inv` coefficient via OpenSSL. The final table is built once in place: a block saves its original base coordinates before writing either folded output bank, and both branches share that block's denominator inverse.

The optional allocation is attempted only on an IFMA-capable CPU when automatic backend selection is enabled and available host/cgroup memory leaves at least the table size plus 512 MiB and two MiB per worker. Allocation failure keeps the old table and old backends. The allocation requests 2 MiB alignment and huge-page advice; that is a request, not a claim that the OS granted huge pages. Builders run in the existing background helper at idle priority, while GPU work proceeds. The CPU-ready flag is published after all builders join.

Worker zero first compares complete coordinates for 41 dense scalars against OpenSSL, including a partial vector, followed by zero, one-bit and all-ones cases at several batch lengths. Only a passing new backend enters the existing productive startup timing alongside the old IFMA52, AVX-512F and AVX2 alternatives. Those native checks are implemented but have not been executed locally. The fastest measured backend is selected on the evaluator's actual CPU. A larger table can lose on cache, TLB, memory bandwidth, startup time or CPU frequency; it is not forced merely because its abstract operation count is smaller. The existing CPU-share check remains active. The GPU-contention controller is revised as follows.

## Contribution-aware CPU control after the negative result

The prior controller only shed workers after GPU interval loss exceeded both the CPU contribution and a fixed 1.5% threshold. A CPU lane adding, for example, 0.5% could therefore remain active while costing the GPU 1.0%. That condition is unfavorable to total candidates even though it was below the fixed threshold. The previous result motivates addressing this concrete control-policy problem; it does not prove that the policy caused the observed 2.695% regression.

This version compares four two-second windows in the order CPU on, off, off, on. Each window contributes one mean GPU batch interval with equal weight, and the first straddling batch is excluded. This cancels first-order linear interval drift in the synthetic model. CPU candidate rate uses only the two on windows, avoiding dilution by the off intervals. A net loss above the measured CPU contribution plus a 0.3 percentage-point tolerance, seen in two consecutive ABBA trials, sheds one quarter of the workers. A good trial clears the strike. The existing reserve, idle priority, stop mechanism and 60-second steady monitoring cadence remain.

If GPU_on is the GPU candidate rate while CPU workers are enabled, then total throughput improves when CPU_rate/GPU_on exceeds the relative GPU batch-interval loss. The comparison uses that ratio directly. The narrower tolerance is paired with longer drift-balanced windows and two-trial confirmation. Pure-Python checks covered 150 synthetic drift/contribution cases, on-only CPU accounting and worker transitions for every count from one through 256. Those checks establish the bookkeeping, not native tuning. Nonlinear thermal behavior and measurement noise can still affect the choice; this is not a proof that the prior score regression is removed.

## Device portion

The selected completed source changes the prepare loop's table loads from a lane-selected cache-policy descriptor to complementary predicated loads with constant hot and cold descriptors. Each lane issues one load in each pair, to the same address with the same cache policy. The first Y-half load retains its 64-byte L2 fetch hint. The endomorphism multiply moves between two passes of the same trip loop and remains immediately before the same term. These changes reduce repeated policy setup and loop-body work without changing the point-add formulas or table decoding.

That completed source also chooses a smaller whole-tree-aligned batch when available device memory cannot accommodate all state slots plus a reserve. The kernel receives the batch length as a runtime parameter. The supported sizes preserve the tree and thread divisibility constraints. This allocation guard does not alter the accepted candidate domain or the hit predicate.

The matching native sm_89 carrier is 304,032 decoded bytes with SHA-256 `9aafe9d7f1eed411e3b68108c6416d55f740fea53399bcfb354dd535c3135d27`. The device source and runtime include closure are checked against that completed commit. The host-only signed backend does not require rebuilding this carrier. A source-only device edit paired with the old promoted image would have been ineffective; the source/image pair is kept together.

## Validation and limitations

Local checks are pure Python and source inspection. They do not establish that the new C++ compiles, that native IFMA lowering is correct, or that total verified throughput improves.

- Exact scalar recoding: 12,415 directed and random inputs, including carries, zero digits, powers of two, curve-order and field-prime boundaries, and all-ones input.
- Complete group-law formulation: 101,376 two-recid endpoint checks on a small prime-order curve, with 438 folded-infinity, 1,584 accumulator-infinity and 1,004 equal-x observations; 64 full-width secp256k1 endpoint comparisons.
- Integrated prefix/reverse model: 168,696 endpoint comparisons across 4,752 batches, including partial vectors, 2,034 regular batches, 423 zero-root retries, and 2,295 infinity retries. Recid-one-only retries occur in 193 batches; output keys are checked for both completeness and absence of duplicates.
- Builder/index bounds: 851,967 nonzero row indices across the old and both new row counts; all parallel window partitions cover each window exactly once. Folded-table construction checks 1,584 rows and sixteen equal-x cases.
- The existing selected field, inverse and SHA primitives retain their previously checked bodies. Protected harness files and the exact publication gate are unchanged.

The portable new check is `python3 candidates/pinning/host_research/check_fold.py`. Existing primitive checks remain in the same directory. The actual ranked build and execution stay `./setup.sh pinning` and `./benchmark.sh pinning`; neither is run locally here. Historical `test_host_gate.py` contains a stale macro-count expectation that already disagrees with the promoted baseline; it is retained unchanged and is not presented as a passing full test.

The source-level 82-to-65 multiplication reduction concerns the CPU curve stage. It is not a 20% improvement to the whole benchmark. The CPU contribution, IFMA availability, worker count, memory pressure, GPU clocks and score sampling noise all affect the final result. The completed GPU donor's score likewise cannot be added arithmetically to a predicted CPU gain. The next decisive observation is the official combined verified-hit result on a fresh problem.

Credit is carried in this description rather than coauthor metadata, as requested by the submitting user. Credits: fkiene and the promoted Pinning lineage; ercumentyildirim, terrapinelf, Meganpark980320 and libsecp256k1/Pieter Wuille for the selected host arithmetic lineage; dukemawex and terrapinelf for the completed predicated-gather/phi source and matching carrier. The independent contribution is the Pinning signed-window layout, folded table builder, compact batch adapter, complete exceptional retry, integration and reproducible mathematical checks.
