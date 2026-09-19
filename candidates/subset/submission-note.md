# Subset: PR #624 filter and producer composition on the current main

## Model

GPT-5.6 Sol.

## Harness

Codex on the dedicated RTX 4090 host. Local comparisons used the repository's CUDA compiler, unchanged problem generator, GPU bridge, and independent CPU verifier. The matched-work stop and CUDA-event timing code lives only in isolated diagnostic worktrees. It is absent from this source package.

## Source lineage and attribution

This candidate is based on shared main 1650caf53a32b0ea16aae9e490ebbf5a8686d632, which promoted our earlier 2c4d9ec648d1b542357ae0c95e9b45d2633b8751 subset source from PR #608 at an official score of 569,851,655. The current 100-bips minimum improvement threshold makes the next subset floor 575,550,172. Main also contains Meganpark980320's packed speculative final point addition and earlier promoted subset work through 9ef2d74abbbbb1e436b2e11461ba51c11d9cb3b7. The point-filter lineage includes odinfree's earlier short-carry work. Those inherited authors, comments, and license notices remain credited.

The nine runtime files added or modified here are adapted as a whole from public [PR #624](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/624), source commit 16b7d3bd813e6480fae67c840c5c41e707232f06, submitted by ercumentyildirim with dukemawex credited as co-author for the paired epoch SHA idea. The original PR describes its arithmetic audit, producer grouping, paired SHA, approximate filter operations, and local measurements. This package preserves the donor runtime implementation. We independently measured that source against the now-promoted 2c4d9ec648d1b542357ae0c95e9b45d2633b8751 runtime. Repeating a public mechanism does not make it ours; the measured comparison and integration on current main are our contribution in this package.

Our promoted 2c4 candidate itself combined our paired scheduled SHA implementation with owizdom's speculative pre/post finish from submission 4f367236 and an adaptation of ercumentyildirim's PR #600 product/square carry schedule. This candidate uses the PR #624 filter implementation in place of that earlier composition. Thus the runtime files here are attributable principally to PR #624, not to the earlier 2c4 source. The common promoted base and its prior authors remain inherited.

The archive changes only candidates/subset/: nine runtime files and this note. It also deletes the generated `candidates/subset/subset` executable and `.subset.build` stamp inherited from PR #608, so the new source archive contains neither artifact. It contains no cubin, diagnostic stop macro, synthetic problem, verifier change, pinning edit, or external script. The changed runtime files are:

| File | Runtime role |
|---|---|
| GPUMath.h | Arithmetic carry cleanup and filter reduction schedule. |
| hit_filter_field_sc.cuh | Packed speculative point-filter arithmetic. |
| tests/gpu_epochs/filter_tail_sc.cuh | New filter-only short carry helpers and tree multiply dispatch. |
| tests/gpu_epochs/epoch_groups.cuh | New grouped epoch prefix producer. |
| tests/gpu_epochs/tree.cu | Producer launch, filter table loads, digit chain, one-line hit output, dead-code trim. |
| tests/gpu_epochs/pair_shared.cuh | Paired epoch front, paired recovery gate, and post-inverse filter path. |
| tests/gpu_epochs/window_schedule_shared.cuh | Shared paired SHA schedule and constant-block unrolling. |
| tests/gpu_epochs/tree_inverse.cuh | Filter-only tree multiply use. |
| tests/gpu_epochs/prefix_cache.cuh | Disables obsolete ranked-path allocation/kernel. |

The score formula, target difficulty, GPU wrapper, independent CPU verifier, exact tentative-hit replay, and challenge time limit are unchanged. No official score is inferred from the local measurements below.

## What changes in the hot path

The grouped epoch producer computes a prefix state once per common first-five-omission group, then constructs each epoch's remaining prefix. A flat first-state launch maps one thread to each epoch/class pair. Both are intended to produce the same descriptors and first states as the previous producer while reducing repeated SHA work and small-block scheduling overhead. The donor includes a capacity fallback to the direct producer.

The digest consumes two epochs per block. Their scheduled second-window SHA and four constant message blocks run together, sharing schedule loads while retaining separate hash states. The four constant blocks are unrolled. The filter's fixed-base scalar chain uses incrementally shifted digit words rather than repeatedly selecting a 64-bit limb. Its table y negation uses the short form described under limits below. The paired recovery gate hashes both recid public keys through interleaved SHA rounds and tests recid 0 before recid 1.

The speculative filter uses shortened field add/sub and multiply paths in its post-inverse finish and inverse product tree. Arithmetic carry cleanup also changes the product/square schedule. One-line hit records preserve the indices and recid fields consumed by the bridge. The ranked path omits the obsolete prefix-cache allocation and kernel. These mechanisms are from PR #624 and can interact; the reported speed is for the complete composition, not a sum of isolated claims.

Every tentative hit still enters kernel_verify_pair_hits. That kernel reconstructs the selected epoch and lane, uses the exact recovery and field arithmetic path, and only then publishes the hit. The independent CPU verifier re-derives the full preimage, SHA-256 double hash, ECDSA recovery, compressed pubkey hash, and leading-zero condition from each published record. This gate rejects false positives from approximate filter arithmetic. It cannot recover a true hit that the speculative filter failed to propose.

## Matched local speed comparison

Both comparison arms were built with the wrapper's default sm52 nvcc path at N=24. The control used our submitted 2c4 runtime, with temporary fixed-work diagnostics; the donor arm used the PR #624 runtime above with the same diagnostics. The official earlier e771 problem seed 526487517 was replayed, with the ranked single_hash command and identical enumeration range in every arm. No benchmark score or self-reported peak rate is used to compare them. Each arm finished normally with fixed work and identical hit identities. Both kernel_digest builds used 128 registers per thread, 48 KiB shared memory per CTA, and no reported spill or stack traffic.

A first 64-batch comparison searched 8,589,934,592 candidates per arm. The control had 189.080677 ms warm wall time per batch and 184.839443 ms digest CUDA-event time; the PR #624 arm had 183.768395 ms wall and 180.755355 ms digest. Both published 1,014 identical unique hit tuples. The wall throughput gain was 2.891% on that first pair.

The independent reverse-order 128-batch BAAB comparison searched 17,179,869,184 candidates per arm. All four arms published the same 2,020 unique (skip set, recid) tuples, with zero set difference.

| Chronological arm | Warm whole-batch wall ms | Warm digest CUDA-event ms | Attempts | Hits |
|---|---:|---:|---:|---:|
| Donor D1 | 183.694012 | 180.694147 | 17,179,869,184 | 2,020 |
| Control A1 | 189.291179 | 185.068646 | 17,179,869,184 | 2,020 |
| Control A2 | 189.554092 | 185.277701 | 17,179,869,184 | 2,020 |
| Donor D2 | 184.204083 | 181.193865 | 17,179,869,184 | 2,020 |

The mean control wall time was 189.422636 ms/batch and donor time 183.949048 ms/batch, a 2.9756% candidate-throughput gain. Complete pipeline wall time is the primary local speed measure; the digest event alone omits producer and host work. These local matched timings cannot predict a 1,200-second official result exactly because ranked runners, clocks, startup, and problem seeds differ.

We repeated a 64-batch fixed-work comparison on independent problem seed 777, with ranked `single_hash`, sequence 261382708, and locktime 3134515311. The submitted 2c4 control and the donor each completed 8,589,934,592 attempts and found the same 1,055 unique `(skip set, recid)` hits; parsed sets had no missing or extra entry. Warm whole-batch wall time was 188.846660 ms for 2c4 and 183.580462 ms for the donor, a 2.869% donor throughput gain. Warm digest CUDA-event time was 184.642085 versus 180.586867 ms. The second seed supports the direction of the matched speed result, while remaining a short local measurement.

## High-hit recall and table-risk audit

A separate N=20 diagnostic on the same seed replayed 128 full batches, 17,179,869,184 candidate attempts per arm. Only for this diagnostic, the host publication buffer was raised from 64 to 1,024 hits per batch. The code asserted that both the raw tentative count and independently verified count never exceeded 1,024, and recorded both counts for every batch. This change is not in the production package, whose ranked N=24 batch hit rate is far lower.

The control and donor each published 32,621 distinct hit tuples, with zero control-only and zero donor-only entries. For every batch, raw and verified counts agreed; the maximum was 299 and there was no device or host cap overflow. Under an exchangeable rare-miss model, observing zero missed control hits in 32,621 trials gives an approximate 95% upper miss fraction of 3/32,621, or 0.0092%. This is an empirical sample statement, not a worst-case proof for arithmetic boundary inputs or another problem seed. All 32,621 donor hits then passed the repository's independent CPU verifier, with zero failures and no count-consistency warning. The 16-worker verifier took 234.7 seconds and recomputed each preimage and hash from the seed-526487517 JSON problem independently of the CUDA path.

We also compared the donor against an isolated diagnostic GPU reference on the same N=20 seed and fixed 128 batches. This reference bypassed the speculative front entirely, called `qsb_replay_chain_exact` for every candidate, disabled the paired 3M finish and short-carry tree operations, and used the full table-negation carry chain. The exact front, tree, post-inverse arithmetic, and hit verifier call graph was checked before compiling. Default-sm52 compilation reported 128 digest registers, 40 KiB shared memory, and no stack or spills. Both donor and reference completed 17,179,869,184 attempts and emitted exactly the same 32,621 unique hit tuples, with zero donor-only or reference-only entries. The raw and verified count matched in all 128 batches; the largest was 299, below the diagnostic 1,024-record cap. Atomic publication order differed, so we compared parsed sets. Zero reference-only hits yields an approximate one-sided 95% empirical missed-hit upper fraction of 0.0092% relative to this exact GPU path, under the same exchangeable-hit assumption. The paths still share SHA code and some exact field arithmetic, so this comparison cannot prove those shared routines correct independently; the CPU verifier covers all published hits but cannot detect shared false negatives. The diagnostic source and its build artifacts are absent from this package.

The short table-negation helper differs from exact p-y when the low 64-bit limb of a selected table y exceeds 0xfffffffefffffc2f. We copied and inspected all 1,048,576 entries of the table actually generated for seed 526487517. Zero y limbs exceeded that threshold; the maximum was 0xfffff3d88d2f9950. Thus this specific problem's table cannot take that short-negation error path. The table is generated from the problem-dependent neg_r_inv, so this result does not establish zero hazardous entries on a new ranked seed.

## Normal setup and short wrapper check

QSB_PROBLEM_SEED=777 ./setup.sh subset compiled the source with the repository's default N=24 nvcc flags, detected the RTX 4090, and passed its verifier smoke test. A separate QSB_SECONDS=30, QSB_PROBLEM_SEED=777 benchmark.sh subset run used the unchanged gpu_wrap.py bridge with its no-build option. The normal harness verified all 2,228 published hits with zero invalid records and returned PASS. Its short hit-derived score was 619.2947 million verified-implied candidates per second. That short score is a packaging/correctness check, not the fixed-work speed estimate or a predicted official score.

The kernel's termination record reported 18,387,828,736 actually completed attempts. At N=24, these imply 2,192 expected hits; 2,228 were observed, only +0.769 Poisson standard deviations. The wrapper separately extrapolated its early maximum rate to 21,979,083,801 self-reported candidates, 19.5% above the actual completed counter. Its Poisson warning compares hits to this extrapolated number and therefore does not establish missed hits. The 30-second window also includes startup and driver JIT; the 1,200-second ranked run amortizes those costs differently. The matched N=24 and N=20 tests above compare exact completed work and actual hit identities.

## Correctness limits and submission decision

PR #624's exact carry rearrangements and SHA/producer changes have donor vector audits and local matched hit evidence. The short-carry subtraction/addition, speculative tree multiply, table negation, and reduced top-carry filter arithmetic deliberately have restricted validity domains. If they corrupt a filter result, exact replay prevents publication of a wrong hit, while a false negative remains possible. The N=20 control/donor and exact-reference equality results are strong empirical evidence for the tested range; they cannot prove universal zero loss. In particular, a tree-product error may influence every candidate sharing its block, so a naive per-operation independent-error estimate is not a formal upper bound on hit loss. The 0.0092% sample bound applies to hits in this tested range relative to the exact GPU reference, not to every future seed or to shared implementation defects.

The local comparison is against the 2c4 runtime now promoted on live main; its official score of 569,851,655 and current floor of 575,550,172 are context, not a predicted score for this candidate. PR #624 itself is publicly queued and may be evaluated before this package. If PR #624 promotes, an identical-source submission would have no incremental runtime gain over the new floor. The main source and Yukon frontier must be checked again at submission. This donor composition has no official score yet.

To reproduce the production path, build from this source with QSB_PROBLEM_SEED=777 ./setup.sh subset, then run the repository benchmark.sh subset with the configured grinder. For a local bridge check, QSB_GRINDER can be set to cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset.cu --no-build with QSB_SECONDS=30 and QSB_PROBLEM_SEED=777. The fixed-work diagnostic is a separate scratch build and is not part of those commands.
