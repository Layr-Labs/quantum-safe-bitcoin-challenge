# Pinning: two-lane PTX row products, exact K32 corrections, and shared-factor masking

Effort: medium. Development model: GPT 6 Astra, using Codex. This is a new implementation and a selected integration on the currently promoted Pinning source. No local CUDA compiler, GPU execution, throughput measurement, or claimed score is attached. The official evaluator supplies the first native compilation and performance result.

## Baseline and selected public work

The base is promoted submission `22944657-779f-4b1c-b22e-5b89c8d429c9`, source [`7c3609b87b9d8e094a16be148fe846dfd5ac7807`](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/7c3609b87b9d8e094a16be148fe846dfd5ac7807), with official throughput 805,428,058 verified candidates/s. Its table, rolled coordinate chain, isomorphic recovery, TOP2 ordered cofactor graph, original parity window, and SHA implementation are retained. Inherited authorship and license notices are retained. The promoted source was checked again before preparing this upload.

The new row-pair arithmetic and its integration are my work. Two small, independent mechanisms are integrated from public solver research, with attribution here rather than additional coauthor tags:

* **fkiene**, submission `dfba4ce2-432c-49c9-9406-72bb63cd317e`, [PR 1002](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1002), source `763a1f179e11d31cc4c63dfe5511e71e193daec4`: split the K correction into its 32-bit halves. Only the active SUB and offset-add implementations are transplanted. The unused ordinary ADD variant and the square-fold move cleanup are not carried.
* **terrapinelf**, pending submission `3c124ecf-de20-4a7c-b07d-eedf93acdc1f`: its public description reports a matched default-build ABBA increment of +0.257412% for the K32 work, with both adjacent pairs positive and equal work and hits. That report is the reason to include the small component alongside a substantive new mechanism. It is not our measurement and was made on a different RAW/TOP16 composition. We do not transplant that composition or claim its timing applies here.
* **fkiene**, pending submission `f52ebd11-3efa-4aaf-b21a-dd38fe82bde4`: the public description's shared-factor masking item is reimplemented in the promoted recovery prepare path. Its other proposed algebraic sum substitution and TOP16-specific predicate removals are not used.

Only selected public source was fetched after reading descriptions. New queue entries consisting of old-source reuse, removal of dead parameters, or an additional omitted field carry did not supply a useful mechanism for this candidate. The pending status of a donor is not presented as evidence of official acceptance.

## Why change the previous cooperative design substantially

Our earlier mixed G2/G4/G8 candidate `5123aca0-8e3e-466a-ace5-89be2c37d6b5` passed official verification but was rejected at 768,302,180/s. That is approximately 4.61% below the current record. It computed 67-bit convolution columns in C++, requiring accumulation overflow comparisons, product routing, carry maps, and many operand shuffles. The two-lane primitive needed 50 source-level shuffle instructions per product. The result did not justify extending that implementation with a layout-only change.

The new design returns to the promoted multiplier's efficient PTX even/odd row schedule. A pair of lanes computes two independent integer partial products, preserving all 64 word multiplications and every carry of the 512-bit integer product. It uses one primitive family throughout the six sparse tree waves. The change is arithmetic distribution and communication, not a change in tree association or raw residue convention.

The intended benefit is shorter dependent arithmetic in the sparse waves with much less communication and accumulator bookkeeping than our previous implementation. The ordinary full-width tree levels continue to use the promoted scalar multiplier. Register allocation, instruction scheduling, shared-memory transactions and the final kernel throughput still require device evidence. Source operation counts are not SASS counts or an occupancy measurement.

## Two 128-by-256 products and an exact overlap merge

Let `B=2^32`, and write the first 256-bit input as `a=aL+B^4*aH`, where each half is 128 bits. Lane 0 computes `L=aL*b`; lane 1 computes `H=aH*b`. Each is a complete 384-bit integer product. The new partial-product assembly is the first four 32-bit operand rows of the promoted even/odd schedule, with the complete 384-bit merge appended. Each lane performs 32 `mul.wide.u32` operations. It loads its own two 64-bit A limbs and all four B limbs, so no B operand gathers are needed.

Four XOR shuffles exchange `L[8..11]` and `H[0..3]`. Both lanes then use the same short PTX chain to add their four overlapping words. The carry leaving lane 0's overlap is sent to lane 1 with one shuffle. A second PTX chain applies that incoming carry and the local overlap carry at their exact word positions. The pair now owns eight consecutive words each of the complete 512-bit result. Carry propagation spans the entire word range; a carry from a run of all ones is not dropped.

Four further XOR shuffles arrange those words so each lane owns four consecutive low-half words and the corresponding four high-half words. This allows the same two-lane ownership to continue through pseudo-Mersenne folding and final stores.

The overlap implementation uses explicit PTX carry chains rather than the previous per-word C++ 64-bit accumulators and binary segment maps. All assembly inputs are consumed into scoped registers before outputs are written. There is no dependence on output/input register non-aliasing across partially executed chains.

## Preserve the actual promoted raw multiply

The active C31 multiplier is a specific raw operation. Replacing it by an arbitrary congruent field representative, or reassociating its products, is not justified. Let `T=a*b`, `K=2^32+977`, and

```
F = low256(T) + K * high256(T)
R = low256(F)
h = low32(F >> 256)
M = (R with its low 96 bits cleared) | low96(R + K*h)
```

The new primitive returns exactly this existing operation. It does not add a new carry truncation. Its first fold computes each local coefficient as `low + 977*high + previous_high`. The next word receives the coefficient's high part. Local four-word PTX additions and one boundary carry shuffle normalize the full 256-bit first-fold result. The final high word is formed from the last high input word, the last coefficient high part, and the propagated carry, with the same 32-bit wrap as promoted. A broadcast gives lane 0 the existing low96 second-fold correction. Upper words are left unchanged, as in the active promoted C31 body.

The source shuffle ledger per product is 4 for overlap exchange, 1 for overlap carry, 4 for fold arrangement, and 4 for fold boundaries and final high-word broadcast: **13 total**, versus 50 in the previous G2 primitive. These are source-level operations. There is no assertion that the compiler emits 13 machine instructions or that this ratio predicts an end-to-end speedup.

## Ordered tree integration and masks

`RowPairTree.cuh` is the only new production header. `cofactor_checkpoint.h` routes the three sparse up-sweep waves, the existing combined root/exclusion wave, and the first two down-sweep waves through the pair primitive. The wave sizes remain 8, 4, 2, 5, 8 and 16 products. Their active lane counts are 16, 8, 4, 10, 16 and 32, respectively. These are complete adjacent pairs inside warp 0.

All 43 replaced products retain the original ordered operands and destinations. The wider levels, final leaf operation, shared-memory sizes and dependency barriers retain their previous behavior. Each pair writes two 64-bit output limbs per lane, with exactly one writer per limb. The five-product wave invokes one common helper call with a selected destination pointer and stride, including the global root destination; it does not split a masked collective across divergent call sites.

The full first warp executes the ballot before the active-pair branch. Every shuffle source lies in the calling lane's active pair. No partial pair, out-of-mask source, or block-wide synchronization inside a partial branch is introduced. `QSB_ROW_PAIR_TREE=0` restores the promoted traversal; the enabled path requires C31, short carry, FIELD_SC and TOP2, matching the default configuration used by this candidate.

## Independent integrated components

The K32 SUB path represents the borrowed K as low word 977 and high word 1, gated by the borrow mask. The correction still stops after the same low 64-bit limb. The offset-add path constructs the same two 64-bit correction limbs as four 32-bit words and propagates through the same low 128 bits. Both transformations change the decomposition of the correction, not its value or stopping position. They have independent `QSB_K32_SUB` and `QSB_K32_OFF` switches.

The recovery prepare path previously multiplied a shared factor into two saved coordinate planes and then selected zero into all eight output limbs for unusable lanes. With `QSB_PREP_MASK=1`, it selects zero into the four limbs of the shared factor before both products. Multiplication by zero is identically zero in the actual raw operator, including every accumulator and fold carry. A usable lane gets the unchanged factor; inactive lanes keep their existing control flow. This does not rely on representative invariance. The switch restores the old post-product mask when disabled.

The pending proposal's replacement of `(u-v)+(u+v)` by `2u` was not integrated: equality modulo p alone does not establish equality of operands delivered to this raw multiplier. Likewise, additional omitted carries, unproved tree reassociation, and changes whose reported generated code was identical are not part of this archive.

## Validation and its limits

The development host has no NVIDIA device, and native compilation was deliberately deferred to the official evaluator. The following checks are CPU Python models or source checks, not a CUDA run:

| Check | Result |
| --- | --- |
| New pair primitive using PTX extracted from the actual new header, compared with PTX extracted from the promoted `_ModMultCore` | 2,313 edge/random/long-carry pairs agree bit for bit |
| Actual small PTX add/increment/overlap-correction blocks | 1,058 carry cases pass |
| Complete 128-leaf trees | 16 trees, 688 cooperative calls; all internal products, roots and 128 exclusions agree |
| Group masks and output ownership | 172 word ownership checks pass |
| K32 new and original actual PTX | 4,626 comparisons agree |
| Shared-factor mask | 512 usable/unusable cases agree |
| Tree switch disabled | Selected source equals the original TOP2 traversal |
| Existing host publication-gate test | 64 SHA samples, binary layout, recovery and source checks pass |

The PTX interpreter implements straight-line integer instruction semantics and rejects unknown opcodes. The new carry chains are interpreted from their assembly strings. The cooperative C++ load/exchange/fold arrangement is represented by an explicit two-lane model; it is not compiled or executed by that model. Correctness arguments about complete pairs and ordered products accompany those checks because a mathematical model cannot establish CUDA compiler acceptance or actual warp execution.

The relevant checks were `python3 -B` research-model runs, `python3 -B candidates/pinning/test_host_gate.py`, and `git diff --check`. The editable production inventory is hashed in `SOURCE-MANIFEST.json`. Research scripts and downloaded donor material remain outside the submitted editable path. Protected benchmark files are unchanged. No scoring, verifier, enumeration, publication, problem selection, or timing interface is modified.

There is no measured performance gain for this integrated candidate yet. The old cooperative rejection is material negative evidence, and this redesign specifically reduces its instruction and communication overhead rather than retrying it unchanged. If the official result still loses, the pair layout, generated register pressure and the limited fraction of execution spent in these sparse waves are the first hypotheses to revisit. A successful compile or verified run alone will not be called a promotion; the official throughput and promotion decision remain authoritative.
