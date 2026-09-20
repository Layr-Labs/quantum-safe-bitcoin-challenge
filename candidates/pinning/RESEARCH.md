# Pinning: interleaved two-tree collective with retained operands and in-place exclusions

Effort: medium. This work was performed with GPT 6 Astra using Codex. The starting point is the latest promoted Pinning submission dcd0147c-8cb3-47f0-8b71-007c87fa7748, landed commit 66fede0cc10d36ad15041861d3eddaad97481ac6, official score 789,011,576 verified candidates per second. This is an unmeasured architectural integration for official remote validation, not a claimed measured speedup.

## Problem and hypothesis

The promoted prepare kernel launches one 128-thread block per candidate cofactor tree. Near the tree root, only a handful of lanes execute a full 256-bit multiplication, while the operation still issues as a warp instruction stream. Each next level also reloads data that a surviving thread just computed. The product and exclusion trees occupy distinct portions of a 12 KiB arena, although their live intervals allow carefully permuted reuse.

This candidate combines three mechanisms: two independent logical trees share a physical 256-thread block; each thread retains its previous result for subsequent same-thread operands; and the down-sweep overwrites consumed sibling slots rather than allocating a separate exclusion plane. The combination needs a new physical-thread mapping so that grouping does not destroy the same-thread identity on which retained operands depend.

The physical prepare thread t handles logical candidate

```
blockIdx.x * 256 + (t & 1) * 128 + (t >> 1)
```

Thus group g=t%2 and within-tree position i=t/2 remain attached to the same physical thread as each active set shrinks or grows. The logical tree width and all finish/root-pipeline indices remain 128. This does not combine the mathematical products of separate trees or alter which candidates share an inverse.

## Exact dataflow and storage invariant

The arena stores level nodes in node-major, group-interleaved order. A logical node i of group g at level base offset resides at `(offset+i)*G+g`. The up-sweep thread retains the left operand from its preceding output, starting from its own input leaf. Its right operand is still loaded from the arena, and its output is still stored for other threads to consume. Both the value and operand order are identical to the promoted operation.

At the combined root/first-exclusion stage, ordinary shared reads are retained. This stage seeds the downward carry for the first four logical lanes of each group. At later downward levels, old active threads reuse their previous result; newly active threads read the parent exclusion from shared memory. The leaf stage applies the same rule to the lower and upper halves. We retain all publication fences appropriate to the combined group population.

For in-place storage, exclusion E_C[i] at width C lives at the former product slot `i XOR (C/2)`. The lane reads precisely that sibling slot and then overwrites it with E_C[i]. No other thread reads that same sibling slot in the same phase. Parent exclusions live in a different, previously completed level. Consequently the reuse introduces no read-after-write race with another lane and requires no additional barrier between each thread's own read and write. Naively writing E_C[i] into slot i would violate this property; the earlier audit retained that as a negative control.

The register carry does not reassociate products, commute operands, duplicate products, replace multiplication by a modular identity, or introduce a different residue representative. Every output is intended to match the promoted raw operation tree bit for bit, including the promoted implementation's existing arithmetic approximations.

## Arithmetic boundary

GPUMath.h is byte-identical to the promoted source. This base already enables C31/carry62 and multiply-tail approximations, and it independently verifies published hits on the host. The new code does not add any omitted carries, square-tail truncation, skipped normalization, changed exception rule, or modified host gate. Claims here are about preserving that existing operation graph, not about proving the inherited multiplier is a complete modular arithmetic implementation.

This distinction matters. A recent public note proposed carrying a raw pre-add product through a parity multiply on the assumption that changing its representative cannot matter. Under the promoted truncated multiply, that generic implication is false: the actual PTX gives a nonzero residue for m(p-1,p), whereas m(p-1,0)=0. That is a counterexample to the generic proof, not a demonstrated reachable full recovery input. No such normalization removal is incorporated here.

## Complete cost ledger

These are source/dataflow counts, not SASS measurements. Compare the same 512 logical candidates:

| Quantity | promoted four 128-thread blocks | new two 256-thread blocks |
| --- | ---: | ---: |
| scalar tree multiplications | 1516 | 1516 |
| warp multiplication issue groups | 68 | 54 |
| warp shared field-read issue groups | 136 | 74 |
| scalar shared field reads | 3032 | 2032 |
| shared arena per prepare CTA | 12288 bytes | 16384 bytes |
| shared arena at intended 512 resident threads | 49152 bytes | 32768 bytes |

A field read contains four 64-bit limbs. Initial leaf stores, internal publication stores and output stores are not counted as reads. The read ledger includes center and leaf operands. The modeled 64-bit/16-lane shared transaction layout has no bank conflict for the new interleaved reads. This is a conservative address model, not a hardware timing report.

The multiplication reduction is 20.59% of this tree's warp issue count, not 20.59% of the whole grinder. Thirteen deferred mixed additions alone account for 1456 warp multiplications per 512 candidates, before squares, seed construction, SHA, table traffic and recovery. Four-tree grouping would reduce the tree issue count further to 49, but the 512-thread CTA is a more substantial scheduling change. Two-tree grouping was selected together with the retained-operand and in-place mechanisms, rather than submitting the earlier small standalone grouping prototype.

Intended resident threads remain 512: previously four 128-thread CTAs, now two 256-thread CTAs. Launch bounds remain a constraint, not a measured register allocation. The change may still lose through register live ranges, block scheduling, instruction selection or altered memory locality. No claim of zero spills, a particular register count, actual occupancy or a predicted score is made.

## Production integration

Only pinning.cu, PackedRecovery.cuh and the new InterleavedCarry.cuh change runtime behavior.

- Prepare launch geometry becomes 256 threads with a minimum-two-block launch-bound target. Logical tree width and finish geometry stay128.
- The digit decoder retains the promoted arithmetic and volatile shared placement. Its plane stride becomes256 physical lanes. Fifteen planes require15360bytes and fit inside the16384-byte tree arena.
- The old unused prepare-scratch argument and allocation function are removed explicitly. The code does not rely on compiler dead-allocation elimination to meet the arena budget.
- Stage0 computes the permuted logical candidate index, and packed state writes use the identical index. Stage1 keeps its original contiguous mapping.
- SHA tail construction uses the same candidate locktime. With the existing256-aligned start, the high bytes equal `(start_lt>>8)+blockIdx.x`; the low byte is `128*(t&1)+(t>>1)`. The high bytes are now uniform over the full256-thread block.
- Root writes use `2*blockIdx.x+group` with a guard against `ceil(batch_size/128)`. Missing groups contribute identity leaves but cannot write beyond the logical root allocation. Active candidate state writes retain their original guard.
- The root-group inversion kernels, weighted root products, finish arithmetic, hash implementation and host verification/publication paths retain their promoted bodies. Optional incompatible diagnostic geometries are rejected by explicit static assertions.

Seven other runtime files retain their promoted hashes, including GPUMath, GPUHash, LeafRecovery, RecoveryConstant, the original cofactor header, and both SHA headers. A reversible bounded change set checks closure of the two edited existing runtime files against packaged promoted snapshots.

## Validation performed

No local C++/CUDA compilation and no local GPU benchmark were performed. The official remote runner is the native compilation, correctness and performance test. Python checks were run with bytecode generation disabled:

```
python3 -B candidates/pinning/research/audit_tree.py
python3 -B candidates/pinning/research/audit_integration.py
```

Results:

- 768 symbolic leaf checks using noncommutative expression hashes preserve operand order and association.
- 50,688 raw-bit leaf comparisons cover random inputs, zeros, p-1/p/p+1, all-ones values and partially filled logical groups.
- 256 leaf results are checked using an interpreter for the actual current multiply PTX against the independent raw model.
- Producer/consumer edge checks verify that each shared value is either block-published or used within its producer warp. Same-phase overwrite checks require every overwritten sibling to have exactly its own writing thread as reader.
- 27,648 SHA index/wrap checks cover aligned starts, wraparound and partial batches. The old128-thread index formula is a negative control and disagrees in27,496 of those checks.
- All15 digit planes fit the16KiB arena. Candidate permutations are bijective, active counts match, root guards match logical tree existence, and finish reads address the corresponding prepared logical candidate.
- Eight full-batch warp store comparisons preserve the number of touched32-byte sectors for each16-byte state plane, although sector count alone does not establish equal measured memory throughput.
- Source closure and seven untouched runtime hashes pass. Token bindings cover the new arena/index expressions and the helper's operand-storage mappings; this is targeted static auditing, not a full C++ interpreter.

## Prior results and attribution

Our prior PR764 was a different register-top16 plus decoder integration. It scored775669591 with verified=true, below this promoted baseline. That composite is not reused. The earlier complete/reassociated top16 paths are also not present.

The retained-operand mechanism is independently implemented from fkiene's public in-flight description db0e237. The note's up/down/leaf carry idea materially contributed to this submission, so fkiene is credited as coauthor. We did not restore its unpromoted cbce550 base or copy its additional arithmetic approximations. The interleaved mapping, in-place exclusion permutation, geometry integration and audits were developed independently here. The promoted source's earlier authorship and license are retained.

Other public in-flight descriptions were screened. New square truncations, unsupported register-count claims, whole-array digit placement, generic representative-invariance claims and inert remeasurement changes were not included. The latest promoted baseline and absence of an own in-flight submission were checked before upload. Discussions are disabled for this benchmark.

This submission requests the first official measurement of the combined architecture. If it regresses, the result should be recorded as evidence against this whole combination; it would not by itself isolate which of grouping, register retention, memory permutation or CTA scheduling caused the regression.
