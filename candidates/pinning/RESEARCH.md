# Pinning: bounded second-fold fusion and packed tree-top arithmetic

Effort: medium. This candidate was developed with GPT 6 Astra in Codex. The base is the promoted Pinning submission `208bbcb6-e235-47a1-b8c5-1d03874ac237`, score 766,671,138, landed at `03e399c9c27037a4acea573a79444a902a5ace91`. The shared branch's later `043b650` tip did not change the Pinning frontier. This is an independently implemented combination developed from public submission descriptions, with source-level proofs and Python PTX semantic checks. There is no local CUDA compilation, GPU execution, ptxas report, or measured speedup for this combination. The official remote run is the first native validation.

## Route selection and attribution

The previous independent wide-window experiment, PR678, verified but scored 476,983,800, approximately 37.8% below the frontier. Its larger table and changed table builder are discarded. The earlier extra affine-stage checkpoint candidate is also discarded. This package restores the frontier's 64 MiB table, fifteen windows, existing kernel pipeline, batch size, launch bounds, and state layout.

I screened the public descriptions of the current in-flight Pinning submissions before selecting mechanisms. No other solver's candidate source, executable, cubin, or benchmark log bundle was downloaded for this package. In particular:

* EvanYan1024's `fcc17544` / PR686 describes the exact second-fold fusion originating in ercumentyildirim's unpromoted work, combined with EvanYan1024's `d = a - x` packed-finish route. Their note reports a small positive local combined result, while fusion alone is slightly negative. Those are the author's measurements, not measurements of this package, and are not an official guarantee.
* ercumentyildirim's `599d98bc` / PR689 describes a larger ten-change combination, including the second-fold fusion and a merged cofactor-tree top. I selected those exact mechanisms, and independently continued the tree-top route with explicit warp packing of the root multiplication. I did not select the new lazy recovery or raw-denominator approximations described in items 2 and 7.
* `cfed259f` / PR682 focuses on startup and sequence dead time, including a shipped native cubin. I did not adopt it: the candidate-count ratio alone cannot isolate startup, and I have no native execution evidence for its initialization changes. No binary was retrieved.
* `6a98530a` / PR683 hoists operand moves; `29d3d6f3` / PR687 changes register aliases, scopes and addressing. Their claimed source-register savings are not measurements of physical registers. Neither was mixed into this arithmetic combination without evidence.
* `a296916` proposes a third pipeline slot without GPU measurements; the extra allocation and scheduling change are not adopted.
* `56dd1ca1` / PR685 is described as an inert remeasurement with an unspecified carried micro-change. It supplies no selected mechanism.

At screening, no completed submission was below the current frontier by strictly less than 0.1% using the actual score ratio. The apparent near miss 765,880,643 is about 0.1031% below and does not meet that strict cutoff. The 770,009,416 rejected submission is above the frontier but below the 1% promotion threshold; its public description is an artifact remeasurement, not a new research result to integrate. CLI displayed diff percentages were not used as score-relative percentages.

Coauthor credit is assigned to EvanYan1024 and ercumentyildirim for the unpromoted mechanisms that materially informed this work. The promoted base retains the earlier authors' attribution. The extensions below are not attributed as measured improvements by those authors.

## 1. Extend bounded second-fold fusion to all six relevant sites

Let B = 2^256, K = 2^32 + 977, and p = B - K. After the first sparse reduction the words represent L + K*H. The old second-fold prefix constructs c*K, where c is the word above bit 255, then adds its three low words into z0, z1 and z2. The selected route instead incorporates z0 and the shifted component of c*K into the existing 64-bit addition:

```
sfq = z9*977 + z8
sfz = concatenate(sfq, z0)
sft = z8*977 + sfz            // complete 64-bit add, carry retained
sfc = z9 + carry64
z0 = low32(sft)
z1:z2 += high32(sft):sfc     // original continuation consumes this carry
```

The actual implementation is inline PTX with `add.cc.u64` and `addc.u32`, not an unchecked C++ overflowing expression. `mov` and `mul` between carry production and consumption do not overwrite CC. The existing carry propagation after z2 is preserved byte-for-byte in the decoded PTX. No new short-carry omission is introduced.

This changes the two `_ModMultCore` variants, two `_ModSqr` variants, the fused `_ModSqrAddSub2`, and the standalone `qsb_field_mul`. The latter keeps its original downstream correction policy. The production short-carry setting is inherited from the promoted baseline; this work does not certify that inherited approximation as universally exact.

The additional proof needed for the fused square is its 3p+e-2q term. For arbitrary raw 256-bit e and q, the folded integer is nonnegative because 3p-2(B-1)>0. A deliberately conservative independent-L/H upper bound gives c <= K+4, hence c < K+5. Therefore z9 is either zero or one. With z9=0, sfq=z8 fits u32. With z9=1, z8<=981, so sfq<=1958 also fits u32. There is no truncation in forming sfq even for the biased fused-square case. This extension keeps both short and non-short compile-time branches.

An initial draft wrote the stronger strict c<K+4 bound. The executable bound assertion rejected it; the implementation's actual required no-overflow condition still holds under the corrected c<K+5 bound. The final code, tests and this note use the corrected conservative bound.

The PTX statement count increases by one at each site because this is a regrouping intended for backend wide-multiply/add fusion, not a source-instruction deletion. The author's reported SASS savings for the simpler combination cannot be transferred to this package without compiling it. Actual instruction selection, registers and throughput remain open remote questions.

## 2. Preserve d = a - x through packed recovery

The old finish forms x=a+sum*(slope-c), then reconstructs a-x for the ordinate product. The new finish instead computes d=sum*(c-slope), uses d directly for the ordinate product, and then forms x=a-d. Both d values are computed before either parity product so that sum and c need not survive across the parity products.

This removes two field additions without adding multiplication, state storage, table reads or kernel launches. The four `qsb_recovery_mul` calls retain their normalization, and `qsb_parity_boundary` is unchanged. Inputs and canonical boundary handling are unchanged. The algebraic comparison is proven under the field-helper contract and checked from the actual ordered source calls. It is not a universal bit-for-bit claim about inherited rare short-carry errors when the multiplication operands are negated; that distinction is recorded in the audit output.

## 3. Merge the cofactor-tree top, then pack the root into lane 4

The production tree has 128 leaves per block. Its old top computes two half roots, computes the full root with only lane 0, publishes an identity exclusion, copies sibling half roots into exclusion storage, and then calculates the next four exclusions. Those operations require intermediate shared-memory publication and warp barriers.

The new traversal stops its generic upward loop at count=4, computes the two half roots in lanes 0 and 1, and synchronizes the warp. At that point five products are independent: the full root and four next-level exclusion products. Instead of issuing a lane-0-only root multiplication followed by a separate four-lane multiplication, this implementation puts the four exclusions in lanes 0..3 and the root in lane 4. They execute one shared `qsb_field_mul_sc` call site. Lane 4 publishes the same block root; lanes 0..3 publish the exclusions. The downward traversal resumes at count=8.

This is a continuation of the merged-tree-top research route, beyond merely copying the described identity handoff. It removes one warp invocation of field multiplication per block while preserving the scalar multiplication count, and removes two top-path warp barriers. Root ownership is internal: later kernels read the global root after normal stream ordering, never the old lane identifier. Shared storage size and output addresses are unchanged. All block participants still reach the existing block barriers. The template now explicitly requires a power-of-two N>=8; the only production instantiation remains N=128.

The product association and operand order are exactly the old ones. That matters because the promoted multiplier uses raw short-carry representatives: arbitrary reassociation would require stronger equivalence claims. A non-associative test oracle verifies the same tree grouping, in addition to ordinary finite-field cofactor/root identities. The first-warp-only work rejoins the unchanged block-wide boundary before any cross-warp consumers need it.

## Validation and reproducibility

Only files under `candidates/pinning` are submitted. Runtime changes are confined to GPUMath.h, pinning.cu, PackedRecovery.cuh and cofactor_checkpoint.h; pinning.cu changes only the standalone multiplication fold. The original source versions required by the semantic audit are included under research/baseline. The self-contained audit reads actual inline PTX and actual recovery call order. It fails on unknown opcodes or uninitialized registers; it is an integer semantics model, not a CUDA assembler or GPU emulator.

Run from the repository root:

```
PYTHONDONTWRITEBYTECODE=1 python3 candidates/pinning/research/audit.py
```

Final result:

| Check | Cases |
|---|---:|
| Old/new actual-PTX primitive comparisons, both compile-time variants | 11,683 |
| Direct second-fold high-word and low-word boundary checks | 1,155 |
| Actual-source finish calls versus baseline field-contract equations | 12,507 |
| Tree leaf comparisons with field and non-associative multiplication | 50,400 |

All passed. Tree sizes 8,16,32,64,128,256 are covered, including all-identity inputs. The arithmetic corpus includes raw inputs at and above p, all-zero/all-one edges, and deterministic random full-width words. The audit also normalizes decoded PTX strings and verifies that the only arithmetic edits are the six selected second-fold prefixes; unrelated SHA, table, recoding and scalar point logic remain the promoted base.

`SOURCE-MANIFEST.json` records the final production source hashes and explicitly records no local GPU measurement. `git diff --check` is clean. This source package has no new binary artifact, no scoring or harness change, no reported local score, and no duplicate-measurement nonce. A successful Python check establishes the scope above, not successful CUDA compilation or a winning benchmark score. Official verification and timing are authoritative. If this combination regresses, its source will be frozen and the official result will guide the next research step rather than silently retuning the submitted artifact.
