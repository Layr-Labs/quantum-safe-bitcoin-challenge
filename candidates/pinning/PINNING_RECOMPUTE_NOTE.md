# Pinning candidate note: recompute the lower checkpoint tree level

## Candidate identity and scope

This note describes one isolated bold candidate for the Yukon `pinning` track.
The candidate is based on the promoted source revision
`d2772418e0f372767b4c59f7382d71f9142585fe`, whose current recorded frontier is
`644546620` verified candidates per second on the official RTX 4090 runner. The candidate preserves the current official rule and verifier contract.

The editable scope is only `candidates/pinning`. The executable change is in
`pinning.cu`; `audit_external_pipeline.py` is updated so the existing CPU
algebra audit models the candidate's packed checkpoint layout. `GPUMath.h`,
`GPUHash.h`, `COPYING`, benchmark files, scorer files, setup files, and every
other official or protected path are unchanged. The candidate keeps the
promoted signed mixed-radix fixed-base table, deferred-Y XYZZ chain, two-level
root batching, vectorized four-field state, fast SHA tail, and both recovery
flags. It changes one pipeline mechanism only: which product-tree nodes cross
the prepare/finish kernel boundary.

Implementation and the original technical note: GPT-5.6-Luna at max effort in Codex.
Architecture review: GPT-6-Astra at high effort. Parent integration, package review,
and publication: GPT-6-Astra at xhigh effort, using HarnessYukon in Codex. No external source or public patch is copied by
this change, so there is no external algorithm coauthor to attribute for the
checkpoint-recompute mechanism. The surrounding arithmetic and pipeline are
inherited from the exact promoted candidate and retain the authorship and
provenance already recorded in the checked-in research log. Parent publication
remains the authority for the final note, current frontier refresh, and upload.

## Hypothesis

The current hierarchical pipeline computes a 256-leaf field product tree in
the prepare kernel, writes 254 non-root internal nodes to a multi-hundred-
megabyte global checkpoint, and reads all 254 nodes back in the finish kernel.
The first internal level has 128 nodes and is the cheapest level to reproduce:
each node is simply the product of two leaves that are already retained in the
128-byte-per-candidate state. Persisting this level pays two global transfers
for every 256-bit product. Rebuilding it in finish costs one field multiply per
pair and one additional block barrier, while leaving the six upper levels
and the root inverse fanout unchanged.

The concrete prediction is that the 32 bytes of saved checkpoint traffic per
candidate can compensate for the additional 0.5 field multiply per candidate,
especially if the 2 GiB candidate state and the old 512 MiB tree checkpoint
put pressure on the RTX 4090 memory system. The prediction is deliberately
provisional. The saved bytes are logical traffic estimates, and the local Mac
cannot compile or execute CUDA. PTXAS register allocation, memory residency,
and the relative cost of an additional field multiply are left to the ranked
server run. A server score must clear the current one-percent requirement; no
local score is claimed here.

## Exact product-tree layout

The shared product array uses the same node numbering as the promoted
`qsb_block_inverse` helper:

```text
nodes 0..255     saved per-candidate leaves (W, with inactive lanes = 1)
nodes 256..383   first pairwise level, 128 products
nodes 384..509   six upper internal levels, 126 products
node 510         complete root, published in the root array
```

The prepare helper still computes every product through node 510. It now
publishes only nodes 384..509, packed as offsets 0..125 in each field plane.
`QSB_CHECKPOINT_STRIDE` is consequently 128 rather than 256. The two trailing
slots in a plane are unused and preserve a power-of-two stride. In finish,
each lane first restores its leaf and the first 126 lanes restore the packed
upper nodes. After the existing load barrier, lanes 0..127 multiply leaves
`tid` and `tid+128` and write the omitted nodes at `256+tid`. A second barrier
makes those writes visible. The original downward inverse expansion then
starts at node 508 and reaches the reconstructed node 256..383 level with no
change to its offsets or its leaf result.

This is a layout change, not a change to the field representation. Every
stored product still uses `qsb_field_mul`, and every restored product is used
as the same congruent representative that the old full checkpoint carried.
Node 510 remains in the compact root array and is normalized immediately
before `_ModInv` by the existing root kernel. Returned leaf inverses are still
normalized exactly once at the existing leaf boundary. The root-group prepare
and finish helpers use the same packed layout, so the auxiliary hierarchy gets
the same reduction without adding a second implementation of the tree.

## Traffic and storage accounting

For a full 16,777,216-candidate batch there are 65,536 search CTAs. A persisted
256-bit internal node occupies 32 bytes. The promoted layout writes and reads
254 nodes per CTA, or 8,128 bytes in each direction. The candidate writes and
reads 126 nodes, or 4,032 bytes in each direction. The exact saving is thus
4,096 bytes per CTA per direction, 8,192 bytes per CTA round trip, and 32
bytes per candidate for the main search tree. Across the full batch this is
256 MiB of writes plus 256 MiB of reads, for 512 MiB less main-tree traffic.

The root hierarchy has 256 root groups for that batch. Its checkpoint has the
same 4,096-byte-per-direction saving per group, or 1 MiB in each direction
(2 MiB round trip). Including that auxiliary level, the logical saving is
approximately 32.125 bytes per candidate. The state and root arrays are
unchanged. The main tree allocation falls from 512 MiB to 256 MiB, and the
root-tree allocation falls from 2 MiB to 1 MiB. With the unchanged 2,048 MiB
state, 2 MiB root array, and small super-root array, the padded allocation
drops by about 257 MiB, from roughly 2,564 MiB to roughly 2,307 MiB.

The current hierarchical external-pipeline accounting is about 320 bytes of
logical cross-kernel traffic per candidate before this candidate's change,
with the large state transfer dominating and the first-level tree checkpoint
the largest removable component. The candidate does not claim that logical
traffic equals DRAM traffic: cache hits, write combining, compression by the
memory system, and kernel overlap are runner observations. The purpose of
the calculation is to bind the exact tradeoff that the ranked run must test.

The arithmetic trade is equally explicit. Recomputing the first level adds
128 `qsb_field_mul` calls per 256-lane CTA, exactly 0.5 call per searched
candidate. It does not repeat fixed-base preparation or the upper tree. The
finish helper gains one pre-downward `__syncthreads()` after the existing
restore barrier. Its seven original downward barriers, the prepare tree, the
root-group kernels, and the single super-root `_ModInv` are otherwise intact.
The shared arrays remain 16 KiB for products plus 8 KiB for inverses, so the
candidate does not increase the 24 KiB shared-memory footprint. A possible
PTXAS register increase or spill in the finish kernel is the main compiler
risk; that is why only the first level is recomputed instead of deleting all
254 checkpoints.

## Correctness argument

For every active lane, the prepare tree's first level is

```text
products[256 + tid] = products[tid] * products[128 + tid] mod p
```

for `0 <= tid < 128`, where `p` is the secp256k1 field prime. The finish code
uses the same operands and the same `qsb_field_mul` schedule, so the rebuilt
values are byte-for-byte equivalent as field residues to the omitted prepare
stores. All upper nodes are restored from their exact prepare values. The
root inverse supplied by the hierarchy is therefore still the inverse of the
same complete product, and each downward child inverse is the same parent
inverse times the unchanged sibling product. The final leaf expression is
unchanged.

Inactive candidate lanes and unusable recovery denominators are represented
by the multiplicative identity before the tree helper, as in the promoted
source. The first-level products of identity leaves remain identity. Partial
last batches therefore have the same fixed-width tree shape and do not read
uninitialized checkpoint slots. The packed global offset is independent for
each field plane and each CTA: node 384 is stored at slot zero and node 509 at
slot 125, with no overlap between neighboring CTAs. The same argument applies
to the root-group tree, whose leaves are compact roots plus identity padding.

The change does not alter the C/Y/W/ZZZ state transform, the common inverse,
the direct XYZZ formulas, SHA-256, DER checks, hit encoding, `single_hash`,
or either recovery flag. It therefore preserves the exact all-input contract
provided the CUDA compilation and ranked verifier accept the source. The
local checks below exercise the tree and finish algebra but cannot establish
device execution or the official score.

## Evidence collected in this checkout

The updated `audit_external_pipeline.py` reconstructs the full 511-node tree
with Python modular arithmetic, returns only nodes 384..509 from prepare, and
rebuilds nodes 256..383 in finish. It checks active counts 0, 1, 31, 32, 33,
127, 128, 129, 255, and 256, injected zero values, 200 random active masks,
and 10,000 independent C/W recovery comparisons against the original
formula. It passed with:

```text
PASS: 210 split trees and 10000 C/W finish comparisons
```

The inherited audits also pass in the same checkout:

```text
audit_deferred_chain.py       20000 arbitrary-field, 1000 curve, 1000 mixed-window cases
audit_fast_tail_contract.py   20 host selection cases
audit_field_final_carry.py    200576 targeted/random cases
audit_shared_tree.py          211 product-tree cases
audit_stream_recode.py        51404 scalars and 1048576 table slots
audit_superbatch_representation.py 10 root-count boundary cases
audit_superbatch_roots.py     137492 roots across 12 boundary sizes
audit_vector_state_layout.py  exact 2 GiB state and alignment/layout checks
```

`git diff --check` is clean. The executable source SHA-256 after this change
is `364ac23e90a132dc49c0eb5b544a0d79f89c9de13a638d26fd63f283db11868c`.
The updated audit SHA-256 is
`5ff2f65f543c3fad2e463f2b261f796aaa89ef0cf56ee81c85e0491b71724da5`.
These hashes bind the exact files in this isolated candidate. No CUDA toolkit,
NVIDIA GPU, local CUDA build, local GPU correctness run, or local GPU timing
was available on Max for this task. The evidence is consequently a semantic
and layout screen, not a verified throughput result.

## Sources, attribution, and ranked validation

The exact public control is the official challenge source at
https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/tree/d2772418e0f372767b4c59f7382d71f9142585fe/candidates/pinning .
This candidate inherits the control implementation and its recorded public solver
contributions; all existing copyright, license, and provenance files are retained.
The changed checkpoint policy is original work in this attempt. No unpublished
external solver material was used.

The official runner must compile this exact package, exercise the official
verifier, and measure its verified candidates per second. No claimed score is
supplied. Performance remains a hypothesis: less global checkpoint traffic may
help when it lies on the critical path; additional multiplication, synchronization,
or compiler register effects may erase the saving or make the candidate slower.
The unchanged public control remains the validated reference until an official
result establishes otherwise. A score, promotion, acceptance, and credited points
are separate outcomes and none is claimed for this candidate.
