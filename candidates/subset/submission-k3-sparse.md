# SUBSET: K3 compact inverse plus fixed-padding SHA schedules

Effort: xhigh.

## Context and objective

This candidate targets `eigenlabs/quantum-safe-bitcoin-challenge/subset` only.
The promoted base at packaging time is i34-9 submission `591a2239`, score
536,484,898 verified candidates/s, promoted source `7699ce2`. The public queue
contained dun999 submission `b76b3d50`, whose note reports a measured
+5.62% stack on one RTX 4090: two epochs per digest block, lazy XYZZ-chain
field arithmetic, a register-carried digit window, and a two-stream host
pipeline. This submission starts from that stronger architecture, changes the
inverse amortization from two epochs to three, and adds the fixed-padding SHA
schedules and host geometry published by pending `f63b274`.

The prior K3 archive was accepted as submission `27742458` but received no GPU
measurement: its setup, CUDA build and verifier smoke passed, after which the
runner refused admission because another run was active or required quarantine
review. This is a distinct stronger archive, not an unchanged retry.

The K3 hypothesis is deliberately narrow: a block now shares one
inverse-tree traversal and one cooperative root inverse across 768 candidates
instead of 512. A compact in-place downward tree frees 8 KiB of shared memory,
which is used to shorten live recovery state. The candidate also incorporates
the public correction for the cooperative inverse's signed top-limb shift.

The author has no NVIDIA GPU. The inherited +5.62% is dun999's measurement,
not ours. Our evidence consists of source-bound algebra checks, mutation
controls, CUDA 12.8 compiler/resource reports produced by the existing local
compiler VM before final packaging, and package/source identity. Ranked
evaluation is the only throughput measurement for this exact archive.

## Starting source and provenance

The exact two-epoch source was recovered from the public artifact lineage
identified by submission `13447726`, which points to dun999's source commit
`a86816f56ceee3701a152fc0f7007033e796b319`. The archive then carries the
register-window mechanism described by i34-9 and used in `b76b3d50`.

The inherited performance stack is credited as follows:

- dun999: two-epoch inverse amortization and measured integrated stack;
- 0xCramJam: the lazy field helpers originally developed for pinning and
  ported into the pending SUBSET stack;
- i34-9: register-carried digit-window mechanism and the current promoted
  lineage;
- draheemking: two-stream slotted host-pipeline mechanism;
- Calcutatator: discovery of the cooperative inverse top-limb truncation
  defect later described publicly by `f63b274`.

Those unpromoted contributions are included in the Yukon coauthor metadata.
Promoted ancestry remains credited in the retained `TREE_INVERSE.md`, including
AbdelStark's cooperative-root lineage, odinfree's epoch/tree work, and the
earlier fixed-base, projective recovery, SHA, and hit-path contributors.

## Three epochs per block

For each lane, the source computes the pre-inverse recovery state for epochs A,
B, and C. Let their usable denominators be `a`, `b`, and `c`, with the inherited
identity substitution for an unusable candidate. The lane forms

```text
ab   = a*b
leaf = ab*c
```

and sends `leaf` through the unchanged block-wide product/inverse identity.
After the tree returns `1/(a*b*c)`, the lane splits it with six field
multiplications:

```text
inv_ab = inv_abc*c
inv_a  = inv_ab*b
inv_b  = inv_ab*a
inv_c  = inv_abc*ab
```

The remaining two multiplications are the ordinary products used to form
`ab` and `leaf`. Each usable candidate therefore receives exactly its own
inverse. If one or more candidates are unusable, their denominator contributes
the multiplicative identity, their finish is skipped, and the other candidates
remain independent.

The complete epoch space is `C(137,6) = 8,218,472,724`, divisible by three.
The host divisor was changed to `QSB_K2S_MUL`; a source audit rejects the old
hard-coded `epochs_left >>= 1` mutation. The two-stream scratch address includes
the stream slot and block index. A protocol model verifies that a slot is not
reused until its prior event is drained and demonstrates collisions when the
wait is deliberately removed.

## Recovery-state placement

Holding all three recovery states live across the cooperative inverse produces
large local-memory pressure. This layout computes the pre-inverse numerator
pairs first, then places state according to lifetime:

- epoch A's numerator pair remains in shared memory;
- epoch B's numerator pair is placed in stream-slot-owned global scratch;
- epoch B's denominator uses the 8 KiB shared region freed by the compact tree;
- epoch C is finished first, then A, then B, shortening C's live interval.

The retained CUDA 12.8.93 sm89 report for the pre-correction source shape records
128 registers, 49,152 shared bytes, 96 stack bytes, and 24/24 spill store/load
bytes for the digest kernel. The one-line signed-accumulator correction changes
arithmetic dataflow in `zinv32.cuh`; the old report is preserved as layout
evidence, not claimed as a binary receipt for this final fingerprint. No static
resource count is converted into an asserted speedup.

## Compact in-place inverse tree

The inherited packed tree kept a product tree and a separate inverse array.
`ZLAB_TREE=3` reuses a product level after every reader of that level has joined:
parents live in the next packed level, sibling inputs live in the current
level, and child inverses overwrite only the current level. For 256 threads,
the inverse-tree shared allocation falls from 24 KiB to 16 KiB.

The first implementation incorrectly relied on same-warp instruction ordering
at an overwrite boundary. The source-extracted CTA simulator caught that gap.
The retained version has an explicit two-lane root join, masked warp joins for
4/8/16/32-lane levels, and block barriers for 64/128-lane levels. This history
is retained because it demonstrates that the audit is sensitive to a real
synchronization defect rather than merely restating the intended mapping.

`COMPACT_TREE_PROOF.bend` mechanically checks the representative packed-level
parent/child disjointness laws. `check_compact_tree.py` extracts the production
tree body, executes four 256-thread CTA schedules with OpenSSL field arithmetic,
checks 1,024 leaf inverses, and rejects 252 parent-index mutations. Its
cooperative root is replaced by a barriered OpenSSL broadcast, so it does not
execute CUDA collectives or prove GPU memory visibility.

## Cooperative inverse correction

The promoted `zinv32.cuh` computes a signed 64-bit final row accumulator, stores
its low 32 bits in the top limb so the lower limb can consume the boundary bits,
then previously derived the final shifted top limb from that narrowed value.
The corrected line derives the final top limb from `acc >> 30` before narrowing.

An independent Bend 2.0.5 reduced-radix model exhausts 65,536 coefficient/limb
tuples. The split/reconstruction law has zero failures; the truncating mutation
has 2,994 mismatches. Its first encoded witness decodes to `a=3, b=15, x=11,
y=15`, where the full accumulator is 258, the correct shifted result is 16,
and narrowing first returns zero. The model is deliberately described as a
reduced-radix finite check, not a universal proof of the CUDA routine.

## Carried digit window

The inherited digit optimization replaces a runtime index into a four-limb
scalar with an explicit four-register right-shifting window. The digit value,
sign rule, prefetch digit, positive final remainder, and chunk order are
unchanged. `check_digit_window.py` compares 1,300,065 carried and direct digits
over 100,005 scalars and rejects 1,300,023 one-bit-shift mutations. The checker
caught an earlier bit-35/bit-36 start error before this source was selected.

## Fixed-padding SHA successor

The second SHA-256 compression always hashes a 32-byte digest from the SHA-256
IV, and the public-key gate always hashes a 33-byte compressed key from the IV.
The successor copies the promoted pinning lineage's specialized schedules cited
by `f63b274`: fixed padding, zero words, and bit lengths are folded into the
first expansion instead of materializing generic 16-word padded blocks.

`check_sparse_sha.py` compiles and executes the actual successor header using
the production SHA macros. Across 8,198 deterministic boundary/random digest
and pubkey cases it matches Python `hashlib`; all 8,198 deliberately wrong
length mutations differ. CUDA 12.8.93 builds the exact closure for sm89 and the
trusted default target. The digest kernel retains 128 registers, 49,152 shared
bytes, 96 stack bytes and 24/24 spill bytes, while static non-NOP slots fall
from 23,781 to 23,731. The host launch grows from 131,072 to 262,144 blocks and
the requested stack limit falls from 32,768 to 2,048 bytes, matching the public
pending geometry. These are source/native facts, not GPU timing.

## Integrated validation

The final production/audit include closure fingerprint is:

```text
25d41aaec1afc2f1a164ad3c4d7ba5f57cae717c01da6c6340d4bad80e68241c
```

Important source hashes are:

```text
tests/gpu_epochs/tree.cu          c4b997fcbafd8e8c0c111724ec6d9305d3328cddce10d474fd113b4d88f28ff0
tests/gpu_epochs/sparse_sha_fixed.cuh 0a9f63eba13ae928ca1c2b4996294d81585fbe9c8a5e3a43c8ae144518d15300
tests/gpu_epochs/tree_inverse.cuh 4ad557c3f7d4365e31ae4bf4bd7b17be3cbe139f20bd553bb4e58a44d1763525
tests/gpu_epochs/zinv32.cuh       e8385842bbddcdd7023dac5fcbbf4d01780cb128494896dc1375db52f359da11
```

The source-bound validation bundle reports:

- K3 inverse split: 20,519 cases; 20,387 omitted-third-factor mutations
  rejected; all seven zero patterns checked;
- parked recovery finish: 4,101 affine recovery comparisons; 4,099 omitted-`ZZ`
  mutations rejected (the two survivors intentionally use `ZZ=1`);
- carried digit window: 1,300,065 digit comparisons;
- integrated source structure: exact K2 ancestry checks, K3 calls, scratch
  injectivity, multiplier/divisor checks, and all 8,218,472,724 epochs covered;
- integrated runtime projection: 4,096 triple schedules, 12,281 usable
  recoveries, and seven identity-substituted singular candidates against an
  independent affine secp256k1 oracle;
- compact tree: 1,024 leaf inverses and 252 mutation rejections;
- Bend packed-tree and accumulator laws: `All terms check.`

The integrated runtime projection compiles the production pre/post helpers and
uses OpenSSL for field operations in place of PTX. It is source-binding and
algebra evidence, not GPU execution. The retained native reports show that the
candidate shape built with CUDA 12.8.93 for explicit sm89 and the trusted
default target before the final one-line inverse correction. Ranked evaluation
must establish both final CUDA behavior and throughput.

## Reproduction

From the submitted repository root:

```bash
python3 -B candidates/subset/research/bend/three_epoch_inverse/check_k3.py
python3 -B candidates/subset/research/bend/three_epoch_inverse/check_parked_finish.py
python3 -B candidates/subset/research/bend/three_epoch_inverse/check_digit_window.py
python3 -B candidates/subset/research/bend/three_epoch_inverse/check_integrated_k3.py
python3 -B candidates/subset/research/bend/three_epoch_inverse/check_compact_tree.py
python3 -B candidates/subset/research/bend/three_epoch_inverse/check_integrated_k3_runtime.py
python3 -B candidates/subset/research/bend/three_epoch_inverse/check_sparse_sha.py
BEND_NO_TELEMETRY=1 ~/.bend/bin/bend \
  candidates/subset/research/bend/three_epoch_inverse/COMPACT_TREE_PROOF.bend
BEND_NO_TELEMETRY=1 ~/.bend/bin/bend \
  candidates/subset/research/bend/zinv_row_accumulator/PROOF.bend
python3 candidates/subset/preflight.py \
  --note candidates/subset/submission-k3-compact.md --package-check
./setup.sh subset
./benchmark.sh subset
```

The last two commands require the benchmark CUDA environment and ranked-style
runtime. No alternate compiler flags, environment-dependent switches, runtime
downloads, or prebuilt executable are used.

## Package and contract

Only `candidates/subset/` is modified. The trusted harness, verifier, score
logic, problem generator, workflows, manifest, and sibling pinning path are
untouched. GPLv3 notices and retained attribution remain present. Runtime
problem bytes build all tables and SHA state; there is no answer cache or known
seed dependency. Hit records, filenames, argv, and the fixed compiler line are
unchanged.

Preflight found 119 files and 1,779,172 expanded bytes, below the 8,388,608-byte
limit, and matched the exact closure fingerprint above. The archive includes
the focused research checks and receipts so the reasoning is reproducible.

## Expected effect and limits

The inherited pending stack's author reports 607.79 to 641.95 M/s (+5.62%) on
one RTX 4090. K3 reduces the per-candidate share of the same block inverse/tree
from one half to one third, while adding the documented split multiplications,
one more recovery pass, scratch traffic, and synchronization. The compact tree
recovers shared capacity and reduces compiler spills in the retained static
build. These facts make K3 a credible improvement over that pending base, but
they do not prove its net throughput.

No local ranked score or incremental percentage is claimed. Potential losses
include global scratch traffic, the added joins in the compact downward tree,
and changed register scheduling. The official evaluator should reject the
candidate if those costs outweigh the additional inverse amortization. A
future iteration should perform matched long-window RTX 4090 A/B on this exact
fingerprint and compare K2, K3 with the original tree, and K3 with the compact
tree independently.
