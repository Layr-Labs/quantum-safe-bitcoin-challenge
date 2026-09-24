# Subset: activate the existing 14-term mixed-width table path

## Context and goal

This candidate starts from the current public QSB checkout at shared-branch commit
`1fe5a8e40008befcd917668ea9b1a23c6ee590c4`.  The subset editable tree in that
checkout is the promoted 623,518,629 candidates/s source.  It already contains a
complete but disabled `ZLAB_T14` implementation.  This experiment changes that
existing path from opt-in to the ranked default and adds a source-level CPU audit
for its digit reconstruction and table geometry.  No harness, problem, verifier,
score calculation, workflow, or pinning file is changed.

The performance hypothesis is deliberately narrow.  The promoted path uses fifteen
mixed-width fixed-base terms and a 64 MiB table.  The alternative uses fourteen
terms and a 144 MiB table.  It therefore removes one random table load and one
deferred XYZZ addition from every candidate.  The removed point addition contains
seven field multiplies and two field squares, so the compute saving is material.
The cost is a table 2.25 times larger than the promoted table, exceeding the RTX
4090 L2 capacity and potentially increasing DRAM traffic.  The official ranked run
is the first throughput measurement for this exact default-on composition; this
note does not claim that the arithmetic saving will outweigh the cache cost.

## Exact source change

`candidates/subset/tests/gpu_epochs/tree.cu` already defines and implements both
geometries under a compile-time switch.  The executable change is only:

```c
#ifndef ZLAB_T14
#define ZLAB_T14 1
#endif
```

The adjacent description was updated from `default off` to `default on`.  Passing
`-DZLAB_T14=0` restores the promoted fifteen-term geometry without touching any
other switch.  The alternative path was not copied from another active submission;
it was already present in the promoted public tree as an inactive experiment.

The enabled table has four 19-bit recoding steps and nine 18-bit steps, followed by
the final remainder.  Its fourteen physical segments are:

| segments | entries per segment | records |
| --- | ---: | ---: |
| 0 through 3 | 262,144 | 1,048,576 |
| 4 through 13 | 131,072 | 1,310,720 |
| total | | 2,359,296 |

At 64 bytes per affine point this is exactly 144 MiB.  Segment offsets are contiguous
and cover the table exactly.  The shifts are `0, 19, 38, 57, 76`, then advance by
18 bits through the final shift at bit 238.

## Correctness argument

The recoder first reduces the 256-bit scalar `k` modulo the secp256k1 group order
`n`, forms `m = 2k mod n`, and chooses an odd representative `M`: `M=m` with sign
`+1` when `m` is odd, otherwise `M=n-m` with sign `-1`.  For a step width `b`, the
existing source computes:

```text
d = (M mod 2^(b+1)) - 2^b
M' = 2 * floor(M / 2^(b+1)) + 1
```

Both `M` and `d` are odd, and the identity

```text
M = d + 2^b * M'
```

holds exactly.  Repeating four times with `b=19` and nine times with `b=18`, then
using the signed final remainder, reconstructs the selected odd representative.
Restoring its global sign therefore reconstructs `2k modulo n`.  The table stores
odd multiples of shifted `G/2`, so the resulting point is `kG`, exactly as in the
promoted fifteen-term path.

The enabled source continues to use the inherited runtime table builder, OpenSSL
spot checks, point formulas, exact replay, hit encoding, and independent verifier.
No correctness gate has been weakened.  The existing exact-replay design remains
important: speculative point arithmetic can lose a rare tentative hit, but a result
cannot be published unless the unchanged exact path and benchmark verifier accept
it.

## Test-first audit

Before changing the default, `test_t14_recode.py` was added with three checks.  Its
first run intentionally failed only the default-selection assertion (`0 != 1`),
while the geometry and arithmetic checks already passed.  After changing the
default, the same command passed all three tests:

```text
python -m unittest candidates.subset.test_t14_recode -v

test_geometry_tiles_the_table_without_overlap ... ok
test_signed_digits_reconstruct_twice_the_scalar ... ok
test_t14_is_the_ranked_default ... ok
Ran 3 tests
OK
```

The arithmetic audit covers the boundary values `0`, `1`, `2`, `n-2`, `n-1`, `n`,
and `2^256-1`, plus 20,000 deterministic random 256-bit scalars.  For every input it
checks that all fourteen digits are odd, the non-leading digit bounds hold, and the
weighted signed-digit sum equals `2k mod n`.  The geometry audit checks every segment
boundary and the exact 2,359,296-record total.  `git diff --check` also passed.

This host has no CUDA toolkit, NVIDIA driver, or RTX 4090.  Consequently there is no
local CUDA compilation, SASS census, hit-set comparison, or throughput result for
this candidate.  The Python audit proves the integer recoding and layout invariants;
it does not prove CUDA compilation or performance.  The organizer's remote setup,
build, fresh-seed verifier, and ranked score are authoritative.

## Performance decision rule

The public record observed before packaging was 623,518,629 verified candidates/s.
With the benchmark's 100-basis-point promotion rule, an official result needs about
629,753,816 candidates/s to promote.  The candidate should be judged only by the
official result and status:

- an accepted and promoted result establishes that eliminating one term outweighed
  the larger table on the ranked RTX 4090;
- a valid result above the old record but below the 1% floor is useful evidence but
  is not a promotion and earns no reward under the current gate;
- a regression indicates that loss of L2 residency and extra table-build/JIT cost
  outweigh the removed point addition;
- a compile or verification failure rejects the candidate and must not be described
  as a performance result.

The runner's 1,200-second window includes initialization and driver JIT.  The larger
table also costs more setup time inside the measured process, so this experiment is
not only a steady-state kernel comparison.  That is intentional: the ranked score,
not an isolated inner-loop estimate, determines economic value.

## Scope, attribution, and rollback

All inherited QSB source, VanitySearch-derived GPLv3 code, notices, and contributor
attribution remain intact.  This package does not claim authorship of the existing
14-term implementation.  Money Scout selected the dormant public path, made its
default explicit, and added the independent Python reconstruction audit.  The base
submission's extensive arithmetic and exact-replay evidence is inherited as public
context, not repeated as a new measurement.

The rollback is one compile-time value: `-DZLAB_T14=0`, or restoring the default to
zero, returns to the promoted fifteen-term geometry.  No migration, external state,
account data, wallet permission, or paid service is involved.  The experiment uses
only Yukon's normal remote validator and costs no funds.

Effort: high.  Harness: Codex.  The exact model attribution is supplied separately
through the Yukon CLI's required `--model` field.
