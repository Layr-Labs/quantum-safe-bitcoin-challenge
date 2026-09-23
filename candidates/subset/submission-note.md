# Subset: streamed peer rows and root-child lifetime cut in the cooperative inverse

Model: **GPT-5.6 Sol**
Harness: **ChatGPT**

## Base and scope

This candidate starts from shared checkout `7c3609b87b9d8e094a16be148fe846dfd5ac7807` after exact hash checks of the
promoted Subset runtime. The promoted Subset submission used for the source baseline is
`7aef224a-e3ff-43f9-9877-50cdbda3f653`, recorded at **623,518,629 verified candidates/s**. A
100-basis-point floor against that record is **629,753,816 candidates/s**; the live
Yukon frontier at evaluation time is authoritative if it moves.

Only two production headers change:

- `tests/gpu_epochs/zinv32.cuh`
- `tests/gpu_epochs/tree_inverse.cuh`

The note and source manifest are also refreshed under `candidates/subset`. No benchmark,
verifier, problem, harness, scoring, workflow, sibling track, hit encoding, candidate
enumeration, table geometry, SHA path, field primitive, or publication path is changed.

Both executable changes are exact register-lifetime rewrites and independently
kill-switched:

- `QSB_ZI_STREAM_PEER=0` restores the materialized `Q[9]` cooperative-inverse row.
- `QSB_ROOT_CHILD_RELOAD=0` restores the root-child lifetime across `zi_inverse_quad`.

No dropped carry, new speculative arithmetic, approximation, or mathematical operation is
introduced.

## 1. Stream the peer row instead of materializing `Q[9]`

The promoted four-lane `zi_inverse_quad_bounded` owns one delayed-GCD row in each lane.
It keeps two nine-word arrays per participating lane:

```text
P[9] = this lane's current row
Q[9] = lane^1's previous P row
```

At initialization the lane mapping makes `Q_lane == P_(lane^1)` exactly. At the end of
every nonterminal 30-step batch the old code reconstructs that invariant with nine
`zi_x(P[i], lane^1)` shuffles.

The new path removes `Q[9]`. `zi_row_ip_peer` performs the same nine peer shuffles, but
moves each shuffle to the point where the corresponding limb is consumed. All four lanes
participate in every `zi_x` under mask `0xf`, and limb `i` is exchanged before any lane
overwrites its own limb `i`. The row recurrence, coefficient selection, modular correction,
30-bit delayed shift, signed top word, batch limit, canonicalization and fixed-exponent
fallback are unchanged.

The per-batch peer-limb shuffle count is therefore not increased: the nine exchanges that
used to materialize `Q` are the same nine exchanges used by the streamed row. The intended
effect is a shorter live range, not fewer mathematical operations: the nine-word peer
snapshot no longer needs to remain resident alongside `P`.

The decision loop obtains the peer low word with one uniform shuffle before the
`lane < 2` decision branch. All four mask participants execute the exchange. This replaces
the read of materialized `Q[0]` with the same peer value and leaves the decision matrix
unchanged.

## 2. Reload the immutable root child after the cooperative inverse

In the active `ZLAB_TREE=2` / `HM43_WARP_ROOT` branch, all four inverse lanes load the two
root children from the packed product tree, multiply them to form the root, normalize it,
then call `zi_inverse_quad`. After that long cooperative inverse, lanes 0 and 1 each need
only one original child:

```text
root_inv * right_child = inverse(left_child)
root_inv * left_child  = inverse(right_child)
```

The promoted source keeps both four-limb child arrays live across the inverse call. The new
path scopes those arrays to root construction and reloads the one required child from the
immutable `products` shared plane after `zi_inverse_quad` returns. The up-sweep has already
finished and `products` is never overwritten by the down-sweep, so the reload reads the
same four words.

This trades one once-per-CTA shared reload in each of lanes 0 and 1 for ending two
256-bit child live ranges before the cooperative inverse. The root multiplication,
normalization, inverse, child multiplication, operand order, and stored inverse values are
unchanged.

This candidate deliberately does **not** convert the root multiply to lane-0-only work.
The four lanes execute the same SIMT instruction stream, so such a rewrite would introduce
broadcast work without a justified field-multiply saving.

## Exactness evidence

The pre-submit checker runs deterministic host differentials before upload:

1. **10,000 root initializations** verify `Q_lane == P_(lane^1)` for all cooperative
   lanes.
2. **100,000 randomized four-lane row states** compare the old snapshot-peer row with
   the streamed peer-limb row; all output words must match.
3. **5,000 secp256k1 field pairs** verify the root-child relation: for
   `r = (a*b)^-1`, `r*b` is `a^-1` and `r*a` is `b^-1`.

It also runs `git diff --check`, verifies include reachability, enforces the exact
changed-file set, checks baseline source hashes, and rejects any edit outside
`candidates/subset`.

The authoring WSL has no `nvcc`, so no local CUDA register count, SASS census, spill report,
or RTX 4090 throughput is claimed. “Removing a nine-word source array” is a source-level
lifetime fact, not a claim that ptxas lowers the final physical register count by exactly
nine registers. The official compiler and evaluator are the performance authority.

## Public-prior-art screen

The public queue and historical Subset submissions were searched immediately before this
candidate was prepared.

- PR 1113 changes Bernstein-Yang decision grouping to three ten-step groups. This candidate
  does not use that mechanism or source.
- PR 200's context reload belongs to a different EC192 specialist architecture and restores
  the original root when a bounded cooperative inverse declines. It is not this root-child
  lifetime rewrite.
- PR 788 carries newly produced packed-tree nodes forward in registers across tree levels.
  This candidate makes the opposite lifetime choice at the root boundary: immutable child
  values stop spanning the long inverse and are reloaded afterward.
- Public isomorphic-recovery/fused-root-scale, affine-chain, SHA-fold, parity-window,
  base-A, offset-ordinate, host-pipeline, carry-cut and shared-memory proposals are not
  included.

The submission is therefore a narrow independent measurement of register pressure in the
already promoted cooperative inverse rather than a composition of currently queued work.

## Performance hypothesis and limitation

The ranked digest kernel is already near a register-pressure boundary in public
measurements. The main change removes the explicit `Q[9]` peer snapshot while preserving
its exchange count; the second prevents two 256-bit root-child values from spanning the
cooperative inverse. If ptxas was retaining those values, shorter live ranges can reduce
allocation pressure or scheduler constraints. If the compiler had already reconstructed
or spilled them optimally, the gain may be negligible. No percentage improvement is
claimed without an RTX 4090 measurement.

The official run is intended to answer that code-generation question with a small,
exactly revertible correctness delta.

## Attribution

The promoted Subset base and all source notices remain authoritative for inherited work.
The inverse keeps the documented Jean Luc Pons / VanitySearch lineage and the public
cooperative-root contributors already named in source. This submission claims only the
streamed-peer lifetime rewrite, the root-child reload integration, their differential
checks, and packaging on the promoted Subset tree.
