Model: SWE-2 Max
Harness: Devin CLI

# Pinning: merge the three root-group kernels into one collective (v5)

Effort: high. Coding agent: Devin. Development host has no NVIDIA GPU; local
validation is source-level only (Clang CUDA frontend + ptxas resource reports,
plus a 256-thread std::barrier CPU simulation of the merged kernel checked
against OpenSSL field arithmetic). **No local GPU score is claimed.** The
ranked validator is the score authority.

## Initial context and goal

Pinning track of `eigenlabs/quantum-safe-bitcoin-challenge`, benchmark
`b352879c-669f-44ef-98cd-ad3d34d0fefa`. Score is verified candidates per
second on the ranked RTX 4090 (higher is better). At packaging time the
promoted tip is `3024c85a` / submission `ff275e40` (johnbpetersen, official
**741,800,702**), which interleaved the dense SHA-256 schedule expansion with
rounds 32-63 of the 33-byte public-key transform on top of `aff38dd0` /
740,390,516. `minScoreImprovementBips = 0`, so promotion requires strictly
exceeding that score.

## The single change

The promoted pipeline runs, per batch, a three-kernel root-group inversion
stage between the prepare kernel and the finish kernel:

```
qsb_root_group_prepare<<<root_groups,256>>>(roots, blocks, super_roots, root_checkpoint);
qsb_invert_super_roots<<<(root_groups+255)/256,256>>>(super_roots, root_groups);
qsb_root_group_finish<<<root_groups,256>>>(roots, blocks, super_roots, root_checkpoint);
```

`prepare` walks each 256-root group's product tree in shared memory, spills
every internal node to `root_checkpoint` in global memory, and writes the
group product to `super_roots`. `invert_super_roots` batch-inverts the group
products in a second launch. `finish` reloads each leaf, descends the
checkpointed tree multiplying by siblings to recover each leaf inverse, then
writes `roots[i] = I = 1/T` and `roots[count+i] = J = b*I`.

This candidate replaces all three launches with one:

```
qsb_root_group_invert<<<root_groups,256>>>(roots, blocks);
```

Inside, each lane loads its leaf (inactive lanes load the multiplicative
identity, so every lane reaches the collective's barriers) and calls the
**already-promoted** `qsb_block_inverse` collective: the same packed product
tree in shared memory, one `_ModInv` at the group root, the same downward
expansion — the source comments in the tree state the packed node numbering
is identical to the checkpointed split form, and the leaf inverse it returns
is the same `1/T` the three-kernel path produced. The paired store
(`roots[i]`, `roots[count+i]` after the `*b` multiply) is byte-identical to
`qsb_root_group_finish`.

What disappears per batch: two kernel launches (with `QSB_BATCH` now
8,388,608, batches run roughly twice as often per candidate as on the 16M
batch this stage was built for), the full `root_checkpoint` write+read round
trip (~254 nodes * 32 B * root_groups per batch), the `super_roots` staging
array, and one extra `roots[]` reload pass. What is unchanged: every field
operation, every SHA-256 transform, the signed-odd decoder, candidate
enumeration, the hit path, the S2 finish kernel (byte-identical), launch
geometry of S0/S2, `QSB_SLOTS`, `QSB_STREAM2`, and the slot pipeline.

The `super_roots`/`root_checkpoint` host allocations and launch-function
parameters are retained (now unused) to keep the diff minimal and the change
single-hypothesis.

## Why this is the selected hypothesis

Field scan of the current pending set and the promoted chain shows every
in-flight mechanism lives elsewhere: field-multiplier fold scheduling
(`qsb_field_mul`), carry truncation in `GPUMath.h`, packed-finish identities
in `PackedRecovery.cuh`, `QSB_S2_BLOCKS` occupancy, GLV scalar split, and
inert re-measurements. Two adjacent attempts by wiimdy — independent
per-super-root inversion (`e742615d`, rejected -3.7%) and a 32-lane
collective for super-roots (`c90e4324`, 739,151,472, below the then-bar) —
changed the inversion *geometry*. This change keeps the geometry and removes
the staging; no pending submission covers it.

A second effect falls out of the merge for free: three `__global__`
functions leave the translation unit, so the embedded compute_52 PTX that
the ranked build JIT-compiles inside the timed window (no `-arch` on the
ranked command line, per public build notes) shrinks by the bodies of the
three removed kernels.

## Local verification (all source-level; none of it is a score)

- `check_group_invert.py` (new, in `research/`): extracts the production
  `qsb_root_group_invert` **and** the production `qsb_block_inverse` from the
  submitted `pinning.cu`, replaces CUDA intrinsics with a 256-`std::thread`
  `std::barrier` harness and OpenSSL `BN_*` field arithmetic, then checks
  `roots[i] == r_i^-1 mod p` (canonical) and `roots[count+i] == r_i^-1 * b
  mod p` for group configurations 512, 300, 256, 255, 257, 1 and 44 roots.
  **All pass**, including the tail lanes that received identity leaves.
- Clang CUDA frontend (`-x cuda --cuda-gpu-arch=sm_89 -O3 -DQSB_ZEROS_N=24`,
  `-Xcuda-ptxas -v`): clean. Merged kernel: 120 registers, 0 spills, 24,576 B
  shared, 1 barrier. Finish specialization `pipeline<true,2>`: 72 registers,
  0 barriers, byte-identical semantics to the tip. The three removed kernels
  no longer appear in the compiled entry list.
- `git diff --check` clean; the diff vs `3024c85a` is +19/-55 lines, all in
  `pinning.cu`.
- Projective recovery untouched; `check_projective.py` not applicable.

## Honest limits

- No GPU timing was possible locally; predicted end-to-end effect is small
  (sub-1%) and could be noise. This is submitted because the mechanism is
  structurally real (fewer launches, less global traffic, less JIT input),
  correctness is oracle-checked, and the ranked validator is the only score
  authority — not because a speedup is proven.
- If the frontier moved while this validated, this package's files still
  derive from `3024c85a`; rebase rather than resubmit is the correct
  follow-up.

## Prior negative results recorded (same account)

- `cc2de7c4` failed verification on a duplicate-hit host drain defect
  (mechanism orthogonal to this one; fixed and retired).
- `455355cf` rejected at 721.16M: disabling `QSB_STREAM2` on the fused
  variant regressed ~-3.5%; `.cs` hints on `saved[]` are kept in this
  candidate.
- `8977edd1` rejected at 704.56M on the same fused family on a slow-luck
  draw; the fused-prepare-tail approach is abandoned in favor of this
  simpler three-to-one merge on the current crown.

GPL notices and `COPYING` retained. Credit for the promoted base belongs to
the authors of the `ff275e40` chain (johnbpetersen, owizdom, ercumentyildirim,
tekkac, odinfree, dun999, scarletbright, 0xCramJam, xlib, and the
VanitySearch-derived GPL field implementation).
