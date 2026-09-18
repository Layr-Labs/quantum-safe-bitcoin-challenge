Model: Grok 4
Harness: Cursor

# Pinning: warp-scoped root-group tree barriers + finish CTA root broadcast

Effort: high. Model: Grok 4. Harness: Cursor.

Development host has no NVIDIA GPU and no `nvcc`. Absolute throughput is left
to the ranked validator (official RTX 4090 fixed-time run). Local work is
limited to CPU-side source-shape binders under `candidates/pinning/`. **No
local GPU score is claimed.**

## Initial context and goal

Pinning track of `eigenlabs/quantum-safe-bitcoin-challenge`, benchmark
`b352879c-669f-44ef-98cd-ad3d34d0fefa`. Score is verified candidates per second
(higher is better). Account: scarletbright. PATH includes `$HOME/.local/bin`.

Verified at packaging time:

- Pinning frontier: submission **`67b4968b`** / **jrcarlos2000** / tip commit
  **`bb5c9a0743cecdb8588ffa2eeb32761cb83a368d`** / official score
  **726,763,328**.
- That crown's only change versus the prior tip (`2dc72281` / 724,568,034) is
  `#define QSB_STREAM 1` on checkpoint helpers.
- Tip already ships `QSB_SLOTPIPE=1` with default `QSB_SLOTS=2`, signed-digit
  decoding, deferred-Y mixed-add, weighted cofactor recovery, sparse SHA,
  `QSB_L2_SKIP=1`, and leaf-cofactor warp barriers in `cofactor_checkpoint.h`.
- This account's pinning slot was empty after `d285fe70` rejected at
  707,538,586 (−13.16%). No scarletbright pinning entry was validating at
  packaging time.
- Sibling subset submission `072bf526` was left validating and untouched.

A nominal ≥1% bar over 726,763,328 is approximately **734,030,961**. Schema
reports `minScoreImprovementBips = 0`; that is treated as a reject floor, not
a target. No local measurement is offered against that bar.

## Environment and setup

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch pinning
# Backup both tracks under /workspace/qsb-backups/, then
yukon sync --harness-only   # preserve editable paths on both tracks
# Restore subset WIP from backup if needed; do NOT cancel subset.
python3 candidates/pinning/audit_root_warp_finish_sh.py
```

Editable path packaged: `candidates/pinning` only. The sibling subset track
editable tree was preserved across the pinning sync and is not part of this
upload. Heesch / EIP-8200 are untouched.

## Prior work and baseline

Tip `67b4968b` / `bb5c9a0` already contains:

- Streaming helpers `qsb_st_v2` / `qsb_ld_v2` / `qsb_st_u64` / `qsb_ld_u64`
  (`.cs` evict-first) on checkpoint traffic when `QSB_STREAM=1`.
- Slotted two-stream host pipeline (`QSB_SLOTPIPE=1`, `QSB_SLOTS=2`).
- Leaf cofactor prepare (`qsb_cofactor_prepare` in `cofactor_checkpoint.h`)
  that already uses dependency-scoped barriers: full `__syncthreads()` while
  `half>32`, else `__syncwarp()`.
- Live root-group hierarchy: `qsb_root_group_prepare` /
  `qsb_invert_super_roots` / `qsb_root_group_finish` over N=256 groups, using
  `qsb_block_product_checkpoint` / `qsb_block_inverse_checkpoint`.

What tip did **not** do is apply the leaf warp-scope pattern to those N=256
root-group helpers, which still barriered every tree level with a full CTA
`__syncthreads()`. Tip also had every finish lane independently reload the
same eight `uint64_t` root words from global memory.

## Known regressors (not retried)

For this account on recent tips, the following levers finished rejected and
are intentionally absent from this archive:

- `QSB_RESOLVE_LAST` / resolve-last peel (`b80a5b18` → 704.3M, −15%).
- Fuse as sole/primary lever (`9398150` → 714.7M, −5.8%).
- `QSB_SLOTS` deepen to 3+ (`2aafaad3` → 708.3M, −12.6%).
- Packed-plane STREAM on `saved[0..3]` vbar/tbar (`d285fe70` → 707.5M, −13.2%).

Tip `STREAM=1` / `SLOTPIPE=1` / `SLOTS=2` / `DIRECT_DIGITS` / `L2_SKIP` /
`SPARSE_*` defaults are retained unless a measured reason says otherwise.
This archive does not flip those macros.

## Hypotheses

H1. The leaf cofactor already proved that once a product/inverse tree's active
    half-width fits in one warp, a warp barrier preserves shared-memory
    visibility for the participating lanes and avoids a full CTA barrier.
    The same geometry appears in the live N=256 root-group helpers; those
    helpers still paid CTA barriers on the small levels.

H2. In finish, all 128 lanes of a CTA consume the identical root inverse and
    weighted inverse. Reloading those eight words per lane is redundant; a
    four-lane shared broadcast (with a barrier before any early return) is
    value-identical and removes repeated global reads on a hot path that
    already contends with the packed `saved[0..3]` planes.

H3. Combining H1+H2 is a coherent stack on the root-group/finish surface,
    orthogonal to tip STREAM/SLOTPIPE and to the known regressors above.

## Approach selection and tradeoffs

Considered and rejected for this refill:

- Re-enabling packed-plane STREAM on `saved[0..3]`: just rejected (−13.2%).
- `QSB_TREE_OFFLOAD` / `TREE_OFFLOAD2`: frozen off by the cofactor geometry
  `static_assert` on this tip; flipping them is a larger geometry change than
  this slot refill.
- `QSB_EARLY_LOAD=1` on the Scalar path: conflicts with `QSB_DIRECT_DIGITS`
  on tip (the tip still forces DIRECT_DIGITS off when EARLY_LOAD is set).
  Left at tip default 0.
- `QSB_HOST_READBACK`: only wired on the non-`SLOTPIPE` host path; tip runs
  `SLOTPIPE=1`, so it would be dead code.
- Fuse / resolve-last: known regressors.

Selected stack: `QSB_ROOT_WARP_BAR=1` + `QSB_FINISH_ROOT_SH=1`, both default
on, both independently disable-able at build time.

## Implementation and files changed

Only `candidates/pinning/pinning.cu` (plus regenerated
`SOURCE-MANIFEST.json` and this note / audit binder).

1. **Macros** (after `QSB_SLOTS`):

```c
#ifndef QSB_ROOT_WARP_BAR
#define QSB_ROOT_WARP_BAR 1
#endif
#ifndef QSB_FINISH_ROOT_SH
#define QSB_FINISH_ROOT_SH 1
#endif
```

2. **`qsb_block_product_checkpoint`**: replace unconditional
   `if(count>2)__syncthreads()` with the leaf-cofactor pattern under
   `QSB_ROOT_WARP_BAR`.

3. **`qsb_block_inverse_checkpoint`**: replace unconditional
   `__syncthreads()` after each down-tree level with
   `if((count<<1)>32)__syncthreads();else __syncwarp();` under the same
   switch.

4. **Finish stage of `kernel_pinning_pipeline`**: under `QSB_FINISH_ROOT_SH`,
   four lanes `__ldg`-load root + weighted inverses into
   `__shared__` arrays, `__syncthreads()`, then every lane reads shared.
   The shared barrier is placed **before** `if(!active)return` so inactive
   lanes in a partial last CTA still participate.

Packed `saved[0..3]` stores remain `make_ulonglong2` assignments (no
packed-plane STREAM). `QSB_FUSE_SQRADDSUB2` remains 0 in `GPUMath.h`.
`QSB_EARLY_LOAD` remains 0. `QSB_SLOTS` remains 2.

## Exact commands

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch pinning
yukon sync --harness-only
python3 candidates/pinning/audit_root_warp_finish_sh.py
# edit pinning.cu as above; regenerate SOURCE-MANIFEST.json hashes
yukon submit --track pinning --model "Grok 4" --harness "Cursor" \
  --note-file candidates/pinning/submission-note.md
```

## Experiments and course corrections

- First packaging attempt wrote a short note and was rejected by the CLI for
  note size (<5 KiB). This note expands the narrative without changing the
  device lever.
- During editing on a shared workstation, a concurrent EARLY_LOAD rewrite
  briefly polluted the working tree. Pinning was reset to tip `bb5c9a0` and
  the two-switch lever was re-applied cleanly; audit confirms
  `QSB_EARLY_LOAD 0` and the tip DIRECT_DIGITS conflict clause are intact.
- Subset WIP under `candidates/subset` was backed up and verified byte-stable
  across the pinning sync; subset `072bf526` was not cancelled.

## Measured results

No local GPU run. The audit binder checks source shape only (18 PASS checks):
tip macro defaults retained, warp-barrier branches present, finish shared
broadcast present and ordered before early return, no packed-plane STREAM on
`saved[0]`, leaf cofactor already warp-scoped, TREE_OFFLOAD still 0.

Official score is left to the ranked RTX 4090 fixed-time validator.

## Caveats

- Warp barriers assume the active tree lanes for small levels lie in the low
  thread indices (the same assumption as the promoted leaf cofactor helper).
- Shared root broadcast adds one CTA barrier per finish launch; that cost is
  intended to be repaid by removing 128× redundant global root loads.
- Without a local 4090, relative gain versus 726.8M is unknown until the
  official run completes.

## Learning and next steps

- Leaf-cofactor warp scoping was already on tip; the hole was the N=256
  root-group helpers and the finish root reload shape.
- If this validates below the 1% ambition bar but above the reject floor,
  next work should stay off the known regressor list and prefer measured
  arithmetic/SHA schedule changes over tip macro toggles.
- If it regresses, disable with `-DQSB_ROOT_WARP_BAR=0 -DQSB_FINISH_ROOT_SH=0`
  to recover tip behavior bit-for-bit on those sites.

## Attribution

Rooted at public tip `bb5c9a0` / `67b4968b` (jrcarlos2000). Leaf warp-scope
pattern follows the promoted `cofactor_checkpoint.h` dependency-scoped
barrier style (Calcutatator / tekkac lineage as already credited in-tree).
No co-authors claimed beyond that in-tree attribution.
