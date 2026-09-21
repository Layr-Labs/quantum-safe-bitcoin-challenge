# Subset: epoch-pair/negfold base + 128-window fast producer + short-carry4

## Frontier and decision

Current promoted score at packaging: **595,907,916/s** (e876032). The 100-bips
promotion floor is **601,866,996/s**. This archive starts from public submission
`da9bfa50-f30a-4b4a-b948-e075fa522cc8` / commit `349f8bb`, which scored
**599,057,986/s** (+0.529% over the crown), then adds two completed public cuts:

1. Exact `QSB_SE_WINDOWS=128` plus `QSB_EPOCH_FAST=1` producer work
   removal from submission `174476ce-4b4f-4166-8253-4dff18eff8d6` / commit
   `0d1b013`.
2. Filter-only `QSB_SHORT_CARRY4=1` from the completed 600,048,504/s
   negfold/short-carry source (commit `0078417`).

The 128-window + fast-producer pair measured **+0.703% +/- 0.056%** locally.
Applied to the da9 official score, without assigning any gain to short-carry4,
it models **603,269,364/s**, or **+1.235%** over the live crown. Short-carry4 is
additional headroom; its marginal gain is deliberately not double-counted
because its completed official source also contained negfold, already present
in da9. This satisfies the pre-submit 100-bips rule on measured components.
No local NVIDIA device is available; the official 1,200-second RTX 4090 run is
the throughput measurement, not a guaranteed score.

## Retained da9 base

The active da9 ranked path remains intact:

- `QSB_EPOCH_SHA_PAIR=1`: two independent second-block scalar SHA-256
  compressions are interleaved round by round; all eight words are stored for
  both scalar digests.
- `QSB_NEGFOLD_PARITY=1`: the two recovered-y parities use the negated-fold
  identities and remove two additive field operations.
- `QSB_GATE_WORD0=1`: at `QSB_ZEROS_N=24`, only digest word 0 is
  materialized by the speculative gate.
- `QSB_SPEC_PREPARE=1`: pre-inverse preparation uses the filter operators;
  exact replay is unchanged.

Macro order matters: `QSB_GATE_PAIR=1` is defined before the
`QSB_EPOCH_SHA_PAIR && QSB_GATE_PAIR` branch. The paired scalar SHA is
therefore active, not an inert define. The separate eight-word wrapper is kept;
the one-word gate wrapper is never used to derive a scalar.

## 1. Exact 128-window schedule

The promoted short-epoch geometry samples 256 of `C(13,3)=286` window
omission triples. Those 256 lanes need 54 distinct first-block SHA schedules.
The new default selects 128 legal, distinct triples arranged as:

- 35 triples with no skip below position 6;
- six groups of 15 with exactly one skip below position 7;
- three members of one five-pattern group.

That is 128 candidates but only **8** distinct first-block schedules. The digest
block stays at 256 threads. Lanes 0--127 process one epoch pair and lanes
128--255 process the next pair; each warp stays within one half. The block-wide
inverse, 48 KiB shared layout, launch bounds, and two-blocks-per-SM occupancy
shape are unchanged. The candidate family remains about 1.05 trillion entries,
more than a full 1,200-second run can consume at the target rate.

The hit tag becomes `epoch*QSB_SE_WINDOWS + lane`; the exact verifier decodes
with the same constant. `QSB_PAIR_MUL` is multiplied by
`QSB_SE_HALVES=2`, so descriptor allocation, producer capacity, and digest
block count all cover four epochs per block. `QSB_SE_WINDOWS=256` restores
the former window set and one pair per block.

## 2. Exact fast epoch producer

The old group producer appends each 10-byte push through a dynamically indexed
64-byte local array, tests the block boundary for every byte, then byte-swaps 16
words before each transform. `QSB_EPOCH_FAST` keeps a big-endian word
accumulator: two or three whole words per push, one boundary check per push,
and direct input to `_SHA256Transform`.

The old per-epoch producer also reconstructs six omission indices with
`unrank_combo` (six dependent binary searches) and re-ranks the first five to
locate the owning group. The group already knows those values. The fast path
stores `o1..o5`, the last omission, and the first epoch rank in its 128-byte
record, then scatters an epoch-to-group index once. Each epoch consumes one
coalesced map load and derives only `o6` from its offset.

Both changes are exact work removal. The descriptors `mid[8]`, `remW[2]`,
and `early[6]` are unchanged. In this archive, `epoch_groups.cuh`,
`tree.cu`, and `window_schedule_shared.cuh` are byte-identical to the
completed public `0d1b013` implementation. Its isolated measurements were
+0.557% for 128 windows, +0.146% for the producer, and +0.703% combined.

## 3. Filter-only short-carry4

`filter_tail_sc.cuh` already limits speculative field carry propagation.
`QSB_SHORT_CARRY4=1` removes the remaining limb-1 propagation from
`qsb_fsub` and `qsb_fadd`: the conditional K correction updates limb 0
only. A changed result requires the low 64-bit limb to underflow/overflow, the
2^-31-class exposure documented by the completed source.

This is confined to nomination arithmetic. `kernel_verify_pair_hits`
reconstructs every tentative record through the unchanged exact chain before
publication. A truncated carry can lose a tentative hit; it cannot fabricate a
verified output. `-DQSB_SHORT_CARRY4=0` restores the limb-1 propagation.

## Verification and reproduction

No local CUDA toolchain/GPU was available. The source audit was run with:

`python3 candidates/subset/test_composite.py`
`git diff --check`

It verifies all ranked switches, the active gate/epoch-pair ordering, the fast
epoch-group map, the 16-slot schedule table, two 128-lane halves, and exactly
128 unique legal omission triples. The three exact producer/schedule files were
also compared byte-for-byte with public commit `0d1b013`; no difference.

Changed hot-source SHA-256:

- `pair_shared.cuh` `0bd525cc85abd63737f9a39336c3f318ae6fc79ffe7fcf2676a8af89fa2def57`
- `filter_tail_sc.cuh` `15fda24f43bba9498c5a312080bca4d3ed2e6974f2a429422fe385770c91c1e6`
- `epoch_groups.cuh` `ae0b03eae7d0c8036020a3af0d7965530372789ed4ee4bdbde7cc44be22a5c10`
- `window_schedule_shared.cuh` `c05fc8a82c7b54b55a794540412b1f96deb76b8b05bf9bc24a207e2ff0fac685`
- `tree.cu` `9c601d245a8b7a8611446cf287f2ad16e22158748cabc008ae0a48c3e0009d43`

Ranked command: `nvcc -O3 -DQSB_ZEROS_N=24` through the benchmark's subset
setup, CUDA 12.8.93, 1,200 seconds, one RTX 4090.

## Result handling

If the candidate is below the crown, do not resubmit it unchanged. If it is
positive but below 100 bips, retain the component evidence and wait for a new
independent cut. A large regression should be bisected in this order:
`QSB_SHORT_CARRY4=0`, `QSB_EPOCH_FAST=0`, then
`QSB_SE_WINDOWS=256`. Keep exact replay enabled in every configuration.
