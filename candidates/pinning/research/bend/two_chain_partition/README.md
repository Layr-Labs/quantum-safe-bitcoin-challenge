# Two-chain partition law

## Decision target

The promoted pinning source at commit `067302c` uses one 15-point signed-window
XYZZ chain. This artifact checks the first prerequisite for a balanced 8+7
split: the two chains contain every original signed, shifted term exactly once,
and merging their results reconstructs the monolithic schedule.

The CUDA source identities used to derive the law are:

- `GT_CHUNKS = 15`;
- widths `[18,17,...,17]`;
- shifts `0` for chunk 0 and `17*c+1` for chunks 1..14, hence
  `[0,18,35,52,69,86,103,120,137,154,171,188,205,222,239]`;
- every table point is the complete signed term for its chunk.

Promoted `pinning.cu` SHA-256:
`f0de8bce0750ce9cf9a6bdd3b9fc5b5097a88d8a0a3493a81874218e3b8e379b`.
Promoted `GPUMath.h` SHA-256:
`b22af2e24f8c2aa503b623447acf9292c44b6c68452aa22ab578c191e238cd0b`.

## Scope and assumptions

Each `Nat` in the theorem denotes an already signed and shifted group term.
The proof establishes schedule/partition preservation only. It does not prove
secp256k1 exceptional additions, XYZZ coordinate invariants, the deferred-Y
anchor transformation, CUDA synchronization, register use, or performance.

Those exclusions are important: two independent chains need two exact XYZZ
states and one final projective merge. The present production deferred-anchor
formula links each step to the preceding affine Y, so each chain must seed and
resolve its own anchor before merge. That adds a second prefix seed, a second
resolution multiply, the merge, and roughly one extra live XYZZ accumulator.

## Commands

```sh
/Users/nolan/.bend/bin/bend --version
/Users/nolan/.bend/bin/bend guide
cd candidates/pinning/research/bend/two_chain_partition
/Users/nolan/.bend/bin/bend PROOF.bend
/Users/nolan/.bend/bin/bend BROKEN.bend
```

Expected: `PROOF.bend` prints `All terms check.`; `BROKEN.bend` is rejected
because the frontier law remains open at `?dropped_chunk_8`.

## Performance gate

The split removes no field operation. Relative to the 95M+28S single chain it
adds another chain seed/resolve boundary plus a projective merge and increases
live state. Its only potential benefit is instruction-level overlap. Therefore
the next CUDA experiment is justified only if an sm_89 compile shows no spills
and enough occupancy, followed by an NVIDIA A/B timing. The Bend result alone
is not performance evidence and is not a submission candidate.

## Native resource gate

An exact `067302c` CUDA 12.8.93 `sm_89` build subsequently measured the live
prepare specialization at 128 registers/thread with zero spills and 8 KiB
shared memory per 128-thread block. A second simultaneously live XYZZ state is
four 256-bit fields, or at least 32 additional 32-bit register-equivalents; its
deferred affine-Y anchor adds eight more before any merge temporaries.

Moving only that 160-byte/thread state to shared memory would add 20 KiB per
block, taking the known allocation to at least 28 KiB before merge scratch and
adding repeated shared traffic to the point chain. Keeping it in local memory
would create exactly the spills the current frontier avoids. Serially computing
the chains with reused registers forfeits the intended instruction overlap.

Therefore the naïve two-XYZZ-state implementation is rejected at the current
geometry. The direction becomes viable only with a compact second-chain
representation, cross-thread ownership, or a merge construction that removes
enough work to change the resource balance. See `resource-gate.json`.
