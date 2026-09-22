## Mathematical construction

Let W=2^32. Write a as eight words a_i and let b be an unsigned 256-bit
integer. Form eight independent nine-word rows

```
P_i = a_i * b
full_product = sum(P_i * W^i, i=0..7)
```

Each row is a 32-by-256 multiply. Starting with carry c=0, process b's words
from least to most significant:

```
t = a_i*b_j + c
row[j] = t mod W
c = floor(t/W)
row[8] = c after the eighth product
```

For arbitrary word inputs and a word carry,

```
0 <= a_i*b_j+c <= (W-1)^2+(W-1) = W^2-W < W^2.
```

Consequently the entire cell fits in an unsigned 64-bit destination. The
first cell uses `mul.wide.u32`; subsequent cells use `mad.wide.u32` with the
zero-extended incoming carry. The returned low word becomes a row digit and
the returned high word becomes the next carry. No overflow bit is omitted:
the bound proves a 65th bit does not exist. The row recurrence also gives
the stronger reachable bound c<a_i when a_i is nonzero; the a_i=0 case
produces an all-zero row without any special branch.

After the first row, each succeeding shifted row is merged into the current
prefix with a complete nine-word 32-bit carry chain. There are seven such
merges. The old prefix has exactly the needed eight overlap words; the last
word starts at zero and receives the new row's high word plus incoming carry.
After row i the accumulated integer is `(a mod W^(i+1))*b`, which is strictly
less than W^(i+9). This proves there is no omitted carry beyond the final
word of that merge. At i=7 the complete sixteen-word, 512-bit product is
available. Zero operands, all-one limbs and long inter-row carry chains are
included in the construction and checks.

The bound is the basis for fusion, not a statistical approximation. Every
one of the original 64 partial products remains. Eight are plain wide
multiplies and 56 are wide multiply-adds. The design separates independent
row construction from their shifted accumulation, permitting a shorter
source dependency graph than a single long in-place multiplication chain.
It does not introduce lane exchanges, shared memory or new kernel stages.

The instruction semantics used in the model and emitted source are documented
in NVIDIA's [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).
In particular, the wide form has a 64-bit destination/addend for 32-bit
multiplicands. The integer bound above is why its finite-width addition is
sufficient for these cells.

## Preserve the promoted raw fold and surrounding pipeline

`RowMac256.cuh` emits the complete product construction followed by the
unchanged promoted C31 reduction text. Scratch names are isolated from the
reduction's registers. Its four result limbs therefore preserve the promoted
raw product output, not merely a congruent field residue. This release adds
no carry omission, exceptional-point skip, normalization shortcut or change
to a verification boundary. It retains the inherited reduction choices.

`GPUMath.h` adds `QSB_ROW_MAC256`, default one, and dispatches the device
`_ModMultCore` to the helper only in the active C31/short-carry configuration.
`QSB_ROW_MAC256=0` restores the original promoted path. The non-C31 path,
non-short-carry path and host transcription remain original. Both in-place
and out-of-place callers are supported: all eight input limbs are captured
before any output operand is written. The separate carry-complete root-group
multiplier, seeded negative-Y MAC, square and fused square routines retain
their promoted code.

The main search source and all other existing production headers are copied
byte for byte from the current promoted baseline. Scalar enumeration, table
layout, block geometry, cofactor association, parity fallback, SHA behavior,
exact host gate and publication remain unchanged. No runtime comparison,
autotuning path, extra GPU allocation or timing-dependent correctness choice
is added. Only the Pinning editable path is submitted; protected harness,
scorer and verifier files are not modified.

## Research used to select this complete implementation

The preceding research first explored 32-bit low/high multiply-add chains,
then bounded wide multiply-add cells with a carry and an old result word.
That larger cell also fits in 64 bits, but its two-word seed can require
33 bits. We implemented exact precombined seeds, retaining their top bit,
and tested them against Python integers. This reduced repeated zero
extensions but could lengthen the carry dependency chain.

We then returned to ordinary schoolbook products and evaluated all 128
ordered partitions of eight operand words. Multiple-word row groups trade
fewer zero-extension packs against longer in-place chains. Independent
single-word rows eliminate old-result additions inside each row; all
accumulation instead happens in the explicit shifted merge. This gave the
best dependency depth in that static partition census and is the selected
source. The 33-bit-seed research is not silently included in production;
the selected rows need only their 32-bit forwarded carry.

Alternative four-tile 128-bit schoolbook constructions were implemented and
checked as well. Their extra live cross products and merge costs did not
provide a better combined source tradeoff. We did not choose a variant just
because its C++ or PTX listing had fewer lines.

The following is a product-prefix source model, before the identical fold.
The arithmetic column counts a 64-bit add/sub as two word operations and a
32-bit operation as one. The cost column assigns two units to a wide
multiply/MAC, one to each word arithmetic operation and treats pack/unpack
moves as aliases. These weights are sensitivity assumptions, not native
instruction counts or device cycle measurements.

| Product prefix | Wide products including MAC | Other word arithmetic | Abstract cost | Peak live words | Predicates | Dependency depth |
|---|---:|---:|---:|---:|---:|---:|
| Promoted even/odd schoolbook | 64 | 134 | 262 | 40 | 0 | 32 |
| Rejected signed-middle Mul48 | 48 | 152 plus 2 predicate ops | 250 | 41 | 2 | 35 |
| Selected independent word rows | 64 | 63 | 191 | 32 | 0 | 31 |

The selected source replaces 56 standalone product-plus-accumulation sites
with bounded wide multiply-adds and has seven full nine-word merges. Its
lower word arithmetic and live-value counts are reasons to test it across
the frequently used ordinary multiply path. They are not throughput results.

The principal uncertainty is physical register packing. The 56 wide-MAC
carry seeds are zero-extended 32-bit values presented as 64-bit operands.
The alias-free source model cannot establish their actual register moves or
pair alignment. Charging one extra move per seed changes the selected cost
from 191 to 247, versus 262 for the promoted model. Charging two changes it
to 303. Added move latency can also lengthen the critical path. Conversely,
the compiler may simplify some packs or fuse the promoted path differently.
Native fusion, spill behavior, occupancy, instruction cache and surrounding
point state must all be decided by actual device code and measurement.
A source peak of 32 words does not claim 32 hardware registers or an
occupancy increase. The official run is needed to determine the net effect.

This is a new full multiply implementation and a concrete response to the
failed signed-middle composition. It is not a remeasurement of that source
or a standalone launch-policy or dead-parameter edit. It also does not
assume the ideal 262-to-191 model translates to a corresponding speedup.

## Checks actually run

The initial carry-seed research passed 6,816 full 512-bit input pairs for
six variants. Omitting the 33rd seed bit caused 5,137 failures, confirming
that the supposedly minor bit is essential. The selected direct schoolbook
family then passed 4,513 full-width input pairs for each of fifteen tested
partition variants, including all selected static frontier points. Cases
include extreme 128-bit half Cartesian combinations, 2,048 deterministic
random full-width pairs and prime-adjacent values. Every product was compared
with Python's exact integer multiplication. The three four-tile alternatives
also passed their own 4,513 pairs.

The actual complete selected PTX plus promoted raw fold passed 1,145
boundary/random pairs against the actual original `_ModMultCore` PTX. The
literal text extracted back from the generated header equals the interpreted
text. After copying into the isolated release, the packaged
`test_rowmac.py` checks another 1,193 source-bound full-product/raw pairs,
checks the enabled dispatch and unchanged fold text, and runs negative
controls that remove row carries and merge carries. Both mutations are
required to fail. A directed maximal allowed row-cell case checks the wide
MAC boundary. The existing host-gate Python test checks 64 SHA256d midstate
samples, problem layout, recovery agreement with the verifier and the source
publication gate.

The source review checks include closure, unchanged inherited production
files, and restoration of the original `GPUMath.h` by removing only the new
dispatch/include block. Archive scope, hashes and `git diff --check` are
checked before upload. These are Python semantic and source checks. They
are not native compilation, device execution, race detection, SASS counts,
register allocation reports or measured throughput.

Reproduce the packaged checks from the repository root:

```
python3 -B candidates/pinning/test_rowmac.py
python3 -B candidates/pinning/test_host_gate.py
```

`rowmac_research/` contains the narrow PTX integer interpreter, selected
product prefix, promoted product and raw references, and the cost record.
`RowMac256.cuh` SHA-256 is
`c8927e4c91cac2908634f3c447888b9392c3ee3a3b67acf0e9d8aa61eb123e92`.
The reference product was copied from the promoted source independently of
the row-MAC generator. No local native compiler or GPU benchmark was run.

