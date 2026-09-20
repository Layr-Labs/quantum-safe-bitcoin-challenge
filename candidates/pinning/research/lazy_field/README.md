# Redundant-field feasibility screen

Decision: retain the idea for research, but do not select a production rewrite
from the current evidence. This follows Muse round five and is independent of
the already rejected Karatsuba schedule. PR74 production remains unchanged.

`screen.py` extracts the actual seed and deferred mixed-add formulas from the
current `GPUMath.h`, verifies the ten-window caller, and propagates conservative
magnitude contracts. No alternative field primitive is implemented. Source
hashes, every operation and the resulting bounds are in `screen-results.json`.

The contracts come from [Bitcoin Core's field wrapper](https://github.com/bitcoin-core/secp256k1/blob/master/src/field_impl.h)
and [5x52 implementation](https://github.com/bitcoin-core/secp256k1/blob/master/src/field_5x52_impl.h):
multiply/square accept magnitude at most8 and return1; addition sums magnitudes;
negation of magnitude m has bound m+1. Subtraction is modeled as add plus
negation. Weak normalization returns magnitude1 while still propagating carries.
The bound argument is conditional on implementing these contracts correctly.

A literal formula conversion exceeds the multiply/square input contract in
three operations: seed `Q*R`, then `P^2` and `PP*P` in the next addition.
Normalizing seed `T` immediately before `Q-T` fixes all three, without further
normalizations within the ten-point chain. This is a concrete schedule, not a
claim that carries may simply be omitted. The abstract chain counts are
60multiplications,18squares,16adds,47subtracts and one proposed weak normalize.

The following are conservative schoolbook arithmetic counts, not SASS or
throughput. They exclude reduction, folding, carries and conversions. Full
wide products are decomposed into narrow partial products; diagonal-specific
or alternative multiplication algorithms could change those counts.

| Representation | 32-bit storage words/field | Max input limb bits | Max raw convolution column bits | Raw narrow products for chain |
| --- | ---: | ---: | ---: | ---: |
| Current8x32, specialized square |8|32|carry-managed|4488|
| Proposed5x52 |10|56|114|7080|
| Proposed9x29 |9|33|69|5670|
| Proposed10x26 |10|30|64|6990|

The9x29 layout violates its32-bit limb storage under this conservative contract.
It requires tighter bounds or extra normalization; its shown product count is
therefore not a ready implementation. For10x26, the raw convolution fits64bits,
but folding could exceed that width and requires a separate proof. The5x52
schedule needs wide intermediates and extra physical storage. None of these
counts proves a measured slowdown, nor does a reduction in carry operations
establish a net gain. An actual efficient multiply/reduction schedule and a
source-matched full-kernel native comparison would be needed before selection.

Recovery, inverse, zero/parity tests, table decoding and checkpoint boundaries
are outside this screen. A complete representation change must handle them.
No third-party implementation was imported; this is a contract-based analysis
of our existing formulas. No GPU execution or timing was performed.

Reproduce from the benchmark directory:

```sh
python3 -B candidates/pinning/research/lazy_field/screen.py
```

The script fails on unknown formula statements, a changed caller/geometry,
unexpected contract violations, or changed operation counts. Its negative
check preserves the three literal-conversion violations; the proposed
single-normalize variant must satisfy all multiplication magnitude contracts.
