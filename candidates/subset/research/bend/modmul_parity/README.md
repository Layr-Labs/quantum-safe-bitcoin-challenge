# Modular-product parity synthesis

The ranked recovery path computes two complete field products whose only
observable output is the canonical Y parity used as the compressed-key prefix.
This experiment asks whether secp256k1's pseudo-Mersenne reduction permits a
smaller parity-only primitive.

For `p = 2^256 - c`, `c = 2^32 + 977`, write `N = L + 2^256 H`, then fold:

```text
T = L + cH = T0 + 2^256 T1
U = T0 + cT1
```

For canonical inputs, `0 <= U < 2p`, so the canonical residue is either `U` or
`U-p`. Because `c` and `p` are odd, its parity is exactly:

```text
parity(L) XOR parity(H) XOR parity(T1) XOR (U >= p)
```

`MODEL.bend` treats the four terms as synthesis features in the reduced field
`p = 2^8 - 5`. It exhaustively evaluates every canonical pair for all sixteen
feature masks. `check_full.py` independently screens boundary values and
200,000 deterministic full-width pairs against Python big integers, retaining
the first counterexample for every incomplete mask.

This establishes a concrete CUDA research target, not a speedup claim. A CUDA
prototype must still show that computing the necessary product halves, fold
carry and comparison is cheaper than the existing full `_ModMult`, and must be
checked against the production reduction near `p` before integration.
