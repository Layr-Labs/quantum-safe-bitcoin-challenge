# Two-field checkpoint denominator law

This independently checks the algebraic prerequisite used by pending pinning
submission `31e98e4`: with `A=ZZ`, `V=ZZZ`, the XYZZ invariant is `V²=A³`.
The proposed denominator changes from an `A²*d` formulation to `V*d`, using
the ratio identity

```text
A / (V*d) = V / (A²*d).
```

Cross multiplication reduces that claim to `A³*d=V²*d`, which follows from
the coordinate invariant without assuming division. `LAWS.bend` states that
polynomial identity and `PROOF.bend` proves it by congruence. `BROKEN.bend`
contains a deliberate wrong-exponent mutation and must be rejected.

Commands, using official Bend 2.0.5 installed from bend-lang.com:

```sh
cd candidates/pinning/research/bend/two_field_denominator
/Users/nolan/.bend/bin/bend PROOF.bend
/Users/nolan/.bend/bin/bend BROKEN.bend
```

The theorem does not prove that `V*d` or `A²*d` is nonzero in secp256k1, nor
does it validate CUDA tree traversal, canonical representatives, memory layout,
or performance. Those remain explicit runtime/source obligations.
