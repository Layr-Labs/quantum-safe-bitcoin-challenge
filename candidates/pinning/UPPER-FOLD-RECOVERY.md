# Conditional upper fold and direct recovery distance

The modulus is `p = 2^256 - 2^32 - 977`. The point multiplication and square
retain the complete pseudo-Mersenne reduction. After the low 96 bits of the
last fold, an `addc` captures the actual carry. If this bit is zero, adding
zero to each remaining limb leaves the upper 160 bits unchanged and cannot
produce another carry. That branch can therefore skip the upper five words.
If the bit is one, the code propagates it through all five upper words and
captures their actual final carry. A nonzero final carry still adds
`2^32 + 977` into the low words. No potential carry is assumed absent.

The three paths are low-fold completion, upper propagation without a final
carry, and upper propagation followed by the final correction. Directed
inputs near `p - 2^k`, for `k` in 48, 64, 80, 96 and 112, exercise the upper
propagation path. Values near `p - 65537` exercise the final correction.
The source audit executes the actual extracted PTX and verifies all three
paths for both multiply and square against Python integers. The native
multiply/square fixture includes all 125 additional upper-carry input pairs.

The recovery change uses the field identity

```
d = S * (c - lambda)
x = a - d
lambda * (a - x) - b = lambda * d - b  (mod p)
```

The previous sequence computed `x = S*(lambda-c)+a` and later recomputed
`a-x` for parity. Keeping `d` through the parity product removes two modular
additions across the two recovered points. The second parity uses the reverse
difference `b - mu*d2`. The existing canonical multiplication boundaries and
parity normalization remain in force. The canonical-RHS subtraction helper
is used only with its established canonical second argument. The existing
point and table contracts are documented in `CANONICAL-CONTRACTS.md` and
`FINAL-WINDOW-PROOF.md`.

The exact-source native recovery audit compares both formulations with an
independent integer oracle on 16,989 canonical and carry-boundary cases.
Both x coordinates must equal the canonical integer outputs exactly; the
checker does not reduce an incorrect returned x coordinate before comparing.
Both y parity bits must also match.

Provenance: the upper-fold scheduling idea also appears in public submission
`8a51e019`; this implementation preserves the complete corrected arithmetic.
The direct-distance identity comes from public submission `e4efcc74`, commit
`1eea69acec2e056d9cf7e59fc6b09067cc16bf56`. Paired product carries follow
public `208bbcb6`, commit `a668c4e5fd80db398c13222453f9c9a645612649`, as
explained in `PAIRED-CARRY-PROOF.md`. Existing GPL notices and `COPYING` are
preserved. The organizer generator, verifier and scoring contract are intact.
