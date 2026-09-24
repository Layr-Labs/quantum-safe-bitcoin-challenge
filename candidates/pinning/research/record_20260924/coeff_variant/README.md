# Exact high-half GLV coefficient variant

This isolated copy changes only `GLVScalar.cuh` relative to the checked-out
production source. `QSB_GLV_HIGH10_HI=0` retains the current coefficient body;
`=1` replaces the five full products and carry accounting of diagonal 10 by
five high-half products and a 64-bit sum. Existing `test_glv_coeff.py` and
v20 notes in the repository document an earlier version of this mechanism;
this is an exact re-port onto the current four-bank/residual129 source.

Let B = 2^32. The original diagonal-10 carry is floor(D10/B). The new
carry is sum(floor(a_i*b_j/B)) over i+j=10. Their difference is at most four.
The fixed reciprocal constants additionally bound all omitted diagonals
0 through 9. The oracle computes a conservative bound directly from these
production constants, including five (B-1) low halves: the full product
exceeds the new lower estimate by less than 9*B^11 for g1 and 8*B^11 for g2.

Thus only a 9-word (g1) or 8-word (g2) band immediately below word-11's
half threshold can change rounding to bit 384. These cases use the existing
out-of-line full-product fallback. Values above the half threshold are safe
even if the omitted positive carry crosses the word boundary: the carry
into the coefficient replaces the previously selected rounding increment.
The guards are 0x7ffffff7 and 0x7ffffff8. Every returned coefficient remains
exact; this change does not spend a missed-hit budget.

`python3 -B test_glv_coeff.py` executes the actual production C++ coefficient
control flow with the single `mul.hi.u32` instruction translated into its
CPU unsigned product-high semantics (the exact PTX string is asserted).
Both flag settings are checked against independent Python integer rounding.
GCC UndefinedBehaviorSanitizer is enabled. Directed cases include coefficient
half thresholds, guard boundaries, word-11 wraparound, all-high and sparse
limbs, plus 100,000 random scalars per reciprocal. Final result: 300,818
cases per reciprocal, two implementations, zero mismatches. Adversarial
cases deliberately trigger the rare fallback; their observed fallback
fraction is not a predicted ordinary-run fraction.

No GPU execution or throughput was measured locally. Native compilation and
register/SASS comparison are separate screening evidence. Benchmark harness,
problem data, publication gate, table geometry, and scalar residual code
are unchanged.
