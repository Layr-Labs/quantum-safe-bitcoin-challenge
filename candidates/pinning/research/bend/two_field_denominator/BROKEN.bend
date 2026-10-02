import Base
import ./LAWS.bend as Laws

# Deliberate mathematical fault: replaces the required A^3 term with A^2.
# The checker must reject this attempted proof.
def Laws.xyzz_denominator_cross_product(a, v, d, invariant):
  ?wrong_a_exponent
