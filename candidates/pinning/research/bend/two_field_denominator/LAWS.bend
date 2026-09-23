import Base

# Algebraic core of the two-field checkpoint substitution.
# A denotes ZZ, V denotes ZZZ, and valid XYZZ coordinates satisfy V^2=A^3.
# d is the recovery factor. Equality is cross-multiplied, so the law does not
# silently assume either denominator is invertible.

def square_scaled(+v: Nat, +d: Nat) -> Nat:
  Nat.mul(Nat.mul(v, v), d)

def cube_scaled(+a: Nat, +d: Nat) -> Nat:
  Nat.mul(Nat.mul(a, Nat.mul(a, a)), d)

law xyzz_denominator_cross_product:
  for a: Nat
  for v: Nat
  for d: Nat
  for invariant: {Nat.mul(v, v) == Nat.mul(a, Nat.mul(a, a)) : Nat}
  {square_scaled(v, d) == cube_scaled(a, d) : Nat}

# The field-level consequence, under explicit nonzero denominator assumptions,
# is A/(V*d) = V/(A^2*d). This law proves its polynomial prerequisite only;
# inversion and canonical field representatives remain outside this Nat model.
