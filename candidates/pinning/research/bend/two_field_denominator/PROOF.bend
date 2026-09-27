import Base
import ./LAWS.bend as Laws

def Laws.xyzz_denominator_cross_product(a, v, d, invariant):
  Equal.cong(
    Nat,
    Nat,
    x => Nat.mul(x, d),
    Nat.mul(v, v),
    Nat.mul(a, Nat.mul(a, a)),
    invariant)
