import Base

# Executable field F_17 model for a three-candidate compound inverse leaf.
# Zero denominators are replaced by one in the shared product and masked back
# to zero after splitting, matching the CUDA unusable-candidate contract.

def mul(+a: U32, +b: U32) -> U32:
  ((a * b : U32) % 17 : U32)

def inv(+a: U32) -> U32:
  +a2 = mul(a, a)
  +a4 = mul(a2, a2)
  +a8 = mul(a4, a4)
  mul(mul(a8, a4), mul(a2, a))

def safe(x: U32, z: Bool) -> U32:
  match z:
    case False{}:
      x
    case True{}:
      1

def masked(x: U32, z: Bool) -> U32:
  match z:
    case False{}:
      x
    case True{}:
      0

def split3(+a: U32, +b: U32, +c: U32) -> U32 & U32 & U32:
  +za = U32.is_eq(a, 0)
  +zb = U32.is_eq(b, 0)
  +zc = U32.is_eq(c, 0)
  +aa = safe(a, za)
  +bb = safe(b, zb)
  +cc = safe(c, zc)
  +ab = mul(aa, bb)
  +abc_i = inv(mul(ab, cc))
  +ab_i = mul(abc_i, cc)
  ia = masked(mul(ab_i, bb), za)
  ib = masked(mul(ab_i, aa), zb)
  ic = masked(mul(abc_i, ab), zc)
  (ia, ib, ic)

law three_nonzero:
  {split3(3, 5, 7) == (6, 7, 5) : U32 & U32 & U32}

law zero_isolated:
  {split3(0, 5, 7) == (0, 7, 5) : U32 & U32 & U32}
