import Base

type Stats is Data:
  MkStats{checked: U32, failures: U32, witness: U32}

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

def bool_u32(x: Bool) -> U32:
  match x:
    case False{}:
      0
    case True{}:
      1

def expect(x: U32, z: Bool) -> U32:
  match z:
    case False{}:
      inv(x)
    case True{}:
      0

def make_leaf(i: U32, bad: U32, is_bad: Bool) -> Stats:
  match is_bad:
    case False{}:
      MkStats{1, bad, 0}
    case True{}:
      MkStats{1, bad, (i + 1 : U32)}

def evaluate(+i: U32, +a: U32, +b: U32, +c: U32, result: U32 & U32 & U32) -> Stats:
  (ia, ib, ic) = result
  +ea = expect(a, U32.is_eq(a, 0))
  +eb = expect(b, U32.is_eq(b, 0))
  +ec = expect(c, U32.is_eq(c, 0))
  +bad = (bool_u32(U32.is_ne(ia, ea)) + bool_u32(U32.is_ne(ib, eb)) : U32)
  +bad2 = (bad + bool_u32(U32.is_ne(ic, ec)) : U32)
  make_leaf(i, bad2, U32.is_ne(bad2, 0))

def checked_leaf(+i: U32) -> Stats:
  +a = (i % 17 : U32)
  +b = ((i / 17 : U32) % 17 : U32)
  +c = ((i / 289 : U32) % 17 : U32)
  evaluate(i, a, b, c, split3(a, b, c))

def leaf(i: U32, in_range: Bool) -> Stats:
  match in_range:
    case False{}:
      MkStats{0, 0, 0}
    case True{}:
      checked_leaf(i)

def merge_with(+lc: U32, +lf: U32, lw: U32, +rc: U32, +rf: U32, rw: U32, left_has: Bool) -> Stats:
  match left_has:
    case False{}:
      MkStats{(lc + rc : U32), (lf + rf : U32), rw}
    case True{}:
      MkStats{(lc + rc : U32), (lf + rf : U32), lw}

def merge(l: Stats, r: Stats) -> Stats:
  MkStats{lc, lf, +lw} = l
  MkStats{rc, rf, rw} = r
  merge_with(lc, lf, lw, rc, rf, rw, U32.is_ne(lw, 0))

def search(+depth: Nat, +index: U32) -> Stats:
  match depth:
    case 0n:
      leaf(index, U32.is_lt(index, 4913))
    case 1n+p:
      left right = search(p, (index * 2 : U32)) search(p, (index * 2 + 1 : U32))
      merge(left, right)

def main() -> Stats:
  search(13n, 0)
