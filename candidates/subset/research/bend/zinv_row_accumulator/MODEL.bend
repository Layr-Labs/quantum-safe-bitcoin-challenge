import Base

type Stats is Data:
  MkStats{split_failures: U32, mutation_mismatches: U32, witness: U32}

def full_shift(x: U32) -> U32:
  (x >> 4n : U32)

def split_shift(+x: U32) -> U32:
  lo = (x .&. 255 : U32)
  hi = (x >> 8n : U32)
  ((hi << 4n : U32) + (lo >> 4n : U32) : U32)

def truncated_shift(x: U32) -> U32:
  lo = (x .&. 255 : U32)
  (lo >> 4n : U32)

def bool_u32(x: Bool) -> U32:
  match x:
    case False{}:
      0
    case True{}:
      1

def make_leaf(i: U32, split_bad: U32, +mutation_bad: U32, nz: Bool) -> Stats:
  match nz:
    case False{}:
      MkStats{split_bad, mutation_bad, 0}
    case True{}:
      MkStats{split_bad, mutation_bad, (i + 1 : U32)}

def merge_with(+ls: U32, +lm: U32, lw: U32, +rs: U32, +rm: U32, rw: U32, left_has: Bool) -> Stats:
  match left_has:
    case False{}:
      MkStats{(ls + rs : U32), (lm + rm : U32), rw}
    case True{}:
      MkStats{(ls + rs : U32), (lm + rm : U32), lw}

def leaf(+i: U32) -> Stats:
  +a = ((i >> 12n : U32) .&. 15 : U32)
  +b = ((i >> 8n : U32) .&. 15 : U32)
  +x = ((i >> 4n : U32) .&. 15 : U32)
  +y = (i .&. 15 : U32)
  +acc = ((a * x : U32) + (b * y : U32) : U32)
  split_bad = bool_u32(U32.is_ne(full_shift(acc), split_shift(acc)))
  +mutation_bad = bool_u32(U32.is_ne(full_shift(acc), truncated_shift(acc)))
  make_leaf(i, split_bad, mutation_bad, U32.is_ne(mutation_bad, 0))

def merge(l: Stats, r: Stats) -> Stats:
  MkStats{ls, lm, +lw} = l
  MkStats{rs, rm, rw} = r
  merge_with(ls, lm, lw, rs, rm, rw, U32.is_ne(lw, 0))

def search(+depth: Nat, +index: U32) -> Stats:
  match depth:
    case 0n:
      leaf(index)
    case 1n+p:
      left right = search(p, (index * 2 : U32)) search(p, (index * 2 + 1 : U32))
      merge(left, right)

def main() -> Stats:
  search(16n, 0)
