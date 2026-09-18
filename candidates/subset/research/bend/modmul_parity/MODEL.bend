import Base

# Reduced-radix synthesis model for p = 2^8 - 5 = 251.
# Each mask selects which one-bit facts are XORed:
#   bit 0: low product parity
#   bit 1: high product parity
#   bit 2: second-fold quotient parity
#   bit 3: final canonical subtraction
# The secp256k1 hypothesis is the structurally identical mask 0b1111 for
# p = 2^256 - (2^32 + 977).

def bool_bit(x: Bool) -> U32:
  match x:
    case False{}:
      0
    case True{}:
      1

def selected(mask: U32, bit: U32, value: U32) -> U32:
  (bool_bit(U32.is_ne(((mask >> U32.to_nat(bit) : U32) .&. 1 : U32), 0)) * value : U32)

def candidate(+mask: U32, a: U32, b: U32) -> U32:
  +n = (a * b : U32)
  +lo = (n .&. 255 : U32)
  +hi = (n >> 8n : U32)
  +t = (lo + 5 * hi : U32)
  t0 = (t .&. 255 : U32)
  +t1 = (t >> 8n : U32)
  +u = (t0 + 5 * t1 : U32)
  ge = bool_bit(U32.is_ge(u, 251))
  (selected(mask, 0, (lo .&. 1 : U32)) .^.
  selected(mask, 1, (hi .&. 1 : U32)) .^.
  selected(mask, 2, (t1 .&. 1 : U32)) .^.
  selected(mask, 3, ge) : U32)

def oracle(a: U32, b: U32) -> U32:
  (((a * b : U32) % 251 : U32) .&. 1 : U32)

def mismatch(mask: U32, +a: U32, +b: U32) -> U32:
  bool_bit(U32.is_ne(candidate(mask, a, b), oracle(a, b)))

# Lexicographic exhaustive scan of every canonical pair in F_251.
def failures(+mask: U32, na: Nat, nb: Nat) -> U32:
  match na nb:
    case 0n _:
      0
    case 1n+pa 0n:
      failures(mask, pa, 251n)
    case 1n+ +pa 1n+ +pb:
      a = U32.from_nat(pa)
      b = U32.from_nat(pb)
      (mismatch(mask, a, b) + failures(mask, 1n+pa, pb) : U32)

def main() -> List<U32>:
  [
    failures(0, 251n, 251n),
    failures(1, 251n, 251n),
    failures(2, 251n, 251n),
    failures(3, 251n, 251n),
    failures(4, 251n, 251n),
    failures(5, 251n, 251n),
    failures(6, 251n, 251n),
    failures(7, 251n, 251n),
    failures(8, 251n, 251n),
    failures(9, 251n, 251n),
    failures(10, 251n, 251n),
    failures(11, 251n, 251n),
    failures(12, 251n, 251n),
    failures(13, 251n, 251n),
    failures(14, 251n, 251n),
    failures(15, 251n, 251n)
  ]
