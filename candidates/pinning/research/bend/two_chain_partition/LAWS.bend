import Base

# Abstract group terms are represented as naturals here. Each value already
# denotes the complete signed, shifted table term for one CUDA chunk. The law
# is about schedule preservation; elliptic-curve addition is handled separately.

def term_sum(+xs: List<&2, Nat>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case Con{+h, +t}:
      Nat.add(h, term_sum(t))

def append_terms(+xs: List<&2, Nat>, +ys: List<&2, Nat>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      ys
    case Con{+h, +t}:
      h <> append_terms(t, ys)

# Splitting a schedule into two independent chains and merging the two chain
# results preserves every term, provided the partition is an exact append.
law split_merge_preserves_terms:
  for xs: List<&2, Nat>
  for +ys: List<&2, Nat>
  {term_sum(append_terms(xs, ys)) == Nat.add(term_sum(xs), term_sum(ys)) : Nat}

# Exact frontier geometry: chunks 0..7 form one chain and chunks 8..14 the
# other. The values include sign and the shifts [0,18,35,...,239].
law frontier_15_split:
  for p0: Nat
  for p1: Nat
  for p2: Nat
  for p3: Nat
  for p4: Nat
  for p5: Nat
  for p6: Nat
  for p7: Nat
  for p8: Nat
  for p9: Nat
  for p10: Nat
  for p11: Nat
  for p12: Nat
  for p13: Nat
  for p14: Nat
  {term_sum([p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14]) ==
   Nat.add(term_sum([p0,p1,p2,p3,p4,p5,p6,p7]),
           term_sum([p8,p9,p10,p11,p12,p13,p14])) : Nat}
