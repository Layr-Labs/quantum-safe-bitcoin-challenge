import Base
import ./LAWS.bend as Laws

law add_assoc:
  for a: Nat
  for b: Nat
  for c: Nat
  {Nat.add(Nat.add(a, b), c) == Nat.add(a, Nat.add(b, c)) : Nat}

def add_assoc(a, b, c):
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_assoc(p, b, c) : {1n+Nat.add(Nat.add(p, b), c) == 1n+_ : Nat}
      {==}

def split_merge_proof(xs: List<&2, Nat>, +ys: List<&2, Nat>) ->
  {Laws.term_sum(Laws.append_terms(xs, ys)) ==
   Nat.add(Laws.term_sum(xs), Laws.term_sum(ys)) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case Con{+h, +t}:
      ih = split_merge_proof(t, ys)
      %Equal.sym(Nat,
                 Laws.term_sum(Laws.append_terms(t, ys)),
                 Nat.add(Laws.term_sum(t), Laws.term_sum(ys)),
                 ih) :
        {Nat.add(h, _) ==
         Nat.add(Nat.add(h, Laws.term_sum(t)), Laws.term_sum(ys)) : Nat}
      Equal.sym(Nat,
                Nat.add(Nat.add(h, Laws.term_sum(t)), Laws.term_sum(ys)),
                Nat.add(h, Nat.add(Laws.term_sum(t), Laws.term_sum(ys))),
                add_assoc(h, Laws.term_sum(t), Laws.term_sum(ys)))

def Laws.split_merge_preserves_terms(xs, ys):
  split_merge_proof(xs, ys)

def Laws.frontier_15_split(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14):
  Laws.split_merge_preserves_terms(
    [p0,p1,p2,p3,p4,p5,p6,p7],
    [p8,p9,p10,p11,p12,p13,p14])
