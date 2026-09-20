# Paired even/odd multiplication carries

The multiplication core accumulates 32-bit partial products in 64-bit even
columns `e[j]` and odd columns `o[j]`. Their weights in the unreduced integer
are respectively `2^(64*j)` and `2^(64*j+32)`.

A saved carry `cy` from even column `j` ordinarily enters `e[j+1]`. The same
integer contribution can enter the upper 32 bits of `o[j]` instead:

```
cy * 2^(64*(j+1)) == (cy << 32) * 2^(64*j+32)
```

At each of the three changed boundaries, both saved carries are bits. The
replacement packs the old odd carry into the low 32 bits and the even carry
into the high 32 bits, then omits the latter from the next even initializer.
Packing and widening instructions do not change PTX carry state. The later
initializer therefore observes the same incoming carry flag.

The odd initializer cannot lose a 65th bit: for any 32-bit multiplicands, its
product plus both saved carries and the incoming carry is at most

```
(2^32-1)^2 + (2^32+1) + 1 = 2^64 - 2^32 + 3 < 2^64.
```

All later product accumulation, folding, final carry correction and point
contracts remain unchanged. The separate truncation and square changes in
public submission 208bbcb6 were not imported. The original public commit is
`a668c4e5fd80db398c13222453f9c9a645612649`; licensing is preserved.

Actual extracted PTX for both `_ModMultCore` and `qsb_field_mul` passed 5,082
CPU input pairs against the unchanged parent and a modular integer oracle.
The combined upper-fold/recovery variant passed 5,207 pairs per function,
covering all three conditional reduction paths. These CPU results do not
establish native GPU correctness or performance. Those require this source's
own native, matched comparison and full-duration evidence.
