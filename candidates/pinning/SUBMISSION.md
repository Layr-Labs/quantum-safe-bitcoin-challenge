# Pinning: double the square cross-term with an 8-step add instead of a 15-step shift

Effort: Grok 4.7, written in Cursor. One mechanism on the promoted pinning
tree. No local GPU, so this note does not claim a throughput number.

## What the previous candidate did

Submission `b1d15d61-7bf7-41fa-98a5-29517e3f183a` scored **742,767,647**,
rejected against the crown at **805,428,058**. It raised stage 0 from four
resident blocks to five and parked `ZZZ` in shared across the fused square,
so the sm_89 JIT would have to schedule in 96 registers instead of 128.

That was the wrong side of the register file. The promoted tree's own static
note already said native sm_89 uses 128 registers with zero spill when the
launch bound allows it, and sm_52 fits in 101. Forcing the tighter cap, and
adding a shared round trip on every mixed addition, cost about 7.8%. Five
resident blocks and the `ZZZ` park should not be retried. The JIT is using
those 128 registers on purpose. This candidate does not change
`QSB_S0_BLOCKS`, shared memory, or the live set.

## Starting point

Promoted commit `7c3609b`, validation of
`22944657-779f-4b1c-b22e-5b89c8d429c9`. The inner addition is the deferred-Y
XYZZ mixed add, two squares per iteration (`PP = P^2` and the fused
`R^2 + PPP - 2V`). Both squares build the off-diagonal cross terms, then
double them, then add the eight diagonal squares.

The double is a left shift by one of the 16 little-endian 32-bit limbs. The
low limb `x0` is zero (the cross terms start at bit 32). The shift is fifteen
dependent `shf.l.wrap.b32` operations, high limb down to limb 1, each taking
the top bit of the limb below. That chain is on the square's critical path:
the diagonal squares cannot be added until the doubled cross terms exist.
It runs twice per mixed addition, and the chain does thirteen mixed additions.

## Change

After the even/odd columns are merged into `x0..x15`, pack them into the
eight `u64` registers the diagonal add already uses and double with

```
add.cc.u64 d0, d0, d0;
addc.cc.u64 d1, d1, d1;
...
addc.u64   d7, d7, d7;
```

The pack `mov.b64 d*, {x2i, x2i+1}` was already the next instruction after
the shift, so it is not new work. The fifteen serial funnel-shifts are gone.
The carry out of `d7` is dropped, which is the same bit the old shift shifted
off the top of `x15`. A fresh `add.cc` starts at `d0` for the diagonal
squares, so that dropped carry does not enter the next chain.

The same replacement is in both `_ModSqr` device bodies and in
`_ModSqrAddSub2`. The host `__uint128_t` transcription is unchanged; it
already computes the same doubled cross term. Ten thousand random 16-limb
inputs with `x0 = 0` matched the old shift bit for bit, including the
discarded top bit.

Register pressure does not go up. `d0..d7` are the registers the diagonal
add consumed on the next line. Nothing else in the addition, the table load,
the cofactor tree, or the launch bounds changed.

## Why this can matter

The chain loop was last measured around a thousand SASS instructions per
round. Seven fewer instructions on the square's dependency chain, twice per
round, is on the order of one percent of that loop if the shift was actually
serializing the square. If `ptxas` was already hiding the funnel-shifts
beside the diagonal multiplies, the ranked score will be flat and this
schedule should be left alone. It is exact either way: a wrong point would
be a bug in the double, not a bounded carry drop.

## Closed by the 742M result

Do not lower the stage-0 register cap to chase a fifth block. Do not park an
accumulator in shared across `_ModSqrAddSub2`. The earlier losses still
stand: `ld.global.cg` in place of `__ldg`, streaming stores on the roots,
three slots, chain unroll, GLV, Karatsuba, the 14-window table, the pinning
SHA shared-table flags, and the merged top-16 traversal.

## Equivalence

`shf.l.wrap.b32 d, a, b, 1` is `(b << 1) | (a >> 31)`. Applied from limb 15
down to limb 1, that is a left shift of the 512-bit cross term, leaving limb
0 untouched. Limb 0 is the constant zero written just before the column
merge, and the merge adds into limbs 2 and up, so limb 0 is still zero at
the shift. Doubling the eight little-endian `u64` pairs with `add.cc` /
`addc` produces the same 512 bits: the low half of pair 0 is `0+0`, and each
carry into the next pair is the bit the funnel-shift used to take from the
limb below. The carry out of pair 7 is discarded on purpose. It is the bit
that used to fall off limb 15. The following diagonal chain opens with
`add.cc.u64 d0, d0, t`, which starts a new carry, so the discarded bit is not
an input to it.

Per mixed addition the shortened chain runs once in `_ModSqr` (`PP = P^2`)
and once in `_ModSqrAddSub2`. The seed `_PointAddXYZZ_mm` uses `_ModSqr` too,
so it picks up the same schedule for free. No point coordinate, digit, table
record, or recovery product changes representation.

## Check without a GPU

`candidates/pinning/test_host_gate.py` is the publication-gate audit and does
not execute the square. The double itself was checked against the old shift
on 10,000 random limb vectors. No claimed score is attached.
