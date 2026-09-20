# Complete conditional upper fold in the tree multiplier

This is a correctness argument and CPU research record, not GPU qualification
or a throughput measurement. Runtime digest:
`e5e60bb1b04795b5d125056826ee2d0646fc23baa63682cee694a08571060dea`.

Let `B = 2^256`, `K = 2^32 + 977`, and `p = B - K`. For arbitrary raw
256-bit operands, write the exact product as `L + B*H`, with `0 <= L,H < B`.
The first fold computes `L + K*H = R + B*c`. Its exact value is at most
`(K+1)*(B-1)`, so `0 <= c <= K` and `0 <= R < B`.

The next fold computes `R + K*c <= B-1 + K^2 < 2*B`. The existing low
96-bit add chain is unchanged. It captures its actual carry in a separate
register. If that carry is zero, every remaining upper-limb operation would
add zero with zero incoming carry; preserving the untouched upper limbs is
exact. If the carry is one, the implementation adds one to limb 3 and
propagates the carry through limbs 4, 5, 6 and 7. No upper carry is discarded.

The overflow after this second fold is at most one. When it is one, the
wrapped value is at most `K^2-1`; adding the final `K` gives at most
`K^2+K-1 < 2^65 < 2^96`. The final correction therefore needs only the same
three low limbs used by the parent. When overflow is zero, adding zero and
propagating zero would leave the result unchanged, so that correction is
skipped. The result is the same raw representative as the complete parent.
Existing caller normalization boundaries remain unchanged.

The production edit transfers the already implemented complete point-field
upper-fold schedule into `qsb_field_mul` only. The product terms, paired
column carry packing, first fold, low-96-bit second fold, output assignment
and host orchestration are unchanged. Each inline assembly block scopes its
own predicates, registers and branch label.

The shipped CPU PTX audit executes the actual extracted PTX against the
parent and an independent modular integer oracle on 5,207 operand pairs.
Every raw output is bit-identical. The three branch paths receive
5,081 / 103 / 23 cases, including directed upper-carry inputs. The prepared
native suite must still test the exact source, tree aliases, point and recovery
fixtures, and the partial/transition pipeline. Performance and a new full
1,200-second all-hit run are also required before this source can be submitted.

