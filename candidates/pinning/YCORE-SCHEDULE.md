# Deferred XYZZ coordinate products

This source changes only the order of two existing products in
`_PointAddXYZZYCore` relative to the merged low-fold/top-two-tree parent.
Every modular primitive, reduction carry, correction and point guard remains.
The implementation still computes both complete XYZZ coordinate products.

After `P = U2 - X1`, the shared prefix computes `PP = P^2`, `PPP = P*PP`
and `Q = U2*PP`. The original order updates `ZZ1 *= PP` before computing
`T = X3`, then updates `ZZZ1 *= PPP` before the final `Q` products. This
source first finishes `Q = R*(Q-T)`, then updates `ZZZ1` and `ZZ1`.

The intervening `T` work reads `R`, `PPP` and the old `Q`. Its lazy path uses
the complete fused square/add/sub helper; its other path uses the original
square and add/sub sequence. Neither path reads or modifies `ZZ1` or `ZZZ1`.
The subsequent two `Q` operations read only `Q`, `T` and `R`. Neither changes
`PP`, `PPP`, `ZZ1` or `ZZZ1`. Therefore the delayed coordinate products read
the same exact operand values and produce the same raw output words.

Both products complete before either branch of `DEFER_Y`. Immediate Y recovery
still reads the updated `ZZZ1` in `Y2*ZZZ1`. Deferred Y recovery stores the same
`Q` and retains the completed coordinate products for its later recovery step.
All input/output arrays in the production point frame are distinct. This
argument does not assert correctness for arbitrary unsupported aliasing among
those arrays. Supported primitive aliases remain covered by their own audits.

The source-extracted CPU audit evaluates the ordered expression graph and
operation multiset for both lazy settings, comparing every resulting frame
value with the parent. Moving either product before its dependencies is a
negative control and must change that graph. The exact source has separate
native point, field, exceptional final-window and pipeline qualification;
inherited mathematical documentation does not transfer the parent's results.

The retained comment on the moved `ZZ1` line describes its former location.
The executable order, dependency audit and this document describe the actual
schedule. Changing that source comment would change the exact source identity,
so the already measured source is preserved.

The scheduling idea was reviewed in public submission `26e1823e`, commit
`5e1d5bdfa85139a6618e8f79c0188fb187379df6`. Its unrelated arithmetic changes
were not imported. The organizer problem, verifier, hashing, reporting and
scoring contract are unchanged. Measured gains and qualification receipts are
recorded in `RESEARCH.md` and `SOURCE-MANIFEST.json` after actual validation.
