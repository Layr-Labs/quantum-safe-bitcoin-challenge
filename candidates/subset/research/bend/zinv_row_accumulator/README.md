# Cooperative inverse row-accumulator check

This experiment independently checks the arithmetic shape behind the promoted
frontier's `zi_row_ip` top-limb update. The exact promoted base is commit
`1248235b7bab9471e6cc4c2f301180837e039fc8`; its
`candidates/subset/tests/gpu_epochs/zinv32.cuh` SHA-256 is recorded by
`RESULTS.md`.

The production routine accumulates a signed 64-bit row value, stores it into a
32-bit limb, then shifts that stored limb by 30. A pending public candidate
claims that this can discard the high accumulator and instead keeps `acc`
through the final shift. This model tests that algebra independently at radix
2^8 with a 4-bit delayed shift, where Bend's u24 arithmetic can exhaust all
65,536 coefficient/limb tuples without overflow.

`LAWS.bend` states mechanically checked boundary and mutation-sensitivity laws.
`PROOF.bend` proves those laws. `MODEL.bend` is the distinct finite, balanced
search; its `MkStats` result contains split failures, mutation mismatches and an
encoded first witness. The expected result has zero split failures, nonzero
mutation mismatches and a nonzero witness. The laws prove only their concrete
reduced-radix statements, while the model is a finite search—not a proof of the
32-bit CUDA routine, signed bounds, warp exchanges or GPU synchronization.
