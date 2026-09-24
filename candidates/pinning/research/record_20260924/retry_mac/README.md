# Exact split-word MAC seed experiment

The narrow instruction form is selected for one official GPU experiment.
No local speedup is claimed. Its exact arithmetic proof and smaller register
footprint support testing it, while the larger native instruction count makes
actual GPU timing decisive. The neutral cold-seed ordering is a separate
mechanism and is disabled by the submission owner.

The pinned baseline negative-Y helper computes `a*b+c` before the inherited
pseudo-Mersenne reduction. Its first product row adds four full 64-bit
words of c into the even chain, propagating their carry through that row
and into its next top word. This experiment distributes c's eight 32-bit
words between the even and odd product chains instead.

For B=2^32, each seeded first-row term becomes

```
a_0*b_j+c_j, j=0..7
```

with even j in the even chain and odd j in the odd chain. Each result fits
64 bits because `(B-1)^2+(B-1)=B^2-B`. Thus eight `mad.wide.u32` operations
can form the seeds independently. No carry can leave a seed, so the
separate `bias_carry` contribution to e4 is zero. The weighted constant
contribution is still exactly `sum(c_j*B^j)=c`. All subsequent product
rows and the entire reduction tail retain their original instructions.

This changes neither coordinates nor the inherited approximate-reduction
contract. In particular, it is not an additional discarded carry justified
by the host publication gate: the changed carry is provably absent.

`prepare.py --narrow --out /tmp/qsb-retry-mac-narrow` recreates the selected
narrow variant in a temporary source closure. Omitting `--narrow` recreates
the rejected wide instruction form. The generator always reads its original
header from Git commit `7e95c40c99e57bded233ce57c7f453fbde9fd21c` and verifies
its SHA-256 (`d8893d08e76acd5c9c00531e89783c309e1dab78c3cce6509e4c667e36ac0335`).
This prevents applying the patch a second time after production is selected.
Only `negative_y_mac.cuh` changes executable behavior. `QSB_MAC_HALF_SEED=0` selects the exact
original PTX strings; the default 1 selects the new seed. The selected production header has SHA-256
`702782b0d2cf51e11f02fd474880955451e599184f508defc5cc97fa18e1e51e`.
`QSB_MAC_HALF_SEED=0` remains available for ablation.

`python3 -B check.py --narrow --production` reads the actual selected production
header, requires that it equals the generated narrow candidate, preprocesses
both branches, and extracts their PTX. Without `--production`, the checker
recreates the chosen historical experiment from the pinned baseline. A source-derived CPU translation preserves register widths and
carry flags and runs with UndefinedBehaviorSanitizer. Both branches match
independent Python arbitrary-precision `a*b+c` on 100,000 random triples
and 888 directed triples, including maximal values, sparse limbs, and
alternating half-word patterns. The existing Python PTX interpreter also
checks 512 directed triples per branch. Removing one odd-column seed is
caught by a mutation test. The OFF PTX and original PTX are identical;
the reduction tail is identical between ON and OFF.

The PTX instruction count increases from 185 to 188 because explicit
zero-extension and unpacking replace the old carry chain. Native SASS and
register evidence are therefore required before selecting this mechanism.
No GPU execution or throughput was measured by this audit. The permanent
artifacts contain only the generator, checker, proof, and compact results;
source closures and native build products stay in temporary storage.

## Native screening result

A second instruction selection (`prepare.py --narrow`) expresses each seed
as `mad.lo.cc.u32` followed by `madc.hi.u32`, avoiding the explicit 64-bit
zero-extended seed operand. It passes the same 100,888-triple audit, including
independent Python PTX checks. `check.py --narrow` reproduces its evidence.
The low MAD produces the carry into the high MAD; the combined 64-bit result
is still the same bounded seed proven above.

All compiler arms disable the statistically neutral cold-seed ordering.

| Native sm89 prepare | Instructions | Registers | Spill bytes |
| --- | ---: | ---: | ---: |
| Original | 6,688 | 122 | 0 |
| Wide half-seed | 6,720 | 122 | 0 |
| Narrow half-seed | 6,720 | 120 | 0 |

Finish remains 4,040 instructions and 64 registers. At the default sm52
compile target, narrow prepare decreases from 25,128 to 25,116 instructions,
with 97 registers and no spills. Native narrow's extra instructions are
mostly moves. The two-register reduction does not change the expected
four-by-128-thread occupancy limit by itself. Both forms remove a seed carry
dependency, but no GPU timing prices that change.

The wide form provides no favorable static evidence. The narrow form is
inconclusive: exact and slightly lower in registers, but larger in native
instructions. These findings do not establish an end-to-end improvement or
justify a record claim. The narrow form is therefore submitted as a measured hypothesis: the official
GPU experiment must distinguish an improvement from another neutral rewrite.
`production-oracle-results.json` records the fresh selected-source validation;
the wide and generated-narrow results remain separate for comparison.
