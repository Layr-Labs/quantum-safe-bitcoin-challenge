# Exact split-word MAC seed experiment

The narrow variant was rejected by the official GPU benchmark: 859,945,955
verified candidates/s versus the 904,971,814 record, a 4.975% decrease.
All 123,117 returned hits verified; elapsed time was 1,200.9827 seconds,
with seed 336189582. Production now defaults to `QSB_MAC_HALF_SEED=0`.
The experimental branch remains available for reproducible ablation.

The pinned negative-Y helper computes `a*b+c` before its inherited
pseudo-Mersenne reduction. Originally, the first product row adds four
64-bit words of c into the even chain, propagating their carry through
that row and into its next top word. This experiment distributes c's eight
32-bit words between the even and odd product chains instead.

For B=2^32, each seed is `a_0*b_j+c_j`, with even j in the even chain and
odd j in the odd chain. Each fits 64 bits because
`(B-1)^2+(B-1)=B^2-B`. No carry leaves a seed, so the separate `bias_carry`
contribution to e4 is zero. The weighted constant contribution remains
`sum(c_j*B^j)=c`. Later product rows and the reduction tail are unchanged.
This is exact product equivalence, not a change to the inherited reduction
contract or an extra discarded carry justified by the host gate.

The wide form uses zero extension and `mad.wide.u32`. The narrow form uses
`mad.lo.cc.u32`, `madc.hi.u32`, and `mov.b64` for each seed. The low MAD
supplies the carry to the high MAD. Later arithmetic resets CC before
consuming carries, including the first reduction operation `add.cc.u64 f0`.

## Reproduction and correctness evidence

Run from this directory:

```sh
python3 -B prepare.py --narrow --out /tmp/qsb-retry-mac-narrow
python3 -B check.py --narrow --production
python3 -B check_full.py
```

The generator always reads the original header from Git commit
`7e95c40c99e57bded233ce57c7f453fbde9fd21c` and verifies SHA-256
`d8893d08e76acd5c9c00531e89783c309e1dab78c3cce6509e4c667e36ac0335`.
It never patches the current production header a second time. Omitting
`--narrow` recreates the wide form, which was screened locally only.

`check.py --narrow --production` preprocesses the actual production header
under explicit flag 0 and flag 1, regardless of its current default. It
compares each branch with the pinned original/generated candidate, then
extracts the PTX. A source-derived CPU translation preserves widths and
carry flags under UndefinedBehaviorSanitizer. Both branches match Python
arbitrary-precision `a*b+c` on 100,000 random and 888 directed triples,
including maximal values, sparse limbs, and alternating half-word patterns.
The Python PTX interpreter independently checks 512 triples per branch.
Omitting an odd-column seed is detected by a mutation test. OFF matches the
original PTX exactly; the reduction tail is identical between ON and OFF.

The additional post-result `check_full.py` expands the production macros
under the actual C31/host-gate selection, then interprets the complete MAC,
including reduction. Explicit flag 0 and flag 1 produce identical outputs
on 216 extreme-value triples and 1,000 random triples. The audit checks the
identical reduction tail and its initial carry reset. No source-level
product or reduction discrepancy was found. These CPU checks do not test
GPU/JIT execution or establish throughput.

The submitted default-1 header SHA-256 was
`702782b0d2cf51e11f02fd474880955451e599184f508defc5cc97fa18e1e51e`.
The audited default-0 production header SHA-256 is
`1342368c46fa2d8523f9b03c8ce9ae83c78438072866ca76d00696c6742175d3`.
`production-oracle-results.json` and `full-mac-audit.json` record the fresh
production audit; generated-wide/narrow results remain separate.

## Static costs and interpretation

All compiler arms disabled the neutral cold-seed ordering.

| Native sm89 prepare | Instructions | Registers | Spill bytes |
| --- | ---: | ---: | ---: |
| Original | 6,688 | 122 | 0 |
| Wide half-seed | 6,720 | 122 | 0 |
| Narrow half-seed | 6,720 | 120 | 0 |

Finish remained 4,040 instructions and 64 registers. At default sm52,
narrow prepare decreased from 25,128 to 25,116 instructions with 97 registers
and no spills. Native extra instructions were mostly moves; the reduction
from 122 to 120 registers did not increase the expected four-by-128-thread
occupancy limit. Product PTX operations were 185 original, 188 wide, and
196 narrow. Static evidence was inconclusive; the official result now
rejects the narrow variant as a performance improvement.

The harness timeout counter can extrapolate a post-initialization rate
over the entire wall interval (`harness/gpu_wrap.py`). Consequently, the
hit-implied rate divided by that counter's rate is not an arithmetic-recall
measurement. Distinguishing startup cost from missing searched candidates
requires the raw progress/coverage evidence; these arithmetic audits alone
cannot explain the official score decrease.
