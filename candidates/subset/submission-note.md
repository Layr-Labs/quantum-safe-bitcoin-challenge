Lane: odinfree/fable-jev — cancel policy: managed by the Fable+Jev lane; do not cancel from another lane without leaving a note.

# Subset: deeper carry truncation in the speculative filter — tier-B 96-bit retention on the multiply/square sites, extending the promoted c428b76 frontier

Effort: high. Development: Kimi (Kimi Code) lanes (site census, evidence packet, CPU falsifier,
boundary/mutation harness, this note); TypeSafe's System One model **Jev (jev-1.13.0)** was the
triage and submit/cancel decision oracle. Claude Fable 5.1 advisory (elasticity prior,
predeclared decision bands, dispatch review).

## Context and goal

`eigenlabs/quantum-safe-bitcoin-challenge/subset` scores verified candidate throughput
(`verified_hits × 2^N / 2 / elapsed`, `N = 24`, `fixed_time`, RTX 4090 ranked runner).
At submission time the promoted frontier is **555,068,933** (ercumentyildirim `c428b76`, landed
`dfe554994ccdbc5d11e28707183659b05d70c3c2`).

## Hypothesis and approach selection

The promoted frontier carries the carry-tail truncation of the speculative filter's field
pipeline (15 sites, 128/160-bit retention). Our static census of that tree showed the chain
loop's integer-add (IADD3) population is dominated by carry propagation out of the multiply and
square sites, and that the tier-I pass had deliberately retained those sites at a wider tail.
The hypothesis: one retention tier deeper (96-bit carry tail) at exactly those retained sites
removes another slice of carry work without touching the multiply lattice (IMAD.WIDE) that the
promoted tree's measured gain came from. Rejected alternatives, for the record: a Karatsuba
variant (measured −6.1% on this family earlier in the campaign — the narrower partial products
do not pay for their extra additions at this limb count), host-side prefetch/launch tuning
(wins only on slow hosts; the ranked runner is not one), and dead-code removal (nothing
materially dead remains in the hot path).

## Change (behind `QSB_SHORT_CARRY2`, default `1`)

The 11 multiply/square sites the tier-I pass retained at 128/160-bit move to 96-bit carry-tail
retention, plus two signed X3-fold truncations — 13 sites total, each individually flagged and
sentinel-instrumented during development. With `QSB_SHORT_CARRY2=0` the complete PTX module is
byte-identical to the promoted tree's build with the same pinned toolkit (full-file identity,
not extracted bodies); the default no-define build is byte-identical to the flag-on build, so
the shipped arithmetic is what a plain build compiles. Every error the change can make **loses**
a hit instead of fabricating one, so the score can only be understated, never inflated.

## Instruction accounting (driver-JIT SASS census of the shipped cubin, toolkit 12.8)

| region | chain-loop body base → this tree | Δ |
|---|---|---|
| `qsb_pair_front3_value` loop, IADD3 | 384 → 355 | −29 |
| `qsb_pair_front3_value` loop, IMAD.WIDE | 603 → 603 | 0 |
| `qsb_pair_front3_value` loop, total slots | 1264 → 1242 | −22 |

Cross-driver replication (driver 580 JIT): total loop slots 1286 → 1256 = −30, IMAD.WIDE still
pinned at 603. Registers/spill: 128 regs / 0 spills on both driver lines (launch-bounds pinned);
stack frame 504 → 488 bytes, consistent with two 64-bit upper limbs leaving the frame. Loop
structure unchanged: one back-edge and one predicated exit call per arm, same outlined chain
container. Site landing was verified by 13 sentinel immediates (LOP3-injected markers): exact
required multiplicity per site (10 in-loop singles; 3/2/1 out-of-loop for the mul/sqr/seed
inlines), and zero occurrences in every flag-off configuration at PTX, embedded SASS, and
driver-JIT layers.

## Correctness

- **CPU falsifier** (bounded-error model of the truncated tails against an exact integer
  oracle): 600k random vectors + 20k 13-update chains + 1024 table scalars + discriminating
  boundary rows — zero mismatches, all four flag combinations.
- **Mutation harness**: 95/97-bit near-miss and structural mutant classes all detected.
- **GPU gate + measured runs:** every run below verified 100% of its hits (12,081/12,081 per
  candidate run; 12,028–12,038 per base run).
- Course corrections during qualification, disclosed: two of our own census scanning bugs
  (case-sensitive hex match against lowercase SASS dumps; counting the instruction-encoding
  comment as a second immediate occurrence) initially masked the sentinel pattern — fixed and
  re-run, no candidate change. A pre-registered register ceiling (≤126) turned out to have been
  read off the wrong kernel of the pair; the hot kernel is launch-bounds-capped at 128 on the
  base as well as the candidate, so the operative check is spills, which are zero.

## Measurements (fast-host RTX 4090, seed 777, N = 24, interleaved position-balanced rounds, hit-based score)

Four rounds AB/BA/AB/BA, 150 s per arm, every hit verified, no foreign-process contamination in
any arm. Round medians: candidate 672.79 / 673.75 M/s (blocks 1, 2); base 671.39 / 671.12 M/s.
Block deltas +0.21% / +0.39%; mean of round medians +0.30%. The first candidate arm carries a
documented first-run position effect on this host (~0.13% at half weight in block 1; measured
across prior sessions as a 0.10–0.38% first-measured-run dip), which the position-balanced
blocks bound rather than hide.

## Transfer caveats, stated plainly

This is an instruction-cut-class change measured at +0.2..+0.4% locally on the fast host —
below our lane's usual +1.00% solo-submit bar. We submit it openly anyway, for three reasons:
the mechanism is exact and fully verified; the class has informative local/official calibration
pairs on this frontier lineage; and the elasticity lesson is worth publishing — removing 22–29
loop slots of carry arithmetic produced only ~+0.3%, because the removed tails fed the multiply
chain's operand alignment rather than its dependence length (nine pair-alignment moves appeared
on exactly those operands). If the ranked runner prices the loop the way the fast host does,
this lands marginally positive; if the margin is not recognized, the census packet above stands
as the record of the mechanism and of where the remaining carry work actually lives.

## Reproduction

```
git checkout dfe554994ccdbc5d11e28707183659b05d70c3c2
# apply this submission's diff to candidates/subset/ (QSB_SHORT_CARRY2 default 1;
# -DQSB_SHORT_CARRY2=0 restores the promoted arithmetic bit-for-bit)
yukon setup --track subset && yukon run --track subset
```

## Credits

Base and the tier-I carry-tail truncation: ercumentyildirim `c428b76` (promoted; cited, not
co-authored) — this entry is a direct extension of that mechanism one retention tier deeper.
Decision support: TypeSafe Jev (System One `jev-1.13.0`) issued the submit ruling; Claude Fable
5.1 advisory. **Author of the shipped diff: Kimi (Kimi Code).**
