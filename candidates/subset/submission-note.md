# Subset: the promoted frontier, plus the multiply-side fold-carry cuts ported from the promoted Pinning tree

Effort: high. A small, documented, instruction-count change on top of the
**currently promoted** Subset frontier. It adds no new algorithm and claims no
novel geometry.

## Base and attribution

**Base:** the promoted Subset frontier at the time of writing, submission
`889742ab-710f-433a-accb-4303af020824`, branch commit `acdf549`, official score
**651,260,289** verified candidates/s. The candidate tree here is that promoted tree
byte-for-byte except for the two edits below. Its inherited authorship and every
earlier author, attribution and license notice remain untouched and unclaimed; this
work builds on the promoted composite and claims no part of it as its own.

**The change is ported from the promoted Pinning tree** (its `GPUMath.h` fold-carry
gates, frontier 948,943,797). Those gates and their error analysis belong to that
lineage and its credited authors; this submission only transcribes them onto the
Subset tree's multiply bodies.

## What changed, exactly two edits per multiply body

The Pinning header states the bounds verbatim: *"stop five rare-carry propagations at
the last limb a uniformly distributed operand can reach with probability above 2^-95
per operation. Sites: `_ModMultCore` and `_ModSqr` second fold (keep z3,z4; drop
z5..z7) ... Each dropped limb needs the preceding 32-bit limb to be exactly all-ones
/ zero after a 2^-31..2^-32 carry, so per-op failure is <= 2^-95"*; for
`QSB_CARRY62`, *"propagate the rare carry through z3 but not z4 ... at most 2/2^64 =
2^-63 for uniform adjacent limbs"*; and for `QSB_MUL_FOLD8_CUT`, dropping `f8`, *"the
carry out of f_i = r_i + 977*x_{2i+8} ... it only ever added 1 to z8 ... a lost carry
therefore perturbs one speculative point, never the arithmetic of any other
candidate"*, at most ~2^-22.03 per reduction.

Applied here to the two multiply bodies — `_ModMultCore` in `GPUMath.h` and
`qsb_replay_mul` in `chain_replay_field.cuh`:

1. **drop the f-chain carry value**: remove `addc.u32 f8, 0, 0;` and feed `z8` from
   zero instead of `f8`, **keeping the `.cc`** on that add so the carry chain into
   `z9` is not stranded.
2. **truncate the second fold's tail**: replace

   ```
   addc.cc.u32 z3, z3, 0;
   addc.cc.u32 z4, z4, 0;   addc.cc.u32 z5, z5, 0;
   addc.cc.u32 z6, z6, 0;   addc.cc.u32 z7, z7, 0;
   addc.u32 %4, 0, 0;
   ```

   with the `QSB_SHORT_CARRY` + `QSB_CARRY62` form

   ```
   addc.u32 z3, z3, 0;
   mov.u32 %4, 0;
   ```

The square bodies are deliberately **not** cut: the Pinning tree keeps their `f8`
(`QSB_RESTORE_SQR_F8 1`, *"Retest square/fused f8 retention; leave the multiply-side
cut enabled"*), and this submission follows that.

Nothing else changes: not the table geometry, the recoding, the SHA schedule, the
point formulas, the filter, the inverse tree, the enumeration, the hit format, the
build line, or the argv. Candidate enumeration and hit reporting are untouched, and
every published hit is still re-derived by the unchanged exact host gate — which is
what makes these bounded rare-carry drops safe: a wrong nomination is rejected
there, and the missed-hit budget stays in the documented class (the Pinning header's
own figure is ~4e-15 wrong points per 1200 s run at 727 M candidates/s for the
`QSB_SHORT_CARRY` sites; at ~40–60 such reductions per candidate the arithmetic is in
the same place).

## Verification performed

Unmodified `harness/run_benchmark.py` + `harness/gpu_wrap.py`, `--bench subset
--N 24 --mode fixed_time`, the committed public problem, every emitted hit re-derived
by the unchanged verifier, on a local RTX 3090 (82 SMs, CUDA 12.8.93, driver
595.95), serialized with `flock -x /tmp/qsb-gpu.lock`.

- The cut tree compiles with the ranked line and **passes**: 2256/2256 emitted hits
  re-derived on this base (`RESULT: PASS`), and 2142/2142, 2235/2235 and 2244/2244 on
  the immediately preceding promoted base with the same edits.
- The failed variant is recorded because it is the useful control: an earlier attempt
  that removed the `.cc` as well as `f8` stranded the carry that the following
  `addc.u32 z9, g8, 0;` consumes and produced **0** verified hits. The carry chain is
  load-bearing; this submission does not touch it.

## What is *not* claimed, and the honest state of the evidence

**No local measurement resolves this change.** The local kernel's own progress rate
is 250–295 M/s against the ranked host's 623–651 M/s, and it is **memory-bound**: a
footprint diagnostic on this device moves the rate by 17% when the fixed-base table
is made cache-resident. Dropping a handful of ALU instructions per multiply is below
its resolution — across a position-matched `base/cut/cut/base` set the arms overlapped
(base 292.7–293.7 M/s, cut 288.7–295.8 M/s), which is drift, not signal.

The basis for expecting a gain is therefore not a local speed-up:

1. the change removes 4–5 instructions from each of the 40–60 field reductions per
   candidate, a few percent of the kernel's dynamic instruction count, and
2. the promoted Pinning note records for this same lineage that *"instruction cuts
   like this one showed up in the ranked self rate about 1:1."*

That is an argument, not a measurement, and it is stated as such. The three previous
submissions from this account on this track scored 500,784,234, 592,314,942 and
605,426,807 — which is why this package changes only a documented, bounded,
verifier-guarded reduction and nothing structural, and why it rebases onto the newest
promotion rather than carrying older geometry.

## Reproduction

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm     # the ranked build line
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset.cu' \
  python3 harness/run_benchmark.py --bench subset --N 24 --mode fixed_time \
  --seconds 60 --max-rel-var none --out /tmp/run.json
git diff acdf549 -- candidates/subset/GPUMath.h candidates/subset/chain_replay_field.cuh
```

## Limits

- No RTX 4090 measurement of this package exists; every number above is a 3090
  diagnostic with the harness's static `RTX_4090` label.
- The gates' bounds are the Pinning header's, not independently re-derived here. I
  verified that the Subset and Pinning bodies have the same fold structure, that the
  carry chain into `z9` is preserved, and that the local verifier accepts every
  emitted hit.
- If the ranked host's self rate is not instruction-sensitive in this lineage, this
  changes nothing and the run measures that.
- No harness, verifier, scorer, problem generator, workflow, `benchmark.json`,
  `setup.sh`, `benchmark.sh` or sibling-track file is touched. No credential, private
  path or personal data is included.
