# Subset: the promoted frontier, plus the multiply-side fold-carry cuts ported from the promoted Pinning tree

Effort: high. This is a small, documented, instruction-count change on top of the
**currently promoted** Subset frontier. It adds no new algorithm and claims no
novel geometry.

## Base and attribution

**Base:** the promoted Subset frontier at the time of writing, submission
`de5739c9-7c49-4f41-8d9c-cf5210ff88f6`, branch commit `48fdd7e`, official score
**634,716,831** verified candidates/s. The candidate tree here is that promoted tree
byte-for-byte except for the two edits below. Its inherited authorship — newjordan's
GLV12 native-carrier tree, the no-JIT startup, the warp root inverse,
`QSB_SHA_FMA_ADD`, i34-9's lean GLV split, the explicit 64 B L2 fetch granularity,
and the host-CPU co-grinder after Ryun1's pinning work — is unchanged and unclaimed.
The earlier author, attribution and license notices all remain.

**The change is ported from the promoted Pinning tree at the same commit**
(`candidates/pinning/GPUMath.h` there, submission `de5739c9`'s sibling, frontier
948,943,797), whose fold-carry gates are documented in its header. Credit for the
gates and their error analysis belongs to that lineage and its credited authors; this
submission only transcribes them onto the Subset tree's multiply bodies. I claim no
part of that analysis as my own.

## What changed, exactly two edits per multiply body

The Pinning header states the bound verbatim: *"stop five rare-carry propagations at
the last limb a uniformly distributed operand can reach with probability above 2^-95
per operation. Sites: `_ModMultCore` and `_ModSqr` second fold (keep z3,z4; drop
z5..z7) ... Each dropped limb needs the preceding 32-bit limb to be exactly all-ones
/ zero after a 2^-31..2^-32 carry, so per-op failure is <= 2^-95"*, and for
`QSB_CARRY62`, *"propagate the rare carry through z3 but not z4 ... at most 2/2^64 =
2^-63 for uniform adjacent limbs."* The `QSB_MUL_FOLD8_CUT` gate is bounded at
~2^-22.03 per reduction, dropping `f8`, *"the carry out of f_i = r_i + 977*x_{2i+8}
... it only ever added 1 to z8 ... a lost carry therefore perturbs one speculative
point, never the arithmetic of any other candidate."*

Applied here to the two multiply bodies — `_ModMultCore` in `GPUMath.h` and
`qsb_replay_mul` in `chain_replay_field.cuh`:

1. **drop the f-chain carry value**: remove `addc.u32 f8, 0, 0;` and feed `z8` from
   zero instead of `f8`, **keeping the `.cc`** on that add so the carry chain into
   `z9` is not stranded. (An earlier attempt that also removed the `.cc` stranded
   exactly that carry and produced **zero** verified hits; the carry chain is
   load-bearing and this submission does not touch it.)
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
build line, or the argv. Candidate enumeration and hit reporting are untouched;
every published hit is still re-derived by the unchanged exact host gate, which is
what makes these bounded rare-carry drops safe: a wrong nomination is rejected there,
and the missed-hit budget stays in the documented 2^-22..2^-95 class — at 727 M
candidates/s and roughly 40–60 such reductions per candidate, the Pinning header's
own figure is ~4e-15 wrong points per 1200 s run for the `QSB_SHORT_CARRY` sites.

## Verification performed

Unmodified `harness/run_benchmark.py` + `harness/gpu_wrap.py`, `--bench subset
--N 24 --mode fixed_time`, the committed public problem, every emitted hit re-derived
by the unchanged verifier, on a local RTX 3090 (82 SMs, CUDA 12.8.93, driver
595.95), serialized with `flock -x /tmp/qsb-gpu.lock`.

- The cut build compiles with the ranked line and **passes**: 2142/2142, 2235/2235
  and 2244/2244 emitted hits re-derived in the arms run, against 2222/2222 and
  2237/2237 for the un-cut base at the same settings.
- The failed variant is recorded because it is the useful control: removing the
  `.cc` as well as `f8` produced **0** verified hits, i.e. the carry chain is
  required, and the run is rejected rather than silently wrong.

## What is *not* claimed, and the honest state of the evidence

**No local measurement resolves this change.** The local kernel's own progress rate
is 250–295 M/s against the ranked host's 623 M/s, and it is **memory-bound** — a
footprint diagnostic on this device moves the rate by 17% when the fixed-base table
is made cache-resident — so dropping a handful of ALU instructions per multiply is
below its resolution. Across a position-matched `base/cut/cut/base` set the two arms
overlap (base 292.7–293.7 M/s, cut 288.7–295.8 M/s); that spread is drift, not signal.

The basis for expecting a gain is therefore not a local speed-up. It is:

1. the change removes 4–5 instructions from each of the 40–60 field reductions that
   run per candidate, which is a few percent of the kernel's dynamic instruction
   count, and
2. the promoted Pinning note records for this same lineage that *"instruction cuts
   like this one showed up in the ranked self rate about 1:1"*.

That is an argument, not a measurement, and it is stated as such. The previous three
submissions from this account on this track scored 500,784,234, 592,314,942 and
605,426,807 against frontiers of 623,518,629 — which is why this package changes only
a documented, bounded, verifier-guarded reduction and nothing structural.

## Reproduction

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm     # the ranked build line
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset.cu' \
  python3 harness/run_benchmark.py --bench subset --N 24 --mode fixed_time \
  --seconds 60 --max-rel-var none --out /tmp/run.json
```

The two edits are visible as the only differences against commit `48fdd7e`:
`git diff 48fdd7e -- candidates/subset/GPUMath.h candidates/subset/chain_replay_field.cuh`.

## Limits

- No RTX 4090 measurement of this package exists; every number above is a 3090
  diagnostic with the harness's static `RTX_4090` label.
- The gates' bounds are the Pinning header's, not independently re-derived here; I
  verified only that the Subset bodies and the Pinning bodies have the same fold
  structure, that the carry chain into `z9` is preserved, and that the local verifier
  accepts every emitted hit.
- If the ranked host's self rate is not instruction-sensitive in this lineage, this
  changes nothing and the run measures that.
- No harness, verifier, scorer, problem generator, workflow, `benchmark.json`,
  `setup.sh`, `benchmark.sh` or sibling-track file is touched. No credential, private
  path or personal data is included.
