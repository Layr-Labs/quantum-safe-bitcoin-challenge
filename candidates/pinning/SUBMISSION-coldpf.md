# SUBMISSION coldpf: L2 hint for the four DRAM-resident table records of every candidate

Effort: Claude Fable 5.1 at maximum reasoning effort, driven from a Claude Code session.

## Summary

The promoted pinning grinder reads twelve 64 B table records per candidate.
Eight of them sit in the 48 MiB persisting L2 window; four are uniformly random
reads over a 9.08 GiB region and each one is consumed by the very next mixed
addition. Nothing in the thread hides that DRAM latency, only the other warps
of the SM do. This submission adds one `prefetch.global.L2` hint per cold record,
issued two mixed additions before the record is loaded (`QSB_COLD_PF=1`,
`QSB_COLD_PF_LEAD=2`). The hint compiles to `CCTL.E.PF2`, costs no register and
no destination, and at most one hint per thread is ever outstanding. Everything
else in the hot loop, the table geometry, the recovery pipeline and the host
gate are the promoted bytes.

## Base

- Promoted tree `7e95c40` (the v27 package of this ledger, official
  904,971,814 verified candidates/s), `pinning.cu` LF sha256 `11bd015c5e4ae21a...`;
  this tree's `pinning.cu` is `9bdb729cf375c0bc...`.
- Only `candidates/pinning` changes: `pinning.cu` (this change plus small init
  diagnostics), a new `FastInit.cuh` that the ranked build does not compile
  (see "Also in the tree"), and this note. `SOURCE-MANIFEST.json` predates this
  change and was not regenerated.

## Why this and not another geometry

From `GLVScalar.cuh` (QSB_FOUR_HOT): terms 0 and 1 hold 262,144 records each
(16 MiB), terms 2 and 3 hold 131,072 each (8 MiB), term 4 holds 67,108,864
(4 GiB) and term 5 holds 85,279,885 (5.08 GiB). Both GLV components share the
six tables, so a candidate performs 8 reads inside the 48 MiB window and 4
random reads over 9.08 GiB.

Two facts from this ledger price the cold reads. GLV10 (ten random DRAM reads
per candidate over 8.9 GiB, two adds fewer) measured 400.0M/s self, which is
4.0 G random reads per second. The promoted tree runs about 905M x 4 = 3.6 G
random reads per second. One extra scattered read per iteration was priced at
-66.9% on a live 4090 by terrapinelf. So the random-read rate of the big region
is the wall and the promoted tree sits at roughly 90 percent of it, while the
compute side (11 mixed additions, 122 registers, 16 warps per SM) has a similar
margin. Fewer cold reads is not available with 64 B records: hot bits are
capped by the persisting window, cold bits by the 24 GB card, and the two-cold-term
split already spends both budgets.

What is left is the exposed part of the cold latency. With 16 warps per SM and
about 0.25 instructions per cycle per warp, one mixed addition is roughly 4,800
cycles, about 1.9 us at 2.5 GHz. A loaded random DRAM read is about 1.4 us by
Little's law at the observed read rate if roughly 8 percent of thread time is
spent waiting on it, which matches this ledger's own "about 92 percent hidden by
occupancy" estimate. Hinting the record two additions ahead (about 3.8 us of lead)
lets the load hit L2 instead of DRAM; the line only has to survive a few
microseconds in the 22 MB of L2 outside the persisting window, which turns over
in about 60 us at the promoted traffic.

This is not the retired `QSB_TBL_PREFETCH` (about ten hints per thread issued at
once, which flooded the LSU and MSHR queues, -37%) and not `QSB_EARLY_LOAD`
(prefetch into registers, which spilled). One outstanding hint per thread, no
registers.

## Implementation

Helper, placed before `_FixedBaseSignedXYZZScalar`:

```cpp
#ifndef QSB_COLD_PF
#define QSB_COLD_PF 1        /* 0: none; 1: hint each streaming record QSB_COLD_PF_LEAD adds ahead; 2: both per component up front */
#endif
#ifndef QSB_COLD_PF_LEAD
#define QSB_COLD_PF_LEAD 2   /* mixed additions between the L2 hint and the load */
#endif
#define QSB_COLD_FIRST (QSB_FOUR_HOT ? 4u : 3u)   /* first streaming term of a component */
__device__ __forceinline__ void qsb_cold_prefetch(const uint8_t *table,unsigned term) {
    volatile uint32_t *codes=(volatile uint32_t*)qsb_digit_arena();
    const uint32_t code=codes[(size_t)term*QSB_TREE_N+threadIdx.x];
    const uint8_t *rec=table+(size_t)(code&0x7fffffffu)*64;
    qsb_prefetch_l2(rec); qsb_prefetch_l2(rec+32);   /* both 32 B sectors */
}
```

In the chain loop, before `qsb_load_glv(table,term,x1,y1)`:

```cpp
if(term+QSB_COLD_PF_LEAD<last && (unsigned)((term+QSB_COLD_PF_LEAD)%GT_CHUNKS)>=QSB_COLD_FIRST)
    qsb_cold_prefetch(table,term+QSB_COLD_PF_LEAD);
```

The predicate is uniform (term is the loop counter), the codes are the same
digit-arena words the load reads later, and the record address is the same
`(code & 0x7fffffff) * 64` the load uses. With the first component at terms 0..5
and the second at 6..11 the hints fire at terms 2, 3, 8 and 9 for records 4, 5,
10 and 11. `QSB_COLD_PF=2` hints both cold records of a component when its hot
adds start (two hints in flight, two to five adds of lead); it is compiled out
here and left for a follow-up draw.

Also in the tree, all at promoted behaviour by default:

- `QSB_COLD_CS` (0): evict-first loads for the cold terms.
- `QSB_FAST_INIT` (0): a GPU ladder and batched-inversion table builder in
  `FastInit.cuh`, bit-exact against the promoted builder on the 22.9M-record
  geometry (same table digest), 3.04 s to 0.83 s locally. It is not compiled
  into this build (the include is guarded) because its full-geometry digest has
  not been checked on a 24 GB card yet, and it is worth about one second of the
  window.
- `[init]` phase marks on stderr, `QSB_STACK_LIMIT` (32768, promoted value) and
  `QSB_PROGRESS_EVERY` (10, promoted cadence) plus an `fflush(stdout)` after the
  progress line so timed runs keep their rate lines.

## Commands

Ranked build (unchanged organizer line, no `-arch`, driver JIT on the runner):

```bash
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm
```

Register envelope of the hot kernel on sm_89 (ptxas 12.6):

```bash
nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_89 -cubin -Xptxas -v -o /dev/null pinning.cu
cuobjdump -sass pinning.cubin | grep -c CCTL.E.PF2
```

Local functional run on a 4 GB card (small geometry, not the ranked table):

```bash
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_FOUR_HOT=0 -DQSB_BATCH=2097152 -DQSB_STACK_LIMIT=4096 \
     -DQSB_PROGRESS_EVERY=1 -o bin_local pinning.cu -lcrypto -lm
timeout 120 stdbuf -oL ./bin_local problems/pinning.bin 0 1 0 single_hash
python3 harness/verify.py --artifact <hits as artifact json> --max-rel-var none
```

## Results

| measurement | promoted | this tree |
| --- | --- | --- |
| hot kernel registers, sm_89, ptxas 12.6 | 122 | 116 |
| spill stores and loads | 0 / 0 | 0 / 0 |
| `CCTL.E.PF2` sites in the hot kernel | 0 | 1 (two sectors) |
| harness setup and verifier smoke | pass | pass |

Functional check of this exact source on the 4 GB card (small geometry): a 120 s run
produced 678 hits and the organizer's `harness/verify.py` verified all 678 with 0
failures (PASS) against the seed-0 public problem. The hint cannot change any
value the kernel computes; this shows the tree builds and runs end to end.

Local throughput on the 4 GB laptop card is not evidence either way and is
reported for honesty. Once the laptop was otherwise idle, four alternating 90 s
arms (cumulative device-counted rate at the last completed sequence) gave
promoted 49.1 and 35.6 M/s against this tree 50.4 and 52.6 M/s; the last promoted
arm coincided with the machine getting busy again, so the pair is 49.1 versus
50.4 and 52.6. Earlier, under CPU contention that also cuts the shared power
budget, with the whole L2 at 1 MiB the hinted line is evicted
before the load, so 150 s arms of hint variants produced 7 to 10 percent fewer
hits than the promoted bytes (about two sigma of Poisson noise). That card has
no persisting window at all and cannot hold the ranked table, so the ranked
run is the first real measurement of the effect.

## Caveats

- Untested on an RTX 4090 before submission; no rental budget was available.
- The upside is bounded by the exposed cold latency, this ledger's own estimate
  is up to about 8 percent of the chain; the realistic expectation is a few
  percent. The downside if the hinted line does not survive until the load is
  neutral: the load merges into the in-flight fill, and the extra cost is four
  hint instructions and four digit-arena reads per candidate.
- If this draw lands below the promoted score, the next draws of this line are
  `QSB_COLD_PF_LEAD=1` and `QSB_COLD_PF=2`, then the same hint applied to the
  first cold term from the decode step.
- Two local runs (out of about fifteen) ended with "unspecified launch failure"
  on the WSL2 laptop, once with the hint and once with the promoted bytes, so
  that is the box and not this change.

## Reproduction and provenance

Apply the two code blocks above to the promoted `pinning.cu`, or use this tree
as is. No new third-party code; the GPL-3.0-only notices of the inherited
sources and the MIT secp256k1 field core are retained unchanged. Builds on the
promoted v27 package (fkiene) and its lineage; the geometry, host gate, priority
pipeline and recovery algebra are theirs.
