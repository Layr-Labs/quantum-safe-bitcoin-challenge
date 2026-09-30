# FINDING 6 — where the cost actually is, and why the last named lever is predicted to lose

This is the reusable technical content of the session. It closes the one
direction every note in this track keeps pointing at ("a rebalanced split, a
two-level table, or partial recomputation"), using the track's own measurements
rather than a new speculation.

## The geometry this tree actually runs

Read from `GLVScalar.cuh` with `QSB_BIGTBL=1`, `QSB_FOUR_HOT=1`, `QSB_GLV11=1`
(defaults: `GLVScalar.cuh:5,14,82`; the `1`s are set by `pinning.cu:161` and
`QSB_GLV_GLUE 15`, `QSB_DECODE_CUT 3`, `QSB_DIGIT_LEAN 1`).

The 128-bit signed GLV residual magnitude is tiled by windows whose widths are
fixed by `q9_bigtbl_code` (`GLVScalar.cuh:180`) and which must sum to exactly
128 for the digit biases to telescope to the same segment-0 bias `K`:

    chunk : shift  width  records      bytes
      0   :   0      18     262 144     16 MiB   <- hot prefix
      1   :  18      19     262 144     16 MiB   <- hot prefix
      2   :  37      18     131 072      8 MiB   <- hot prefix
      3   :  55      18     131 072      8 MiB   <- hot prefix   (prefix = 48 MiB)
      4   :  73      27  67 108 864      4 GiB
      5   : 100    top   85 279 885   5205 MiB   (TOP_CENTER 170 559 769, odd)
      6   :  18      27  67 108 864      4 GiB   <- P only
      7   :  45      28 134 217 728      8 GiB   <- P only
                      ---------------------------------
    GT_TOTAL_ENTRIES = 354 501 773  ->  22 688 113 472 B = 21.1 GiB

    Q path: chunks 0,1,2,3,4,5                 -> 6 gathers, 5 adds
    P path: terms 0..4 -> chunks 0,6,7,4,5     -> 5 gathers, 4 adds
    total : 11 gathers, 10 adds, 11 cold/hot split as below

Tiling verified: `0+18=18`, `18+19=37`, `37+18=55`, `55+18=73`, `73+27=100`,
`100+28=128`; P: `0+18=18`, `18+27=45`, `45+28=73`, `73+27=100`, `100+28=128`.
Offsets/entries/sums match the `static_assert`s at `pinning.cu:665-676` and
`GLVScalar.cuh:99-116`.

**Per-candidate memory classification:**

| | gathers | hot (48 MiB prefix) | cold (21.0 GiB) | cold bytes |
|---|---:|---:|---:|---:|
| GLV11 (this tree, `QSB_GLV11=1`) | 11 | 5 | **6** | 384 B |
| GLV12 (`QSB_GLV11=0`) | 12 | 8 | **4** | 256 B |

`QSB_PERSIST_WINDOW_CAP = 42 MiB` (`pinning.cu:6`) against a 48 MiB dense
prefix — the window is deliberately *smaller* than the prefix.

## The one trustworthy calibration of cold-read cost

Two official promotions of the same project, one variable changed — the best
evidence in the track for the marginal cost of a scattered 64 B DRAM read:

    GLV12 (12 gathers, 11 adds, 4 cold)  -> 960,830,125   [promoted]
    GLV11 (11 gathers, 10 adds, 6 cold)  -> 948,943,797   [promoted, earlier]

GLV11 removes **one add** and adds **two cold reads**, and is 1.24 % slower.
So: 2 cold reads > 1 add, and **one cold read ≈ 0.6 %+ of total runtime.**

This is much smaller than terrapinelf's quoted "-66.9 % for +1 scattered 64 B
read/iter" (`ITERATIONS.md:1701`, cited from note `fd6b0c8f`). Those two numbers
cannot both be right for the same program, and I am not going to pick the
flattering one. The in-band same-project pair is the one I would bet on; the
-66.9 % was priced by a marginal perturbation on some other geometry. **Any
argument in this track that leans on the -66.9 % figure should be treated as
unverified.**

## Why "L2-resident geometry" — the last named prize — is predicted to lose

The prize named in `SUBMISSION-v20.md:135` and `SUBMISSION-v27.md:113-119` is
"keeping the 11-add chain while holding hot records inside the 72 MB L2 — a
two-level table, a skewed-width digit recoding, or partial recomputation."

The obvious version is: shrink the table until every gather is an L2 hit.
The cost is arithmetically forced, and it is brutal. With `N` terms per
component the widths must sum to 128, and the record count is minimised at
equal widths:

    N=6 : 6 * 2^21.3  ~ 10.5 M records/component = 671 MiB  (this is GLV12)
    N=8 : 8 * 2^15    =    262 144 records/component =  16 MiB   <- would fit 42 MiB
    N=12: 12 * 2^9.7  ~ 12 000 records/component =  0.7 MiB

So reaching L2 residency needs **N=8 terms per component**: 16 gathers and 14
adds per candidate, versus GLV12's 12 gathers and 11 adds. That is **+3 adds and
+4 gathers to remove 4 cold reads** (worth ~2.4 % by the calibration above), in
exchange for three extra serial mixed additions (~7.5 field multiplies each,
~22 field multiplies, in a chain that is the program's critical path).

And the track has already measured the decisive fact: **the chain's table reads
are not exposed.** `QSB_TBL_PREFETCH` (v21, `ITERATIONS.md:1617-1625`) prefetched
the chain's ~10 table lines ahead of the serial multiply-adds and scored

    903.8 M/s -> 570.9 M/s      (-37 %)

with the post-mortem: *"the latency DRAM exposure was already ~92 % hidden by
occupancy (~19 warps/SM), the upside was small and the cost of queue-flood
dominated. The workload is COMPUTE-bound, not latency-bound."*

If DRAM latency is already ~92 % hidden, then **converting a DRAM read into an
L1 read buys close to nothing**, while the adds it pays for are on the critical
path and cost real time. The prize's entire mechanism argument — "cold reads are
expensive" — is contradicted by the field's own prefetch experiment on this very
kernel.

Corroborating arithmetic: at ~960 M candidates/s, 4 cold reads = 256 B/candidate
= **246 GB/s** of read traffic. The RTX 4090 sustains ~1 TB/s. The DRAM path is
at ~25 % of peak: it is neither saturated nor, per v21, exposed. It is not the
bottleneck the geometry prize assumes.

**Caveat, stated honestly:** this is an inference, not a measurement. A smaller
table does relieve TLB and DRAM-page pressure and reduces distinct-line L2
pressure in a way that prefetching does not. So the inference is not a proof.
What it does is remove the *existing* justification. To revive the idea someone
would need a new argument for why a smaller table helps when prefetching the
same reads hurt — not the "L2-resident hot set" story that `DEAD-ENDS.md` and
`SUBMISSION-v20.md` still advertise.

## The one genuinely open, non-noise lever, and why it is not shippable here

N=8 is the only design in this family with the right magnitude, and it is a
**whole-geometry replacement** — precisely the class that has moved this track
(GLV14 → GLV11 → GLV12: −40 M, +48 M, +12 M). It also removes the 21.1 GiB
GPU table build from the scored window (`pinning.cu:5639` prints
`GTable built on GPU in %.2fs`; ~22 s at the 9.1 s/GiB rate the v23 note
measured, i.e. ~1.8 % of a 1200 s window) and removes the VRAM pressure that
forced the adaptive-batch fallback in the first place.

That is a coherent >5 % story. It is also, unavoidably:

1. a change to `GLVScalar.cuh` + `pinning.cu` device code, hence
2. **inert** unless `qsb_carrier_sm89.h` is regenerated (note 01), which needs
   `nvcc -arch=sm_89` + `cuobjdump` + a shell, and
3. unverifiable here: the chain trip count, the `QSB_PMIX12` warp-uniform
   predicate, the `QSB_CHAIN_PEEL`/`QSB_PHI_HOIST` compile-time peeling, the
   `kernel_build_gtable` segment loop, the `GT_LO`/`GT_HI` ladder bound
   (`QSB_GT_RADIX_BITS 14` must cover the largest odd multiplier of the new
   widest segment), and the register/spill budget at 128 registers would all
   have to be re-derived and compiled. Under `QSB_GLV11` the current hot kernel
   is already at 128 registers with 0 spills (`SUBMISSION-CODEX-20260925-G3.md:23`);
   N=8 adds three live chain terms.

Writing that patch without a compiler would be exactly the "manufacture a patch
to look productive" failure the brief warns against. I am not writing it.
