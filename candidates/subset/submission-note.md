# Subset: terrapinelf's `80212db2` host side (v3 epoch producers, round-9 9-window co-grinder with the 16-lane key-hash schedule, single-accumulator IFMA columns and pattern groups) on the promoted `521075fe` device side built with the record's `QSB_Q_MIX` 2 (image `f7454842`, `QSB_R_CBANK` 0), which our in-run ranked A/B measured faster than `003e3d39`

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 + i7-13700K host (CUDA 12.8.93), with measurements on a rented AMD EPYC 9554 (Zen 4, SMT on) + RTX 4090 host.

## Starting point

The promoted subset record is `5c7e36c5` (jacklightChen), 708.41 M/s: GPU part 644.83 M/s and co-grinder part 63.58 M/s from its public hit list (window-triple classification: the GPU walks the 128 triples of the late window, the co-grinder the other 158). The promotion bar is 715.49. Our previous ticket `6b37a2f8` (ranked 697.67 M/s, GPU 633.84 + CPU 63.83 M/s) was our `QSB_Q_MIX` 4 package with an in-run A/B of this package's image (the record's `QSB_Q_MIX` 2, `f7454842`) on a second epoch region; its public hit list decodes to B/A = 1.01133 95% CI [1.00772, 1.01493]. A per-SM phase-skew ticket (`f947af05`) queued after it was cancelled in favour of this package once the decode was known. Before it, `2ba96cf0` (ranked 706.83 M/s, GPU 643.01 + CPU 63.82 M/s), `0cdb0861` (ranked 702.74 M/s: GPU 638.62 + CPU 64.12) and `86c643ae` (705.37 M/s: GPU 641.69 + CPU 63.69) carried the same device side with our own host side.

This package keeps terrapinelf's host side and switches the device image to the record's `QSB_Q_MIX` 2 image `f7454842`: our in-run A/B ticket `6b37a2f8` ran it and `003e3d39` in alternating 60 s slices on disjoint epoch regions under one thermal state on the ranked card, and its public hit list decodes to B/A = 1.01133, 95% CI [1.00772, 1.01493] (`QSB_Q_MIX` 2 over `QSB_Q_MIX` 4).

## What this package is

Every file is terrapinelf's `80212db2` (ranked 701.49 M/s: GPU 635.52 + CPU 65.97), byte for byte, except two lines of `subset.cu`:

- `#define QSB_R_CBANK 1` is removed, so the promoted default `QSB_R_CBANK` 0 applies (with `QSB_Q_MIX` 4 this tree builds the promoted image `003e3d39…`, the image of our previous packages).
- The inert re-measurement tag on line 1 is renamed (unreferenced).
- `#define QSB_Q_MIX 2` is added on line 2 of `subset.cu` (the record's layout mix: every second warp decodes Q with the six GLV12 terms), so `build_carrier.sh 24` with CUDA 12.8.93 regenerates the record's image byte for byte: cubin sha256 `f74548427859ec03…`, 0 bytes stack, 0 spills (the image of `5c7e36c5`).

So the device code is the promoted `521075fe` chain (`QSB_Y_PAIR` pair addition with one reduction, shared-memory park, the P18 five-term GLV11 chain, `QSB_Q_MIX` 2), and the host side is terrapinelf's: host-built epoch descriptors and first-block SHA states on three SHA-NI producer threads (v3), their round-9 host loop with blocking waits, and their co-grinder (8-lane AVX-512 IFMA + SHA-NI on the co-grinder's window triples; weighted batch-affine prefix, L2-targeted row prefetch, a memory-gated signed 9-window table with the top window's C fold in place, the 16-lane key-hash message schedule with 4-wide SHA-NI rounds, single-accumulator IFMA product columns, always-inlined small field operations, and the block-0 pattern-group filter after cefika's `4a197f06`).

## Why: a head-to-head on the ranked topology

The ranked host exposes 32 logical CPUs; the epoch-walk diagnostics of our `86c643ae` and `0cdb0861` show SMT siblings among the workers' CPUs and the 9-window table, and the co-grinder parts of the public ranked runs (61–66 M/s) match 16 Zen 4 cores with two workers each. We rented an AMD EPYC 9554 (Zen 4, SMT on) with an RTX 4090 and ran the complete ranked binaries (N = 24, real problem, the GPU grinding, the whole process pinned to 16 cores x 2 threads, 9-window table), alternating our `0cdb0861` package and terrapinelf's `80212db2`, and read each co-grinder's own final line (candidates since start) and its table-ready time:

| package | runs: co-grinder M/s over the run | table ready | steady rate, candidates / (run - table ready) |
|---|---|---|---|
| our `0cdb0861` host side | 56.63, 55.55, 56.30 | 10.34, 10.99, 10.93 s | 61.26, 60.88, 61.17 M/s |
| terrapinelf `80212db2` host side | 58.76, 58.78 | 6.95, 6.97 s | 61.83, 61.64 M/s |

terrapinelf's co-grinder is about 1.0% faster in steady state on this topology and builds its 9-window table about 4 s sooner, together about +1.3% of the co-grinder part over a 1,200 s run. The public ranked draws point the same way: `80212db2` drew a co-grinder part of 65.97 M/s against 64.12 for our `0cdb0861`.

On the same host we also swept our own co-grinder's knobs on 16 cores x 2 workers (CPU-only bench, 9-window table, 14 base runs at 64.29 M/s +- 0.03): table-row prefetch distance 4 / 6 / 12 and first-window prefetch 4 within +-0.14%, 80 instead of 100 patterns -0.13%, batch 1,280 +0.02%, batch 768 -0.51%, 31 instead of 32 workers -0.91%, 10 windows instead of 9 -3.98%. Our lane was at its optimum; the remaining gain was terrapinelf's lane itself.

## Why `QSB_Q_MIX` 2

The `QSB_Q_MIX` knob trades cold 64 B table records for field additions (Q_MIX 4: 7.5 cold records and 9.25 additions per candidate; Q_MIX 2: 7 and 9.5). On a power-capped local card Q_MIX 2 read -0.2 to -0.45%, and the public ranked means of the two images were equal within noise (`003e3d39` 636.1 over 18 draws, `f7454842` 635.0 over 9). Measured in one run on the ranked card, Q_MIX 2 came out ahead (B/A = 1.01133, 95% CI [1.00772, 1.01493]), so this package takes it as its only image. `QSB_R_CBANK` stays 0 as in our previous packages.

## Validation of this exact package

| check | result |
|---|---|
| `build_carrier.sh 24`, CUDA 12.8.93 | cubin sha256 `f74548427859ec03…`, 0 bytes stack, 0 spills: byte-identical to `5c7e36c5`'s image |
| unmodified harness (`benchmark.sh subset`, `QSB_GRINDER=cmd:… gpu_wrap.py`), N = 24, 120 s, fresh seed | 12,278 / 12,278 verified, `RESULT: PASS` |
| the same, 1,200 s | 124,423 / 124,423 verified, `RESULT: PASS`; no duplicate co-grinder hits |
| full ranked binary on the EPYC 9554 + RTX 4090 (above) | `Native sm_89 carrier: on`, 9-window table, co-grinder and GPU running |

terrapinelf's own validation of this host side (their note for `80212db2`) covers the co-grinder's exactness under their AVX-512 checks; the same host side ran on the ranked host in their `80212db2` and, without the pattern groups, in `a33e04c3`.

## Caveats

- The GPU part of a single ranked draw varies by several M/s between draws of the same bytes (`003e3d39`: 634.19 to 642.62 in our tickets).
- The head-to-head is two to three 150 s runs per package on a shared rented host; the steady-rate column removes the table build but not all host noise.

## Base and attribution

- **terrapinelf** (co-author): the whole host side of this package (`80212db2`: v3 producers, round-9 host loop, the co-grinder with its 9-window table, key-hash schedule, IFMA columns and pattern groups), and the r7 lane our earlier tickets carried.
- **cefika** (co-author): the block-0 pattern-group selection idea (`4a197f06`) that the pattern groups follow.
- **RealAdii** (promoted `521075fe`, cited): the promoted device side and host loop this tree is built on; **kshitij-hash** (promoted `d052bc3d`, cited): the `QSB_Y_PAIR` chain; **fkiene**, **Meganpark980320**, **newjordan**, **HyeokxC**, **i34-9** and every contributor credited in the source notices through the promoted lineage.
- **Ours:** the head-to-head and sweeps on the ranked topology, the choice of host side and image, the ranked diagnostics of our tickets that established the topology, and our earlier host-side work that this lineage carries (the 9-window rule and floating producer placement from `a141df2b`).

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes. Kill switches are terrapinelf's (see the source comments: `-DQSB_CPU_KH16=0`, `-DQSB_CPU_MRG=0`, `-DQSB_CPU_AINL=0`, `-DQSB_CPU_GRIND=0`, `-DQSB_HOST_PRODUCERS=0`).
