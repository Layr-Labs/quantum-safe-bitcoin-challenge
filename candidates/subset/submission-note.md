# Subset: the promoted `d052bc3d` GLV11 P18 chain with kshitij-hash's paired single-reduction chain addition (`QSB_Y_PAIR`, `8f99a3e9`) and half of the warps on the six-term Q layout (`QSB_Q_MIX` 2), a native image rebuilt from this exact tree, and ercumentyildirim's `bfe57794` host side (terrapinelf's `2d1631b0` co-grinder lane on `82d8493f`'s floating producers, blocking GPU waits, the host core for the co-grinder, a memory-gated 9-window host table)

Prepared with Claude Opus 5.5 in Claude Code. This host has no GPU and no AVX-512; everything below that says "measured" was measured by the ranked runner or by the authors named, and everything I did myself is a build, a byte comparison or an analysis of public ranked artifacts.

## Summary

This package adds no new arithmetic of its own. It is a composition of the two strongest queued subset tickets, which are disjoint in the files they change on top of the promoted record:

- the device side of kshitij-hash's `8f99a3e9`: the chain addition carries the pair `(Q', R)` and reduces once, so one reduction and one product per chain addition go away (`QSB_Y_PAIR` 1, `QSB_SC_PARK` 1; 127 registers, 0 spills);
- the whole host side of ercumentyildirim's `bfe57794` and its one device knob, `QSB_Q_MIX` 4 → 2 (terrapinelf's `92a51c8c` setting);
- a native sm_89 image rebuilt from this exact tree with the tree's own `build_carrier.sh` and CUDA 12.8.93.

The composition is chosen from an analysis of 76 public ranked runs, split exactly into GPU and co-grinder parts (below).

## Starting point

The promoted subset record: submission `d052bc3d` (kshitij-hash), commit `61cb94f`, 691,630,437/s. The checkout was the benchmark branch tip `f0e453d`, whose `candidates/subset/` is byte-identical to `61cb94f`'s.

## What the ranked runs say (analysis of public artifacts)

Every ranked subset run uploads `run-subset.json` with its verified hits (artifact `benchmark-diagnostics-subset-<run>`). The GPU grinds 128 of the C(13,3) = 286 window patterns and the co-grinder the other 158, so counting each hit's last three skip indices splits a score exactly into a GPU part and a CPU part (the 128th most frequent pattern had 612–656 hits and the 129th 52–81 in every run, so the split is unambiguous). I did this for all 76 successful subset runs from 2026-09-25 20:36 UTC to 2026-09-27 08:28 UTC. All of them ran on the same host (`starkware-rtx4090-leadergpu-2-…-3512797`, one job at a time), so there is no runner lottery on this track.

| family (classified by the self-reported peak rate) | runs | GPU part, M/s | CPU part, M/s | total, M/s |
|---|---:|---:|---:|---:|
| GLV12 trees (`9f8a33d8` line), CPU > 40 | 23 | 628.6 (sd 3.4) | 46.3 | 674.9 (sd 5.0) |
| GLV11 P18 trees, light co-grinder (CPU < 55) | 17 | 629.0 (sd 7.6) | 47.6 | 676.6 |
| GLV11 P18 trees, heavy co-grinder (CPU ≥ 57) | 8 | 628.4 (sd 4.0) | 60.1 | 688.5 (sd 4.4) |

What I take from it:

1. **The ranked GPU part has been flat at about 628–632 M/s for a day.** The P18 chain reads +4.4% on cool, unthrottled cards (843 vs 806 M/s in `d052bc3d`'s note) but only about +0.1–0.4% on the ranked card, which is thermally limited (about 317 W, 1.7 GHz, 90 °C per earlier notes). This agrees with terrapinelf's energy model in `92a51c8c` (runner energy ≈ 0.66 × core energy + 7.6 nJ per cold DRAM record): P18 trades field additions for cold records, and the ranked card charges for cold records.
2. **The record's 643.32 GPU part is a draw about two standard deviations high.** Its exact-source redraws read 634.72 (`f10929b4`) and 618.72 (`4bbb03cc`). The promotion bar (698.55) therefore needs about +1.5% over the heavy-co-grinder family mean, not +1% over the record.
3. **A heavier co-grinder does not cost the P18 GPU part** within this sample (629.0 light vs 628.4 heavy), so co-grinder rate is additive. The best ranked family is `789aed1b` / `a7727680` / `fa8a3d22` (terrapinelf's `2d1631b0` lane or its w10 predecessor on floating producers): 692.74, 689.87, 692.93.
4. There is a visible slow stretch on 2026-09-27 05:00–07:00 UTC (GPU parts 616.7–622.6 for trees that drew 628–634 at other hours), probably ambient temperature at the host.

## What this package contains, against `d052bc3d`

| file | source | change |
|---|---|---|
| `hit_filter_field_sc.cuh`, `y_pair_sc.cuh` (new), `hit_filter_field.cuh`, `tests/gpu_epochs/tree_inverse.cuh` | `8f99a3e9`, unchanged | `QSB_Y_PAIR`: the chain trip carries `(Q', R)` with `Q' = X3 − V` and the next trip forms `R = red(S·ZZZ + Q'·R)`, one reduction for both products; the seed emits the pair form and the last addition resolves it. `QSB_SC_PARK`: the first candidate's Z product waits in the idle product-tree arena of shared memory across the second candidate's front call, so the pair fits in registers |
| `tests/gpu_epochs/tree.cu` | `8f99a3e9`, plus one line | the `QSB_Y_PAIR` call sites, the park and reload, the new knobs in the image fingerprint, blocking slot events (`QSB_HOST_BLOCKING` 1, identical to `bfe57794`); **and `QSB_Q_MIX` 4 → 2** |
| `CpuGrindSubset.h`, `tests/gpu_epochs/host_producers.h` | `bfe57794`, unchanged | terrapinelf's `2d1631b0` lane (weighted batch-affine prefix, L2-targeted row prefetch, memory- and huge-page-gated 10/11/12-window table, safegcd inversion, split fold) with ercumentyildirim's host-core placement under blocking waits and the memory-gated 9-window table (`QSB_CPU_TRY9`); `82d8493f`'s floating producers plus the one `g_share_cpu = -1` symbol |
| `qsb_carrier_sm89.h` | rebuilt here | native sm_89 image of this tree |
| `SOURCE-MANIFEST.json`, this note | here | regenerated |
| `tests/gpu_epochs/tree.cu.orig`, `hit_filter_field_sc_aluz.cuh` | dropped | an unused backup (as in `bfe57794`) and a generated variant compiled only under `QSB_SC_ALUZ=1` (default 0) |

Everything else is `d052bc3d`'s, byte for byte. `subset.cu` is unchanged.

### Why these two tickets and not others

- `8f99a3e9` is the only queued ticket with new device arithmetic that was measured with interleaved controls: +0.47%, +0.38%, +0.47% GPU rate at a 350 W cap in its three passes, bitwise identical point-addition outputs on 19,800 random states, 127 registers and 0 spills. It ships the `2d1631b0` engine at the record's light thread footprint; the ranked split above says the heavier footprint is worth about +12 M/s of co-grinder rate at no measurable GPU cost, so this package takes the device side of `8f99a3e9` and the host side of `bfe57794`.
- `bfe57794` is `789aed1b` (ranked 692.74: GPU 631.56 + CPU 61.18) plus blocking waits with the host core given to the co-grinder, the 9-window table, and `QSB_Q_MIX` 2.
- `QSB_Q_MIX` 2: terrapinelf's local sweep in `92a51c8c` read −0.47% GPU energy per candidate against 4 (8 of 8 ABBA rounds lower); its ranked draws so far (634.72 in `eb9ee8f3`, 622.84 in `32688f7a`, 632.00 in `fa8a3d22`) average 629.9, inside the family band. It only chooses which precomputed layout a warp reads partial sums from, so it is exact and cheap to keep.
- Not taken: `QSB_Q_MIX` 1 (+1.0% energy locally against 4); `QSB_R_CBANK` 1 without an image rebuild (the carrier would fall back to JIT); the light co-grinder footprint (lower ranked totals); start-up work (already about 0.75 s to the first launch).

## Build and checks done here

All builds ran in an amd64 container with CUDA 12.8.93 (`nvcc` release 12.8, V12.8.93), the ranked runner's toolkit.

| check | result |
|---|---|
| toolchain reproduces `bfe57794`'s committed image from `bfe57794`'s tree | byte-identical, cubin sha256 `45f988c8c5e4529d…` |
| toolchain reproduces `8f99a3e9`'s committed image from `8f99a3e9`'s tree | byte-identical, cubin sha256 `003e3d39b7a6283f…` |
| `build_carrier.sh 24` on this tree | cubin sha256 `f74548427859ec03273f05e9151c810716c6475596e0213a0db3915e468aa5dc`, 462,496 bytes, digest kernel LTC64B loads: 3 |
| `kernel_digest` resources | 127 registers, 0 bytes stack, 0 spill stores, 0 spill loads |
| the harness's own build line (`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`) | builds; no kernel spills |
| build fingerprint: the image's `qsb_carrier_knobs` against the host binary's `QSB_CARRIER_KNOBS` | identical, 1,793 bytes, includes `QSB_Q_MIX=2;QSB_Y_PAIR=1;…;QSB_SC_PARK=1`, so `qsb_carrier_init` will load the image |
| CPU reference smoke test of the harness (`QSB_GRINDER=cpu`, N = 4) | `RESULT: PASS` (harness only; it does not run this kernel) |

## Exactness

- `QSB_Y_PAIR`: kshitij-hash's argument in `8f99a3e9` holds unchanged, because the files are theirs byte for byte. The four-operand routine equals `red(a·b + c·d)` on 101,452 cases; its only mismatches are limb-2 carry drops of the class the record's own product already drops, each predicted by the fold model; the whole point addition gives bitwise identical X3, ZZ3, ZZZ3 and Yoff on 19,800 random canonical states. The speculative filter can only lose a candidate, and every GPU tentative is re-derived by the exact host gate before it is written.
- `QSB_Q_MIX` 2: the chain loop is the rolled `QSB_SC_PP` 0 loop, which walks either descriptor list per warp (`g = 0` P18, `g = 1` six-term); `8f99a3e9` ran it at `QSB_Q_MIX` 4, where one warp in four already takes the six-term list, so both paths of this image were exercised by its validation. The start-up self-check `qsb_s3_selfcheck` replays both descriptor lists on every run.
- Co-grinder: unchanged from `bfe57794`; every CPU hit passes the exact OpenSSL gate `qsb_hv_check` before it is appended, and any failure on the 9-window path (mmap, huge-page coverage, table check) falls back to 10, 11, 12 windows.

## Expected result and caveats

- Expected on the ranked host: GPU part about 628–634 (family mean plus +0.4% for the pair chain if it carries over) and co-grinder part about 60–64 (the `2d1631b0` lane, a few percent more if the 9-window rule fires). Total about 690–698, against the promotion bar of 698.55. I estimate the chance of promotion on one draw at 10–20%; I state this so nobody reads the composition as a claimed gain.
- Not run on a GPU by me. The device files are `8f99a3e9`'s and the knob is `bfe57794`'s, but this exact combination has not been executed anywhere before this ticket.
- The 9-window host table has not run on the ranked host before `bfe57794`; that ticket runs earlier in the queue and will show whether it fires (epoch-walk diagnostic bit 19).
- If `8f99a3e9` or `bfe57794` is promoted before this ticket runs, the bar moves to about +1% over it; this package contains the device side of the first and all of the second.

## Reproducing

```
./setup.sh subset
./benchmark.sh subset
```

Off the ranked runner, the harness's command grinder runs the same build:

```
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/{bench}/{bench}.cu --no-build' ./benchmark.sh subset
```

Regenerate the native image after any device-code edit (in `candidates/subset`): `NVCC=/path/to/cuda-12.8/bin/nvcc ./build_carrier.sh 24`. The GPU/CPU split of any ranked run: download its `benchmark-diagnostics-subset-<run>` artifact, count hits per `skip[6:9]`, take the 128 most frequent patterns as the GPU's, and rate = hits × 2^23 / `elapsed_s`.

Kill switches: `-DQSB_Y_PAIR=0` (the record's chain; set `QSB_SC_PARK=0` with it), `-DQSB_Q_MIX=4` (the record's mix), `-DQSB_CPU_TRY9=0` (at most 10 windows), `-DQSB_HOST_BLOCKING=0` (spin wait), `-DQSB_CPU_GRIND=0`, `-DQSB_HOST_PRODUCERS=0`. Any device knob change needs `build_carrier.sh` again, or the carrier falls back to the compute_52 image.

## Base and attribution

- **kshitij-hash** (co-author): the pair chain `QSB_Y_PAIR` and the shared-memory park `QSB_SC_PARK` (`8f99a3e9`), and the promoted `d052bc3d` composition this builds on.
- **ercumentyildirim** (co-author): the `bfe57794` host side (host-core placement under blocking waits, the 9-window table), the `789aed1b` / `a7727680` compositions, and the ranked GPU/CPU split method that I extended here.
- **terrapinelf** (co-author): the co-grinder lane (`2d1631b0`), the host-built epoch producers and warp-uniform root (`82d8493f`), the `QSB_Q_MIX` 2 setting and the ranked energy model (`92a51c8c`), the promoted `de5739c9` tree.
- **HyeokxC** (co-author): blocking GPU waits with the host core given to the co-grinder (`0735233a`, `888f5fce`).
- **fkiene**: the GLV11 P18 five-term chain and the per-warp Q-layout mix (`413f83e7`, promoted in `d052bc3d`).
- **Through the base:** Meganpark980320 (`QSB_SHA_FMA_ADD=0`, `296e5e53`), newjordan (`d1ddefca`, `212237f4`), i34-9, Ryun1, Akashneelesh (`7aef224a`) and every contributor those notes credit. libsecp256k1 (MIT, `COPYING-secp256k1`).
- **Mine:** the 76-run GPU/CPU split and family analysis above, the choice of composition, the merge, the image rebuild and the byte checks.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
