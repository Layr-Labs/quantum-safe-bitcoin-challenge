# Subset: the four-bank GLV12 table geometry from pinning, ported to subset — 12 lookups and 11 additions, large banks streamed evict-first

Effort: max. Development context: Claude Opus 5.5 driving Claude Code, with sub-agents on the same
model for implementation and measurement (planning by a Claude Fable 5.1 sub-agent). Measured on RTX
4090s at their 450 W cap, every run serialised behind a machine lock.

## Base and credit

The base is our previous subset draw tree, **fk-lean**: **@fkiene**'s subset tree `eaba5205` with two
pipe-routing defaults turned off (`QSB_SHA_FMA_ADD`, `QSB_R_CBANK`). Underneath it is the chain credited
in that tree: terrapinelf's PR1088 composite, fkiene's and hybridnoise's `QSB_K32` / `QSB_FUSE_X3`,
Akashneelesh, Saviour1001, DrCleverHans, mitchuski, Babbaragga, dun999, Meganpark980320, @EvanYan1024,
owizdom, DPZZxlz, jacklightChen and our own PR868. All license and attribution notices are retained.

**The table geometry is the pinning track's.** The GLV12 six-segment shared table, its signed-digit
decoder with the telescoping bias and the GPU table builder are **odinfree**'s `QSB_BIGTBL` (pinning
`d71d3b7b`). The four-cached-bank cut used here — segment widths [18, 19, 18, 18, 27] with the top field at
shift 100 — was first published by **0xCramJam** (pinning `c13f3832`), carried onto the promoted pipeline
by **Saviour1001** (`3ecc74b2`) and promoted on pinning by **fkiene** (`871963fd`); we built the same cut
independently for pinning as well. `GLVScalar.cuh` is the pinning tree's file; the GLV lattice constants
are libsecp256k1's (Pieter Wuille and contributors, MIT; notice in `COPYING-secp256k1`). What is new here is the port to subset's kernel
and its fixed base, and the choice of load path for the large segments.

## What changed

One switch, `QSB_S3` (default 1; 0 restores fk-lean byte for byte — the preprocessed translation unit is
identical). Files: `tests/gpu_epochs/tree.cu`, `tests/gpu_epochs/pair_shared.cuh` (two guards), new
`GLVScalar.cuh`, new `COPYING-secp256k1`.

* **Scalar split.** Each candidate's fixed-base scalar is split with GLV into two ~128-bit signed
  components (λ, β the cube-root-of-unity pair; ψ(x,y) = (βx, y) applied once between the two halves of the
  walk). This works for subset's base point A as for any point of secp256k1.
* **Table.** Six shared segments per component instead of fk-lean's 15-chunk 64 MiB table:
  records [2^18, 2^18, 2^17, 2^17, 2^26, 85,279,885] = 153,175,181 records (9.35 GiB). **12 lookups and 11
  additions per candidate** instead of 15 and 14.
* **Residency.** Segments 0-3 (48 MiB) are stored first and the persisting L2 window is clamped to exactly
  those bytes. Segments 4 and 5 stream from DRAM and are loaded with **`ld.global.cs` (evict-first)**; the SASS
  shows `LDG.E.EF` on exactly those loads and the ordinary cached path for segments 0-3. On subset the load
  path matters: measured with probes on fk-lean, four streamed lookups through the ordinary path cost about
  16% of rate, the same four through evict-first loads about 5.4%.
* **Builder.** The pinning builder's three-level split (host ladders ≤ ~4k points per segment, GPU build
  about half a second). A heal pass (`QSB_GT_HEAL`) checks every record for being on the curve and rewrites
  any that is not from OpenSSL before the unchanged spot check; on subset's builder it found zero such
  records on every instance we ran and is kept as insurance, because at 9 GiB the host fallback cannot
  rebuild the table inside a ranked window.
* **Spot check.** 240 samples read directly from device memory (no whole-table copy to the host) —
  **terrapinelf**'s device-side spot check from pinning (`ee1c795d`), extended to the new segment edges.

## Correctness

* **Host math:** the decoder is exact over every field value of every segment, both signs; 2.65M
  magnitudes × 2 signs including constructed extremes reconstruct exactly; the device walker equals the
  reference decode on 10.6M walks; split and ψ constants checked against libsecp256k1; builder ladders equal
  OpenSSL on sampled records including the split edges. The checker was validated by injecting five bugs;
  each failed loudly.
* **Whole table:** a test-only build proves the healed table record by record (on curve, consecutive
  differences equal the step, OpenSSL anchors) on three problem instances; device decode checked end to
  end on 3 × 1M scalars.
* **Harness** (`benchmark.sh` → `harness/gpu_wrap.py`, 240 s): verified == reported on seeds 777 and 4242,
  zero shortfall; hits × 2^23 / candidates 1.0015 and 0.9949 (fk-lean 1.0055 and 1.0009).
* **Exact hit-set diff** against fk-lean over the common launch prefix: seed 4242 identical; seed 777 zero
  lost and one extra (the fk family's known filter-only miss, recovered); all eight A/B pairs identical
  (126,920 common hits, 0 only-fk-lean, 0 only-this). The candidates and public keys are the same; only
  the scalar-multiplication route changed.
* Stage-0 kernel: 128 registers, no stack, no spills. Peak VRAM 11.5 GiB of 24.

## Measurement

Mirrored A/B against fk-lean, 8 rounds of 180 s (4 in each order), **verified hits divided by the process
wall time** (so CUDA init, the larger table build, heal and spot check are included — about +0.37 s per
process), 450 W on every run: **+6.71% ± 0.03%**; the candidate rate from the epoch counter agrees
(+6.99% ± 0.02%). This is one card's local number; the ranked runner will give its own.

## Reproduction

```bash
yukon clone eigenlabs/quantum-safe-bitcoin-challenge qsbc && cd qsbc
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm
```

This tree prints a progress line every 15 s. Do not time anything from the integer seconds printed inside a
progress line; one second on a 75 s window is 1.33%.
