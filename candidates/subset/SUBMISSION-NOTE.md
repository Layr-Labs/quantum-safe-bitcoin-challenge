Model: Claude Opus 5.5
Harness: Claude Code

# Subset: `ef1b37e9` (GLV12 + native carrier + no-JIT) with i34-9's lean GLV scalar split — 2.6 fewer warp instructions per candidate (−0.37% of the digest kernel's dynamic instructions), bit-exact

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on RTX 4090s (CUDA 12.8.93 toolchain). The ranked build line and argv are unchanged, and only `candidates/subset/` changes.

## What this is

This package is terrapinelf's public subset tree `ef1b37e9` (official 626.61) with one device change: the GLV scalar split in `GLVScalar.cuh` is replaced by i34-9's lean version from `adfa8aaa`, as carried unchanged in terrapinelf's `7ee5c52a`. We take only the scalar split. The table geometry stays GLV12 (12 gathers, 11 additions per candidate); none of the GLV11 geometry of `7ee5c52a`/`14675ab0` is included, and `q11_bigtbl_code` in the file is compiled out (`QSB_GLV11` is undefined).

The split runs once per candidate, before the table walk, and decomposes the scalar `k` into `±mag0 + λ·(±mag1)`. i34-9's version has three compile-time switches, all on:

| switch | what it does |
|---|---|
| `QSB_GLV_LEAN` | the 32×32→64 partial products of the rounding coefficients and of `q9_product129` are written as explicit `mul.wide.u32` / `mad.wide.u32`, so ptxas issues one `IMAD.WIDE` per product instead of widening both operands first |
| `QSB_GLV_ROUND_CC` | the 128-bit rounding add takes its carry from the add's carry flag (`add.cc` / `addc`) instead of a compare |
| `QSB_GLV_HIGH15_HI` | in the high-half coefficient product, diagonal 10 is only observed through its carry into diagonal 11, so only the high 32 bits of its five products are kept; the discarded low halves can change word 11 by at most 8 (g1) or 7 (g2), and the exact-fallback band of the coefficient wrappers is widened by exactly that amount (`0x7ffffff8` / `0x7ffffff9` instead of `0x7ffffffc` / `0x7ffffffd`) |

The first two are pure instruction-selection changes with identical results. The third is exact by construction: any input where the omitted carry could change the rounded coefficient falls into the widened band and takes the unchanged exact fallback (`q9_coeff_fallback`). All three knobs are added to the carrier build fingerprint (`QSB_CARRIER_KNOBS` in `tree.cu`), so a host/image mismatch still switches the carrier off.

## Measurements

**Instructions (NVBit, dynamic, one full batch of 134,217,728 candidates):**

| build | warp instructions / candidate | Δ |
|---|---:|---:|
| `ef1b37e9` | 698.416 | — |
| this package | 695.822 | **−2.594 (−0.371%)** |

That is about 83 fewer thread instructions per candidate. Static SASS of `kernel_digest`: 15,056 → 14,968 (−88; `IMAD` −16, `IADD3` −16, `SEL` −10, `ISETP` −28). Registers stay at 128, with 0 bytes of stack and 0 bytes of spill.

**Local throughput (`benchmark.sh subset`, 180 s arms, same RTX 4090, fixed problem seed 2310742569, order A B B A):**

| arm | `ef1b37e9` verified hits | this package verified hits |
|---|---:|---:|
| 1 | 17,346 | 17,399 |
| 2 | 17,346 | 17,399 |

That is **+0.31%** candidates searched in the same window, with every run `RESULT: PASS`. Repeated runs of the same build gave identical hit counts, so this card's run-to-run spread at 180 s is about one batch (~0.1%).

**Exactness.** The `ef1b37e9` run's 17,346 hits are all contained in this package's 17,399; the 53 extra hits come from searching further in the same time. i34-9's own check (`adfa8aaa`) and terrapinelf's (`7ee5c52a`, identical 30 s hit sets with and without the split) agree.

**1200 s runs (full ranked length, unmodified harness, same seed), one pair on each of two further RTX 4090 hosts, in opposite orders:**

| host, order | `ef1b37e9` verified hits (rate) | this package verified hits (rate) | Δ |
|---|---:|---:|---:|
| host 2, base first | 114,074 (796.12 M/s) | 114,608 (799.89 M/s) | +0.47% |
| host 3, package first | 115,427 (805.00 M/s) | 116,156 (810.00 M/s) | +0.62% |

Both runs: `RESULT: PASS`, every hit verified. In each pair all of `ef1b37e9`'s hits are in this package's set: 114,074 of 114,074 and 115,427 of 115,427.

## Files changed relative to `ef1b37e9`

- `GLVScalar.cuh`: i34-9's lean split (`adfa8aaa`, byte-identical to the copy in `7ee5c52a`).
- `tests/gpu_epochs/tree.cu`: `QSB_GLV_LEAN`, `QSB_GLV_ROUND_CC`, `QSB_GLV_HIGH15_HI` added to the carrier fingerprint (one line).
- `qsb_carrier_sm89.h`: regenerated. cubin sha256 `841aaeee654a9734…`, 470,944 B, 2 `LTC64B` loads in the digest kernel, 0 spills.
- `SOURCE-MANIFEST.json`, `SUBMISSION-NOTE.md` (this note; it replaces the inherited `submission-note.md`).

Default-build PTX sha256 prefix: `40f055385b7f` (only JIT-compiled if the carrier is off).

## Unchanged from `ef1b37e9`

Everything not listed above is byte-identical to terrapinelf's `ef1b37e9` tree:
- newjordan's GLV12 four-bank geometry (ercumentyildirim's `933abead` port) and native sm_89 carrier with the one-access `.L2::64B` cold-record fetch (`d1ddefca`);
- the warp-distributed root inverse (newjordan `5b198ddf`, file set via i34-9 `78208a18`);
- the no-JIT startup (`ef1b37e9`): every search-path kernel, the table build and the heal scan launch from the native image, and the compute_52 PTX is only compiled if the carrier falls back;
- the two-slot host pipeline, the persisting L2 window, the batch geometry (262,144 blocks × 256 threads, 49,152 B shared, 128 registers), and the exact OpenSSL host publication gate.

The carrier image `qsb_carrier_sm89.h` was regenerated with the tree's own `build_carrier.sh` and CUDA 12.8.93. Before editing anything we rebuilt the unchanged `ef1b37e9` tree the same way and got its shipped cubin bit for bit (sha256 `af8b2c96dd90f7cd…`), and its default-build PTX prefix `1590a6ee0d3e`, so the toolchain matches. The same package was rebuilt independently on three RTX 4090 hosts and gave the same cubin hash each time.

Every tentative hit is still re-derived by the unchanged exact host gate before publication, and the harness verifier re-derives every published hit.

## How the numbers were measured

- **Instructions per candidate** are dynamic counts, not static ones. We ran NVBit 1.7.7.3 with a small per-PC counter tool (a variant of NVBit's `instr_count` that keeps one counter per SASS instruction of `kernel_digest`) on the native carrier image, over one full batch (262,144 blocks, 134,217,728 candidates), and divided the warp-level instruction total by the candidate count. On `ef1b37e9` that is **698.42 warp instructions per candidate** (about 22,350 thread instructions). The per-PC counts sum exactly to NVBit's own kernel total.
- **Local throughput** is `benchmark.sh subset` itself, unmodified harness, 180 s per arm, fixed problem seed, alternating the two builds on one RTX 4090 (stock 450 W limit) with each arm started below 45 °C. Candidate order is deterministic on a fixed problem, so the number of verified hits in a fixed window counts the work done (one hit ≈ 8.4 M candidates). Every run: `RESULT: PASS`, all hits verified.
- **Exactness**: on the fixed problem the two builds enumerate the same candidates in the same order, so the slower build's hit set must be a subset of the faster build's. We compared the complete hit identities (index set plus recovery id) of paired runs.

## Commands

```bash
NVCC=/usr/local/cuda-12.8/bin/nvcc ./build_carrier.sh 24          # regenerate the image (already done)
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm        # ranked build line
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset.cu' QSB_SECONDS=180 ./benchmark.sh subset
```

## Base and attribution

- **Tree:** terrapinelf's `ef1b37e9` (no-JIT startup and the warp root inverse port; the package this one extends). Co-author.
- **GLV12 + native carrier base:** newjordan's `d1ddefca` (GLV12 four-bank geometry on the fk-lean tree, native sm_89 carrier with the one-access cold-record fetch, two-slot pipeline, persisting L2 window, exact host gate, `sha_gate_fma.cuh`). Co-author.
- **GLV12 four-bank port:** ercumentyildirim's `933abead`. Co-author.
- **fk-lean tree and FMA-pipe gate adds:** fkiene's `eaba5205` and `b864a72c`. Co-author.
- **Carrier design:** Ryun1's pinning submission `25bd990a` (`QsbCarrier.h`, `build_carrier.sh`). Co-author.
- **Warp root inverse:** newjordan's `5b198ddf`; the file set is i34-9's port `78208a18`.
- **Promoted crown:** `7aef224a` (Akashneelesh) and every contributor it credits, and the whole chain credited in the inherited files.
- **Lean GLV scalar split (`GLVScalar.cuh`, `QSB_GLV_LEAN` / `QSB_GLV_ROUND_CC` / `QSB_GLV_HIGH15_HI`):** i34-9's `adfa8aaa`, as carried into the GLV12 native-carrier line by terrapinelf's `7ee5c52a`. Co-authors.
- **Measurement and packaging of this combination:** ours.

All inherited source, GPLv3 notices (`COPYING`, `COPYING-secp256k1`, VanitySearch headers) and attributions are retained.

## Packaging

Only `candidates/subset/` changes. The harness, verifier, problem, setup, benchmark, workflow and sibling track are untouched, and no binary or build stamp is included. There are no includes outside `candidates/subset/`.
