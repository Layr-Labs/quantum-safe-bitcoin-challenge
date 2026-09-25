# Subset: i34-9's GLV11 geometry on our GLV12 native-carrier tree (de0d4f55): one fewer addition per candidate, with every cold record fetched with the one-access `.L2::64B` hint (+2.74% ± 0.06 locally vs de0d4f55)

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on a dedicated RTX 4090 (driver 595.71, CUDA 12.8.93 toolchain). The ranked build line and argv are unchanged.

## What this is

This package is our `de0d4f55` with i34-9's GLV11 table geometry from their public `14675ab0`, ported into it. In `de0d4f55`:
- GLV12 four-bank table plus the native sm_89 carrier from newjordan's `d1ddefca`;
- warp-distributed root inverse;
- our no-JIT startup;
- `QSB_SHA_FMA_ADD`.

**GLV11 (P18 layout).**
- **Q component:** keeps its six GLV12 terms.
- **P component:** uses five terms: segment 0 (18 bits, in the pinned 48 MiB), two new cold segments of 27 and 28 bits, and GLV12's cold segments 4 and 5. The new segments hold 2^26 and 2^27 records, appended after the GLV12 table.
- **Per candidate:** 11 table gathers and 10 point additions, instead of 12 and 11, at the price of two more cold (DRAM) records: 6 instead of 4.
- **Table:** 354,501,773 records, 22.7 GB (21,637 MiB). It fits the 24 GB card because the epoch-group buffers are sized exactly (below).

i34-9's `14675ab0` has the geometry but not the native image, so its cold records are two DRAM accesses each. It was ranked at 618.46 with a peak self rate of 811.7. That is above every GLV12 draw (789–793), but with a lower decay ratio (0.762), as expected from more DRAM traffic.

This package fetches all six cold records per candidate with newjordan's one-access `ld.global.cs.nc.L2::64B` load: the cold segments are read evict-first, and the first 16 B slice carries the 64 B prefetch-size hint. Our reading of the runner data:
- that hint lifted GLV12's decay ratio from 0.776–0.785 to 0.794;
- with 6 cold records per candidate instead of 4, it should matter more here.

## Measurement (local, same protocol as our earlier notes)

Paired ABBA on one RTX 4090 at the 450 W cap, fixed generated problem (seed 424242), 62 s arms, 3 rounds, rate between the first and last progress lines:

| build | steady M/s | nJ/candidate | Δ vs `de0d4f55` |
|---|---:|---:|---:|
| `de0d4f55` (GLV12) | 825.55 | 544.3 | — |
| **this package (GLV11)** | **848.19** | **530.0** | **+2.743% ± 0.060** (energy −2.63%) |

For context, our no-DRAM probe on GLV12 put the four cold records at about 4.6% of the energy locally. The two extra cold records here cost about 2.3%, and the saved addition returns about 5%.

**Exactness.**
- On the fixed problem the search is deterministic. In 30 s, `de0d4f55` wrote 2,800 hits and this package 2,828. All 2,800 are in this package's set; the 28 extra come from searching further in the same time.
- At startup the host self-check runs the 11-term walker against `q9_bigtbl_code` (Q) and `q11_bigtbl_code` (P) on 4,096 random scalars.
- The table heal reported 0 off-curve flags, and the 256-sample OpenSSL spot check passed.
- **Unmodified harness** (`benchmark.sh subset`, 90 s, fresh problem seed): see the validation line at the end of this note.
- Every published hit is still recomputed by the exact host (OpenSSL) publication gate.

**Startup.** The search starts at 1.23 s instead of 0.70 s: the GPU table build grows from 0.51 s to 0.96 s for 2.3× the records. That costs about 0.04% of the 1200 s window. The compute_52 image is still never JIT-compiled while the carrier is on.

## Memory: exact epoch-group capacity

The GLV11 table leaves about 1.2 GB for everything else. `QSB_GROUP_CAP_EXACT` (i34-9's, from `14675ab0`) sizes the two epoch-group buffers at 228,771 records instead of 2·epochs+4 = 2,097,156, which frees about 460 MiB.

We checked the bound independently. The epochs are the 6-subsets of {0..136} in lex order, 1,048,576 per launch, and a launch spans rank5(last) − rank5(first) + 1 groups. With exact combinatorics over all 7,838 launches of the search space, the maximum is **181,498 groups**, at the last launch. The bound holds with margin, and the host still checks every launch against it.

Local peak GPU memory is about 23.3 GB of 24.5 GB.

## Implementation

- `tests/gpu_epochs/tree.cu`:
  - `QSB_GLV11` / `QSB_GLV11_P18` knobs: 8 segments, 11 terms, `gt_entries/offset/shift` for segments 6 and 7, static asserts, and the H2-ladder bound for m < 2^28.
  - The 11-term `QSB_S3_DESC_INIT`, and the 11-term host self-check.
  - `QSB_GROUP_CAP_EXACT` with `qsb_group_capacity()`.
  - The three new knobs are added to the carrier build fingerprint.
- `GLVScalar.cuh`: `q11_bigtbl_code`, used by the self-check.
- `qsb_carrier_sm89.h`: regenerated with `build_carrier.sh` and CUDA 12.8.93. cubin sha256 `0810c43218e80725…`, 478,240 B, 2 `LTC64B` loads in the digest kernel, 0 spills in every function.
- **Unchanged:**
  - the chain, walker and loaders, which are generic in the term count; the cold test `off >= GT_DENSE_ENTRIES` picks up segments 6 and 7;
  - the builder and heal kernels, the pipeline, the host gate and the no-JIT startup.

Default-build PTX sha256 prefix: `2804dda9cf09`. It is only JIT-compiled if the carrier is off.

## Other measurements this morning (on `de0d4f55`, not shipped)

| variant | Δ |
|---|---:|
| σ shifts via IMAD.HI (`QSB_SHA_FMA_ROT=8`) | −0.212% ± 0.042 |
| SHA256d second compression, constant-folded (`_SHA256TransformDigest32Q`) | −0.051% (hit sets identical) |
| the same with FMA-pipe adds | −0.688% |
| `QSB_SHA_FMA_ROT=3`, `=12` | spill 12 B |

FMA-pipe adds pay only in the pubkey gate.

## Base and attribution

- **GLV11 geometry, the P18 layout and the exact group capacity:** i34-9's `14675ab0`. Co-author.
- **Tree:** newjordan's `d1ddefca` (GLV12 four-bank geometry, native sm_89 carrier with the `.L2::64B` cold-record fetch, two-slot pipeline, persisting L2 window, exact host gate, `sha_gate_fma.cuh`). Co-author.
- **GLV12 four-bank port:** ercumentyildirim's `933abead`. Co-author.
- **fk-lean tree and FMA-pipe gate adds:** fkiene (`eaba5205`, `b864a72c`). Co-author.
- **Carrier design:** Ryun1's pinning `25bd990a`. Co-author.
- **Warp root inverse:** newjordan's `5b198ddf`, file set via i34-9's `78208a18`.
- **Ours:** the no-JIT startup (`ef1b37e9`), `QSB_SHA_FMA_ADD` enabled with its census (`de0d4f55`), and this port with the group-capacity proof.
- **Promoted crown:** `7aef224a` (Akashneelesh) and every contributor it credits.

All inherited source, GPLv3 notices (`COPYING`, `COPYING-secp256k1`, VanitySearch headers) and attributions are retained.

## Packaging

Only `candidates/subset/` changes. The harness, verifier, problem, setup, benchmark, workflow and sibling track are untouched, and no binary or build stamp is included. There are no includes outside `candidates/subset/`.

**Validation of this exact package:** the unmodified harness (`benchmark.sh subset`, 90 s, fresh problem seed) verified 8,953 of 8,953 hits: `RESULT: PASS`.
