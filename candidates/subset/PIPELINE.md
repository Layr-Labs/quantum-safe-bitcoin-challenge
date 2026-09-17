# Subset: external hierarchical inversion on the promoted tree

Effort: max. Model: Claude Opus 5 (Anthropic), run in Claude (Cowork) on behalf
of the account owner. We had no GPU, so **no local GPU score is claimed**; the
ranked run is the first timing measurement.

## Starting point

The base is the promoted subset tree at `65cd138` (i34-9 / dun999, submission
`99ce8411-5d19-453e-8a4c-9fd9bb14b445`, **495,193,826 verified
candidates/s**). `candidates/subset/` is unchanged at the current `main`
(`372a325`).

What this candidate changes in the base:

- **Unchanged:** every base file is byte-for-byte, except
  `tests/gpu_epochs/tree.cu`.
- **`tree.cu`:** one `#include`, a small host-side dispatch, and an advisory
  host-side L2-persistence hint for the table (see below).
- **New files:**
  - `tests/gpu_epochs/ranked_pipeline.cuh`;
  - this note;
  - `verify/`, CPU-only checks that the official build never reads.

## What changes

The promoted ranked launch is one monolithic `kernel_digest` per 256-lane CTA.
Its product tree lets lane 0 run `_ModInv` inside the CTA while the other 255
lanes wait. This candidate splits the launch into five ordered kernels on the
default stream.

1. **`qsb_ranked_prepare`.** It first runs the promoted work unchanged:
   scheduled window hash, SHA-256d, and the direct-digit 15-point fixed-base sum
   on the 64 MiB mixed table. It then runs `qsb_xyzz_finish_prepare`. Its new
   work:
   - saves `C=ZZ*d^2`, `Y`, `W=ZZ^2*d` and `ZZZ` in eight `ulonglong2` planes
     (128 B per candidate);
   - checkpoints the 254 internal product-tree nodes;
   - publishes one root per CTA.
2. **`qsb_root_group_prepare`.** Groups the roots 256 at a time.
3. **`qsb_invert_super_roots`.** Inverts the group roots with the promoted
   `qsb_block_inverse_tree`. That is one `_ModInv` per 16.8M-candidate launch
   instead of 65,536 inside the search CTAs.
4. **`qsb_root_group_finish`.** Expands the group inverses back to the CTA
   roots.
5. **`qsb_ranked_finish`.**
   - Restores each tree and expands 1/W.
   - Runs the promoted `qsb_xyzz_finish_precomputed`, with u2R from the promoted
     constant `QSB_U2R`.
   - Hashes each compressed key once (recid 0 first).
   - Writes the promoted packed hit records (`ZLAB_HITPATH`).

**How the checkpointed trees work**

- **Canonicalization is lazy, as in the promoted tree.** Internal products use
  `qsb_field_mul_raw`, the super-root inverse normalizes its root, and every
  returned leaf inverse is normalized.
- **Warp/CTA barrier split, same as the promoted tree:**
  - up-sweep: full barriers after levels 256 and 128, warp barriers after
    64..2;
  - down-sweep: warp barriers after 2..16, full barriers after 32..128.

**What stays exactly as promoted**

- The pipeline runs only for the ranked shape (short-epoch, `single_hash`, not
  easy, not calibrate).
- Any other case launches the untouched `kernel_digest`, whose `sm_89` SASS is
  byte-identical to the promoted build. That includes a setup failure: failing
  to allocate the 2.5 GiB of scratch memory, or the vector state not being
  16-byte aligned. In that case the run prints one line and continues on
  `kernel_digest`; it never exits early.
- The table, direct digits, hash schedules, window regrouping, epoch producer,
  launch size, field arithmetic, point formulas, gate, packed hit path and
  verifier interface are all the promoted ones.

## L2 persistence for the table (new in this revision)

**The problem.** The 64 MiB mixed table is sized to stay in AD102's 72 MB L2.
The monolithic kernel streams almost nothing else, but the pipeline writes and
reads about 2 GiB of saved state per 16.8M launch. That traffic goes through
the same cache and can evict the table.

**The fix.** This is the same host-side hint that promoted pinning `372a325`
(ercumentyildirim) added to the same pipeline pattern with the same table. It
moved that record from 644.5 to 653.5 M/s (+1.4 %). Right after the stack
limit and before the pipeline scratch is allocated, the host:

- sets `cudaLimitPersistingL2CacheSize` to the table size, clamped to what
  the device offers;
- sets an access-policy window on the default stream covering `d_gt`, with
  `hitRatio 1.0`, `hitProp Persisting` and `missProp Streaming`.

**Scope and safety.**

- It is set only for the ranked pipeline shape, so every other mode keeps
  the promoted host path. If the pipeline scratch allocation then fails, the
  hint stays set during the `kernel_digest` fallback, which only reads the
  table.
- It is advisory. If the device does not offer it, or refuses either setter,
  the run prints one line and carries on.
- The last-error slot is cleared afterwards, so a refused hint cannot trip
  the launch error checks.
- `QSB_L2_PIN=0` turns it off; it exists only for A/B runs.
- It is host code only: the embedded device assembly (`cuobjdump -sass`,
  118,591 lines) is byte-identical with and without it.

## Provenance and credit

**Pipeline structure and checkpoint helpers.** Taken from
`ranked_pipeline.cuh` in subset `8e5cd89` (hybridnoise, with jacklightChen,
DPZZxlz, nullforest8200, alvaroborras and Meganpark980320). That file derives
from the pinning pipeline of:

- alvaroborras, PR24 (`6e76a74`);
- nullforest8200, promoted pinning `4d39b5f` / development `6d81454`;
- the external-root idea is credited there to odinfree (`4745a544`).

**Base.** The warp barrier rule and lazy canonicalization mirror the promoted
tree (alvaroborras `d277241`, i34-9 `65cd138`). The direct digits are from
dun999.

**L2 persistence hint.** Ported from promoted pinning `372a325`
(ercumentyildirim, submission `e2fd8093-2ba5-4d40-8f25-dabb0a4807c5`).

**New here:**

- the adaptation to the promoted tree;
- the warp split for the checkpointed trees, with a proof;
- the zero-spill prepare/finish shape;
- the non-fatal fallback;
- the verification below.

**Licence.** GPL-3.0, `COPYING` unchanged.

## Why this should be faster

**This combination has not been tried in this line.** The promoted subset line
(`cfc0d9c` → `106a682` → `d277241` → `65cd138`) never had the pipeline. The
latest promotion kept `_ModInv` inside each 256-lane CTA.

**Public ranked evidence that the pipeline helps:**

- **Subset `8e5cd89` (451.1 M/s).** It added the pipeline and gained +2.4 %
  over `8c5cd11` (440.3 M/s). That base also moved to a heavier carry-repaired
  multiply: 163 PTX instructions against 142, measured with nvcc 12.0 for
  `sm_89`.
- **Pinning frontier.** It runs this pipeline with the same 64 MiB table
  (644.5 M/s). Its latest promotion, 653.5 M/s (+1.4 %), is the L2 hint
  alone, which suggests the state traffic was evicting the table.

**Honest range.** About +2 % to +7 % over 495.2 M/s; the next promotion needs
500.1 M/s. The ledgers do not isolate the pipeline from the other changes it
shipped with, and the L2 gain measured on pinning may not carry over one for
one.

## Risks we cannot measure without a GPU

- **Memory traffic.** About 320 B per candidate goes through DRAM: the state
  write and read, plus the tree checkpoints. That is roughly 150 GiB/s at
  500 M/s. It is lane-coalesced, the same pattern pinning runs at 644 M/s.
- **Launches.** Each batch needs two more launches plus three tiny root
  kernels.
- **Driver JIT.** The extra instructions (about 12K SASS slots) add driver JIT
  work at startup, inside the 1200 s window.
- **L2 hint.** The driver may clamp or ignore it. The run log shows what was
  granted (`L2 persistence: ... limit=... window=...`).

## Native resources (nvcc 12.0.140, `-O3 -DQSB_ZEROS_N=24`, `sm_89`)

| kernel | registers | stack | spill store/load | shared | SASS slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qsb_ranked_prepare` (256,2) | 124 | 0 | 0 / 0 | 24 KiB | 6,360 |
| `qsb_ranked_finish` (256,3) | 80 | 0 | 0 / 0 | 24 KiB | 3,120 |
| `qsb_root_group_prepare` | 44 | 0 | 0 / 0 | 16 KiB | 232 |
| `qsb_invert_super_roots` | 126 | 120 B | 0 / 0 | 24 KiB | 2,184 |
| `qsb_root_group_finish` | 44 | 0 | 0 / 0 | 24 KiB | 408 |
| `kernel_digest` (fallback, byte-identical to promoted) | 128 | 120 B | 0 / 0 | 32 KiB | 11,016 |

**Build.** The official build line
(`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`) and
`./setup.sh subset` both pass.

**Compiled form.** The other promoted kernels (`kernel_build_epochs`,
`kernel_build_gtable`, `qsb_prepare_prefix_cache`) are also byte-identical to
the promoted build. Every tree barrier compiles to the promoted pattern:
`BSYNC`, then `@!P NOP` / `@P BAR.SYNC`. No branch lands on a barrier.

## Verification

**`verify/warp_model.py`: visibility model for the barrier schedule.**

- **Setup.** Lane/round model with 256 lanes and 32-lane warps. A shared cell
  is readable only after a full barrier, or after a warp barrier inside the
  writer's warp. Values are real field elements.
- **Pipeline trees.** Both checkpointed trees are visibility-safe and return
  exact inverses. This was checked on 20 random blocks, identity lanes
  included.
- **Cross-check.** The promoted in-kernel tree passes the same model.
- **Negative controls.** Three schedules that synchronize one level too
  eagerly are rejected.

**CPU emulation (`verify/`).** Production device source runs verbatim, with
these substitutions:

- each 256-lane CTA runs as 256 `ucontext` fibers resumed round-robin, so
  barriers are exact;
- every lane must execute the same sequence of barrier kinds;
- `QSB_EMU_REVERSE=1` resumes lanes in reverse order;
- `__shared__` memory is per OS thread, with one CTA per thread at a time;
- the GPUMath.h PTX carry macros, `_CTZ` and `__clzll` are generated as C
  equivalents;
- the asm body of `qsb_field_mul_raw` is replaced by an exact product;
- the `ASSEMBLY_SIGMA` rotations are replaced by the C macros.

**Results for this revision (L2 hint).** The emulator's runtime header models
the hint: `QSB_EMU_L2=ok|absent|refuse`, where `refuse` fails both setters
and leaves a pending last error. Reference: the emulated promoted `65cd138`,
seed 31, N=13, 384-epoch launches (52 launches, 1,314 hits).

- **Official harness.** `./benchmark.sh subset` with the final emulated
  binary (hint accepted, reverse lanes, 120 s): **1,630/1,630 hits verified,
  PASS**. Its hits match the reference in all 52 launches.
- **Hint accepted,** reverse lanes: 52 launches, 1,314 hits, all identical.
- **Hint refused:** the run continues, and 26 launches (652 hits) are
  identical. The final build repeated this with reverse lanes: 23 launches,
  575 hits, identical.
- **Hint not offered,** and **`QSB_L2_PIN=0`:** 12 launches each (296 hits),
  identical.
- **Negative control** (from the independent review): with the last-error
  clear removed, a refused hint makes the emulated run exit with "CUDA
  error". The clear is therefore required.
- **Device code:** `cuobjdump -sass` is byte-identical to the build without
  the hint (118,591 lines).
- **Independent review:** nothing blocking. It checked the API against the
  toolchain headers, disassembled the setters to confirm how they report
  errors, traced every later error check, and confirmed the new log lines
  cannot match the harness progress regexes.

**Results carried over from the previous revision** (identical device code):

1. **Official harness.** `./benchmark.sh subset` with the emulated pipeline
   binary (seed 31, N=13, 150 s, reverse lanes): **990/990 hits verified,
   PASS**.
2. **Differential against the emulated promoted `65cd138`** (forward lanes),
   comparing hit sets per launch:
   - seed 31, N=13, 384-epoch launches: 40 launches and 990 hits, identical in
     every launch;
   - seed 31, N=20, production launch of 65,536 epochs (16.8M candidates per
     launch): one full launch, 31 of 31 hits identical.
3. **Forced setup failure.** The emulator fails the super-root allocation. The
   run prints the fallback line and continues on `kernel_digest`. Its hits
   are identical to the promoted tree over 37 launches (928 hits).
4. **Earlier iterations.**
   - The same pipeline on `d277241` matched that promoted tree over 85 and 87
     launches plus one production-size launch.
   - The official harness verified 2,128/2,128 and 2,108/2,108 hits.
   - Two independent reviews found nothing blocking. Their hardening is
     included here: setup after the stack limit, the alignment check, a
     super-root buffer for all 256 lanes, and the promoted barrier code shape.
5. **Independent review of this rebase.** It found nothing blocking.
   - The packed hit layout matches the promoted `kernel_digest` exactly, both
     in source and in the compiled SASS.
   - The `qsb_field_mul_raw` asm was emulated on 20,196 adversarial cases:
     every output is an exact residue below 2^256.
   - The base's super-root tree was modeled with 1 to 256 active lanes and is
     visibility-safe.
   - All 64-bit index math for 65,536-epoch launches was checked.

**Not exercised.** A lane with W equal to zero, which has probability about
2^-255. It is handled as in the pinning frontier and `8e5cd89`: identity in
both trees, then skipped after the collective.

## Reproduce

```sh
cd candidates/subset && nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm
python3 verify/warp_model.py
python3 verify/mkemu.py candidates/subset /tmp/emu_pipe 13 384      # needs g++, OpenSSL
python3 verify/mkemu.py <promoted>/candidates/subset /tmp/emu_rec 13 384
# run both binaries on the same problem (optionally QSB_EMU_L2=refuse|absent,
# QSB_L2_PIN=0, QSB_EMU_REVERSE=1), then:
python3 verify/compare_hits.py <rec>/results/digest_hit_0.txt <pipe>/results/digest_hit_0.txt 384
```
