# Subset: our `94ba3929` tree (GLV11 P18 + `QSB_Y_PAIR` + `QSB_Q_MIX` 2, v3 producers, 9-window co-grinder, 24 MiB L2 cap) with ercumentyildirim's run-time 16-lane AVX-512 SHA-256 (`3ff68d21`), the record's completed-slot host loop (`521075fe`), shift-form round Sigmas for the 16-lane key hashes (Zen 4 pipe balance, new here) and 80 of the co-grinder's 158 window patterns (whole block-0 groups, new here)

Prepared with Claude Opus 5.5 in Claude Code. This host has no GPU and no AVX-512. Everything below that says "measured" was measured by the ranked runner or by the authors named. What I did myself is the composition, a static Zen 4 pipe analysis, builds, byte comparisons and an analysis of public ranked artifacts.

## Summary

Four host-side changes on top of our queued `94ba3929`. The device code and the native sm_89 image are unchanged: the cubin is byte-identical to `bf001729`'s, `57065b7d`'s and `94ba3929`'s (`f74548427859ec03…`).

1. **16-lane AVX-512 SHA-256 in the co-grinder (ercumentyildirim's `3ff68d21`, S16), unchanged.** Key hashes, epoch tails and the second SHA-256 can run 16 lanes wide. A start-up ABBA timing test keeps the 16-lane path only if it is at least 2% faster than the 4-lane SHA-NI path on the host. Under Intel SDE, `3ff68d21` measured 2,385 → 2,332 instructions per candidate with identical hit sets.
2. **Shift-form round Sigmas for the 16-lane key hashes (`QSB_CPU_S16_SHR`, default 1, new here).** See the next section.
3. **The co-grinder grinds 80 of its 158 window patterns (`QSB_CPU_PAT_MINGRP`, default 5, new here).** See "80 patterns" below.
4. **The record's completed-slot snapshot (`521075fe`).** A completed slot's hit data is copied out before its replacement readback is queued, and the unchanged exact host gate runs after the next GPU work is launched. The ranked regression below reads it as neutral (+0.2 ± 3.0 M/s of GPU part). It is included so this tree carries everything the record carries.

## Shift-form round Sigmas: why

Zen 4 issues `vpmadd52*q` zmm (the co-grinder's IFMA field arithmetic), `sha256rnds2` (SHA-NI) and `vprord`/`vprold` zmm (rotates) on the same two FP pipes, FP0/FP1, at one per cycle (uops.info). Immediate shifts (`vpsrld`/`vpslld` zmm) issue on FP2/FP3, and adds and `vpternlogd` on any of the four. I compiled the co-grinder with gcc 11.4 `-O3` (the ranked host's compiler) and counted the hot loops per candidate at 9 windows in pipe-pair cycles:

| pipe group | cycles per candidate |
|---|---:|
| FP0/FP1 (IFMA 425, SHA-NI 290) | 715 |
| FP2/FP3 | 93 |
| permutes | 77 |
| any pipe | 371 |
| total | 1,255 |

The ideal-scheduler bound is 715 cycles and the even-spread bound 939. The observed rate is about 1,040 cycles per candidate (4.1 GHz × 16 cores / 63 M/s). FP0/FP1 is the binding pair.

S16 computes each round's Σ0/Σ1 with three `vprord`s, all on FP0/FP1, the same pipes as IFMA and SHA-NI. In the model, the unmodified 16-lane key hashes save only 0.7–0.8% of cycles, so they would probably fail the 2% start-up gate. `ror(x,a) ⊕ ror(x,b) ⊕ ror(x,c)` equals the XOR of six shifts, because the two halves of a rotate cover disjoint bits (OR equals XOR). With two `vpternlogd 0x96` and one XOR, the rotates move to FP2/FP3. The expression is written with XOR because gcc turns an OR of shift pairs back into `vprord`, which I checked in the assembly. The FP0/FP1 cost of one 16-key hash drops from 562 to 178 pipe cycles. The model puts the saving at 4.1% (even spread) to 6.3% (ideal) of co-grinder cycles, about +4% co-grinder rate after non-pipe overhead. The message schedule keeps `vprord`: converting those rotates too helps in one model and hurts in the other.

Safety: an always-on start-up self-check (`s16_key_selfcheck`) compares the 16-lane key hash against the 4-lane SHA-NI key hash on 4,096 pseudo-random keys, some with leading zero bytes. It turns the 16-lane path off on any mismatch, before the ABBA test. I also checked the identity on 100,000 random words for both Sigma rotation sets (6, 11, 25 and 2, 13, 22).

## 80 patterns: why

Within an epoch, a candidate's SHA-256 work after the epoch state is block 0 (the epoch's 8 buffered bytes plus the first 56 bytes of kept window pushes) and five later blocks. `hash_plan` already computes block 0 once per group of patterns with the same block-0 bytes. The C(13,3) = 286 window patterns split by block 0 into groups of 35, 6 × 15, 21 × 5 and 56 × 1. The GPU takes the 35-group, the six 15-groups and 3 members of one 5-group (the 128 GPU patterns; see `tree.cu`). The co-grinder's 158 are therefore 20 × 5 + 1 × 2 + 56 × 1: 77 block-0 compressions per epoch for 158 candidates, 0.49 per candidate.

This package keeps whole block-0 groups of at least 5 patterns, in the largest prefix whose total is a multiple of 16: the first 16 of the twenty 5-groups, 80 patterns. Block 0 then costs 16/80 = 0.2 compressions per candidate. The 4-lane SHA-NI groups (20) and the 16-lane chunks (5) carry no padding lanes, whichever hashing mode the start-up test picks. Per candidate that is about 8.2 SHA-256 compressions instead of 8.5. The per-epoch prefix re-hash is spread over 80 candidates instead of 158, and the walk covers about 1.98× more epochs: about 9.5e8 of the 8.2e9 epochs in 1,200 s, starting from the diagnostic-coded base. The model net is about +1% co-grinder rate.

Checks: a Python replay of `hash_plan`'s grouping on the problem's own pushes reproduces the 20 × 5 / 1 × 2 / 56 × 1 split. It confirms that the selection keeps 80 patterns in 16 groups, all disjoint from the GPU's 128. The kept patterns are a subset of the complement of the GPU's, so no candidate is ground twice. Every CPU hit still passes `qsb_hv_check`. `QSB_CPU_PAT_MINGRP=1` (or `QSB_CPU_ALLPAT` in the environment) keeps all 158. The start line prints `CPU co-grind: 80 of 158 window patterns kept (…)`.

## Starting point

The promoted record is `521075fe` (RealAdii), 700,953,730/s, kshitij-hash's `8f99a3e9` tree with the completed-slot host loop. This package replaces the whole editable tree. Against the record, it keeps the record's `QSB_Y_PAIR` device files and host loop, and adds `QSB_Q_MIX` 2, the heavy co-grinder (9-window table, host-core placement under blocking waits) with S16 and the shift-form Sigmas, terrapinelf's v3 producers, and the 24 MiB L2 cap.

## What the ranked runs say (public artifacts)

Every ranked subset run uploads its verified hits. The GPU grinds 128 of the C(13,3) = 286 window patterns and the co-grinder the other 158, so each score splits exactly into a GPU part and a CPU part.

| submission | what it carries | GPU part | CPU part | total |
|---|---|---:|---:|---:|
| `4da17ebc` (i34-9) | record image (`QSB_Q_MIX` 4) + our host files | 639.38 | 61.83 | 701.22 |
| `521075fe` (promoted) | `8f99a3e9` + completed-slot loop | 642.62 | 58.33 | 700.95 |
| `8c3822c4` | copy of our `bf001729` | 637.63 | 60.70 | 698.33 |
| our `bf001729` | this device side, `bfe57794` host side | 637.61 | 60.16 | 697.77 |
| `cd33f1b6` | `a141df2b` + 24 MiB L2 cap | 635.60 | 61.89 | 697.49 |
| `fc2f35f2` | copy of `a141df2b` | 632.42 | 63.69 | 696.11 |
| our `57065b7d` | this tree without the L2 cap, S16, snapshot loop, Sigmas | 629.61 | **63.15** | 692.76 |
| our `94ba3929` | this tree without S16, snapshot loop, Sigmas, 80 patterns | 635.11 | 62.74 | 697.84 |
| `3ff68d21` (ercumentyildirim) | record image + our host files + S16 (rotate-form key hashes) | 634.19 | 61.36 | 695.55 |

A regression of the GPU part of 40 P18 draws on code features, with the mean GPU part of the two previous and two next runs as a control for the card's phase: phase coefficient 0.94 ± 0.20; blocking waits +2.2 ± 1.9 M/s; L2 cap +1.6 ± 3.5; `QSB_Y_PAIR` +1.3 ± 2.9; completed-slot loop +0.2 ± 3.0; `QSB_Q_MIX` 2 +0.0 ± 2.4; −0.28 ± 0.18 M/s of GPU part per +1 M/s of co-grinder. The card's phase moves the GPU part by about ±6 M/s and dominates single draws. S16 on its own drew a 61.36 co-grinder part in its first ranked run, no higher than the same host side without it (62.2–63.7), which fits the pipe model: with rotate-form Sigmas the 16-lane key hashes save under 1% of cycles and probably fail the 2% start-up gate. The shift-form Sigmas in this package target exactly that. The host side is the one lever that moves reliably: the v3 producers lifted the co-grinder part from 60.2 to 62–63.7 M/s. This package works on the co-grinder's pipe balance.

## What changes against `94ba3929`

| file | source | change |
|---|---|---|
| `CpuGrindSubset.h` | `3ff68d21` (S16), plus the Sigma patch | S16 16-lane hashing with its start-up ABBA selection; **new here:** `QSB_CPU_S16_SHR`, `S16SIGS`/`S16_ROUNDS`, `s16_full_s` for the key hashes, `s16_key_selfcheck` before the ABBA test; `QSB_CPU_PAT_MINGRP` and the pattern selection after `hash_plan` in `start()` |
| `tests/gpu_epochs/tree.cu` | `521075fe` | completed-slot snapshot in the two-slot host loop (`sp_collect` / `sp_publish`) |
| `qsb_carrier_sm89.h` | rebuilt here | same cubin; only the source-hash comment changes |
| `SOURCE-MANIFEST.json`, this note | here | regenerated |

Our `CpuGrindSubset.h` before the Sigma patch is byte-identical to `3ff68d21`'s. The host-loop code is byte-identical to `521075fe`'s. Our `tree.cu` differs from `521075fe`'s only in `QSB_Q_MIX` 2, the L2 cap and the final `[HP]` line.

## Build and checks done here

All builds ran in an amd64 container with CUDA 12.8.93, the ranked runner's toolkit.

| check | result |
|---|---|
| `build_carrier.sh 24` | cubin sha256 `f74548427859ec03273f05e9151c810716c6475596e0213a0db3915e468aa5dc`, 462,496 bytes, unchanged; 0 spills |
| the harness's build line (`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`) | builds with no errors; with `-Wall`, the only new diagnostic is nvcc's front end not recognising `#pragma GCC unroll`, as elsewhere in the file |
| image `qsb_carrier_knobs` against host `QSB_CARRIER_KNOBS` | identical, so the native image loads |
| Sigma identity | shift form equals rotate form on 100,000 random words, both rotation sets |

## Execution check under emulation (AVX-512 IFMA + SHA-NI)

This host has no AVX-512, so the new co-grinder code was run under Intel SDE (`sde64 -spr`) inside a full x86_64 Linux guest (`qemu-system-x86_64 -accel tcg -cpu max`, Ubuntu 22.04). Intel SDE does not run under Rosetta, and QEMU user mode has SHA-NI but no AVX-512. A CPU-only harness copies tree.cu's loader, unrank helper and 128-pattern rule verbatim and calls `qcpu::start` in the header's own dev mode (`QSB_CPU_DEVBENCH`, `QSB_CPU_DEVCAND`) at `QSB_ZEROS_N=12`, on the seed-0 problem with a 15-window table. A separate brute-force oracle runs `qsb_hv_check` over every (epoch, pattern) pair.

| run | patterns | epochs | hits vs oracle |
|---|---:|---:|---|
| `3ff68d21`'s header (base), 16-lane modes and calibrated | 158 | 1,024 | 72 = 72, exact |
| this header, `QSB_CPU_PAT_MINGRP=1` | 158 | 1,024 | 72, byte-identical to base |
| this header, default (80 patterns), modes 3 / 0 / calibrated | 80 | 2,048 | 88 = 88 each, exact |
| this header, `QSB_CPU_S16_SHR=0` | 80 | 1,024 | 44, exact |
| QEMU user mode (scalar EC + SHA-NI): base / all patterns / default | 158 / 158 / 80 | 2,048 | 150 / 150 / 88, exact |

- The start line prints `80 of 158 window patterns kept (block-0 groups of >= 5: 16 groups, was 77)`, and the hash plan goes from 77 groups / 10 chunks to 16 groups / 5 chunks. Every hit lies in the kept set, which an independent Python replay reproduces.
- The 16-lane key hash with shift-form Sigmas matches SHA-NI on 4,800 keys (0 mismatches). A deliberately broken copy (one Sigma rotation 25 → 24) gives 4,800 mismatches; the start-up check prints `16-lane key hash self-check FAILED; 4-lane SHA-NI only`, and its hits stay exact.
- No crash, no duplicate, no extra hit in any run.
- Limits: built with g++ 11.4, not through nvcc's host pass; 15-window table only; SDE runs at about 260 candidates/s, so it checks correctness, not the Zen 4 speed or the ranked calibration choice.

## Exactness

- The shift-form Sigmas compute the same SHA-256 values. The start-up self-check disables the 16-lane key hash on any mismatch, and every CPU hit still passes the exact OpenSSL gate `qsb_hv_check` before it is written, so a fault can only lose hits.
- S16 is `3ff68d21`'s code, checked there under SDE with identical hit sets. It runs only if AVX-512F is present and the start-up timing selects it.
- The completed-slot loop changes when the host gate runs, not what it checks; every GPU tentative is still re-derived exactly.
- Device side: unchanged from `94ba3929` (`QSB_Y_PAIR` files byte-identical to the record's; `qsb_s3_selfcheck` replays both Q layouts at start-up).

## Expected result and caveats

- Expected: GPU part about 630 plus the card's phase (±6), co-grinder part about 63–67 (16-lane key hashes if selected, plus the 80-pattern block-0 saving), for a total of about 693–703 in an average phase. The promotion bar is 707.96. This is a pipe-model estimate, not a measurement: nothing here ran on AVX-512 hardware.
- The ABBA gate may keep SHA-NI. The package then equals `94ba3929` plus the neutral host loop, at a cost of about 18 s of mixed-mode calibration (about −0.02 M/s).
- If `sha256rnds2` occupies only half an FP0/FP1 pair-cycle on Zen 4, the Sigma gain roughly halves.

## Reproducing

```
./setup.sh subset
./benchmark.sh subset
```

To regenerate the native image after any device-code edit: `NVCC=/path/to/cuda-12.8/bin/nvcc ./build_carrier.sh 24` in `candidates/subset`. To get the GPU/CPU split of any ranked run: download its `benchmark-diagnostics-subset-<run>` artifact, count hits per `skip[6:9]`, take the 128 most frequent patterns as the GPU's, and compute rate = hits × 2^23 / `elapsed_s`.

Kill switches: `-DQSB_CPU_PAT_MINGRP=1` (all 158 CPU patterns), `-DQSB_CPU_S16_SHR=0` (rotate-form S16 key hashes), `QSB_CPU_S16=0` in the environment (SHA-NI only, dev override), `-DQSB_TABLE_L2_WINDOW_MIB=0`, `-DQSB_Y_PAIR=0` (with `QSB_SC_PARK=0`), `-DQSB_Q_MIX=4`, `-DQSB_CPU_TRY9=0`, `-DQSB_HOST_BLOCKING=0`, `-DQSB_CPU_GRIND=0`. Any device knob change needs `build_carrier.sh` again.

## Base and attribution

- **ercumentyildirim** (co-author): the 16-lane AVX-512 SHA-256 with its start-up selection (`3ff68d21`), the `bfe57794` / `a141df2b` host side, and the ranked GPU/CPU split method.
- **AvinashNayak27** (co-author): the isolated 24 MiB persisting-L2 cap (`cd33f1b6`).
- **terrapinelf** (co-author): the v3 producer code, the co-grinder lane (`2d1631b0`), the host-built producers (`82d8493f`) and the `QSB_Q_MIX` 2 setting (`92a51c8c`).
- **kshitij-hash** (co-author): `QSB_Y_PAIR` and `QSB_SC_PARK` (`8f99a3e9`, in the promoted record) and the `d052bc3d` composition.
- **HyeokxC** (co-author): blocking GPU waits with the host core given to the co-grinder (`0735233a`, `888f5fce`).
- **RealAdii**: the promoted `521075fe` record and its completed-slot host loop.
- **fkiene**: the GLV11 P18 chain and the per-warp Q-layout mix (`413f83e7`), and the first persisting-L2 cap (`8009bfb9`).
- **Through the base:** Meganpark980320, newjordan, i34-9, Ryun1, Akashneelesh and every contributor their notes credit. libsecp256k1 (MIT, `COPYING-secp256k1`).
- **Mine:** the Zen 4 pipe analysis, the shift-form Sigma change, the 80-pattern selection, the SDE-in-QEMU execution check against a brute-force oracle, the ranked split and regression analysis, the composition, the image rebuild and the byte checks.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
