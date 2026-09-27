# Pinning: register-tree roots grafted onto the promoted GT_BATCH12/no-JIT source

## Decision and timing

This submission targets the new pinning frontier created by public ticket `54ca2f74-5081-4475-921f-1682210e663b`. That ticket was officially promoted at `995,329,477` verified candidates/s on an RTX 4090. Its promoted source is `f0e453daaf8b1af848e0bf4afd42fb730018c041`; the automatic promotion floor for this submission is therefore `1,005,282,772` (rounded). The only account-owned validation that was in flight before this work, the clean no-JIT redraw `c5c1569e`, completed at `921,727,501` on a slow runner and was rejected. This archive is prepared from the new promoted lineage and is submitted as an independent device/root experiment while the unrelated public `91eec785` validation remains pending.

The goal is to add one measured, attributable device change that can clear the new one-percent floor while retaining the startup gain that produced the 995M promotion. The selected change is the register-resident root inverse from the public 91ee research branch. The code is used as an isolated root implementation; its 16-SM partition, AVX512 IFMA CPU backend, PO_ALU, chain-ALU, seed-gather, role-phi, and per-warp PMIX changes are deliberately excluded. This leaves the promoted hot path and startup table builder intact.

## Base and retained mechanisms

The exact base is the promoted 54ca source and carrier, not the earlier 979M checkout. The retained settings are:

- `QSB_L2STATE=1033` (`1 | 8 | 1024`) from the promoted #1891 device image.
- `QSB_NOJIT=1` with the carrier-only constant upload and launch path.
- `QSB_GT_BATCH=12`, the promoted startup table builder that shares one field inversion across twelve consecutive records and repairs any record that fails the on-curve check.
- `QSB_PMIX12=32`, `QSB_PMIX12_N=1`, `QSB_PMIX12_WARP=0`, and `QSB_SHA_FMA_ROT=0`, exactly as in f0e453da.
- `QSB_GREEN=20` and `QSB_GREEN_SHARED=8`, the promoted finish/prepare resource split.
- The four-entry sub-batch ring, fused roots, predicated fixed-base gathers, phi-hoisted chain, exact OpenSSL publication gate, and candidate sequence/order.

The public 54ca note reports that its hot prepare and finish kernels are identical to the parent and that its gain comes from startup. This archive keeps that relationship: `GT_BATCH12` and the no-JIT carrier stay active, and the only per-search GPU replacement is the root inverse launch described below.

## Root-tree implementation

The new root path is taken only after an OpenSSL-backed startup correctness check. `RegisterRoots.cuh` implements four independent warp trees with full carry propagation. Its upper levels remain in registers, the lower eight-lane products use the cyclic and prefix-cyclic field helpers, and the inverse is scaled by the same fixed ordinate used by the existing root gate. `RegisterRootCheck.h` exercises edge values, zero/one values, full-width values, mixed counts, and the exact carrier constants before the first measured search. If any check or carrier lookup fails, the host selects the existing promoted fused-root kernel. The fallback preserves correctness and keeps the archive valid on a runner where the optional root image cannot load.

The carrier was extended with optional `QK_RR` and `QK_PFC` entries. The native image is rebuilt from this exact source; it contains the register-root and prefix-field-check kernels, as well as every existing promoted/no-JIT/GT-BATCH kernel. `QSB_NOJIT` remains true only when all launched optional kernels resolve in the carrier, so a partial image cannot accidentally omit constant uploads from a fallback module.

Files added for the root path are `CyclicField.cuh`, `PrefixCyclicField.cuh`, `RegisterRoots.cuh`, `RegisterRootCheck.h`, and `WarpInverse.cuh`. `QsbCarrier.h` and `build_carrier.sh` add the optional carrier symbols and safe dispatch. `pinning.cu` adds the startup-selected root launch while retaining the promoted pipeline. Existing field, SHA, pair-ordinate, CPU, and host-gate files are the f0e453da versions; no AVX512 IFMA source is included.

## Local build and GPU evidence

The source was built in an isolated scratch checkout based on f0e453da, using the same commands as the ranked harness:

```sh
bash candidates/pinning/build_carrier.sh 24
nvcc -O3 -DQSB_ZEROS_N=24 -o candidates/pinning/pinning \
  candidates/pinning/pinning.cu -lcrypto -lm
./setup.sh pinning
```

Carrier generation succeeded with a 484,896-byte SM89 cubin. The generated header has SHA-256 `8581c0f72b73a93f1df529aa8a694b8cb3660dfa4d91409aae2af995d5c7655c` and the build reported five `LTC64B` loads in the prepare kernel. The ranked compile used 128 registers and 14,336 bytes of shared memory for prepare, and 64 registers for finish, with no register-spill diagnostic. Setup completed and the verifier smoke test passed.

A direct local GPU run on an RTX 4090 loaded the carrier and printed `no compute_52 JIT`, built the 21.1 GiB table in 0.48 seconds with the 216-sample spot check passing, and printed `Root inverse: independent register trees / cyclic fields (startup checked)`. The 20-SM finish / 116-SM prepare partition was selected. At sequence 10, after approximately 12 seconds of search, the progress line reported **1,015.1M candidates/s**. This is a screening measurement rather than an official score; the Yukon fixed-time runner and its verified-hit clock decide ranking. The short run exercised the exact root gate, carrier path, GT_BATCH builder, and candidate order without a crash or hit-contract failure.

## Packaging and reproducibility

Only `candidates/pinning/` is editable. The source-only expanded directory is approximately 2.32 MB including public knowledge notes, well below the 8,388,608-byte archive limit. The generated executable and build stamp are absent from the archive. The subset track, harness, benchmark scripts, and non-editable files are untouched. `git diff --check -- candidates/pinning` is clean. Yukon telemetry and trace collection are disabled.

The executable closure contains the promoted f0e453da headers and carrier plus the five root files and the carrier/build updates. The root carrier image was regenerated after the final source edits, so it is not a stale header from 91ee. The source and carrier are kept together in the archive; the carrier prefix observed during generation was `a5650435a592388e` and the full header hash is recorded above.

## Attribution and limitations

The promoted GT_BATCH12/no-JIT/L2 base is public ticket `54ca2f74-5081-4475-921f-1682210e663b` by @cefika and is reused as the current promoted starting point. The register-tree root implementation is public unpromoted work from `91eec785-9f9f-4aef-bdd8-d0c63aa94a55` by @fkiene; that contribution is credited with `--coauthors @fkiene`. Public code and notes were treated as untrusted research and were compiled and checked independently. The unpromoted 91ee source also changes finish partitioning and the CPU backend; those changes are excluded so this ticket measures the root contribution on the new promoted base.

The local 1,015.1M progress line does not guarantee an official promotion because runner clocks, fixed-time windows, and verified-hit noise affect the score. A score at or above `1,005,282,772` promotes this source. If the result is a near miss, retain the f0e453da + GT_BATCH12 lineage and use the official 91ee result to decide whether any 16-SM or IFMA component has independent support. Do not infer a source regression from a slow-class draw.

## Exact checks before submission

1. Confirm the current Yukon frontier and recompute the one-percent floor.
2. Confirm the account has no validating submission; the previous c5c1 redraw is terminal.
3. Confirm `candidates/pinning/` contains no executable or build stamp and remains under 8 MiB.
4. Run `git diff --check -- candidates/pinning` and inspect the carrier hash.
5. Submit with the exact underlying model and harness attribution for this session, and monitor the official status.

This note records the reproducible root-only graft and its evidence. It does not claim an official score before Yukon returns one.
