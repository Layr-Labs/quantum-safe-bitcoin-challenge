# FINDING 4 — a device-code change in this tree cannot be shipped from a host without CUDA

This is the most important result of the session. It is a **hard** block, not a
soft caveat, and it is not recorded anywhere in `DEAD-ENDS.md`,
`NEXT-OPTIMIZATIONS.md`, or `ITERATIONS.md`.

## The mechanism, read out of the source

`candidates/pinning/pinning.cu` is compiled by the ranked line

    nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm

with **no `-arch`**, so it produces compute_52 PTX + SASS. On an RTX 4090 that
SASS cannot run and the driver JIT-compiles the PTX. That image is not what runs.

`QsbCarrier.h` loads a **second, prebuilt image of the same source**, compiled
offline for sm_89 and embedded as base64 in `qsb_carrier_sm89.h`:

    qsb_carrier_sm89.h:3   cubin sha256 913a97b2a8e6354e7632a2f41a997a05bf0e61e5a8b9a5c5c793f931e983b463;
                           476832 bytes; prepare kernel LTC64B loads: 5

`QSB_CARRIER` defaults to 1 (`QsbCarrier.h:29-31`) and `QSB_NOJIT` defaults to 1
(`QsbCarrier.h:40-42`); neither is overridden anywhere in `pinning.cu`. On an
sm_89 device `qsb_carrier_init` succeeds (image decodes, `cudaLibraryLoadData`
succeeds, all required kernels resolve, `qsb_carrier_zeros == QSB_ZEROS_N`) and
sets `g_qsb_carrier.on = 1`, `nojit = 1`.

Every hot-path launch site is a two-armed `if`:

    pinning.cu:4104-4111   if (FAST_TAIL && qsb_carrier_has(QK_S0))
                              qsb_carrier_launch(kernel_pinning_pipeline<FAST_TAIL,0>, QK_S0, ...)
                          else kernel_pinning_pipeline<FAST_TAIL,0><<<...>>>(...)

    pinning.cu:4139-4165   QK_RGP / QK_ISR / QK_RGF   (root-group prepare/invert/finish)
    pinning.cu:4189-4196   QK_S2
    pinning.cu:4478-4485   QK_S0   (slotted pipeline path)
    pinning.cu:4498-4512   QK_RGP / QK_ISR / QK_RGF
    pinning.cu:4519-4526   QK_S2
    pinning.cu:5598-5602   QK_BUILD (kernel_build_gtable)
    pinning.cu:5662-5666   QK_YOFF  (qsb_table_offset_y)

`QsbCarrier.h:161-177` is explicit about what the C++ kernel pointer is used for:

    /* Launch the carrier image of `kern`. The static kernel pointer only supplies the
     * parameter types: every argument is converted to its declared parameter type before
     * its address goes to cudaLaunchKernel, exactly as a <<<>>> launch would. */

    cudaLaunchKernel((const void *)g_qsb_carrier.k[kid], g, b, argv, 0, st);

So the **executed device code is the frozen cubin**, resolved from the embedded
image by mangled name. The text of `kernel_pinning_pipeline`, of
`cofactor_checkpoint.h`, of `GLVScalar.cuh`, of `GPUMath.h`, of
`PackedRecovery.cuh` — none of it is what runs on the 4090.

`QSB_CARRIER_BUILD` is not a no-op relative to the ranked build. It changes at
least three device behaviours:

- `pinning.cu:799-803` — the first 16 B of each 64 B record becomes
  `ld.global.nc.L2::64B.v2.u64`.
- `pinning.cu:1420-1421` — `QSB_TBL_L2POL_ON` (cold-gather `.L2::evict_...`
  policy) exists only under `QSB_CARRIER_BUILD`.
- `pinning.cu:1479-1480` — the native image gathers the Y half first so it
  carries the 64 B L2 fetch.

## Consequence

| what I edit | what happens |
|---|---|
| device code (`pinning.cu`, `GLVScalar.cuh`, `GPUMath.h`, `PackedRecovery.cuh`, `cofactor_checkpoint.h`, `GPUHash.h`, `LeafRecovery.cuh`, `ParityWindow.cuh`, `NegativeY`-style headers) | **no effect on the hot path.** The build succeeds, the run succeeds, the score is a draw of the frozen 476,832-byte cubin. |
| a kernel **signature** | **silent wrongness.** `static_assert(sizeof...(P) == sizeof...(A))` (`QsbCarrier.h:174`) compares my edited signature against my edited call — it still passes. The frozen kernel is then handed a mismatched `argv`. This is the v22 dropped-`qsb_table_offset_y` failure class, with the argument-count guard defeated. |
| a `__constant__` symbol | fails loudly: `cudaLibraryGetGlobal` by name returns an error (`QsbCarrier.h:193`) or `n > sz` (`:195`). This one class is safe. |
| host code (launch geometry, streams, batching, CPU co-grind controller) | works, but see note 02 — that knob space is already measured out. |

Regenerating the carrier needs `nvcc -arch=sm_89`, `cuobjdump`, `python3` and a
shell (`build_carrier.sh`). `build_carrier.sh:4` states it outright:
"Rerun after ANY edit to `pinning.cu` or a header it includes."

## Why this matters as a correctness result, not just a convenience

The brief's named trap was: *"A missing kernel launch produces NO compile error.*
… *the spot check that passed sampled the RAW table, so it could not catch a
defect downstream of it. A control that verifies before the consumer does not
cover the consumer."*

The carrier is a strictly larger instance of the same defect. A validation that
proves the *source* correct proves nothing about the *bytes that run*, and
**there is no local check at all that source and carrier agree** — the G3 note
admits it: "a stale image may still load if only the zeros count is unchanged, so
the source/header pairing is part of this submission's reproducibility contract."
The only binding check is `qsb_carrier_zeros == QSB_ZEROS_N`, which is a
SHA-prefix count and would not notice a changed chain loop, a changed digit
width, or a changed gather count.

## Launch-site inventory (the audit the brief asked for)

Per the v22c post-mortem I audited `<<<` against the carrier launch sites:

- 16 raw `<<<>>>` sites in `pinning.cu`: 4076, 4079, 4111, 4122, 4143, 4154,
  4164, 4181, 4196, 4485, 4502, 4507, 4512, 4526, 5602, 5666.
- 14 have a `qsb_carrier_launch` counterpart (each is the `else` arm of the same
  `if (qsb_carrier_has(...))`).
- 2 are compiled out: 4122 under `QSB_TREE_OFFLOAD` (=0, line 259) and 4181 under
  `QSB_TREE_OFFLOAD2` (=0, line 271). `pinning.cu:4066-4068` makes that a hard
  requirement under `QSB_NOJIT`, so the two cannot both be on.

**No launch is currently missing.** The v22 defect is not present in this tree.
That part of the archive is sound.

## The only escape hatch, and why it is a regression

`QSB_CARRIER` can be set to 0 from source, which would make the edited device
code actually execute. It is a measured loss, not a neutral:

- the record load loses `ld.global.nc.L2::64B`, i.e. both 32 B sectors of a
  64 B record stop being fetched as one DRAM access (the G3 note prices the
  hinted form at +0.25% locally, and that is the *only* device difference
  Ryun1's PR #1447 contributed);
- `QSB_TBL_L2POL_ON` disappears;
- the native gather ordering disappears;
- `QSB_NOJIT` no longer protects the compute_52 module, so ~1.1 s of PTX JIT
  returns inside the timed window (`pinning.cu:328`, measured cold-cache).

Trading an unmeasurable device change for those four known losses is not a bet
worth taking blind.
