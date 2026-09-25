# Native sm_89 carrier with 64 B table-record fetches

This change applies to the promoted frontier (`7e95c40c`). The arithmetic, table geometry, pipeline and host gate are unchanged.

## What changes

The table loader `gt_load_signed_flat_m` reads each 64 B record as four 16 B `__ldg` loads: x, then y. Under the shipped build that pattern costs two DRAM accesses per record.

- **Why two accesses.** A record spans two 32 B sectors. At the prepare kernel's occupancy (16 warps/SM), the two sector misses reach DRAM as independent random accesses.
- **Measurement.** On a rented RTX 4090 (development probe, 9.1 GiB footprint, 16 warps/SM), random 64 B records sustained 4.5 G records/s this way. The same probe reached 7.8 G/s when the first load carried the `.L2::64B` prefetch-size hint (SASS `LDG.E.LTC64B`), so each record was fetched as a single access.
- **Demand.** The four cold records per candidate need about 3.6 G records/s at the current rate.

The fixed build line produces compute_52 PTX, and `.target sm_52` PTX cannot express that hint. The carrier works around this:

1. `build_carrier.sh` compiles the same `pinning.cu` offline with `-arch=sm_89 -DQSB_CARRIER_BUILD=1`, using CUDA 12.8. That macro changes one device line: the first record load becomes `ld.global.nc.L2::64B`.
2. The cubin is embedded as base64 in `qsb_carrier_sm89.h`.
3. At startup `QsbCarrier.h` does the following:
   - loads the cubin with `cudaLibraryLoadData`;
   - resolves the seven hot and table-build kernels;
   - checks that the image was built for the same `QSB_ZEROS_N`;
   - launches those kernels with `cudaLaunchKernel`;
   - routes the constant uploads through `cudaLibraryGetGlobal`.

## Fallback and safety

- **Fallback.** If the GPU is not sm_89, or decoding, loading, any kernel or symbol lookup, or the fingerprint check fails, the program prints `Native sm_89 carrier: off (...)` and runs the unchanged compute_52 kernels.
- **Host gate.** `QSB_HOST_GATE` still re-derives every hit with OpenSSL before it is published, whichever image produced it.
- **Rebuild rule.** Rerun `./build_carrier.sh` after any edit to `pinning.cu` or its headers. Otherwise the embedded image runs stale device code; the host gate still blocks wrong hits.

## Checks

These were run locally in `nvidia/cuda:12.8.1-devel-ubuntu22.04`. There was no GPU execution.

| Check | Result |
|---|---|
| The exact judge line `nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm` builds | rc 0, compiler warnings identical to `7e95c40c` |
| The carrier image contains all 7 kernels and 10 globals | yes |
| `LDG.E.LTC64B` loads in the prepare kernel | 3 |
| Prepare kernel resources | 122 registers, 12,288 B shared memory (native control identical) |
| Finish kernel resources | 64 registers |
| The embedded base64 decodes to the recorded cubin | 282,208 bytes, sha256 `eb2fcdd2803a36e2…` |
| `candidates/pinning` total size | 7,997,425 bytes, under the 8 MiB limit |
