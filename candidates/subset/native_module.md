# Native sm89 module

The payload contains eleven kernels and twenty globals for N=24, GLV10,
128-window paired search with disjoint A/B families and 64 first-state slots.
Four kernels build, gather and audit the new table; the seven original kernel
entry points remain, including independent exact replay on the old 64 MiB table.
Every fresh problem builds its own tables. No problem data or hits are packaged.

The fixed build remains:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm
```

On Linux sm89, the executable reads `native_sm89.cubin` beside `/proc/self/exe`,
checks its exact size and SHA-256, loads it with the CUDA 12.8 runtime library API,
and resolves all kernels and thirteen host-accessed globals before symbol
initialization. The seven remaining initialized globals stay internal to the
native image and require no host lookup. Kernel
arguments are converted using the declared CUDA kernel signature, then their
widths are checked against the generated PTX ABI. Symbol transfers use checked
native pointers and `cudaMemcpy`; both reads of SHA-256 K use the chosen module.
A successful native selection is permanent. Load, lookup, bounds, ABI and launch
errors terminate with nonzero status; there is no per-kernel fallback.

Before the first explicit CUDA call, eligible builds override both
`CUDA_MODULE_LOADING` and `CUDA_MODULE_DATA_LOADING` to `LAZY`, including if the
caller set `EAGER`. This is intended to leave the unused registered source module
unloaded. Whether this fully avoids its PTX JIT must be verified on a GPU.

A missing payload or non-sm89 device selects the entire source pipeline before
any symbol initialization. N other than 24, CUDA headers older than 12.8, and
`-DQSB_NATIVE_MODULE=0` compile the source pipeline. The explicit killswitch also
leaves the loading environment untouched. Other experimental device compile
flags require disabling native mode or regenerating a matching package; the
shipped payload represents the default device configuration only.

Regenerate with CUDA **12.8.93**, from any working directory:

```sh
python3 candidates/subset/regenerate_native.py
# From the repository on a Mac, with the repository mounted at /work:
python3 candidates/subset/regenerate_native.py --docker qsb-cuda
```

The generator runs `nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_NATIVE_MODULE=0 --ptx
subset.cu`, then `ptxas -arch=sm_89 -O3`. It emits the cubin, a C++ lookup/hash
manifest and a JSON manifest with exact toolchain, flags, PTX/cubin hashes,
kernel parameter widths, global sizes and hashes of recursive source inputs.
Temporary build files are removed. The cubin contains generic executable code;
regeneration requires no NVIDIA device. No driver library link is needed.

Compilation, CPU mock tests and disassembly comparisons cannot establish GPU
loader behavior, lazy loading, hit equivalence or a throughput improvement.
Official device execution remains necessary.

APIs: [CUDA 12.8 library management](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__LIBRARY.html)
and [NVIDIA dynamic loading example](https://developer.nvidia.com/blog/dynamic-loading-in-the-cuda-runtime/).
