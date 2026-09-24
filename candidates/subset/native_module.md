# Fixed native sm89 carrier and full-source regeneration

The fixed command compiles a host-only carrier from the original shared setup,
search and publication bodies:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm
```

`subset.cu` selects `QSB_NATIVE_MODULE=1`; `native_role.cuh` gives both nvcc
passes the explicit `QSB_HOST_CARRIER=1` role. Candidate device definitions and
host/device helper device variants are excluded in that role. No source kernel
addresses or device-symbol placeholders are used. `native_abi.cuh` supplies
readable typed tags for seven kernels and thirteen globals. Parameters convert
to their declared types before argument addresses are assembled, including all
forty digest arguments. Both SHA K reads use the selected native module. The
thirteen globals have fourteen host transfer sites: twelve writes and two reads.

On Linux sm89, the carrier reads `native_sm89.cubin` beside `/proc/self/exe`,
checks 933536 bytes and SHA256
`994014fb153b73eb4338b013d373734dc8350041378b2c39ed0199987cc38ae8`,
then resolves the seven kernels and thirteen globals. The payload is the
immutable native-v1 arithmetic baseline, with twenty total device globals.
Carrier-only compile guards reject incompatible host geometry/mapping options
(including T14, windows, digit shift, paired/epoch modes and launch size).
Wrong device architecture, missing/corrupt payload, lookup, bounds, ABI or launch
failure terminates. Unsupported CUDA headers or N fail compilation. There is no
source fallback in the carrier. Environment setup still sets both module loading
variables to LAZY before the first explicit CUDA call, as in the baseline.

All fresh-instance initialization, table construction, allocations, the late
32768-byte stack setter, default stream, timings, search mapping, verification
and publication remain in the original shared host bodies. This preparation
changes neither device arithmetic nor host scheduling policy. Nominal loading
behavior is not a GPU measurement.

The complete original device path remains included. Explicit
`-DQSB_NATIVE_MODULE=0` also selects `QSB_HOST_CARRIER=0`; it is the full-source
regeneration role, not a fallback selected by the carrier. Shared epoch/group
POD layouts are in `native_types.cuh`. Field/SHA arithmetic headers remain
included by the full-source role and are excluded from the carrier role.

With CUDA **12.8.93**, regenerate from any working directory:

```sh
python3 /path/to/candidate/regenerate_native.py --artifacts /path/to/build
# If the same directory tree is mounted at /work in an existing container:
python3 /path/to/candidate/regenerate_native.py --docker qsb-cuda \
  --docker-workdir /work/path/to/candidate --artifacts /path/to/build
```

The generator uses the original `nvcc -O3 -DQSB_ZEROS_N=24
-DQSB_NATIVE_MODULE=0 --ptx subset.cu` frontend (compute_52 default), followed by
`ptxas -arch=sm_89 -O3`. Build products, commands, logs and preprocessor dependency
files stay outside the candidate. An exact cubin mismatch stops before installing
outputs. Successful regeneration updates only the native payload and its two
native manifests. The JSON manifest records PTX/cubin identity, all entry ABIs,
all globals, recursive source union and actual local preprocessor dependencies
for both roles. The generated header is lookup/hash metadata, not a host body.
The public submission note and package source manifest are a separate packaging
step and are not rewritten by this generator.

CPU mocks, compilation and executable inspection cannot establish GPU loading
latency, hit equivalence or throughput. Device execution remains necessary; this
preparation does not authorize packaging or submission.

APIs: [CUDA 12.8 library management](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__LIBRARY.html)
and [NVIDIA dynamic loading example](https://developer.nvidia.com/blog/dynamic-loading-in-the-cuda-runtime/).
