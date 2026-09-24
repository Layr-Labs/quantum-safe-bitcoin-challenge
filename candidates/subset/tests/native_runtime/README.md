# Native host-routing checks

Run from any working directory, using the path to this script:

```sh
python3 candidates/subset/tests/native_runtime/check.py --docker qsb-cuda
```

The existing container must have this repository mounted at `/work`, plus g++
and OpenSSL development files. Use `--mount-root` for another mount location.
No container, account, credentials or GPU is created. Omit `--docker` to use
local g++ and OpenSSL; `--cxx` selects the compiler. `--out result.json` saves
machine-readable results.

The script builds a small C++ test against the **actual `native_runtime.cuh`**
and a mocked CUDA API. Kernel parameter declarations are extracted from the
included production source. Every argument byte is checked for all eleven kernel
signatures, including all forty digest arguments, literal-zero pointer conversion,
integer conversion and repeated launches. It verifies all thirteen native symbol
copy round-trips, whole-source fallback selection, and failures for malformed ABI,
copy bounds, repeated/uninitialized selection, launch/load errors, wrong payload
length and wrong payload hash. Twelve process-isolated cases must pass.

Temporary generated files and executable are created beneath this test directory
and removed. The candidate source and payload are never modified. The assertions
in the mock intentionally do not execute CUDA arithmetic or simulate NVIDIA's
loader. Passing establishes host argument/routing behavior, not GPU lookup,
synchronization, hit equivalence, lazy loading or speed. Those need real hardware.
