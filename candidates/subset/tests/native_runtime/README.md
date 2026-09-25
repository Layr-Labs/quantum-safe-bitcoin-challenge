# Native routing checks

The source defaults to `QSB_NATIVE_MODULE=0`. No generated native payload is included in this preparation commit. Enabling native requires generating the exact selected source image first.

CPU-only preparation gate:

```
python3 candidates/subset/tests/native_runtime/check.py --mock-image
```

This compiles the actual routing header against mocked CUDA calls and a synthetic payload in a temporary directory. It is not a CUDA build or GPU correctness test. It checks all six typed launch ABIs on three streams, all sixteen host globals, native/source routing, resource attributes, and failure cases including corrupt payload, old descriptor size, wrong descriptor contents and changed host bank geometry.

After authorization for CUDA compilation, generate the current source image with CUDA12.8.93 using the default compute52 frontend and offline SM89 assembler:

```
python3 candidates/subset/regenerate_native.py
python3 candidates/subset/regenerate_native.py --verify-only
python3 candidates/subset/tests/native_runtime/check.py
```

`--verify-only` is mandatory after any source or generated-file change before a native build/release. It checks recursive source hashes, the selected geometry contract, generated headers and embedded payload. Any source change requires regeneration. Generation checks the emitted eleven-row descriptor initializer against the independent source contract before publishing the payload. Metadata is published last and inconsistent artifacts fail verification.

At startup on SM89, source descriptors and all eight host bank records must match the selected contract; the loaded module's actual descriptor size and bytes must then match the source initializer. Six matching kernel ABIs alone are insufficient. Unsupported devices select the source route before work. Supported-device load, ABI, descriptor, global, attribute or launch errors abort; no mixed-module continuation occurs.

Actual CUDA identity, resource allocation, memory reserve, engagement, hit identity, independent verifier and performance gates remain separate requirements. A passing CPU mock does not establish any of them.
