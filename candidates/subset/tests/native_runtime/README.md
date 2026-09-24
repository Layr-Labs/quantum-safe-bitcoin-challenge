# Native-only carrier host checks

Run from any working directory using the path to these scripts:

```sh
python3 candidates/subset/tests/native_runtime/check.py --docker qsb-cuda
python3 candidates/subset/tests/native_runtime/check_config.py
```

The first command uses an existing container with this repository mounted at `/work`, g++, and OpenSSL development files. Override with `--mount-root`; omit `--docker` for local g++/OpenSSL. `--cxx` selects that compiler and `--out result.json` saves results. The configuration check needs a local C++ preprocessor (`CXX`, default `c++`). No resources are provisioned and no GPU code executes.

`check.py` compiles the actual `native_runtime.cuh`, `native_abi.cuh`, `native_types.cuh`, and role header against a fake CUDA API. It compares all seven kernel tag types with parameter declarations extracted from the included full device source, checks all thirteen global tag types, and verifies POD size/alignment/offsets for epoch/group records and `uint4`. Argument bytes are checked across fourteen typed launches, including the forty-parameter digest, literal-zero/NULL pointers, addresses above 4 GiB, signed conversions and repeated launch calls.

The suite executes the actual host push-word, constant-schedule and window-schedule preparation functions and the literal window selector. Their ten transfers plus the remaining four main-upload bindings exercise all fourteen host transfer routes: twelve writes and both reads of K. It additionally checks roundtrips through all thirteen typed global adapters. This exercises routing; it does not execute the full search main.

Twenty-seven isolated runtime cases cover success; unsupported architectures; missing, short, oversized and corrupt payloads; executable-path failures; repeated or premature use; library/kernel/global lookup failures; global size/null-pointer errors; ABI count/width mismatch; unknown tokens; declared-size/copy bounds; H2D/D2H copy errors; and launch errors. Four rejected compilations cover unsupported CUDA headers, wrong N, mismatched roles and nonboolean role selection. The mock supplies no source-symbol-copy functions or candidate source function handles: carrier fallback references cannot silently link.

`check_config.py` preprocesses the literal carrier guard extracted from `tree.cu`. It accepts the baseline, rejects all eighteen individually incompatible host configuration settings in carrier mode, and confirms that the same guard leaves the explicit full-source regeneration role unrestricted. It does not compile eighteen candidate variants.

Temporary mock files/executables are removed automatically. Runtime headers, payload and manifests are not edited. These are host contracts, not real CUDA library behavior, GPU correctness, synchronization, hit recall, JIT or throughput measurements. A successful fixed-command build must be inspected separately for candidate device images and registration. Empty nvcc PTX/ELF metadata containers and generic fatbin registration may remain; their presence is not a claim of candidate device code or measured startup cost.
