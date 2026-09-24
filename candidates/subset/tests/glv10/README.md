# GLV12 and coverage audits

These tests use the real candidate headers. No problem-specific hits are included. Local validation is on macOS without an NVIDIA GPU: CPU checks execute locally; CUDA validation is compile-only. Compilation does not establish GPU correctness or speed. Historical `glv10` file and kernel names are retained for ABI continuity.

From this candidate directory (the directory containing `subset.cu`):

```sh
python3 tests/glv10/check_contract.py
python3 tests/glv10/check_coverage.py
python3 tests/glv10/check_policy.py
python3 tests/glv10/gpu_audit.py --docker qsb-cuda
python3 tests/native_runtime/check.py --docker qsb-cuda
```

CPU checks require Python 3 and a C++14 compiler (`CXX` defaults to `c++`). The CUDA command uses an existing CUDA 12.8 container with the repository mounted at `/work`; override that path with `--mount-root`. Without `--docker`, it uses local `nvcc`. No resources are provisioned. Build outputs are temporary and removed automatically. The native mock requires regenerated native artifacts matching the current runtime sources.

To execute the CUDA audit on an NVIDIA GPU with sufficient memory, explicitly provide a fresh benchmark input:

```sh
python3 tests/glv10/gpu_audit.py --run path/to/digest_params.bin
```

With `--docker`, the input must be within the mounted repository and that container must have GPU access.

## CPU coverage

`check_contract.py` compiles the literal six-bank geometry, checks 378,552 signed codes at bit/carry/range boundaries, verifies rational residual/lattice bounds, and compares 145 complete affine chains and 290 signed recovery outputs. All 87 production scalars, including six forced coefficient-fallback boundaries per reciprocal, are included. All nine P/Q sign/zero classes are covered. It checks physical bank order `[0,1,2,5,3,4]`, interval boundaries and invalid-record rejection, 22,893,641 records, and the full-table final CTA of 73 live/183 padded lanes. The host test does not execute the PTX scalar splitter.

`check_coverage.py` compiles the actual selector and host schedule with memory-copy CUDA mocks. It checks 512 ordered triples, five invalid inputs, disjoint A/B families, 1,152 lane outputs across three seed patterns and A->B->A replacement, and 296 continuation/tail models. It tests host data and enumeration; stream drain/publication ordering still needs source review.

`check_policy.py` compiles the real stack/cache setup bodies with a fake CUDA runtime. Twenty cases cover the 32 KiB stack request, rounding and readbacks, API failures, cache caps, and the actual 48 MiB stream-0 access-policy window. Both source and native modes use that policy. Mocks do not establish driver acceptance, residency, or cache hits.

## CUDA coverage

`audit.cu` includes the actual `tree.cu` with its main renamed. Two temporary adapters retain the current exact-replay bodies, replacing only SHA generation with an explicit scalar. This tests replay arithmetic and recovery IDs, not SHA-to-candidate mapping or nomination plumbing.

The harness exercises the real `q9_glv_split`, recoder, twelve-term chain and front on all 87 production scalars. Actual split outputs are compared with independent integer expectations. OpenSSL normalizes GPU XYZZ output after checking denominator residues, so zero is never inverted. Two additional scalars force singular finishes using synthetic C=17A and the current instance table. Exact replay retains its separate old 64 MiB table. Zero/raw-p nomination is checked directly.

`builder_audit.cuh` compares ordinary and batch builders with OpenSSL over 1,602 records: indices 0–256 in all six banks, low/high 12-bit boundaries, and each bank's final two entries. Its final CTA has 66 live/190 padded lanes. The scalar oracle independently constructs the bias and odd multiples. Compact comparison buffers avoid a full-table download.

Execution builds one 1,465,193,024-byte GLV12 table plus the old exact table; the two host/device ladders total 3 MiB per side while building. Every mismatch fails. Inherited speculative field carries can cause chain differences requiring diagnosis. The isolated source-kernel audit does not certify native module selection or the ranked process's cache/memory policy. The production startup audits separately check the actual instance before search.
