# GLV10 and coverage audits

These tests use the real candidate headers. They do not submit results, change runtime sources, or contain problem-specific hits. Local development used macOS without an NVIDIA GPU: CPU checks passed; CUDA checks were compiled only. A successful compile does not establish GPU correctness or speed.

From the repository root:

```sh
python3 candidates/subset/tests/glv10/check_contract.py
python3 candidates/subset/tests/glv10/check_coverage.py
python3 candidates/subset/tests/glv10/gpu_audit.py --docker qsb-cuda
```

The first two commands need Python3 and a C++11 compiler (`CXX` defaults to `c++`). They create temporary files and print JSON. The CUDA command uses an existing CUDA12.8 container with the repository mounted at `/work`; override that path with `--mount-root`. Without `--docker`, it uses local `nvcc`. Compile-only is the default. Build outputs live temporarily under repository `.scratch` and are removed automatically, including on failure.

To **execute** on an NVIDIA GPU with sufficient memory, explicitly supply the existing binary problem input produced by the benchmark wrapper:

```sh
python3 candidates/subset/tests/glv10/gpu_audit.py --run path/to/digest_params.bin
```

With `--docker`, the input must be within the mounted repository and the existing container must have GPU access. No container or GPU resource is provisioned by these scripts.

## CPU coverage

`check_contract.py` compiles the actual `glv10_geometry.h`; checks315460 signed codes with bit/carry/range boundaries; independently minimizes table size using integer dynamic programming; verifies exact rational residual/prefix/lattice bounds; and compares58 complete affine chains and116 signed recovery outputs. All nine P/Q sign/zero classes are included. It does not execute the PTX scalar splitter.

`check_coverage.py` compiles the actual family selector and host schedule preparation function with memory-copy CUDA mocks. It checks512 ordered triples, five invalid inputs, disjoint A/B families,1152 lane schedule outputs across three seed patterns and A->B->A replacement, and296 continuation/tail models using the actual128-window paired-block tag formula. It checks host data and enumeration, not CUDA stream synchronization. The source review must still establish drain/publication ordering.

## CUDA coverage

`audit.cu` includes the actual `tree.cu` under a renamed main. `gpu_audit.py` generates two temporary adapters from the current exact-replay bodies. Only SHA generation is replaced by an explicit scalar argument; the point recovery and gate bodies remain unchanged. This seam tests replay arithmetic and recids, not SHA-to-candidate mapping or kernel nomination plumbing.

The harness exercises the real `q9_glv_split`, recoder, GLV chain and front on all87 production audit scalars, including12 forced high-product fallback boundaries. Split outputs are checked against independently computed integer expectations. Host OpenSSL normalizes actual GPU XYZZ output after checking both denominator residues, so zero is never inverted. Two additional scalars force the singular finishes by using synthetic C=17A with the current instance A and its existing table. No additional GLV table is built for that synthetic C. Exact replay keeps its separate old table. Zero/raw-p nomination is checked directly.

`builder_audit.cuh` compares ordinary and batch GPU builders with OpenSSL over1335 records: every index0–256 in all five segments, low/high ladder boundaries, and each segment's final two entries. Its final CTA has55 live and201 padded lanes. Small comparison buffers are used; no full-table D2H copy is needed for this builder audit.

Runtime execution builds one8.27GiB GLV table plus the old exact table. It needs an NVIDIA GPU and sufficient memory. Every mismatch fails the harness; chain, replay and zero failures are reported separately. Inherited speculative carry approximations can cause chain differences, which require diagnosis rather than silent tolerance. The production startup audit separately tests the actual instance before its search loop.
