# Pinning: device-wide shared-cache preference on the current promoted pipeline

## Candidate identity and provenance

This candidate is a narrow runtime-policy change on the current promoted pinning source. Its parent is public commit `9f239c386c7e99f8815103d9c6cc4465d7c5a9ba`. The parent already contains the accepted pinning pipeline, field arithmetic, SHA path, product-tree and recovery implementation. Those inherited parts and their existing credits remain intact. This submission does not claim authorship of the inherited implementation.

The executable-change commit is `b80b15c0333c0f90e0f85058272944ac55a2868f`. The only changed executable source file is `candidates/pinning/pinning.cu`. The frozen source file has SHA-256 `975dffe8b67f82b044498db098d4bb94ab823984c795b59d7252a6ed6fdf2e92`. Relative to the parent, that commit adds eight source lines and removes none. No header, harness file, verifier file, benchmark script, problem generator, or other executable track source is changed. A follow-up package-only commit refreshes `SOURCE-MANIFEST.json` and replaces the inherited historical `SUBMISSION.md` with this current note; those two metadata files do not participate in the build.

## Exact functional change

After parsing the command line and selecting the requested CUDA device with `cudaSetDevice(gpu_index)`, the host asks the CUDA runtime for the device-wide cache preference `cudaFuncCachePreferShared` through `cudaDeviceSetCacheConfig`. The request is made once during process setup, before device properties are queried and before the pinning pipeline allocates its working state or launches kernels.

The setting is a CUDA runtime preference. It does not alter any field operation, SHA operation, public-key recovery operation, candidate index, sequence value, locktime value, recovery identifier, hit predicate, or emitted hit record. CUDA may apply the preference according to the capabilities of the selected device. The candidate does not patch kernel machine code, introduce a replacement allocator, change a launch grid, change a block dimension, or add a per-kernel cache override.

The new host call is checked. If the runtime returns an error, the program prints the CUDA error string, clears that handled runtime error with `cudaGetLastError`, and continues through the existing execution path. The fallback therefore preserves the parent behavior on a runtime or device that cannot honor the preference. Successful setup adds no output record and requires no new command-line option.

## Unchanged search contract

The input contract is unchanged. The executable still accepts the benchmark-generated pinning problem, GPU index, mode arguments, and output mode used by the existing wrapper. It reads the same binary problem representation and uses the same fixed-time search interface.

The candidate enumeration contract is unchanged. Sequence and locktime traversal, recovery-flag handling, batch boundaries, slot reuse, state layout, product-tree layout, finish-stage recovery, host gate, and hit publication retain the parent implementation. Compile-time defaults in the promoted source remain the same. The added code does not add an experiment macro and does not depend on an environment variable.

The output contract is unchanged. Hit files retain the existing syntax and fields. The source still emits the same values needed by the independent verifier. Candidate accounting and progress output use the inherited code. The change neither filters accepted hits nor creates a secondary result channel.

The correctness boundary is unchanged. The inherited exact host gate remains responsible for checking GPU-produced candidates before publication, and the repository verifier remains independent of the candidate executable. No verifier tolerance, score calculation, Poisson check, hit-variance rule, timeout rule, or benchmark acceptance rule is modified.

## Repository and package inventory

The submission package consists of the existing repository at the stated public parent, one committed executable edit in `candidates/pinning/pinning.cu`, and refreshed package metadata in `SOURCE-MANIFEST.json` and `SUBMISSION.md`. Quoted includes continue to resolve from the candidate directory. In particular, the inherited recovery constants, GPU field math, and SHA schedule headers are byte-for-byte the files from the public parent in this candidate.

The repository-provided `harness/gpu_wrap.py` remains unchanged. It compiles the pinning source with the ordinary no-explicit-architecture command shape `nvcc -O3 -DQSB_ZEROS_N=<N> -o <binary> <source> -lcrypto -lm`. This candidate does not require an `-arch`, `-gencode`, link-time optimization flag, CUDA launch environment override, or precompiled binary to be included in the submission.

The repository-provided `benchmark.sh`, benchmark runner, score script, verifier, and pinning problem generator remain unchanged. The candidate does not carry generated problems, generated hits, score JSON, logs, object files, cubins, profiling reports, build stamps, or local binaries as tracked submission content. A copy of this note replaces the inherited historical `SUBMISSION.md` so that the packaged description matches the packaged source.

The source continues to use the CUDA runtime already required by the parent and the OpenSSL and math libraries already present in the parent build command. No new library, package, service, network call, background process, device daemon, or persistent state is introduced. There is no new file-system path embedded in the source.

## Build reproduction

Start from public commit `9f239c386c7e99f8815103d9c6cc4465d7c5a9ba`, apply candidate commit `b80b15c0333c0f90e0f85058272944ac55a2868f`, and confirm that `git status --short` is empty. Confirm the source identity with:

```text
sha256sum candidates/pinning/pinning.cu
```

The expected digest is:

```text
975dffe8b67f82b044498db098d4bb94ab823984c795b59d7252a6ed6fdf2e92
```

For the benchmark's current leading-zero parameter, build through the unchanged repository wrapper or use the equivalent wrapper command. The wrapper deliberately leaves the CUDA architecture unspecified so the ordinary official build path is exercised. A representative wrapper invocation is:

```text
python3 harness/gpu_wrap.py --src candidates/pinning/pinning.cu --bench pinning --zeros 24 --mode fixed_time --out <run-json> --seconds <seconds> --hits <hit-limit>
```

The wrapper expands the build portion to the inherited command form:

```text
nvcc -O3 -DQSB_ZEROS_N=24 -o candidates/pinning/pinning candidates/pinning/pinning.cu -lcrypto -lm
```

No local native-architecture binary is required to reproduce the candidate. Rebuilding from the frozen source is the intended path.

## Correctness qualification

The candidate passed the unchanged standard harness and independent verifier, with every emitted hit accepted. No locally supplied verifier patch or score override was used.

## Scope statement

This is intentionally one runtime-policy experiment. Its review surface is the single checked CUDA runtime call and its error-handled fallback. It is source-distinct from the public parent, preserves the full search and verification contract, and can be evaluated by the standard official build and harness. No throughput improvement and no official score are asserted in this note; the official runner is the measurement authority.

zarar@1337 <3 🎲
