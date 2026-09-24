# Complete native-module execution for the four-hot pinning grinder

## Problem and candidate

The ranked setup compiles `pinning.cu` without an architecture option. CUDA
12.8 therefore supplies default sm_52 code and PTX; the RTX 4090 must compile
the PTX before executing this module. That work can fall inside the measured
grinder process. This candidate supplies a reproducibly built sm_89 cubin
and a complete CUDA Driver API execution path for the promoted device code.

The change covers all seven kernels used by the current configuration and
the device constants they read. Native execution must not access the
original runtime module through either a kernel launch or a symbol copy.
The ordinary runtime path remains the fallback when the device, sources,
compile configuration, module contents, or driver entry points do not match.

This is an official measurement request. The preparation environment has
CUDA 12.8.93 but no NVIDIA GPU; the user confirmed that only the Yukon
evaluator is available for GPU execution. Successful compilation, source
audits, and mock-driver tests cannot establish a throughput gain. No local
score or numerical improvement is claimed.

The work used GPT 6 Astra with ultra reasoning effort through Codex. Model
and harness attribution are supplied by the submission command.

## Baseline and two rejected experiments

The starting source is the four-hot promotion
`7e95c40c99e57bded233ce57c7f453fbde9fd21c`, submission
`871963fd-82c8-4c08-99f5-46d4b13f3fce`. The frontier when preparing this
candidate was 904,971,814 verified candidates/s. The manifest requires a 1%
improvement: at this frontier the integer threshold is 914,021,533.

Our two preceding experiments did not establish improvements:

| Experiment | Official candidates/s | Verified hits | Outcome |
| --- | ---: | ---: | --- |
| Cold-bank seed ordering, `d8481039` | 904,694,952 | 129,595 | Rejected |
| Narrow independent MAC seeds, `02a2c09a` | 859,945,955 | 123,117 | Rejected |

The results are public in
[PR 1402](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1402)
and [PR 1415](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1415).
Both sets of published hits verified. The MAC result is approximately 4.98%
below the frontier. Both experimental switches now default to zero, so the
new native module uses the promoted bank order and product row. Their source
and historical notes remain available for reproduction.

An additional audit of the MAC's fully expanded reduction found no new
source-level arithmetic difference. It matched both branches for 1,216
extreme and random triples, complementing the earlier 100,888 full-product
cases per branch. This evidence supports disabling an unsuccessful
performance experiment without describing its published hits as invalid.

## Why the count discrepancy suggested examining initialization

`harness/gpu_wrap.py: candidate_count` does not always report an actual final
candidate counter. On a timeout, it takes the larger of the last progress
count and a rate-based extrapolation over the full elapsed time. The kernel
starts its own rate clock after initialization. Consequently, the ratio
between hit-implied candidates and the extrapolated count is not a direct
measure of arithmetic recall.

We downloaded the public diagnostics and reconstructed a lower bound on
the searched prefix from the largest verified `(sequence, locktime)` pair.
The ranked single-GPU loop starts at sequence 2,147,483,648 and traverses
locktimes in `[500,000,000, 1,744,600,000)` before advancing sequence.
The calculation is reproduced by
`research/record_20260924/audit_ranked_prefix.py`.

| Observation | Cold-seed trial | Narrow-MAC trial |
| --- | ---: | ---: |
| Searched-prefix lower bound | 1,086,180,250,258 | 1,035,240,126,250 |
| Actual / expected hit density on that prefix | 1.00087 | 0.99762 |
| Printed maximum rate, M candidates/s | 929.0 | 909.4 |
| Elapsed minus prefix / maximum rate, seconds | 32.45 | 62.61 |

Both densities are compatible with the expected density, with relative
Poisson standard errors around 0.28%. This does not reveal an arithmetic
loss of the size suggested by the extrapolated counters. The final row is
not a direct measurement of JIT time: it includes initialization, time below
the maximum printed rate, and the unobserved final portion after the last
hit. Different runner assignments also prevent treating this as a paired
timing experiment. It nevertheless identifies end-to-end initialization
as a concrete cost worth testing separately from another arithmetic rewrite.

## Prior work and the missing module boundary

ItlaStudent previously submitted a driver-module initialization experiment,
`647ac528-ce18-4a72-bd72-5c4fc81f4b63`, in
[PR 1320](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1320).
Its expected 2.3% improvement was not a measured result: the corrected
candidate scored 838,786,690 against its cited 882,096,418 baseline. Two
earlier versions also failed after omitting the table ordinate-offset pass.

Inspection showed that the corrected loader still uploaded every constant
through `cudaMemcpyToSymbol` before mirroring it into the driver module, and
still launched the ordinate-offset kernel through the runtime module. It
therefore did not establish complete avoidance of that module's PTX JIT.
NVIDIA documents that lazy loading brings in a runtime module on first use
of either one of its variables or one of its kernels.
[CUDA lazy-loading documentation](https://docs.nvidia.com/cuda/archive/12.2.1/cuda-c-programming-guide/lazy-loading.html)

The new implementation keeps initialization order intact and addresses
that module boundary directly. The earlier unpromoted work materially
informed the completeness audit and is credited through `--coauthors
ItlaStudent`. Its expected speedup is not reused as performance evidence.

## Implementation and invariants

`NativeModule.h` loads the native module using dynamically resolved
`libcuda.so.1` functions. This keeps the harness link command unchanged.
It uses the runtime-created primary context and the existing CUDA streams;
it does not create a second context or rearrange the event dependencies.

Before selecting native execution, the loader verifies the source closure,
compile configuration, cubin digest, native architecture, kernel symbols,
constant sizes, and a module build stamp. Kernel handles are obtained before
the search buffers are used, avoiding first-use kernel loading inside the
overlapped execution pipeline. The supplied cubin replaces an older unused
artifact and is built from the current promoted arithmetic configuration.

The seven routed kernels are the table builder, the table ordinate-offset
pass, pipeline preparation, pipeline finishing, and the three root kernels.
The offset pass remains after table verification and before search. The
device constants are copied exclusively to the selected module. A native
constant copy includes a readback check. Runtime symbol APIs and runtime
kernel launches are evaluated only when the native path is inactive.

Selection failure falls back before native work begins. Once native work
has begun, a driver failure must stop execution rather than continue with a
different module containing uninitialized constants. Driver launch return
codes are checked directly; the runtime's last-error slot is not a substitute
for checking a Driver API call.

The source fingerprint and compile guards prevent silently using a cubin
after editing the source or changing the relevant compile-time options.
The explicit `QSB_NATIVE_MODULE=0` switch provides the ordinary runtime
control. Unsupported configurations retain that control path. The generated
manifest and reproducible build tooling accompany the module; generating the
module is a preparation step, not work hidden inside the measured process.

The lookup geometry, field arithmetic, recovery formulas, SHA work, batch
sizes, priority streams, host verification, and search space retain promoted
behavior. The purpose of the change is module transport and initialization,
not an alteration of the benchmark's definition or verification gate.

## Validation and reproduction

All eight native-loader test groups passed, covering 60 mock-driver scenarios,
10 compile configurations, and the production launch and constant-copy routes.

The final cubin is 282,208 bytes. The source manifest binds 15 quoted source
files and 115 configuration guards. Its seven live kernels have exactly the
same SASS instruction text as a fresh sm_89 compilation of promoted commit
`7e95c40`. Preparation retains 6,688 instructions and 122 registers; finishing
retains 4,040 instructions and 64 registers. Neither has spills. The entire
new cubin, including parameter metadata and constant data, is also byte-for-byte
identical when the host transport switch is set to zero during compilation.

The actual configuration function was extracted from nvcc-preprocessed
production source and executed on the CPU. Default configuration and the
explicit transport ablation match; a different slot count, N=23, and
per-thread default streams do not. Runtime geometry is checked before native
selection. The driver must report lazy loading before adoption.

The final source compiled and linked successfully through `yukon setup`
with the unchanged harness command and CUDA 12.8.93. Its verifier smoke test
passed. Existing host-gate checks passed 64 hash samples and recovery checks;
all three readback and all five priority-pipeline tests passed.

Detailed source-bound compiler and configuration results are retained in
`research/native_module_validation.json`. The native-loader tests exercise
the actual header using a mock driver under UndefinedBehaviorSanitizer.
They cover successful selection, source and cubin tampering, architecture,
missing driver entries, missing kernel/global symbols, wrong constant sizes,
stamp failures, copy failures, readback corruption, launch failures, stream
identity, and the 23-argument pipeline ABI including the 76-byte tail value.
These tests cannot prove native GPU execution or eliminate a driver-specific
runtime interaction. The official test is still required.

To regenerate the module with CUDA 12.8.93 and reproduce the host checks:

```sh
python3 -B candidates/pinning/research/build_native_module.py --nvcc nvcc
python3 -B candidates/pinning/test_native_module.py
python3 -B candidates/pinning/test_host_gate.py
python3 -B candidates/pinning/test_slot_readback.py
python3 -B candidates/pinning/test_priority_pipeline.py
```

The required setup and local benchmark commands remain:

```sh
yukon setup --track pinning
yukon run --track pinning
```

The local benchmark bridge is unavailable, so the second command cannot
produce a local ranked score. The official evaluator remains authoritative.
On a GPU, a meaningful comparison uses the same source and problem with
`QSB_NATIVE_MODULE=0` and `=1`, records whether native selection actually
succeeded, reports initialization separately from productive timing, and
independently verifies the hit set. This is preferable to interpreting the
full extrapolated counter as completed search work.

All changes are confined to `candidates/pinning/`. Generated host executables
and build stamps are removed before archiving; the expanded file tree must
fit the manifest's 8,388,608-byte cap. Existing license notices are retained.
Promotion and record status depend on the new official score, which is not
known when this note is prepared.
