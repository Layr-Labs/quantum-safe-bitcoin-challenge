# Independent native transport audit

This is a CPU and source audit. It does not measure GPU execution, startup
latency, throughput, or the official score.

Reproduce from the repository root:

```sh
python3 -B candidates/pinning/test_native_module.py
```

The suite compiles the actual `NativeModule.h` under UndefinedBehaviorSanitizer
against a mock dynamic loader and CUDA driver. Fixture files exercise the real
SHA-256 and ELF checks. Its 60 driver scenarios cover successful initialization,
unsupported architecture/configuration, all three per-thread stream macros,
missing or modified files, missing driver entries, eager loading, failed module
loading, every missing kernel/constant, constant-size mismatch, bad source stamp,
and copy/readback/launch errors. Failed initialization cleans up before any
native write or launch. Errors after selection propagate without selecting a
different execution backend.

Successful paths verify all seven kernel handles and ten runtime constants,
exact uploaded bytes, unchanged launch geometry and stream identity, and the
actual variadic argument packer. Both pipeline stages preserve all 23 typed
arguments, including the 76-byte by-value SHA-tail structure. Invalid argument
counts are rejected before the driver call.

Independent source checks compare each native route with its runtime launch's
argument order and geometry, demangle every required kernel, and confirm that
every `cudaMemcpyToSymbol` call is routed through the selected backend. The
table builder and Y-offset kernel are both routed. Existing prepare/root and
root/finish event handoffs retain their positions around the same stages.
The runtime native-geometry predicate equals the existing FAST_TAIL predicate
and precedes table construction, preventing generic input from selecting the
FAST_TAIL-only native functions.

The actual generated manifest accepts its baseline configuration and rejects
nine altered math/layout definitions. The module validates all 15 source files,
the cubin digest, ELF architecture, required symbol sizes, and its embedded
source/configuration stamp. The generated manifest is intentionally excluded
from the source closure to avoid a hash cycle; the stamp covers the source
hashes, compiler identity and build flags. Final hashes are recorded in
`native_module_audit.json`.

The initial review caught missing runtime input-format gating and the
per-thread-default-stream incompatibility; both were corrected before this
audit. CUDA device initialization and property errors are checked. Every driver
launch's return status is checked directly, because CUDA Runtime last-error
state does not substitute for a driver result. Native table validation failure
aborts instead of switching to the runtime module after native work begins.

Remaining empirical questions require the official GPU: driver/runtime
interoperability on the evaluator, achieved startup savings, and native cubin
versus driver-JIT scheduling. CPU mocks and source identity do not establish
those outcomes. Existing event-dependency tests cover the unchanged
`PriorityPipeline.h`; this audit verifies transport and routing around it.
