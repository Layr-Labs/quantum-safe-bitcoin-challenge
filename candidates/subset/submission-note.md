# Subset: native Ada module loading with promoted arithmetic and fresh runtime data

Effort: high. Research, implementation and independent validation use Codex
GPT-6 Astra at high effort, coordinated through separate Herdr panes.

## Problem and hypothesis

The subset benchmark builds its CUDA source using a fixed command without an
architecture flag. With the CUDA 12.8 toolchain this produces a device image that
includes compute_52 PTX. An RTX 4090 cannot execute the sm_52 machine image and
normally needs the driver to compile the PTX for its own architecture. That
compilation happens when the candidate process initializes or first uses the
module. The benchmark's root-owned clock includes candidate startup.

This experiment packages a native sm_89 device image generated from the promoted
subset implementation and routes the host through CUDA 12.8's public library
loading API. The intended benefit is more time spent evaluating fresh candidates
inside the fixed wall-clock window. There is no intended change to field
arithmetic, SHA-256, candidate enumeration, point-table geometry, or the exact
tentative-hit replay.

This is a startup experiment, with an unmeasured ranked effect. The existence of
PTX JIT is established by NVIDIA documentation; the amount of the current
benchmark's score deficit caused by it is not established. We do not claim that
the entire difference between a reported peak rate and hit-derived score is JIT.
It could also include changing GPU rate, initialization, or other pauses.

## Base and provenance

The source base is subset promotion `9ac2515450446dbadbe061e98ebfc317c36d4999`,
submission `7aef224a-e3ff-43f9-9877-50cdbda3f653`, with an official score of
623,518,629 verified candidates/s. That promotion is the starting point, not a
gain produced by this experiment. All inherited notices and GPLv3 source remain
included. The sibling pinning track and the benchmark harness are outside the
change surface.

The immediate motivation includes terrapinelf's public submission note
`a75cf15a`, which reports local cold-start measurements for a compact PTX variant.
That note's proposed 14–18x local-to-ranked JIT multiplier is an inference from
aggregate score data. We treat it as a hypothesis and credit that research; no
compact SHA or cold-helper patch from that submission is copied into this
experiment. The native-loading approach gives a different test of the underlying
startup hypothesis. Another pending compact-source experiment, `1e6215d2`, is
also recorded in the research ledger to avoid presenting that approach as new.

## Reproducible device image

The first baseline image was generated with CUDA 12.8.93 as follows:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 --ptx -o base.ptx candidates/subset/subset.cu
ptxas -arch=sm_89 -O3 -v -o base.cubin base.ptx
```

The native assembly step uses the same compute_52 PTX frontend output as the
ordinary benchmark build. It does not silently change architecture-conditional
source by asking the frontend to compile for compute_89. The offline assembler's
schedule can still differ from the installed driver's JIT schedule, so identical
CUDA/PTX arithmetic is not evidence of identical runtime performance.

The initial baseline PTX is 3,515,568 bytes with SHA-256
`8599b133fe9a27ffc33651db33e5dcdec4385955bc831907762a6430c7b674f5`.
Its sm_89 image is 933,536 bytes with SHA-256
`994014fb153b73eb4338b013d373734dc8350041378b2c39ed0199987cc38ae8`.
Regeneration from the integrated source produced the same PTX and cubin hashes
byte for byte. The JSON manifest records the current recursive source hashes.

Seven kernel entries are present: the table builder, direct epoch producer,
group producer, incremental epoch producer, first-state producer, main digest
consumer, and exact tentative-hit verifier. The independent ABI extraction finds
40 parameters in the digest consumer and 20 named device global/constant symbols.
Main digest resource usage is 128 registers/thread, 49,152 bytes shared memory
per block, and no stack frame. These are static compiler facts, not throughput.

## Host integration requirements

The implementation uses the public CUDA 12.8 runtime library interface rather
than generated registration internals. An early experiment trying to recompile
`nvcc --cuda` generated host output as a CUDA source failed duplicate CUDA vector
type declarations. That approach was abandoned before production integration.

Native and source modes are selected for the entire invocation before any
problem-dependent device symbol is initialized. Once a mode is chosen, all
kernel launches and all symbol reads and writes belong to that module. Switching
one kernel to the original module after native initialization would use separate,
uninitialized constants and is therefore invalid.

Each launch retains its declared parameter types, order, grid and block
dimensions and the promoted zero-shared-memory/default-stream configuration. This matters because a raw generic
argument pack can deduce an integer for a literal null argument where the kernel
ABI expects an eight-byte pointer. The independent review checks the wrapper
against the compiled PTX entry declarations, including this case.

Symbol routing includes both directions. The SHA schedule preparation reads the
round constants K with `cudaMemcpyFromSymbol` in two host helpers; redirecting
only symbol writes would leave an access that can initialize and JIT the original
module. Native global pointers are obtained with `cudaLibraryGetGlobal`,
checked against the declared size, and passed to ordinary memory-copy APIs.
They are not valid CUDA Symbol API arguments.

The image is generic program code. The fresh benchmark instance still supplies
every preimage, scalar, recovery coordinate, constant schedule, and folded-base
table at runtime. No hit, problem seed, or instance-specific point table is
embedded. The existing text output and exact GPU replay remain the source of
candidate nominations consumed by the independent host verifier.

## Validation and measurement limits

Authoring is on a CPU-only macOS machine using an x86_64 CUDA 12.8.93 container
for compilation and static inspection. There is no local NVIDIA device. Neither
the loader nor its synchronization nor its throughput has been executed on a GPU
here. The ranked runner is required to establish real device behavior and a
verified score. No local claimed score is attached.

The independent CPU contract work checks 5,810 exhaustive small combinadics,
10,004 full-domain rank/unrank cases, 274,560 candidate/launch mappings, and 36
folded recovery boundary comparisons. It reconstructs 120 full preimages across
three fresh seeds and passes 34 low-difficulty hits through the unchanged
independent verifier with zero failures. It also checks duplicate
canonicalization, invalid indices and recids, seed binding, and zero/infinity
oracle cases. These tests establish the host mathematical contract; they do not
claim to execute the native kernel or its loader.

The separate structure model checks 9,728 complete SHA256d messages and proves
that the current 128-window candidate family has 1,051,964,508,672 unique skip
sets. At a sustained target near 997.63M/s it would exhaust before 1200 seconds.
The present native experiment preserves that family to isolate startup effects;
any future variant capable of that rate needs a separately validated disjoint
extension. A proposed second family requires 47 first-state classes, whereas the
promoted 128-window build provides only 16 slots. Changing that later requires
matching constants, allocation and strides, plus a full pipeline drain between
families. A larger domain is not itself a throughput gain.

## Interpretation and next experiments

The user-defined campaign target is an additional 60% over the starting record,
or 997,629,806.4 verified candidates/s. That is a research target, not a projected
result of this package. Even treating the entire gap to the crown's roughly
718.8M/s peak as removable startup gives only about 15.3% improvement and leaves
another 38.8% steady-rate gain to find. Improvements from initialization,
arithmetic and occupancy cannot be multiplied without a matched measurement.

The arithmetic audit also analyzes GLV12 and larger fixed-base tables. GLV12
removes three mixed additions but adds scalar splitting, an endomorphism multiply,
and a roughly 1.365 GiB table with more DRAM traffic. Its benefit on another
track does not establish its benefit here. Tensor-core convolution has exact
integer models but significant packing and reduction cost; it is not part of
this package. These remain separate follow-up experiments.

## Sources

- Repository `spec/PROBLEM.md`, `spec/SCORING.md`, `benchmark.json`, and
  `harness/gpu_wrap.py` define the instance, editable surface, fixed build,
  output interface and verified-hit score.
- [CUDA 12.8 runtime library API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__LIBRARY.html)
  documents native image loading, kernel lookup and global lookup.
- [NVIDIA dynamic runtime loading examples](https://developer.nvidia.com/blog/dynamic-loading-in-the-cuda-runtime/)
  show loading and launching with the runtime alone.
- [NVIDIA CUDA compiler documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
  documents PTX, cubin and the compilation phases.
- [CUDA 12.8 programming guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html)
  documents lazy module/data loading and CUDA cache controls.

## Integration checks

The fixed-command N=24 build, N=3 source fallback build, and explicit
`-DQSB_NATIVE_MODULE=0` N=24 source build all compiled with CUDA 12.8.93.
The independent 12-case mocked-CUDA suite invokes all seven real kernel signatures in
14 launches and checks every argument byte, grid and block, including null
pointer conversion and the 40-parameter digest signature. It exercises native
selection, unsupported-architecture and missing-payload source selection, and
expected process failure for ABI, bounds, declared-size, launch, load, hash and
selection-state errors. These are host integration checks, not GPU execution.

The runtime preflights the 13 globals actually read or written by host code. The
full 20-symbol inventory remains in the provenance manifest and the native
image. This avoids making unused library globals a prerequisite for execution.
All 14 host symbol transfers are routed, including both reads of K.

The integrated package reproduces the baseline PTX and cubin exactly, and all
35 ELF sections compare equal. The 21 recursive source hashes in
`native_manifest.json` match the packaged inputs. The payload remains 933,536
bytes. The source directory, including the native payload and tests, is checked
against the 8,388,608-byte limit before upload.

Reproduce the packaged host-routing checks without a GPU:

```sh
python3 candidates/subset/tests/native_runtime/check.py
# Or with the existing CUDA container and repository mounted at /work:
python3 candidates/subset/tests/native_runtime/check.py --docker qsb-cuda
```

This suite includes distinct bad-length and equal-length bad-hash cases, plus
round-trip copies through all 13 host-accessed native globals. It produces no
claimed candidate throughput. Temporary test files are removed after execution.

Independent scratch regeneration also reproduced the entire generated JSON and
header exactly, with all 21 source hashes verified. Independent review found
no standards or specification correctness defects; device lookup behavior and
ranked performance still require the official GPU evaluation.
