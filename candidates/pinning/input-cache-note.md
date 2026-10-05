# Reuse unchanged per-slot input uploads

## Scope and starting point

This candidate targets only the pinning track and starts from promoted source
`582a99408761f904f7f92a5d64d9ca0dcc76924a`. The advertised frontier when this work
started was 1,036,462,054 verified candidates per second. That number belongs to
the inherited implementation, not to this candidate. No full candidate score is
claimed here. The official Linux CUDA 12.8 / RTX 4090 evaluation must determine
whether this change clears the benchmark's promotion threshold.

The submitted work was developed with GPT 6.1 Sol, medium effort, using Codex.
This patch is independent of the separately submitted graph activation change.
It leaves graph configuration at the promoted base setting. It does not claim
authorship of the inherited cryptographic kernels, native carrier, green-context
pipeline, compact readback, CPU verification, or batch accounting. Their original
notices and attribution are preserved. This contribution is a small host-side
input cache and an independent focused CUDA correctness probe.

## Observed repeated work and hypothesis

In the slot pipeline, every batch first waits for the previous use of its slot,
copies a small input record into a pinned host mirror, and enqueues a transfer
into the slot's device input buffer. The subsequent input event orders that
transfer and the counter reset before the compute stream reads the input. This
is correct but often redundant. With the default ASICBOOST / B0CONST path,
`grp_slot` is constructed outside the inner locktime batch loop. Its contents
remain unchanged across the iterations of that loop. The five slots can therefore
reuse their already uploaded bytes on most subsequent visits during that group.

The default record is four 96-byte sequence records, or 384 bytes per slot. A
byte comparison on the CPU is cheap relative to submitting another asynchronous
host-to-device copy. The hypothesis is that removing those redundant copies
reduces host submission work and input-stream traffic without changing any
device arithmetic or per-batch state. The likely benefit depends on the number
of batches per input group and how much submission overhead is exposed in the
official pipeline. A large microbenchmark saving does not imply a similarly
large improvement in overall candidates per second.

I chose exact byte comparison instead of indexing the cache by sequence or
locktime. A semantic key would need to include every property affecting each
supported input layout. Byte comparison uses the payload that the original
implementation actually sent and automatically includes every represented field.
It also avoids assuming that equal sequence numbers imply equal midstate data.

## Implementation

`SlotInputCache.h` defines a non-owning cache for one fixed-size device allocation
and its pinned host mirror. Initialization records the existing pointers and
size. It does not allocate memory, replace streams, or change ownership. A cache
starts invalid, regardless of bytes already present in the allocations. That
ensures the first use always uploads its complete record and never compares an
uninitialized pinned allocation.

On enqueue, an invalid pointer or unexpected size returns
`cudaErrorInvalidValue`. If the cache is valid and the proposed bytes equal the
pinned mirror, the method returns success without transferring them again.
Otherwise it invalidates the cache, copies the proposed bytes into that mirror,
and enqueues exactly the original `cudaMemcpyAsync` to the device input buffer.
Only a successful enqueue marks the cache valid again. A failed enqueue cannot
cause a later retry to skip the upload because of an optimistic validity flag.

In `pinning.cu`, each existing slot receives one cache object. Its initialization
occurs after successful device allocation and the original initialization copy.
Each of the three input-upload branches delegates to its slot cache: B0CONST
sequence records, ordinary ASICBOOST tail records, and the 32-byte midstate path.
The unused-midstate branch still does no input upload. No changed byte layout is
introduced, and allocation sizes remain calculated by the original code.

The cache's important precondition is that the caller waits for slot completion
before calling enqueue, including before a potential hit. This is required
because the pinned mirror can still be the source of a prior asynchronous DMA.
Both supported refill branches already provide that precondition: `collect_slot`
or `drain_slot` synchronizes `slot_done` before the next use of the slot. The cache
call remains below that synchronization. The initial unbusy slot has no prior
cache DMA to await. Final draining and resource cleanup are unchanged.

Counter reset remains unconditional on every successful input enqueue. Compute
launches, input/completion events, locktime and sequence updates, host hit
validation, and candidate credit also remain unchanged. Cache hits only remove
the redundant input transfer. The reset and input event continue to establish
the normal ordering chain. Device input storage is read by the compute kernels;
it is not an output scratch allocation and is not intentionally overwritten
between uses. If future work writes this storage independently, it must invalidate
or stop using the cache.

## Focused correctness experiment

`slot-input-cache-probe.cu` includes the exact submitted helper. It creates five
independent nonblocking CUDA streams, device input and output buffers, pinned
input mirrors and readback buffers, and completion events. It cycles over those
five slots for 10,000 batches. Before reusing a busy slot, it synchronizes that
slot's completion event and compares every readback word with the expected input.
A simple kernel copies all input words to output, followed by asynchronous
readback and an event record. This directly tests that the consumer sees the
correct bytes after first upload, unchanged-input reuse, and replacement.

The first input is all zero, while the pinned mirror is deliberately filled with
nonzero bytes before cache initialization. Later inputs include long identical
runs and changes in every word. The test also checks rejection of repeated
initialization, null input, and an incorrect payload size. The final version uses
96 words, matching the default 384-byte record. Earlier exploration used 128
words and passed twice; those runs also exercise a different fixed input size.

Local reproduction uses CUDA 13.4 with MSVC on Windows and an RTX 3080, sm_86:

```text
nvcc -O3 -arch=sm_86 -ccbin <MSVC-compiler-directory> candidates/pinning/slot-input-cache-probe.cu -o slot-input-cache-probe.exe
slot-input-cache-probe.exe
```

The compiler directory is the host system's ordinary MSVC x64 compiler location.
On Linux, omit `-ccbin` when nvcc can locate its supported host compiler and use
the appropriate architecture for the installed GPU. The probe requires no
credentials, blockchain connection, benchmark API, or external service.

Result for the final 384-byte probe:

```text
PASS: 10000 full-payload checks over 5 slots; zero first input, cache hits, changed inputs, invalid arguments
uncached identical input + completion: 14.037 us/iteration
cached identical input + completion: 0.249 us/iteration
```

The timing section performs 20,000 iterations with stream completion before each
enqueue. The baseline copies identical bytes to the pinned mirror and sends them
on every iteration; the cached case calls the submitted helper with identical
bytes. Both finish with a stream synchronization. This measures the deliberately
isolated redundant-transfer case, not cryptographic throughput. CUDA scheduling,
Windows driver behavior, and an idle stream differ from the official pipelined
Linux run. The timing is evidence that eliminating repeated transfers removes
real work locally, not a portable prediction or a submitted score.

## Validation boundaries and review

The probe tests the exact cache helper and full-payload observation across slot
reuse. It does not execute the entire pinning algorithm, test every compile-time
configuration, inject asynchronous hardware failures, or validate the official
green-context pipeline on an RTX 4090. Full cryptographic hit validation and
throughput measurement remain the official evaluator's responsibility. Static
review confirmed that both refill branches synchronize before the replacement
upload and that the input allocations have stable per-slot ownership.

The full Linux harness was not run locally because this machine is Windows and
has no configured Linux environment. The candidate changes host logic only. The
checked-in native sm89 carrier and device code are unchanged, so a device carrier
rebuild is unnecessary for this patch. No changes are made outside the editable
pinning directory. The standalone probe is not part of the ranked executable.

If the official result fails correctness, investigate the lifetime and ownership
of the device input first and disable the cache until reproduced. If correctness
passes but the score fails promotion, the input submission cost may already be
hidden by GPU work or input groups may be too short to yield enough cache hits.
That is a useful negative result and does not justify changing credit accounting,
reducing validation, or relabeling the local timings as ranked throughput.

The next step is official evaluation of this exact candidate. Only an accepted
and promoted result can support an eligible external reward claim; neither the
probe nor a successful upload establishes a cash payment.
