# Share immutable input records across completed-slot lifetimes

## Starting point and scope

This pinning candidate starts from promoted commit
`582a99408761f904f7f92a5d64d9ca0dcc76924a`. The inherited frontier was reported
as 1,036,462,054 verified candidates per second. This submission does not claim
that score as its own and has no full local ranked score. It preserves the
promoted device code and checked-in native sm89 carrier. Only host input
ownership and upload scheduling change. The existing CUDA graph switch remains
disabled as at the base commit.

The work was developed with GPT 6.1 Sol at medium effort in Codex. The existing
cryptographic kernels, host gate, native carrier and pipeline are prior work;
their original licenses and attribution remain. The new work consists of
`SharedInputPool.h`, its host integration, and an independent CUDA probe. No
unpromoted third-party implementation was used for this pool.

## A correction to the earlier per-slot hypothesis

The initial attempt cached input separately in each of the five slots. Its
standalone repeated-input test passed and showed that a cache hit avoids a real
transfer, but closer review of the actual ranked geometry revealed a limitation:
that design ordinarily gets no hits. The low-locktime-byte loop produces a
constant input record for five batches. Five slots each handle exactly one of
those batches, and on the next visit to the same slot the low byte has changed.

The calculation is reproducible from the base source. `LT_MIN` is 500,000,000,
`LT_MAX` is 1,744,600,000, `QSB_BATCH` is 4,194,304, and `QSB_AB_K` is four.
`batch_lts` is therefore 1,048,576. The code computes `ab_hn` as
`((LT_MAX-LT_MIN)/256) & ~31u`, which is 4,861,696. The ceiling of `ab_hn /
batch_lts` is five. Changing the low byte changes the `r01` part of `grp_slot`,
so exact per-slot comparison then misses. This correction matters: a useful
microbenchmark does not establish that the real program reaches its fast path.

This candidate uses a pool shared across slots instead. All five batches in the
same input group can reference one uploaded allocation. It retains the original
five-slot pipeline and does not insert a global drain between input groups.
Different groups can remain in flight at once, using different pool entries.

## Ownership and lifetime design

`SharedInputPool<Slots>` is non-owning with respect to memory. The caller passes
the existing five device input allocations and their five pinned host mirrors.
The pool creates one transfer-ready event per entry. It does not increase the
number or sizes of input allocations, move the cryptographic buffers, or change
the byte layout expected by the native device image.

Each entry stores a validity flag, a reference count and its transfer-ready event.
Each pipeline slot stores an entry binding. Before requesting new input for a
slot, the existing `collect_slot` or `drain_slot` waits for its completion event.
The pool then releases that slot's prior reference. This is a conservative
reference scheme: a completed slot may keep a reference until its next visit,
but a still-running slot can never lose its reference through another slot's
visit. Both supported refill paths retain their original completion barriers.

The pool compares proposed bytes with every valid pinned mirror. A matching
record may belong to an active entry. Sharing it is safe because the record is
immutable while referenced and the compute kernels read the input allocation.
On a match, the new slot's stream waits for that entry's transfer-ready event.
That wait is required even when the byte comparison succeeds: another stream
may still be uploading the bytes. The pool increments the reference count,
records the new slot binding, and returns that entry's device pointer.

On a miss, only an entry with zero references can be selected. The pool marks
it invalid, copies the proposed bytes to its pinned mirror, enqueues the normal
host-to-device transfer, and records its ready event on that same stream. Only
successful transfer submission and event recording make the entry valid. The
caller receives the new pointer and subsequently enqueues the ordinary counter
reset and input event. Its existing compute ordering then follows the ready
event or transfer, as appropriate.

There is always a free entry on a miss under the caller precondition. Immediately
after releasing the current slot, at most `Slots-1` slot bindings remain. Those
bindings can refer to at most `Slots-1` distinct entries. A pool with `Slots`
entries therefore has one unreferenced entry. No assumption about consecutive
group sizes or regular round-robin payloads is needed for this argument.

Re-recording an entry's ready event happens only after its reference count reaches
zero. Every former consumer completed before releasing its reference. Thus no
consumer can still be reading its bytes, and the previous asynchronous transfer
also completed before its former slot was released. Pending waits cannot be
redirected to an unrelated new record through premature reuse.

An asynchronous CUDA submission failure poisons the pool, causing later calls
to fail rather than retry with potentially outstanding DMA. Production already
exits on such an input enqueue failure. Invalid slot indices, pointer arguments,
payload sizes and incomplete initialization are rejected before releasing a
binding. The first use always uploads: initially valid is false, so uninitialized
mirror bytes are never read by a comparison.

## Host integration

`pinning.cu` initializes one pool beside the existing slot arrays. Each successful
device input allocation and original initialization copy is followed by pool
entry initialization. The existing memory remains owned by the original arrays;
the pool does not overwrite those allocation pointers or cause duplicate frees.

At the input-upload site, a local pointer initially selects the original slot
allocation. The three upload branches request shared input for their respective
payloads: B0CONST sequence records, ordinary ASICBOOST tail records, or the
32-byte midstate. The unused-midstate branch still does no upload. The returned
pointer is passed to the existing sub-batch or monolithic launch path. Hit counter
reset, readback, slot completion, candidate enumeration, host verification, CPU
co-grinder, and candidate accounting remain unchanged.

The default B0CONST payload is four 96-byte sequence records, totaling 384 bytes.
Exact byte comparison includes each represented field and padding; the existing
code zeroes the group before filling it. This avoids a semantic cache key that
might omit a midstate-dependent field. If future code writes the device input
allocation as scratch storage, this pool's read-only assumption must be revisited.

## Meaningful CUDA probe and results

`shared-input-probe.cu` includes the exact helper. It uses five independent
nonblocking streams, five existing device input allocations and pinned mirrors,
separate device output/readback storage, and per-slot completion events. Before
reusing a busy slot, it synchronizes its completion and checks every one of the
96 readback words against the expected record. A consumer kernel reads the
selected shared pointer and copies the whole 384-byte payload to output.

The first phase deliberately matches the real geometry: five consecutive batches
share one payload and each of the five slots is used once. The second phase
changes every payload, forcing eviction and zero reuse. The third uses groups of
17 batches, exercising repeated slot reuse, sharing across group boundaries, and
eviction. Every fifth batch delays its producer stream before enqueue. This is
especially useful for the ready-event dependency: a hit on another stream must
not race the delayed producer's transfer. The first record is all zero while the
pinned allocations were filled with nonzero bytes before initialization.

On the local Windows CUDA 13.4 / MSVC / RTX 3080 environment, the probe produced:

```text
phase 0: 2000 uploads, 8000 reuses
phase 1: 10000 uploads, 0 reuses
phase 2: 589 uploads, 9411 reuses
PASS: 30000 full-payload checks; 12589 uploads, 17411 cross-slot reuses; delayed input producer, changed inputs, eviction, invalid arguments
```

The phase-zero counts are explicitly asserted by the probe: one upload per group,
four reuses, and no extra transfers. Phase one explicitly asserts no false reuse.
The probe also checks invalid slot, null input and wrong size rejection. These
are measured correctness and transfer-count results, not ranked throughput.

Reproduce from the repository root, selecting the installed host compiler:

```text
nvcc -O3 -arch=sm_86 -ccbin <MSVC-x64-compiler-directory> candidates/pinning/shared-input-probe.cu -o shared-input-probe.exe
shared-input-probe.exe
```

On Linux, use the supported default host compiler and the installed GPU's target
architecture. The probe needs no credentials or external services. Its helper
uses ordinary host C++ and CUDA runtime APIs available before CUDA 12.8.

## Remaining uncertainty

The full ranked Linux executable was not built locally: this host is Windows,
and no Linux environment is configured. The probe uses sm86 with CUDA 13.4,
whereas official evaluation uses sm89 with CUDA 12.8. It does not exercise the
complete cryptographic algorithm or every compile-time configuration. Static
review and the producer/consumer test support the ownership and ordering design;
the official evaluator must verify full hits and determine the actual score.

Eliminating 80% of tiny input copies does not imply eliminating 80% of runtime.
The added byte comparisons and ready-event waits have costs, and most benchmark
time belongs to the GPU kernels. Some or all transfer overhead may already be
hidden by the pipeline. No claim is made that the measured transfer-count change
will clear the 1% promotion threshold. If it does not, the correct outcome is a
negative ranked result rather than altering verification or accounting.

The next step is evaluation of this exact candidate with the normal harness.
Only an accepted and promoted record increment can support a reward claim;
successful local tests, an uploaded archive, or a displayed score alone do not
establish earned money.
