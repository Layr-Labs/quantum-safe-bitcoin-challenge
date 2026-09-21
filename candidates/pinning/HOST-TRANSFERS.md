# Pinning: one asynchronous transfer per slot batch

Effort: max. This submission was prepared using GPT 6 Astra through Codex.
It combines our previously submitted bounded 18-product parity window with
a reduction in host-side transfer commands. Development took place on a
CPU-only host. No local GPU throughput or claimed score is supplied.

## Starting point and attribution

The promoted source is commit
`94abdd0d72847b780c7d4f99da4f367e6f9f0fd1`, submission
`07009ac3-94a4-428e-b030-1f6ce317ccb7` by @terrapinelf. Its recorded score
is 797,446,582 verified candidates per second. The benchmark still reported
that source and score immediately before this submission. Its minimum
promotion improvement is 100 basis points, or 1 percent.

Our earlier submission `56043b5d-ed1f-4798-bfb9-e834314b4bf5` narrowed the
inherited parity window from 27 to 18 word products. Its validation was
cancelled before obtaining a score to release Yukon's single in-flight
submission slot. The present candidate retains its `ParityWindow.cuh`
exactly. It adds a host-transfer change in `pinning.cu`; it is therefore a
combined candidate, not an isolated measurement of host transfers against
the promoted source. There is no ranked parity-only result to compare yet.

The inherited source credits @stffinfcti for PR827 field arithmetic,
@EvanYan1024 for the original PR885 bounded parity window, and @terrapinelf
for the recovery-coordinate isomorphism. It also credits @draheemking for
the slotted pipeline and @jungjipdo's `a91746ca` for the non-slotted
`QSB_HOST_READBACK` counter/index grouping. This submission adapts that
existing grouping pattern to the active per-slot asynchronous readback.
Those mechanisms were already present in the promoted source; their
authorship is preserved. This does not claim that grouped readback is a
new technique.

Public notes, including `7244626`, were consulted for prior transfer work.
Older scores combine other changes and different starting versions, so
they do not establish a speedup for this candidate. No other pending
solver's code was merged. The top-16 cofactor and register-resident seed
changes discussed elsewhere are not part of this archive.

## The unnecessary work

The active configuration has `QSB_SLOTPIPE=1`, `QSB_SLOTS=2`, and
`QSB_TAIL_PRE=1`. The slot loop always calls
`launch_pinning_pipeline<true>`, after the existing geometry check has
restricted this path to the supported fast-tail layout.

Before this change, every slot batch issued three asynchronous copies:

1. Upload 32 bytes of SHA midstate from pinned host memory.
2. Download the four-byte hit counter.
3. Download the first 64 four-byte hit indices.

For `FAST_TAIL=true` with `QSB_TAIL_PRE=1`, the SHA state already comes
from the precomputed `tp` kernel argument passed by value. The device
midstate pointer is unused in this specialization. The first transfer
therefore updates a buffer that the compiled kernels do not read.

The two downloads are adjacent in the same stream and have the same
completion event. Placing the counter and indices in contiguous storage
allows their first 260 bytes to be returned with one asynchronous copy.
The hypothesis is that eliminating two transfer commands reduces host/API
overhead and stream work. It is not a hypothesis about changing the
cryptographic kernel's instruction schedule.

| Per batch in the active configuration | Before | Candidate |
| --- | ---: | ---: |
| Host-to-device asynchronous copies | 1 | 0 |
| Device-to-host asynchronous copies | 2 | 1 |
| Total asynchronous copy calls | 3 | 1 |
| Bytes copied by those calls | 292 | 260 |

The byte reduction is small. The main proposed benefit is fewer API and
stream operations, not a material reduction in transfer bandwidth.

## Implementation and invariants

`QSB_SLOT_SKIP_MID_UPLOAD` defaults to 1. The existing midstate upload is
retained whenever this option is disabled or `QSB_TAIL_PRE=0`. Thus a
configuration that needs the device midstate buffer still updates it.
The startup allocation and initialization remain valid in both modes.

`QSB_SLOT_PACKED_READBACK` also defaults to 1. For each slot, the GPU
allocation contains 1,025 `uint32_t` values: one counter followed by
the original capacity of 1,024 indices. `d_hit_idx_s[s]` points one word
after `d_hit_cnt_s[s]`. Only the counter is zeroed before each launch,
just as before. Device writes still use the unchanged `pos < 1024` bound.

The pinned host report has 65 words per slot. A single copy returns the
counter and the first 64 indices. The drain code reads that slot's counter
and consumes `min(counter, 64)` indices. The counter may exceed the device
index capacity, but the existing limit on reads is retained. Stale values
beyond the counter are ignored, as they were with separate buffers.

The result copy remains on the same slot stream, after the kernels and
before the same completion event. Slot ownership, event waits, sequence
rollover drains, batch accounting, candidate generation, and the exact
OpenSSL publication gate are retained. No rollover barrier was removed.
Disabling both new options restores the prior host-transfer operations,
while keeping the parity optimization available as an independent flag.

`HOST-TRANSFERS.patch` records the host-only delta from the preceding
candidate. It is already applied to this submission's `pinning.cu` and
must not be applied a second time. The updated test reads the active
source directly. `NARROW-PARITY.md` documents the preceding parity
experiment, and `SUBMISSION.md` preserves the inherited source history.
`SOURCE-MANIFEST.json` records the final file hashes and comparison base.

## Local correctness and compiler checks

The added `test_host_transfers.py` extracts the actual allocation, upload,
copy, and result-view statements from `pinning.cu`. It compiles them
against CPU implementations of CUDA memory operations with AddressSanitizer
and UndefinedBehaviorSanitizer. It tests both flags separately, together,
and disabled, plus `QSB_TAIL_PRE=0` and a three-slot configuration.

Across those six configurations, the test checks 13,403 cases and 830,752
returned indices with zero failures. Counts cover every value from 0 to
1,030, including the 63/64/65 and 1,023/1,024/1,025 boundaries. Buffers are
reused with stale tails to verify that only the intended entries are read.
The test also checks the expected number of copy calls and preservation
of the midstate upload in the fallback configuration. It was rerun against
the applied source before submission.

The existing exact host-gate test also passes on the final source: 64
SHA256d midstate cases, binary layout, EC recovery agreement, and source
checks for the publication gate and inherited C31 behavior. The earlier
parity audit passed 2,161,035 rows and 8,644,140 parity comparisons; its
header is unchanged. That earlier audit is documented in detail in
`NARROW-PARITY.md` and is not a GPU execution test.

Four complete CUDA 12.8.93 builds passed during preparation: default sm52,
native sm89, sm89 with both transfer options disabled, and sm89 with
`QSB_TAIL_PRE=0`. The applied `.cu` file and all nine included `.h`/`.cuh`
files were verified byte-for-byte against that compiled candidate. A
portable GCC 13 toolchain and compatible glibc development headers were
used because the host's system toolchain is newer than this CUDA release
supports. Toolchain files and compiled binaries are not in the archive.

For both sm52 and sm89, the complete GPU SASS dumps are byte-for-byte
identical between the preceding parity candidate and the transfer variant.
Thus this host change does not alter the compiled GPU instructions,
register counts, or spill instructions in those builds. Host disassembly
contains three `cudaMemcpyAsync` call sites before the change, one after
it, and three with both new options disabled.

The generated PTX independently confirms that `d_midstate`, the first
parameter of `kernel_pinning_pipeline<true,0>` and `<true,2>`, occurs in
the parameter declarations but is never loaded. This supports the
unused-upload analysis for the actual compiled configuration.

The setup CPU verifier smoke test passed earlier. The unchanged local
`yukon run --track pinning` could not produce a ranked result because
the development host lacks both an NVIDIA GPU and the organizer's
benchmark bridge. No harness or evaluation rule was changed to bypass
that limitation. The current benchmark records claimed scores only and
does not require a local score as a prefilter.

## Reproduction

Run from the benchmark work directory. The following commands assume
an installed CUDA compiler with a supported host compiler and headers:

```sh
PYTHONDONTWRITEBYTECODE=1 ASAN_OPTIONS=detect_leaks=0 python3 candidates/pinning/test_host_transfers.py
PYTHONDONTWRITEBYTECODE=1 python3 candidates/pinning/test_host_gate.py
nvcc -O3 -DQSB_ZEROS_N=24 candidates/pinning/pinning.cu -o /tmp/pinning-transfers -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_89 candidates/pinning/pinning.cu -o /tmp/pinning-transfers-sm89 -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_89 -DQSB_SLOT_SKIP_MID_UPLOAD=0 -DQSB_SLOT_PACKED_READBACK=0 candidates/pinning/pinning.cu -o /tmp/pinning-transfers-control -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_89 -DQSB_TAIL_PRE=0 candidates/pinning/pinning.cu -o /tmp/pinning-transfers-fallback -lcrypto -lm
git apply --reverse --check candidates/pinning/HOST-TRANSFERS.patch
yukon run --track pinning
```

LeakSanitizer was disabled because it cannot operate under this host's
tracing environment; address and undefined-behavior checks stayed enabled.
The host test frees its allocations. Its CPU memory stubs do not model
CUDA concurrency, asynchronous transfer timing, or GPU execution.

For an isolated GPU comparison, build the same source with the two new
options set to 0 and then to 1, keeping every other flag fixed. Setting
only one flag to 1 separates the unused upload from packed readback.
Setting `QSB_PARITY_WINDOW_NARROW=0` independently restores the promoted
27-product window. Rebuild explicitly after changing headers or flags.

## Interpretation and remaining uncertainty

The evidence establishes fewer host transfer commands and unchanged
compiled GPU instructions for the tested targets. It does not establish
a measured throughput gain. The old transfers may already overlap GPU
work, and changed buffer addresses can affect memory behavior despite
identical instructions. A CPU buffer test cannot replace a GPU hit-set
check or a matched timing experiment.

The inherited arithmetic's approximation limits and exact host gate are
the same as in the preceding parity submission. The gate rejects invalid
nominations but does not recover candidates that were never nominated.
No new claim of universal field correctness or perfect recall is made.

The remote ranked run will measure this combined candidate. A matched GPU
comparison with both host-transfer options disabled would help isolate
their effect from the parity change. Separate ranked runs are subject to
sampling and machine-state variation, so a matched experiment is preferable
for attributing a small gain. In particular, removing two transfer commands
does not by itself demonstrate the 1 percent improvement required for
promotion.
