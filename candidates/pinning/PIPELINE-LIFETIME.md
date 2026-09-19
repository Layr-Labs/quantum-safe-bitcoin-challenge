# Two independent reports per stream

Before batch k uses parity k mod 2, that parity is either unused or was consumed
while enqueueing k-1. The host asserts it is unarmed. H2D, all kernels, D2H and
the event are ordered within the stream. The next batch writes the other pinned
report and midstate. Even when that next batch finishes during a CPU pause, it
cannot overwrite the preceding report. Metadata also has two independent slots.
Consumption completes before a parity is reused. At each sequence boundary all
older reports are already consumed, so only the latest report remains to drain.

The CPU model covers 21 distinct stream/batch-count shapes, five sequences each,
under the conservative schedule that completes all queued GPU copies before the
CPU reads the previous report. This is an abstract lifetime argument. The exact
native 0143 audit adds a 10ms delay after the old event and checks all emitted
records in eight partial-batch/sequence modes. The combined carry-complete source
repeats those eight delayed-read modes in native 0240; the original outputs were
downloaded and independently checked before source-bound registration.

This source preserves 9bb3's checked persistent output. The host lookahead change
was first isolated with unchanged device code in a63. This combined candidate
also changes the point-chain square and multiply final carry correction in
GPUMath.h. It therefore has its own native arithmetic, point and pipeline proof
and requires its own full production qualification. The earlier device-image
identity does not apply to this combined source.
