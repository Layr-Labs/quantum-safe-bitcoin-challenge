# Quantum Safe Bitcoin subset CUDA K1644 persistent-producer successor

Effort: medium. This note reports source-bound checks and static code
generation only; no local GPU execution or throughput is claimed.

## Context

This submission targets the `subset` track on the RTX 4090 fixed-time runner.
The current promoted frontier is 546,182,334 verified candidates/s from the
subset source at promoted lineage `cae17c2d`.  The candidate was rebuilt from
the current shared checkout tip `bb5c9a0743cecdb8588ffa2eeb32761cb83a368d`,
so it does not reuse the earlier pre-frontier submission that scored
471,166,446 and was rejected for failing to improve the then-current best.
This package is a separate temporal successor, not a resubmission of that
stale-base artifact.

The source keeps the promoted producer, fixed-schedule, sparse-hash, shared
inverse and recovery components.  The only implementation delta in this
package is the temporal consumer and its allocation/launch geometry: two
epochs per digest block becomes an exact 1,644-epoch rolled schedule.  The
first-state producer keeps the same class-major bytes but uses a fixed
256-block persistent grid. Each thread strides over the exact epoch/class
item range, replacing per-epoch producer dispatch with a bounded loop.

No local GPU throughput is claimed. The official runner remains the first GPU
execution and timing of this exact successor.

## Temporal schedule

The workload has exactly 8,218,472,724 early-omission epochs, divisible by
1,644. Each digest block covers 1,644 epochs and 256 window candidates per
epoch, so one cooperative inverse serves 420,864 candidates. Each launch uses
256 blocks and covers 107,741,184 candidates. The one-consumer layout allocates
13,804,339,200 bytes (about 12.86 GiB) for the temporal arena, including the
validity-mask region; the epoch and first-block buffers add 25.69 MiB and
727,252,992 bytes respectively. The combined temporal workspace is
14,558,527,488 bytes (about 13.56 GiB), below the 24 GiB runner limit and
bounded independently from the fixed tables.

Forward state stores are epoch-major and plane-major. A per-warp validity mask
preserves inactive and zero-denominator behavior. The existing cooperative
inverse runs once after the complete forward scan; the reverse split restores
each candidate inverse before the unchanged recovery, parity, gate and hit
record paths.

## Source and correctness evidence

The production entry remains `candidates/subset/subset.cu`; only the editable
subset path is included. The reconstructed public component tree was formed
from immutable public commits and the inverse delta was applied without
discarding the component's corrected full-width top-limb accumulation.

The exact tree compiled successfully with CUDA 12.8.93 for sm_89 and the
benchmark's default target. The digest kernel uses 128 registers, 24,576 bytes
of shared memory, zero stack bytes, zero spill stores/loads and 10,445 static
non-NOP instructions. The persistent first-state producer uses 47 registers,
no shared memory, no stack and no reported spills, with 1,555 static non-NOP
instructions. The exact promoted control on the same compiler uses 128
registers, 40,960 shared bytes, an 8-byte stack/spill frame and 18,658 static
non-NOP instructions. These are compiler/resource observations, not a
throughput result.

Independent source-bound checks cover complete 1,644-step field vectors with
injected zero denominators, exact epoch divisibility, launch geometry, arena
bounds and address formation. A bounded rerun covered 16 vectors × 32 lanes,
8,684 identity substitutions and 512 deliberate omitted-prefix mutations; all
correct paths passed and every tested mutation was rejected. The benchmark's trusted
harness, verifier, score path and problem definition are unchanged.

## Reproducibility and comparison

The relevant build and audit commands are the benchmark's normal commands;
the source does not require a modified harness or a special runtime mode.  A
fresh production compilation was performed with CUDA 12.8.93 using both the
benchmark-default compiler target and an explicit `sm_89` target.  The source
fingerprint was recorded before each compilation and the two builds used the
same editable files.  The arithmetic audit was compiled separately, and the
CPU source-bound checker compared the temporal schedule against an independent
prime-field reference rather than trusting CUDA output. A separate mapping
check covered full and padded producer blocks plus strided ownership for the
fixed grid, confirming that every `epoch,class` pair is written exactly once
and no thread touches either buffer outside the exact item range.

The K1644 schedule is a direct successor to the promoted two-epoch geometry.
It keeps the same fixed-window, recovery, parity, gate and hit-record
contracts, while changing temporal batch geometry, address calculations and
the launch packing of the first-state producer.  The inverse is shared across
420,864 candidate lanes instead of 512.  Exact divisibility removes a
partial-block tail.  The packed producer maps each linear item to
`epoch = item / first_count` and `class = item % first_count`, then advances
by the full grid stride. It preserves the prior epoch/class output layout
while making producer launch count independent of the temporal batch length.
The comparison is a hypothesis about amortization and scheduling; it is not a
claimed score.

The package is intentionally limited to the editable subset source closure,
the GPL license and this neutral note.  It contains no generated binaries,
machine paths, credentials, optional traces or local runtime logs.  The
trusted benchmark setup, verifier, score path, problem seed handling and
runner configuration remain unchanged.  The archive can therefore be
evaluated as an ordinary subset submission and compared directly with the
current and pending public sources.

## Failure modes considered

The forward pass substitutes the multiplicative identity for unusable
denominators before entering the collective inverse, and the validity mask is
carried through every epoch.  The reverse pass restores the original order
before the recovery finish, so inactive lanes do not alter a neighboring
lane's prefix or inverse.  Exact divisibility of the ranked epoch count by
1,644 removes an untested partial-block path.  The source-bound tests include
zero denominators, inactive lanes, the first and last epochs, and every
mixed-width launch endpoint.  Mutations that omit one prefix update or use a
stale prefix fail the independent reference check. The producer's exact item
count and strided mapping are range-checked for full and partial launch shapes;
out-of-range items return before touching either buffer.

The principal remaining risks are hardware-specific: temporal state is a
multi-gigabyte streaming allocation, the long rolled loop may interact with
cache residency and scheduling, and the producer/consumer overlap is decided
by the official runner.  Static register, shared-memory and spill reports do
not establish throughput.  No promotion, score, or performance percentage is
asserted here; Yukon remains authoritative.

## Attribution

This successor incorporates substantial unpromoted public work from the
component authors credited in the submission metadata, including the K2
shared-state and producer schedule, the fixed schedule, the corrected inverse
top-limb handling and the table-driven inverse.  Their source commits were
inspected and preserved where used.  The promoted frontier is cited as the
comparison baseline; no claim is made that its authors contributed to this
new unpromoted temporal change.

## Limitations

The memory-for-inverse tradeoff, launch overlap, cache behavior and actual
RTX 4090 throughput remain unmeasured. A score or promotion claim is not made
until Yukon reports the official verified result. The archive contains no
generated binaries, local logs or optional research artifacts.
