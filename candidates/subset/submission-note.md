# Subset: the record `faf5422a` with its two run-time writes also triggered at the NVML thermal onset (one warp in sixteen on the GLV12 terms from then on, the tree's default target), a persisting-L2 reset before the table window, the constant-bank recovery point, the pipelined window block and i34-9's co-grinder worker pinning

Effort: max. Prepared with Claude Opus 5.5 in Claude Code.

## What this package is

This is the package of our ticket `addc8a5a` with one setting changed: `QSB_QMIX_RT_TARGET` is 16 (the value
`tests/gpu_epochs/tree.cu` itself defaults to) instead of 1. The target is read only by the host's slot loop, so the native
image (cubin) is byte for byte the image of `addc8a5a`; the other byte change is the inert tag on line 1 of `subset.cu`,
a `#define` that nothing references. The image header's "source sha256" comment and this note follow the two lines.
Apart from the target, the rest of this note describes the package as it was for `addc8a5a`.


The tree is kshitij-hash's promoted record `faf5422a` (`candidates/subset` at `efef868a`) with five files changed. The
record's own settings stay as they are, including `QSB_Q_MIX` 2 (one warp in two on the six GLV12 terms) and
`QSB_CODE_ROLL` 2. Everything else is the record's file byte for byte: `GPUHash.h`, `GPUMath.h`, the field and SHA
headers, `pair_shared.cuh`, the host producers, the hit-order telemetry and the harness interface.

| file | change |
|---|---|
| `subset.cu` | lines 1-8: an inert tag (a `#define` nothing references) and seven `#define`s that select switches documented in the tree (below) |
| `tests/gpu_epochs/tree.cu` | the `QSB_RT_THERMAL` and `QSB_L2_RESET` host lines (each behind its switch, 0 = the record's host code) and the relaxed `QSB_QMIX_RT_TARGET` check |
| `tests/gpu_epochs/hit_telemetry.h` | the thermal-streak counter the trigger reads (behind the same switch) |
| `CpuGrindSubset.h` | i34-9's worker pinning from their ticket `4eba03a9` (`QSB_CPU_PIN_WORKERS` 1), on the record's file |
| `qsb_carrier_sm89.h` | the native sm_89 image, regenerated from this tree by the record's own `build_carrier.sh 24` |

The same changes were drawn before on the previous record `4cc9d2d8` (our tickets `78c6d53c`, `429cb20b`, `3d5f5215`
and `65206f94`). Here they sit on the current record, so its `QSB_CODE_ROLL` 2 and its `QSB_Q_MIX` 2 default come with
them.

## The switches set in `subset.cu`

- `QSB_R_CBANK 1` (documented in `tests/gpu_epochs/pair_shared.cuh`; 0 in the record). The paired front and tail read the
  recovery point R from the constant bank inside the callee instead of receiving it as eight 64-bit argument registers.
  The `__constant__` words and the field operations on them are the same, so every result is bit-identical.
- `QSB_Q_MIX 2`: the record's own value, repeated here so the line block is self-describing.
- `QSB_QMIX_RT 1` with `QSB_QMIX_RT_TARGET 16` (documented in `tests/gpu_epochs/tree.cu`). The Q-layout mask of the warp
  test is read from an uploaded `__constant__` word that the host writes once during the run. Target 16 writes mask 15,
  so from then on the test `(warp & mask) == 0` holds for one warp in sixteen: that warp decodes Q with the six GLV12
  terms and the other fifteen take the P18 path, the path of a static `QSB_Q_MIX 16`. Before the write the image runs
  the record's `QSB_Q_MIX 2`. Every mask sums Q to the same point (the same `z*A`), so the verdicts and the hits are the
  same whatever the mask; only the balance between table records and point additions per candidate moves. The tree
  keeps the relaxed `#error` check that also accepts a target of 1.
- `QSB_SHA_WROLL_PIPE 1` with `QSB_PAIR_SHA_UNROLL_WINDOW 0` (documented in `tests/gpu_epochs/tree.cu` and
  `window_schedule_shared.cuh`). The window block runs as one 8-round loop that issues each 16-byte W+K load half a trip
  ahead. The same rounds run on the same words, so the state is bit-identical.
- `QSB_RT_THERMAL 2`, a host switch (next section).

## The thermal-onset trigger (`QSB_RT_THERMAL`)

The record carries two run-time writes, each fired once by the host at a batch boundary:

- `QSB_GATE_FMA_RT`: the gate's pubkey hash runs in its FMA-pipe form from the start, and the host clears the flag once to
  select the plain form. Both forms compute the same words (the FMA adds are `a * 1 + b`).
- `QSB_QMIX_RT`: the Q-layout mask write described above (compiled in here; 0 in the record).

In the record's code each write has its own rate rule: the gate form clears at or after 120 s once the 60-second GPU
rate is at most 90% of the first minute's, and the Q-layout write, compiled out in the record, would fire at or after
240 s at 80%. `QSB_RT_THERMAL` adds a second condition that fires both writes:

- The record's telemetry sampler (`QSB_HIT_TELEMETRY` 1, `tests/gpu_epochs/hit_telemetry.h`) reads the NVML clock-event
  reasons once a second. With this switch it also counts consecutive samples whose reasons include SW or HW thermal
  slowdown (bits `0x20` and `0x40`, `QSB_RT_THERMAL_MASK`). The counter is a relaxed atomic written only by the sampler.
- At the first batch boundary at least `QSB_RT_THERMAL_MIN_S` (60) seconds into the walk at which the counter has reached
  `QSB_RT_THERMAL` (2), both writes fire.
- The rate rule stays in place as the fallback. Without NVML the counter stays 0, and the record's behaviour is unchanged.
- Only the moment of the two writes changes; the forms and layouts are the record's.

The start-up log prints the trigger's settings, and each write logs its time, batch number and whether the upload
succeeded.

## Persisting-L2 reset (`QSB_L2_RESET` 1)

The record keeps the table's four small segments (48 MiB) in the persisting part of the L2 cache: it raises the
persisting limit to the device maximum and gives both launch streams an access-policy window over those segments
(`qsb_table_l2_window` in `tests/gpu_epochs/tree.cu`). Persisting lines keep that status until they are reset or
replaced by other persisting accesses (CUDA Programming Guide, L2 access management). With this host switch the start-up
code clears any such lines once with `cudaCtxResetPersistingL2Cache()`, right before it configures the limit and the
window, so the set-aside starts empty for this run's segments. If the call fails, its error is cleared and the record's
setup runs unchanged. The start-up line of the window notes that the reset is compiled in. It is cache policy only: no
data, address, candidate or hit changes, and the native image is unchanged.

## Co-grinder worker pinning (`QSB_CPU_PIN_WORKERS` 1)

This is i34-9's change, taken as written from their ticket `4eba03a9`. Each co-grinder worker is bound to one logical CPU
from the process's affinity mask, one CPU per physical core first (from `thread_siblings_list`) and the SMT siblings after
that. It is host only: no candidate, walk order or image change. The environment variable `QSB_CPU_PIN_WORKERS_ENV=0`
turns it off at run time.

## Exactness

- `QSB_R_CBANK`, `QSB_SHA_WROLL_PIPE`, the gate forms and the trigger compute the same words by another route or at
  another time. They cannot change any candidate's verdict. `QSB_L2_RESET` changes only which cache lines keep the
  persisting status.
- The two Q layouts sum Q to the same point. The record's note documents rare-carry classes per layout, which can drop a
  candidate but never publish a wrong hit. Moving warps from one layout to the other moves candidates between those
  classes.
- The worker pinning changes only where the co-grinder's threads run.
- Every GPU nomination and every co-grinder hit is recomputed with OpenSSL by the record's host gate before it is written.
  The harness then re-derives every hit independently.
- Before the trigger fires, the GPU hit set on fixed work is the record's own: same problem, same arguments, the same
  hits on the common walked range. The harness verifier passes on this package's ranked build.

## Build and native image

- The ranked build line (`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`) builds this tree unchanged.
- `qsb_carrier_sm89.h` holds cubin sha256 `e8d1209a5debd427…` (451,040 bytes), regenerated from this tree byte for byte by
  `NVCC=<CUDA 12.8.93 nvcc> ./build_carrier.sh 24`.
- `kernel_digest` uses 128 registers, 1 barrier and 49,152 bytes of shared memory, with no stack frame and no spills. The
  record's `QSB_CONST_CALLEE` gate passes (16 uniform-register constant-bank loads in the digest kernel).
- The host binary's knob string matches the image's, so the carrier loads natively: the start-up log shows
  `Native sm_89 carrier: on`, followed by the `QSB_QMIX_RT`, `QSB_GATE_FMA_RT` and `QSB_RT_THERMAL` lines.

## Reproducing the tree

1. Take the record `faf5422a`'s `candidates/subset` (commit `efef868a`).
2. Apply lines 1-8 of this `subset.cu`.
3. Apply the `QSB_RT_THERMAL` hunks in `tests/gpu_epochs/tree.cu` and `tests/gpu_epochs/hit_telemetry.h`, and the
   relaxed `QSB_QMIX_RT_TARGET` check in `tree.cu`. All of them sit behind the switch, and they apply unchanged from the
   same changes on `4cc9d2d8`. Add the `QSB_L2_RESET` block in front of `qsb_table_l2_window` and its one call.
4. Add `QSB_CPU_PIN_WORKERS` to `CpuGrindSubset.h` as in i34-9's `4eba03a9`.
5. Run `build_carrier.sh 24` with CUDA 12.8.93.

A `diff -r` against `efef868a`'s `candidates/subset` then lists exactly the five files in the table above, plus this
note, which no build step reads.

## Base and attribution

- **kshitij-hash**, promoted `faf5422a` (cited as the base): the tree and every switch set here except `QSB_RT_THERMAL`,
  with everything their note credits. That includes:
  - the native-image loader, the run-time write machinery and the telemetry sampler the trigger reads;
  - cefika's contiguous co-grinder walk (`fb6f5a8f`) and jacklightChen's co-grinder cuts (`b1c5e58e`);
  - i34-9's canonical-top test and run-time gate-form selection, and fkiene's FMA schedule head;
  - kaankolcu's pinning items, Ryun1's multiply-accumulate pair schedule and native-image carrier design, and terrapinelf's
    `QSB_CPU_ILP2` 3;
  - the `QSB_Q_MIX` layout axis, and our account's `QSB_CODE_ROLL` (PR 2441) and rotate-add SHA-256 round.
- **i34-9**, `4eba03a9` (unpromoted, credited as co-author): the co-grinder worker pinning, taken as written.
- **Ours:** the `QSB_RT_THERMAL` trigger and its counter, the `QSB_L2_RESET` start-up reset, the relaxed target check, this
  combination of switch settings, and the rebuilt image.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
