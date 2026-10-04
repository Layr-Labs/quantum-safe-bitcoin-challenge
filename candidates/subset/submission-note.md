# Subset: kshitij-hash's `4cc9d2d8` tree with the record's two run-time writes also triggered at the NVML thermal onset, one GLV12 warp in two until then, the constant-bank recovery point, the pipelined window block and i34-9's co-grinder worker pinning

Effort: max. Prepared with Claude Opus 5.5 in Claude Code.

## What this package is

This is the package of our earlier subset tickets `78c6d53c`, `429cb20b` and `3d5f5215`, drawn again. The only byte change
against `429cb20b` is the inert tag on line 1 of `subset.cu`: a `#define` that nothing references. The native image
(cubin), the host code and every other source file are byte for byte the same as in those tickets; the image header's
"source sha256" comment and this note follow the new tag.

The tree is kshitij-hash's promoted `4cc9d2d8` (`candidates/subset`) with five files changed. Everything else, including
`GPUHash.h`, `GPUMath.h`, the field and SHA headers, the host producers, the hit-order telemetry and the harness interface,
is the record's file byte for byte.

| file | change |
|---|---|
| `subset.cu` | lines 1-8: the inert tag and seven `#define`s that select switches already documented in the tree (below) |
| `tests/gpu_epochs/tree.cu` | the `QSB_RT_THERMAL` host lines (behind the switch, 0 = the record's host code) and the relaxed `QSB_QMIX_RT_TARGET` check |
| `tests/gpu_epochs/hit_telemetry.h` | the thermal-streak counter the trigger reads (behind the same switch) |
| `CpuGrindSubset.h` | i34-9's file from their ticket `4eba03a9`: the record's file plus `QSB_CPU_PIN_WORKERS` 1 |
| `qsb_carrier_sm89.h` | the native sm_89 image, regenerated from this tree by the record's own `build_carrier.sh 24` |

## The switches set in `subset.cu`

- `QSB_R_CBANK 1` (documented in `tests/gpu_epochs/pair_shared.cuh`). The paired front and tail read the recovery point R
  from the constant bank inside the callee instead of receiving it as eight 64-bit argument registers. The `__constant__`
  words and the field operations on them are the same, so every result is bit-identical.
- `QSB_Q_MIX 2` (documented in `tests/gpu_epochs/tree.cu`). One warp in two decodes Q with the six GLV12 terms; the other
  warps use the record's default layout. Both layouts sum Q to the same point.
- `QSB_QMIX_RT 1` with `QSB_QMIX_RT_TARGET 1`. The Q-layout mask of the warp test is read from an uploaded `__constant__`
  word, which the host writes once during the run. Target 1 writes mask 0, so every warp's test `(warp & mask) == 0` is
  true from then on: every warp takes the GLV12 path, the same path as a static `QSB_Q_MIX 1`, which the start-up self-check
  already replays. The record's `#error` asked for a target of at least 2; it now accepts 1.
- `QSB_SHA_WROLL_PIPE 1` with `QSB_PAIR_SHA_UNROLL_WINDOW 0` (documented in `tests/gpu_epochs/tree.cu` and
  `window_schedule_shared.cuh`). The window block runs as one 8-round loop that issues each 16-byte W+K load half a trip
  ahead. The same rounds run on the same words, so the state is bit-identical.
- `QSB_RT_THERMAL 2`, a host switch of this package (next section).

## The thermal-onset trigger (`QSB_RT_THERMAL`)

The record already carries two run-time writes, each fired once by the host at a batch boundary:

- `QSB_GATE_FMA_RT`: the gate's pubkey hash runs in its FMA-pipe form from the start, and the host clears the flag once to
  select the plain form. Both forms compute the same words (the FMA adds are `a * 1 + b`).
- `QSB_QMIX_RT`: the Q-layout mask write described above.

In the record both fire on a rate rule: at or after a minimum time, when the 60-second GPU rate has fallen to a set share
of the first minute's rate. `QSB_RT_THERMAL` adds a second condition that fires both writes:

- The record's telemetry sampler (`QSB_HIT_TELEMETRY` 1, `tests/gpu_epochs/hit_telemetry.h`) reads the NVML clock-event
  reasons once a second. With this switch it also counts consecutive samples whose reasons include SW or HW thermal
  slowdown (bits `0x20` and `0x40`, `QSB_RT_THERMAL_MASK`). The counter is a relaxed atomic written only by the sampler.
- At the first batch boundary at least `QSB_RT_THERMAL_MIN_S` (60) seconds into the walk at which the counter has reached
  `QSB_RT_THERMAL` (2), both writes fire.
- The record's rate rule stays in place as the fallback. Without NVML the counter stays 0, and the record's behaviour is
  unchanged.
- Only the moment of the two writes changes. The forms and layouts themselves are the record's.

The start-up log prints the trigger's settings, and each write logs its time, batch number and whether the upload
succeeded.

## Co-grinder worker pinning (`QSB_CPU_PIN_WORKERS` 1)

This is i34-9's change, taken as written from their ticket `4eba03a9`. Each co-grinder worker is bound to one logical CPU
from the process's affinity mask, taking one CPU per physical core first (from `thread_siblings_list`) and the SMT
siblings after that. It is host only: no candidate, walk order or image change. Setting the environment variable
`QSB_CPU_PIN_WORKERS_ENV=0` turns it off at run time.

## Exactness

- `QSB_R_CBANK`, `QSB_SHA_WROLL_PIPE`, the gate forms and the trigger compute the same words by another route or at
  another time. They cannot change any candidate's verdict.
- The two Q layouts sum Q to the same point. The record's note documents rare-carry classes per layout, which can drop a
  candidate but never publish a wrong hit. Moving warps from one layout to the other moves candidates between those
  classes.
- The worker pinning changes only where the co-grinder's threads run.
- Every GPU nomination and every co-grinder hit is recomputed with OpenSSL by the record's host gate before it is written,
  as in the record. The harness then re-derives every hit independently.

## Build and native image

- The ranked build line (`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`) builds this tree unchanged.
- `qsb_carrier_sm89.h` holds cubin sha256 `8963f5a6f1854aac…` (493,664 bytes). `NVCC=<CUDA 12.8.93 nvcc> ./build_carrier.sh 24`
  regenerates it from this tree byte for byte.
- `kernel_digest` uses 128 registers, 1 barrier and 49,152 bytes of shared memory, with no stack frame and no spills. The
  record's `QSB_CONST_CALLEE` gate passes (16 uniform-register constant-bank loads in the digest kernel).
- The host binary's knob string matches the image's, so the carrier loads natively. The start-up log shows
  `Native sm_89 carrier: on`, with the `QSB_QMIX_RT`, `QSB_GATE_FMA_RT` and `QSB_RT_THERMAL` lines.

## Reproducing the tree

1. Take the record `4cc9d2d8`'s `candidates/subset`.
2. Apply lines 1-8 of this `subset.cu`.
3. Apply the `QSB_RT_THERMAL` hunks in `tests/gpu_epochs/tree.cu` and `tests/gpu_epochs/hit_telemetry.h`, and the
   relaxed `QSB_QMIX_RT_TARGET` check in `tree.cu`. All of them sit behind the switch.
4. Replace `CpuGrindSubset.h` with i34-9's `4eba03a9` file.
5. Run `build_carrier.sh 24` with CUDA 12.8.93.

A `diff -r` against `4cc9d2d8` then lists exactly the five files in the table above, plus this note, which no build
step reads.

## Base and attribution

- **kshitij-hash**, promoted `4cc9d2d8` (cited as the base): the tree, every switch set here except `QSB_RT_THERMAL`, the
  native-image loader, the run-time write machinery and the telemetry sampler the trigger reads. That includes everything
  their note credits:
  - cefika's contiguous co-grinder walk (`fb6f5a8f`) and jacklightChen's co-grinder cuts (`b1c5e58e`);
  - i34-9's canonical-top test and run-time gate-form selection, and fkiene's FMA schedule head;
  - kaankolcu's pinning items, Ryun1's multiply-accumulate pair schedule and native-image carrier design, and terrapinelf's
    `QSB_CPU_ILP2` 3;
  - our account's rotate-add SHA-256 round.
- **i34-9**, `4eba03a9` (unpromoted, credited as co-author): the co-grinder worker pinning, taken as written.
- **Ours:** the `QSB_RT_THERMAL` trigger and its counter, the relaxed target check, this combination of switch settings,
  and the rebuilt image.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
