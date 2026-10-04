# Subset: the record `faf5422a` (kshitij-hash) with two of its own exact device switches on, a larger co-grinder batch on shared cores, and a per-worker batch split (new here)

Prepared with Claude Opus 5.5 in Claude Code. This host has no GPU. The device code is the record's with two of its own switches set to 1, and the native image is rebuilt for them. The host changes are four file-scope switches; one of them is new, and one sets the record's batch constant.

## What this package is

Every file is the record `faf5422a` (benchmark commit `efef868`), except `subset.cu`, `CpuGrindSubset.h` and the rebuilt `qsb_carrier_sm89.h`.

| switch | kind | change | source |
|---|---|---|---|
| `QSB_SHA_WROLL_PIPE` 1 | device | the window block as one 8-round loop, each W+K load issued half a trip ahead | the record's (kshitij-hash) |
| `QSB_R_CBANK_TAILS` 1 | device | R's words come from the constant bank in both tails | the record's (kshitij-hash) |
| `QSB_HIT_TELEMETRY` 0 | host | no runtime telemetry in the hit order | the record's switch |
| `QSB_CPU_DIAG_EPOCH` 0 | host | the co-grinder's ranges start at epoch 0 instead of at a diagnostic code | the record's switch |
| `QSB_CPU_BATCH` 4096 | host | the co-grinder batch used when workers share cores (the record: 1,024) | the record's constant |
| `QSB_CPU_BATCH_ODD` 8192 | host | odd-numbered co-grinder workers use 8,192-candidate batches; even ones use `QSB_CPU_BATCH` | new here (first in `171d06c5`) |

The two device switches with these two host settings are the combination terrapinelf drew as `5c27aeab` and its redraws. With them, the rebuilt image is byte for byte the one in `5c27aeab` and `9405f3fd` (`5554da9f…`).

`subset.cu` also drops the record's inert tag line `QSB_REDRAW_09260102`, which nothing reads.

## The two device switches

Both are kshitij-hash's, documented in the record's own source as bit-identical. Their device code is unchanged here; only their value in `subset.cu` changes from 0 to 1.

- **`QSB_SHA_WROLL_PIPE` 1** (`tree.cu`): the SHA-256 window block runs as one 8-round loop instead of fully unrolled, with each round's W+K load issued half a trip ahead of its use. The record pairs it with `QSB_PAIR_SHA_UNROLL_WINDOW` 0. The same rounds run on the same words in the same order, so every digest is unchanged. The effect is on code size and load scheduling only.
- **`QSB_R_CBANK_TAILS` 1** (`tree.cu`, `pair_shared.cuh`): the two tails (`qsb_pair_tail3_value`, `qsb_pair_finish3_value`) read the original R from the constant bank inside their callee instead of taking it as eight 64-bit ABI arguments. `kernel_digest` then no longer reloads R after the tree and holds its 16 registers across both tails. The front keeps its argument list. In the source's words: same `__constant__` words, same field operations, in the same order.

Both switches are in the image's knob list, so they need the rebuilt native image; `build_carrier.sh 24` builds it from this tree's own source.

## `QSB_CPU_BATCH_ODD` (host only)

The record's co-grinder picks one batch size at start-up: 1,024 when two workers can share a physical core, 4,096 when none can (`QSB_CPU_BATCH_AUTO`).

- **Change:** with `QSB_CPU_BATCH_ODD` non-zero, the odd-numbered workers use that size instead. Each worker sizes its own buffers from its batch, so nothing else changes.
- **Exactness:** each worker still walks its own contiguous epoch range in the same order. Only how many candidates share one batched inversion changes, so the set of candidates searched and every hit are unchanged. Every hit still passes the exact OpenSSL gate.
- **Reading it:** each worker's range starts at t × span (with `QSB_CPU_DIAG_EPOCH` 0), so the public hit list shows how far each worker got. The odd and even workers can therefore be compared within one run.

## The batch sizes in this draw

The record's batch rule (`batch_choose`, 86c643ae's) keeps `QSB_CPU_BATCH` whenever SMT siblings are among the workers' CPUs, and moves to `QSB_CPU_BATCH_SOLO` (4,096) only when every worker has a core to itself. Its source cites a rival's measurement on a Zen 4 EPYC with SMT off: 2,048 at +1.9 %, 4,096 at +3.5 % co-grinder rate over 1,024. The record's comment on `QSB_CPU_BATCH` gives the reason for 1,024: both SMT threads' EC state (2 x 0.25 MB) and the prefetched rows stay in the core's 1 MB L2.

- **`QSB_CPU_BATCH` 4096:** every worker uses 4,096, the size my previous draw `3d96fe49` ran on half the workers (after 2,048 in `171d06c5`). It is also the record's size when cores are not shared.
- **`QSB_CPU_BATCH_ODD` 8192:** the odd workers go one step further, to the largest batch the code accepts. The rival's SMT-off measurement cited above has 8,192 slightly behind 4,096, so this half of the draw tests where the curve turns with shared cores.
- **Memory:** a worker's batch buffers grow with its batch (points, field elements, digits and z words, a few hundred bytes per candidate), so 8,192 adds some tens of MiB in all. The 9-window table and its huge pages are untouched.
- **Topology:** consecutive worker numbers may sit on the two SMT siblings of one core; such a pair then runs one batch of each size. Nothing in the walk or in the hit gate depends on which worker runs which size.

## What does not change

- The GPU's search, its 128 patterns and its epoch order; the co-grinder's patterns (the complement subset) and its contiguous walk.
- The start-up self-checks, the host producers, the gate-form switch (`QSB_GATE_FMA_RT`) and the exit path.
- The exact OpenSSL gate on every published hit, so a faulty change could lose hits but could never publish a wrong one.

## Checks done here

All builds ran in an amd64 container with CUDA 12.8.93, the ranked runner's toolkit.

| check | result |
|---|---|
| `build_carrier.sh 24` | cubin sha256 `5554da9f3ba62973…`, 451,168 bytes; 0 spills, no stack frame in `kernel_digest`; the `QSB_CONST_CALLEE` gate passes (16 UR-indexed `c[0x3]` loads) |
| image knob string against the host binary's `QSB_CARRIER_KNOBS` | byte-identical (3,182 bytes), so the native image loads. The three host switches are not in it |
| the harness's build line (`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`) | builds with no errors |
| `QSB_CPU_BATCH_ODD` | a `static_assert` limits it to 0 or a multiple of 32 in 32..8192, the same range the record's run-time batch override accepts |

I have no GPU or Zen 4 host here, so nothing was timed or run end to end.

## Kill switches

- `QSB_CPU_BATCH_ODD` 0 in `subset.cu`: every worker on `QSB_CPU_BATCH`.
- `QSB_CPU_BATCH` 1024 (or delete its line in `subset.cu`): the record's batch.
- `QSB_SHA_WROLL_PIPE` 0 and `QSB_R_CBANK_TAILS` 0 in `subset.cu`, then `NVCC=<CUDA 12.8 nvcc> ./build_carrier.sh 24`: the record's image (`5f1f8111…`).

## Reproducing

```
./setup.sh subset
./benchmark.sh subset
```

## Base and attribution

- **kshitij-hash** (co-author): the record `faf5422a` in full, including both device switches, the telemetry and diagnostic switches and the co-grinder's batch rule.
- **terrapinelf** (co-author): the public draws of this device image with telemetry and the diagnostic start off (`5c27aeab`, `2e43819a`).
- **ercumentyildirim** (co-author): `QSB_CODE_ROLL`, which the record carries.
- **Credited through the record** (co-authors, up to Yukon's limit): fkiene, i34-9, kaankolcu, jacklightChen, HyeokxC, newjordan. Every contributor named in the record's note is credited through it, including cefika for the contiguous co-grinder walk. The GPU arithmetic headers derive from VanitySearch (GPLv3, `COPYING`); the co-grinder's field and scalar code follow libsecp256k1 (MIT, `COPYING-secp256k1`).
- **Mine:** `QSB_CPU_BATCH_ODD`, the batch choice, the composition and the checks above.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
