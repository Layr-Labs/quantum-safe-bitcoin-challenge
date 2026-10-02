# Subset: the record `4cc9d2d8` (kshitij-hash) with its own `QSB_Q_MIX` 2 and two host-only switches from open submissions: ercumentyildirim's `QSB_RT_THERMAL` 2 and jungjipdo's `QSB_HP_TAIL_BUCKETS` 1

Prepared with Claude Opus 5.5 in Claude Code. This host has no GPU. The device code is the record's with one of its own switches set: `QSB_Q_MIX` 2 instead of 4. The native image is rebuilt for it. The two host changes are other contributors' published code, ported unchanged.

## What this package is

Every file is the record `4cc9d2d8` (benchmark commit `2f57d80`), except for four files. The code in them comes from two open submissions:

| file | change | source |
|---|---|---|
| `subset.cu` | sets `QSB_Q_MIX` 2 and `QSB_RT_THERMAL` 2 in place of the record's inert tag line | — |
| `qsb_carrier_sm89.h` | the native image rebuilt for `QSB_Q_MIX` 2 by the record's own `build_carrier.sh 24` | — |
| `tests/gpu_epochs/tree.cu` | the `QSB_RT_THERMAL` switch (default 0) | ercumentyildirim, `2a5fdb8f` |
| `tests/gpu_epochs/hit_telemetry.h` | the thermal streak counter that `QSB_RT_THERMAL` reads | ercumentyildirim, `2a5fdb8f` |
| `tests/gpu_epochs/host_producers_v3.h` | `QSB_HP_TAIL_BUCKETS` (default 1 in its source) | jungjipdo, `ed2137f5` |

`2a5fdb8f` also sets `QSB_R_CBANK` 1, which is not taken here.

## `QSB_Q_MIX` 2 (the record's device switch)

The record lays out the GLV terms of Q per warp. At 4, one warp in four decodes Q with the six GLV12 terms; at 2, one warp in two does. Both layouts sum Q to the same point, so every candidate's z·A and the hit set are unchanged. The public ranked draws `31edf1f2` (terrapinelf), `2a5fdb8f` (ercumentyildirim) and `3616a5fb` (i34-9) run the record with `QSB_Q_MIX` 2.

## `QSB_RT_THERMAL` 2 (host only, not an image knob)

The record runs the gate's pubkey hash in its FMA form from the start. It switches once to the plain form (`QSB_GATE_FMA_RT`) when the 60 s GPU rate has fallen to 90% of the first minute's, after 120 s at the earliest.

- **What the switch adds:** the same one write also fires at the first batch boundary after `QSB_RT_THERMAL_MIN_S` (60 s). The condition is that the record's NVML sampler (`QSB_HIT_TELEMETRY`) has seen thermal slowdown in at least 2 consecutive 1 s samples. Thermal slowdown here means the clock-event reasons SW thermal slowdown 0x20 or HW thermal slowdown 0x40 (`QSB_RT_THERMAL_MASK`).
- **Fallback:** the rate rule stays. When NVML is not usable, the streak stays 0 and the record's behaviour is unchanged.
- **Exactness:** both gate forms compute the same verdicts, so the hit set is unchanged.
- **Unaffected:** `QSB_QMIX_RT`, which the same condition also feeds, is 0 in the record and stays 0.

## `QSB_HP_TAIL_BUCKETS` 1 (host only)

The host producers hash the epochs' suffix blocks four epochs at a time. Before this change, one group of four could mix different suffix lengths, so the shorter lanes kept compressing until the longest one ended.

- **Grouping:** with the switch, each epoch is queued by its count of complete suffix blocks. A group is flushed as soon as four epochs with the same count are queued, through the existing flush (`sha_pre4` when all four counts are equal).
- **Leftovers:** at the end of a chunk, at most three leftovers per count drain through the original flush. Counts outside the table go to the existing immediate path.
- **Exactness:** descriptors are written by epoch index, and every SHA input is the same, so the descriptors and first-block states are the same as before. No epoch is dropped and the search range is unchanged.
- **Safety net:** the record's start-up check stays. It builds batch zero with both the host and the GPU producers and compares every descriptor and every used first-state word. A mismatch disables host production.

## Checks done here

All builds ran in an amd64 container with CUDA 12.8.93, the ranked runner's toolkit.

| check | result |
|---|---|
| `build_carrier.sh 24` | cubin sha256 `21456a67b163d52a...`, 516,960 bytes; 0 spills; the `QSB_CONST_CALLEE` gate passes (16 UR-indexed `c[0x3]` loads); the image's knob string carries `QSB_Q_MIX=2` |
| image `qsb_carrier_knobs` against the host binary's `QSB_CARRIER_KNOBS` | byte-identical (3,165 bytes), so the native image loads |
| the harness's build line (`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`) | builds with no errors; the binary carries the `RT_THERMAL: on` start-up line |
| the two patches against the record | both apply cleanly to `2f57d80` |

I have no GPU here, so these changes were not run end to end on a card. The record's start-up check guards the producer change at run time, and the gate change only moves the time of a switch that the record already makes.

## Kill switches

- `-DQSB_RT_THERMAL=0` (or delete the line in `subset.cu`): the record's rate rule only.
- `-DQSB_HP_TAIL_BUCKETS=0`: the record's producer grouping.
- These two are host-only and not in the knob list, so the native image is unaffected either way.
- `-DQSB_Q_MIX=4` with `NVCC=<CUDA 12.8 nvcc> ./build_carrier.sh 24`: the record's Q layout and image (`cef81a9f...`).

## Reproducing

```
./setup.sh subset
./benchmark.sh subset
```

The start-up log prints the `RT_THERMAL: on` line. When the switch fires, the `GATE_FMA_RT` line carries `(thermal slowdown seen)`.

## Base and attribution

- **kshitij-hash** (co-author): the record `4cc9d2d8` in full, including `QSB_Q_MIX`, `QSB_GATE_FMA_RT` and the NVML sampler that `QSB_RT_THERMAL` builds on.
- **ercumentyildirim** (co-author): `QSB_RT_THERMAL` (`2a5fdb8f`).
- **jungjipdo** (co-author): `QSB_HP_TAIL_BUCKETS` (`ed2137f5`).
- **terrapinelf, i34-9** (co-authors): public ranked draws of the record with `QSB_Q_MIX` 2 (`31edf1f2`, `3616a5fb`).
- **Credited through the record** (co-authors, up to Yukon's limit): fkiene, kaankolcu, HyeokxC, jacklightChen, newjordan. Every contributor named in the record's note is credited through it, including cefika for the contiguous co-grinder walk the record keeps. The GPU arithmetic headers derive from VanitySearch (GPLv3, `COPYING`); the co-grinder's field and scalar code follow libsecp256k1 (MIT, `COPYING-secp256k1`).
- **Mine:** the composition and the checks above.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
