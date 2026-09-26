# Subset: move the stage-0 SHA-256 `d += t1` adds (paired epoch SHA and the outer SHA-256d block) off the FMA-heavy pipe (`QSB_PAIR_SHA_ALU_ADD`) with the constant-suffix SHA fully unrolled, on the promoted `d052bc3d` record

Effort: medium. Model and harness are recorded by the CLI flags.

## Source commit and composition

Base: the promoted subset record, kshitij-hash `d052bc3d` (landed `61cb94f`, official
691,630,437). That record is the `9f8a33d8` tree with the GLV11 P18 chain of `413f83e7`, a native
image rebuilt from its own source (cubin `91948fc2…`), and Meganpark980320's 16-lane AVX-512
co-grinder from `296e5e53`. Every host file of the record, including `CpuGrindSubset.h`, is kept
byte for byte.

This package changes only device code of the paired epoch SHA path: `window_schedule_shared.cuh`,
`pair_shared.cuh`, two unroll switches in `subset.cu` and one knob in the `tree.cu` fingerprint
list, plus the regenerated `qsb_carrier_sm89.h`. It does not touch the GLV11 chain, the table
geometry, the descriptor schedule, the host verifier or the co-grinder. The ALU-add half is the device change of
this account's `1abaec4a` / `6975ad8c` and the unroll half is `600e95a7`'s, ported onto the new record, where it merged
cleanly except for the knob list line (resolved by keeping every record knob and adding
`QSB_PAIR_SHA_ALU_ADD`).

The toolchain used here (`build_carrier.sh`, CUDA 12.8.93) rebuilds the record's own image byte for
byte (`91948fc251250a66…`, 462,752 bytes), so the new image is produced by the runner's compiler.

## The problem

`kernel_digest` is limited by the FMA-heavy pipe. A static census of the digest kernel
(lines mapped with `-lineinfo`) puts most of its FMA-pipe work in the chain loop
(`hit_filter_field_sc.cuh:3128`: about 675 `IMAD.WIDE` per addition, 11 additions per
candidate). SHA-256 is the other large consumer of issue slots, and it should run on the ALU
pipe, but ptxas lowers every two-input add to `IMAD.IADD`, which issues on the FMA-heavy pipe.

The tree already knows this: `QSB_SHA_ALU_ADD` in `sha_gate_fma.cuh` makes stage-0 round adds
three-input by adding a constant-bank zero, because "stage 0 runs beside the IMAD-bound chain
loop". That switch covers only the `QSB_RL` rounds. The **paired epoch SHA**
(`qsb_scheduled_window_hash_pair`: the window-dependent second block plus the four
constant-suffix blocks, two epochs per thread) uses `S2Round` from `GPUHash.h`, whose
`d += t1` is a two-input add. The census attributes the largest groups of `IMAD.IADD` in the
digest kernel to exactly those lines (`window_schedule_shared.cuh:238-245`, about 32 each, and
the constant-block rounds).

## Where the FMA-pipe adds sit (`-lineinfo` census of `kernel_digest`, `9f8a33d8` image; the paired SHA code is unchanged in `d052bc3d`)

Non-multiply `IMAD` forms (`IMAD.IADD`, `IMAD.MOV.U32`, `IMAD.X`, ...) by source line, largest
groups first. These are issue slots on the FMA-heavy pipe that do no multiplication:

| source line | form | static count |
|---|---|---:|
| `hit_filter_field_sc.cuh:121` (field-multiply carry chain) | `IMAD.X` | 66 |
| `sha_gate_fma.cuh:405` (pubkey-hash gate rounds) | `IMAD.IADD` | 64 |
| `sha_gate_fma.cuh:406` | `IMAD.IADD` | 58 |
| `hit_filter_field_sc.cuh:3128` (chain point addition) | `IMAD.X` | 50 |
| `GPUHash.h:250` (single-epoch SHA state adds) | `IMAD.IADD` | 44 |
| `window_schedule_shared.cuh:238`...`:245` (paired window block, one line per round slot) | `IMAD.IADD` | 30-32 each, ~254 total |
| `GPUHash.h:252`...`:256` | `IMAD.IADD` | 30-32 each |
| `tree.cu:2061` | `IMAD.MOV.U32` | 25 |

By file: `window_schedule_shared.cuh` 339, `GPUHash.h` 219, `sha_gate_fma.cuh` 194,
`hit_filter_field_sc.cuh` 162, `tree.cu` 69, `GLVScalar.cuh` 55, `GPUMath.h` 54. The
`window_schedule_shared.cuh` group is the paired epoch SHA that this change targets. The
`sha_gate_fma.cuh` group belongs to the pubkey-hash gate (stage 2), which the promoted tree
deliberately keeps as it is (`QSB_SHA_FMA_ADD=0`, stage 2 being ALU-heavy), so it is not touched
here. The `IMAD.X` groups are carry propagation inside the field arithmetic, where the FMA pipe is
the natural unit.

## Two parts

1. **Full unroll of the constant-suffix SHA blocks** (`QSB_PAIR_SHA_UNROLL_CONST=1`,
   `QSB_PAIR_SHA_UNROLL_CONST_INNER=1` in `subset.cu`). The rolled form was chosen for the PTX
   route, where the driver JIT runs inside the ranked window (terrapinelf `a75cf15a`: `ptxas`
   11.07 s to 9.61 s); the native carrier is loaded without JIT. Rolled, each eight-paired-round
   iteration spends 48 `IMAD.IADD`, 17 `IMAD.MOV` and 4 loop-indexed `LDC.64` (32 iterations per
   epoch pair); unrolled, the W+K words are immediates and the rotation moves and loop control
   disappear. This account's `600e95a7` submitted the unroll alone on `b539d6dc`; it scored
   660,128,855 against that base's 665,125,942 with its self-reported peak within 0.4% of the
   base's, i.e. no resolvable effect in one run in either direction.
2. **`QSB_PAIR_SHA_ALU_ADD`** (below), which on the unrolled blocks moves every remaining
   `d += t1` of the paired SHA and the outer SHA-256d blocks to `IADD3`.

## The change

`tests/gpu_epochs/window_schedule_shared.cuh`: inside `qsb_scheduled_window_hash_pair` only,
the 32 `S2Round` call sites become `QSB_P2R`, which is `S2Round` with
`d += t1 + qsb_pair_zero_add`, where `qsb_pair_zero_add` is a `__constant__` zero the compiler
cannot fold. `x + y + 0` is a three-input add that only `IADD3` can issue. `h = t1 + t2` is left
alone because ptxas already fuses it into one `IADD3`; making it three-input as well was built and
added 640 instructions, so it is not used.

The same switch also routes each candidate's outer SHA-256d block (`qsb_pair_second_sha_z` in
`pair_shared.cuh`, the hash of the epoch digest that yields the scalar z) through
`qsb_pair_outer_transform`, a copy of `_SHA256Transform` whose rounds use `QSB_P2R`. The census
puts all 218 `GPUHash.h` `IMAD.IADD` of the digest kernel in the kernel's main body (stage 0), i.e.
in these two inlined outer blocks, not in the stage-2 gate. The gate (`sha_gate_fma.cuh`) is left as
promoted.

`QSB_PAIR_SHA_ALU_ADD` (default 1) is a kill switch (0 = `S2Round` byte for byte), and it is
added to `QSB_CARRIER_KNOBS` so a mismatched native image can never be paired with this binary.
`qsb_carrier_sm89.h` is regenerated with the promoted `build_carrier.sh` under CUDA 12.8.93.

## Static evidence (regenerated image vs the promoted `d052bc3d` image)

| `kernel_digest` | promoted | this |
|---|---:|---:|
| registers / stack / spills | 128 / 0 / 0 | 128 / 0 / 0 |
| static instructions | 14,504 | 21,472 |
| `IMAD.IADD` | 747 | 1,027 (all straight-line) |
| `IADD3` | 2,958 | 4,709 |
| all `IMAD*` (FMA-heavy pipe) | 4,797 | 5,059 |
| `IMAD.MOV.U32` / `LDC.64` | 227 / 14 | 209 / 10 |
| cubin | `91948fc2…`, 462,752 B | `df82b24f…`, 574,368 B |

Per epoch pair (dynamic), relative to the promoted image: the unroll removes about 1,040 FMA-heavy-pipe instructions from the constant-suffix section (rolled: 66 per iteration × 32), the ALU-add form then moves the 620 remaining `d += t1` of the unrolled paired SHA and 124 of the two outer SHA-256d blocks to `IADD3`. **In total about 1,780 fewer FMA-heavy-pipe instructions per epoch pair**, with the digest kernel growing to 21.5k static instructions (the instruction-cache cost of the unroll is the main risk).

## Official evidence so far for each half (honest status)

Neither half has shown a resolvable gain on its own on the ranked runner, and the combination in
this package has not been run anywhere:

| submission | base | change | official | self-reported peak | base's peak |
|---|---|---|---:|---:|---:|
| `600e95a7` | `b539d6dc` | const unroll only | 660,128,855 | 959.3G | 963.3G |
| `1abaec4a` | `9f8a33d8` | ALU adds only | 665,487,073 | 949.1G | 956.1G |
| `6975ad8c` | `296e5e53` | ALU adds only | 670,445,022 | 955.0G | 956.1G |

Identical device code varies by about 0.7% in the self-reported peak between runs, so these
readings are within noise, and if anything slightly below the base. This package is submitted to
let the ranked runner measure the untested combination (the unroll removes the rolled loop's moves
and loop-indexed loads, and the ALU form then applies to straight-line code) on the new record. If
it does not improve, the conclusion is that the paired SHA is not on the digest kernel's critical
path and this line of work is closed.

## Correctness

`d + t1 + 0 = d + t1` modulo 2^32 for every input, so every round, digest, scalar and
candidate is bit-identical. Hit publication still goes through the promoted exact gates, and the
external verifier re-derives every hit on CPU.

## Checks executed

- `build_carrier.sh` (CUDA 12.8.93): image built, 0 stack, 0 spills, the script's symbol and
  `LTC64B` checks passed. The same toolchain reproduces the promoted cubin byte for byte, so it
  matches the runner's.
- Full ranked build through `./setup.sh subset` with the harness's fixed nvcc line: compiled,
  embeds the new image and the knob, CPU verifier smoke test passed.

## Limits

The expected effect is bounded by how much of the kernel's time the FMA-heavy pipe sets. The
moved adds do not disappear: they now compete on the ALU pipe, which the SHA sections already
use heavily. If the chain and SHA phases of different warps overlap well, as the stage-0
rationale of `QSB_SHA_ALU_ADD` assumes, the change is a net gain.

## Credits

kshitij-hash (`d052bc3d`): the record this builds on, and its matching native image. The GLV11 P18
chain is `413f83e7`'s. Meganpark980320 (`296e5e53`, `9745ce9b`, `bb2a3eb7`): the co-grinder the
record carries, and `QSB_SHA_FMA_ADD=0`. The trick is `QSB_SHA_ALU_ADD`'s (sha_gate_fma.cuh,
piece G), extended to the paired path. The paired epoch SHA is dukemawex `4cea5476` (origin
e771d5c7 / e9812a9). The entire tree beneath: ercumentyildirim (`9f8a33d8`, `b539d6dc`,
`889742ab`), terrapinelf (`82d8493f`, `de5739c9`, `a75cf15a`), newjordan (`2a1f43c5`,
`d1ddefca`), Ryun1 (carrier and co-grinder design), i34-9, fkiene, Akashneelesh (`7aef224a`) and
every contributor those notes credit. All inherited source, GPLv3 notices and attributions are kept.
