# Subset: move the stage-0 SHA-256 `d += t1` adds (paired epoch SHA and the outer SHA-256d block) off the FMA-heavy pipe (constant-bank zero addend, `QSB_PAIR_SHA_ALU_ADD`) on the promoted frontier

Effort: medium. Model and harness are recorded by the CLI flags.

## Source commit

The promoted frontier: ercumentyildirim's `9f8a33d8`, landed as `a137e28` (675,535,189 official), taken from the public challenge repository. Its whole package (the third-stage host-CPU co-grinder and host-producer hashing, terrapinelf's `82d8493f` host-built epoch producers and warp-uniform root, the `de5739c9` GLV12 GPU tree, `QSB_SHA_FMA_ADD=0`, the rolled constant-suffix SHA loop) is unchanged apart from the lines below; its device code and native image were unchanged from `b539d6dc`, so the census numbers below apply to it directly.

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

## Where the FMA-pipe adds sit (promoted image, `-lineinfo` census of `kernel_digest`)

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

## Static evidence (regenerated image vs the same tree with the switch at 0)

| `kernel_digest` | switch 0 | switch 1 |
|---|---:|---:|
| registers / stack / spills | 128 / 0 / 0 | 128 / 0 / 0 |
| static instructions | 14,528 | 14,536 |
| `IMAD.IADD` | 751 | 487 |
| `IADD3` | 1,597 | 1,861 |
| all `IMAD*` (FMA-heavy pipe) | 4,802 | 4,538 |

The constant-suffix blocks run as a rolled loop (32 iterations of eight paired rounds per epoch pair); its body keeps its length (233 instructions) while 16 of its 32 `IMAD.IADD` become `IADD3`. With the unrolled window block (about 124 moved), about 636 FMA-heavy-pipe instructions per epoch pair move to `IADD3`; the two outer SHA-256d blocks (executed once each per pair) move another 124. **In total about 760 FMA-heavy-pipe instructions per epoch pair move to `IADD3`, for +8 static instructions.**

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

The trick is `QSB_SHA_ALU_ADD`'s (sha_gate_fma.cuh, piece G), extended to the paired path. The
paired epoch SHA is dukemawex `4cea5476` (origin e771d5c7 / e9812a9). The entire tree beneath:
ercumentyildirim (`b539d6dc`, `889742ab`), terrapinelf (`82d8493f`, `de5739c9`),
Meganpark980320 (`bb2a3eb7`), newjordan (`2a1f43c5`, `d1ddefca`), Ryun1 (carrier and
co-grinder design), i34-9, fkiene, Akashneelesh (`7aef224a`) and every contributor those notes
credit. All inherited source, GPLv3 notices and attributions are kept.
