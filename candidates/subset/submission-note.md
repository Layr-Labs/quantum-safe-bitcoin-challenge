# Subset: fully unroll the paired constant-suffix SHA blocks on the native sm_89 carrier route (the rolled form was a PTX-JIT-time choice that the carrier made obsolete)

Effort: medium. Model and harness are recorded by the CLI flags.

**Status of the evidence, stated up front:** this change was compiled, register/spill-checked
and statically censused on the regenerated sm_89 image, and the full ranked build passed the
verifier smoke test, but **it was not timed on a GPU before submission**. The official run is
its first measurement. Nothing below claims a speed-up; it states the hypothesis and the
static evidence for it.

## Source commit

The promoted frontier: ercumentyildirim's `b539d6dc`, landed as `1968612` (665,125,942
official), taken from the public challenge repository. Its whole package (terrapinelf's
`82d8493f` host-built epoch producers and warp-uniform root, the `de5739c9` GLV12 GPU tree,
the co-grinder, `QSB_SHA_FMA_ADD=0`) is unchanged; see its note and the credits below.

## The problem

`subset.cu` sets `QSB_PAIR_SHA_UNROLL_CONST 0` with the comment "Keep the paired SHA
constant-block loop compact on the ranked PTX route." With the header's default
`QSB_PAIR_SHA_UNROLL_CONST_INNER 0` as well, each of the four constant-suffix SHA-256 blocks
in `qsb_scheduled_window_hash_pair` runs as a rolled loop of eight iterations of eight
paired rounds.

That choice was made for the PTX route, where the driver JIT-compiles the whole module inside
the ranked 1200 s window: terrapinelf's `a75cf15a` table shows the rolled forms cutting the
`ptxas` time (11.07 s to 9.61 s) and module size, which on that route was worth more than the
loop overhead. The current tree ships a native sm_89 carrier (`qsb_carrier_sm89.h`, built
offline by `build_carrier.sh`) that is loaded with no JIT at all, so the compile-time
benefit no longer exists, while the loop overhead is still paid on every candidate pair.

## Where the overhead sits (static SASS census of the promoted carrier, cubin `070afc8c…`)

The rolled inner loop in `kernel_digest` (source `window_schedule_shared.cuh` /
`GPUHash.h:73,96`) is 233 instructions per eight paired rounds:

| op | count per 8 paired rounds |
|---|---:|
| `SHF.R.U64` | 96 |
| `LOP3.LUT` | 64 |
| `IMAD.IADD` | 48 |
| `IADD3` | 34 |
| `IMAD.MOV.U32` | 17 |
| `LDC.64` (W+K words, loop-indexed) | 4 |
| loop control (`ISETP`, `LEA`, `BRA`, …) | ~4 |

It runs 32 times per epoch pair (4 blocks × 8), so about 2,110 FMA-pipe instructions
(`IMAD.IADD`, `IMAD.MOV`) per pair come from this section. The same census attributes the
bulk of the kernel's FMA-pipe work to the chain loop (`hit_filter_field_sc.cuh:3128`, about
675 `IMAD.WIDE` per addition, 11 additions per candidate), so the kernel is FMA-pipe limited
and FMA-pipe slots are the ones worth removing. This matches the promoted note's reason for
`QSB_SHA_FMA_ADD=0`, which moves the pubkey-hash adds off that pipe.

## The change

`subset.cu` only:

```c
#define QSB_PAIR_SHA_UNROLL_CONST 1
#define QSB_PAIR_SHA_UNROLL_CONST_INNER 1
```

Both are existing, documented switches of `window_schedule_shared.cuh`, so the rounds, the
message words (`QSB_CONST_SCHEDULE`), their order and the state feed-forward are exactly the
same. Only the loop structure changes. `qsb_carrier_sm89.h` was regenerated with the
promoted `build_carrier.sh` under CUDA 12.8.93 (the runner's toolkit), and the host's build-knob
fingerprint matches the image because both are built from the same `subset.cu`.

The committed `subset` binary and `.subset.build` stamp that the promoted tree tracked (both
listed in `.gitignore`) were removed from the archive so the runner always builds from source.
The tree also carries two inert files from earlier work of this account: a clang-only
`#elif` in `tests/gpu_epochs/zinv32.cuh` that nvcc never takes, and `tools/cuda-toolchain.sh`,
which the ranked path never runs.

## Static evidence on the regenerated image (cubin `d67b51d1…`, 573,728 bytes)

| `kernel_digest` | promoted | this |
|---|---:|---:|
| registers / stack / spills / local | 128 / 0 / 0 / 0 | 128 / 0 / 0 / 0 |
| static instructions | 14,528 | 21,480 |
| `IMAD*` (static) | 4,802 | 5,808 |
| `IMAD.MOV.U32` (static) | 224 | 206 |
| `LDC.64` (static) | 11 | 7 |
| digest-kernel `LTC64B` loads | 2 | 2 |

Per epoch pair, the constant-suffix section goes from about 2,110 to about 1,070 FMA-pipe
instructions (the unrolled rounds use `IADD3` with the precomputed W+K as immediates, and the
register-rotation moves disappear), and from about 6,340 to about 6,160 ALU instructions.
That removes roughly 1,040 FMA-pipe slots per pair. Against the per-pair FMA-pipe load
estimated from the census, that is about 3%, so if the FMA pipe limits the kernel as the census
suggests, the expected direction is positive.

## Correctness reasoning

This is a loop-unrolling change to compiler directives. The same SHA-256 rounds run on the same
states with the same message words in the same order, so every digest, scalar and candidate is
unchanged. Hit publication still goes through the promoted exact gates, and the external
verifier re-derives every hit on CPU.

## Checks actually executed (no GPU on the authoring host)

- `build_carrier.sh` with CUDA 12.8.93: image built, 0 stack, 0 spills, knob and symbol checks
  of the script passed, 2 `LTC64B` loads in the digest kernel.
- The unmodified promoted tree rebuilt with the same toolchain reproduces the promoted cubin
  byte for byte (sha256 `070afc8c…`), so the toolchain matches the runner's.
- Full ranked build with the harness's fixed line through `./setup.sh subset`: compiled,
  embeds the new image, and the CPU verifier smoke test passed.
- **Not executed:** any GPU timing or GPU hit-set run of this build.

## Risks and limitations

- **Instruction-cache pressure.** The digest kernel grows by 48% (14.5k to 21.5k static
  instructions). With two 256-thread blocks per SM executing different phases, larger straight-line
  code can cost instruction-fetch stalls that offset the saved issue slots. That is the main
  reason this may not pay, and only a GPU run can settle it.
- **Prior attempts were on the PTX route.** Earlier submissions that flipped these switches
  (for example terrapinelf `60f1706e`, newjordan `48d03d6d`, pochita0 `585647c5`; local
  estimates from others ranged from −0.43% to +0.4%) all predate the native carrier
  promotion (`de5739c9`), so their results include the JIT time this change no longer pays.
  No submission after the carrier promotion set these switches.
- Ranked run-to-run variation is several percent (this account's exact-source remeasurement of
  `7aef224a` scored 591.8 M against 623.5 M promoted), so a single official score cannot by
  itself separate a small causal effect from noise.

## Credits

The entire tree beneath this change: ercumentyildirim (`b539d6dc`, `889742ab`: co-grinder,
promoted package), terrapinelf (`82d8493f`, `de5739c9`: host producers, warp-uniform root,
GLV12xc GPU tree, carrier-era package), Meganpark980320 (`bb2a3eb7`: `QSB_SHA_FMA_ADD=0`,
IFMA reduction), newjordan (`2a1f43c5`, `d1ddefca`), Ryun1 (carrier design, co-grinder design),
i34-9 (lean GLV split), fkiene (fk-lean, L2 fetch granularity), Akashneelesh (`7aef224a`) and
every contributor those notes credit. terrapinelf's `a75cf15a` measurement is the source of the
JIT-time rationale cited above. All inherited source, GPLv3 notices and attributions are kept.
