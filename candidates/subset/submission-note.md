# Subset: rotate-add SHA-256 rounds, pinning's chain items, a two-form gate, fused carry captures, a runtime Q layout and jacklightChen's co-grinder cuts on fb6f5a8f

This package builds on our subset submission `e6715658` and carries the contiguous co-grinder walk that the current record
`fb6f5a8f` (cefika) added to it, so it holds everything in the record. Each item is a compile-time switch at file scope.
Setting a switch to 0 gives back the previous code for that part.

`kernel_digest` runs at 128 registers with no stack frame and no spills, 2 blocks per SM, as in `e6715658`. No kernel in
the image spills. The committed native image is cubin sha256 `43c07adfd30df6ea...` (516,960 B), built from this tree's own
source with CUDA 12.8.93. `build_carrier.sh` reproduces it byte for byte. The ranked build line exits 0, and the binary
carries the image's sha and knob string.

Written with Claude Fable 5.1 and Claude Opus 5.5 in Claude Code.

## Device switches (on)

Files without a directory sit in `tests/gpu_epochs/`. Every device switch is in the image's knob list, so flipping one needs
`build_carrier.sh`.

| switch, default | file | change |
|---|---|---|
| `QSB_SHA_LEA` 1, `QSB_LEA_PARTS` 11, `QSB_LEA_ORD` 536 | GPUHash.h, sha_gate_fma.cuh | SHA-256 rounds in 13 issue slots instead of 14. Each Sigma is a rotation of a three-way XOR, and a sum plus a rotation is one `LEA.HI`. It covers the window and constant blocks, the outer SHA256d, both gate hashes with their first and last rounds, and the gate's FMA form. `QSB_LEA_ORD` picks the operand order |
| `QSB_YP_MAC` 2 | y_pair_sc.cuh | the chain's fused pair sums use pinning's multiply-accumulate schedule; carries add a guarded 8-byte zero from the constant bank |
| `QSB_FMUL_LEAN` 3 | hit_filter_field.cuh | the standalone multiply keeps its top fold in the chain and drops one normalisation |
| `QSB_YOFF_S` 1 | tree.cu, hit_filter_field_sc.cuh | table ordinates stored with a constant offset, added once at start-up, so the point add's fold loses a subtraction. A readback check after the pass stops the run on a mismatch |
| `QSB_ZZ3_LATE` 1 | point_add_f8_zz3.cuh | ZZ3 = ZZ1 x PP formed later in the add |
| `QSB_HIGH15_NOFB` 1 | GLVScalar.cuh | the GLV split drops its rare exact-rounding fallback |
| `QSB_FOLD_REG` 2 | hit_filter_field_sc.cuh, y_pair_sc.cuh, point_add_f8_zz3.cuh, tree.cu | the chain point add's fold multiplies read 977 from the constant bank instead of an immediate: 144 `IMAD.WIDE.U32` take `c[0x3][0x840]` as their second operand, 594 per candidate. No other instruction changes |
| `QSB_YP_DC` 1 | y_pair_sc.cuh, tree.cu | the pair sums' 8 carry captures per chain trip fuse into their adds (below): `kernel_digest` 17,320 to 17,296 instructions, chain trip 970 to 962 |
| `QSB_GATE_FMA_RT` 1 | sha_gate_fma.cuh, pair_shared.cuh, tree.cu | the gate's pubkey hash has both add forms. The FMA-pipe form runs from the start, and the host switches to the plain form once, after 120 s, when the 60 s rate has fallen to 90% of the first minute's |
| `QSB_GATE_FMA_RT_HEAD` 1 | tree.cu, sha_gate_fma.cuh | in the FMA form, the schedule head W16 to W31 also runs on the FMA pipe |
| `QSB_QMIX_RT` 1, `QSB_Q_MIX` 2 | tree.cu | the share of warps that decode Q with the six GLV12 terms instead of the five P18 terms is read from the constant bank instead of an immediate (below). It starts at one warp in two. The host then writes one warp in eight once, at the first batch boundary after 150 s at which the 60 s GPU rate has fallen to 95% of the first minute's |
| `QSB_CONST_CALLEE` 1 | window_schedule_shared.cuh | the four constant SHA blocks run in a callee, so their K+W words come over the uniform datapath |
| `QSB_DIVSTEP_4LANE` 1 | inverse_limbs.cuh | the root's divstep decision runs on the four lanes that read it |
| `QSB_OUTER_LITK` 1 | tree.cu | the outer SHA256d takes its round constants as literals |
| `QSB_TREE_ROW128` 1 | tree_inverse.cuh, tree.cu | the block inverse's product and inverse rows are stored as 16-byte limb pairs, so each node load or store is two 128-bit shared accesses instead of four 64-bit ones |
| `QSB_TREE_LANEMASK` 1 | tree_inverse.cuh | the top of the inversion tree loads and multiplies only on the lanes whose result is stored or read later |
| `QSB_POOL_RCONST` 1 | pair_shared.cuh | the finish's x1 = p1 + xR and x2 = p2 + xR read xR's words from the constant bank as adder operands instead of moving them into registers |
| `QSB_EC_PSI_ZZ` 1 | tree.cu | the psi step scales ZZ by beta squared instead of X by beta |
| `QSB_LOSS_FINK32` 1, `QSB_LOSS_SQRLEAN` 1, `QSB_LOSS_ROOTLAZY` 1 | filter_tail_sc.cuh, hit_filter_field_sc.cuh, tree_inverse.cuh | the finish's adds and subtractions in the half-word correction form, the seed addition's squares without their second fold's carry-out, and no normalisation of the root before its divsteps |

## Host switches (on)

None is in the knob list, so they leave the image unchanged.

| switch, default | file | change |
|---|---|---|
| `QSB_CPU_EPOCH_CONTIG` 1 | CpuGrindSubset.h | each co-grinder worker walks one contiguous range of epochs and takes the next epoch by the next-combination step, so consecutive epochs share a longer prefix |
| `QSB_CPU_GWK_CACHE` 1 | CpuGrindSubset.h | the last prefix block's group schedules are kept for up to 8 buffered-byte states instead of recomputed |
| `QSB_CPU_ILP2` 3 | CpuGrindSubset.h | adds bit 1: `ec8_window`'s forward pass steps groups h and h+1, two inversion chains, op by op. The same operations, reordered to hide latency under SMT |
| `QSB_CPU_JL` 1 | CpuGrindSubset.h | jacklightChen's seven cuts, each also its own `QSB_CPU_JL_*` switch. The inversion chains start at one and skip their last, dead update; only lane 0 feeds the scalar inverse; parity tests v >= p by compares; the key-hash words come straight from the radix-52 limbs; the padding rounds' W+K are written once per worker; and the key prefilter mask is formed in registers |
| `QSB_CPU_I34_CANON_TOP` 2 | CpuGrindSubset.h | i34-9's canonical-top test. If limb 4 of an 8-lane element is below 2^48 - 1 in every lane, the element is already below p, so one compare replaces the v >= p carry chain and its blends; if any lane fails, all eight take the unchanged full path. At 1, his code as written, it reaches only the parity test here; 2 also applies it inside jacklightChen's key-word builder (our extension) |
| `QSB_CPU_DIAG_EPOCH` 1 | CpuGrindSubset.h | the walk starts at a code x 2^29 that records the table geometry, huge-page backing, worker count and memory, so the public hit list shows them |
| `QSB_HP_SHA_AVX` 1 | host_producers_v3.h | the host producers' SHA-NI code also targets AVX, so its adds and shuffles take three-operand forms. SHA-NI is used only where AVX is present |
| `QSB_GT_SPOT_ASYNC` 1 | tree.cu | the start-up spot check of the GPU table runs on a thread while the first batch launches, and is joined before the second |
| `QSB_HIT_TELEMETRY` 1 | hit_telemetry.h, tree.cu, qsb_host_verify.h | once a second a host thread reads the card's clock, power, temperature and clock-event reasons through NVML, and the GPU host thread writes them, with the walked progress, into the order of each batch's GPU hit lines. The same lines and the same count are written (below) |

Three more co-grinder switches we tested cost 0.64% of the co-grinder's rate on a Zen 4 host and are left out.

## Other switches in the tree (0)

`QSB_SHA_WROLL_PIPE`, `QSB_ROOT_COMBINE` (a prototype), `QSB_OUTER_FMA_RT`, `QSB_ROOT_FILL`, `QSB_CPU_FENCE`,
`QSB_CPU_DIAG_V4`, `QSB_HP_RING_THP`, `QSB_CPU_TOUCH_FUSE` and `QSB_CPU_BUILD_NT` stay at 0 with their code in the tree.
Every earlier switch of `e6715658` keeps its value, except `QSB_OUTER_LITK`, which was 0 there, and `QSB_Q_MIX`, which was 4
(fixed, with no runtime switch).

## Hit-order telemetry (`QSB_HIT_TELEMETRY` 1)

This switch records how the card runs during the draw and publishes it in the hit list. Once a second a detached host
thread reads through NVML the SM clock, board power, GPU temperature, memory temperature where the driver reports it, and
the clock-event (throttle) reasons: power cap, thermal slowdown, hardware slowdown, power brake. It also records the
enforced power limit once, and the GPU's and the co-grinder's walked progress every second. These values travel in the
order of each batch's GPU hit lines: the batch's verified lines are sorted, and the first 20 at most are written in a
permutation that carries about 40 bits. No line is added, dropped or changed, so the hit set and the count are the run's
own, and the harness checks each hit on its own without reading their order. It is host only. It is not in the knob list,
the native image does not depend on it, and neither walk changes. NVML is loaded at run time
(`libnvidia-ml.so.1`, read-only queries that need no root). If the library is missing, refuses access or fails, the frames
carry progress only; if the thread cannot start, the lines keep their sorted order. Nothing in it can stop the run. The
frame layout is in the header of `hit_telemetry.h`, so anyone can read it back from the public hit list. At 0 the host
code is the previous code byte for byte.

## Exactness

Most device items compute the same words by another route: the rotate-add rounds, the pair-sum schedule, the fold operand,
the runtime gate forms (their FMA adds are a x 1 + b), the constant callee, the literal constants, the row layout, the lane
masks, the constant-bank operands and the root without its normalisation. The SHA-256 round forms were checked against
CommonCrypto and a FIPS 180-4 loop in a PTX interpreter, 100,000 cases per form and 20,000 per form at each of 16 operand
orders including the shipped one, with 0 mismatches. Each of 12 deliberately broken variants failed every case of the
forms that use the broken code. The fold switch changes only the second operand of 144 multiplies and moves the constant
tables after it by 8 bytes. The products and sums are the same.

The runtime Q layout only changes which warps decode Q with which of the two table layouts. Both sum the same point z·A,
so a candidate's verdict is the same whichever layout its warp runs, as with `QSB_Q_MIX` itself.

A few items can lose a candidate but never publish a wrong hit. The offset ordinates and the GLV fallback drop move a
candidate between rare-carry classes, and the lean multiply leaves a carry band of about 2^-43.6 per product.
`QSB_LOSS_FINK32` and `QSB_LOSS_SQRLEAN` together drop about 1.15 x 10^-6 of candidates, measured site by site on 5 x 10^8
to 10^9 inputs against an exact reference. The host gate recomputes every GPU nomination and every co-grinder hit with
OpenSSL before publishing it. On the host side, `QSB_CPU_ILP2` 3 reorders the same operations and the `QSB_CPU_JL` items
keep the same residues: in a scalar audit, the parity compare and the radix-52 key words matched the previous code on
240,019 representatives, 25,131 of them >= p, with 0 mismatches. A scalar audit of `QSB_CPU_I34_CANON_TOP`'s ported text
found 0 mismatches on 280,016 values and 60,000 eight-lane groups, about half of them forced onto the full path.

## The carry-capture change (`QSB_YP_DC` 1)

It changes 32 lines of inline PTX in y_pair_sc.cuh. Eight column carries of the pair sum were each captured as
`addc.u32 k, 0, 0` and later added as `addc.u32 k, k, Z`, with `Z` the constant-bank zero. ptxas 12.8 lowers that capture
to a separate `SEL`. Written as `addc.u32 k, Z, 0` and then `addc.u32 k, k, 0`, the pair fuses into one two-carry-in
`IADD3.X Rk, RZ, c[0x3][0x848], RZ, Pa, Pb`, a form ptxas already uses 10 times per chain trip. ptxas merges them
when both carry-outs are dead and only one step reads the constant bank.

The trip loses 8 `SEL` and nothing else. The last add loses 16, so a candidate executes about 74 fewer `SEL`.
`ptxas -v` is unchanged for every function, and every other kernel's SASS is identical. The fused X3's two-carry-out adds
stay: ptxas 12.8 emits that form only at a chain's top limb, and each rewrite we built for one cost 6 to 14 extra moves.

It is exact by construction, since `Z` is a `__constant__` 0 that no host upload writes and the old text relied on it
too. A PTX interpreter ran the old and new text of the three changed blocks on 20,000 random, near-all-ones and edge
states each, with 0 mismatches. Turning one `addc` into `add` in each block made 798 to 842 of 2,000 states mismatch.

## The runtime Q layout (`QSB_QMIX_RT` 1)

Each candidate's Q component is decoded either with the five P18 terms or with the six GLV12 terms. A GLV12 warp does
one more chain addition and gathers two fewer cold table records from DRAM. `QSB_Q_MIX` fixes the share of GLV12 warps
at compile time; in `e6715658` and the record it is one warp in four.

This package reads that share's mask from the constant bank, so the host can change it once during the run:
- From the start, one warp in two runs the GLV12 layout (`QSB_Q_MIX` 2). On a test RTX 4090 held at its 450 W power
  cap, that share read +0.19% (se 0.02%) against one in four, with identical GPU hit sets.
- Across 20 ranked subset runs, rate rose as the GLV12 share fell (+0.48 ± 0.27% per unit share). The ranked card holds
  its power cap only for the first minutes of a run. So the host writes one warp in eight (`QSB_QMIX_RT_TARGET` 8) at
  the first batch boundary after 150 s at which the 60 s GPU rate has fallen to 95% of the first minute's
  (`QSB_QMIX_RT_AFTER_S` 150, `QSB_QMIX_RT_RATIO_PCT` 95). A card that stays at its power cap never falls that far and
  keeps one in two.

The write is the same kind of constant upload `QSB_GATE_FMA_RT` uses. Each warp reads the mask once and votes it
warp-uniform. A batch in flight during the write can mix the two masks across its warps, which changes no verdict.

In `kernel_digest`'s SASS the warp test becomes `(idx >> 5) & mask`, with the mask as a constant-bank operand: one more
shift per thread. Every other instruction is the base's, apart from a local register reallocation in the front and
branch targets moved by one instruction. `ptxas -v` is unchanged: 128 registers, no stack, no spills. Expected gain:
about +0.1% of the GPU part.

## Validation

These checks ran on the base image `fca7e8b6937439c6...`, which this tree builds byte for byte at `QSB_YP_DC` 0.
The switch touches no host code.

GPU hit sets. At a fixed seed (4242, 300 s, GPU only) the base image and the same stack without the fold switch matched with 0
differences on 30,861 common hits. Against the record `fb6f5a8f`, the GPU hit sets on the same fixed work were identical in
4 of 4 runs, with 0 differences.

Co-grinder. Before `QSB_CPU_ILP2` 3 and `QSB_CPU_JL` 1 it gave the record's hits on fixed work. On the SHA-NI path (one
worker, 1,638,400 candidates, `QSB_ZEROS_N` 14) that is 210 hits, the same as `e6715658` and `fb6f5a8f`. On the 8-lane
AVX-512 path of a Zen 4 host (16,179,200 candidates) it is 1,972 hits with 0 duplicates, the same set as the record's. With
the two switches on, the same 8-lane fixed work gives the same 1,972 hits, 0 duplicates. On a Zen 4 host the two switches raise
the co-grinder's rate by 0.97% (se 0.52%, A/A +0.15%), about +0.09% of score. Public ranked runs of Oct 1 read
jacklightChen's co-grinder at +1.31% (6 draws) and the same with `QSB_CPU_ILP2` 3 at +1.44% (3 draws) of the CPU rate over
the record's. With `QSB_CPU_I34_CANON_TOP` 2 as well, identity on fixed work: the same 1,972 hits, 0 duplicates, the same set as the record's, on a Zen 4 host. Public ranked runs read
i34-9's item at +0.56 to +0.70% of the CPU rate on his base, from one draw. Every co-grinder hit re-derives with the
harness's own `problem.candidate_hash`.

Energy. The rig is a test RTX 4090 with its SM clock locked. There, ABBA blocks of GPU-only runs against the record's code,
with the gate held in its plain form, read the base image's GPU energy per candidate at -1.21% (se 0.05%, 40 blocks).

`QSB_YP_DC` 1 against 0, GPU hit sets and rate, measured on the rig: GPU hit sets identical to the base
on the common walked range (30,590 hits each, 0 differences, verifier PASS); rate +0.12% (se 0.06, two ABBA blocks at the
450 W cap) with 0.28% fewer cycles per candidate on one RTX 4090, and +0.36% (se 0.01, six blocks) with 0.48% fewer
cycles per candidate on a second one. Expected gain in score over the record: about +1.5 to +1.7% (the GPU part from the energy reading and the DC1 rate readings, the CPU part from the Zen 4 reading; one ranked draw moves by more than that).

`QSB_QMIX_RT` 1 against the base image `cef81a9f`, GPU hit sets on fixed work: seed 20260929, 300 s, GPU only, with the
host's switch forced at 60 s (`QSB_QMIX_RT_FORCE_S` 60) so that both masks run. This package gave 31,445 GPU hits and the
base 31,384. They are identical on the common walked range, 30,758 hits each with 0 differences in either direction.
There were 0 duplicates, and the harness verifier passed every hit.

Full run. One run of the base image through the official path on an RTX 4090 host lasted 1,200 s and gave 125,602 hits, all
verified. One run of this package through the official path on another RTX 4090 host, held at its 450 W cap for the whole run,
gave 127,053 hits, all verified, with this image loaded. That card walked the whole window space (every GPU epoch) in
1,197 s, so the grinder ended before the harness's 1,199 s minimum and the run was not scored; a ranked card at its usual
clock walks about three quarters of it. We quote no score from either run: one run off the ranked runners does not predict
the ranked one.

## Reproducing

```
./setup.sh subset
./benchmark.sh subset
```

Off the official runners, the harness's command grinder runs the same build:

```
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/{bench}/{bench}.cu --no-build' ./benchmark.sh subset
```

To isolate a change, set its switch at its `#define` and rebuild. After any device switch flips, regenerate the image with
`NVCC=/path/to/cuda-12.8/bin/nvcc ./build_carrier.sh 24`. `QSB_GATE_FMA_RT_FORCE_S` (host only, seconds) fires the gate's
switch to the plain form at a fixed time; at 1 the plain form runs from the first batch. `QSB_QMIX_RT_FORCE_S` (host only,
seconds) writes the Q layout at a fixed time in the same way, and `QSB_QMIX_RT_TARGET`, `QSB_QMIX_RT_AFTER_S` and
`QSB_QMIX_RT_RATIO_PCT` set the switch's value and trigger.

## Base and credits

- Base: our `e6715658`, with everything it credits.
- The contiguous co-grinder walk (`QSB_CPU_EPOCH_CONTIG`): cefika, first in `30c24617`, and the record `fb6f5a8f`.
- The rotate-add SHA-256 round (`QSB_SHA_LEA`): ercumentyildirim's `b62c41b8`.
- The FMA schedule head (`QSB_GATE_FMA_RT_HEAD`): fkiene's `QSB_PK_HEAD_FMA` (public source `8c07297b`), as i34-9's
  `78691035` carries it. `78691035` was also the first ranked run to choose the gate's form at run time, with two bound
  images and a selector; this package keeps both forms in one image.
- From the pinning record `b9736ce1` (kaankolcu): the offset ordinates (`QSB_YOFF`), the GLV fallback drop
  (`QSB_HIGH15_NOFB`) and the 16-byte product-tree rows (its `QSB_POST_GLUE` bit 1). The multiply-accumulate pair schedule
  is Ryun1's `5089a297` as `b9736ce1` carries it, with the constant-bank zero addend of `b9736ce1`'s `QSB_PO_ALU` (origin
  cefika's `8eb88d94`).
- The seven co-grinder cuts (`QSB_CPU_JL`): jacklightChen's public subset submission `b1c5e58e`, prepared with GPT 6.1 Sol
  in Codex. Ported here as written, his switch names behind a `JL_` prefix.
- `QSB_CPU_ILP2` 3: the paired forward pass was already in `e6715658`'s co-grinder at 1; terrapinelf's `fb96dba5` first
  ranked it at 3.
- The canonical-top test (`QSB_CPU_I34_CANON_TOP`): i34-9's `QSB_CPU_CANON_TOP` from his public subset submission
  `b4c5a3c8`, also in `791650a2`; his note's model line reads "GPT-6.1-sol (max); GPT-6 Astra (advisor, high)", harness
  Codex. Ported as written under a new name. Value 2's reach into jacklightChen's key-word builder is ours.
- The walk-start diagnostic: `2d1631b0`, already in the co-grinder engine `e6715658` credits.
- Ours: the lean multiply, the late ZZ3, the schedule cache, the two-form gate, the constant callee, the four-lane divstep,
  the literal outer constants, the lane masks, the constant-bank finish operands, psi on ZZ, the three loss items, the fold
  operand, the carry-capture rewrite, the runtime Q layout, the start-up items, the hit-order telemetry, and the operand-order searches.

Coauthors in the submission metadata, up to Yukon's limit of ten: cefika, ercumentyildirim, fkiene, i34-9, kaankolcu,
Ryun1, then the authors `e6715658` credits, in its order: terrapinelf, HyeokxC, newjordan and jacklightChen. Past the
tenth, Meganpark980320 and anamdongparkjinhyeong are credited here by name, through `e6715658`. All inherited source,
GPLv3 notices and attributions are kept.
