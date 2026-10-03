# Subset measurement run: a 7-arm in-run probe of two Q layouts and the outer-block roll on the record `faf5422a` (not a promotion candidate)

Prepared with Claude Opus 5.5 in Claude Code. This host has no GPU. This is a **measurement submission**: it time-slices seven digest-kernel images of the current record on the ranked card, so each arm's rate can be read from the public hit list. Its own score is not meant to beat the record.

## What this run measures

The record `faf5422a` (kshitij-hash; the same tree as `d7c57dd4`, image `5f1f8111`) runs `QSB_Q_MIX` 2 and `QSB_CODE_ROLL` 2. This run measures three neighbouring values of those two switches on the ranked card, each between two controls:

| arm | extra device flags | what it changes (all bit-identical, per the record's own documentation) |
|---:|---|---|
| 0 | none | control: the record's device code |
| 1 | `-DQSB_Q_MIX=1` | every warp decodes Q with the six GLV12 terms: 6 cold table records and 10 field additions per candidate, against 7 and 9.5 in the record |
| 2 | none | control |
| 3 | `-DQSB_CODE_ROLL=3` | adds bit 0 to the record's rolled gate: the SHA256d outer block runs once inside the paired front instead of twice inline in `kernel_digest` |
| 4 | none | control |
| 5 | `-DQSB_Q_MIX=4` | one warp in four on the six GLV12 terms (7.5 cold records, 9.25 additions), the value before the record |
| 6 | none | control |

All three are values of the record's own switches, and their device code is unchanged here. `QSB_Q_MIX` is kshitij-hash's. `QSB_CODE_ROLL` is ercumentyildirim's (`dea321f0`, `667cfead`, PR 2441) as the record ports it; the record's source notes that value 3 was drawn once (DPZZxlz, `e69d3dce`). Switches with host-side parts (`QSB_ROOT_FILL` changes the launch grid, `QSB_QMIX_RT` changes uploads during the run) cannot be sliced this way and are not included.

## How it works

The probe is patternrecognition9-del's in-run multi-arm framework from `9b2fbb14` (`QsbCarrier.h`, `build_carrier.sh`, and the probe lines in `tree.cu`), ported onto the record. Only its arm table and its list of free knobs differ, plus two merges into the record's `build_carrier.sh`, described below.

- **Images:** the carrier holds one cubin per arm. Each is built from this source with that arm's extra device-only flags.
- **Loader check:** the loader refuses an arm whose knob string differs from the host binary's outside `QSB_PROBE_FREE_KNOBS`. This run adds `QSB_CODE_ROLL` to that list. `QSB_Q_MIX` is already on it. Every arm keeps `QSB_CODE_ROLL` non-zero, so its key is in every knob string.
- **Time slicing:** the epoch ranks are split into 7 equal arm spans A = floor(C(137,6)/7), each into 256 slots S = floor(A/256). Round j runs arms 0..6 in turn, about 0.7 s each, and arm k searches ranks from k·A + j·S upward.
- **No overlap:** every candidate is searched once, by one arm, and no rank is searched twice.
- **Per-arm rate:** the work of (k, j) is the largest verified GPU-hit rank in its slot minus the slot start.
- **Probe mode:** the host producers are off, because their build-ahead cannot follow a time-driven schedule. Every batch is then built by the GPU producers on its slot stream. The score is therefore not comparable with a normal draw of the record.

## Merges into the record's `build_carrier.sh`

The probe's per-arm loop replaces the record's single-image block. Two record features are kept per arm:

- **Optional `kernel_gt_offset_y` entry:** the `QSB_YOFF_S` images carry this kernel. Each arm's kernel list includes it when the arm's image does, and the header's name list includes `QK_YOFF`, matching the record's `QsbCarrier.h` enum.
- **Switch-dependent checks:** `QSB_QMIX_RT` needs the `QSB_QMIX_MASK_C` global, and `QSB_GATE_FMA_RT` needs the `QSB_GATE_FMA_C` global. The `QSB_CONST_CALLEE` image must have 16 UR-indexed `c[0x3]` loads in `kernel_digest`. These checks now run for every arm.

## Exactness

Every arm is a value of one of the record's own switches, which the record documents as bit-identical: every `QSB_Q_MIX` value sums Q to the same point, and `QSB_CODE_ROLL` bit 0 runs the same compression on the same eight words. The record's start-up self-checks run unchanged. The host gate recomputes every GPU nomination and every co-grinder hit with OpenSSL before publishing it, so a faulty arm could lose hits but could never publish a wrong one.

## Checks done here

All builds ran in an amd64 container with CUDA 12.8.93, the ranked runner's toolkit.

| check | result |
|---|---|
| `build_carrier.sh 24`, 7 arms | every kernel of every arm 0 spill bytes; `kernel_digest` 128 registers with no stack frame in every arm. Arms 0, 2, 4 and 6 are the same cubin (`e829500564db6dc0…`), and every arm resolves the same kernels |
| control image against the record's image `5f1f8111` | the SASS of every kernel is identical; only the knob string differs (two more probe keys) |
| arm 3 (`QSB_CODE_ROLL` 3) | `kernel_digest` 14,632 to 13,376 static instructions (one copy of the outer compression fewer), 16 UR-indexed `c[0x3]` loads as in the control |
| arms 1 and 5 (`QSB_Q_MIX` 1 and 4) | `kernel_digest` 14,632 static instructions, as in the control; only the warp's layout choice changes |
| arm knob strings against the host binary's `QSB_CARRIER_KNOBS` (3,216 bytes) | same keys in the same order. Each arm differs only in its own free knob (arms 1 and 5: `QSB_Q_MIX`; arm 3: `QSB_CODE_ROLL`), so every arm loads |
| the harness's build line (`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`) | builds with no errors |

## Reproducing

```
./setup.sh subset
./benchmark.sh subset
```

To read the arms from the public hit list:

1. Take the GPU hits: the 128 most frequent `skip[6:9]` patterns.
2. Take each hit's lexicographic rank of its first six skip indices. Then arm k = rank // A and slot j = (rank − k·A) // S.
3. Take, for each (k, j), the largest rank − (k·A + j·S) + 1.
4. For each round j ≥ 1 in which every arm has hits, compare each measured arm's work with the controls of that round.

## Base and attribution

- **kshitij-hash** (co-author): the record `faf5422a` (`d7c57dd4`) and the `QSB_Q_MIX` switch.
- **patternrecognition9-del** (co-author): the in-run multi-arm probe framework (`9b2fbb14`).
- **ercumentyildirim** (co-author): `QSB_CODE_ROLL` (`dea321f0`, `667cfead`, PR 2441), as well as items the record credits him for.
- **Credited through the record** (co-authors, up to Yukon's limit): fkiene, i34-9, kaankolcu, Ryun1, terrapinelf, HyeokxC, newjordan. Also credited through it: jacklightChen, Meganpark980320 and anamdongparkjinhyeong. Every contributor named in the record's note is credited through it. The GPU arithmetic headers derive from VanitySearch (GPLv3, `COPYING`); the co-grinder's field and scalar code follow libsecp256k1 (MIT, `COPYING-secp256k1`).
- **Mine:** the port onto the record, the choice of arms, the free-knob addition, the two `build_carrier.sh` merges and the checks above.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
