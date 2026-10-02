# Subset measurement run: a 7-arm in-run probe of three exact device switches on the record `4cc9d2d8`, one of them cadamcat's full-table fetch hint (not a promotion candidate)

Prepared with Claude Opus 5.5 in Claude Code. This host has no GPU. This is a **measurement submission**: it time-slices seven digest-kernel images of the current record on the ranked card, so each arm's rate can be read from the public hit list. Its own score is not meant to beat the record.

## What this run measures

This run measures three exact device switches on the record `4cc9d2d8` (kshitij-hash) on the ranked card, each between two controls:

| arm | extra device flags | what it changes (all bit-identical, per the record's own documentation) |
|---:|---|---|
| 0 | none | control: the record's device code |
| 1 | `-DQSB_S3_HINT_MODE=2` | cadamcat's: the `.L2::64B` fetch hint on all four 16-byte loads of a cold table record, not only on X's first (same bytes, same `.cs.nc` operator) |
| 2 | none | control |
| 3 | `-DQSB_R_CBANK=1` | the record's switch: R read from the constant bank in the paired front and tail instead of ABI register arguments |
| 4 | none | control |
| 5 | `-DQSB_R_CBANK_TAILS=1` | the record's switch: the same for the tails only |
| 6 | none | control |

`QSB_S3_HINT_MODE` is cadamcat's code from the open submission `4281592d`, ported unchanged except that its knob always enters the knob string (so every arm has the same keys) and that `QSB_CODE_ROLL` stays at the record's 0. The other two are the record's own switches. Switches with host-side parts (`QSB_ROOT_FILL` changes the launch grid, `QSB_QMIX_RT` changes uploads during the run) cannot be sliced this way and are not included.

## How it works

The probe is patternrecognition9-del's in-run multi-arm framework from `9b2fbb14` (`QsbCarrier.h`, `build_carrier.sh`, and the probe lines in `tree.cu`), ported onto the record. Only its arm table and its list of free knobs differ, plus two merges into the record's `build_carrier.sh`, described below.

- **Images:** the carrier holds one cubin per arm. Each is built from this source with that arm's extra device-only flags.
- **Loader check:** the loader refuses an arm whose knob string differs from the host binary's outside `QSB_PROBE_FREE_KNOBS`. This run adds `QSB_S3_HINT_MODE`, `QSB_R_CBANK` and `QSB_R_CBANK_TAILS` to that list.
- **Time slicing:** the epoch ranks are split into 7 equal arm spans A = floor(C(137,6)/7), each into 256 slots S = floor(A/256). Round j runs arms 0..6 in turn, about 0.7 s each, and arm k searches ranks from k·A + j·S upward.
- **No overlap:** every candidate is searched once, by one arm, and no rank is searched twice.
- **Per-arm rate:** the work of (k, j) is the largest verified GPU-hit rank in its slot minus the slot start.
- **Probe mode:** the host producers are off, because their build-ahead cannot follow a time-driven schedule. Every batch is then built by the GPU producers on its slot stream. The score is therefore not comparable with a normal draw of the record.

## Merges into the record's `build_carrier.sh`

The probe's per-arm loop replaces the record's single-image block. Two record features are kept per arm:

- **Optional `kernel_gt_offset_y` entry:** the `QSB_YOFF_S` images carry this kernel. Each arm's kernel list includes it when the arm's image does, and the header's name list includes `QK_YOFF`, matching the record's `QsbCarrier.h` enum.
- **Switch-dependent checks:** `QSB_QMIX_RT` needs the `QSB_QMIX_MASK_C` global, and `QSB_GATE_FMA_RT` needs the `QSB_GATE_FMA_C` global. The `QSB_CONST_CALLEE` image must have 16 UR-indexed `c[0x3]` loads in `kernel_digest`. These checks now run for every arm.

## Exactness

Every arm is one of the record's own switches, which the record documents as bit-identical. The record's start-up self-checks run unchanged. The host gate recomputes every GPU nomination and every co-grinder hit with OpenSSL before publishing it, so a faulty arm could lose hits but could never publish a wrong one.

## Checks done here

All builds ran in an amd64 container with CUDA 12.8.93, the ranked runner's toolkit.

| check | result |
|---|---|
| `build_carrier.sh 24`, 7 arms | every arm 0 spill bytes, 112 B stack, digest registers 128. Arms 0, 2, 4 and 6 are the same cubin (`30926eac…`); arm 1 has 16 `LTC64B` loads in `kernel_digest` against 4; every arm resolves the same kernels |
| arm knob strings against the host binary's `QSB_CARRIER_KNOBS` (3,219 bytes) | same keys in the same order. Each arm differs only in its own free knob, so every arm loads |
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

- **cadamcat** (co-author): the full-table fetch hint `QSB_S3_HINT_MODE` (`4281592d`, after `ffcf41bf`).
- **kshitij-hash** (co-author): the record `4cc9d2d8` and the `QSB_R_CBANK` switches.
- **patternrecognition9-del** (co-author): the in-run multi-arm probe framework (`9b2fbb14`).
- **Credited through the record** (co-authors, up to Yukon's limit): ercumentyildirim, fkiene, i34-9, kaankolcu, terrapinelf, HyeokxC, newjordan. Also credited through it: jacklightChen. Every contributor named in the record's note is credited through it. The GPU arithmetic headers derive from VanitySearch (GPLv3, `COPYING`); the co-grinder's field and scalar code follow libsecp256k1 (MIT, `COPYING-secp256k1`).
- **Mine:** the port onto the record, the choice of arms, the free-knob additions, the `build_carrier.sh` merges and the checks above.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
