# Subset measurement run: an 8-arm in-run probe of six exact device switches on the record `fb6f5a8f` (not a promotion candidate)

Prepared with Claude Opus 5.5 in Claude Code. This host has no GPU. This is a **measurement submission**: it time-slices eight digest-kernel images on the ranked card and reads each arm's rate from the public hit list. Its own score is not meant to beat the record.

## What this run measures

The record (`fb6f5a8f` = kshitij-hash's `e6715658` + my contiguous co-grinder walk) keeps several exact device switches at 0. They were rejected on a local RTX 4090 that is not thermally limited. The ranked card runs thermally limited, so energy per candidate decides there, and a switch that saves energy without saving time locally could win on the ranked card. Two more are ercumentyildirim's energy cuts from PR 2441 (`dea321f0`), measured neutral on a power-capped local card. None of the six has a ranked measurement yet.

| arm | extra device flags | what it changes (all bit-identical) |
|---:|---|---|
| 0 | none | control: the record's device code |
| 1 | `-DQSB_Q_SPREAD=1` | the two six-term Q warps on sub-partitions 0 and 3 instead of both on 0 |
| 2 | `-DQSB_TREE_UNROLL=1` | the inversion tree's up and down loops fully unrolled |
| 3 | none | control again (A/A) |
| 4 | `-DQSB_GATHER_L1_POLICY=1` | cold table-record loads with `.L1::no_allocate` |
| 5 | `-DQSB_CODE_ROLL=2` | ercumentyildirim's: the pair gate's two recovery-id hashes as one two-trip loop (cubin 21 KB smaller) |
| 6 | `-DQSB_WSEC_L1LAST=1` | ercumentyildirim's: window-schedule and first-state loads with `.L1::evict_last` |
| 7 | `-DQSB_DIGEST_MINB=1` | `kernel_digest` with minBlocksPerMultiprocessor 1 (174 registers, one block per SM) |

## How it works

The probe is patternrecognition9-del's in-run multi-arm A/B framework from `9b2fbb14` (`QsbCarrier.h`, `build_carrier.sh`, the probe lines in `tree.cu`), ported unchanged onto the record. Only its arm table and its list of free knobs differ.

- **Loading:** the carrier holds one cubin per arm, all built from this source with that arm's extra device-only flags. The loader refuses an arm whose knob string differs outside `QSB_PROBE_FREE_KNOBS`; this run adds `QSB_Q_SPREAD`, `QSB_TREE_UNROLL`, `QSB_GATHER_L1_POLICY`, `QSB_CODE_ROLL` and `QSB_WSEC_L1LAST` to that list.
- **Time slicing:** the epoch ranks are split into 8 equal arm spans A = floor(C(137,6)/8), each into 224 slots S = floor(A/224) (4.59M epochs, about 30% above a 0.7 s slice). Round j runs arm 0..7 in turn for about 0.7 s each, and arm k searches ranks from k·A + j·S upward. Every candidate is searched once, by one arm.
- **Reading the result:** the work of (k, j) is the largest verified GPU-hit rank in its slot minus the slot start. Arms of the same round share the round's slice time, so each arm's work is compared per round with the control arms'.
- **Precision:** on `9b2fbb14`, 240 rounds gave about ±0.11–0.18% per arm, and the two control arms agreed to 0.12%.
- **Side effects of probe mode:** the host producers are off (the probe schedule is time-driven), so the GPU builds its own epoch descriptors, and the co-grinder gets that CPU. The score is therefore not comparable with a normal run.

## Exactness

Every arm is one of the record's own switches, documented there as bit-identical. The start-up self-check replays the walker and the gather forms, and the host gate recomputes every GPU nomination with OpenSSL before publishing it. A broken arm could lose hits, never publish a wrong one.

## Checks done here

All builds ran in an amd64 container with CUDA 12.8.93, the ranked runner's toolkit.

| check | result |
|---|---|
| `build_carrier.sh 24`, 8 arms | every arm 0 spill bytes, 112 B stack; digest registers 128 (arm 7: 174); arms 0 and 3 are the same cubin; arm 5's cubin is 452,384 B against 473,376 B |
| arm knob strings against the host binary's `QSB_CARRIER_KNOBS` | same keys in the same order; each arm differs only in its own free knob, so every arm loads |
| the harness's build line (`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`) | builds with no errors |

## Reproducing

```
./setup.sh subset
./benchmark.sh subset
```

To read the arms from the public hit list:

1. Take the GPU hits: the 128 most frequent `skip[6:9]` patterns.
2. Take each hit's lexicographic rank of its first six skip indices; arm k = rank // A, slot j = (rank − k·A) // S.
3. Take, for each (k, j), the largest rank − (k·A + j·S) + 1.
4. For each round j ≥ 1 in which every arm has hits, divide each arm's work by the mean of arms 0 and 3.

## Base and attribution

- **patternrecognition9-del** (co-author): the in-run multi-arm probe framework (`9b2fbb14`).
- **ercumentyildirim** (co-author): `QSB_CODE_ROLL` and `QSB_WSEC_L1LAST` (PR 2441, `dea321f0`), ported unchanged.
- **kshitij-hash** (co-author): the record's device and host code (`e6715658`) and the other four switches this run measures.
- **Credited through `e6715658`** (co-authors, up to Yukon's limit): terrapinelf, i34-9, HyeokxC, jacklightChen, fkiene, kaankolcu, newjordan. Every contributor named in `e6715658`'s note is credited through it.
- **Mine:** the record's contiguous co-grinder walk (`fb6f5a8f`), the choice of arms, the free-knob additions, the checks above, and the per-arm reading of ranked probe hit lists.

All inherited source, GPLv3 notices and attributions are kept. Only `candidates/subset/` changes.
