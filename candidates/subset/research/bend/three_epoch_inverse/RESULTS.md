# Results

## Identity

- Date: 2026-09-17 (America/Indiana/Indianapolis)
- Promoted base considered: `1248235b7bab9471e6cc4c2f301180837e039fc8`
- Two-epoch comparison mechanism: pending submission `b76b3d5` (the cancelled
  `7140869` stack plus the public register-carried digit window)
- Bend: 2.0.5, telemetry disabled
- Native compiler: Apple clang 21.0.0, eight CPU threads
- `LAWS.bend`: `f5dde0aa76e660a2398a2e5e9684a342efd0959c7b5097cbb5b457a6c56c6bec`
- `PROOF.bend`: `1b544d0abdf78c8da37d2ced451da2472bc655c3bc1b320cedef8c528016a00b`
- `MODEL.bend`: `0c8f617dbd25098b5ca6bafcec8f20416e6101ac397cfe58901c3139026e07d3`

## Commands

```text
BEND_NO_TELEMETRY=1 ~/.bend/bin/bend PROOF.bend
All terms check.

BEND_NO_TELEMETRY=1 ~/.bend/bin/bend MODEL.bend -o model
./model --threads 8
MkStats{4913, 0, 0}
```

The native model checked every `(a,b,c)` in F_17, including all seven nonempty
zero patterns. It found no mismatch between the split outputs and independently
computed inverses, with zero inputs expected to produce a masked zero output.

## Interpretation

The algebraic schedule and zero isolation survive the complete finite field
search. The Bend laws additionally pin one all-nonzero case and one case where
the first candidate is unusable but the other two must remain correct.

This does not establish the secp256k1 field implementation, canonicalization,
CUDA tree indexing, scratch-buffer ownership, synchronization, compilation or
throughput. A CUDA prototype must start from the exact eventual two-epoch
source, retain the source-bound corrected inverse, and compare against an
unchanged two-epoch control. No production source or submission was changed by
this experiment.

## Full-width and ownership oracle

```text
PYTHONDONTWRITEBYTECODE=1 python3 check_k3.py
status=PASS
inverse_cases=20519
zero_patterns=7
mutation_rejections=20387
scratch.words=135168
scratch.wrong_slot_collisions=67584
slot_protocol.launches=257
slot_protocol.wrong_no_wait_conflicts=255
```

Final `check_k3.py` SHA-256:
`42feab5582981ec0a4c0ce09aecc638fe29dd976f41cdc616bee9aafc367db22`.

The arithmetic oracle uses Python big integers modulo the actual secp256k1
prime. The scratch test checks the complete `(slot,block,lane,word)` mapping for
two slots, 33 blocks, 256 lanes and eight 64-bit words per parked pair. Removing
the slot dimension produces the expected full cross-slot collision set. The
protocol model requires launch `i-2` to be drained before launch `i` reuses its
slot and includes the mandatory final drain; its no-wait mutation conflicts on
every launch after the first two. These are address/ownership checks, not proof
of CUDA event recording, stream ordering or device memory visibility.

## CUDA 12.8 promoted-source component

The exact six-multiply schedule was compiled against promoted commit `1248235`
using its real `qsb_field_mul_raw` include closure in the existing ARM Linux
CUDA compiler VM. The component receives `invabc`; it does not execute the
inverse tree, zero masks, full digest kernel or GPU hardware.

```text
CUDA 12.8.93 sm89: PASS
trusted no-architecture build: PASS
registers: 90
stack / spill stores / spill loads: 0 / 0 / 0 bytes
shared memory / barriers: 0 / 0
instruction slots / non-NOP: 920 / 905
local-memory opcodes: 0
```

- `component/k3_split_audit.cu` SHA-256:
  `bcf56fd04f9ef59985a1eb34d8178c7fe830a7af51cf6eb816448d77c7f115ed`
- `component/native-results.json` SHA-256:
  `5d73ecec759f1ecf637aecdebbbf990dce6231b35cb94adef443e33db6ef141c`

Component allocation does not transfer to an integrated kernel. In particular,
this result says nothing about values live across three SHA/EC passes, scratch
loads, achieved occupancy, latency or throughput.

## Parked preinverse / finish seam

```text
PYTHONDONTWRITEBYTECODE=1 python3 check_parked_finish.py
PASS: 4101 parked-finish recoveries; 4099 omitted-ZZ mutations rejected

CUDA 12.8.93 sm89: PASS
trusted no-architecture build: PASS
registers: 88
stack / spill stores / spill loads: 0 / 0 / 0 bytes
shared memory: 0
instruction slots / non-NOP: 1096 / 1082
local-memory opcodes: 0
```

The bigint oracle covers boundary and seeded random projective scales and
scalar multiples. It compares both `P+R` and `P-R` with independent affine
addition. The mutation deliberately omits the required `ZZ` factor from the
parked first numerator; two `ZZ=1` boundary cases correctly survive it.

- `check_parked_finish.py` SHA-256:
  `563708c97ba57cfd1aa68383380446485ceff37cd6908da0014bbbc2c909bc3c`
- `component/k3_parked_finish_audit.cu` SHA-256:
  `d9c18306487ea0c64e3870960f50b987d52123358ccec29bce515ac5b23963c0`
- `component/native-finish-results.json` SHA-256:
  `8372798670ae3f6c28fe81cbcaf33e04857e4cfdef9859c4ee093fd9777b31d0`

This closes formula correspondence and isolated post-reload allocation only.
It does not validate global scratch visibility, the integrated three-epoch
kernel, GPU execution, or throughput.

## Exact pending K2 source and integrated K3 prototype

Submission `1344772` disclosed `7140869`'s immutable source commit
`a86816f56ceee3701a152fc0f7007033e796b319`. The advertised hashes for
`GPUMath.h`, `tree.cu`, and `window_schedule_shared.cuh` all matched after an
isolated clone. CUDA 12.8.93 compiled that exact K2 source and the K3 derivative
for sm89 and the trusted default target:

| Source | Registers | Shared | Stack | Spill store/load | Non-NOP | Work unit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact K2 | 128 | 40,960 B | 80 B | 8 / 8 B | 16,138 | 512 candidates |
| K3, A-first initial | 128 | 49,152 B | 104 B | 120 / 148 B | 24,105 | 768 candidates |
| K3, C-first retained | 128 | 49,152 B | 152 B | 96 / 100 B | 23,876 | 768 candidates |
| K3, C-first + B denominator scratch | 128 | 49,152 B | 152 B | 96 / 116 B | 23,883 | 768 candidates |
| K3, C-first + carried digit window | 128 | 49,152 B | 136 B | 60 / 64 B | 23,764 | 768 candidates |
| K3, carried window + park B/C pairs | 128 | 49,152 B | 96 B | 28 / 32 B | 23,762 | 768 candidates |
| K3, above + park B denominator | 128 | 49,152 B | 88 B | 12 / 12 B | 23,738 | 768 candidates |
| K3, above + park C denominator | 128 | 49,152 B | 72 B | 0 / 0 B | 23,739 | 768 candidates |
| K3, compact tree + shared B denominator (retained) | 128 | 49,152 B | 96 B | 24 / 24 B | 23,781 | 768 candidates |

Line-info SASS showed that A-first's extra local traffic occurs on the ordinary
recovery path, not only the rare hit path. Finishing C first shortens the live
range of its numerator pair, `leaf`, and `prodAB`; that lowers aggregate spills
from 120/148 to 96/100 bytes and static work from 24,105 to 23,876 non-NOPs.
Pre-materializing A/B inverses retained 96/100 spills at 23,880 non-NOPs.
Parking `invA` raised spills to 104/108, while parking B's denominator raised
loads to 116; both were rejected. The exact C-first source and dual-target
report are retained. Normalized non-NOP counts are 31.52/candidate for K2 and
31.09/candidate for retained K3, but this is not a speed result and does not
offset the remaining spill and scratch risks without GPU evidence.

The carried digit-window port initially used bit 35. Its dedicated oracle
rejected the first boundary scalar and established bit 36 as the correct start
(`gt_shift(2)+1`). After correction it passes 100,005 scalars / 1,300,065
digit comparisons and rejects 1,300,023 one-bit-shift mutations. CUDA 12.8.93
then produced the carried-window row above for both sm89 and the default target.

Line-mapped SASS localized most remaining carried-window local traffic to the
cooperative `zinv32` phase, where C recovery state and B/C denominators remain
live across the inverse. Explicitly parking those values progressively reaches
zero compiler spills. It is not retained: scratch round-trip rises from 128 to
384 bytes/lane before counting spills, while the retained layout's 128 explicit
+ 124 compiler-spill bytes/lane remains lower overall. Without GPU
latency/timing evidence, spill-free resource totals do not justify tripling the
explicit streaming scratch.

The retained alternative instead compacts the packed inverse tree from 24 to
16 KiB by overwriting each product level after a load-completion join. The
freed 8 KiB holds B's denominator, reducing spills to 24/24 without increasing
global scratch. A first version relied on implicit same-warp ordering; the
source-extracted CTA audit rejected it. The corrected source uses masked joins
for 2/4/8/16/32-lane phases and block joins for 64/128-lane phases.

```text
BEND_NO_TELEMETRY=1 bend COMPACT_TREE_PROOF.bend
All terms check.

python3 -B check_compact_tree.py
status=PASS
leaf_inverses=1024
uniform_barriers_per_lane=8
parent_index_mutation_rejections=252
```

The first structural audit found and rejected an inherited host bug: the loop
still divided `epochs_left` by two while advancing by three. The corrected
source divides by `QSB_K2S_MUL`. Since `C(137,6)=8,218,472,724` is divisible by
three, every epoch is covered without a compound-block tail.

```text
python3 -B check_integrated_k3.py
PASS: exact K2 identity, K3 source structure, scratch slots, and 8218472724 complete epochs

python3 -B check_integrated_k3_runtime.py
status=PASS
triple_schedules=4096
candidate_recoveries=12281
identity_substituted_singular_candidates=7
finish_order=C,A,B
```

- `check_integrated_k3.py` SHA-256:
  `5a8b841ff68a79c14bbbf7adb1c2e083f75c9474cb99aa8d8e62b025b74dbc7d`
- `check_integrated_k3_runtime.py` SHA-256:
  `9c640efc556c132408bd655596e0b097b27018bc9fbb9fede374406212256fd3`
- exact K2 report SHA-256:
  `37d77dcc6b72b8a01965af944c5816095074d04eafec1b7b9a8fb79d6cb9010c`
- retained K3 report SHA-256:
  `8cf1d049843ee447b606fd7436190f699e42846fe8f991ad5d8adce783540fde`
- carried-window oracle SHA-256:
  `a4df6e6ddc0fa145690726e1466c2ea086a9af16f5da3384c1f851f0a7417db5`
- carried-window K3 report SHA-256:
  `6aec68ca150c2589ee1dcc5164d625d7ea54f93feb7ef777720c8ad0d873fa28`
- B/C-pair parking report SHA-256:
  `4b6f96d337c55f3e87ece65fc0efbeb5bc0508d128836acd73a2598462a429ec`
- B/C-pair plus B-denominator report SHA-256:
  `e67b446c2c78a4abadbe6477e497c723fd50698e6a4df0c36755f33f4cd6a7cc`
- spill-free B/C-pair plus B/C-denominator report SHA-256:
  `09edbd79adf9f7c08f7aa44ff6bd7d8a4f850db77b559f9b478d846d8f0a3206`
- compact-tree laws / proof SHA-256:
  `49e99b6198c77b71aac1e0ff79127e4fc927c771673a501e84d04da6d35a6b1e` /
  `496c3cfadd09d305bd361337f62b63a70e34a6f0ba94c9fbc8dce24eda6d8d9b`
- compact-tree source audit / result SHA-256:
  `8b2a48d9b2b47fa8314581566322fd1a0b8174d3509b95b855e4a7909df64e17` /
  `76e201ef25709e2deba1b1552f6722a72d4c7ae36b8da6c3d68f35e0d5be3543`
- retained compact-tree CUDA report SHA-256:
  `7309816beb434e7ca1e1f5d7726fc967bb58eae62df27bd5f2675ad392d57b21`
- rejected global-denominator report SHA-256:
  `f17f57c31ef0bdce122d33cd63165cbc63736f9a6fe0db3ebdc99018307b8a03`

This prototype is not submission-qualified. Static reports do not locate every
spill on the ranked critical path, no NVIDIA GPU was executed, and there is no
evidence it beats `7140869`'s measured +5.25% stack.
The live comparison is stricter: `b76b3d5` claims +5.62% after adding the
register-carried digit window. The runtime audit uses source-extracted helpers
and the exact split schedule, but OpenSSL replaces PTX arithmetic and it is not
GPU timing evidence.

After the frontier advanced to promoted `591a223` at 536,484,898/s, pending
`f63b274` disclosed a rare cooperative-inverse top-limb truncation defect. The
active K3 `zinv32.cuh` now shifts the signed 64-bit accumulator before narrowing
(`e8385842bbddcdd7023dac5fcbbf4d01780cb128494896dc1375db52f359da11`).
The Bend accumulator laws still pass and the integrated runtime audit passes
with source fingerprint
`2f8b41e0b79b360a02c28cae4338b5ac81687a059a5c7a513f8b3808d6ce312f`.
The earlier CUDA resource report predates this one-line repair and is retained
only as static evidence for the compact-tree layout, not as a current binary
receipt.

## Sparse-SHA K3 successor

The first successor snapshot is `k3_sparse_successor/`. It adds the promoted
pinning lineage's fixed-padding SHA schedules referenced by pending `f63b274`,
raises the launch to 262,144 blocks, and lowers the requested device stack limit
to 2,048 bytes. The actual header is compiled into a host shared library and
executed by `check_sparse_sha.py`: 8,198 digest/pubkey cases match `hashlib`,
and all 8,198 wrong-length mutations differ. Source hashes:

- `sparse_sha_fixed.cuh`:
  `0a9f63eba13ae928ca1c2b4996294d81585fbe9c8a5e3a43c8ae144518d15300`
- successor `tree.cu`:
  `c4b997fcbafd8e8c0c111724ec6d9305d3328cddce10d474fd113b4d88f28ff0`
- `check_sparse_sha.py`:
  `bc0fa335fd6c077861adb68fc9699ca044928e6b8472cf54302e78e195dc9704`
- CUDA 12.8.93 native report:
  `2a07fa753f3dfb6bef228791d75c665c65ed52425417b03cb8c2350e9c22bee7`

Both sm89 and trusted-default builds pass. The digest kernel holds the submitted
K3 resource class (128 registers, 49,152 shared bytes, 96 stack bytes, 24/24
spill store/load bytes) while non-NOP slots drop 23,781 -> 23,731. This remains
static codegen, not GPU timing. It is a successor research snapshot, not a
replacement for validating submission `27742458`.
