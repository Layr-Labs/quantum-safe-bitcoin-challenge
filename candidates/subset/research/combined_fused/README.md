# Combined fused small48 successor

Frozen source fingerprint:
`581e22633f7120ace8605576e64c5022ac67b93660920b100aebd5944e35f8e9`.

`prepare.py` composes three individually reviewed components and refuses to
overwrite the result:

1. Exact cubic-recovery fused source
   `7afe2d8642468c1245febc0aa9d159c76e6724e44035497a1d6623b75256c37e`.
   This retains the corrected PR77 fused architecture, guarded fourteen-window
   geometry, bounded mandatory 48 MiB + 4 GiB table and optional L2 policy.
   Its recovery identity saves three field squares with two extra modular
   additions/subtractions and unchanged multiply count per ordinary candidate.
2. Exact repaired leaf-pair header from
   `94e46e120d921d87273fd9e50584cc2294ba58c57f9de6e7c8aa67b102a14b5f`.
   It retains the uniform n=64 block join that fixes the public PR138 race.
3. Exact PR137 host `qsb_pack_second_classes` body, hash
   `5b18ba41b5d44f423691b3115d2e6c0e11c75aafdc059871eca90fdf27401caa`.
   The new call follows the existing host sort and precedes both `WIN3` upload
   and hash-schedule preparation. Existing schedule and device hash bodies are
   unchanged. No paired-SHA or host-drain source was imported.

Relative to the cubic base, only `tree.cu`, `tree_inverse.cuh` and
`window_schedule_shared.cuh` change. `tree.cu` changes by exactly one host call;
its fused kernel and cubic recovery bodies remain exact. The complete identity,
function hashes and provenance are in `prepared-source.json`.

The PR137 source is IvanLudvig's unpromoted commit
`103a6adc15391e07aac210a2653f8dfd7eb4d4c0`; its exact public header snapshot is
retained here. The PR138 source is AbdelStark's unpromoted commit
`2dc49dc4e4083050ffc34b9be61eb027eb8c291f`; its original snapshot and correction
are retained in `../leaf_pair`. Credit **IvanLudvig and AbdelStark** as coauthors
if these contributions are submitted before promotion, in addition to the
existing alvaroborras, MakiRH4, jacklightChen and ercumentyildirim provenance.
Original GPL notices and `COPYING` are preserved.

The packer's separate CPU audit preserves all 256 windows and 54/56 first/second
classes. Its sector model changes the sum of second-schedule sector groups
from 14 to 13 with the tested first-state bank model unchanged. These are address
models, not cache measurements. Native component results reported at preparation
time have identical 128-register/24 KiB shared/336-byte stack/no-spill resources:
leaf-pair has 30205 non-NOP instructions, cubic recovery has 29536, versus
29902 for the common fused base. Those numbers cannot be added to predict this
combined kernel, its cache behavior or performance.

The composition was followed by exact-source checks recorded below. No GPU
execution or throughput is claimed. The pending submission and its preserved
production source are unchanged.

## Exact combined validation

All final-source frontend, recovery, actual-chain/recovery, builder and native
production/audit checks pass; see validation-summary.json. Kernel resources are
128 registers, 24 KiB shared memory, 336 stack bytes, zero reported spills and
29830 static non-NOP instructions. There is no GPU measurement. This exact
closure is staged for the next available subset slot; submission-ready.json
records the package and checks. Refresh the frontier and pending state before
upload. Never cancel c571025c.
