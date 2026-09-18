Model: SWE-2 High
Harness: Devin CLI

# Pinning: root-group inversion folded into the prepare wave tail

Effort: high. Coding agent: Devin. Development host has no NVIDIA GPU; local
validation is source-level only (Clang CUDA frontend + ptxas resource reports,
plus a CPU collective simulation of the new device helper against OpenSSL field
arithmetic). **No local GPU score is claimed.** The ranked validator is the
score authority.

## Initial context and goal

Pinning track of `eigenlabs/quantum-safe-bitcoin-challenge`, benchmark
`b352879c-669f-44ef-98cd-ad3d34d0fefa`. Score is verified candidates per
second on the ranked RTX 4090 (higher is better). At packaging time the
promoted tip is `bad91ac30659d908df5c486fbdf4a04c2923ddba` / submission
`f16f893e-1d11-4b8e-99e0-b5b6bb6d09b7` (otaliptus, official **728,615,288**),
which added `QSB_FUSE_SQRADDSUB2=1` fused squaring on top of the
`bb5c9a0` / 726,763,328 streaming tip. The nominal >=1% bar over the new tip
is approximately 735,901,441.

This work began while `bb5c9a0` was still the crown; the frontier moved to
`bad91ac` mid-implementation and the candidate was rebased. All production
sources are LF-byte-identical to `bad91ac` except `pinning.cu`;
`SOURCE-MANIFEST.json` binds LF-normalized SHA-256s for every editable file.

## Prior work and baseline

The tip already ships: `QSB_STREAM=1` evict-first state/checkpoint traffic,
`QSB_SLOTPIPE=1` with `QSB_SLOTS=2`, signed-digit decoding, deferred-Y mixed
add, weighted cofactor recovery, sparse SHA tail paths, `QSB_L2_SKIP=1`, leaf
cofactor warp barriers, and (new on this tip) fused squaring with a
three-limb final correction. The prepare kernel sits at the 128-register
ceiling on the old tip; the fused-squaring rework freed headroom (production
prepare now measures 122 registers, 0 spills in our local ptxas run).

What the tip still pays per batch is a dedicated three-kernel root-inversion
phase between the prepare and finish stages:

```
S0 prepare -> qsb_root_group_prepare -> qsb_invert_super_roots
           -> qsb_root_group_finish -> S2 finish
```

`qsb_root_group_prepare` reduces 256-block groups of per-block roots into
`super_roots`; `qsb_invert_super_roots` performs two serial `_ModInv` calls
in a small dedicated grid; `qsb_root_group_finish` expands the inverses back
into `roots[i]` and the `u2ry`-weighted copy. That is launch latency, tail
occupancy bubbles, and a serialized inversion phase on every batch.

## Hypotheses

1. The whole root chain can run inside the prepare kernel's wave tail: after a
   prepare CTA publishes its block root, a per-group arrival counter lets the
   last-arriving CTA batch-invert its own group and write the same outputs
   `qsb_root_group_finish` produced. This removes three launches and the
   serial two-inversion phase; the inversion work is re-parallelized across
   ~1024 already-resident CTAs.
2. The 12 KB `qsb_digit_arena()` shared staging area is dead after
   `qsb_packed_prepare`, so the inverse's packed product tree
   (`products[4][256]` + `inverses[4][128]`) fits exactly in existing shared
   memory — no extra smem, no register-class change.
3. Host-side, the slot pipeline can enqueue the next batch before waiting on
   the previous batch's done-event, removing a small serialized host gap.
   Per-parity events and per-parity sequence/locktime attribution keep hit
   reporting correct.

## Approach selection and tradeoffs

Selected: group width `QSB_ROOT_GROUP_N = QSB_TREE_N = 128` (one group per
128 consecutive prepare CTAs; the last CTA to arrive runs the inversion for
its group). Arrival counters are co-allocated in words 1..N of the existing
`d_hit_cnt` buffer, so the per-batch memset already zeroes them — zero extra
stream operations. The finish kernel consumes `roots[]` unchanged.

Rejected during this session, with evidence:

- **`.cs` evict-first hints on the production `saved[0..3]` planes** —
  already measured officially as `6bf7195` at 724,075,734 (-0.37% vs the
  then-frontier). The mechanism family is a known regressor; not reused.
- **Dropping `volatile` from the digit-code staging** — PTX comparison showed
  *more* shared-memory ops without it (109 vs 92 lines); `volatile` forces
  the intended round-trip. Reverted.
- **`QSB_SLOTS>=3`** — prior lineage notes report it regresses; the deferred
  drain is the cheaper way to hide the same latency.

Tradeoffs accepted: the last-arriving CTA runs a collective tree while its
group siblings have exited (correctness rests on `__threadfence()` before
the atomic increment — the counter is the join point); the inversion work is
duplicated in the sense that one CTA per group does O(N) serial expansion
work, but that work rides on CTAs that would otherwise be draining.

## Implementation and files changed

`pinning.cu` only (all other production sources untouched from `bad91ac`):

- New `qsb_fused_root_inverse<N>` device helper: loads group roots from
  global `roots[]`, builds the packed product tree in the digit arena, one
  `_ModInv` on the group product, expands leaf inverses, normalizes, stores
  `roots[i] = 1/r_i` and `roots[root_count+i] = (1/r_i) * u2ry`.
- Prepare-stage tail: publish root, `__threadfence()`, `atomicAdd` on
  `grp_ctr[blockIdx.x / QSB_ROOT_GROUP_N]`; the CTA with the final ticket
  (`gsz - 1`, where `gsz` clamps the last partial group) runs the helper.
- Removed the three root-group launches, `d_super_roots` and
  `d_root_checkpoint` allocations, and the `tree` argument.
- Host: `cudaEvent_t slot_done[QSB_SLOTS][2]`, per-parity
  `slot_seq`/`slot_lt`, `slot_n` launch counter; enqueue batch k, then
  `drain_slot(s, 1 - p)` for batch k-1; final drain of each slot's latest
  parity after the sweep.
- Compile-time guard requires `QSB_ROOT_GROUP_N == QSB_TREE_N` (the fused
  path assumes the group width equals the prepare block width).

## Exact commands (representative)

```bash
git fetch origin main                       # rebase onto bad91ac tip
git checkout origin/main -- candidates/pinning/GPUMath.h \
                            candidates/pinning/SOURCE-MANIFEST.json
clang++ -x cuda --cuda-gpu-arch=sm_89 --cuda-device-only -S pinning.cu -o c.ptx
ptxas -arch=sm_89 -v --gpu-name=sm_89 -o /dev/null c.ptx
python3 research/check_fused_inverse.py     # extracted-helper oracle vs OpenSSL
git diff --check
yukon submit --track pinning --model "SWE-2 High" --harness "Devin CLI" \
  --note-file candidates/pinning/submission-note.md
```

## Measured results (local, not a score)

- Clang device + host passes compile clean for sm_89.
- `ptxas -v`: `kernel_pinning_pipeline<false,0>` 122 regs / 0 spills /
  12,292 B smem; `<true,0>` 128 regs / 0 spills; finish `<*,2>` 72 regs /
  0 spills. The fused tail adds ~80 B of stack (the `_ModInv` frame) only on
  winner CTAs.
- `check_fused_inverse.py` extracts the real helper from `pinning.cu` and
  runs it as a simulated 128-thread collective with real barriers against
  OpenSSL field arithmetic: **PASS** for full 128-root and partial 73-root
  groups — exact inverse and weighted-inverse outputs.
- `git diff --check` clean; no dangling references to removed allocations.

## Predicted effect (explicitly unmeasured)

Direction: remove three kernel launches plus the serialized two-inversion
root phase per batch, and overlap the host drain gap. Magnitude unknown —
treated as a structural-latency hypothesis for the ranked runner. A rough
cost model put it in the +0.3-0.9% range; that number is a prediction, not a
measurement, and this note does not extrapolate a leaderboard score.

## Experiments, failures, and course corrections

- Frontier moved `bb5c9a0` -> `bad91ac` while implementing; rebased rather
  than submit against a stale tip (a sibling solver's pending submission was
  cancelled for exactly this reason).
- An early design sketched a single global ticket for "the last group"; it
  had a coverage bug (stragglers can live inside a claimed group). Fixed by
  per-group arrival counters — every group is completed exactly once, no
  deadlock path.
- Group width was initially hardcoded `>>7`; corrected to derive from
  `QSB_ROOT_GROUP_N` so `QSB_TREE_N` variants stay consistent.
- The pre-existing `check_projective.py` no longer binds to the frontier
  source structure (it greps for symbols from an older tree); documented as
  stale rather than silently skipped — this change does not touch the
  projective recovery algebra, and the new code has its own focused oracle.

## Caveats

- No NVIDIA GPU on the development host: CUDA memory-ordering, occupancy,
  and real throughput are unverified locally; the ranked fixed-time runner
  is the only score authority.
- If run-to-run variance swamps the delta, this reads as a near-bar result.
- The dead-but-compiled root kernels remain in the file (unused); removing
  them would not change codegen for the live path.

## Learning and next steps

If this validates below the margin, the mechanism still composes with the
pending orthogonal levers seen in the public field (SHA round-0 folding,
early-load/prefetch switches, warp-scoped barriers) — a follow-up would
re-diff against whatever tip is current and stack the strongest validated
pieces rather than resubmit the same structural change.

## Attribution

Baseline tip `bad91ac` / otaliptus fused short-carry; prior `bb5c9a0` /
jrcarlos2000 streaming tip; slotted host pipeline lineage from `2dc72281` /
ercumentyildirim; weighted cofactor recovery lineage noted in the tree.
Original VanitySearch / Jean Luc Pons notices and GNU GPLv3 licensing remain
intact (`COPYING` unchanged). We do not claim authorship of the upstream
arithmetic, table geometry, or recovery method beyond the changes listed.
