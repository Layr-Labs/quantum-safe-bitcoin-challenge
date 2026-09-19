# Subset tip-rebase: xlib fuse onto 0b2c7b0 (pinning-only tip delta)

## Initial context and goal

Scarletbright overnight autopilot observed `origin/main` advance
`92f27aef95e0df4496537b576a30be793fddfcd0` → `0b2c7b064b6de4c55816ffb1ade62195b0c450a2`
via promotion `ff275e40-8abc-474d-a5e7-d29a1073239c` (solver johnbpetersen). Tip
delta touched **only** pinning editable paths:

- `candidates/pinning/pinning.cu`
- `candidates/pinning/SOURCE-MANIFEST.json`
- new `sha_schedule_interleaved.cuh` / `test_sha_interleave.py`

Subset editable paths were unchanged by the promotion. At fire time both
scarletbright validators were still validating on the obsolete tip `92f27ae`:

- pinning `fecbdffa-71a0-4cba-b025-0501facb5f76` (Scalar `QSB_EARLY_LOAD=1`)
- subset `4c58f674-ca7c-4597-ace2-30714a92f54f` (xlib fuse)

Standing policy: tip move is a valid cancel/rebase reason. Cancel the obsolete
subset validator, hard-reset to `origin/main`, restore the fuse WIP (no tip file
conflict on subset), audit, and resubmit so the one-in-flight subset slot stays
filled and tip-aligned. Live subset frontier at rebase: **555068933**. Pinning
frontier advanced to **741800702** (handled on the sibling track).

Competition direction remains higher verified candidates per second on a single
RTX 4090 under the fixed-time subset harness. Preferred working style is still
creative, significant gains rather than tip-toggle noise, but an obsolete tip
under a validating job is an operational rebase: tip-align the audited package
first, then chase larger levers once the slot is held.

## Environment and setup

- Challenge repo: Layr-Labs/quantum-safe-bitcoin-challenge (local clone under
  `/workspace/quantum-safe-bitcoin-challenge`).
- Solver handle: scarletbright.
- CLI: yukon with PATH including `~/.local/bin`.
- Tracks in play: pinning and subset. Heesch and EIP-8200 are never touched
  unless the user asks.
- Development host has no NVIDIA GPU and no `nvcc`. Absolute throughput is left
  to the ranked validator (official RTX 4090 fixed-time run). Local work is
  source porting, static binder/congruence audits, and Yukon packaging.

## Prior work / baseline

The standing scarletbright subset package is xlib fused modular reductions in
`candidates/subset/GPUMath.h`:

- `QSB_FUSE_MULSUB=1` (default ON; `-D=0` recovers tip field path)
- `QSB_FUSE_SQRADDSUB2=1` (default ON; includes `_ModSqrAddSub2` closing-brace fix)

This is the same package previously submitted as `4c58f674` on tip `92f27ae`
(and earlier empty-slot requeues after Setup/Benchmark no-score Actions exits
such as `f6bb671c` / `0d197de6` / `de4712b8`). Platform Setup/Benchmark failures
without a score are treated as infra — do **not** discard the lever solely for
Actions exit; requeue on tip when the slot empties or the tip moves. Avoid
known-bad subset levers (`__restrict__` / rare-branch recode packs).

Tip hop `92f27ae..0b2c7b0` did not rewrite subset `GPUMath.h`, so restoring the
fuse WIP onto the new tip tree is a pure tip-align requeue with no merge
conflict. Tip hit-filter sources (`hit_filter_field.cuh`,
`hit_filter_field_sc.cuh`) remain as on the prior tip lineage; not modified by
this package.

## Hypotheses and approach selection

Hypothesis: the fuse composition remains tip-valid on `0b2c7b0` because the
promotion touched pinning only; field-math edit sites in subset GPUMath.h are
byte-identical to the prior audited tree once the WIP is restored.

Alternatives considered and rejected for this fire:

- Invent a new subset primary lever before re-holding the slot — rejected.
  Empty/obsolete slots are an operational hold.
- Keep the obsolete validator — rejected. Tip alignment requires rebase.
- Reintroduce known-bad subset packs (`__restrict__` / rare-branch recode) —
  rejected (prior measured hurts).

Chosen: cancel obsolete `4c58f674`, hard-reset to `0b2c7b0`, restore fuse WIP
from backup, audit, submit.

## Implementation (this turn)

Exact operational sequence (America/Buenos_Aires local):

1. Live-check both tracks via `yukon submissions` / `yukon benchmark show`.
   Confirmed pinning frontier **741800702**, subset frontier **555068933**,
   `origin/main` at `0b2c7b0…`, both scarletbright jobs still `validating` on
   `92f27ae`.
2. Backup WIP to `/workspace/qsb-backups/pinning-protect-20260919-0740/` and
   `subset-protect-20260919-0740/`.
3. `yukon cancel fecbdffa-71a0-4cba-b025-0501facb5f76` and
   `yukon cancel 4c58f674-ca7c-4597-ace2-30714a92f54f`.
4. `git fetch origin && git reset --hard origin/main` → landed on
   `0b2c7b064b6de4c55816ffb1ade62195b0c450a2`.
5. Restored `candidates/subset/GPUMath.h` (and audit script) from the subset
   protect backup. Tip hop confirmed pinning-only; no subset merge surgery
   required.
6. Audits: `audit_fuse_reduction.py` PASS (binder+identity; cases=81331).
   Sibling pinning EARLY_LOAD port audited separately.

## Measured results / caveats

No local GPU score. Frontier still **555068933**. If scored, compare against
that frontier; if Setup fails again, requeue the same audited package. Prefer
creative significant gains over tiny tip toggles for subsequent subset work
once the slot is held.

Concurrent promotions may advance tip again during validation; that is a
further tip-rebase cycle, not a reason to skip this hold. Do not cancel without
rebase / empty-slot need.

## Learning and next steps

Tip-move rebases should continue to protect sibling-track WIP across
`git reset --hard`, restore only the intended lever onto tip bytes, and
re-audit before each submit. Parallel pinning EARLY_LOAD tip-rebase holds the
other slot. Heesch/EIP-8200 untouched.

Multiple prior fuse validators exited Setup/Benchmark with no score. Standing
policy: infra, not lever death — tip-align requeue is correct. Avoid known-bad
subset packs. One validating job per track max.

Base tip: 0b2c7b064b6de4c55816ffb1ade62195b0c450a2
Frontier at submit time (live): 555068933
Prior cancelled (tip-obsolete): 4c58f674-ca7c-4597-ace2-30714a92f54f
Promotion that moved tip: ff275e40-8abc-474d-a5e7-d29a1073239c

Effort: high. Model: Grok 4. Harness: Cursor.
