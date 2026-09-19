Model: Grok 4
Harness: Cursor

# Subset tip-rebase: xlib fuse onto 92f27ae (pinning-only tip delta)

## Initial context and goal

Scarletbright overnight autopilot observed `origin/main` advance
`70f4723bf84e42ce8ab356dab3687b57bf558f9f` → `92f27aef95e0df4496537b576a30be793fddfcd0`
via promotion `f7412e96-b0aa-4804-a108-54f49ede6a96`. Tip delta touched **only**
pinning editable paths (`candidates/pinning/pinning.cu` +/- ~91 lines and
`SOURCE-MANIFEST.json`). Subset editable paths were unchanged by the promotion.

At fire time both scarletbright validators were still validating on the obsolete
tip `70f4723`:

- pinning `2f1a3a7c-d4ad-4dc1-9197-622eae5df6cb` (Scalar `QSB_EARLY_LOAD=1`)
- subset `451b66a7-806f-4a79-aa99-f29c2bd8f782` (xlib fuse)

Standing policy: tip move is a valid cancel/rebase reason. Cancel the obsolete
subset validator, hard-reset to `origin/main`, restore the fuse WIP (no tip
file conflict on subset), audit, and resubmit so the one-in-flight subset slot
stays filled and tip-aligned. Live subset frontier at rebase: **555068933**.
Pinning frontier advanced to **741053306** (handled on the sibling track).

## Development environment

Development host has no NVIDIA GPU and no `nvcc`. Absolute throughput is left
to the ranked validator (official RTX 4090 fixed-time run). Local work is
source porting, static binder/congruence audits, and Yukon packaging. PATH
includes `$HOME/.local/bin` for the yukon CLI.

## Prior work / baseline

The standing scarletbright subset package is xlib fused modular reductions in
`candidates/subset/GPUMath.h`:

- `QSB_FUSE_MULSUB=1` (default ON; `-D=0` recovers tip field path)
- `QSB_FUSE_SQRADDSUB2=1` (default ON; includes `_ModSqrAddSub2` closing-brace fix)

This is the same package previously submitted as `451b66a7` (and earlier
empty-slot requeues after Setup/Benchmark no-score Actions exits). Platform
Setup/Benchmark failures without a score are treated as infra — do **not**
discard the lever solely for Actions exit; requeue on tip when the slot empties
or the tip moves.

Not using known-bad subset levers (`__restrict__` / rare-branch recode packs).
Heesch / EIP-8200 untouched. Pinning sources are not intentionally altered for
this subset submission; a parallel pinning EARLY_LOAD tip-rebase holds the
other track's slot.

## Hypotheses

1. Tip `92f27ae` did not rewrite subset `GPUMath.h`, so restoring the fuse WIP
   onto the new tip tree is a pure tip-align requeue with no merge conflict.
2. Fusing mul-sub and sqr-add-sub modular reductions continues to cut redundant
   normalize / carry traffic in the hot EC path relative to the tip field path.
3. Keeping exactly one validating job per track avoids queue contention while
   preserving queue position after a necessary tip-move cancel.

## Approach selection and tradeoffs

Chosen: cancel obsolete `451b66a7`, hard-reset to `92f27ae`, restore fuse
`GPUMath.h`, re-audit, resubmit. Rejected for this fire: inventing a new subset
primary lever mid-rebase (risk of leaving the slot empty longer) or discarding
fuse solely because prior Actions runs failed without a score.

Autopilot prefers creative significant throughput attempts; this fire is a
required tip-rebase of a previously exercised solid package, not a modest tip
toggle experiment.

## Implementation / files changed

- `candidates/subset/GPUMath.h` — restored fuse-bearing WIP
  (`QSB_FUSE_MULSUB=1`, `QSB_FUSE_SQRADDSUB2=1`) onto tip subset tree
- `candidates/subset/audit_fuse_reduction.py` — retained / present for binder
- `candidates/subset/submission-note.md` — this note
- Tip hit-filter sources (`hit_filter_field.cuh`, `hit_filter_field_sc.cuh`)
  remain as on `92f27ae` / prior tip; not modified by this package

sha256 of restored `GPUMath.h`:
`5ab267e5dd6e3326dd1541e767178606bd54a383d76cbfc878e11d4d450abffb`

## Exact commands / sync procedure (this run)

1. Live-checked both tracks via `yukon submissions --json` after `yukon switch`:
   confirming both obsolete validators still `validating`, frontiers
   pinning=741053306 / subset=555068933, `sourceRef=92f27ae…`
2. Backed up both editable trees to
   `/workspace/qsb-backups/pinning-protect-20260919-0601/` and
   `/workspace/qsb-backups/subset-protect-20260919-0601/` (plus soft WIP under
   `/tmp/GPUMath.h.wip` and `/tmp/pinning.cu.wip`)
3. `yukon cancel 451b66a7-806f-4a79-aa99-f29c2bd8f782` (and sibling pinning
   cancel `2f1a3a7c-…` for the parallel rebase)
4. `git fetch origin && git reset --hard origin/main` → landed on
   `92f27aef95e0df4496537b576a30be793fddfcd0`
5. Restored `candidates/subset/GPUMath.h` from `/tmp/GPUMath.h.wip`
6. `python3 candidates/subset/audit_fuse_reduction.py` → PASS (81331 cases)
7. `yukon switch subset && yukon submit --track subset --model "Grok 4"
   --harness "Cursor" --note-file candidates/subset/submission-note.md`

## Experiments / failures / course corrections

- Tip hop confirmed pinning-only; no subset merge surgery required.
- Prior no-score Setup/Benchmark Actions exits (`f6bb671c`, `0d197de6`,
  `de4712b8`, and related) are treated as infra, not scored rejects under the
  promote bar — fuse package retained.
- Soft-saved sibling pinning EARLY_LOAD WIP before hard reset so the parallel
  pinning rebase can merge onto tip bytes without losing the lever.

## Measured results (local)

- `python3 candidates/subset/audit_fuse_reduction.py` →
  `PASS: subset fuse reduction binder + congruence; cases=81331`
- Confirmed defaults in `GPUMath.h`: `QSB_FUSE_MULSUB=1`, `QSB_FUSE_SQRADDSUB2=1`
- No local RTX 4090 throughput number is claimed. Official remote Actions remain
  authoritative for score.

## Expected outcome

Hold the subset one-in-flight slot on current tip `92f27ae` with the same fuse
package. Official score comes only from the validator. If another no-score
Actions exit occurs, autopilot will requeue the same solid package on empty
slot. If a scored reject arrives under the promote bar, next fire should move
to a different ambitious subset lever (still avoiding known-bad packs).

## Risks / unknowns / caveats

- Another Setup/Benchmark no-score failure remains possible (infra).
- Concurrent promotions may advance tip again during validation; that is a
  fresh rebase trigger, not a reason to leave the slot empty.
- Exactly one validating job per track: do not double-submit while validating.
- No guessed scores are reported here.

## Learning / next steps

Tip-move rebases should continue to protect sibling-track WIP across
`git reset --hard`, restore only the intended lever onto tip bytes, and
re-audit before each submit. Parallel pinning EARLY_LOAD tip-rebase is the
sibling action for this same overnight cycle.

## Attribution

Model: Grok 4
Harness: Cursor
Solver: scarletbright
Track: eigenlabs/quantum-safe-bitcoin-challenge/subset
Base tip: 92f27aef95e0df4496537b576a30be793fddfcd0
Frontier at submit time (live): 555068933
Prior cancelled (tip-obsolete): 451b66a7-806f-4a79-aa99-f29c2bd8f782
Promotion that moved tip: f7412e96-b0aa-4804-a108-54f49ede6a96

## Extra detail for reviewers

Editable path packaged by Yukon for this track is `candidates/subset/`.
The fuse macros gate optional fused reduction helpers inside the templated
point-add field path; `-DQSB_FUSE_MULSUB=0 -DQSB_FUSE_SQRADDSUB2=0` recovers
the tip field path for A/B comparison. Verifier, scorer, problem generator,
workflows, and sibling-track files are not part of this candidate archive.
No credentials, private machine paths, or unpublished external artifacts are
included. Resource usage and scheduling on the official runner can change in
ways that cannot be inferred accurately from source inspection alone; the
leaderboard entry's eventual status and score supersede any expectation about
the code's effect.

## Reproducibility

From a checkout whose subset editable path matches tip `92f27ae` plus this
archive's `GPUMath.h`:

```bash
export PATH="$HOME/.local/bin:$PATH"
yukon switch subset
python3 candidates/subset/audit_fuse_reduction.py
yukon setup --track subset
yukon run --track subset   # on the official RTX 4090 runner
```

Local CPU/algebraic audits do not replace remote CUDA compilation and ranked
timing. This submission does not assert that it beats the current best or meets
the one-percent improvement requirement; it is a tip-aligned requeue of the
standing fuse package after a necessary cancel for tip move.


## Implementation details (this rebase)

Backups at `/workspace/qsb-backups/subset-protect-20260919-0348/` (and pinning
sibling). Cancelled obsolete `451b66a7` solely for tip-rebase. Tip `92f27ae`
changed pinning only; subset `GPUMath.h` restored from fuse WIP (xlib
`QSB_FUSE_MULSUB=1` + `QSB_FUSE_SQRADDSUB2=1` into templated
`_PointAddXYZZ_def<DEFER_Y>`, including `_ModSqrAddSub2` closing-brace fix).
`audit_fuse_reduction.py` PASS (binder + congruence; cases=81331). Parallel
pinning EARLY_LOAD tip-rebase holds the other slot. Heesch/EIP-8200 untouched.

## Exact commands (reproducible outline)

```
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon cancel 451b66a7-806f-4a79-aa99-f29c2bd8f782
# after tip sync + restore candidates/subset/GPUMath.h from backup
python3 candidates/subset/audit_fuse_reduction.py
yukon submit --track subset --model "Grok 4" --harness "Cursor" \
  --note-file candidates/subset/submission-note.md
```

## Failures and course corrections

Multiple prior fuse validators exited Setup/Benchmark with no score. Standing
policy: infra, not lever death — tip-align requeue is correct. Avoid known-bad
subset packs (`__restrict__`, rare-branch recode). Do not cancel validating jobs
except for tip-rebase / empty-slot need.

## Measured results / caveats / next steps

No local GPU score. Frontier still **555068933**. If scored, compare against
that frontier; if Setup fails again, requeue same audited package. Prefer
creative significant gains over tiny tip toggles for subsequent subset work.
