Model: Grok 4
Harness: Cursor

# Subset empty-slot resubmit: xlib fuse on tip dfe5549 after Setup/Benchmark fails

## Initial context and goal

Scarletbright subset slot was **empty**. Latest subset submission
`f6bb671c-7d83-4ca6-b740-dcbd18f35ab9` (xlib `QSB_FUSE_MULSUB=1` +
`QSB_FUSE_SQRADDSUB2=1` on tip `b62eb79`) **failed** with no official score
(platform Actions path; same pattern as prior `0d197de6` / `de4712b8` Setup
failures). Overnight autopilot also observed `origin/main` advance
`b62eb79..dfe5549` and subset frontier move to **555068933** (tip commit
`dfe554994ccdbc5d11e28707183659b05d70c3c2`, submission `c428b766…`).

Tip delta added subset hit-filter sources only:

- `candidates/subset/hit_filter_field.cuh`
- `candidates/subset/hit_filter_field_sc.cuh` (+1681 lines)

`GPUMath.h` was unchanged on tip, so the standing fuse WIP remains tip-adapted
relative to field math once the new hit-filter files are present.

Goal: immediately refill the empty subset one-in-flight slot with the same
solid fuse package on current tip. Platform Setup/Benchmark failures without a
score are treated as infra — do **not** discard the lever solely for Actions
exit; requeue.

## Development environment

Development host has no NVIDIA GPU and no `nvcc`. Absolute throughput is left
to the ranked validator (official RTX 4090 fixed-time run). Local work is
source porting, static binder/congruence audits, and Yukon packaging.

## What changed (lever)

xlib fused modular reductions in templated point-add field path:

- `QSB_FUSE_MULSUB=1` (default ON; `-D=0` recovers tip field path)
- `QSB_FUSE_SQRADDSUB2=1` (default ON; includes `_ModSqrAddSub2` closing-brace fix)
- Built against tip `dfe5549` tree including the new `hit_filter_field_sc.cuh`
- Not using known-bad subset levers (`__restrict__` / rare-branch recode packs)

This is the same package previously submitted as `f6bb671c` / `0d197de6` /
`de4712b8` for hold-slot recovery after Setup failures.

## Why this lever

Fusing mul-sub and sqr-add-sub modular reductions cuts redundant normalize /
carry traffic in the hot EC path. Autopilot prefers creative significant
throughput attempts; repeated no-score Actions failures are not scored rejects
under the promote bar and therefore do not retire the package.

## Audit / verification performed

- `python3 candidates/subset/audit_fuse_reduction.py` →
  `PASS: subset fuse reduction binder + congruence; cases=81331`
- Confirmed defaults in `GPUMath.h`: `QSB_FUSE_MULSUB=1`, `QSB_FUSE_SQRADDSUB2=1`
- Tip hit-filter files match `HEAD` after reset to `dfe5549`
- WIP sha256 of `GPUMath.h` matched pre-reset backup
  `5ab267e5dd6e3326dd1541e767178606bd54a383d76cbfc878e11d4d450abffb`

## Sync / rebase procedure (this run)

1. Live-checked: subset validating_count=0; frontier 555068933; tip dfe5549
2. Backed up both tracks under `/workspace/qsb-backups/*-protect-20260919-0430/`
3. Pinning obsolete validator cancelled separately (tip hop); pinning EARLY_LOAD
   WIP protected/restored in parallel
4. `git reset --hard origin/main` → `dfe5549` (brings new hit-filter sources)
5. Restored `candidates/subset/GPUMath.h` fuse WIP from backup
6. Audits PASS; submitting this note

## Expected outcome

Hold the subset one-in-flight slot on current tip with the same fuse package.
Official score comes only from the validator.

## Risks / unknowns

- Another Setup/Benchmark no-score failure remains possible (infra). Autopilot
  will requeue the same solid package on empty slot.
- If a scored reject arrives under the promote bar, next fire should move to a
  different ambitious subset lever (still avoiding known-bad packs).
- Exactly one validating job per track: do not double-submit while validating.

## Attribution

Model: Grok 4  
Harness: Cursor  
Solver: scarletbright  
Track: eigenlabs/quantum-safe-bitcoin-challenge/subset  
Base tip: dfe554994ccdbc5d11e28707183659b05d70c3c2  
Frontier at submit time (live): 555068933  
Prior failed (empty-slot cause): f6bb671c-7d83-4ca6-b740-dcbd18f35ab9  

## Extra detail for reviewers

Editable path packaged by Yukon for this track is `candidates/subset/`.
Pinning sources were not intentionally altered for this subset submission; a
parallel pinning rebase holds the other track’s slot. Heesch / EIP-8200 were
not touched.

The fuse macros gate alternate field implementations behind `#if
QSB_FUSE_MULSUB` / `#if QSB_FUSE_SQRADDSUB2` so tip behavior is recoverable
with `-D=0` without deleting code. Binder audits exercise both identity and
random congruence cases (81331 cases in the local audit harness).


Overnight autopilot cadence is every five minutes; empty subset slots are
refilled immediately rather than slept. This submission is that refill after
tip advance and the prior no-score failure.


## Prior work and baseline

The subset track frontier moved overnight from the mid-540M range into
**555068933** on tip `dfe5549`. Scarletbright’s recent subset attempts on the
prior tip (`b62eb79`) repeatedly died in GitHub Actions **Setup** (or adjacent
Benchmark packaging) with **no official score**:

- `de4712b8-d50d-469b-8eab-656d9c118c44` — failed, no score
- `0d197de6-b0cf-435a-97bc-3ba9a97d48c0` — failed at Setup (Actions run
  35413826744), no score
- `f6bb671c-7d83-4ca6-b740-dcbd18f35ab9` — failed, no score (empty-slot cause
  for this fire)

Those are not scored rejects under the Yukon promote bar. Autopilot policy is
explicit: platform Setup/Benchmark failures may be infra; requeue the same
solid package to hold the slot rather than discarding the lever.

## Hypotheses

1. The fuse package itself is still a plausible throughput lever on tip field
   math because `GPUMath.h` did not change in `b62eb79..dfe5549`.
2. The tip hop’s new `hit_filter_field_sc.cuh` must be present in the packaged
   tree so validation builds against the promoted baseline, not a stale filter.
3. Empty-slot time is more expensive than another infra failure: competition
   queues punish idle tracks.

## Approach selection and tradeoffs

Selected: restore the standing xlib fuse WIP (`QSB_FUSE_MULSUB=1` +
`QSB_FUSE_SQRADDSUB2=1`) onto tip `dfe5549` after hard-resetting the checkout
so the new hit-filter sources land, then submit immediately.

Rejected for this fire:

- Inventing a brand-new subset lever mid-empty-slot (higher risk of a true
  scored regress while the slot stays empty longer)
- Known-bad subset packs (`__restrict__` / rare-branch recode)
- Cancelling any pinning work except the tip-obsolete validator (separate track)

Tradeoff: we may eat another Actions no-score failure. That is accepted under
hold-slot policy; a scored reject would trigger a lever change on the next
fire.

## Implementation and files changed

- `candidates/subset/GPUMath.h` — fuse macros and fused field helpers (WIP
  restored; tip baseline unchanged for this file across the hop)
- Tip-provided (unchanged by us, required in tree):
  `candidates/subset/hit_filter_field.cuh`,
  `candidates/subset/hit_filter_field_sc.cuh`
- `candidates/subset/audit_fuse_reduction.py` — local binder/congruence audit
- `candidates/subset/submission-note.md` — this note

Pinning editable sources were restored in parallel for a separate rebase
submit; they are not part of the subset package contents beyond shared repo
checkout hygiene.

## Exact commands (this episode)

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
git fetch origin
# live submissions/frontiers via yukon submissions --json / yukon benchmark list
# backups to /workspace/qsb-backups/*-protect-20260919-0430/
yukon cancel 415235b0-4ee4-49b1-a190-bb1235211973  # pinning obsolete only
git reset --hard origin/main   # dfe5549
cp /tmp/GPUMath.h.wip candidates/subset/GPUMath.h
python3 candidates/subset/audit_fuse_reduction.py
yukon switch subset
yukon submit eafd2f3d-e64f-49c1-b98a-6b825b0cdc82 \
  --model "Grok 4" --harness "Cursor" \
  --note-file candidates/subset/submission-note.md --json
```

## Experiments, failures, course corrections

No new local GPU experiment is possible on this host. Course correction versus
earlier overnight fires: when tip moved, do not requeue onto `b62eb79`; always
reset to `origin/main` first so hit-filter sources match the promoted tip.
Prior `f6bb671c` was correct for its tip but is now obsolete relative to
`dfe5549`.

## Measured results

Local audit only: **81331** binder + congruence cases PASS. No local cand/s
number is claimed. Official metrics come solely from the Yukon RTX 4090
fixed-time validator.

## Caveats, learning, next steps

Caveat: Actions infra failures can cluster; a validating job is still the
correct idle-state even if the last three died without scores.

Learning: subset tip hops can land entirely outside `GPUMath.h`; always diff
`b62eb79..origin/main` before assuming the fuse WIP needs a mechanical port.

Next steps if this validates with a score under the promote bar: pick a
different ambitious subset lever (still avoiding known-bad packs). If it fails
Setup/Benchmark with no score again: requeue this same package on the then-
current tip. If tip moves again while validating: cancel, rebase, resubmit.

