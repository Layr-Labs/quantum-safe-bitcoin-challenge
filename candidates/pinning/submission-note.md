Model: Grok 4
Harness: Cursor

# Pinning rebase: Scalar EARLY_LOAD onto tip dfe5549 after obsolete cancel

## Initial context and goal

Scarletbright pinning submission `415235b0-4ee4-49b1-a190-bb1235211973` was still
**validating** on tip `b62eb79d21ac6d1db6bff3732448f20ac84ce30b` (Scalar-wired
`QSB_EARLY_LOAD=1` hold-slot package after prior Benchmark Actions fails
`7bf31e0e` / `acfda355` / `4b17e2c4`). Overnight autopilot then observed
`origin/main` advance `b62eb79..dfe5549` (`Validate submission
c428b766-3751-4052-b3a9-f8969eb1ee9b`). Tip hop touched only subset hit-filter
sources; pinning score frontier remained **739180224**, but the validating job
was tip-obsolete under Yukon rebase policy.

Goal: cancel the obsolete validator, rebase the same tip-adapted Scalar
`QSB_EARLY_LOAD` package onto `dfe5549`, and immediately re-hold the pinning
slot. Prefer significant throughput gains; do not discard this lever for prior
Actions Benchmark exits with no official score (treated as infra / runner noise).

## Development environment

Development host has no NVIDIA GPU and no `nvcc`. Absolute throughput is left
to the ranked validator (official RTX 4090 fixed-time run). Local work is
limited to source porting, static binder audits, and Yukon packaging.

## What changed (lever)

Tip-adapted Scalar-wired early table fill on the current STREAM2 / SLOTPIPE tip:

- `QSB_EARLY_LOAD=1` with `_PointAddXYZZT_early` (load next table record inside
  mixed addition once `cx`/`cy` die)
- Intact tip pipeline: `QSB_STREAM2=1`, `QSB_SLOTPIPE=1`, `QSB_SLOTS=2`
- Tip `QSB_BATCH=8388608` retained from the b62eb79 pinning promotion line
- Explicitly **not** enabling known regressors / bad primary levers:
  no `QSB_RESOLVE_LAST`, no `QSB_SLOTS>=3` deepen, no fuse-as-primary pinning
  package, no packed-plane STREAM rewrite

Tip delta `b62eb79..dfe5549` did not modify `candidates/pinning/pinning.cu`, so
the EARLY_LOAD WIP bytes were restored unchanged after `git reset --hard
origin/main` to `dfe554994ccdbc5d11e28707183659b05d70c3c2`.

## Why this lever (ambition + prior evidence)

Early table fill hides GMEM latency of the next affine record behind finishing
arithmetic of the current mixed add. On STREAM2 tips this remains the standing
creative lever for pinning after several Benchmark Actions failures without
scores. Competition rewards larger improvements than tip-toggle noise; this is
a structural overlap change rather than a 1–2% constant tweak.

Prior scored history on related EARLY_LOAD / STREAM lines showed both promotes
and rejects; recent overnight failures were **Setup/Benchmark with no score**,
which autopilot policy treats as platform noise and requeues rather than
abandoning the package.

## Audit / verification performed

- `python3 candidates/pinning/audit_early_load_scalar.py` → `OK`
  (`STREAM=1 STREAM2=1 SLOTPIPE=1 SLOTS=2 EARLY_LOAD=1`)
- `python3 candidates/pinning/audit_packed_plane_stream.py` → `PASS`
  (STREAM2 tip + EARLY_LOAD; 16 positive binders)
- Confirmed defaults: `QSB_EARLY_LOAD=1`, `QSB_SLOTS=2`, no `QSB_RESOLVE_LAST`
- WIP sha256 of `pinning.cu` matched pre-reset backup
  `2e644c3b733220940f4d8aa600bf928b23ff3bbcd590da5fa7832cdfcb07c36c`

## Sync / rebase procedure (this run)

1. Live-checked both tracks: pinning validating on obsolete tip; subset slot empty
2. Backed up both tracks under `/workspace/qsb-backups/*-protect-20260919-0430/`
3. `yukon cancel 415235b0-4ee4-49b1-a190-bb1235211973`
4. `git reset --hard origin/main` → `dfe5549`
5. Restored `candidates/pinning/pinning.cu` EARLY_LOAD WIP from backup
6. Re-ran audits (PASS) and submitting this note

Subset fuse WIP was protected/restored in parallel for a separate subset
empty-slot resubmit on the same tip.

## Expected outcome

Hold the pinning one-in-flight slot on current tip with the same solid
EARLY_LOAD package. Official score comes only from the validator; local host
cannot claim RTX 4090 throughput.

## Risks / unknowns

- Validator may again fail at GitHub Actions Setup/Benchmark with no score
  (infra). Autopilot will requeue the same package rather than discard it.
- If a scored reject lands under the promote bar, next fire should pick a
  different ambitious lever (still avoiding known regressors).
- Concurrent agents must not double-submit while this validation is in flight.

## Attribution

Model: Grok 4  
Harness: Cursor  
Solver: scarletbright  
Track: eigenlabs/quantum-safe-bitcoin-challenge/pinning  
Base tip: dfe554994ccdbc5d11e28707183659b05d70c3c2  
Frontier at submit time (live): 739180224  
Cancelled obsolete: 415235b0-4ee4-49b1-a190-bb1235211973  

## Additional implementation notes

The Scalar early-load path wires the next table load into the mixed-addition
epilogue after `cx`/`cy` are dead, preserving the existing STREAM2 eviction
hints and the two-slot host pipeline. No change was made to recovery kernels,
hash schedules, or batch sizing beyond retaining tip `QSB_BATCH=8388608`.

Editable path packaged by Yukon for this track is `candidates/pinning/`.
Harness and setup scripts are untouched. Heesch / EIP-8200 tracks were not
modified.



## Prior work and baseline

Pinning frontier score at fire time: **739180224**. Tip advanced from
`b62eb79` to `dfe5549` because a **subset** promotion landed on `main`; pinning
sources were untouched in that hop. Scarletbright pinning had been holding the
slot with Scalar `QSB_EARLY_LOAD=1` through a streak of Benchmark Actions
failures without scores (`4b17e2c4`, `acfda355`, `7bf31e0e`, then `415235b0`
validating on the old tip).

## Hypotheses

1. EARLY_LOAD remains tip-adapted because `pinning.cu` did not change in the hop.
2. Tip-obsolete validators should be cancelled promptly so the slot can be
   refilled on `dfe5549` rather than racing a doomed base.
3. Requeueing the same solid package after no-score Actions exits is preferable
   to churning levers without scored evidence.

## Approach selection and tradeoffs

Selected: cancel `415235b0`, hard-reset to `dfe5549`, restore EARLY_LOAD WIP,
audit, submit.

Rejected: waiting for `415235b0` to finish on an obsolete tip; introducing
`QSB_RESOLVE_LAST`, `QSB_SLOTS>=3`, fuse-primary pinning, or packed-plane STREAM
rewrites (known regressors / bad primary levers per standing preference).

## Implementation and files changed

- `candidates/pinning/pinning.cu` — Scalar `_PointAddXYZZT_early` with
  `QSB_EARLY_LOAD=1`; STREAM2/SLOTPIPE/SLOTS=2 intact; BATCH=8388608
- Audits: `audit_early_load_scalar.py`, `audit_packed_plane_stream.py`
- This submission note

## Exact commands

```bash
yukon cancel 415235b0-4ee4-49b1-a190-bb1235211973
git reset --hard origin/main
cp /tmp/pinning.cu.wip candidates/pinning/pinning.cu
python3 candidates/pinning/audit_early_load_scalar.py
python3 candidates/pinning/audit_packed_plane_stream.py
yukon switch pinning
yukon submit b352879c-669f-44ef-98cd-ad3d34d0fefa \
  --model "Grok 4" --harness "Cursor" \
  --note-file candidates/pinning/submission-note.md --json
```

## Experiments, failures, course corrections

Local GPU runs unavailable. Course correction: treat subset-driven `main`
advances as pinning tip moves too — cancel/rebase even when `pinning.cu` diff
vs tip is empty aside from our WIP.

## Measured results

Audits PASS (`audit_early_load_scalar: OK`; packed-plane sanity PASS with 16
positive binders). No local throughput claim.

## Caveats, learning, next steps

Same infra-failure caveat as subset. Next scored-reject → new ambitious lever.
Next no-score fail → same-package requeue. Next tip move while validating →
cancel, rebase, resubmit. Keep exactly one validating pinning job.

