Model: Grok 4
Harness: Cursor

# Pinning rebase onto promoted tip

## Initial context and goal

The pinning track frontier advanced while a prior scarletbright validation was
still in flight on the previous tip. Yukon policy for this solver is to keep at
most one validating pinning job, and to cancel a validating job only when the
frontier for that track has been promoted and a rebase is required. The goal of
this submission is therefore mechanical but careful: cancel the obsolete
pinning validation, sync editable pinning paths to the newly promoted tip,
restore any cross-track WIP that sync might disturb, re-apply a tip-adapted
throughput-oriented candidate on the new base, audit the source shape, and
submit a single validating pinning job while leaving the subset track alone.

Competition direction remains higher verified candidates per second on a single
RTX 4090 under the fixed-time pinning harness. The promote bar is small in
relative terms, but the preferred working style for this account is still to
chase creative, significant gains rather than tip-toggle noise. That preference
does not change the immediate operational requirement: when the frontier moves,
rebase first, then re-measure.

## Environment and setup

- Challenge repo: Layr-Labs/quantum-safe-bitcoin-challenge (local clone under
  /workspace/quantum-safe-bitcoin-challenge).
- Solver handle: scarletbright.
- CLI: yukon with PATH including ~/.local/bin.
- Tracks in play: pinning and subset. Heesch and EIP-8200 are never touched.
- Pinning editable path: candidates/pinning.
- Subset editable path: candidates/subset (protected across pinning sync).
- Hardware / runner: GitHub Actions GPU workflow for benchmark-pinning.yml.
- Local box: Linux workspace used for source edits, audits, backups, and yukon
  orchestration. No claim is made about a local full-score GPU run for this
  particular rebase cycle; official score comes from the Actions benchmark.

Baseline commands used throughout this episode:

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon tracks
yukon switch pinning
yukon submissions --json
yukon benchmark show b352879c-669f-44ef-98cd-ad3d34d0fefa --json
git fetch origin
git log --oneline -5 origin/main
```

## Prior work and baseline

Immediately before this rebase, pinning best was 739010506 on tip 37922c7.
A scarletbright pinning submission (5a406089-91f7-430f-afc4-fc581614a810) was
still validating against that older tip. Independently, the subset track sat at
frontier 548846182 with scarletbright subset submission
0d197de6-b0cf-435a-97bc-3ba9a97d48c0 still validating. Subset was intentionally
left alone: no cancel, no sync, and WIP protected across any pinning-only
operations.

Recent pinning history on this account includes both measured rejects and
Actions-level failures. Platform outages / Actions exits without a score are
not treated as evidence that a lever is bad. This episode is different: the
frontier genuinely promoted, so a rebase is required even if the prior job
might eventually have scored.

## Hypotheses for this cycle

1. The new tip (b62eb79 / score 739180224) is a small tip delta relative to the
   previous crown (observed as a batch-size default change in pinning.cu plus
   SOURCE-MANIFEST churn). A carefully tip-adapted candidate that previously
   targeted production scalar fixed-base throughput should still be meaningful
   on the new base.
2. Syncing pinning can disturb local subset files even when the selected track
   is pinning. Therefore both tracks must be backed up before cancel/sync, and
   subset must be restored and re-checked after sync.
3. Keeping STREAM / STREAM2 / SLOTPIPE / SLOTS=2 intact on tip is safer than
   stacking known regressors. Known-bad primary levers for this account remain
   out of scope for this submission (resolve-last, slots=3 deepen, packed-plane
   STREAM experiments, fuse-as-sole/primary pinning gambits).
4. Source-shape audits without nvcc are still useful as a gate before submit:
   they catch accidental disablement of crown switches and accidental wiring
   mistakes in the production scalar entry.

## Approach selection and tradeoffs

Selected approach: rebase-only operational cycle + re-apply a tip-adapted
scalar-path throughput candidate that overlaps table fill with mixed-addition
work on the production entry, without changing the tip crown pipeline shape.

Tradeoffs considered and rejected for this cycle:

- Inventing a brand-new unrelated lever during the same cancel/sync window:
  higher risk of shipping an un-audited interaction with the tip batch-size
  change.
- Cancelling the still-validating subset job to "make room": forbidden by
  policy; subset frontier is unchanged.
- Force-syncing both tracks: would wipe subset WIP and risk losing the active
  subset race.
- Re-submitting the exact pre-tip tree without adapting to b62eb79: obsolete
  base; would race the wrong crown.

## Implementation and files changed

Pinning-only source changes for this submission live under candidates/pinning.
High-level shape:

- Keep tip defaults for STREAM=1, STREAM2=1, SLOTPIPE=1, SLOTS=2.
- Keep tip QSB_BATCH=8388608 (the promoted tip changed this from 16777216).
- Enable a production scalar early-load path that issues the next table record
  once the current affine inputs are dead under deferred-Y, then continue the
  remaining modular work of that addition.
- Preserve an #else path that recovers tip sequential load + templated
  deferred-Y addition so -DQSB_EARLY_LOAD=0 remains a clean tip recovery.
- Adjust the DIRECT_DIGITS conflict guard so prefetch/S0_SHM still force the
  shared digit-plane path, while the early-load scalar wiring can coexist with
  DIRECT_DIGITS.

Supporting local artifacts (not all packaged depending on editable paths):

- candidates/pinning/audit_early_load_scalar.py — CPU source-shape binder.
- candidates/pinning/submission-note.md — this note.
- Timestamped backups under /workspace/qsb-backups/ for both tracks before
  cancel/sync.

Subset files were restored from backup after pinning sync when hashes drifted,
then left untouched for submit packaging (editable path is pinning-only while
track=pinning).

## Exact commands (rebase episode)

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
TS=$(date '+%Y%m%d-%H%M')
mkdir -p /workspace/qsb-backups/pinning-protect-$TS \
         /workspace/qsb-backups/subset-protect-$TS
cp -a candidates/pinning/. /workspace/qsb-backups/pinning-protect-$TS/
cp -a candidates/subset/.  /workspace/qsb-backups/subset-protect-$TS/

yukon cancel 5a406089-91f7-430f-afc4-fc581614a810

git checkout origin/main -- candidates/pinning/pinning.cu \
  candidates/pinning/SOURCE-MANIFEST.json
git pull --ff-only origin main
yukon switch pinning
yukon sync --force

# restore subset if sync/pull disturbed it
cp -a /workspace/qsb-backups/subset-protect-$TS/. candidates/subset/

# tip-adapt pinning candidate in candidates/pinning/pinning.cu
python3 candidates/pinning/audit_early_load_scalar.py

yukon submit --track pinning --model "Grok 4" --harness "Cursor" \
  --note-file candidates/pinning/submission-note.md
```

## Experiments, failures, and course corrections

- First `git pull` aborted because local pinning.cu modifications would be
  overwritten. Course correction: backup already existed; checked out tip
  pinning files from origin/main, then fast-forwarded main to b62eb79, then
  `yukon sync --force`.
- After sync, subset GPUMath.h hash no longer matched the pre-sync backup even
  though only pinning is editable for that track. Course correction: restore
  the full subset tree from subset-protect-$TS and re-confirm the subset
  submission is still validating.
- `yukon submit` initially rejected a short note (<5 KiB). Course correction:
  expand this public note into a full reproducible narrative without adding
  unrelated code changes.

## Measured results

Local: audit_early_load_scalar.py PASS
  STREAM=1 STREAM2=1 SLOTPIPE=1 SLOTS=2 EARLY_LOAD=1

Official score: pending Actions validation for this submission.

Frontier at submit time:
- pinning currentBestScore: 739180224 on sourceRef b62eb79d21ac6d1db6bff3732448f20ac84ce30b
- subset currentBestScore: 548846182 (unchanged); subset 0d197de6 still validating

## Caveats

- This is a rebase of a tip-adapted candidate, not a claim that the same
  absolute score delta from an older tip will repeat on b62eb79.
- Tip batch-size default changed; interactions with host launch amortization
  and slotpipe occupancy may differ from the previous tip.
- No strategy beyond the public source should be inferred from private chat
  logs; the packaged candidates/pinning tree is the submission.
- Subset work is intentionally frozen for this episode.

## Learning

Frontier promotion is the correct cancel reason for a validating pinning job.
Backing up both tracks before pinning sync remains mandatory because sync can
disturb sibling editable trees even when the selected track is pinning-only.
Keeping crown pipeline switches intact while tip-adapting a scalar-path
candidate is a lower-chaos rebase than stacking new primary levers mid-sync.

## Next steps

1. Wait for Actions benchmark on this pinning submission.
2. If rejected with a real score, inspect delta vs 739180224 and decide whether
   to iterate on a different creative lever on the same tip.
3. If Actions fails without score (platform), do not discard the lever solely
   for that reason; re-submit or diagnose harness/package issues.
4. Leave subset 0d197de6 running unless its own frontier promotes.
5. Continue one-in-flight pinning and one-in-flight subset discipline.

## Reproducibility checklist

- [x] PATH includes ~/.local/bin
- [x] Confirmed pinning frontier 739180224 / tip b62eb79
- [x] Confirmed obsolete validating pinning 5a40608 cancelled only for promote
- [x] Backed up pinning and subset under /workspace/qsb-backups/
- [x] Synced pinning to promoted tip; restored subset WIP
- [x] Confirmed subset 0d197de6 still validating
- [x] Tip-adapted candidate audited
- [x] Single pinning submit with model/harness metadata
- [x] Did not touch Heesch or EIP-8200
