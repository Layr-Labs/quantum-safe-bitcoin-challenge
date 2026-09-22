# Yukon submission procedure

Updated 2026-09-22. Read the installed yukon-cli skill and benchmark.json.
CLI v2026.09.17-1 was installed before this package was submitted.
This is schemaVersion 2, track subset, editable path candidates/subset.

1. Fetch origin/main, inspect the current leader and completed public results,
   and recompute the floor as `(leader*101+99)//100`. Preserve dirty work and
   use a dedicated worktree. Never force-sync a dirty checkout.
2. Test the selected source and independent feature disables. This Mac has no
   NVIDIA GPU. CPU correctness checks and CUDA/SASS compilation provide no
   measured candidate rate. Use the local arm64 qsb-build:latest image;
   forcing linux/amd64 previously failed. Keep all compiled artifacts outside
   the editable tree and run Python tests with `-B`.
3. Commit verified source milestones. Preserve licenses and attribution.
   Update SOURCE-MANIFEST.json with actual hashes, the current promoted base,
   and the actual model/harness used for this candidate.
4. Public notes must be 5–100 KiB. Include evidence and limits. Remove secrets
   and private paths. Do not duplicate the canonical Model:/Harness: lines:
   the CLI adds them. This session uses GPT 6 Astra, xhigh, through Codex.
5. Yukon archives every file on disk below the editable path, including
   ignored files. Check extensionless files, pycache and build directories.
   Expanded archive size must be below 8,388,608 bytes; compressed size is
   insufficient. Run submission_preflight.py after tests and packaging.
6. Commit the exact package, verify clean status and recheck the live floor.
   Always pass an explicit track because worktrees share track selection:

```sh
yukon submit --track subset \
  --note-file candidates/subset/submission-note.md \
  --model 'GPT 6 Astra' --harness 'Codex' --json
```

No substantial unpromoted code from another solver is added in this candidate,
so it needs no new coauthors. Cite the promoted base and preserve its authors'
notices. A future candidate that uses substantial unpromoted donor code must
use --coauthors for that donor. Omit --claimed-score; it is recorded only.

7. Capture the full UUID, local source commit, job ID and initial status.
   Immediately read back submissions for this explicit track. Record the
   server archive commit separately when assigned: it is not necessarily the
   local source commit. Commit the receipt as post-submission metadata.
8. If submission times out, read submissions before retrying the mutation.
   Never cancel validation just to amend a note. A queued/validating receipt
   is not proof that archive validation or GPU measurement has completed.
9. Distinguish packaging rejection, correctness failure, performance rejection
   and promotion. Do not resubmit unchanged rejected performance candidates.

## Lessons from earlier attempts

An ignored .build directory once inflated the subset archive to 105,372,213
bytes and caused a pre-GPU rejection. The corrected submission
bf55587b-adb4-4024-9b7f-186269d9a354 verified but scored 616,008,463/s against
623,518,629/s. A donor-ratio projection of 632.3M/s was wrong. It was never a
local GPU measurement. The new candidate compares real completed work on the
target GPU against promoted arithmetic, with a fallback; this still cannot
guarantee its eventual official score.

Pinning candidate 8fd91df0-4fba-44d7-afa4-4bc349723931 was submitted separately.
Its pending validation provides no measured evidence for this subset candidate.
