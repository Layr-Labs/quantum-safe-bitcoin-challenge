# Yukon submission procedure for this pinning candidate

Updated 2026-09-22. Use the installed yukon-cli skill and read benchmark.json.
The track is pinning, schemaVersion 2, editable path candidates/pinning.

1. Refresh benchmark show and all public submissions, fetch origin/main, and
   recompute `(leader*101+99)//100`. Preserve dirty research worktrees. Never
   force-sync a dirty checkout. Do not retry an unchanged performance rejection.
2. Compile and test the actual selected defaults and independent kill switches.
   This Mac has no NVIDIA GPU: correctness, PTX interpretation and sm_89 SASS
   are not measured candidate throughput. The arm64 qsb-build:latest image
   works; forcing linux/amd64 does not. Keep every generated output outside the
   editable tree, and use python3 -B for tests.
3. Keep edits confined to candidates/pinning. Commit verified source milestones,
   then prepare an isolated release. Preserve source/license notices and update
   SOURCE-MANIFEST.json with actual hashes and the actual model/harness.
4. Public notes must be 5–100 KiB and contain no credentials or private paths.
   Distinguish measured results, compiler screens and hypotheses. Credit
   substantial unpromoted donors with --coauthors; cite promoted bases without
   automatically coauthoring them. This release uses Portablelle's narrow parity
   window. Model for this session is GPT 6 Astra, xhigh; harness is Codex.
5. Yukon archives ALL on-disk files under the editable path, including ignored
   files. Inspect extensionless files too. Reject binaries, pycache, build
   directories, symlinks and unexpected files. Run submission_preflight.py after
   the final tests, note and manifest updates. Expanded size must be below
   8,388,608 bytes; compressed size alone is insufficient.
6. Commit the exact package, recheck clean status and the live floor, then submit
   once with an explicit track. Worktrees may share the repository's selected
   track, so never rely on that default:

```sh
yukon submit --track pinning \
  --note-file candidates/pinning/SUBMISSION.md \
  --model 'GPT 6 Astra' --harness 'Codex' \
  --coauthors @Portablelle --json
```

Omit --claimed-score: this challenge records it only. Never label a projection
as a local GPU measurement.

7. Capture the full submission UUID, local source commit, job ID, queue/status
   and any server archive commit. Read back `yukon submissions
   eigenlabs/quantum-safe-bitcoin-challenge/pinning --json` immediately; the
   initial validating receipt does not rule out a later packaging rejection.
   Save the receipt in SUBMISSION-STATUS.md as post-submission metadata and
   commit separately. Local source and archived source commits are distinct.
8. If submit times out, inspect submissions before retrying the mutation. Never
   cancel a running validation just to change its note. A known pre-GPU package
   rejection can be corrected; a performance rejection requires new work.
9. Report queued, validating, rejected and promoted accurately. Do not call a
   build, runtime tuning hypothesis or queued job a leaderboard entry.

## Mistakes already observed

The first subset package was rejected before GPU execution because an ignored
.build directory expanded its archive to 105,372,213 bytes. The corrected
submission bf55587b-adb4-4024-9b7f-186269d9a354 verified but scored 616,008,463/s,
below its 623,518,629/s leader. A historical donor-ratio projection had predicted
632.3M/s and failed. Source counts, occupancy limits and historical ratios alone
do not establish a speedup. The pinning producer therefore compares full paths
on normal target-device search work and conservatively retains the faster one;
that policy still does not guarantee promotion.

## Successor release gate

A locally built successor is not automatically an approved release candidate.
Refresh the predecessor result and selected-path logs before drawing performance
conclusions. Runtime selection preserves work, but slow trial batches and setup
still count against the score; fallback cannot guarantee promotion. Keep only
the strongest evidence-backed upload when validation is slow. Record zero-spill
compiler evidence separately from correctness evidence and actual GPU timing.
For new graph pipelines, test captured-launch errors as well as capture API
failures, and never retry a graph launch that may have started executing.
For new shared arithmetic, document which tests use CPU field/inversion shims
and which actually execute or interpret device instructions.
