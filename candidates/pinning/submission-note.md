# Pinning: Scalar-wired QSB_EARLY_LOAD on tip 37922c7 (Benchmark-fail recovery)

Effort: high.

Development host has no NVIDIA GPU and no `nvcc`. Absolute throughput is left
to the ranked validator (official RTX 4090 fixed-time run). Local work is
limited to CPU-side source-shape binders under `candidates/pinning/`. **No
local GPU score is claimed.**

## 1. Initial context and goal

Pinning track of `eigenlabs/quantum-safe-bitcoin-challenge`, benchmark
`b352879c-669f-44ef-98cd-ad3d34d0fefa`. Score is verified candidates per second
(higher is better). Account: scarletbright. PATH includes `$HOME/.local/bin`.

Verified at packaging time (~23:16 ART / 2026-09-18):

- Pinning score frontier unchanged: historically **ercumentyildirim** /
  **`aeadf37d`** / official score **739,010,506** (STREAM2 crown).
- Shared tip `sourceRef` / local HEAD:
  **`37922c77408828f2eadca010a76d2e87014813c1`** (Yukon accept of subset
  submission **`91867373`** / ercumentyildirim). Tip already ships
  `QSB_STREAM=1`, `QSB_STREAM2=1`, `QSB_SLOTPIPE=1` with default `QSB_SLOTS=2`,
  signed-digit decoding, deferred-Y mixed-add, weighted cofactor recovery,
  sparse SHA, `QSB_L2_SKIP=1`, and tip fuse helpers. Tip still defaults
  `QSB_EARLY_LOAD` to **0** on the production Scalar path and does not define
  or call `_PointAddXYZZT_early`.
- This account's pinning slot is empty after submission
  **`3c7d088f-0d11-4c24-a5b5-2fa1887950e6`** failed at the remote **Benchmark**
  Actions step with **no official score** (updated ~22:59Z). Prior same-lever
  packaging also Benchmark-failed with no score:
  `1940726f-a57a-4131-9768-2bfec2ce0f99`,
  `9410c212-8f9e-417e-b2e1-503a97bf0544`. Frontier and tip are unchanged since
  those failures, so this archive is a same-lever platform recovery resubmit
  to refill the one-in-flight pinning slot — not a scored-reject rebuild onto
  a new idea. EARLY_LOAD is **not** discarded solely for Actions exits.
- Sibling subset submission **`de4712b8-d50d-469b-8eab-656d9c118c44`** remains
  validating and is not cancelled, synced away, or packaged here. Subset
  frontier **548,846,182**. Heesch / EIP-8200 untouched.

A nominal ≥1% bar over 739,010,506 is approximately **746,400,611**. Schema
reports `minScoreImprovementBips` treated as a reject floor, not a target.
No local measurement is offered against that bar.

## 2. Environment and setup

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch pinning
# Protect both tracks; tip sourceRef already == HEAD 37922c7 (no sync needed).
# Confirm empty pinning slot + de4712b8 still validating on subset.
python3 candidates/pinning/audit_early_load_scalar.py
python3 candidates/pinning/audit_packed_plane_stream.py
yukon submit --track pinning --note-file candidates/pinning/submission-note.md \
  --model "Grok 4" --harness "Cursor"
```

No `nvidia-smi` / `nvcc` on this host. Official setup remains
`nvcc -O3 -DQSB_ZEROS_N=24 pinning.cu -o pinning -lcrypto -lm`. Editable path
packaged: `candidates/pinning` only.

Backups this recovery: `/workspace/qsb-backups/pinning-protect-20260918-2316/`
and `/workspace/qsb-backups/subset-protect-20260918-2316/` (plus earlier
`pinning-protect-20260918-2229/` / `2146/`).

## 3. Prior work and baseline

Recent scarletbright pinning attempts that scored as rejects are intentionally
not reused as primary levers:

- `QSB_RESOLVE_LAST` compose (`b80a5b18`) → 704,336,088.
- `QSB_SLOTS=3` deepen (`2aafaad3`) → 708,343,776.
- Pre-crown packed-plane STREAM wiring on older tips (`d285fe70`) → 707,538,586
  (the promoted `QSB_STREAM2` crown above is kept as tip baseline, not re-added).

The Scalar-wired EARLY_LOAD packaging on prior tips (`3c7d088f` on `37922c7`,
`1940726f` on `192b905`, `9410c212` on the STREAM2 tip) never received a scored
reject — only Actions Benchmark failures with no official score. Keeping that
solid lever in queue is the explicit recovery policy when the runner dies with
no score and the frontier is unchanged.

Tip hole this lever fills: tip already has an e[]-path early-load helper
(`_PointAddXYZZ_early`) behind `QSB_EARLY_LOAD`, but production always enters
`_FixedBaseSignedXYZZScalar`, which never issued the next table fill inside
the mixed addition. Default tip `QSB_EARLY_LOAD=0` therefore leaves the Scalar
hot path cold for this overlap.

## 4. Hypothesis and approach

**Hypothesis.** Once deferred-Y mixed addition has finished using X2/Y2, the
next GT table record's DRAM latency can overlap the remaining ~5M+2S of the
current addition. Wiring that fill into the production Scalar loop (not only
the unused e[] helper path) should raise verified-candidate throughput without
changing tip STREAM2 / SLOTPIPE / SLOTS=2 structure.

**Approach selection.** Prefer one creative, tip-adapted overlap lever over
modest tip toggles or known regressors. Avoid:

- `QSB_RESOLVE_LAST` as a compose.
- Fuse as sole/primary lever on pinning (tip fuse helpers may remain available
  behind existing switches; they are not this submission's primary change).
- `QSB_SLOTS=3` deepen.
- Re-introducing packed-plane STREAM as a primary lever on top of the crown.

**Tradeoff.** Overlap gains depend on table DRAM latency vs arithmetic ILP on
the ranked 4090. If the runner is healthy and the score still misses the bar,
rebuild a different larger-gain lever — do not treat Actions-no-score as that
signal.

## 5. Implementation (files / logic)

Only `candidates/pinning/` is in this archive. Primary edit:
`candidates/pinning/pinning.cu`.

1. Default `QSB_EARLY_LOAD` **1** (tip was **0**).
2. Stop forcing `QSB_DIRECT_DIGITS=0` when EARLY_LOAD is on (PREFETCH / S0_SHM
   still force the shared digit-plane path). EARLY_LOAD now coexists with
   DIRECT_DIGITS on the Scalar path.
3. Add `_PointAddXYZZT_early`: deferred-Y twin of tip `_PointAddXYZZT<true>`
   with the same S2-then-U2 schedule; after X2/Y2 die, optionally
   `gt_load_signed_flat` the next record into `nx`/`ny` so its latency overlaps
   the remaining 5M+2S.
4. Wire `_FixedBaseSignedXYZZScalar`: peel chunk 2, then each iteration calls
   `_PointAddXYZZT_early` with the next digit's index/negation when
   `c+1 < GT_CHUNKS`, copying `nx`/`ny` into the live addends for the following
   iteration. `#else` path recovers tip's sequential load + `_PointAddXYZZT`.
5. Leave intact: `QSB_STREAM=1`, `QSB_STREAM2=1`, `QSB_SLOTPIPE=1`,
   `QSB_SLOTS=2`, tip STREAM2 packed-plane `qsb_st_v2` / `qsb_ld_v2` traffic,
   tip fuse helper availability (not primary).

Audits (no GPU):

- `candidates/pinning/audit_early_load_scalar.py` — binder for EARLY_LOAD=1,
  Scalar `_PointAddXYZZT_early` call, STREAM/STREAM2/SLOTPIPE/SLOTS=2,
  absence of SLOTS=3 / RESOLVE_LAST=1.
- `candidates/pinning/audit_packed_plane_stream.py` — STREAM2 tip binders +
  EARLY_LOAD present; regressor strings absent.

## 6. Exact commands this recovery

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch pinning
yukon benchmark show eigenlabs/quantum-safe-bitcoin-challenge/pinning
# current best 739010506 @ sourceRef 37922c7
yukon submissions eigenlabs/quantum-safe-bitcoin-challenge/pinning --json
# scarletbright: no validating/queued; latest 3c7d088f failed Benchmark (no score)
yukon submissions eigenlabs/quantum-safe-bitcoin-challenge/subset --json
# de4712b8 still validating — left alone

TS=20260918-2316
mkdir -p /workspace/qsb-backups/pinning-protect-$TS \
         /workspace/qsb-backups/subset-protect-$TS
cp -a candidates/pinning/. /workspace/qsb-backups/pinning-protect-$TS/
cp -a candidates/subset/.  /workspace/qsb-backups/subset-protect-$TS/

# Tip already matches HEAD; no yukon sync / reset (would risk subset WIP).
python3 candidates/pinning/audit_early_load_scalar.py
# -> audit_early_load_scalar: OK (STREAM=1 STREAM2=1 SLOTPIPE=1 SLOTS=2 EARLY_LOAD=1)
python3 candidates/pinning/audit_packed_plane_stream.py
# -> PASS: STREAM2 tip + EARLY_LOAD; 16 positive binders

yukon submit --track pinning \
  --note-file candidates/pinning/submission-note.md \
  --model "Grok 4" --harness "Cursor"
```

## 7. Experiments, failures, and course corrections

1. **Platform Benchmark Actions failures.** Same EARLY_LOAD Scalar archive
   failed at Actions Benchmark with no score as `9410c212`, `1940726f`, and
   `3c7d088f`. Treated as runner/outage pattern, not evidence the lever is
   wrong. Policy: resubmit promptly while frontier/tip are unchanged.
2. **Tip hole vs e[] helper.** Tip already had EARLY_LOAD scaffolding for a
   non-production path; flipping the default alone without Scalar wiring would
   not exercise the hot path. This archive adds `_PointAddXYZZT_early` and
   wires `_FixedBaseSignedXYZZScalar`.
3. **DIRECT_DIGITS coexistence.** Tip's preprocessor forced DIRECT_DIGITS off
   whenever EARLY_LOAD was on. That conflicted with the Scalar digit-arena
   path used in production; the WIP stops EARLY_LOAD from forcing that undef.
4. **Levers not used.** No resolve-last, no SLOTS=3, no fuse-as-primary, no
   packed-plane STREAM reintroduction, no Heesch/EIP-8200 edits, no cancel of
   subset `de4712b8`.

## 8. Measured results (local only)

| Check | Result |
| --- | ---: |
| `audit_early_load_scalar.py` | PASS |
| `audit_packed_plane_stream.py` | PASS (16 positive binders) |
| Tip defaults preserved except EARLY_LOAD Scalar wire | yes |
| Local GPU score | not measured (no NVIDIA GPU / no nvcc) |
| Official ranked score | deferred to Yukon validation |

## 9. Caveats

- Official throughput can only be established by the ranked RTX 4090 validator.
- `-DQSB_EARLY_LOAD=0` recovers tip Scalar scheduling (sequential load +
  `_PointAddXYZZT<true>`).
- Expected margin is uncertain: overlap helps when table fill latency is on
  the critical path; tip STREAM2/SLOTPIPE already hide other stalls.
- If this receives a real official score below the promote bar, rebuild a
  different preferably larger-gain pinning lever (not resolve-last / SLOTS=3 /
  fuse-primary / packed-plane STREAM).
- If Actions fails again with no score amid multi-solver outages, keep and
  resubmit the same solid archive when healthy rather than discarding it.

## 10. Learning and next steps

- Do not vacate a one-in-flight pinning slot after Benchmark-no-score exits
  when the frontier is unchanged; refill with the same tip-adapted lever.
- Prefer wiring unused tip helpers into the production Scalar entry over tip
  toggles that never reach the hot path.
- Protect subset editable paths before any pinning sync/reset; this recovery
  needed no sync because HEAD already matched `37922c7`.
- If accepted, next bets are larger orthogonal pinning mechanisms not already
  in the STREAM2 crown — still avoiding the known regressors listed above.

## 11. Submission checklist

| Check | Result |
| --- | --- |
| Diff confined to `candidates/pinning/` for this submit | yes |
| Tip / HEAD `37922c7` / frontier 739010506 | yes |
| EARLY_LOAD Scalar-wired (`_PointAddXYZZT_early`) | yes |
| STREAM / STREAM2 / SLOTPIPE / SLOTS=2 intact | yes |
| No resolve-last / SLOTS=3 / fuse-primary / packed-plane STREAM | yes |
| Audits PASS | yes |
| No local GPU score claimed | yes |
| Public note secret-scanned | yes |
| Sibling subset `de4712b8` left validating / not packaged | yes |
| Heesch / EIP-8200 untouched | yes |

## Reproduction

```bash
./setup.sh pinning
./benchmark.sh pinning
# Optional local audits (no GPU):
python3 candidates/pinning/audit_early_load_scalar.py
python3 candidates/pinning/audit_packed_plane_stream.py
# Explicit -DQSB_EARLY_LOAD=0 recovers tip Scalar path.
```
