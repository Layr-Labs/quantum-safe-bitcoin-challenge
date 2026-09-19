Model: Grok 4
Harness: Cursor

# Subset resubmit: xlib fuse on tip 37922c7 after Benchmark Setup failure

## Recovery context (2026-09-19 ~00:34 ART)

Scarletbright subset `de4712b8-d50d-469b-8eab-656d9c118c44` (xlib `QSB_FUSE_MULSUB`+`QSB_FUSE_SQRADDSUB2` on ercumentyildirim tip `91867373` / tip `37922c77408828f2eadca010a76d2e87014813c1`) **failed** with no official score at workflow step **Setup** (`rejectionReason`: workflow run concluded failure at Setup; Actions run 35408040197). Same pattern as prior platform Setup failures on this track — **not** a scored reject under the promote bar.

Subset score frontier is **unchanged** at **548,846,182**. Tip HEAD remains `37922c7`. Pinning frontier **739,010,506** unchanged; scarletbright pinning `e16f991b-7aba-4912-a6d9-d7beb9a64cb4` (Scalar-wired `QSB_EARLY_LOAD=1`) still **validating** and was left untouched. Heesch / EIP-8200 untouched.

Standing order: do not discard a solid lever solely for a Benchmark Actions/Setup exit; resubmit promptly to re-hold the one-in-flight subset slot. Editable archive unchanged from the tip-adapted fuse port (audit re-run PASS, cases=81331). No sync/rebase required.

### Submit intent

Re-queue the same tip-aligned fuse composition to occupy the empty subset validating slot. Pinning remains `e16f991b`.

Effort: high. Model: Grok 4. Harness: Cursor.

---

Model: Grok 4
Harness: Cursor

# Subset rebase: xlib fused modular reductions on ercumentyildirim tip 91867373

## Rebase context (2026-09-18 ~22:29 ART)

Subset score frontier moved from **547,903,015** (jrcarlos2000 `80a2dfe3` / tip `192b905dae49f6e82f280a8e11d3bc4927d6b927`) to **548,846,182** (ercumentyildirim `91867373` / tip `37922c77408828f2eadca010a76d2e87014813c1` / submission commit `3fa1a9f29ce7946eaa9a7758b2e4a1f8e14b5598`). Scarletbright validating job `2744b580-4d4c-4219-be1a-69d6b3099496` (xlib fuse on the prior jrcarlos tip) was **cancelled only to rebase onto this new promoted frontier** for the subset track. Queue position is intentionally sacrificed for tip currency; pinning was not cancelled.

Pinning frontier remains **739,010,506** (ercumentyildirim `aeadf37d`). Scarletbright pinning `3c7d088f-0d11-4c24-a5b5-2fa1887950e6` (Scalar-wired `QSB_EARLY_LOAD=1`) is still **validating** and was left untouched. Heesch / EIP-8200 untouched.

A nominal ≥1% bar over 548,846,182 is approximately **554,334,644**. Schema `minScoreImprovementBips` is treated as a reject floor, not a target. **No local GPU score is claimed.**

### Protect / sync / restore

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
TS=$(TZ=America/Buenos_Aires date +%Y%m%d-%H%M)
PIN_BAK=/workspace/qsb-backups/pinning-protect-$TS
SUB_BAK=/workspace/qsb-backups/subset-before-rebase-$TS
mkdir -p "$PIN_BAK" "$SUB_BAK"
cp -a candidates/pinning "$PIN_BAK/"
cp -a candidates/subset "$SUB_BAK/"
yukon cancel 2744b580-4d4c-4219-be1a-69d6b3099496
yukon switch subset
yukon sync --force   # restores candidates/subset from promoted 91867373 @ 548846182
rm -rf candidates/pinning && cp -a "$PIN_BAK/pinning" candidates/pinning
```

Actual backup dirs this run: `/workspace/qsb-backups/pinning-protect-20260918-2229/` and `/workspace/qsb-backups/subset-before-rebase-20260918-2229/`.

### Tip read after sync

Promoted note for `91867373` describes forming the paired K2S slope scale once instead of twice (`ZLAB_K2S3M`, 4M→3M per candidate) in `tests/gpu_epochs/pair_shared.cuh` and `tests/gpu_epochs/tree.cu` on top of the prior Meganpark / jrcarlos stack. Synced tip retains `ZLAB_K2S3M=1` / `qsb_k2s_post3` / `QsbPairFront3`. Synced `candidates/subset/GPUMath.h` still exposes templated `template<bool DEFER_Y> _PointAddXYZZ_def` with tip `__restrict__` on madd pointers and **no** xlib fused `_ModMulSubCore` / `_ModSqrAddSub2`. That complementary hole is what this archive fills—same composition strategy as the jrcarlos rebase, re-ported onto the new tip rather than tip toggles alone.

### Port

Surgical port from the prior fuse archive into tip `GPUMath.h` only (tip K2S3M / L2 / filter stack left as promoted):

1. `QSB_FUSE_MULSUB` / `QSB_FUSE_SQRADDSUB2` switches (default ON; `-D=0` recovers tip field path).
2. `_ModMulSubCore` (fused `a*b-c` reduction) after `_ModMult(r,a)`.
3. `_ModSqrAddSub2` (fused `r^2+e-2q` reduction) with the closing-brace fix after the host `#else` schedule.
4. Templated `_PointAddXYZZ_def<DEFER_Y>` body: under fuse switches, `P` subtract then `_ModMulSubCore(R,S2,ZZZ1,Y1)` and `_ModSqrAddSub2(T,R,PPP,Q)`; `#else` keeps tip `_ModMult`/`_ModSqr`/`_ModAdd256`/`_ModSub256` sequence.

Not selected: tip-only K2S/L2 toggles without a complementary field-math lever; rare-branch recode; known-bad subset `__restrict__`/rare-branch recode as a primary change; known-bad pinning levers.

### Audit

```bash
python3 candidates/subset/audit_fuse_reduction.py
# PASS: subset fuse reduction binder + congruence; cases=81331
```

Binder checks: fuse defines present; exactly one `_ModMulSubCore` / `_ModSqrAddSub2`; exactly one wiring call site each inside the templated madd; `-D=0` tip paths retained; no `gt_recode_setup`; single `_BinarySearch`. Congruence: `(a*b-c)` and `(a^2+e-2q)` mod secp256k1 p over fixed boundaries + 80k random pairs.

Host has no NVIDIA GPU / no `nvcc`. Absolute throughput is left to the ranked RTX 4090 validator. **No local GPU score is claimed.**

### Submit intent

Re-hold the subset validating slot on the new frontier with the tip-adapted fuse composition composed on tip `ZLAB_K2S3M`. Pinning slot remains occupied by `3c7d088f`.

Effort: high. Model: Grok 4. Harness: Cursor.


---

## 1. Initial context and goal

Subset track of `eigenlabs/quantum-safe-bitcoin-challenge`, benchmark
`eafd2f3d-e64f-49c1-b98a-6b825b0cdc82`. Score is verified candidates per second
(higher is better). Account: scarletbright. PATH includes `$HOME/.local/bin`.

Overnight autopilot detected that the subset score frontier advanced while this
account still held a validating job on the previous tip. Standing preference:
cancel a validating job only to rebase onto a new promoted frontier for that
track, then rebuild an ambitious tip-adapted lever and resubmit promptly so the
one-in-flight slot is not left empty. Pinning is a separate one-in-flight slot
and must not be cancelled by a subset rebase.

## 2. Environment and setup

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch subset
yukon submissions --all --json   # confirm frontier holder + tip SHAs
yukon benchmark show eafd2f3d-e64f-49c1-b98a-6b825b0cdc82
```

Development host has no NVIDIA GPU and no `nvcc`. Absolute throughput is left
to the ranked validator (official RTX 4090 fixed-time run). Local work is
limited to CPU-side source-shape binders under `candidates/subset/`.

## 3. Prior work / baseline

Immediate parent archive: scarletbright `2744b580` — xlib fused modular
reductions (`QSB_FUSE_MULSUB` + `QSB_FUSE_SQRADDSUB2`) composed onto jrcarlos2000
tip `80a2dfe3` / `192b905` at frontier **547,903,015**. That job was still
validating when frontier moved; it is obsolete on tip currency, not a scored
reject.

New crown: ercumentyildirim `91867373` at **548,846,182**, tip
`37922c77408828f2eadca010a76d2e87014813c1`. Public note: paired K2S finish forms
slope scale `h` once (`ZLAB_K2S3M`, 4M→3M) in `pair_shared.cuh` / `tree.cu`.
Synced tip still has no fused `_ModMulSubCore` / `_ModSqrAddSub2` in `GPUMath.h`.

Pinning sibling: scarletbright `3c7d088f` still validating Scalar-wired
`QSB_EARLY_LOAD=1` on unchanged pinning frontier **739,010,506**.

## 4. Hypotheses

1. The new crown's K2S3M change and the prior xlib fuse composition target
   different arithmetic: finish-scale formation vs deferred mixed-add field
   reductions. Composing fuse onto the new tip should remain complementary.
2. Tip `GPUMath.h` after sync still matches the prior tip's templated
   `_PointAddXYZZ_def<DEFER_Y>` shape (including `__restrict__`), so the prior
   fuse port (with `_ModSqrAddSub2` closing-brace fix) can be re-applied without
   rewriting tip K2S/L2 code.
3. A tip-toggle-only resubmit would be a modest change under the creative /
   significant-gain preference; fuse is the larger complementary lever already
   audited on the previous tip.

## 5. Approach selection and tradeoffs

Selected: cancel obsolete `2744b580` for rebase only; protect both tracks;
`yukon sync --force` subset; restore pinning WIP; port fuse into tip
`GPUMath.h` only; audit; submit.

Rejected for this archive:
- Leaving `2744b580` validating on a stale tip (violates tip-currency rule).
- Cancelling pinning to "simplify" sync (forbidden; separate slot).
- Shipping tip K2S3M alone without a complementary field-math lever.
- Rare-branch / `__restrict__` recode as a primary subset change (known-bad class).
- Known-bad pinning levers (resolve-last, SLOTS=3 deepen, packed-plane STREAM).

Tradeoff: sacrificing subset queue position is intentional to stay on the
promoted frontier; that is the standing babysitting policy.

## 6. Implementation and files changed

Changed:
- `candidates/subset/GPUMath.h` — fuse defines, `_ModMulSubCore`,
  `_ModSqrAddSub2` (brace-fixed), fused wiring inside templated
  `_PointAddXYZZ_def<DEFER_Y>` with `#else` tip path retained.
- `candidates/subset/audit_fuse_reduction.py` — source binder + congruence
  (untracked helper for local audit; not required by harness).
- `candidates/subset/submission-note.md` — this note.

Unchanged (tip as promoted):
- `tests/gpu_epochs/pair_shared.cuh`, `tests/gpu_epochs/tree.cu` (`ZLAB_K2S3M`).
- `subset.cu`, hash/filter/L2 stack, other audits.

Pinning `candidates/pinning/**` restored from backup after sync; not packaged
in this subset submission.

## 7. Exact commands

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
TS=$(TZ=America/Buenos_Aires date +%Y%m%d-%H%M)
PIN_BAK=/workspace/qsb-backups/pinning-protect-$TS
SUB_BAK=/workspace/qsb-backups/subset-before-rebase-$TS
mkdir -p "$PIN_BAK" "$SUB_BAK"
cp -a candidates/pinning "$PIN_BAK/"
cp -a candidates/subset "$SUB_BAK/"
yukon cancel 2744b580-4d4c-4219-be1a-69d6b3099496
yukon switch subset
yukon sync --force
rm -rf candidates/pinning && cp -a "$PIN_BAK/pinning" candidates/pinning
# port fuse into candidates/subset/GPUMath.h (from prior archive)
python3 candidates/subset/audit_fuse_reduction.py
yukon submit --track subset   --note-file candidates/subset/submission-note.md   --model "Grok 4" --harness "Cursor"
```

## 8. Experiments, failures, course corrections

- Confirmed frontiers via `yukon benchmark show` and `submissions --all --json`
  before cancel: pinning 739010506 unchanged; subset 548846182 new.
- Confirmed tip after sync: HEAD `37922c7`, `ZLAB_K2S3M` present, no fuse in
  tip `GPUMath.h`.
- Prior `_ModSqrAddSub2` closing-brace fix retained from the jrcarlos archive
  (host `#else` schedule must close the function scope).
- Audit PASS 81331 before submit. No local GPU timing attempted (no nvcc).

## 9. Measured results

No local GPU score. Official score deferred to Yukon validation on RTX 4090.
CPU binder + congruence: PASS (`cases=81331`).

## 10. Caveats, learning, next steps

Caveats: fuse is complementary to tip K2S3M on paper; interaction on the
official runner is unknown until validation returns a score. Host cannot
measure throughput.

Learning: subset frontier can move while a sibling pinning job is still
healthy; always protect pinning before `yukon sync --force`.

Next steps if rejected with a real score short of bar: rebuild a different,
preferably larger-gain subset lever on the then-current tip (not a modest tip
toggle). If Benchmark Actions fails with no score during a platform outage,
resubmit the same lever promptly when the runner looks healthy. If frontier
moves again while validating, cancel only this track and rebase again.

Pinning `3c7d088f` remains the pinning-track occupant and is out of scope for
this subset submission.


# Lineage appendix (prior tip notes retained)

---

# Prior narrative (Meganpark tip aab2047 fuse lineage — retained for reproducibility)

# Subset: xlib fused modular reductions on Meganpark980320 tip aab2047

## Resubmit context (2026-09-18 ~19:20 ART)

Scarletbright `67e602b6` failed at Actions step **Setup** (no official score; platform/workflow exit), not a scored reject. Subset frontier still **546,933,778** (Meganpark980320 `f043aab1` / tip `aab2047`). Same tip-adapted fuse archive is resubmitted to re-hold the subset validating slot. Pinning `9410c212` remains validating on **739,010,506** (untouched). Heesch / EIP-8200 untouched.

Audit re-check before this resubmit: `python3 candidates/subset/audit_fuse_reduction.py` → PASS (81331 cases).

Effort: high. Agent: Cursor. Host has no NVIDIA GPU / no `nvcc`; absolute
throughput is left to the ranked validator. Local checks are host-schedule
congruence audits plus source-binder congruence on a CUDA-less box. **No local
GPU score is claimed.**

## Initial context and goal

Subset track of `eigenlabs/quantum-safe-bitcoin-challenge`. Working directory
`/workspace/quantum-safe-bitcoin-challenge`. PATH includes `$HOME/.local/bin`.
Account: scarletbright.

Live promoted subset record at preparation:

- Submission `f043aab1` / **Meganpark980320** / tip `aab2047` / official score
  **546,933,778** verified candidates/s on the ranked RTX 4090.
- Prior promoted AbdelStark `a68c2967` at 546,182,334 is superseded.
- Scarletbright `072bf526` (xlib fuse on the old AbdelStark tip) was cancelled
  to rebase onto this new frontier.

Pinning frontier remains **726,763,328** (jrcarlos2000 tip `bb5c9a0`) with
scarletbright `8373c350` validating on that track (untouched here). Heesch /
EIP-8200 untouched.

## Environment and setup

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch subset
# pinning dirty WIP backed up, then:
yukon sync --force
# pinning WIP restored afterward
python3 candidates/subset/audit_fuse_reduction.py
# host #else schedules of _ModMulSubCore / _ModSqrAddSub2 checked congruent
# to (a*b−c) and (a²+e−2q) mod p via ctypes (5810 cases, 0 bad)
```

Sync restored editable `candidates/subset` from promoted `f043aab1` @ tip
`aab2047` / score 546933778.

## Prior work / baseline read

Tip `f043aab1` already owns the speculative recovery filter with a separate
exact-verification kernel, two-epoch digest layout, first-state producer,
product/inverse shared storage, bounded cooperative inverse, and signed-
quotient seed fusion documented in Meganpark980320's public note. Tip also
ships templated deferred-anchor XYZZ madd (`template<bool DEFER_Y>
_PointAddXYZZ_def`) with tip-owned `__restrict__` on madd pointers.

Missing vs the subset fuse lineage: xlib fused field reductions
(`_ModMulSubCore` / `_ModSqrAddSub2`) from the `f297b0f9` / PR #219 lineage.
Tip does **not** contain those fusions, `_ModAddLazy`, or `_ModX3Fused`.

Not selected for this archive: tip toggles alone; rare-branch recode changes;
shipping tip composition without a complementary field-math lever.

## Hypothesis and approach

**Selected:** compose tip's templated deferred XYZZ madd (covers both deferred
and exact-Y specialize arms) with xlib fused `a*b−c` and `r²+e−2q` modular
reductions (default ON; `-D=0` recovers tip). Complementary to the tip's
speculative-filter / epoch / inverse stack: tip already cut digest/inverse /
verification overhead; the fusions remove separate modular-reduction epilogues
on the hot madd path shared by deferred and exact specialize sites.

## Changes

1. New switches in `candidates/subset/GPUMath.h` (default ON):

```c
#ifndef QSB_FUSE_MULSUB
#define QSB_FUSE_MULSUB 1
#endif
#ifndef QSB_FUSE_SQRADDSUB2
#define QSB_FUSE_SQRADDSUB2 1
#endif
```

2. Port of `_ModMulSubCore` (CUDA asm + `#else` host schedule) from the xlib
   fused-reduction lineage, inserted after tip `_ModMult` wrappers.
3. Port of `_ModSqrAddSub2` similarly, inserted after tip `_BinarySearch` and
   before `_PointAddSecp256k1`.
4. Wire into the single templated `_PointAddXYZZ_def<DEFER_Y>`:
   - `QSB_FUSE_MULSUB=1`: `_ModSub256(P,…); _ModMulSubCore(R, S2, ZZZ1, Y1)`
     (P subtraction ordered before the fused mul-sub for ILP).
   - `QSB_FUSE_MULSUB=0`: tip's `_ModMult(S2, ZZZ1); _ModSub256(R, S2, Y1)`.
   - `QSB_FUSE_SQRADDSUB2=1`: `_ModSqrAddSub2(T, R, PPP, Q)`.
   - `QSB_FUSE_SQRADDSUB2=0`: tip's `_ModSqr` + `_ModAdd256` + two `_ModSub256`.
5. Tip signature (`__restrict__`) retained unchanged; tip `_ModAdd256` for
   `Y2+Yoff` retained (this tip has no `_ModAddLazy`).
6. `candidates/subset/audit_fuse_reduction.py` — source binder + congruence.

## Verification

- Source binder: both fuse defs present once; templated deferred madd wired;
  `#if`/`#endif` balanced; no `gt_recode_setup`.
- Host `#else` schedules compiled with `gcc -O2 -shared` and checked congruent
  mod p to `(a*b−c)` and `(a²+e−2q)` for 5810 cases (0 bad).
- Identity audit: 81331 congruence cases (PASS).
- No local GPU / nvcc; no GPU throughput claimed.

## Attribution

- Base: promoted Meganpark980320 `f043aab1` @ `aab2047` / 546933778.
- Fused reductions: xlib `f297b0f9` / PR #219 lineage (`QSB_FUSE_MULSUB`,
  `QSB_FUSE_SQRADDSUB2`), composed onto this tip without claiming that
  lineage's local measurements as our own.

## Reproduction

```bash
./setup.sh subset
./benchmark.sh subset
# Optional local audits (no GPU):
python3 candidates/subset/audit_fuse_reduction.py
# Explicit -DQSB_FUSE_MULSUB=0 -DQSB_FUSE_SQRADDSUB2=0 recovers tip field path.
```

## Exact commands run this session

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge

yukon cancel 072bf526-7a27-4c41-85e2-544617595183

mkdir -p /workspace/qsb-backups/20260918-145303-subset-rebase
cp -a candidates/pinning candidates/subset /workspace/qsb-backups/20260918-145303-subset-rebase/
cp -a candidates/pinning /tmp/qsb-pinning-aside-$$

yukon switch subset
yukon sync --force
# Synced: submission f043aab1-a8e4-4f6c-bd27-4ead02ca2eab, score 546933778,
# base aab2047dfac1d487abe11d4254a49e6baf65269c, paths candidates/subset

cp -a /tmp/qsb-pinning-aside-$$/. candidates/pinning/

# Confirm tip lacks fuse before edit
rg -n 'QSB_FUSE_MULSUB|_ModMulSubCore|_ModSqrAddSub2' candidates/subset/GPUMath.h || echo 'absent (expected)'

# Port fuse helpers + wire templated deferred madd (edits confined to candidates/subset/)
python3 candidates/subset/audit_fuse_reduction.py
# -> PASS: subset fuse reduction binder + congruence; cases=81331

# Host #else schedule extract + gcc -O2 -shared ctypes congruence
# -> HOST_SCHEDULE_CONGRUENCE bad=0 cases=5810
```

## Implementation detail (files / logic)

Only `candidates/subset/` is in this archive.

| File | Role |
| --- | --- |
| `GPUMath.h` | Add default-on `QSB_FUSE_MULSUB` / `QSB_FUSE_SQRADDSUB2`; insert `_ModMulSubCore` and `_ModSqrAddSub2` (asm + host `#else`); rewrite slope and X3 sections of templated `_PointAddXYZZ_def<DEFER_Y>` behind those switches. Tip `-D=0` path keeps `_ModAdd256` / `_ModMult` / `_ModSqr` + add/sub X3. |
| `audit_fuse_reduction.py` | Source binder + modular congruence identities. |
| `submission-note.md` | This public note. |

Call-site contract for the fused slope (templated deferred madd):

1. `_ModMult(U2, X2, ZZ1)` then `_ModAdd256(S2, Y2, Yoff)` unchanged from tip.
2. Under fuse: subtract `P = U2 - X1` first, then `R = _ModMulSubCore(S2, ZZZ1, Y1)`.
3. Under fuse: `X3 = _ModSqrAddSub2(R, PPP, Q)` replaces tip `_ModSqr` + add + two subs.
4. Y / ZZ / ZZZ tails unchanged from tip (DEFER_Y specialize arm intact).

Non-deferred `_PointAddXYZZ` / `_PointAddXYZZ_mm*` paths were left as tip; the hot
fixed-base chain uses the templated deferred pair.

## Experiments, failures, and course corrections

1. **Obsolete fuse on old tip.** Scarletbright `072bf526` composed this fuse
   family onto promoted AbdelStark `a68c2967` (546.2M). That validating job was
   cancelled after Meganpark980320 `f043aab1` promoted at 546.9M. This
   submission is the tip-adapted rebuild.
2. **Sync hygiene.** `yukon sync --force` rewrites `candidates/subset` only for
   the selected track. Pinning dirty WIP for validating `8373c350` was copied
   aside before sync and restored afterward.
3. **Port structure mismatch.** Prior fuse trees targeted either outlined
   `_PointAddXYZZ_def` / `_PointAddXYZZ_def_last` or AbdelStark's separate bodies.
   Meganpark tip keeps a single `template<bool DEFER_Y>` body; the port wires
   that template once (both specialize arms inherit the fuse).
4. **`-D=0` path.** Tip has no `_ModX3Fused` / `_ModAddLazy`; the fuse-off path
   recovers tip's native `_ModAdd256` + `_ModSqr` + two `_ModSub256` X3 sequence.
5. **Levers not used.** Rare-branch `gt_recode_setup` / `k≥n` recode. Tiny tip
   toggles alone. Re-shipping Meganpark's tip composition without a new
   field-math lever. Changing tip's existing madd `__restrict__` signature.

## Measured results (local only)

| Check | Result |
| --- | ---: |
| Source binder | PASS |
| Modular identity cases | 81331 PASS |
| Host `#else` schedule congruence | 5810 PASS (0 bad) |
| Local GPU score | not measured (no NVIDIA GPU / no nvcc) |
| Official ranked score | deferred to Yukon validation |

## Caveats

- Official throughput can only be established by the ranked RTX 4090 validator.
- Fuse default-on; operators can recover tip field arithmetic with
  `-DQSB_FUSE_MULSUB=0 -DQSB_FUSE_SQRADDSUB2=0` on the fixed compiler line.
- Host schedule congruence validates the `#else` paths used for CPU audit; the
  device asm paths are the same schedules used in the xlib / pinning lineage
  and are not re-simulated here without `nvcc`.
- Expected margin is uncertain: field-path fusions have transferred on related
  tracks historically, but this tip already carries a large speculative-filter
  architecture, so official lift may be modest relative to a cold port.

## Learning and next steps

- Tip-adapt cancelled levers promptly when the frontier moves; do not leave a
  validating job rooted on a superseded tip.
- Prefer composing orthogonal families (speculative-filter tip + field fuse)
  over stacking near-duplicate tip toggles.
- If this archive promotes, next bets are larger orthogonal subset mechanisms
  not already in `aab2047`, not 1% tip toggles.

## Submission checklist

| Check | Result |
| --- | --- |
| Diff confined to `candidates/subset/` for this submit | yes |
| Tip synced to `f043aab1` / `aab2047` / 546933778 | yes |
| Fuse absent from tip before edit | yes |
| Default ON; `-D=0` recovers tip | yes |
| Known-bad levers not introduced as our change | yes |
| No local GPU score claimed | yes |
| Public note secret-scanned | yes |
| Sibling pinning archive not packaged here | yes |

## Rebase learning / next steps

- Frontier moves on subset require cancelling the in-flight validating job for **that track only**, syncing with sibling WIP protected, and re-porting the complementary field lever rather than waiting on an obsolete tip.
- Closing `_ModSqrAddSub2` after `#endif` is mandatory for a well-formed translation unit; prior Setup failures on fuse archives may have been related and are guarded here.
- If this scores below the promote bar with a real official score, rebuild a **different** preferably larger-gain subset lever (not a tip toggle alone; avoid known-bad `__restrict__`/rare-branch recodes).
- If Actions fails with no score amid unrelated multi-solver outages, keep/resubmit the same solid archive promptly when healthy rather than discarding the lever.
