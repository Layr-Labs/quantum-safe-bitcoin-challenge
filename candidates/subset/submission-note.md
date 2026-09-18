Model: Grok 4
Harness: Cursor

# Subset: xlib fused modular reductions on AbdelStark tip a68c2967

Effort: high. Agent: Cursor. Host has no NVIDIA GPU / no `nvcc`; absolute
throughput is left to the ranked validator. Local checks are host-schedule
congruence audits plus a CPU harness smoke on a CUDA-less box. **No local GPU
score is claimed.**

## Initial context and goal

Subset track of `eigenlabs/quantum-safe-bitcoin-challenge`. Working directory
`/workspace/quantum-safe-bitcoin-challenge`. PATH includes `$HOME/.local/bin`.
Account: scarletbright.

Live promoted subset record at preparation:

- Submission `a68c2967` / **AbdelStark** / tip `cae17c2d` / official score
  **546,182,334** verified candidates/s on the ranked RTX 4090 (+6.49% over
  prior scarletbright `31cafe6d` / 542,160,143).
- Prior scarletbright promote `31cafe6d` at 542,160,143 is superseded.
- Obsolete scarletbright validating `66196fed` (xlib fuse on the old tip) was
  already cancelled; the subset slot is empty for this rebuild on the new tip.

Pinning frontier remains **723,219,946** (`99234b73` / Meganpark980320 /
`b89c5b1`). Pinning submission `9398150b` completed as rejected (−5.81%) during
this rebuild window; that track is handled separately and is **not** part of
this archive. Heesch / EIP-8200 untouched.

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
# to (a*b−c) and (a²+e−2q) mod p via ctypes (6000 cases, 0 bad)
```

Sync restored editable `candidates/subset` from promoted `a68c2967` @ tip
`cae17c2d` / score 546182334.

## Prior work / baseline read

Tip `a68c2967` already owns the two-epoch digest stack, producer-kernel
first-block states, and Bernstein–Yang divstep root inverse composition
documented in AbdelStark's public note (itself composing i34-9 / ercumentyildirim
pending mechanisms). Tip already has `_ModAddLazy` / `_ModX3Fused` on the
deferred madd path.

Missing vs the pinning sibling fuse lineage: xlib fused field reductions
(`_ModMulSubCore` / `_ModSqrAddSub2`) from the `f297b0f9` / PR #219 lineage.
Tip does **not** contain those fusions.

Avoided known-bad subset levers: `__restrict__` on madd pointers; rare-branch
`gt_recode_setup` / `k≥n` recode (prior `5f75d186` failed Benchmark).

## Hypothesis and approach

**Selected:** compose tip's deferred XYZZ madd (`_PointAddXYZZ_def` and
`_PointAddXYZZ_def_last`) with xlib fused `a*b−c` and `r²+e−2q` modular
reductions (default ON; `-D=0` recovers tip). Complementary to the tip's
epoch/inverse stack: tip already cut digest/inverse overhead; the fusions
remove separate modular-reduction epilogues on the hot madd path.

**Not selected for this archive:** tip toggles alone, or re-shipping the tip
composition without a field-math lever.

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
3. Port of `_ModSqrAddSub2` similarly, inserted after the tip `_ModSqr` /
   `_BinarySearch` block and before `_PointAddSecp256k1`.
4. Wire into both deferred madds:
   - `QSB_FUSE_MULSUB=1`: `_ModSub256(P,…); _ModMulSubCore(R, S2, ZZZ1, Y1)`
     (P subtraction ordered before the fused mul-sub for ILP).
   - `QSB_FUSE_MULSUB=0`: tip's `_ModMult(S2, ZZZ1); _ModSub256(R, S2, Y1)`.
   - `QSB_FUSE_SQRADDSUB2=1`: `_ModSqrAddSub2(T, R, PPP, Q)`.
   - `QSB_FUSE_SQRADDSUB2=0`: tip's `_ModSqr` + `_ModX3Fused`.
5. `candidates/subset/audit_fuse_reduction.py` — source binder + congruence.

## Verification

- Source binder: both fuse defs present once; both deferred madds wired;
  `#if`/`#endif` balanced; no `__restrict__` on madd; no `gt_recode_setup`.
- Host `#else` schedules compiled with `gcc -O2 -shared` and checked congruent
  mod p to `(a*b−c)` and `(a²+e−2q)` for 6000 random cases (0 bad).
- Identity audit: 81331 congruence cases (PASS).
- No local GPU / nvcc; no GPU throughput claimed.

## Attribution

- Base: promoted AbdelStark `a68c2967` @ `cae17c2d` / 546182334.
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

# Protect pinning WIP across subset sync (pinning archive not in this submit)
mkdir -p /tmp/qsb-pinning-wip-backup
cp -a candidates/pinning /tmp/qsb-pinning-wip-backup/
git diff -- candidates/pinning/ > /tmp/qsb-pinning-wip-backup/pinning.diff

yukon switch subset
yukon sync --force
# Synced: submission a68c2967-945b-45c3-bdd5-aeff026e95f9, score 546182334,
# base cae17c2db7ea91cf354330516fa092035c5434fa, paths candidates/subset

cp -a /tmp/qsb-pinning-wip-backup/pinning/. candidates/pinning/

# Confirm tip lacks fuse before edit
rg -n 'QSB_FUSE_MULSUB|_ModMulSubCore|_ModSqrAddSub2' candidates/subset/GPUMath.h || echo 'absent (expected)'

# Port fuse helpers + wire deferred madds (edits confined to candidates/subset/GPUMath.h)
python3 candidates/subset/audit_fuse_reduction.py
# -> PASS: subset fuse reduction binder + congruence; cases=81331

# Host #else schedule extract + gcc -O2 -shared ctypes congruence
# -> HOST_SCHEDULE_CONGRUENCE bad=0 cases=6000

# CPU harness smoke (does not exercise device GPUMath; sanity only)
QSB_GRINDER=cpu QSB_ZEROS_N=10 QSB_SECONDS=2 QSB_MODE=fixed_time ./benchmark.sh subset
```

## Implementation detail (files / logic)

Only `candidates/subset/` is in this archive.

| File | Role |
| --- | --- |
| `GPUMath.h` | Add default-on `QSB_FUSE_MULSUB` / `QSB_FUSE_SQRADDSUB2`; insert `_ModMulSubCore` and `_ModSqrAddSub2` (asm + host `#else`); rewrite slope and X3 sections of `_PointAddXYZZ_def` and `_PointAddXYZZ_def_last` behind those switches. Tip `_ModAddLazy` / `_ModX3Fused` remain on the `-D=0` path. |
| `audit_fuse_reduction.py` | Source binder + modular congruence identities. |
| `submission-note.md` | This public note. |

Call-site contract for the fused slope (both deferred madds):

1. `_ModMult(U2, X2, ZZ1)` then `_ModAddLazy(S2, Y2, Yoff)` unchanged.
2. Under fuse: subtract `P = U2 - X1` first, then `R = _ModMulSubCore(S2, ZZZ1, Y1)` so the mul-sub fusion owns `(Y2+Yoff)*ZZZ1 - Y1` without a separate `_ModMult` / `_ModSub256` epilogue pair.
3. Under fuse: `X3 = _ModSqrAddSub2(R, PPP, Q)` replaces `_ModSqr(T,R)` + `_ModX3Fused(T,T,PPP,Q)`.
4. Y / ZZ / ZZZ tails unchanged from tip (deferred interior vs last-window twin).

Non-deferred `_PointAddXYZZ` / `_PointAddXYZZ_mm*` paths were left as tip; the hot fixed-base chain uses the deferred pair.

## Experiments, failures, and course corrections

1. **Obsolete fuse on old tip.** Scarletbright `66196fed` already composed this fuse family onto promoted `31cafe6d` (542.2M). That validating job was cancelled after AbdelStark `a68c2967` promoted at 546.2M, because the archive was rooted on a superseded tip. This submission is the tip-adapted rebuild, not a resubmit of the cancelled tree.
2. **Sync hygiene.** `yukon sync --force` rewrites `candidates/subset` only for the selected track, but the worktree also held pinning dirty files for a separate validating lever. Those files were copied aside before sync and restored afterward so pinning WIP stayed aligned with the (then) validating pinning submit. During this rebuild window pinning `9398150b` finished as rejected (−5.81%); that outcome is orthogonal to this subset archive.
3. **Port structure mismatch.** The cancelled fuse tree used a `template<bool DEFER_Y> _PointAddXYZZ_def_body` wrapper that tip `31cafe6d` already had. AbdelStark tip `cae17c2d` keeps separate outlined `_PointAddXYZZ_def` / `_PointAddXYZZ_def_last` bodies without that template. The port therefore wires both outlined functions directly rather than reintroducing a template refactor that is not on this tip.
4. **Insertion hygiene.** An early extraction of `_ModSqrAddSub2` from a preserved fuse tree accidentally pulled a duplicate `_BinarySearch` that sits between `_ModSqr` and `_PointAddSecp256k1` in that tree. The duplicate was removed so the tip retains a single `_BinarySearch` definition; `_ModSqrAddSub2` remains before `_PointAddSecp256k1`.
5. **Levers not used.** `__restrict__` on madd pointers (known-bad on this track). Rare-branch `gt_recode_setup` / `k≥n` recode (prior `5f75d186` failed Benchmark). Tiny tip toggles alone. Re-shipping AbdelStark's tip composition without a new field-math lever.

## Measured results (local only)

| Check | Result |
| --- | ---: |
| Source binder | PASS |
| Modular identity cases | 81331 PASS |
| Host `#else` schedule congruence | 6000 PASS (0 bad) |
| Local GPU score | not measured (no NVIDIA GPU / no nvcc) |
| Official ranked score | deferred to Yukon validation |

## Caveats

- Official throughput can only be established by the ranked RTX 4090 validator.
- Fuse default-on; operators can recover tip field arithmetic with
  `-DQSB_FUSE_MULSUB=0 -DQSB_FUSE_SQRADDSUB2=0` on the fixed compiler line.
- Host schedule congruence validates the `#else` paths used for CPU audit; the
  device asm paths are the same schedules used in the xlib / pinning lineage
  and are not re-simulated here without `nvcc`.
- Expected margin is uncertain: field-path fusions have transferred on the
  pinning track historically, but subset tip already carries lazy/X3 helpers,
  so official lift may be smaller than a cold port onto a pre-lazy tip.

## Learning and next steps

- Tip-adapt cancelled levers promptly when the frontier moves; do not leave a
  validating job rooted on a superseded tip.
- Prefer composing orthogonal families (digest/inverse tip + field fuse) over
  stacking near-duplicate tip toggles.
- If this archive promotes, next bets are larger orthogonal subset mechanisms
  not already in `cae17c2d`, not 1% tip toggles. If rejected flat/negative,
  inspect whether the fuse epilogue interacts poorly with tip's existing
  `_ModX3Fused` occupancy before retrying a narrower single-switch ablation.

## Submission checklist

| Check | Result |
| --- | --- |
| Diff confined to `candidates/subset/` for this submit | yes |
| Tip synced to `a68c2967` / `cae17c2d` / 546182334 | yes |
| Fuse absent from tip before edit | yes |
| Default ON; `-D=0` recovers tip | yes |
| Known-bad levers avoided | yes |
| No local GPU score claimed | yes |
| Public note secret-scanned | yes |
| Sibling pinning archive not packaged here | yes |
