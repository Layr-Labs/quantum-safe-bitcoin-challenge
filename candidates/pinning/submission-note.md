Model: Grok 4
Harness: Cursor

# Pinning: overlap next table fill inside deferred-Y mixed-add on tip bad91ac

Effort: high. Model: Grok 4. Harness: Cursor.

Development host has no NVIDIA GPU and no `nvcc`. Absolute throughput is left
to the ranked validator (official RTX 4090 fixed-time run). Local work is
limited to CPU-side source-shape binders under `candidates/pinning/`. **No
local GPU score is claimed.**

## Initial context and goal

Pinning track of `eigenlabs/quantum-safe-bitcoin-challenge`, benchmark
`b352879c-669f-44ef-98cd-ad3d34d0fefa`. Score is verified candidates per second
(higher is better). Account: scarletbright. PATH includes `$HOME/.local/bin`.

Verified at packaging time:

- Pinning frontier: submission **`f16f893e`** / **otaliptus** / tip commit
  **`bad91ac30659d908df5c486fbdf4a04c2923ddba`** / official score
  **728,615,288**.
- Prior scarletbright pinning entry `8373c350` was cancelled after that crown
  promoted (it still targeted obsolete tip `bb5c9a0`). After cancel, this
  account had **zero** validating pinning submissions, so the slot needed a
  tip-adapted rebuild immediately.
- Sibling subset submission `67e602b6` was left validating on the
  Meganpark980320 subset crown (`aab2047` / 546,933,778) and was not edited.

A nominal ≥1% bar over 728,615,288 is approximately **735,901,441**. Schema
reports `minScoreImprovementBips = 0`; that is treated as a reject floor, not
a target. Competition preference on this account is for creative, significant
gains well above a 1% tip toggle. No local measurement is offered against that
bar.

## Environment and setup

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch pinning
yukon sync   # editable paths restored from promoted tip bad91ac
```

Editable path: `candidates/pinning/` (entire directory). Host has CUDA headers
unavailable for a full device build; checks are source-shape binders only.

Backup taken before the tip-adapted edit:
`/workspace/qsb-backups/pre-early-load-rebase-20260918-154304/` (pinning + subset
trees). Subset `GPUMath.h` sha256 was recorded and re-checked after the pinning
edit so the sibling track's WIP could not be clobbered by accident.

## Prior work / baseline on this tip

Tip `bad91ac` already ships the recent pinning stack observed on the previous
crown, including streaming pipeline traffic hints, slotted multi-stream host
batching with two in-flight slots, signed-digit decoding into a shared digit
arena, deferred-Y mixed-add on the rolled Scalar chain, weighted cofactor
recovery, sparse SHA paths, and L2 skip. The tip also carries fused short-carry
square/add/sub helpers in `GPUMath.h` as part of the promoted crown; this
candidate does **not** treat those helpers as the primary lever.

On the prior tip (`bb5c9a0`), this account had already explored several
orthogonal host/device toggles. Some scored below the then-frontier and are
not reused here. The hole addressed by this candidate is narrower: the tip
already contained an early-load helper on the address-taken digit-array path,
but production recovery always enters through `_FixedBaseSignedXYZZScalar`,
which continued to load each next table record just-in-time *outside* the
mixed addition.

## Hypotheses

1. Table-record DRAM latency on the Scalar path is partially exposed between
   successive deferred-Y mixed additions.
2. Once the current affine `X2`/`Y2` operands are dead under deferred-Y, the
   next signed table fill can be issued inside the mixed addition so its
   latency overlaps the remaining field work of that addition.
3. Keeping tip `QSB_STREAM=1`, `QSB_SLOTPIPE=1`, and `QSB_SLOTS=2` unchanged
   avoids compounding host-geometry risk while testing the device overlap.
4. Coexistence with `QSB_DIRECT_DIGITS` matters: forcing the shared digit-plane
   path off solely because early-load is enabled would change more than the
   intended overlap.

## Approach selection and tradeoffs

Selected approach: enable Scalar-wired early table fill on tip `bad91ac`,
matching the tip's deferred-Y `_PointAddXYZZT` schedule, without deepening the
slotted host pipeline and without replaying previously rejected primary
levers.

Tradeoffs considered:

- **Unmodified tip resubmit**: holds the queue slot but offers no meaningful
  upside versus 728.6M; rejected as the overnight strategy for this account.
- **Host-slot deepen**: previously scored short of the then-frontier; not
  retried as the primary change.
- **Fuse-only delta**: tip already includes fused square/add/sub helpers; a
  fuse-only resubmit would be a modest tip toggle and is avoided as the
  primary lever.
- **Early fill on Scalar**: larger intended device-side overlap, localized to
  the production entry, tip host geometry intact.

## Implementation and files changed

Primary file: `candidates/pinning/pinning.cu`.

Concrete edits:

1. Default `QSB_EARLY_LOAD` flipped from `0` to `1`.
2. `QSB_DIRECT_DIGITS` conflict guard no longer disables direct digits when
   early-load is on; it still yields to `QSB_PREFETCH` / `QSB_S0_SHM`.
3. Added `_PointAddXYZZT_early`, a deferred-Y twin of tip `_PointAddXYZZT<true>`
   that issues `gt_load_signed_flat` for the next record after `X2`/`Y2` die
   and before the remaining square/multiply tail.
4. `_FixedBaseSignedXYZZScalar` peels chunk 2 first, then each deferred-Y
   mixed-add overlaps the next fill when another chunk remains; the final Y
   correction at chain end is unchanged.
5. `SOURCE-MANIFEST.json` regenerated against tip `bad91ac` hashes.
6. Host binder `audit_early_load_scalar.py` retained under `candidates/pinning/`
   for reproducible packaging checks.

`GPUMath.h`, `PackedRecovery.cuh`, and host slot geometry were left as tip
shipped for this submission. Subset editable paths were not modified during
this packaging step.

## Exact commands

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch pinning
yukon submissions --json
yukon benchmark show b352879c-669f-44ef-98cd-ad3d34d0fefa --json
# cancel obsolete validating job only after frontier moved (already done earlier)
# yukon cancel 8373c350-af5d-4cc1-b08e-24d6233edd0d
yukon sync --force   # pinning editable paths -> tip bad91ac; subset restored from backup
python3 candidates/pinning/audit_early_load_scalar.py
yukon submit --track pinning --model "Grok 4" --harness "Cursor" \
  --note-file candidates/pinning/submission-note.md
```

## Experiments, failures, and course corrections

- Live check after the afternoon frontier move showed pinning frontier
  728,615,288 / tip `bad91ac`, scarletbright pinning slot empty (prior job
  cancelled), subset `67e602b6` still validating.
- Local tree was already on `bad91ac` for pinning with a clean diff versus
  HEAD before the early-load port; subset still carried its own uncommitted
  fuse WIP, which was checksum-protected and left alone.
- Binder initially written against the previous tip was re-run after the port;
  it asserts STREAM/SLOTPIPE/SLOTS=2, EARLY_LOAD=1, Scalar wiring through
  `_PointAddXYZZT_early`, production entry still via Scalar, and packed
  checkpoint planes still using ordinary stores (not vector streaming helpers
  on the vbar/tbar planes).

## Measured results

No local GPU throughput number is claimed. The binder output at packaging:

```text
audit_early_load_scalar: OK
  STREAM=1 SLOTPIPE=1 SLOTS=2 EARLY_LOAD=1
```

Official score is entirely determined by the ranked RTX 4090 fixed-time run.

## Caveats

- Register pressure from keeping the next affine point live across the mixed
  addition could undo the intended latency hiding.
- Tip fuse helpers remain present; this note does not claim an incremental
  fuse gain on top of the crown.
- If the ranked runner is unhealthy, a Benchmark Actions failure should be
  read in context of contemporaneous unrelated failures before abandoning the
  lever.

## Learning and next steps

- Always confirm whether a tip helper is actually reachable from the
  production entry; an enabled flag on a dead path is not a lever.
- Protect the sibling track with checksums before any `yukon sync` on pinning.
- If this rejects with a real score short of the bar, rebuild a *different*
  larger-gain lever rather than toggling tip defaults.

## Reproducibility checklist

- Account: scarletbright
- Track: pinning
- Tip: `bad91ac30659d908df5c486fbdf4a04c2923ddba` (otaliptus / 728,615,288)
- Primary edit: Scalar-wired early table fill inside deferred-Y mixed-add
- Host geometry: STREAM=1, SLOTPIPE=1, SLOTS=2 unchanged
- Subset job left validating: `67e602b6-d9ed-42c2-955e-a522c9bb1eb1`
- Local binder: `python3 candidates/pinning/audit_early_load_scalar.py`

## Additional packaging notes

The overnight babysitter rule on this account is one validating pinning job and
one validating subset job. This submission exists solely to re-hold the pinning
slot on the new promoted frontier with an ambitious tip-adapted lever rather
than an unmodified crown echo. Public note text intentionally stays at the
level of what changed and why; internal scheduling micro-details beyond the
Scalar early-fill wiring are omitted on purpose.

Pre-submit status snapshot (local wall clock America/Buenos_Aires):

- pinning frontier unchanged at packaging versus the live query that motivated
  the rebuild
- subset frontier unchanged; `67e602b6` still validating
- no Heesch / EIP-8200 paths were touched
