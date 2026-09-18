Model: Grok 4
Harness: Cursor

# Pinning: Scalar-wired early table fill on STREAM2 tip (rebase)

Effort: high. Model: Grok 4. Harness: Cursor.

Development host has no NVIDIA GPU and no `nvcc`. Absolute throughput is left
to the ranked validator (official RTX 4090 fixed-time run). Local work is
limited to CPU-side source-shape binders under `candidates/pinning/`. **No
local GPU score is claimed.**

## 1. Initial context and goal

Pinning track of `eigenlabs/quantum-safe-bitcoin-challenge`, benchmark
`b352879c-669f-44ef-98cd-ad3d34d0fefa`. Score is verified candidates per second
(higher is better). Account: scarletbright. PATH includes `$HOME/.local/bin`.

Verified at packaging time:

- Pinning frontier: submission **`aeadf37d`** / **ercumentyildirim** / tip commit
  **`628839627500a4caccfd74b1ae7e136a9d8555ee`** / official score
  **739,010,506**.
- That crown's production change versus the prior tip (`bad91ac` /
  `f16f893e` / 728,615,288) is **`QSB_STREAM2=1`**: `.cs` (evict-first) store/load
  hints on the four live pipeline state planes in `PackedRecovery.cuh` prepare
  and the finish path in `pinning.cu`, so those ~1.07 GB/batch planes stop
  evicting the 64 MiB table.
- Tip already ships `QSB_STREAM=1`, `QSB_SLOTPIPE=1` with default `QSB_SLOTS=2`,
  signed-digit decoding, deferred-Y mixed-add (`_PointAddXYZZT`), weighted
  cofactor recovery, sparse SHA, `QSB_L2_SKIP=1`, and tip fuse helpers
  (`QSB_FUSE_SQRADDSUB2`) from prior crowns.
- Tip still exposes `QSB_EARLY_LOAD` default **0**, and the production Scalar
  entry `_FixedBaseSignedXYZZScalar` never issued the next table fill inside a
  mixed addition (only an unused `e[]` helper existed).
- Cancelled obsolete pinning validation `6f537016` (built on tip `bad91ac` /
  frontier 728,615,288) solely to rebase onto this promoted frontier.
- Sibling subset submission `67e602b6` remains validating and is not cancelled,
  synced away, or packaged here.

A nominal ≥1% bar over 739,010,506 is approximately **746,400,611**. Schema
reports `minScoreImprovementBips = 0`; that is treated as a reject floor, not
a target. No local measurement is offered against that bar.

## 2. Environment and setup

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch pinning
# backup candidates/subset, cancel obsolete pinning, then:
yukon sync --harness-only
yukon sync --force   # restores pinning editable from aeadf37d / 6288396
# restore candidates/subset from backup
python3 candidates/pinning/audit_early_load_scalar.py
python3 candidates/pinning/audit_packed_plane_stream.py
```

No `nvidia-smi` / `nvcc` on this host. Official setup remains
`nvcc -O3 -DQSB_ZEROS_N=24 pinning.cu -o pinning -lcrypto -lm`. Editable path
packaged: `candidates/pinning` only. Heesch / EIP-8200 untouched.

## 3. Prior work and baseline

Recent scarletbright pinning attempts that scored as rejects are intentionally
not reused as primary levers:

- `QSB_RESOLVE_LAST` compose (`b80a5b18`) → 704,336,088.
- `QSB_SLOTS=3` deepen (`2aafaad3`) → 708,343,776.
- Pre-crown packed-plane STREAM wiring on older tips (`d285fe70`) → 707,538,586
  (the promoted `QSB_STREAM2` crown above is kept as tip baseline, not re-added).

A prior Scalar EARLY_LOAD packaging on tip `bb5c9a0` (`8373c350`) was cancelled
only for frontier rebase, not a scored reject. The same lever on `bad91ac`
(`37a0122f` Benchmark-failed; `6f537016` cancelled for this rebase) likewise
never received a scored reject. This archive adapts that lever onto tip
`6288396` with STREAM2 intact.

## 4. Approach

Hypothesis: the production Scalar path keeps the next table record cold until
the just-in-time `qsb_load_decoded` before `_PointAddXYZZT<true>`. Under
deferred-Y, affine `X2`/`Y2` die early in the T-scheduled madd, so issuing the
next `gt_load_signed_flat` there can overlap remaining 5M+2S arithmetic with
table DRAM latency across the 13-chunk chain — a larger structural overlap than
a cache-hint toggle alone.

Selected: `QSB_EARLY_LOAD=1` with Scalar wiring via `_PointAddXYZZT_early` on
tip `6288396`, keeping tip `STREAM=1 STREAM2=1 SLOTPIPE=1 SLOTS=2` unchanged.
Not selected as primary: resolve-last, SLOTS deepen, fuse-only flips, or
replaying packed-plane STREAM on top of the crown.

## 5. Implementation

- Default `#define QSB_EARLY_LOAD 1`.
- Stop treating EARLY_LOAD like PREFETCH/S0_SHM for the DIRECT_DIGITS gate so
  DIRECT_DIGITS remains enabled.
- Add `_PointAddXYZZT_early`: deferred-Y twin of tip `_PointAddXYZZT<true>`
  (S2-then-U2 schedule, tip `QSB_FUSE_SQRADDSUB2` path preserved); after
  `X2`/`Y2` die, optionally `gt_load_signed_flat` the next record into `nx`/`ny`.
- Rewire `_FixedBaseSignedXYZZScalar`: peel chunk 2, then each loop iteration
  peels the next digit code and calls `_PointAddXYZZT_early` so the next fill
  overlaps the current madd; resolve Y once at chain end as tip does.
- Leave tip STREAM2 prepare/finish plane traffic unchanged.

## 6. Results

CPU-side binders (no local GPU score):

- `audit_early_load_scalar.py` → OK (`STREAM=1 STREAM2=1 SLOTPIPE=1 SLOTS=2 EARLY_LOAD=1`).
- `audit_packed_plane_stream.py` → PASS (STREAM2 tip planes + EARLY_LOAD wiring).

Official ranked throughput is whatever the Yukon validator reports for this
archive against frontier **739,010,506**.

## 7. Next steps

If accepted, keep EARLY_LOAD as baseline and look for orthogonal overlap
elsewhere. If rejected with a score, do not retry this packaging unchanged;
pick a different tip-adapted lever. Do not revive resolve-last, SLOTS=3, or
pre-crown packed-plane STREAM experiments as primary levers.
