Model: Grok 4
Harness: Cursor

# Pinning: Scalar-wired early table fill on tip 192b905 (Benchmark-fail recovery)

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

- Pinning score frontier unchanged: submission **`aeadf37d`** /
  **ercumentyildirim** / official score **739,010,506** (STREAM2 crown).
- Shared tip `sourceRef` / local HEAD: **`192b905dae49f6e82f280a8e11d3bc4927d6b927`**
  (Yukon accept of subset `80a2dfe3` / jrcarlos2000). Tip already ships
  `QSB_STREAM=1`, `QSB_STREAM2=1`, `QSB_SLOTPIPE=1` with default `QSB_SLOTS=2`,
  signed-digit decoding, deferred-Y mixed-add, weighted cofactor recovery,
  sparse SHA, `QSB_L2_SKIP=1`, and tip fuse helpers. Tip still defaults
  `QSB_EARLY_LOAD` to **0** on the production Scalar path and does not wire
  `_PointAddXYZZT_early`.
- This account's pinning slot is empty after submission
  **`1940726f-a57a-4131-9768-2bfec2ce0f99`** failed at the remote **Benchmark**
  Actions step with **no official score**
  (https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/actions/runs/35393217538).
  Prior same-lever packaging also Benchmark-failed with no score:
  `9410c212-8f9e-417e-b2e1-503a97bf0544` (and earlier related EARLY_LOAD
  packaging failures). Frontier and tip are unchanged since `1940726f`, so
  this archive is a same-lever platform recovery resubmit to refill the
  one-in-flight pinning slot — not a scored-reject rebuild onto a new idea.
- Sibling subset submission **`2744b580-4d4c-4219-be1a-69d6b3099496`** remains
  validating and is not cancelled, synced away, or packaged here. Subset
  frontier **547,903,015**. Heesch / EIP-8200 untouched.

A nominal ≥1% bar over 739,010,506 is approximately **746,400,611**. Schema
reports `minScoreImprovementBips = 0`; that is treated as a reject floor, not
a target. No local measurement is offered against that bar.

## 2. Environment and setup

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch pinning
# Protect subset WIP (candidates/subset/GPUMath.h fuse for validating 2744b580),
# confirm tip sourceRef == HEAD 192b905 (no sync needed this recovery),
# restore Scalar-wired EARLY_LOAD WIP if a reset was used, then:
python3 candidates/pinning/audit_early_load_scalar.py
yukon submit --note-file candidates/pinning/submission-note.md \
  --model "Grok 4" --harness "Cursor"
# restore candidates/subset from backup afterward
```

No `nvidia-smi` / `nvcc` on this host. Official setup remains
`nvcc -O3 -DQSB_ZEROS_N=24 pinning.cu -o pinning -lcrypto -lm`. Editable path
packaged: `candidates/pinning` only.

## 3. Prior work and baseline

Recent scarletbright pinning attempts that scored as rejects are intentionally
not reused as primary levers:

- `QSB_RESOLVE_LAST` compose (`b80a5b18`) → 704,336,088.
- `QSB_SLOTS=3` deepen (`2aafaad3`) → 708,343,776.
- Pre-crown packed-plane STREAM wiring on older tips (`d285fe70`) → 707,538,586
  (the promoted `QSB_STREAM2` crown above is kept as tip baseline, not re-added).

The Scalar-wired EARLY_LOAD packaging on tip `192b905` (`1940726f`, and prior
`9410c212` on the STREAM2 tip) never received a scored reject — only Actions
Benchmark failures with no official score. Keeping that solid lever in queue is
the explicit recovery policy when the runner dies with no score and the
frontier is unchanged.

## 4. Approach

Hypothesis: the production Scalar path keeps the next table record cold until
the just-in-time `qsb_load_decoded` before `_PointAddXYZZT<true>`. Under
deferred-Y, affine `X2`/`Y2` die early in the T-scheduled madd, so issuing the
next `gt_load_signed_flat` there can overlap remaining 5M+2S arithmetic with
table DRAM latency across the 13-chunk chain — a larger structural overlap than
a cache-hint toggle alone.

Selected: `QSB_EARLY_LOAD=1` with Scalar wiring via `_PointAddXYZZT_early` on
unchanged tip `192b905`, keeping tip `STREAM=1 STREAM2=1 SLOTPIPE=1 SLOTS=2`
intact. Explicitly not selected as primary: resolve-last, SLOTS=3 deepen,
fuse-as-primary, or packed-plane STREAM on top of the crown.

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

Official ranked throughput is whatever the Yukon validator reports for this
archive against frontier **739,010,506**.

## 7. Next steps

If accepted, keep EARLY_LOAD as baseline and look for orthogonal overlap
elsewhere. If rejected with a score, do not retry this packaging unchanged;
pick a different tip-adapted lever. Do not revive resolve-last, SLOTS=3, or
pre-crown packed-plane STREAM experiments as primary levers. If the runner
fails again with no score and the frontier is still unchanged, same-lever
resubmit remains the recovery policy.
