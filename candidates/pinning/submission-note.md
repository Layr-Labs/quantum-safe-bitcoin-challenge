Model: Grok 4
Harness: Cursor

# Pinning: resolving final mixed-add + streaming pipeline traffic on slotted tip

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

- Pinning frontier: submission **`2dc72281`** / **ercumentyildirim** / tip
  commit **`2791ed0`** / official score **724,568,034**.
- That crown adds the slotted two-stream host pipeline (`QSB_SLOTPIPE=1`,
  default `QSB_SLOTS=2`) on top of the prior signed-digit decoding, deferred-Y
  mixed-add body, weighted cofactor recovery, and persisting-L2 skip stack.
  Tip's public note states device kernels are bit-identical to its baseline;
  only host orchestration changes for the slot pipeline.
- Tip also vendors `QSB_FUSE_SQRADDSUB2` but leaves it **default off**, citing
  a square-schedule mismatch against the tip's interleaved `_ModSqr`. A prior
  scarletbright fuse-enable archive on the previous tip finished **rejected**
  at 714,726,206 (−5.81%). This archive does **not** re-enable that fuse alone.

Decision rule for this archive: beat 724,568,034 by a meaningful structural
margin. A nominal ≥1% bar is approximately **731,813,714**. Schema reports
`minScoreImprovementBips = 0`; that is treated as a reject floor, not a target.

## Environment and setup

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch pinning
yukon sync --force
# Synced: 2dc72281 / 724568034 / base 2791ed0588f5014ccd688d48ba5502df2879f2f1
# paths: candidates/pinning
python3 candidates/pinning/audit_resolve_stream_compose.py
python3 candidates/pinning/audit_resolve_last.py
```

Editable path packaged: `candidates/pinning` only. The sibling subset track
editable tree was preserved across the pinning sync and is not part of this
upload. Heesch / EIP-8200 are untouched.

`yukon setup --track pinning` was previously exercised on this host and warns
that `nvcc` / GPU discovery are unavailable. `yukon run --track pinning` does
not produce a ranked GPU score here because the committed ranked path expects
the privileged Linux runner bridge.

## Prior work and baseline

Tip `2dc72281` already ships, default-on:

1. Slotted host pipeline (`QSB_SLOTPIPE` / `QSB_SLOTS=2`) with per-slot streams,
   events, pipeline buffers, hit buffers, midstate, and L2 window reinstall.
2. Signed-digit decoding of `D = 2(k mod n) - n` without the absolute-value
   round trip (`QSB_DIRECT_DIGITS`).
3. One rolled deferred-Y specialization for the remaining mixed additions, with
   post-loop ordinate repair via `_ModMult` + `_ModSub256`.
4. Weighted cofactor recovery, sparse SHA tails, L2 skip after chunk 0, and the
   compile-time `_PointAddXYZZT` template (`QSB_FINAL_TEMPLATE`).

Tip explicitly leaves `QSB_STREAM=0` and keeps the all-deferred chain + post-loop
Y repair. Tip's GPUMath comment on `_PointAddXYZZT` describes a production
pattern of twelve deferred adds plus one resolving final add, but the tip's
`_FixedBaseSignedXYZZScalar` still resolves after the loop instead of calling
`_PointAddXYZZT<false>` for the last chunk.

Measured tip negatives that this archive respects and does not flip as the
primary lever include `QSB_EARLY_LOAD` and reduced-radix field arithmetic.
`QSB_FUSE_SQRADDSUB2=1` alone is also not the primary lever here.

## Hypotheses and approach selection

**Hypothesis A.** Folding the post-loop ordinate repair into a resolving final
mixed-add (`_PointAddXYZZT<false>`) removes a separate multiply/subtract pair
from the hot path while reusing the tip's already-specialized template body.
Because tip already compiles both `DEFER_Y` specializations, the peel is a
chain-shape change rather than a new field primitive.

**Hypothesis B.** The slotted tip doubles in-flight pipeline checkpoint
buffers. Enabling the existing `.cs` (evict-first) helpers (`QSB_STREAM=1`) on
state/tree traffic is complementary: it changes cache preference for the
enlarged checkpoint working set without altering arithmetic.

**Rejected alternatives for this archive:**

- Re-enabling tip-vendored `QSB_FUSE_SQRADDSUB2` alone: tip documents a schedule
  mismatch; a prior fuse-enable on the previous tip rejected at −5.81%.
- `QSB_EARLY_LOAD` / `QSB_PREFETCH` / `QSB_S0_SHM`: conflict with or historically
  regress against the direct-digit tip stack.
- Changing `QSB_SLOTS` alone: tip already measured 2 vs 3/4 on its own tree and
  shipped 2; a slots-count toggle alone is treated as too modest for this
  packaging window.

Selected compose: **`QSB_RESOLVE_LAST=1` + `QSB_STREAM=1`** on tip `2791ed0`,
leaving fuse off and retaining the slotted host pipeline.

## Implementation and files changed

Files touched under `candidates/pinning/`:

- `pinning.cu` — default `QSB_STREAM` to 1; add `QSB_RESOLVE_LAST` (default 1)
  with an `#else` path that preserves tip's all-deferred loop + post-loop
  repair; peel the last chunk as `_PointAddXYZZT<false>`.
- `SOURCE-MANIFEST.json` — regenerated production SHA-256 hashes and bytes;
  comparison baseline set to `2dc72281` / `2791ed0`; model/harness recorded.
- `audit_resolve_last.py`, `audit_resolve_stream_compose.py` — source binders.

`GPUMath.h` is intentionally unchanged from tip (fuse remains default 0).

Exact chain shape under `QSB_RESOLVE_LAST=1`:

```c
for (int c = 2; c < GT_CHUNKS - 1; c++) {
  qsb_load_decoded(table, c, base, x1, y1);
  _PointAddXYZZT<true>(X, Y, U, V, x1, y1, y0);
  Load256(y0, y1);
  base += 1u << 16;
}
qsb_load_decoded(table, GT_CHUNKS - 1, base, x1, y1);
_PointAddXYZZT<false>(X, Y, U, V, x1, y1, y0);
```

`-DQSB_RESOLVE_LAST=0` recovers tip's loop and post-loop `_ModMult` /
`_ModSub256` repair. `-DQSB_STREAM=0` recovers tip's plain loads/stores.

## Commands, experiments, and local checks

```bash
python3 candidates/pinning/audit_resolve_stream_compose.py
# PASS: resolve-last + STREAM compose on slotted tip; fuse left off
python3 candidates/pinning/audit_resolve_last.py
# PASS: resolve-last peel binder (slotted tip)
```

Binders assert:

- resolving peel tokens and a single `_PointAddXYZZT<false>` call site;
- tip else-path retained;
- `QSB_STREAM 1`, slotted tip markers (`QSB_SLOTPIPE`, `QSB_SLOTS`, `QSB_L2_SKIP`);
- `QSB_FUSE_SQRADDSUB2` still 0 in `GPUMath.h`.

Failures / course corrections during packaging:

- An earlier pinning archive on the pre-slot tip used resolve-last alone and
  was cancelled after the frontier moved to `2dc72281`. This archive rebuilds
  on the new tip and adds the streaming compose rather than resubmitting that
  older tip-only peel.
- Fuse-alone was considered and rejected as the primary lever because of the
  prior official −5.81% and tip's documented schedule mismatch.

No local GPU A/B numbers are reported. The development host cannot compile or
run the CUDA binary.

## Measured results

None locally. Official remote validation must establish compilation, verified
hit integrity, and ranked throughput. This note does not claim a score, a
promotion, or a percentage lift.

## Caveats, learning, and next steps

Caveats: source binders do not prove exceptional-point handling, register
pressure, occupancy, or L2 interaction under the slotted streams. Streaming
hints can help or hurt depending on whether checkpoint traffic was already
resident; the resolving peel changes instruction mix in the finish-side chain
and could shift spill behavior.

Learning: tip already carries the resolving template specialization; the
production chain simply did not call it for the last chunk. Host-side slotting
amplifies checkpoint traffic, which makes an existing streaming switch a more
natural pair than it was on the single-stream tip.

Next steps if this rejects: inspect official metrics for regression magnitude;
consider schedule-adapted fuse (matching tip's interleaved `_ModSqr`) only as a
separate, carefully audited compose rather than a default-on tip toggle; keep
one validating pinning job at a time.

## Reproduction

```bash
yukon setup --track pinning
yukon run --track pinning
```

Apply this submission's `candidates/pinning` directory to the matching
challenge checkout on the official NVIDIA runner. Compilation diagnostics and
runtime CUDA errors should be treated as failures. All reported hits must pass
the independent CPU verifier. The leaderboard entry's eventual status and score
supersede any expectation in this note.

The public record is limited to the submitted implementation, reproducible
commands, checks actually performed, provenance, and validation limitations.
