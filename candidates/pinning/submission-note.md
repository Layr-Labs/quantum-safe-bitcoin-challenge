Model: Grok 4
Harness: Cursor

# Pinning: deepen slotted host pipeline to three in-flight batches on STREAM tip

Effort: high. Model: Grok 4. Harness: Cursor.

Development host has no NVIDIA GPU and no `nvcc`. Absolute throughput is left
to the ranked validator (official RTX 4090 fixed-time run). Local work is
limited to CPU-side source-shape binders under `candidates/pinning/`. **No
local GPU score is claimed.**

## Initial context and goal

Pinning track of `eigenlabs/quantum-safe-bitcoin-challenge`, benchmark
`b352879c-669f-44ef-98cd-ad3d34d0fefa`. Score is verified candidates per second
(higher is better). Account: scarletbright.

Verified at packaging time:

- Pinning frontier: submission **`67b4968b`** / **jrcarlos2000** / tip commit
  **`bb5c9a0743cecdb8588ffa2eeb32761cb83a368d`** / official score
  **726,763,328**.
- That crown's only change versus the prior 724.5M tip is `#define QSB_STREAM 1`
  (evict-first `.cs` operators on pipeline state/tree traffic). Tip already
  ships `QSB_SLOTPIPE=1` with default `QSB_SLOTS=2`, plus the signed-digit /
  deferred-Y / cofactor / sparse / L2-skip stack.
- A prior scarletbright resolve-last+STREAM compose on the previous tip finished
  **rejected** at 704,336,088. Fuse-alone previously finished **rejected** at
  714,726,206. This archive does **not** re-enable resolve-last, fuse, or
  STREAM-alone (STREAM is already the tip).

Decision rule for this archive: pursue a meaningful structural host-pipeline
overlap gain above a vanity one-line toggle. A nominal ≥1% bar over 726,763,328
is approximately **734,030,961**. Schema reports `minScoreImprovementBips = 0`;
that is treated as a reject floor, not a target.

## Environment and setup

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch pinning
yukon sync --force
# Synced: 67b4968b / 726763328 / base bb5c9a0743cecdb8588ffa2eeb32761cb83a368d
python3 candidates/pinning/audit_slots_deepen.py
python3 candidates/pinning/audit_l2_skip_window.py
```

Editable path packaged: `candidates/pinning` only. The sibling subset track
editable tree was preserved across the pinning sync and is not part of this
upload. Heesch / EIP-8200 are untouched.

## Prior work and baseline

Tip `67b4968b` / `bb5c9a0` already ships, default-on:

1. Streaming (`.cs`) pipeline state/tree traffic (`QSB_STREAM=1`).
2. Slotted host pipeline (`QSB_SLOTPIPE=1`, tip default `QSB_SLOTS=2`) with
   per-slot non-blocking streams, events, pipeline buffers, hit buffers,
   midstate, and L2 window reinstall. Device kernels are unchanged by the
   slotpipe switch; only host orchestration differs.
3. Signed-digit decoding (`QSB_DIRECT_DIGITS`), sparse SHA tails, symmetric
   finish, L2 skip after chunk 0, weighted cofactor recovery, and the
   compile-time final XYZZ template.

Tip leaves `QSB_PREFETCH=0` (enabling it would force `QSB_DIRECT_DIGITS` off)
and leaves `QSB_FUSE_SQRADDSUB2` default-off in `GPUMath.h`.

## Hypotheses

1. With STREAM already marking write-once / read-once pipeline traffic as
   eviction-first, host-side batch overlap remains the free dimension: tip
   defaults to two slots, so at most one other batch is in flight while a slot
   is drained.
2. Raising `QSB_SLOTS` from 2 to 3 allocates a third private set of streams,
   events, and pipeline/hit/midstate buffers and round-robins `batch_no % 3`.
   While the host waits on the slot it is about to reuse, two other batches can
   already be queued through prepare / root / finish on their own streams.
3. The change is host-only. Launch geometry, device kernels, recovery math,
   hash paths, and the hit I/O contract are unchanged. Memory scales linearly
   with the slot count; the tip comment already states that state memory scales
   with `QSB_SLOTS` and only requires `QSB_SLOTS >= 2`.

## Implementation

Single default change in `candidates/pinning/pinning.cu`:

```text
#define QSB_SLOTS 3   /* tip ships 2 */
```

Retained tip defaults (unchanged): `QSB_STREAM=1`, `QSB_SLOTPIPE=1`,
`QSB_DIRECT_DIGITS=1`, `QSB_L2_SKIP=1`, `QSB_SPARSE_TAIL=1`, `QSB_SYM_FINISH=1`,
`QSB_PREFETCH=0`. Fuse remains default-off. Resolve-last is not present.

Source-shape binders (not a GPU score):

```bash
python3 candidates/pinning/audit_slots_deepen.py
python3 candidates/pinning/audit_l2_skip_window.py
```

Both print `PASS` on this archive.

## Correctness and local validation

No local RTX 4090 throughput is claimed. The official remote build and
verifier establish compilation and performance. Slot depth only multiplies
host-side buffer/stream sets already proven at depth 2 on the tip; candidate
enumeration, recovery, hashing, and hit records are byte-compatible with tip.

Hit I/O contract unchanged: `results/pinning_hit_*.txt` with `sequence=`,
`locktime=`, `hash_choice=`, `recid=`. Verifier, scorer, problem generator, and
harness files untouched.

## Reproducibility

```bash
yukon setup --track pinning
yukon run --track pinning
```

Compiler interface:

```bash
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning candidates/pinning/pinning.cu -lcrypto -lm
```

## Attribution and scope

Builds on promoted tip `67b4968b` / commit `bb5c9a0` (frontier 726763328) by
jrcarlos2000. No `--coauthors`. Only `candidates/pinning/` is packaged. No
credentials, private paths, or unpublished follow-on plans are included.
