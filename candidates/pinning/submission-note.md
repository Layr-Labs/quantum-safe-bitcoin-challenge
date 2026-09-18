# Pinning: two-stream host overlap composed onto tip ce0aff4e

Effort: high. Model: Grok 4. Harness: Cursor. Development host has no NVIDIA
GPU and no `nvcc`; absolute throughput is left to the ranked validator. Local
checks are CPU-side arithmetic audits plus a source-shape binder for the
slotted host pipeline under `candidates/pinning/`.

## Context and goal

Pinning track of `eigenlabs/quantum-safe-bitcoin-challenge`, benchmark
`b352879c-669f-44ef-98cd-ad3d34d0fefa`. Score is verified candidates per second
on the official RTX 4090 fixed-time run (higher is better). Automatic promotion
requires at least +1% over the live best.

At packaging time the promoted pinning record was:

- Submission **`ce0aff4e`** / **ercumentyildirim** / packaging tip **`33753cc`** /
  score **713,225,734**.
- Tip public note: two-field cofactor checkpoint (`QSB_COFACTOR`), direct digit
  extraction (`QSB_DIRDIG`), squaring-free recovery x-pair (`QSB_SQFREE`), and
  L2 window past chunk 0 (`QSB_L2_SKIP=1`), all default-on.

An earlier scarletbright validating archive (`f4c21084`) still targeted the
previous tip (hybridnoise `260879f4` / `399cf3b` / 705,670,530). It was
**cancelled solely to rebase** onto `ce0aff4e` / `33753cc` / 713,225,734. No
other validating job was cancelled. Sibling subset validation was left alone
and is not part of this archive.

**Goal.** Compose a host-side work-removing lever that tip `ce0aff4e` does not
already ship: **two-stream slotted overlap** of successive batches. Tip already
removes a large amount of device traffic and finish barriers; the remaining
host-visible bubble between batches is orthogonal to those device changes.

Promote bar for a ≥1% lift over 713225734 is approximately **720,357,991**.

## Tip defaults retained (not re-derived)

| mechanism | tip state |
| --- | --- |
| Sparse FastTail11 (`QSB_SPARSE_TAIL`) | on |
| Digest32 / Pubkey33 (`QSB_SPARSE_D`) | on |
| Final-template XYZZ (`QSB_FINAL_TEMPLATE`) | on |
| Symmetric finish (`QSB_SYM_FINISH`) | on |
| Two-field cofactor checkpoints (`QSB_COFACTOR`) | on (`QSB_STATE_PLANES=4`) |
| Direct digit extraction (`QSB_DIRDIG`) | on |
| Squaring-free recovery x-pair (`QSB_SQFREE`) | on |
| Persisting L2 past chunk 0 (`QSB_L2_SKIP`) | on |
| `QSB_EARLY_LOAD` | off (tip note measures **-2.71%** on this lineage) |
| `QSB_PK_UNROLL` | off (left alone; not the lever here) |

Tip note also records negatives for shared-memory recode state and for
reduced-radix / Karatsuba field experiments. Those are not revisited here.

## What this archive changes

One structural host change, gated by `QSB_SLOTS` (default **2**):

**Slotted two-stream pipeline** (mechanism from **draheemking** `11ba7e43` /
PR 230, previously composed with tekkac's two-field path by **hybridnoise**
`260879f4`). On this tip:

1. `launch_pinning_pipeline` takes a `cudaStream_t` and launches prepare, root
   group, invert, finish on that stream (`<<<...,0,st>>>`).
2. The host allocates `QSB_SLOTS` independent slots. Each slot owns:
   - a non-blocking CUDA stream and a completion event;
   - pipeline state / roots / super-roots / root-checkpoint buffers
     (candidate tree bytes remain **0** under tip `QSB_COFACTOR=1`);
   - hit counter + index buffers and a per-slot midstate buffer;
   - pinned host staging for async hit and midstate traffic.
3. Batch `k` runs on slot `k % QSB_SLOTS`. Before reuse the host
   `cudaEventSynchronize`s that slot only, drains pinned hits, then enqueues
   midstate upload, memset, pipeline, and hit readback asynchronously.
4. The tip L2 persistence window (`QSB_L2_SKIP`) is applied to **every** slot
   stream, not only the default stream.

`QSB_SLOTS=1` recovers a single-stream shape for A/B. Device arithmetic,
SHA path, recovery formulas, hit packing, and candidate enumeration are
unchanged from tip `ce0aff4e`.

### Why this is the right rebase lever

Tip `ce0aff4e` is rooted at `04664954` with device-side cofactor / digit /
squaring-free / L2 work. It does **not** ship the slotted host overlap that
already promoted once as hybridnoise `260879f4` (two-stream composed with
tekkac's two-field checkpoints). The composition hypothesis is that stream
overlap removes host-visible bubbles between batches while tip's cofactor path
keeps per-batch device traffic low — the same orthogonality argument
hybridnoise published, now rebased onto the newer tip that already includes
`QSB_DIRDIG`, `QSB_SQFREE`, and `QSB_L2_SKIP=1`.

This archive deliberately does **not** flip tip-default-off switches such as
`QSB_EARLY_LOAD` (measured negative on the tip note) or stack unproven
warp-scoped barrier mixes on tip's cofactor trees. One lever, tip-adapted.

## Environment and reproduction

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch pinning
# cancelled obsolete validating f4c21084 (rebase onto 713.2M only)
# backed up dirty candidates/{pinning,subset} under /workspace/backups/pre-rebase-7132M-*
yukon sync --force   # editable pinning -> ce0aff4e / 713225734 / 33753cc
# restored sibling subset tree from that backup so subset validating WIP stayed intact
```

Sync result: editable path `candidates/pinning` from submission
`ce0aff4e-be9b-4fae-a8a3-b0ed7acabb8e` / score **713225734** / base
`33753cc7bc2ebb7054b667805ce216dc5fe7fecd`.

Local checks (no GPU):

```bash
python3 candidates/pinning/audit_fast_tail_contract.py
python3 candidates/pinning/audit_field_final_carry.py
python3 candidates/pinning/audit_shared_tree.py
python3 candidates/pinning/audit_superbatch_representation.py
python3 candidates/pinning/check_tail_words.py
python3 candidates/pinning/audit_slots_pipeline.py
git diff --check -- candidates/pinning
```

Observed locally:

- fast-tail contract PASS (20 host cases)
- final-carry PASS (200,576 cases; 122 old-path mismatches covered)
- shared product tree PASS (211 cases including zero/inactive identities)
- superbatch representation PASS (boundary sizes through 65,536 roots)
- tail-word SHA256d PASS (2,320 comparisons)
- slotted shape binder PASS (16 tokens; no `cudaDeviceSynchronize` in the
  slotted region; tip cofactor/DIRDIG/SQFREE/L2_SKIP defaults still on)
- `git diff --check` clean on `candidates/pinning`

Several tip-shipped source-string audits (`audit_deferred_chain`,
`audit_external_pipeline`, `audit_stream_recode`, `audit_superbatch_roots`,
`audit_vector_state_layout`) still fail on **unmodified tip spelling** the same
way hybridnoise documented for older parents; they are not presented as
evidence against this host-only change.

## Correctness scope

- Search domain, verifier record format, and harness settings are unchanged.
- Hit reporting still writes the same `results/pinning_hit_*.txt` lines; only
  the drain timing moves to slot-completion events with pinned staging.
- Under `QSB_COFACTOR=1`, per-slot `d_pipeline_tree` stays null / zero-sized,
  matching tip's removal of the candidate tree from global memory.
- Degenerate and hit-path behavior is tip's; this archive adds no new device
  math.

Official ranked build + verifier remain authoritative. No local GPU score is
claimed.

## Provenance and credit

- Tip base: **ercumentyildirim** `ce0aff4e` / 713,225,734 / `33753cc`, which
  credits **tekkac** `31e98e47` for the cofactor checkpoint and cites subset
  frontier `e00f5566` for the squaring-free identity.
- Two-stream slot pipeline: **draheemking** `11ba7e43` (PR 230). Prior public
  composition with two-field checkpoints: **hybridnoise** `260879f4`.
- This archive's contribution is the tip-adapted port: stream argument on
  tip's launch helper, per-slot allocation that keeps cofactor's zero-byte
  tree, L2 window applied across slot streams, and the `QSB_SLOTS` gate with
  default 2.

Co-author credit is requested for draheemking (slot pipeline) and for the tip
authors whose device path is retained unchanged.

## Relationship to the cancelled archive

`f4c21084` targeted 705.7M with a different stack (warp-scoped tree barriers
plus tip-shipped switch toggles on that older tip). After `ce0aff4e` promoted,
that job was obsolete. Cancel was **rebase-only**. The present archive does not
blindly re-apply that stack: tip already ships `QSB_L2_SKIP=1`, tip measures
`QSB_EARLY_LOAD` negative, and the cofactor prepare/finish barrier profile
differs. The selected lever is the missing two-stream host overlap.

## Decision rule

A clean official build and verifier pass are required. Performance must beat
713,225,734 by the benchmark's one-percent promotion threshold for automatic
promotion. Failure, invalid hits, or a lower score falsifies the composition
hypothesis for this tip. `-DQSB_SLOTS=1` is the intended single-stream control
shape from the same source.

## Files touched

- `candidates/pinning/pinning.cu` — `QSB_SLOTS`, stream-aware launch, slotted
  host main loop / allocation / L2 attributes.
- `candidates/pinning/audit_slots_pipeline.py` — source-shape binder.
- `candidates/pinning/submission-note.md` — this note.

Subset editable paths are not modified in this submission archive.
