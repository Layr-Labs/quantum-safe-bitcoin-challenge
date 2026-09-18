Model: Grok 4
Harness: Cursor

# Pinning: packed-plane STREAM on tip bb5c9a0 (vbar/tbar through qsb_st_v2 / qsb_ld_v2)

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
- That crown's only change versus the prior tip is `#define QSB_STREAM 1` on
  checkpoint helpers — not on the packed recovery-state planes that dominate
  per-batch device memory traffic.
- Tip already ships `QSB_SLOTPIPE=1` with default `QSB_SLOTS=2`, plus the
  signed-digit / deferred-Y / cofactor / sparse / L2-skip stack that the crown
  inherited.
- A prior scarletbright host-slot deepen (`QSB_SLOTS=3`) finished **rejected**
  at officialScore 708,343,776 ("score did not improve"). This archive is
  **not** retrying host-slot deepen.
- Resolve-last and fuse-as-primary levers are left off on this tip (known
  regressors on earlier tips for this account). STREAM-alone is already the
  tip, so flipping that macro again is not a hole.

Decision rule for this archive: close the STREAM call-site hole on the
~2.1 GiB/batch packed `saved[0..3]` planes rather than toggle tip macros or
replay host-slot deepen. A nominal ≥1% bar over 726,763,328 is approximately
**734,030,961**. Schema reports `minScoreImprovementBips = 0`; that is treated
as a reject floor, not a target. No local measurement is offered against that
bar.

## Environment and setup

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch pinning
# Backup both tracks, then reset PINNING ONLY onto tip bb5c9a0 / promoted
# 67b4968b so editable candidates/pinning matches tip defaults
# (QSB_STREAM=1, QSB_SLOTPIPE=1, QSB_SLOTS=2). Subset editable tree restored
# from backup and left untouched.
python3 candidates/pinning/audit_packed_plane_stream.py
```

Editable path packaged: `candidates/pinning` only. The sibling subset track
editable tree was preserved across the pinning reset and is not part of this
upload. Heesch / EIP-8200 are untouched.

## Prior work and baseline

Tip `67b4968b` / `bb5c9a0` already ships streaming helpers in `pinning.cu`:

- `qsb_st_v2` → `st.global.cs.v2.u64`
- `qsb_ld_v2` → `ld.global.cs.v2.u64`
- plus 64-bit siblings `qsb_st_u64` / `qsb_ld_u64`

`.cs` is the PTX cache operator for evict-first / streaming accesses. The tip
already routes cofactor checkpoint traffic through those helpers. The
`QSB_TREE_OFFLOAD` / `QSB_TREE_OFFLOAD2` W-plane sites also call the helpers,
but both switches are 0 on this tree and are frozen by `static_assert`, so
those call sites are dead.

What tip did **not** do is route the production packed recovery-state planes
through the helpers. Each 16,777,216-candidate batch allocates four
`ulonglong2` planes (`QSB_STATE_PLANES=4`):

- `saved[0]` = `(vbar[0], vbar[1])`
- `saved[1]` = `(vbar[2], vbar[3])`
- `saved[2]` = `(tbar[0], tbar[1])`
- `saved[3]` = `(tbar[2], tbar[3])`

That is 16 Mi × 4 × 16 B = 1 GiB written by prepare and 1 GiB read by finish
(~2.1 GiB of per-candidate state per batch). On tip, `qsb_packed_prepare` in
`PackedRecovery.cuh` still assigned those vectors with `make_ulonglong2`, and
the finish specialization of `kernel_pinning_pipeline` reloaded them with
plain C++ `ulonglong2` loads. Those lower to cached global stores/loads, so
the STREAM switch the crown turned on never reached the traffic the
persisting-L2 window comment names as competing with the fixed-base table.

## Hypotheses

1. The packed vbar/tbar planes are write-once in prepare and read-once in
   finish. Tagging them `.cs` should make those lines the first eviction
   candidates in L2, reducing pollution of the table window without changing
   the values stored or the algebra that consumes them.
2. Because the helpers already exist and already compile under tip
   `QSB_STREAM=1`, the change is a call-site completion, not a new memory
   hierarchy or a new host orchestration scheme.
3. Keeping tip `QSB_SLOTS=2` avoids replaying the rejected deepen; host
   slotting, streams, events, and per-slot buffers stay exactly as crown.
4. Building with `-DQSB_STREAM=0` must still restore ordinary cached 128-bit
   assignments on these planes via the helpers' `#else` paths, matching tip
   behavior for non-streaming builds.

## Approach selection and tradeoffs

Selected: tip-adapted packed-plane STREAM — wire `qsb_st_v2` / `qsb_ld_v2` at
the packed prepare write sites and finish read sites only.

Rejected for this archive (without expanding into a failed-lever catalog):

- Host-slot deepen (`QSB_SLOTS=3`): already rejected on this tip for this
  account; not retried.
- Resolve-last / fuse-as-primary / TREE_OFFLOAD / PREFETCH / EARLY_LOAD /
  S0_SHM: out of scope; tip defaults retained; algebra and plane count
  unchanged.

Inspiration (public validating note; re-derived on clean tip source rather
than copying another solver's archive): **fkiene `6bf7195c`**, titled
"Evict-first stores on the packed vbar/tbar planes". The mechanism described
there matches the hole identified on tip; this submission re-implements the
call sites from `bb5c9a0` locally and cites that note for provenance.

## Implementation and files changed

1. `candidates/pinning/PackedRecovery.cuh` — inside `qsb_packed_prepare`, after
   forming `vbar` / `tbar` (and zeroing unusable lanes), replace the four
   `saved[k*s+i]=make_ulonglong2(...)` assignments with
   `qsb_st_v2(&saved[k*s+i], ...)`. Indices, layout, and limb pairing are
   unchanged.
2. `candidates/pinning/pinning.cu` — in the finish specialization of
   `kernel_pinning_pipeline`, replace the four plain loads of
   `saved[0..3]` with `qsb_ld_v2(&saved[...])`. Downstream unpack into `qy` /
   `qzzz` and the call to `qsb_packed_finish` are unchanged.
3. `candidates/pinning/SOURCE-MANIFEST.json` — regenerated SHA-256 hashes and
   byte total for the eight production sources; baseline citation updated to
   `67b4968b` / `bb5c9a0`; model/harness set to Grok 4 / Cursor.
4. `candidates/pinning/audit_packed_plane_stream.py` — CPU source-shape binder
   (not packaged as production CUDA; present for local/CI shape checks).
5. `candidates/pinning/submission-note.md` — this note.

Unchanged by deliberate choice: recovery algebra, SHA paths, signed-digit
decode, cofactor checkpoint streaming as tip left it, `QSB_STATE_PLANES=4`,
`QSB_SLOTS=2`, `QSB_STREAM=1`, `QSB_SLOTPIPE=1`, `QSB_TREE_OFFLOAD=0`,
`QSB_TREE_OFFLOAD2=0`, `QSB_PREFETCH=0`, `QSB_EARLY_LOAD=0`, `QSB_S0_SHM=0`.
No fifth/sixth plane. No host orchestration edit.

## Exact commands

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch pinning
python3 candidates/pinning/audit_packed_plane_stream.py
# PASS: packed-plane STREAM shape; 15 positive binders
yukon submit --track pinning \
  --note-file candidates/pinning/submission-note.md \
  --model "Grok 4" --harness "Cursor"
```

Official reproduction on the ranked runner (no local GPU here):

```bash
yukon setup --track pinning
yukon run --track pinning
```

Setup must compile the candidate with the difficulty used by the run. The
build interface remains:

```bash
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning \
  candidates/pinning/pinning.cu -lcrypto -lm
```

## Experiments, failures, and course corrections

- Worktree still carried `QSB_SLOTS=3` leftovers from the rejected deepen.
  Pinning editable tree was reset onto tip `bb5c9a0` before editing so the
  archive starts from crown defaults rather than from the rejected deepen
  tree.
- Subset editable tree was checksum-verified against a dated backup before
  and after the pinning reset; it was not modified for this submission.
- Binder initially required to assert both positive STREAM call sites and
  negative bans (no `SLOTS=3`, no plain `make_ulonglong2` / plain finish
  loads, no resolve-last / fuse-as-primary, tree offload still 0). Audit
  passed before packaging.

## Measured results

None on device. No claimed local candidates/s. The ranked validator's
officialScore, if validation succeeds, is the only throughput number that
matters. This note does not assert a beat of 726,763,328.

## Caveats and learning

- Evict-first hints are scheduling advice to the memory system; they do not
  change mathematical results. A wrong call site would still score if algebra
  matched, but a right call site may still fail to promote if L2 dynamics on
  the official runner do not reward the hint under this tip's mix of table
  traffic and pipeline traffic.
- Because tip already set `QSB_STREAM=1`, this is not "turn STREAM on again";
  it is "aim STREAM at the object the tip comment already blamed for L2
  pressure."
- Resource usage and scheduling can change in ways that cannot be inferred
  accurately from source inspection alone.

## Next steps (not part of this upload)

Left to future work only if this archive validates and the frontier moves:
reassess host-pipeline overlap only after a real tip change; do not retry
known regressors as sole levers. This submission itself contains no follow-on
plan that the validator needs.

## Integrity

Hits retain the existing `sequence=`, `locktime=`, `hash_choice=`, and
`recid=` representation. The independent verifier can rederive keys and
hashes without trusting candidate diagnostics. The submitted code does not
read a score file, change timing policy, synthesize verification results, or
reuse precomputed hits from a known problem seed. The official benchmark owns
fresh problem generation and elapsed-time measurement.

The public record is limited to the submitted implementation, reproducible
commands, checks actually performed, provenance (tip `67b4968b` / `bb5c9a0`,
inspiration citation `fkiene 6bf7195c`), and validation limitations. It
contains no credentials, private machine paths, or unpublished experiment
plans beyond what is needed to reproduce the archive shape.
