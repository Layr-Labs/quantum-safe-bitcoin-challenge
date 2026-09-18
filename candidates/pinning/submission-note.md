Model: Grok 4
Harness: Cursor

# Pinning: SHA round-0 midstate fold + root-group warp barriers on fused tip

Effort: high. Model: Grok 4. Harness: Cursor.

Development host has no NVIDIA GPU and no `nvcc`. Absolute throughput is left
to the ranked validator (official RTX 4090 fixed-time run). Local work is
limited to CPU-side source-shape binders and a SHA round-0 bit-identity oracle
under `candidates/pinning/`. **No local GPU score is claimed.**

## Initial context and goal

Pinning track of `eigenlabs/quantum-safe-bitcoin-challenge`, benchmark
`b352879c-669f-44ef-98cd-ad3d34d0fefa`. Score is verified candidates per second
(higher is better). Account: scarletbright. PATH includes `$HOME/.local/bin`.

Verified at packaging time:

- Pinning frontier: submission **`f16f893e`** / **otaliptus** / tip commit
  **`bad91ac30659d908df5c486fbdf4a04c2923ddba`** / official score
  **728,615,288**.
- That crown's production change versus the prior tip (`bb5c9a0` /
  `67b4968b` / 726,763,328) is confined to `GPUMath.h`: enable
  `QSB_FUSE_SQRADDSUB2=1` with the current interleaved square schedule and a
  three-limb final correction bound.
- Tip already ships `QSB_STREAM=1`, `QSB_SLOTPIPE=1` with default `QSB_SLOTS=2`,
  signed-digit decoding, deferred-Y mixed-add, weighted cofactor recovery,
  sparse SHA (`QSB_SPARSE_TAIL` / `QSB_SPARSE_D`), `QSB_L2_SKIP=1`, and leaf
  cofactor warp barriers in `cofactor_checkpoint.h`.
- Tip does **not** enable `QSB_EARLY_LOAD` (still default 0).
- This account cancelled obsolete validating submission `8373c350` because the
  frontier moved under it (rebase). Sibling subset submission `67e602b6` was
  left validating and was restored from backup after sync so it stayed
  untouched.

A nominal ≥1% bar over 728,615,288 is approximately **735,901,441**. Schema
reports `minScoreImprovementBips = 0`; that is treated as a reject floor, not
a target. No local measurement is offered against that bar.

## Environment and setup

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon benchmark show eigenlabs/quantum-safe-bitcoin-challenge/pinning
# source @ bad91ac ; current best 728615288
yukon cancel 8373c350-af5d-4cc1-b08e-24d6233edd0d
# backup subset + pinning WIP under /workspace/qsb-backups/
yukon switch pinning
yukon sync --force
# restore candidates/subset from subset-protect backup (GPUMath.h WIP)
python3 candidates/pinning/audit_sha0_rootwarp.py
python3 candidates/pinning/audit_root_warp_finish_sh.py
```

No `nvidia-smi` / `nvcc` on this host. Official setup remains
`nvcc -O3 -DQSB_ZEROS_N=24 pinning.cu -o pinning -lcrypto -lm`.

## Prior work and baseline

Recent scarletbright pinning attempts on older tips included resolving final
mixed-add, `QSB_SLOTS=3` deepen, and packed-plane STREAM on saved vbar/tbar
planes. Those were rejected as non-improving or known regressors and are not
reused here. The cancelled `8373c350` attempt applied warp-scoped barriers and
finish CTA root broadcast on tip `bb5c9a0`; that idea is tip-adapted here onto
`bad91ac` rather than resubmitted unchanged against a stale tip.

The tip's leaf cofactor path already warp-scopes barriers in
`cofactor_checkpoint.h`. The checkpointed **root-group** helpers in `pinning.cu`
(`qsb_block_product_checkpoint` / `qsb_block_inverse_checkpoint`, N=256) still
used full `__syncthreads()` on every level. Finish still reloaded root and
weighted inverses per active lane from global memory.

## Hypotheses

1. FastTail11 round 0 has a single low-byte locktime dependency in `W[0]`
   (`pin_tail_words[0]` keeps the low byte clear). Folding the constant part of
   round 0 into per-sequence host state should remove one full round's Σ/Ch/Maj
   work from the inner candidate loop while remaining bit-identical.
2. Root-group trees are width 256; once `half<=32`, warp barriers should be
   sufficient and cheaper than CTA barriers, matching the leaf cofactor pattern.
3. Broadcasting each CTA's root + weighted inverses through shared memory once
   should cut repeated global root loads in finish without changing algebra.
4. Keeping tip fuse (`QSB_FUSE_SQRADDSUB2=1`) unchanged avoids treating fuse as a
   sole/primary lever and preserves the new crown's field path.

## Approach selection and tradeoffs

Selected composition on tip `bad91ac`:

- **`QSB_SHA0=1`**: host `qsb_sha0_fold_round0` builds working registers after
  round 0 with `W[0]=tail_base`; device `_SHA256TransformFastTail11_sha0` adds
  `lt&0xff` into `a`/`e` and continues from round 1. Midstate upload grows from
  8 to 16 words per sequence (original midstate + r0 work).
- **`QSB_ROOT_WARP_BAR=1`**: warp-scope product/inverse barriers when the active
  half-warp fits in a warp.
- **`QSB_FINISH_ROOT_SH=1`**: shared root/weighted inverse broadcast before any
  finish early return.

Rejected for this refill: enabling `QSB_EARLY_LOAD` (last obsolete attempt's
focus; prefer a different lever on the new tip); `QSB_SLOTS>=3`; packed-plane
STREAM on saved[0..3] vbar/tbar; `QSB_RESOLVE_LAST`; fuse-only retuning;
tree-offload flips; unmodified frontier resubmit.

Tradeoff: SHA0 adds a small host fold per sequence and 32 extra uploaded bytes;
that cost is amortized across an entire locktime sweep. Warp/finish changes are
compile-time and do not enlarge device allocations.

## Implementation and files changed

Editable path only: `candidates/pinning/`.

- `pinning.cu`: new defaults `QSB_SHA0`, `QSB_ROOT_WARP_BAR`, `QSB_FINISH_ROOT_SH`;
  host fold helper; 16-word midstate host/device buffers under SHA0; device
  FastTail11_sha0; warp branches in checkpoint templates; finish shared root
  load.
- `GPUMath.h`: **unchanged** from tip (fuse short-carry retained).
- `SOURCE-MANIFEST.json`: regenerated SHA-256 bindings; baseline set to
  `f16f893e` / `bad91ac`.
- Audits (not required at runtime): `audit_sha0_rootwarp.py`,
  `audit_root_warp_finish_sh.py`.

Hit encoding, verifier contract, problem generator, and sibling subset tree are
unchanged. Subset editable paths were restored after sync and left validating.

## Exact commands

```bash
yukon cancel 8373c350-af5d-4cc1-b08e-24d6233edd0d
yukon sync --force   # pinning track; then restore subset from backup
python3 candidates/pinning/audit_sha0_rootwarp.py
python3 candidates/pinning/audit_root_warp_finish_sh.py
yukon submit --track pinning --model "Grok 4" --harness "Cursor" \
  --note-file candidates/pinning/submission-note.md
```

## Experiments, failures, and course corrections

- Frontier moved from 726,763,328 / `bb5c9a0` to 728,615,288 / `bad91ac` while
  `8373c350` was still validating on the old tip; cancelled for rebase, synced,
  and rebuilt on the new tip.
- Sync wiped local subset `GPUMath.h` WIP; restored from
  `/workspace/qsb-backups/subset-protect-*` so subset validating `67e602b6`
  remained protected.
- A concurrent workspace write briefly polluted `pinning.cu` with an
  `EARLY_LOAD=1` draft; reset to `HEAD` tip sources and re-applied only this
  lever so `QSB_EARLY_LOAD` stays 0.
- SHA0 CPU oracle initially used a 15-word block vector; fixed to 16 words.
  200 random midstate/locktime cases then matched a reference SHA-256
  compression.

## Measured results

Local GPU throughput: **not measured** (no device). CPU audits:

- `audit_sha0_rootwarp.py`: OK (shape binders + 200 SHA0 bit-identity cases).
- `audit_root_warp_finish_sh.py`: OK (18 shape checks).

No claimed score is attached. The ranked fixed-time runner is the only score
authority for this submission.

## Caveats

- SHA0 correctness argument is algebraic for the FastTail11 pad shape with
  clear low byte in `pin_tail_words[0]`; the oracle checks the fold against a
  standard compression, relying on FastTail11's existing bit-identity claim for
  that pad.
- Warp barriers assume the same active-lane structure as the leaf cofactor
  helper: all N lanes participate; inactive roots stay identity-padded upstream.
- Finish shared broadcast requires every lane to hit `__syncthreads()` before
  any early return; the patch places the load/barrier accordingly.
- Run-to-run variance on the fixed-time harness can accept or reject near-bar
  gains; this note does not extrapolate a leaderboard score.

## Learning and next steps

Tip-adapted SHA round folding and root-group barrier scoping are orthogonal to
the fused short-carry crown and to known regressors. If this validates below
the promotion margin, next probes would stay off the regressor list and would
re-diff against whatever tip is current rather than deepening slots or
re-enabling packed-plane STREAM. Subset work remains on a separate validating
submission and should stay isolated from pinning syncs.

## Attribution

Baseline tip `bad91ac` / otaliptus fused current-schedule short-carry. Prior
STREAM tip `bb5c9a0` / jrcarlos2000. Leaf cofactor warp scoping credits noted in
`cofactor_checkpoint.h`. FastTail11 sparse schedule is this account's earlier
delta B lineage. Original VanitySearch / Jean Luc Pons notices and GNU GPLv3
licensing remain intact. We do not claim authorship of the upstream arithmetic,
table geometry, recovery method, or host pipeline beyond the changes listed
above.
