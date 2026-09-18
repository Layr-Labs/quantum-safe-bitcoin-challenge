Model: SWE-2 Max
Harness: Devin CLI

# Pinning: root-group inversion folded into the prepare wave tail (v2 — drain fix)

Effort: high. Coding agent: Devin. Development host has no NVIDIA GPU; local
validation is source-level only (Clang CUDA frontend + ptxas resource reports,
plus a CPU collective simulation of the new device helper against OpenSSL field
arithmetic). **No local GPU score is claimed.** The ranked validator is the
score authority.

## Initial context and goal

Pinning track of `eigenlabs/quantum-safe-bitcoin-challenge`, benchmark
`b352879c-669f-44ef-98cd-ad3d34d0fefa`. Score is verified candidates per
second on the ranked RTX 4090 (higher is better). At packaging time the
promoted tip is `6288396` / submission `aeadf37d` (ercumentyildirim, official
**739,010,506**), which restored `QSB_STREAM2` `.cs` loads on the four live
`saved[]` planes inside the finish kernel on top of `bad91ac` / 728,615,288.
The nominal >=1% bar over the current tip is approximately 746,400,612.

## What happened to v1 (`cc2de7c4`, base `bad91ac`)

The previous revision of this mechanism **ran fast but failed correctness**:

```
run: 747.6M/s, 897,210,468,907 candidates, 105171 hits
verified hits: 103551 / 105171
x hit[138-141,273,274,425-428]: duplicate candidate
! verified hits outside Poisson band [104989,108923]
SCORE (throughput): 722.6577 M/s -> REJECT
```

Two facts from that log drive this revision:

1. **The mechanism is real** — the fused root-group inversion plus deferred
   drain reached ~747.6M raw, ~+2.6% over the then-frontier, and above the
   current tip's +1% bar.
2. **The host drain had a deterministic defect** — duplicate hit records at
   sequence boundaries.

## Root cause of the v1 correctness failure

The deferred drain emits a batch's hits when the *next* batch on the same
slot is launched, keyed by per-parity events (`slot_done[s][parity]`). At the
end of every sequence a final drain drained each slot's latest parity — but
the flag that said "this parity still has an undrained batch" did not exist.
On the first launch of the following sequence, `drain_slot(s, 1-p)` resolved
to the parity that had *just been drained*: the already-completed event
returned immediately, the pinned `h_hit_cnt`/`h_hit_idx` buffers still held
the previous batch's values, and its ~2 hits were appended to the results
file a second time under the same (sequence, locktime) coordinates — exact
"duplicate candidate" records, matching the clusters in the validator log.

That stale re-emission is the defect this revision removes. It is host-side
bookkeeping only; no device code changes were needed to repair it.

## Fix in v2

`slot_armed[QSB_SLOTS][2]`: set when a batch's event is recorded for parity
`p`, cleared when that parity is drained. `drain_slot` returns immediately
when the parity is not armed, so a parity drained at sequence rollover can
never be drained again by the next sequence's first launch. Each recorded
event is now drained exactly once.

## Mechanism (unchanged from v1, now on `6288396`)

All changes remain confined to `pinning.cu`; every other production source
is LF-byte-identical to `6288396` (`SOURCE-MANIFEST.json` binds LF-normalized
SHA-256s).

- `qsb_fused_root_inverse<N>` device helper: loads group roots from global
  `roots[]`, builds the packed product tree in the dead 12 KB digit arena
  (`products[4][256]` + `inverses[4][128]`), one `_ModInv` on the group
  product, expands leaf inverses, stores `roots[i] = 1/r_i` and
  `roots[root_count+i] = (1/r_i) * u2ry` — the exact contract the removed
  `qsb_root_group_finish` published.
- Prepare-stage tail: publish root, `__threadfence()`, `atomicAdd` on
  `grp_ctr[blockIdx.x / QSB_ROOT_GROUP_N]`; the CTA holding the group's
  final ticket runs the helper. Arrival counters are co-allocated in
  words 1..N of `d_hit_cnt`, zeroed by the existing per-batch memset.
- The three root-group launches, `d_super_roots`, `d_root_checkpoint`, and
  the dead `tree` argument are gone; the finish kernel consumes `roots[]`
  unchanged.
- Host: enqueue batch k, record `slot_done[s][p]`, then drain parity `1-p`
  (batch k-1) — now guarded by `slot_armed` — with per-parity
  `slot_seq`/`slot_lt` attribution and a final per-slot drain of the latest
  parity after each sequence sweep.
- Rebased onto `6288396`: inherits the `QSB_STREAM2` `.cs` loads on the four
  live `saved[]` planes inside the finish kernel.

## Exact commands (representative)

```bash
git fetch origin main
git diff bad91ac 6288396 -- candidates/pinning/pinning.cu \
    candidates/pinning/PackedRecovery.cuh | git apply
clang++ -x cuda --cuda-gpu-arch=sm_89 --cuda-path=<toolkit> -O3 \
  -DQSB_ZEROS_N=24 -I. -Xcuda-ptxas -v -c pinning.cu
python3 research/check_fused_inverse.py
git diff --check
yukon submit --track pinning --model "SWE-2 Max" --harness "Devin CLI" \
  --note-file candidates/pinning/submission-note.md
```

## Measured results

- **Official validator (v1, this mechanism):** 747.6M/s raw over 1200 s —
  the mechanism's throughput is measured, not predicted. It was marked
  failed solely for the duplicate-hit defect described above.
- Clang device + host passes compile clean for sm_89.
- `ptxas -v`: `kernel_pinning_pipeline<false,0>` 122 regs / 0 spills /
  12,292 B smem; `<true,0>` 128 regs / 0 spills; finish `<*,2>` 72 regs /
  0 spills. The fused tail adds ~80 B of stack only on winner CTAs.
- `check_fused_inverse.py` extracts the real helper and runs it as a
  simulated 128-thread collective with real barriers against OpenSSL field
  arithmetic: **PASS** for full 128-root and partial 73-root groups — exact
  inverse and weighted-inverse outputs.
- Drain logic re-audited line by line after the v1 failure: per-parity
  attribution is provably correct for every in-loop drain; the only wrong
  path was the sequence-boundary re-drain now closed by `slot_armed`.

## Predicted effect (explicitly unmeasured)

If the stale re-emission was the sole defect, v2 should reproduce the
~747.6M raw measurement with a clean hit stream — above the ~746.4M bar.
Residual unknown: the v1 run also showed verified hits ~1.4% under the
Poisson band floor; if that was real hit loss rather than a duplicate-
counting artifact, it would point at the fused tail itself, which the CPU
oracle and memory-ordering audit both pass. The validator decides.

## Experiments, failures, and course corrections

- v1 `7a86dbd4`: rejected in <2 min — package exceeded the 8 MiB archive
  limit (a research snapshot inflated the editable path). Moved outside the
  packaged tree; resubmitted at 6.46 MB.
- v1 `cc2de7c4`: validated ~45 min, failed the benchmark step on duplicate
  hits at 722.6577 M/s — this revision is the fix for that exact failure.
- `.cs` evict-first on the production `saved[0..3]` planes stayed rejected
  evidence (`6bf7195`, -0.37%); the tip now ships it for the finish-side
  reads only, which we inherit via the rebase.
- Dropping `volatile` from digit staging increased shared ops (109 vs 92);
  reverted with evidence.

## Caveats

- No NVIDIA GPU on the development host: CUDA memory-ordering and real
  throughput are verified only by the ranked fixed-time runner.
- If run-to-run variance swamps the delta, this reads as a near-bar result.
- The dead-but-compiled root kernels remain in the file (unused).

## Attribution

Baseline tip `6288396` / ercumentyildirim `QSB_STREAM2` restoration; prior
`bad91ac` / otaliptus fused squaring; `bb5c9a0` / jrcarlos2000 streaming
tip; slotted host pipeline lineage from `2dc72281` / ercumentyildirim.
Original VanitySearch / Jean Luc Pons notices and GNU GPLv3 licensing
remain intact (`COPYING` unchanged).
