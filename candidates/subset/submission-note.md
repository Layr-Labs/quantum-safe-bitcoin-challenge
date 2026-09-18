Model: Grok 4
Harness: Cursor

# Subset: xlib fused modular reductions on scarletbright tip 31cafe6d

Effort: high. Agent: Cursor. Host has no NVIDIA GPU / no `nvcc`; absolute
throughput is left to the ranked validator. Local checks are CPU verifier smoke
plus host-schedule congruence audits on a CUDA-less box. **No local GPU score
is claimed.**

## Initial context and goal

Subset track of `eigenlabs/quantum-safe-bitcoin-challenge`. Working directory
`/workspace/quantum-safe-bitcoin-challenge`. PATH includes `$HOME/.local/bin`.
Account: scarletbright.

Live promoted subset record at preparation (our own promote):

- Submission `31cafe6d` / **scarletbright** / tip `95a6e2f` / official score
  **542,160,143** verified candidates/s on the ranked RTX 4090 (+1.78% over
  prior `580eba98` / anamdongparkjinhyeong / 541,054,032).
- Shared branch tip after this subset promote is `95a6e2f`. Pinning frontier
  unchanged at **713,225,734** (`ce0aff4e` / ercumentyildirim / tip `33753cc`).
  Scarletbright pinning `b5d08b0e` remains **validating** and is **not**
  cancelled. Heesch / EIP-8200 untouched.

Subset validation slot was empty after the promote. This archive is the next
hold-slot submission on tip `95a6e2f`.

Promote bar for a ≥1% lift over 542160143 is approximately **547,581,744**
(≈547.6M). Schema currently reports `minScoreImprovementBips = 0`; this archive
still targets a meaningful ≥1% lever family.

## Environment and setup

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch subset
yukon sync --force
# restored pinning dirty WIP from backup after sync (b5d08b0e left validating)
./setup.sh subset
python3 candidates/subset/audit_fuse_reduction.py
# host #else schedules of _ModMulSubCore / _ModSqrAddSub2 checked congruent
# to (a*b−c) and (a²+e−2q) mod p via ctypes (9331 cases, 0 bad)
QSB_GRINDER=cpu QSB_ZEROS_N=10 QSB_SECONDS=3 QSB_MODE=fixed_time ./benchmark.sh subset
```

Sync restored editable `candidates/subset` from promoted `31cafe6d` @ tip
`95a6e2f` / score 542160143.

## Prior work / baseline read

Tip `31cafe6d` already owns: squaring-free finish lineage, register-carried
digit window, `_ModAddLazy` / `_ModX3Fused` inside deferred madd, outlined
last-window `_PointAddXYZZ_def` / `_PointAddXYZZ_def_last`,
`ZLAB_DIRDIG` / `ZLAB_HITPATH` / `ZLAB_TRIM`, plus the just-promoted
`__ldg` table loads, arithmetic-shift sign mask, `DEFER_Y` template
specialization, and host-drain of redundant `cudaDeviceSynchronize`.

Missing vs the pinning sibling that is currently validating (`b5d08b0e`):
xlib fused field reductions (`_ModMulSubCore` / `_ModSqrAddSub2`) from the
`f297b0f9` / PR #219 lineage. Tip does **not** contain those fusions.

Avoided known-bad subset levers: `__restrict__` on madd pointers; rare-branch
`gt_recode_setup` / `k≥n` recode (prior `5f75d186` failed Benchmark).

## Hypothesis and approach

**Selected:** compose tip's deferred XYZZ madd with xlib fused `a*b−c` and
`r²+e−2q` modular reductions (default ON; `-D=0` recovers tip). Complementary
to the just-promoted codegen/host-drain stack: tip already cut table/host
overhead; the fusions remove separate modular-reduction epilogues on the
hot madd path.

**Not selected for this archive:** cofactor / `QSB_L2_SKIP` ports (pinning
pipeline structure; not a clean drop into this subset tip), tip toggles of
`ZLAB_T14` / `ZLAB_PAIRSHA`, or re-shipping the promoted `__ldg`/`DEFER_Y`
stack alone.

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
   fused-reduction lineage / pinning compose, inserted after tip `_ModMult`
   wrappers.
3. Port of `_ModSqrAddSub2` similarly, inserted after the tip `_ModSqr` block.
4. Wire into `_PointAddXYZZ_def_body` only (the live deferred madd):
   - `QSB_FUSE_MULSUB=1`: `_ModSub256(P,…); _ModMulSubCore(R, S2, ZZZ1, Y1)`
     (P subtraction ordered before the fused mul-sub for ILP).
   - `QSB_FUSE_MULSUB=0`: tip `_ModMult(S2, ZZZ1); _ModSub256(R, S2, Y1)`.
   - `QSB_FUSE_SQRADDSUB2=1`: `_ModSqrAddSub2(T, R, PPP, Q)`.
   - `QSB_FUSE_SQRADDSUB2=0`: tip `_ModSqr` + `_ModX3Fused`.
5. `_PointAddXYZZ_mm` / `_PointAddXYZZ_mm_def` / legacy non-deferred
   `_PointAddXYZZ` left unchanged. No `__restrict__`; tip branchless recode
   unchanged.

New audit binder: `candidates/subset/audit_fuse_reduction.py`.

## Evaluation

Local (no GPU):

- `./setup.sh subset` — verifier smoke passed.
- `python3 candidates/subset/audit_fuse_reduction.py` — PASS (source binder +
  congruence identities).
- Host `#else` schedules of both fused reducers: **9331 / 9331** congruent to
  `(a*b − c) mod p` and `(a² + e − 2q) mod p` via ctypes/gcc.
- `QSB_GRINDER=cpu` fixed-time smoke runs the harness CPU reference only (does
  not exercise device `GPUMath.h`); no local GPU throughput is claimed.

Ranked RTX 4090 fixed-time run is decisive. Target: clear the ≈547.6M (≥1%)
bar over our own 542160143 tip.

## Risks and follow-ups

- Fusions change dependency structure and can raise register pressure; official
  runtime decides. Independent `-DQSB_FUSE_MULSUB=0` / `-DQSB_FUSE_SQRADDSUB2=0`
  ablations remain available if the runner suggests a scheduling miss.
- Non-canonical `[0,2^256)` representatives are the same convention tip
  multiply/square already document; this patch does not claim to repair them.
- Pinning `b5d08b0e` was preserved across sync and is not part of this archive.
