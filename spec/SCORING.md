# Scoring & verification

## The score

**Primary — candidate throughput (M/s):** `verified_hits × 2ᴺ / 2 / elapsed / 1e6`
(candidates per second *estimated from verified hits*). The `/2` is the two
recids tried per candidate. Statistical noise of the estimate is ≈ `1/√K`
(`K` = verified hits), so `K` must be large: at `N=24` a 20-min 4090 run gives
`K≈22k` pinning / `9k` subset ⇒ ≈0.7% / 1.0%.

`elapsed` is **a clock the candidate cannot write to**, never the grinder's own.
With the `cpu`/`cmd:` grinders `run_benchmark.py` brackets the grinder process
itself. On the ranked runner (`bridge:` grinder) the kernel runs inside a
root-owned sandbox and `elapsed` is the wall time root measured around that
sandboxed command (`metrics.json` `wall_s`), which excludes GPU admission and
is not writable from inside the sandbox. Throughput is recomputed from that
measurement and the verified-hit count. The grinder's own `candidates` /
`throughput_Mps` are recorded (`self_reported`, `candidates_self_reported`) but
not ranked. Otherwise a submission could inflate its rate simply by
under-reporting its runtime, or by reporting a peak rate instead of a mean. A
grinder that claims to have run *longer* than the harness watched it run is
rejected outright — its accounting is untrustworthy.

Everything inside the grinder command is timed, including compilation. Build in
`setup.sh <track>` (which pre-builds that candidate kernel at the ranked `N`), not
inside the run.

This is the metric to optimize. Higher is better.

**Secondary — verified hits/second:** `verified_hits / elapsed`. Reported
alongside; useful as a sanity/telemetry number. It is *not* the primary score
because at 100–1,000 hits its Poisson variance is ~3–10%, not 0.5%.

Both `N` (leading-zero bits) and the target hit count are tunable
(`harness/config.json`). Pick `N` so a 15–20 min run yields ~100–1,000 verified
hits: expected hits ≈ `2 · candidates · 2⁻ᴺ` (two recids per candidate). Calibrate `N` on the GPU being
characterized once its throughput is known (see `baselines/`).

## Why the two benchmarks are separate

Pinning and subset-selection assemble their preimages differently — a 75-byte
varying tail vs. a ~1.7 KB section rebuilt per candidate — so a single score
would hide where an implementation wins or loses. Score them independently.

## Anti-cheat: independent hit verification

A run is **scored only if every reported hit independently re-derives**
(`harness/verify.py`). For each hit the verifier, from scratch:

1. rebuilds the preimage from the (synthetic) problem + the hit's candidate,
2. `z = SHA256d(preimage)`,
3. `Q = recover(r, s, z, recid)`, `h = SHA256(compress(Q))`,
4. asserts `leading_zero_bits(h) ≥ N`,
5. rejects duplicates after canonicalizing the candidate (32-bit masked
   `sequence`/`locktime`; sorted skip set; `recid` included).

Re-derivation of the hits that survive that check runs across all CPU cores of
the judge host (`os.cpu_count()`). The verified set and the failure list do not
depend on the worker count; `QSB_VERIFY_WORKERS` overrides it for diagnostics
only.

A kernel that fakes throughput cannot produce verifiable hits, so throughput is
only credited when backed by real work.

**Hit-sample sufficiency (hard gate).** The verified-hit sample must be large
enough for the hit-rate estimate to be meaningful. With `p = 2⁻ᴺ` and `K` the
number of verified hits, the relative variance is

```
Var = sqrt((1 − p) / K)
```

and the run is **rejected** unless `Var ≤ max_relative_variance`
(`harness/config.json`, default `0.1` ⇔ `K ≥ ~100`). Both `verify.py` and
`run_benchmark.py` enforce it and record `hit_relative_variance` in the artifact.

**Statistical consistency check.** The verifier also flags runs whose verified
hit count is grossly inconsistent with the claimed candidate count at difficulty
`N` (expected ≈ `2 · candidates · 2⁻ᴺ`, checked within a wide Poisson band). This
is a diagnostic of the kernel's own counter only — it never rejects a run,
because the ranked number no longer depends on `candidates`. `--strict` is
accepted but has no effect on the band.

## Anti-replay: a fresh instance per ranked run

A committed, publicly-seeded problem can be ground offline at leisure. The hits
are real and verify perfectly — so a submission could bank them, return them
instantly, and claim any throughput it likes. Independent hit verification does
not catch this, because the hits are not fake.

A ranked run therefore generates a **fresh instance from an unpredictable seed**
(`benchmark.sh`, or `run_benchmark.py --seed random`), chosen after the
submission is fixed. Precomputed hits do not verify against it, so the only way
to produce hits inside the timed interval is to grind.

`harness/verify.py` also binds a run to its instance: a run whose recorded
`problem_seed` differs from the instance being verified is rejected. The seed is
written by the harness, not claimed by the grinder.

The committed `problems/` instance (seed 0) remains the public example for the
quick start and for `candidates/`. A ranked instance is written to a scratch
directory and pointed at via `QSB_PROBLEM_DIR`.

## Reproducibility

The generator is deterministic in its seed, so a ranked run is fully
reproducible from the seed recorded in its artifact — which is what CI does
before re-verifying. Record the seed, `N`, mode, GPU, and both score numbers
with each result (see `baselines/TEMPLATE.json`).
