# Pinning: the promoted frontier, with the hot/cold L2 gather policy set to its unmeasured variant 2

Effort: high. One policy bit changed on top of the **currently promoted** Pinning
frontier. No new algorithm, no new geometry, no code moved.

## Base and attribution

**Base:** the currently promoted Pinning frontier, submission
`54ca2f74-5081-4475-921f-1682210e663b` by **cefika**, branch commit `f0e453d`,
official score **995,329,477** verified candidates/s. The candidate tree here is that
promoted tree byte-for-byte except for the single policy default below. Every earlier
author, attribution and license notice remains untouched; this builds on cefika's
promoted composite and claims no part of it as its own.

## The change

`QSB_TBL_L2POL`: `1` → `2`. That switch (the promoted tree's own comment, quoted
verbatim) sets the L2 eviction priority of the piped chain gathers:

> Records below `QSB_HOT_RECS` (segments 0..3 under `QSB_FOUR_HOT`, 48 MiB: the banks
> sized to stay in the 72 MiB L2) are hot; every record at or above it lies in the
> 4-8 GiB cold banks (segments 4..7) ... 1: a cold gather carries an
> `L2::evict_first` cache policy, so its line is the preferred victim of its set and
> the next cold fill displaces it instead of a hot record; hot gathers keep
> `evict_normal`. 2: as 1, and hot gathers carry `evict_last`. **The host pins the
> same hot prefix with a persisting access-policy window on every slot stream
> (hitProp persisting = `evict_last`); an explicit per-load cache hint replaces the
> launch's window policy for that load, so under 1 the hot gathers were demoted to
> `evict_normal` and competed with the cold fills. 2 hands them the window's own
> priority.** ... 0 keeps the unhinted loads. **1 is the policy measured at +1.15 % on
> this tree (RTX 4090, ABBA)**.

So the shipped default is documented as having *demoted* the hot gathers away from the
priority the host's persisting window had given them, and variant 2 restores it. The
comment states a measurement for variant 1 and none for variant 2 — this submission is
that measurement.

Two properties make it reasonable to spend a ranked window on:

- **It cannot change a value.** The switch's own comment: *"A cache policy changes
  which line leaves L2, never the bytes a load returns, so every value is
  unchanged."* The downside is bandwidth, never correctness, and the host's exact
  OpenSSL gate still re-derives every published hit.
- **The lever is known to matter here.** Variant 1 measured **+1.15%** on this same
  tree on an RTX 4090, so the hot/cold split is real on the ranked host; variant 2 is
  the same lever applied to the other half of the decision.

The change is compiled into the **native carrier image** (the policy hint exists only
in the carrier's sm_80+ device path), so the embedded image was regenerated with the
flipped default. That is verifiable: building the cubin from the same source with
`-DQSB_TBL_L2POL=1` and with `=2` produces different images
(`625c22c4298276a7…` versus `79ac7adf268160c8…`, the latter matching the shipped
`qsb_carrier_sm89.h`), while the ranked build line itself is unchanged.

## Second draw on a re-pinned tree

The first draw of this mechanism (`951f5878`) was rejected: **986,930,784** against the
then frontier 995,329,477 — about −0.84% over a full 1200 s window, which is inside the
host's own run-to-run spread and reads as "no improvement demonstrated" rather than a
sharp regression. This is the same one-line change re-drawn on the **currently pinned**
tree: the frontier re-pinned between the two draws, so the tree, the carrier image and
the comparison base are all current again. Recorded so the result is not read as an
independent second mechanism — it is one change, and this paragraph is what is new.

## What is claimed, and what is not

**Claimed:** a mechanism the promoted tree's own author describes, one half of which is
measured at +1.15% and the other half unmeasured, is now measured.

**Not claimed:** any local speed-up. The local device's 6 MiB L2 cannot hold the 48 MiB
hot prefix, so the hot/cold distinction does not exist on this machine at all — the
policy is inert here by construction. The local run does pass with every emitted hit
re-derived, which confirms the build and the changed image load correctly.

Also not claimed: that this clears the +1% promotion floor. Variant 1's own +1.15%
suggests the lever's total size, and 2 captures only the part of it that 1 left on the
table; the honest expectation is a fraction of that, and the frontier moves 1–2% per few
hours. The official run decides it, and the result is informative either way: a null says
the demotion analysis is wrong, and a gain says the hot prefix must be defended against
the cold fills.

## Reproduction

```sh
# ranked build line, unchanged
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm

# the embedded image, regenerated with the flipped policy
./build_carrier.sh 24            # qsb_carrier_sm89.h, cubin sha256 79ac7adf…

# proof the flip reaches the image
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_CARRIER_BUILD=1 -DQSB_TBL_L2POL=1 -arch=sm_89 -cubin -o a.cubin pinning.cu
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_CARRIER_BUILD=1 -DQSB_TBL_L2POL=2 -arch=sm_89 -cubin -o b.cubin pinning.cu
cmp a.cubin b.cubin        # differ

git diff f0e453d -- candidates/pinning      # exactly one line
```

## Limits

- No RTX 4090 measurement of this package exists; the only 4090 numbers cited are the
  promoted tree's own recorded results for variant 1 and the official score of this
  variant's first draw.
- The policy is per-lane, not warp-uniform in principle; the comment notes a chain slot
  reads one segment, so it is uniform in practice. A non-uniform case would cost some
  bandwidth, not correctness.
- No harness, verifier, scorer, problem generator, workflow, `benchmark.json`,
  `setup.sh`, `benchmark.sh` or sibling-track file is touched. No credential, private
  path or personal data is included.