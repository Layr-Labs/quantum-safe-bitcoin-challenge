# Mixed64 control for the next subset architecture

2026-09-16: full isolated candidate, not submitted or GPU measured.
Source fingerprint:
`3205a7177cb0a51fcb269b5ea1f06daedc7ef0b6c7000029095ad86c49d1bb0b`.

Public [PR62](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/62)
has now been promoted at **477182283 verified candidates/s**, commit
`106a6826ecf8998e684169f46e8de9dce1cf3f0f`. Our PR60 scored451135044.
PR62's fifteen-window64MiB geometry is therefore an evidence-backed direction
to include before choosing the next large architecture. Its score measures
its monolithic executable, not this new combination.

This control applies widths[18,17x14] to our existing corrected external-inversion,
direct-recovery and rolling-prefetch framework. It retains the bounded builder,
using256-entry low and1024-entry high ladders with one batch normalization per
window. Table entries and bases are runtime-dependent. It does not copy the
promoted header over our repaired field arithmetic. The chain costs95M+28S,
between PR60's102M+30S and the wide prototype's60M+18S.

`prepare.py` verifies its frozen wide-prefetch donor and refuses overwrite.
`provenance.json` records identities and attribution. Existing GPL notices and
the PR60/PR74 contribution lineage remain. The geometry direction is credited
to welttowelt's promoted PR62; no additional unpromoted code was imported here.

## Source-bound evidence

| Check | Result |
| --- | --- |
| Recoder | 12,769 cases |
| OpenSSL curve/recovery | 414 chains, 822 recovered keys |
| Actual interleaved loader | 570 cases |
| Next-load prefetch addresses | 5,382 comparisons |
| Builder | 101,491 decompositions, 4,473 affine entries over3 runtime bases |
| Batched host ladders | 10,097 points per runtime base |
| Native production | CUDA12.8.93 sm89 and default flags pass |
| Native audit | CUDA12.8.93 sm89 passes |
| Prepare resources | 128 registers,24KiB shared, zero spills |
| Finish resources | 80 registers,24KiB shared, zero spills |

CPU tests execute source-derived expressions with OpenSSL field/table oracles;
they do not execute CUDA arithmetic or collectives. Native compilation occurs
in a local ARM Linux VM on this Mac, not on an NVIDIA GPU. Static zero-spill
results do not establish latency or performance. Startup remains inside the
ordinary program and must be included when evaluating total performance.

```sh
python3 -B candidates/subset/research/wide_windows/check_wide.py --source candidates/subset/research/mixed_windows/candidate --mixed --report candidates/subset/research/mixed_windows/cpu-results.json
python3 -B candidates/subset/research/wide_windows/check_builder.py --source candidates/subset/research/mixed_windows/candidate --mixed --report candidates/subset/research/mixed_windows/builder-results.json
```

The next integration should compare this64MiB method with the16GiB method on
real candidate batches and preserve the corrected optional-allocation fallback
and memory cleanup from `../adaptive_tables/`. Keep the submitted/promoted PR60
control intact. Do not submit a32MiB-versus16GiB comparison while ignoring the
new measured64MiB alternative. The public PR77 and PR68 compositions still use
per-CTA inversion; their smaller supporting changes require compatibility
review and are not automatically additive speedups.
