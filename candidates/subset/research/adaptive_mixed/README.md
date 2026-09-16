# Adaptive mixed-table subset candidate

The candidate combines the checked64MiB compact and16GiB wide branches with
real-batch runtime selection. Source fingerprint:
`6bfe0e61fbbcabce237877c95e238f0bd22914ed6ceade6baf97e058c5e51ac6`.

`provenance.json` identifies frozen donors. `validation-summary.json` binds
CPU/native results; `inherited-evidence.json` documents exact unchanged tested
primitives and pipeline bodies. `external-review.json` records completed and
incomplete model reviews with rejected claims. No NVIDIA GPU execution occurred.

Public mechanism, attribution, commands and limitations are in
`../../submission-adaptive-mixed.md`. The compact path is mandatory; optional
wide requires16GiB plus2GiBfree reserve, falls back on allocation exhaustion,
and is selected only for measured real GPU pipeline time2%lower than compact.
All comparison ranges/hits are retained. Startup and noisy timing remain risks.

Both prepare kernels compile at128registers/24KiBshared/zero spills; finish is
80registers/24KiB/zero spills. These counts are not runtime performance.

The corrected generalcheck_candidate.py accepts--source and validates actual
mixed-window geometry, enabling the same fullSHA/unrank/inverse regressions
before production replacement. Builderprojection fixes are test-only changes.
