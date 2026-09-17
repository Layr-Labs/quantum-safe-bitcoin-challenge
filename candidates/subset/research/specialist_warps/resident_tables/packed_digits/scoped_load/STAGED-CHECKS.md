# Resident32 specialist staged checks

Production source is the root `candidates/subset` closure, fingerprint `cd3d11f24e6fd42459eed3ba3a431c7f952de506cb1d22dfddc9b291e4c98667`.
Run from the benchmark root. Place generated diagnostics outside the editable directory before packaging. The scripts require Python, a native C++ compiler and OpenSSL; they do not execute CUDA kernels.

```sh
OUT=$(mktemp -d)
python3 -B candidates/subset/research/specialist_warps/resident_tables/small32/check_policy.py --source candidates/subset --report "$OUT/policy.json"
python3 -B candidates/subset/research/specialist_warps/resident_tables/packed_digits/check_digits.py --source candidates/subset --output "$OUT/digits"
python3 -B candidates/subset/research/specialist_warps/resident_tables/packed_digits/scoped_load/check_field.py --source candidates/subset --output "$OUT/field"
python3 -B candidates/subset/research/specialist_warps/resident_tables/packed_digits/scoped_load/check_control.py --source candidates/subset --output "$OUT/control.json" --debug-output "$OUT/control-debug"
python3 -B candidates/subset/research/specialist_warps/resident_tables/small32/check_builder.py --source candidates/subset --output "$OUT/builder"
python3 -B candidates/subset/research/specialist_warps/two_six_ring/sha-check.py --source candidates/subset --report "$OUT/sha.json"
python3 -B candidates/subset/research/specialist_warps/two_six_ring/check_inverse.py --source candidates/subset --report "$OUT/inverse.json"
python3 -B candidates/subset/preflight.py --package-check
```

Policy, digit and field/curve checks were rerun from this stage. Complete control, builder, SHA, inverse and native evidence comes from the qualified source/component reports; package construction does not claim fresh execution of those expensive checks. Exact unchanged component bodies bind reused evidence to the production closure. No GPU performance is claimed.

The staged control checker adds explicit source/output/debug destinations. The staged digit checker pins this closure instead of the earlier packed-only closure. The staged inverse checker accepts an explicit source root. Numerical helper extraction, CPU shims, oracles and negative mutations are unchanged by these interface adaptations; preparation receipts record original and staged script hashes.

Historical candidate material is limited to the ranked donor closure, the resident32 control closure and small header fragments actually read by the adapters. They are oracle/provenance dependencies, not alternate production sources. Unrelated research, provider records, compiler directories, host debug logs and generated binaries are excluded.

Public report copies omit private build/workspace paths and retain `original_report_sha256`. Digests inside the original qualification record refer to those preserved original main reports; sanitized copies have different file digests. Native public copies are `public-native-results.json` and `public-audit-native-results.json` beside the scoped checker. The final submission note is added separately after review.
