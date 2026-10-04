# Signed-Q5 CPU math proof

Copy this directory to `candidates/pinning/tests/signedq5` alongside the unchanged 39 production files. From the repository root:

```sh
python3 candidates/pinning/tests/signedq5/check_signed_q5_math.py
python3 candidates/pinning/tests/signedq5/check_signed_q5_math.py --receipt signed-q5-math-receipt.json
```

Python 3 with its standard library is sufficient. Do not use `python -O`. No compiler, CUDA installation, OpenSSL, network, checkout, or research directory is required. For another placement, pass `--source-root PATH` naming the production `candidates/pinning` directory. The optional receipt path must be outside that production directory; stdout also contains the complete receipt.

## Binding

`source-manifest.json` gives exact SHA-256 values for the 39 production files. The test reads those files; it never writes them. It also decodes the production carrier header and checks the complete signed-Q5 cubin image hash and byte count, without loading or executing the image. `baseline-to-candidate.patch` is the exact three-hunk pinning source change relative to commit `b59a947d5c4d0ac61d2b1136ffdd7b3362010434`. The script strictly reverses it in memory and checks the complete reconstructed baseline pinning source hash. There is no dependency on a separate baseline source tree.

## Checks

The deterministic generated inputs cover 15,105 distinct bounded magnitudes, both signs and both routes (60,420 rows); literal LOP3 truth tables; seed/sign and physical table-record indices; equality of intermediate prefix sums; 4,108 synthetic full128-bit endpoint cases; five inherited negative-borrow constructions; 1,024 CTA selectors plus the rare P6 boundary at65536; slot-two omission and initial tree overwrite; and the slot/phi model for both P schedules and conditional zero cases. The script fails on any assertion and reports actual counts and observed index ranges. See `MATH-REVIEW.md` for the algebra and inherited exceptional paths.

## Limits

This is a finite Python integer recoding and ideal slot/phi model. It does not execute CUDA/PTX or prove universal native field correctness, exact active compiler branches, performance, or canonical hit coverage. Existing decode, multiply/square carry, reduction and Y-offset cuts remain unchanged; different projective intermediates can expose different rare-cut behavior. Conditional zero-component schedules include source fallback semantics even though the default NZ-cut returns3. Synthetic out-of-bound residuals test route endpoint equality, not valid upstream GLV decomposition. Native qualification requires separate source/image-bound canonical output checks. No GPU or native build is part of this suite.
