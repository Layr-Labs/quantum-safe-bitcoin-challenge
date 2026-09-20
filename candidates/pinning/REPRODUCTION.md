# Reproduce the arithmetic and point checks

Run from `candidates/pinning`, using new output paths each time. These are
diagnostic fixtures for the public QSB benchmark. Performance qualification
uses the separate unchanged organizer problem generator and production wrapper.

CPU checks of the actual shipped PTX:

```sh
python3 -B audit_paired_upper_fold.py . NEW_PAIRED_PTX_PROOF.json
python3 -B audit_conditional_small_tail.py GPUMath.h NEW_SMALL_PTX_PROOF.json
python3 -B audit_paired_native.py NEW_PAIRED_INPUTS
python3 -B audit_small_fields.py NEW_SMALL_INPUTS --inputs-only
python3 -B audit_signed_points.py NEW_POINT_INPUTS --inputs-only
```

The paired fixture has 5,207 multiply input pairs and 16,989 recovery cases.
Their SHA256 digests are respectively
`3a3afd975c56b600a784824f6b66f739ab1ce34c295cb589d231be4187bf099e` and
`82782182f343f530ef575a03d1c2f08d43119548f65e558b1f0f67ba0b9ee6fd`.
The generator reproduces the exact original native inputs byte for byte.

Native multiplication and recovery checks require CUDA 12.8 and OpenSSL
development headers/libraries:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 field_core_audit.cu -o field_core_audit -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 sum_audit.cu -o sum_audit -lcrypto -lm
python3 -B audit_paired_native.py NEW_PAIRED_RESULTS --field-binary ./field_core_audit --recovery-binary ./sum_audit
nvcc -O3 audit_final_fold.cu -o audit_final_fold
python3 -B audit_final_fold.py ./audit_final_fold NEW_MULTIPLY_RESULTS
```

The direct field audit checks separate, left-aliased and right-aliased outputs.
The multiply/square fixture has 4,502 pairs, including 125 directed upper-carry
pairs. It checks five outputs per pair, including in-place operations. Recovery
checks the exact canonical coordinates and both parity bits against independent
integer formulas for both implementations. To verify previously saved outputs
without executing CUDA, use `--field-output PATH --recovery-output PATH` instead
of the two binary arguments.

Native small-field and point checks:

```sh
nvcc -O3 small_field_audit.cu -o small_field_audit
python3 -B audit_small_fields.py NEW_SMALL_RESULTS --binary ./small_field_audit
nvcc -O3 canonical_helper_audit.cu -o canonical_helper_audit
python3 -B audit_small_fields.py NEW_HELPER_RESULTS --binary ./canonical_helper_audit
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_AUDIT_SAFE_NEG=1 audit_signed_points.cu -o audit_points_guarded -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_AUDIT_SAFE_NEG=0 audit_signed_points.cu -o audit_points_full -lcrypto -lm
python3 -B audit_signed_points.py NEW_POINT_RESULTS --guarded ./audit_points_guarded --full ./audit_points_full
```

Small-field checks cover raw and canonical caller domains, all aliases, and
carry/borrow boundaries. Point checks use OpenSSL on two bases and both table
paths, including the finite-doubling witnesses in `FINAL-WINDOW-PROOF.md`.

On the shared research Pod, compilation and every CUDA execution above must
run inside `bash /workspace/shared/run-gpu pinning-b <command...>` with the
installed CUDA environment. The local coordination wrapper is unnecessary
on a separate unshared machine. None of these diagnostic checks is an official
score or a replacement for the full 1,200-second production qualification.
