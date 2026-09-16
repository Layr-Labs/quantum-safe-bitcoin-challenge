# Arithmetic research while the first subset submission validates

Updated 2026-09-16. Pending PR27 is preserved. The prepared PR36+PR40
production control remains fingerprint
`d0ccab7c66830429f8bc39dc619ad4b369883d0ea9c7929b64a569592c65512e`.
The variants below are isolated research, not submitted or GPU measured.

## Concrete inherited reduction defect

For `p=2^256-2^32-977` and **a=b=p-65537**, the actual `_ModMultCore` C++
host branch returns **130096**, while independent arithmetic gives
**4295098369**. Both inputs are canonical. The difference is exactly
`2^32+977`, proving a lost final `2^256` carry, not merely a noncanonical
representation of the right residue. In contrast, `(p-1)^2` returns `p+1`,
which is congruent to one but noncanonical. Keep those failure classes separate.

The second-fold host expression drops `t>>32` after the highest limb. The
corresponding PTX also omits the final carry capture. `square32.cuh` explicitly
removed the same third fold, so its host and device branches share this issue.
The separate inverse-tree `qsb_field_mul` already has the correction. The
header's claim that canonical inputs cannot expose it is disproved by the
counterexample; the defect is inherited from the promoted arithmetic.

Earlier point-chain tests replaced field primitives with OpenSSL. They correctly
tested chain algebra and source ordering, but could not catch this primitive
defect. This is why source-specific field checks now accompany those tests.
The targeted sample below is not an estimate of failure frequency on random
benchmark coordinates, nor evidence that the pending run has failed.

Run from the repository root:

```sh
python3 candidates/subset/research/check_field_core.py
# Optional: create a fresh, isolated nine-file source closure with both fixes.
python3 candidates/subset/research/check_field_core.py --output /tmp/qsb-corrected-copy
```

The script compiles the extracted host core and an isolated corrected version;
it never changes production. Its optional output also corrects the square PTX.
It captures the second-fold carry, adds that carry times `2^32+977`, then
canonicalizes with at most one subtraction. The third-fold bound is below
`2^256`, so there is no further carry to discard.

| Actual extracted host source | Calls | Wrong residue | Noncanonical |
| --- | ---: | ---: | ---: |
| Existing multiply | 60540 | 69 | 165 |
| Corrected multiply | 60540 | 0 | 0 |
| Existing square | 80032 | 8 | 12 |
| Corrected square | 80032 | 0 | 0 |

Multiplication uses 20180 input pairs in distinct-output and both input-alias
modes; square checks distinct and in-place output. The data includes explicit
near-p carries and full-width random inputs. The oracle is independent Python
integer arithmetic. The corrected output is canonical on every tested case.
Host square calls the host multiply; this does not execute the separate square
PTX body. No CUDA compilation, PTX execution, resource count or throughput
result exists for either correction.

Corrected isolated production/audit fingerprint:
`0b850fce55b78e24b9cac953e26215213782c68019487b41ee2ae16aeae3d349`.
Reports are in `field-core-results.json`. Any future arithmetic implementation
must retain the final carry. Before selecting this correction for submission,
review both device paths and revise the public note/fingerprint; do not upload
the frozen control as though this finding did not exist.

## Larger speed hypothesis: 48 word products

`karatsuba_model.py` implements a difference-based one-level split with explicit
32-bit limbs. For `H=2^128`, it computes `z0=a0*b0`, `z2=a1*b1`, and
`d=abs(a0-a1)*abs(b0-b1)`. The middle term is `z0+z2-sign*d`, then the
full product is `z0 + middle*H + z2*H^2`. This uses 48 32-bit word products
instead of 64 before reduction, with additional comparison, carry and merge work.

```sh
python3 candidates/subset/research/karatsuba_model.py
```

The model passed 21465 exact full-product and field-reduction comparisons,
covering all four difference signs, 1573 cases with a 257-bit middle term,
9 final reduction carries and partial columns requiring 66 bits. A uint64_t
column accumulator alone would therefore be insufficient. Results are in
`karatsuba-model-results.json`. This is a Python limb model, not a CUDA kernel.

The 25% reduction in product count does not establish a 25% faster field
multiply, let alone a full-kernel gain. Reconstruction and live-state scheduling
could offset it. Next work is a scheduled low-level prototype with explicit
carry semantics and a count of added work, retaining the specialized square.
Do not use native ARM instruction counts as a CUDA performance verdict.

The accompanying table-size screen shows 16/15/14/13 ordinary signed windows
need at least 32/64/144/352 MiB under its full-coordinate, balanced-width model.
An idealized seven-window-per-128-bit GLV layout shares a 72 MiB table and uses
14 lookups, but on-load endomorphism adds seven beta multiplications. Scalar
decomposition and signed-recode corrections remain unimplemented. These narrow
counts do not establish a large gain or prove alternatives impossible.

## Review corrections

Grok and Gemini reviewed the algebra and independently agreed that the concrete
carry counterexample is wrong modulo p. Their initial numeric GPU instruction,
register, latency and optimality assertions lacked evidence and were rejected.
No source established a universal lower bound for the deferred point formula,
an ARM-to-CUDA falsification threshold or a guaranteed prefetch gain. Historical
prefetch spill regressions should be considered before repeating that experiment.

Gemini's suggested PTX repair still omitted `.cc` on the highest second-fold
addition. The actual isolated repair uses `addc.cc.u32 z7` before capturing
carry. NVIDIA specifies that `addc` writes carry-out only when `.cc` is present:
[PTX addc semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-addc).
This reinforces the requirement to verify model suggestions against source and
instruction semantics rather than accept model agreement as validation.
