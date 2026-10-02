# Narrow MAD fusion in the current even/odd product

Decision: not selected. This experiment tests a direct carry-fusion proposal
without changing the limb representation or product decomposition.

Fifty-six `mul.wide` plus64-bit-add pairs become narrow low/high MAD pairs.
Low-word addition consumes the old carry when required; the high-word MAD adds
the high product plus its addend and the low-word carry. Terminal flag liveness
is checked because the low instruction changes a flag that the old terminal
64-bit instruction preserved. The next flag consumer must begin a fresh chain.
The repaired reduction, host fallback, square and canonicalization are intact.

The [official PTX madc semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-madc)
extract the high product **before** adding the third operand and carry.
Gemini's review claimed the reverse, incorrectly declaring this valid pattern
invalid. The semantic model now contains an explicit witness for that rule.
Grok timed out without a final response. `external-review.json` records the
review limitations and corrections; neither model supplies performance proof.

Actual generated PTX passes20,961full-product/field-residue CPU semantic cases,
including near-prime and carry boundaries. Mutations that drop MAD carry-in or
the final overflow flag fail. This does not establish GPU execution correctness.

CUDA12.8.93 sm89 and defaultflags full builds pass. The unchanged source hashes
match the exact pending PR74 control report before resource comparison.

| Ranked kernel | Control/new registers | Control/new static non-NOP instructions | New spill stores/loads |
| --- | ---: | ---: | ---: |
| Prepare |122/128|7817/7819|0/0bytes|
| Finish |78/80|4598/4601|8/8bytes|
| Fused |126/128|13607/13605|0/0bytes|

There is no meaningful static instruction reduction and finish acquires a
spill. That does not prove equal GPU speed or exclude every scheduling variant,
but provides no basis for expecting the substantial improvement sought here.
The decision rests on actual codegen, not Gemini's unsupported optimality or
instruction-latency assertions. No production edit or new submission follows.

Reproduce from the benchmark directory with the documented compiler VM:

```sh
python3 -B candidates/pinning/research/narrow_mac/check.py
python3 -B candidates/pinning/research/compile_local.py --source candidates/pinning/research/narrow_mac/candidate --default-build --report /tmp/narrow-mac-native.json
```

The compiler VM was stopped after testing. The generated experiment is
immutable; `prepare.py` refuses an existing destination. Check reports bind the
exact header, PTX, semantic model and checker. SASS counts are static and are
not used as a substitute for GPU timing or a universal arithmetic lower bound.
