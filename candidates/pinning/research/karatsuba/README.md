# One-level difference Karatsuba: not selected

This isolated experiment turns the existing 48-product Python hypothesis into
actual straight-line PTX and compiles the complete pinning candidate. Pending
PR74 and sibling subset PR86 are unchanged. No GPU execution or timing occurred.

The three128-bit products use4x32 even/odd accumulation. Branchless absolute
differences and signed reconstruction produce the middle257-bit term, which
is merged into the512-bit product before the exact corrected production
reduction tail. Raw word products fall64→48; nine wide reduction products and
one carry-fold low product remain. The host fallback and specialized square
are unchanged. This is not a claim that48 products means25% less total work.

`check.py` interprets the actual generated PTX against Python integer arithmetic
for21,465 full512-bit products and field reductions. It covers all four
difference-sign combinations, a257-bit middle, final-fold overflow, and mutation
regressions for the difference sign and stale final carry flag. The actual
field PTX must equal the generated artifact. Unknown instructions fail closed.
The model is not GPU instruction execution or a proof over all inputs.

Borrow semantics were checked against
[NVIDIA PTX subc](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-subc).
Model checks explicitly cover borrow propagation, preservation and overwrite.

Native CUDA12.8.93 sm89 and default-flag production builds pass on the local
ARM Linux compiler VM. `comparison.json` binds the baseline to every exact
submitted include hash; the earlier adaptive report has stale host-loop hashes
and was not used as the final control. Only the device multiply changes.

| Ranked kernel | Original registers | Karatsuba registers | Original static non-NOP instructions | Karatsuba static non-NOP instructions | New spill stores/loads |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prepare |122|128|7817|9221|4/8bytes|
| Finish |78|80|4598|5059|44/48bytes|
| Fused |126|128|13607|15415|12/12bytes|

The original kernels have zero register spill stores/loads. Static sizes are
not dynamic executed instruction counts. Even without assigning an exact
performance penalty, added instructions, registers and spills undermine the
expected substantial-gain case for this implementation. It is not selected
for production or submission. This does not prove every Karatsuba schedule
loses; revisit only with a materially better signed-merge/carry schedule or
relevant GPU evidence, not the same raw48-versus64 argument.

Independent background read after implementation:
[NVIDIA forum discussion of 256-bit multiply code generation](https://forums.developer.nvidia.com/t/squeezing-the-last-17-5-out-of-a-compute-bound-256-bit-modular-arithmetic-kernel-sm-89-82-5-sm-throughput/367788?page=2).
Its authors describe carry/recombination overhead and application-specific
measurements. Those reports are not measurements of this candidate, and its
general ILP assertions are not used as universal architectural facts. No forum
code was imported into this prototype.

Reproduce from the benchmark directory:

```sh
python3 -B candidates/pinning/research/karatsuba/check.py
python3 -B candidates/pinning/research/compile_local.py --source candidates/pinning/research/karatsuba/candidate --default-build --report /tmp/karatsuba-native.json
```

`prepare.py` creates the immutable experiment from the exact pending source and
refuses to overwrite an existing experiment. Source/report identities are in
`provenance.json`, `ptx-results.json`, `native-results.json` and `comparison.json`.
The compiler VM is stopped after these builds.
