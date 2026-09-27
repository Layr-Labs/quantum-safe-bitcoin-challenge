# Carry-free product columns: radix29

This is an isolated product-schedule experiment on exact pending PR74. It is
not selected: semantic checks pass, but full-kernel native code grows sharply.
The source-matched control is `wide_windows/production-native-results.json`.

Unlike the previous magnitude-lazy proposal, each input is fully unpacked into
eight29-bit limbs and one24-bit limb before multiplication. Thus the raw
product-column bound, including incoming carry, is at most61bits. This permits
81`mad.wide.u32` accumulations without an explicit carry after every product.
Each column is normalized to radix29, the full512-bit product is repacked to
sixteen32-bit limbs, and the exact existing corrected reduction is retained.
Host fallback, specialized square and canonicalization are unchanged.

This removes many per-product carry operations while adding17raw products,
packing, shifts and masks. It is a real tradeoff; product count alone neither
establishes nor disproves a speedup. The first bounded screen deliberately
keeps existing field interfaces, making conversion overhead explicit.

`check.py` interprets the actual generated PTX against Python integer products
and residues for23,025cases. It includes radix29/radix32 boundaries, near-prime
witnesses, full-width random inputs and rejected carry/repack mutations. This
is CPU semantics, not GPU execution or an inline-assembly runtime audit.

CUDA12.8.93 production builds forsm89 and defaultflags pass. Exact source
hashes in both control and experiment reports were matched before comparison.

| Ranked kernel | Control/new registers | Control/new static non-NOP instructions | New spills |
| --- | ---: | ---: | --- |
| Prepare |122/128|7817/11661|0|
| Finish |78/80|4598/5882|0|
| Fused |126/128|13607/18694|0|

This schedule has no evidence supporting a substantial gain. Native counts are
static, not executed work or a measured slowdown. A representation kept across
point operations or a different fused reduction could behave differently; it
would need a new concrete cost case. Do not rerun this same conversion-heavy
schedule under a new name.

Reproduce from the benchmark directory (start the documented local compiler
VM before compilation and stop it afterward):

```sh
python3 -B candidates/pinning/research/radix29/check.py
python3 -B candidates/pinning/research/compile_local.py --source candidates/pinning/research/radix29/candidate --default-build --report /tmp/radix29-native.json
```

`prepare.py` creates the immutable experiment and refuses to overwrite it.
`provenance.json` records source and exact column bounds; `ptx-results.json`
binds the field source, semantic model and checker. No production code changed.
