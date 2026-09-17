# Corrected PR77 fused grinder with guarded small48 tables

Prepared source fingerprint:
`b832d7ba4ec2b2554aa5279e46858a5163e83e3ad7e714e9b108551165c95f7b`.

The exact corrected PR77 control supplies the complete fused search kernel,
per-CTA inverse, hash frontend, field arithmetic, recovery and host search loop.
Its control fingerprint is
`e18e355f7d8c90452e6ad61a6cbf4be1a495b5bb7d0817a1e8d00a69d935c918`.
The isolated external-pipeline candidate `140133ddb7a0408750a5396ce20658a1b48591422b5974315c8b76110de3117c`
supplies the guarded fourteen-window table geometry, bounded builder and checked
optional L2 policy. Neither donor's score or performance is assigned to this port.

`prepare.py` verifies both complete source fingerprints and refuses to overwrite
an existing candidate. `prepared-source.json` records dependencies, preserved
function hashes, checked startup changes and source-derived resource costs.
The GPL license and original source notices are preserved.

The fixed-base chain loads windows 12 and 13 first, then windows 0 through 11.
Its normal path costs 88 multiplications and 26 squarings. The donor's final-add
guard is unchanged. Two calls use PR77's equivalent deferred-Y symbols:
`_PointAddXYZZ_mm_def` and `_PointAddXYZZ_def(..., true)`. The exact corrected
PR77 `GPUMath.h` and `square32.cuh` remain unchanged.

Search retains its original per-CTA inverse and allocates no external search
checkpoints. `builder_checkpoint.cuh` contains only the donor's checkpoint and
root-helper prefix, used during table construction. These helpers now resolve
`qsb_field_mul` and `qsb_block_inverse_tree` to corrected PR77, so the integrated
builder must be checked with these dependencies rather than relying solely on
the older donor report.

The mandatory table occupies 4,345,298,944 bytes. Construction uses 65 bounded
chunks, 48,497,152 bytes of temporary GPU buffers and 14,680,064 bytes of host
ladder arrays, plus OpenSSL point objects. Only 23,552 sampled table bytes are
read back; there is no full-table host allocation or OpenSSL fallback. Temporary
builder buffers are freed before search. The ranked stack request is established
at 4096 bytes per thread before table construction. Forty standalone startup
CUDA calls now check their results; existing checked calls remain checked.
Unsupported L2 policy retains the same table geometry with ordinary caching.

`check_projection.py` passes production and audit C++ type/syntax projections.
It removes CUDA launches and PTX and is not a CUDA build or correctness test.
Parent-owned checks cover the actual chain, exception witnesses, field
primitives, startup policy, integrated builder and native compilation.

Concrete unresolved performance risks are cold-table latency, cache residency,
fused-kernel register pressure and spills, driver compilation and table startup.
Startup counts against the official harness clock. There is no local GPU
execution, measured throughput or established improvement over PR77.

Production and pending submission source are not modified by this experiment.

## Parent validation,2026-09-17

Exact b832d7ba source now passes6144SHA256d/3840inverse/10262recode/8086unrank
cases,628pointchains/1242keys/477exception-helpercases and13107integratedbuilder
entries. Production and audit nativeCUDA12.8.93sm89/defaultbuilds pass. The
comparison and report bindings are in `native-comparison.json` and
`validation-summary.json`. These are CPU/reference/compiler evidence, not GPU
execution or throughput. The current c571025c submission remains preserved.

The distinct inverse-tree multiplier also passes2180actualPTXsemantic-model
andcompiledcanonical-tail cases, with a rejected stale-carry mutation; see
`tree-field-results.json`. This is separate from inherited hot-math evidence.
