# Asymmetric fourteen-window experiment

This isolated subset candidate tests mostly small fixed-base tables plus two
large windows. It is suitable for later comparison on both tracks; only subset
has been implemented. Production PR86 and pinning PR74 remain preserved.

Widths `[18,17,17,17,17,17,17,17,17,17,17,17,26,25]` span 256 bits. Twelve
tables occupy 52 MiB; two occupy 3 GiB. Total: 51,183,616 points, 3,275,751,424
bytes. There are fourteen point lookups and a normal deferred-Y chain costing
88 multiplications and 26 squarings, versus 95 and 28 for compact15. This is
an arithmetic count, not an estimated throughput gain.

The generator inherits `../startup_budget/candidate`, including checked
allocations and the early stack request. It changes the compact branch's
geometry, decoder and ladder radix, and disables the optional16GiB alternative
for this isolated experiment. Field, SHA, inverse and point-add formulas are
unchanged. Retained `compact_` names refer to the dispatch slot; they do not
mean this candidate allocates a64MiB table. The runtime comparison message is
unreachable with the optional wide pointer null. Unused inherited kernels
remain available in the compile prototype.

The8192×8192 builder ladders generate69,826 points per runtime base. The GPU
builder still processes bounded chunks and checks sampled entries. CPU audits
use sparse virtual table mappings; they do not allocate or construct a full
resident3GiB table on this Mac.

## Evidence

All reports bind source fingerprint
`230f651785e7bcd9affd8cb188e41287a057a06caa8d50749487c5f7a73dec0a`.

| Check | Result |
|---|---|
| Actual recoder |12,769 scalars; independent signed reconstruction |
| Curve chain and recovery |414 chains,822 recovered keys vs OpenSSL |
| Actual vector loader and prefetch addresses |532 loads,4,968 next-load address checks |
| Builder decode |101,494 independent mapping checks |
| Actual host ladders and affine builder |4,482 sampled points across3 runtime bases |
| Actual builder prepare/inverse/finish CPU projection |13,107 entries;3 bases, partial blocks, window transitions, canaries |
| Full native CUDA12.8.93 sm89 and default builds |Pass; no GPU execution |

| Ranked kernel | Compact15 control | Asymmetric14 |
|---|---:|---:|
| Prepare registers |128|128|
| Prepare shared memory |24KiB|24KiB|
| Prepare spill stores/loads |0/0|0/0|
| Prepare static non-NOP instructions |8,412|8,428|
| Finish registers / spills / non-NOP |80 /0 /4,685|80 /0 /4,685|

Static instruction size includes the rolled loop once. It does not count
dynamic loop repetitions or measure cycles. The geometry adds a little
control logic and removes one point-add iteration. Other search/root kernel
resource/opcode reports are unchanged; the generic fallback still has its
inherited spills. `comparison.json` records the complete differences.

## Reproduce

From the benchmark directory:

```sh
python3 -B candidates/subset/research/asymmetric_windows/prepare.py
python3 -B candidates/subset/research/wide_windows/check_wide.py --compact --widths 18,17,17,17,17,17,17,17,17,17,17,17,26,25 --source candidates/subset/research/asymmetric_windows/candidate --report candidates/subset/research/asymmetric_windows/curve-results.json
python3 -B candidates/subset/research/wide_windows/check_builder.py --compact --widths 18,17,17,17,17,17,17,17,17,17,17,17,26,25 --low-bits 13 --source candidates/subset/research/asymmetric_windows/candidate --report candidates/subset/research/asymmetric_windows/builder-results.json
python3 -B candidates/subset/research/pr86_failure/check_builder_pipeline.py --compact --widths 18,17,17,17,17,17,17,17,17,17,17,17,26,25 --low-bits 13 --source candidates/subset/research/asymmetric_windows/candidate --report candidates/subset/research/asymmetric_windows/builder-pipeline-results.json
python3 -B candidates/subset/research/asymmetric_windows/check_mutations.py
python3 -B candidates/subset/research/asymmetric_windows/cache_budget.py
```

With the existing local compiler VM running, compile using the unchanged
`candidates/pinning/research/compile_local.py` helper, `--entry subset.cu`,
`--source candidates/subset/research/asymmetric_windows/candidate`,
`--default-build` and `--report candidates/subset/research/asymmetric_windows/native-results.json`.
The VM is stopped after compilation. It contains no NVIDIA GPU.

## Decision

Retain as a researched candidate; do not select an upload on these checks alone.
There is no measured cache hit rate or GPU throughput. Nominal52MiB capacity
does not establish residency. Two large random lookups, cache pollution and
runtime construction can consume the arithmetic saving.

`cache_budget.py` exposes this dependence rather than inventing a hit rate:
with80% hits in the compact15 control and zero cold-table hits, small-table
hits must reach91.67% merely for equal logical table miss bytes. At more than
86.67% control hits, perfect small-table reuse still adds table bytes. Extra
bytes may still be worthwhile for saved arithmetic; byte parity is not time
parity. A large expected lead over the488,210,159/s subset frontier and pending
work is not established. This experiment also has not incorporated PR77's
subset ordering/constants; the public PR96 composite showed that combining
those with external inversion is not automatically faster.
