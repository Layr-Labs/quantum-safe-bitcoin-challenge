# Three-field checkpoint experiment

The pending PR74 production files remain unchanged. Three isolated layouts were
implemented, checked and compiled. None is selected for a new submission:
all introduce finish spills and instruction growth, so this screen does not
establish an expected substantial lead. It does not prove every such layout
slower on a GPU.

Let the XYZZ state be X,Y,A,B, with A=Z², B=Z³. Define
d=xR*A-X, W=A²*d and C=A*d². The current code saves C,Y,W,B and
batch-inverts W. The relation C*B²=W² allows C to be reconstructed if
the inversion denominator is changed.

The first two variants save Y,W,B, invert D=W*B, then compute
delta=W*(W/D)²=W/B² and h=B²/D=B/W. These replace C/W and B/W in
the same recovery formulas. The final variant saves U=Y*B, W and V=B²,
inverts D=W*V, then uses delta=W²/D, inverseW=V/D and slope numerators
yR*V±U. This moves one multiply and one square from finish into prepare;
the overall change remains two additional multiplications and one square.

Every version preserves identity factors for singular/inactive lanes and
returns from unusable finish lanes only after the collective. Recovery targets
P+R and P-R, retaining recid-zero hit priority. The outer inversion hierarchy,
table builder and fused alternative retain the submitted implementation.

| Version | Search tree nodes stored | State+tree logical read/write bytes per candidate | Extra tree multiplies per candidate |
| --- | ---: | ---: | ---: |
| PR74 |254|319.5|0|
| Full rebuild |0|192|254/256|
| Half rebuild |126|223.5|128/256|
| Scaled half rebuild |126|223.5|128/256|

Full rebuilding adds seven up-sweep synchronization stages to finish.
Half rebuilding recreates only the first 128 nodes, with one extra stage.
The remaining upper nodes use a 128-node stride; the outer root checkpoints
keep their original 256-node stride. The scaled half candidate also loads U
after the collective. These byte counts exclude table/outer-root traffic,
caching and generated spills; they are not overall speedup estimates.

All variants passed 5,000 arbitrary scaled-coordinate recoveries, 2,048
individual tree inverses, and the actual extracted prepare/root/finish kernel
bodies for sizes 1,31,32,255,256,257,511,513. The pipeline tests include
singular and inactive lanes, actual SHA/hit packing, an independent affine
oracle, one global inverse, and the exact tree multiplication counts.
OpenSSL replaces field primitives and controlled XYZZ inputs replace the
fixed-base producer. This is CPU evidence, not CUDA execution or timing.

Real CUDA 12.8.93 builds passed for sm_89 and default flags:

| Version | Prepare registers / static instructions | Finish registers / spill stores+loads / static instructions |
| --- | --- | --- |
| PR74 |122 /7817|78 /0+0 bytes /4598|
| Full rebuild |128 /7674|80 /8+8 bytes /5325|
| Half rebuild |123 /7684|80 /32+32 bytes /5346|
| Scaled half |123 /7956|80 /16+16 bytes /5045|

Shared memory remains 16 KiB prepare and 24 KiB finish; fused resources match
PR74. Static instruction totals are not dynamic work or cycle counts. The
compiler VM was stopped after the checks.

`prepare.py` reproduces the scaled half candidate from the frozen production
control; `check_pipeline.py` checks it. The earlier source snapshots and
source-bound reports are retained beside it. `comparison.json` records costs,
hashes and limitations. `external-review.json` distinguishes Gemini's completed
but unsupported defect claims from Grok's timed-out review. No new external
source was imported for this algebraic transformation.
