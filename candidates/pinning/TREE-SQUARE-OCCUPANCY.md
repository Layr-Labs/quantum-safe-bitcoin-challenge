# Conditional tree fold, exact square rows and prepare occupancy

This package keeps complete finite-field arithmetic and the existing canonical
point/recovery boundaries. The point and tree multipliers retain paired column
carries and every required modular correction.

The tree multiplier now skips the upper five second-fold limbs only when the
actual carry out of the low 96 bits is zero. A nonzero carry is propagated
through all five upper limbs. If it overflows, the final 2^32+977 correction is
retained. TREE-FOLD-PROOF.md gives the integer bound and all branch cases.

In the point square, five row-initializer carry captures are provably zero.
For 32-bit x,y and a carry bit c, x*y+c <= (2^32-1)^2+1 < 2^64.
The o9/e10 initializers have exactly this form, so their outgoing carry bits
are zero. The o11/e12 initializers then add a product and a carry to those
zero values, giving the same bound; o13 similarly leaves o15 zero. Later
chains begin with add.cc, so they do not consume a removed condition flag.
Only those five zero captures are removed. Every diagonal product, the
complete reduction suffix and its actual final-carry correction remain.
This pre-reduction simplification also appears in public208bbcb6, commit
a668c4e5fd80db398c13222453f9c9a645612649. Its truncated reductions are not used.

The prepare launch bound requests five 128-thread blocks. CUDA 12.8.93 uses
96 registers for these preparation kernels and 72 for the finish kernel,
with no spills in these three primary specializations. These are static
compiler observations, not performance measurements or a guarantee about
residency during every execution.

The CPU audit executes actual extracted PTX on 5,207 pairs per arithmetic
function. It covers all three complete-fold paths: 5,081/103/23 for point and
tree multiplication and 5,055/75/77 for square. The independent modular
integer oracle agrees. Exact-source native and full-production qualification
are separate requirements; RESEARCH.md and SOURCE-MANIFEST.json record their
status and the actual receipts when available.
