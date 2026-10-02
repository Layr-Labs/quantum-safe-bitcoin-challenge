# Iteration 12 — legacy pm9 dual-carry fusion

## Hypothesis and scope

Current leader 4cc9d2d8 (source 2f57d80) reports dual-carry fusion in its MAC
schedule. That schedule is not present in our qualified centered-square source.
This probe ports the mechanism, not the leader's whole multiply emitter: one
odd-accumulator two-carry sum and optionally one high-word two-carry sum in the
existing pm9 emitter. QSB_YP_LEGACY_DC is a default-off two-bit mask.

No enumeration, SHA, slot ownership, host publication, or candidate-domain
change. This is distinct from the centered-offset fusion and SHA probes.

## Logical review

The first odd add chain emits carry A. A non-.cc addc captures it. The next
independent chain emits carry B. Existing code captures B and adds captured A;
the probe's non-.cc addc directly computes A+B. Moving an opaque constant zero
to the first capture permits the compiler to represent both carry inputs in an
IADD3.X. Since the constant is initialized to zero and no host upload changes
it, all four (A,B) combinations give the same 0,1,1,2 results. Neither capture
changes CC, and all following chains retain their original .cc instruction.
The optional top-word bit changes only the spelling of its first zero capture;
its later carry-add instruction remains unchanged. No rare carry is dropped.

Both yp (loop) and yf (final resolve), in both fold-drop preprocessor arms, use
the new macros. Production fold-drop remains 0. Default-off macros reproduce
the original PTX strings exactly and add no constant symbol. Carrier identity
includes the knob only when nonzero, preventing accidental use of the old
native image for enabled probes. Enabled native images require regeneration;
none is installed in production.

No new allocation, cleanup, empty-work, or failure paths are added. Existing
host exact verification remains the publication boundary. Successful hits do
not establish exhaustive field-helper correctness.

## Native screen

CUDA 12.8.93 sm_89, same-tool cuobjdump with its nvdisasm on PATH:

| mask | instructions | registers | shared bytes | stack bytes | ptxas spill bytes |
|---|---:|---:|---:|---:|---:|
| 0 | 14520 | 128 | 24576 | 0 | 0 |
| 1 | 14552 | 128 | 24576 | 8 | 0 |
| 3 | 14560 | 128 | 24576 | 0 | 0 |

Mask1 removes 3 SEL and adds 3 IADD3.X but also adds predicate/LOP3/move
machinery and one LDL.LU.64/STL.64 pair. Mask3 removes 6 SEL and adds 6 IADD3.X
but adds more conversion/move machinery. The intended local fusion is emitted;
the surrounding costs defeat a static-instruction-count win. Do not equate
ptxas zero reported spills with zero local-memory traffic in mask1.

Default mask0 CUBIN sha256 remains exactly the qualified
9c9aab2abc1a6c7368627ae239a40a8e0f4f917df08de43930ee90ee3d48f5ad.

Initial census orchestration failed because nvdisasm was absent from PATH;
corrected tool location produced all three disassemblies. No failed build is
counted as verifier evidence. Both local benchmark screens use the existing
N24 benchmark, frozen qualified control, serial GPU flock, 120s per arm. See
iter12 findings for the actual scored outcomes (pending when this was written).

## Concrete legacy results

- Mask1: 389.775892 -> 355.265565 M/s (-8.853890%), 5634 and5172
  independently verified hits; both existingN24 benchmark RESULT PASS.
- Mask3: 395.751801 -> 352.340156 M/s (-10.969412%), 5722 and5121
  independently verified hits; both RESULT PASS.
- sm89 full digest count14520 ->14552 /14560 (+32/+40). Mask1 adds an8B
  stack frame and LDL.LU.64/STL.64 despite reporting zero spill bytes. Mask3
  retains0Bstack. Both128registers/24KiBshared. Native SEL captures -3/-6,
  but extra moves, predicate conversions and logic dominate. Different original
  emitter liveness means the leader's cheap eight-fusion trick is not portable
  as a one/two-fusion spelling change. Reject both, no repeat gate.
- These are individual negative screens, not confirmed regressions. Exact-hit
  verification validates published hits, not all-domain field arithmetic.

## Follow-on: isolate the leader's complete MAC schedule

Added y_pair_mac_leader.cuh as a GPL3-derived public-source emitter from
2f57d80, with source SHA attribution. Default-off QSB_YP_MAC_PORT selects it
without importing any other leader device knobs. QSB_FKM falls back to literal
977; no leader fold-register switch is implicitly enabled. QSB_YP_MAC2/DC1
interleaves both products' rows and merges once into the existing513-bit sum;
retains exact original fold tail and exact host publication. This addresses the
liveness deficiency identified in the legacy spelling probe instead of sweeping
constant-zero variations.

sm89 candidate:14464 native instructions (-56),128registers/24KiBshared,
0Bstack/0Bspills,18fewerSEL. ExistingN24 paired score pending at note creation.
Final default native build9c9aab2... remains byte-identical to qualified carrier.

GPU runs: frozen immutable binary/source manifest, flock serializes each arm;
benchmark.sh subset through cached gpu_wrap,120s,N24,seed1789110211. No changes
to judge/harness. ActualGPU3090; harness's declared4090 label is not hardware.
