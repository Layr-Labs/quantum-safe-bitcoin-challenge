# Iteration 7: fuse the centered-square offset into its reduction

The current best was submitted as 414fe58f at806b050 and is validating.
Watcher owns that slot. A single board check at iteration start found the same
728337167 cefika official best. Read cefika public note and fetched its tree
ff27a2b66990a3eb554a1d4453e896c0397337ba; inherited field helper text is unchanged
there, so this is not a previously published fused centered-square primitive.

Current mode2 finish forms p2=(m1+m2-c)^2-c^2-p1 after parity0. The previous
sum-basis probe changed liveness without reducing field work and screened -2.8%.
This probe changes the primitive: insert both subtractions in the square's
288-bit first-fold result and consume their signed high word with ONE second
fold. It reuses the inherited signed-X3 fold pattern, retains the square's
product schedule and first fold, and removes two separate modular subtraction
folds. The sum scratch no longer needs to be published: post3 has no consumer
of it after center_offset, including the sum-basis branch, which reconstructs
m2 first. p1 itself remains immutable until p2 is complete; output aliases are
safe because asm inputs are copied to local registers before publication.

QSB_K2S_CENTER_FUSED defaults0, only supported for qualified mode2. Non-short
arithmetic uses the unchanged exact ModSqr/ModSub route (existing diagnostic
would not exercise the new filter!). Enabled configuration adds a nonempty
carrier knob fingerprint, so it cannot silently load the old native image.

Arithmetic caveat: this does not make the inherited filter exact. Its first
fold has dropped carries, and the sign-extended second fold stops at z3, like
inherited signed-X3. Final publication and benchmark OpenSSL checks remain
mandatory. The algebraic identity alone is not evidence of identical speculative
residues or unbiased successful hits. Do not claim all-domain exactness.

Source recipe: lab/iter7_generate_center_fused.py. It generated
center_square_sc.cuh from hit_filter_field_sc.cuh's square; attribution retained.
Build: bash candidates/subset/lab/iter7-build.sh (CUDA12.8 local and native).
Existing benchmark pair: lab/iter6_ab.sh FROZEN_CURRENT n24L_centerfused.cu
centerfused-screen 1 120, N24 seed1789110211, serialized /tmp/angel-gpu.lock.
A uses the qualified combined gate's frozen executable, not fill2 or launch4.
No production enable/carrier replacement until measured qualification.
