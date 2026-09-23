# Seed X3 fused square experiment

This branch starts from PR #1205 source commit
`e3e413bb820dc339a11cf30df4de7ade8d179845`. PR #1205 measured
829,282,307 verified candidates/s on r5 and missed the 835,195,327
promotion floor. `QSB_SEED_FUSE_X3=0` restores the exact prior seed source.

The seed uses `P=X2-X1`, `R=Y2-Y1`, `PP=P²`, `PPP=P³`, and `Q=X1*PP`, with
`X3=R²-PPP-2Q`. The ordinary mixed-add has a fused square for
`R²+PPP-2Q`. Define `P'=-P` and `R'=-R` instead. Then `PP'=PP`, `PPP'=-PPP`,
`Q'=Q`, and `R'²+PPP'-2Q'` is the same `X3`. Both the deferred ordinate
and `ZZZ3` reverse sign. Dividing the corresponding projective numerator by
`ZZZ3` therefore yields the same affine point. The seed's existing
`QSB_NEG_Y_MAC` convention and the ordinary convention were both checked.

`test_seed_fuse.py` passed 4,096 edge tuples and 10,000 random tuples over
the secp256k1 field, including zero, p-1, p, and 2^256-1 inputs reduced
modulo p. This is an algebraic test, not execution of the GPU PTX. Both
switch states built with CUDA 12.6 for native `sm_89`; the all-on version
also built with the organizer-style default target. Stage 0 had zero frame
and zero spills in both versions. The native on version used **114 registers**
versus **122** for the off control, with the same **6,696** static SASS
instructions. The reduction in registers did not cross a 128-thread block
residency threshold in a simple 64 Ki-register calculation. The static
instruction count therefore gives no evidence of a speed gain despite the
mathematically useful rewrite.

No local NVIDIA GPU is present. Do not submit this change alone on a
projection: measure actual stage-0 latency and full verified throughput on
the ranked RTX 4090, or use it as register headroom for one isolated,
correctness-checked follow-up that needs those registers.
