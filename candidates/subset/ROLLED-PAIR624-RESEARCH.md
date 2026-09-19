# Rolled constant blocks in the credited PR624 paired SHA

This ablation uses the parent's existing QSB_PAIR_SHA_UNROLL_CONST=0 switch.
The four constant blocks and their rounds execute in the same order, retaining
both independent SHA states. The purpose is to compare code size and register
pressure on the actual default CUDA12.8 JIT pipeline. Static code generation
alone cannot establish a speed gain. Credits remain with ercumentyildirim and
dukemawex for the parent implementation.

The first-state oracle uses the actual flattened producer. All expected values
and coverage remain unchanged. No device result or submission is claimed yet.
