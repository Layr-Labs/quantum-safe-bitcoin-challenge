# Fuse inverse and tail after separate SHA and point

Research only. Preserve the four-stage SHA and point kernels. Fuse the collective inverse stage with the two per-thread finishes. Keep inverse A and B in registers instead of storing four limbs and reloading them for each candidate; save 64 bytes of global traffic per candidate and one launch. Read finish fields only after the tree; preserve pairing, arithmetic, validity, record format and exact replay. All lanes still enter the tree with identity for absent candidates; emit is called only for present candidates. Check compiled resources before measurement. The pending production candidate remains unchanged and no external source upload is authorized by this experiment.

## Result

Rejected after warm comparison. Baseline 246.060295 M/s, fusion 246.131967 M/s (+0.0291%). Both complete exactly 14562623488 candidates and independently verify the same 1765 hits. The common 113770496-epoch prefix matches exactly; no missing or extra hits. First-use fusion verified 1683 hits but startup reduced wall throughput to 234.615643 M/s. Fusion provides no qualifying gain and remains research only. Sources, logs, compressed run artifacts and binary hashes are retained; production remains the four-stage candidate awaiting submission authorization.
