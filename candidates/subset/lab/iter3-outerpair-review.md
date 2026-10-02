# SHA256d outer-pair probe, review before verification

The live qsb_pair_epoch_z_value first produces two SHA256 mid-results with
the inherited scheduled paired transform. Baseline subsequently hashes each
32-byte result separately, from the standard IV with word 8 = 0x80000000,
words 9..14 = 0, word 15 = 256. This probe uses the existing round-interleaved
generic transform used by the pubkey gate to hash those SAME two blocks.
This is not the earlier literal-K solo transform, and not ZLAB_PAIRSHA, which
only changes problem startup. It changes per-candidate outer SHA dependency
scheduling, using an already implemented helper rather than new SHA rounds.

Each stream has its own 16-word schedule and eight result words. The helper
starts from the same IV, performs exactly 64 rounds independently for each
stream and adds back the IV. Scalar conversion remains big-endian reverse
word pairs (6,7), (4,5), (2,3), (0,1). There is no cross-stream data exchange.
All digest lanes still execute both hashes before the field front, including
identity-substituted partial-tail lanes. No allocation, synchronization,
class lookup, hit encoding, cleanup or error path changes. Register pressure
and instruction-cache footprint may swamp the ILP benefit; compile reports
are evidence of resources, not throughput or exact verification.

Knob off by default; enabled value is added to the carrier fingerprint. The
generic paired helper requires QSB_GATE_PAIR=1 and QSB_OUTER_LITK=0; both
conditions have compile-time guards. The old defaults are unchanged.
Existing benchmark.sh subset plus independent verifier is the gate.
