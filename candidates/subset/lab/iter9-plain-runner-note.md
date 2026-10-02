# Label and timing caveats (all iteration9 runs)

The hardware is dev RTX3090. The unchanged harnessconfig's gpu string RTX4090 is
a declared label, not detection; it appears in score files but must NEVER be
reported as measured4090 speed. Official leader score736585478 is separately
4090 runnermeasurement. Compare only interleaved local same3090 rates.

All screens used pre-existing N24/120s fixedtime protocol and fixedseed1789110211.
The benchmark owns clock and hit verification. Throughput is verified_hits*2^24/2
/elapsed, NOT self-reported GPU count. Existing filter/Poisson-band warning arises
on both arms and is retained; no attempt to normalize away candidate yield.

Default host verifier re-derives each nominated recovery, then published hits
are re-derived by unchangedbenchmark verifier. A pure cache hint cannot alter
arithmetic by design but exact measurements are still required. A SHA rewrite is
exact modulo2^32 by reviewed identity; exact benchmark demonstrates integration,
not exhaustive correctness of every possible input. Arithmetic guards and
randomized official evaluation remain unchanged.

Concurrent process scheduling means a pair may have another job's locked arm
between its own arms; only ONE GPU/scored instance ever runs. Persist manifests,
scorefiles/full logs; never edit the running shell script or wrappers/stamps.
All archived device source frozenbefore measuredbinary compilation.

Native census refinement: total instruction counts14520/14256/14128 are stable.
Previous plain string" NOP;" count in iteration7 missed spacing"NOP ;" (including
two predicated NOP). Fullopcode census properly counts20(default),21(scoped),
17(gateextension), versus older14/15/11literal counts. NonNOPdefault14500,
scoped14235,gate14111. This changes neither code nor instruction-totaldelta,
nor any score. Persist both literal and opcodecounts for audit; avoid mixing them.
