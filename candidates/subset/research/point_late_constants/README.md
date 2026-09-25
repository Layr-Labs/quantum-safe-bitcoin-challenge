# Late point constants: compile-gate rejection

The hypothesis was to avoid carrying eight `QSB_U2R` coordinate limbs through
the fixed-base point chain by reading the immutable constants just before
final preparation. Arithmetic order and packet layout were retained.

On both sm_86 and sm_89, the default two-block configuration still used 128
registers. Requesting three blocks reduced this to 80 registers but produced
a 280-byte cumulative stack plus substantial spills in the helper. The
intended register-pressure benefit was not achieved, so no GPU run or
correctness/performance qualification was attempted.

Production was restored byte-for-byte. The rejected header and variant
wrappers are hash-verified in `rejected-source-overlay.tar.xz`; extract that
overlay only into an isolated copy of the prepared `candidates/subset` tree
to reproduce the experiment. `compile-evidence.json` records PTX fingerprints
and target resource reports. The default production PTX remained unchanged
while the experiment's default-off macro existed.
