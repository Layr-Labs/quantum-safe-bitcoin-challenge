# Pinning successor status, 2026-09-22

Source milestone: `f89859255f8933beb5eab20f17a4e356f995e31e`. Branch: codex/pinning-affine-next.
Built locally; **not submitted**. Prior submission
8fd91df0-4fba-44d7-afa4-4bc349723931 remains separate and undisturbed.

The produced route now uses a collective affine comb with exact root inverses,
16 KiB shared arena reuse, and warp synchronization for small tree levels.
Fused fallback retains the previous projective path. Default native and
compute52-to-sm89 produced kernel: 108 registers, zero stack and spills.
Fused/producer/finish/replay: 126/40/62/72 registers, zero stack and spills.
The inherited table builder and super-root inversion still have call stacks.

Checks: 384 OpenSSL collective combs, 80,000 canonical add/sub alias cases,
42,048 interpreted device PTX add/sub cases; SHA producer, parity replay and
stream-drain selection regression tests. See SUBMISSION.md for test limitations.

This is an algorithmic experiment, with no defensible expected score. Fourteen
serial collective roots may erase the point-arithmetic savings. Zero spills
are necessary evidence, not proof of faster execution. Five-block bound spilled
and is rejected. Read CUDA-CENSUS.json and YUKON-SUBMISSION.md before release.
