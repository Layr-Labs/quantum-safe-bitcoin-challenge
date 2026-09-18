# Three-epoch compound inverse screen

This experiment evaluates a distinct extension beyond the K2 mechanism now
pending as `b76b3d5`: combine three candidate denominators into one inverse-tree leaf,
then split the leaf inverse back into three candidate inverses.

The Bend model uses F_17 so it can exhaust every triple, including all zero
patterns. `LAWS.bend` states concrete nonzero and zero-isolation laws;
`PROOF.bend` proves them. `MODEL.bend` separately checks all 17^3 = 4,913
triples. This does not prove secp256k1 arithmetic, CUDA transcription, shared
memory synchronization, register placement or throughput.

For factors `a,b,c`, the schedule is:

1. `ab = a*b`, `abc = ab*c`;
2. one shared tree inverse gives `1/abc`;
3. `1/ab = c/abc`;
4. `1/a = b/ab`, `1/b = a/ab`, `1/c = ab/abc`.

That is six field multiplies around the common inverse, or 2.0 per candidate,
versus three multiplies / 1.5 per candidate for a two-way split. Zero factors
are replaced by one in the shared product and their own outputs are masked;
other candidates remain valid.

The pending two-epoch design parks one 64-byte pre-inverse `(m1,m2)` pair per
lane in 16 KiB shared memory, bringing the promoted 24 KiB tree to 40 KiB. A
third epoch needs another 16 KiB pair. The initial candidate therefore keeps
one parked pair in shared memory and places the second in a compact 16 KiB
global scratch slot per block/stream. That adds one 32-byte field-pair write and
read for 256 of every 768 candidates: 16 KiB round-trip per block, or about
42.7 bytes per candidate. This is a cache/latency hypothesis, not a bandwidth
or performance claim.

Abandon before submission if native compilation introduces repeated local
traffic in the EC chain, if scratch indexing cannot be made slot-safe under the
two-stream pipeline, or if GPU A/B does not beat the exact two-epoch control by
more than noise. The optimistic ceiling from the published 13% inverse-tree
share is roughly 13%/6 = 2.17% over two epochs before the extra multiply and
scratch costs.

`check_k3.py` is an independent Python-bigint oracle over the actual secp256k1
field. It checks boundary/random triples, all zero patterns, a deliberately
omitted-third-factor mutation, two-slot scratch-address injectivity and the
wait-before-reuse ring protocol. It does not execute CUDA or establish memory
visibility.

`component/k3_split_audit.cu` binds the six-multiply schedule to the exact
promoted `qsb_field_mul_raw` implementation. Its CUDA 12.8 sm89/default report
is in `component/native-results.json`. This is a component allocation screen,
not integrated-kernel or GPU evidence.

`check_parked_finish.py` covers the other K3-specific seam: it stores
`(yR*ZZZ-Y)*ZZ` and `(yR*ZZZ+Y)*ZZ` before inversion of `W=ZZZ*d`, reloads
them after the inverse, and checks both recovered points against independent
affine secp256k1 additions. `component/k3_parked_finish_audit.cu` compiles the
corresponding six post-inverse multiplies against the promoted field primitive;
`component/native-finish-results.json` records the static codegen result.

The later public artifact submission `1344772` disclosed pending `7140869`'s
exact source commit, `a86816f56ceee3701a152fc0f7007033e796b319`.
`exact_k2/` preserves that byte-verified K2 source and its native baseline;
`k3_candidate/` is the first integrated K3 derivative. It parks epoch A's
numerators plus denominator in 24 KiB of shared state, parks epoch B's 16 KiB
numerator pair in stream-slot-owned global scratch, and keeps epoch C live.
`check_integrated_k3.py` binds the base hashes, split schedule, scratch slots,
host multiplier, and complete epoch count. `check_integrated_k3_runtime.py`
compiles the production pre/post helpers unchanged, executes the exact K3 split
and C -> A -> B finish order with OpenSSL field operations, and compares all
usable outputs with an independent affine oracle. The integrated prototype is
research only: its static spills are substantially worse than K2 and it has no
GPU A/B.

`check_digit_window.py` covers the register-carried 17-bit digit window added
after `b76b3d5` disclosed that mechanism. It checks 1,300,065 carried/direct
digit pairs and rejects a one-bit-shift mutation. The port reduced the retained
K3 sm89 spills from 96/100 to 60/64 bytes, but remains static-only evidence.

The retained follow-up uses `ZLAB_TREE=3`, an in-place packed inverse tree. It
overwrites a product level only after all sibling readers join, preserving the
cooperative root inverse while removing the separate 8 KiB inverse array. The
freed shared memory parks B's denominator on-chip. `COMPACT_TREE_PROOF.bend`
checks representative packed-level boundaries; `check_compact_tree.py` executes
the production tree body with 256-thread CTA simulation and OpenSSL arithmetic.
The CUDA result is 24/24-byte spills with unchanged global scratch. This is
still static/source-bound evidence, not a measured speedup.

The active K3 snapshot also carries the minimal cooperative-inverse correction
disclosed by pending `f63b274`: the signed 64-bit final row accumulator is
shifted before narrowing the top limb. The prior form narrowed first. The
independent Bend reduced-radix proof/model in `../zinv_row_accumulator/`
demonstrates mutation witnesses for that truncation. This is a correctness
repair, not a claimed speedup; the ranked comparison still requires GPU A/B.
