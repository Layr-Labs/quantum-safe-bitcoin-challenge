# Compact32 root LUT: design audit before code/testing

The 832 exact 64-bit six-divstep entries encode four signed byte coefficients
(a,b,c,d), a signed delta adjustment byte and one sign bit. Census of every
entry found a,b even in [-16,64],[-14,64]; c in [-32,63]; d odd in [-31,63];
adjust even in [-4,6]. Lossless 31-bit packing:
 bits0..6 a/2 signed7; 7..13 b/2 signed7; 14..20 c signed7;
 21..26 (d-1)/2 signed6; 27..29 adjust/2 signed3; 30 sign.
Total LUT shrinks 6656 ->3328B. Bit31 unused. Decode restores exact operands,
sign mask, adjustment and decision delta for every possible table address.

Constant route: single uint32 lookup replaces uint64 lookup then exact decode.
Shared route: same cp.async/wait lifetime as current LUT but 208 16B copies,
not416. Compact table fits 128-leaf inverse arena (4096B), avoids the existing
8192B full-LUT arena that erases 4-block residency. Source global mirror is
constant-qualified aligned16; destination is alias of dead inverse arena.
LUT reads finish before root/cofactor writes; block barrier is unchanged.
Keep ordinary 64-bit table for scalar/fallback helper and original switch OFF.

Smallest preexisting arithmetic entry: tree_audit.cu with exact tree operands
SHORT_CARRY3=0, unused narrow finish K2S_PARITY_WINDOW=0, both ISO scale switches
OFF. This audits all input lengths, sparse final blocks, direct roots including
zero and bounded fallback unchanged. Production score must use unchanged
benchmark.sh exact hit gate with original short-carry and ISO config.

## Measured proof

All832 entries were reconstructed bit-for-bit from the packed value before
compilation (no dropped precision or runtime/seed specialization). Existing
tree_audit with shared32LUT returned exit0: every32/64/128/256-thread block
for counts1/129/255/256/257/8191 has zero wrong residues; roots8192/8192exact,
zero bounded status/fallback errors.

Local const+shared builds: 128regs24KiB0spills. Native shared32 build:
128regs24KiB0spills,481248B,sha2564f51c8a69534da18...
Constant32 scored380.052120vs392.313841M/s **-3.1255%**, bothPASS.
Shared32 scored screen in flight. Native ready in the frozen tree, not installed.

Zero-knob native rebuild: cubin hash exactly equals production's
e370d476d6ebfd4b51157257ce0f9d4a94d1d8068f953244ce20e9ff446eff18. All native
header bytes except the source-hash COMMENT are identical; note that the full
header compare exits1 because experimental source changes the comment. Payload
and embedded knob string unchanged; production image kept untouched.
