> Release update: the selected mechanisms are now default ON. See SUBMISSION.md
> for the current package, runtime selection and evidence. Earlier sections below
> retain research history; rejected patch files remain in the research branch.

# Negative deferred Y and seeded field multiply-add

Research checkpoint on promoted 7c3609b plus exact parity replay. GPT 6 Astra,
xhigh, Codex. No local NVIDIA GPU or device throughput result.

The current deferred point ordinate is Ycore, with actual Y = Ycore-yoff*V.
Store N=-Ycore instead. The next slope numerator becomes
R=(y2+yoff)*V+N. A field multiply-add replaces a field multiply followed by a
borrow-corrected subtraction. The new deferred ordinate is Nnew=R*(Xnew-Q),
so reversing the existing final subtraction produces the required sign for
free. The two-affine seed uses the same reversed subtraction. Resolve N's
sign only once at chain completion, then perform the existing anchor removal.
QSB_NEG_Y_MAC is default OFF until the complete integration is validated.

negative_y_mac.cuh starts with the promoted _ModMultCore instruction schedule.
It seeds the low 256-bit even product chain with the addend and carries its
initial overflow into e4. The full integer result is exactly a*b+c: its upper
bound (B-1)^2+(B-1)=B^2-B fits 512 bits for B=2^256. It uses the inherited
pseudo-Mersenne reduction and C31/RP truncation sites; these are not a new
claim of all-input exactness. The host publication gate is unchanged.

Native CUDA 12.6.20 sm_89 N24: prepare 126 registers versus 128, zero stack
or spills; same 12 KiB shared memory. Finish stays at 62 registers. The hot
mixed-add loop is 1035 SASS instructions versus 1048: -1 IMAD, -10 IADD3,
-2 LOP3 per addition, repeated thirteen times. Static prepare size is 5680
versus 5672 (entry/tail differences). These are resource and operation counts,
not a measured throughput gain. The resident block limit has not changed.

Tests: test_negative_y_ptx.py checks 34176 triples through the actual generated
PTX schedule. All 512-bit seeded products match Python exactly. 548 directed
field cases reach inherited truncation sites; no unexplained field mismatch.
The predicates count g3 carry, z8 overflow, and only the final z2+sfc carry.

test_negative_y_chain.py compiles the extracted seed, mixed add and complete
chain twice with exact OpenSSL-backed CPU arithmetic. 2048 random fifteen-point
chains match both the original representation and independent OpenSSL EC sums
(30720 point addends). No mismatches. Device instruction correctness for the
new MAC is covered separately by the PTX test, not by this host oracle.

A preliminary positive-Y multiply-subtract variant grew the loop to 1061
instructions and is rejected. It is not included by production code. The
negative representation removes the complement and constant correction that
made that first attempt unattractive.

## Full negative-ordinate checkpoint and recovery

Keep Nfinal=N+yoff*V=-Yactual at chain completion, using the seeded MAC
instead of a negation plus multiply/subtract. The saved first plane is
therefore -Y*hc; the packed finish restores the original slopes as l=u+v
and m=u-v. The second plane and denominator are unchanged. The hot chain
and checkpoint need no new normalization branch. The runtime point-add
resolver still normalizes before its required positive-ordinate negation.

The full CPU chain test now also computes both recovered compressed keys
and compares them against OpenSSL EC additions: 2048 chains, 30720 input
points, 4096 compressed keys, zero mismatches. The full extracted finish
and replay test compares the control positive ordinate with the new
negative ordinate: 111190 executions, 17990 queued replays, 388 matching
hits, zero mismatches. The actual seeded MAC PTX remains separately tested.

Native sm_89 with NEG_Y_MAC=1: stage 0 5648 static instructions and 1037
in the thirteen-iteration loop, 126 registers; stage 2 3784 instructions,
62 registers. Both have zero stack and spills. The original promoted loop
is 1048 instructions; this is an operation saving, not a throughput result.
The flag remains default OFF pending candidate selection and packaging.

Additional screens were rejected: a fused affine seed changed the hot loop
to 1051 instructions despite its shorter algebraic formula. Seeding the MAC
with paired 32-bit mad.lo/madc.hi instead of wide multiply plus carry adds
passed 34176 raw-product/field cases but grew the loop to 1046 instructions
and the static stage 0 to 5656. The latter header is retained as research
only and is not included by production. RAW finish gives no additional
register saving with replay and grows stage 2 to 3792 instructions. Packed
absolute digit offsets add more decoder work than the loop saving on this
source; their patch and RAW are retained in composition-screen-rejected.patch.
