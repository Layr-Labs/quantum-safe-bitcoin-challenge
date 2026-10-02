# Denominator cross-factor relocation: logical audit before measurement

Live baseline computes leaf = inv(WA*WB) (including the inherited isomorphism
scale), invA = leaf*WB, invB = leaf*WA; each paired tail starts by multiplying
its ZZ by invA/B to obtain the shared slope scale h. Thus hA = ZZ_A*WB*leaf
and hB = ZZ_B*WA*leaf. This probe moves the opposite-W multiplication to the
ZZ words BEFORE the tree: parked A's rows8..11 become ZZ_A*WB; B's four ZZ
words become ZZ_B*WA. Both tails subsequently receive leaf unchanged.

Field multiplication is associative modulo p; isomorphism's leaf scale is
unchanged. The same canonical five-word QSB_TREE_MUL is relocated, NOT
replaced by speculative arithmetic. Inputs' fifth limbs are setzero as in
the original post-tree multiplication. The inherited tail and parity window
then see congruent slope factors; exact verifier is still required since
noncanonical/speculative representations can affect the hit filter.

WA/WB are set to identity for unusable or partial-tail candidates BEFORE
forming the product and applying this relocation. If B is unusable, WB=1
preserves A's scale; likewise for A. The old guards still suppress invalid
candidate tails. Every active digest lane executes identical tree arithmetic
and barriers; no new synchronization, allocation, hit path or cleanup changes.
If both unusable, extra early arithmetic is unused, but leaf remainsidentity.

Four A ZZ words loaded and rewritten in existing vector park rows; no extra
shared allocation. Twelve B words stilllive, but prodA/prodB no longer need
to survive the root/tails; native resource/SASS inspection will test whether
that creates a meaningful register liveness improvement. Multiplication count
unchanged for full valid lanes (two relocated, not added).

Off default; signatureonlyaddswhenenabled; compileguards exclude PRE3,
stagger/weave/rootparking and require paired12wordvectorfront. Existing
benchmark.sh exact verifier is the relevant pre-existing integration entry.

## B-only diagnostic, value 2

After building the both-candidate form, SASS confirms two extra LDS128 and
two extra STS128 for A's ZZ update. Value 2 isolates liveness versus shared
cost: only B's register ZZ is multiplied by WA before the tree. A still uses
its original invA=leaf*WB and original untouched parked ZZ. This retires WA
but not WB, preserves exactly the same canonical multiplication count and
avoids A's shared round trip. B's tail still uses leaf directly. Guards,
invalid identities, isomorphic factor and exact integration test unchanged.
Separate frozen form1 measurement is not altered by adding form2 source.

## Clarification from implementing helper audit (iteration4)

Production QSB_TREE_MUL routes through qsb_field_mul_tree -> qsb_filter_mul when SHORT_CARRY3=1; this is an inherited speculative helper, not exact canonical multiply for every adversarial operand. Associativity only describes exact target field algebra. Exact unchangedOpenSSLpublication preventsinvalidhits; anyspeculativeyieldchangeaffectsmeasuredscoreandmustbeverifiedempirically. Allgatesaboveverifiedeverypublishedhit. Noteupdatedtoavoidstrongerclaim.
