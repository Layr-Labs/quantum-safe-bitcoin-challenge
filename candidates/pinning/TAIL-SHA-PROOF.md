# Complete per-sequence SHA precomputation

The block contains live words W0 and W1, a run-constant W2, zero words W3
through W14, and length word W15=79960. Each sequence has an immutable SHA
midstate. The host computes six terms that depend only on that state and W2:

- The constant parts of round 0's new A and E.
- The initial H and round constant contributions for rounds 1, 2, and 3.
- The constant part of schedule word 17, s1(W15)+s0(W2).

Round 0 still adds W0 at the original weight. Rounds 1 through 3 still compute
every state-dependent rotation and Boolean term. The expanded schedule retains
the contributions of W0 and W1, all 64 rounds, and the final midstate addition.
No digest bits, carries, elliptic-curve operations, or output checks are omitted.

The midstate and terms travel together in a by-value kernel parameter. The
three-stream host loop creates it after obtaining the current sequence's state;
each asynchronous launch owns its parameter copy. The alternate host path also
updates it whenever the sequence state changes. This preserves sequence identity
without mutable device-global precompute data.

`audit_tail_sha.py` extracts the actual helpers and SHA macros from the shipped
source. It checks 32,768 arbitrary midstates and three live words against a full
independent OpenSSL compression, with undefined-behavior sanitization. A wrong
precomputed round-0 term is required to fail. This CPU argument is separate from
the exact-source native pipeline checks and full production evidence recorded
in RESEARCH.md and SOURCE-MANIFEST.json.

The complete SHA helper was reviewed in public submission 736a5884, commit
82d33da84bf144b3cc016544d00f96cd37fdf631. Its unrelated public field arithmetic
was not imported. All eight arithmetic, point, recovery and hash headers remain
byte-identical to the frozen 88567233 parent. Its point schedule and arithmetic
arguments are retained in the other proof documents.
