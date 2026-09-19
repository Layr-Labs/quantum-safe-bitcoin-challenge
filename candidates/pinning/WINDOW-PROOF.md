# Signed odd windows and complete final addition

Letn be the secp256k1 group order andA=(-r^-1)G, withr supplied by the
current problem. For a raw256-bit hashk, reducek modulo n, then represent
D=2k-n. D is an odd signed integer in[-n,n). Tables contain
positive odd multiples ofA/2 at each window's bit offset, with runtime signs.
Thus the reconstructed group point is(D/2)A=kA modulo n.

For W=11..16 letq=floor(256/W),r=256 modW. The first r windows have widthq+1
and the rest widthq. The widths sum256. Chunkc starts atq*c+min(c,r).
Its signed digit is odd, nonzero, with absolute value at most2^width-1.
The code extracts digits directly from the signed residue; tests/window_cases.py
independently reconstructs the integer and checks the production identity.

Before chunkc, the prefix is odd and its magnitude is at most2^shift-1,
where shift is that chunk's bit offset. The next term has magnitude at least
2^shift. Their sum of magnitudes is at most2^(shift+width)-1. Before the final
chunk this is less than n. Consequently neither their difference nor sum
can be0 modulo n: the intermediate mixed additions are nonsingular.

The final prefix and table point are finite, but can agree or be opposite.
The final mixed-add guard distinguishes unequal y (infinity) from equal y
(doubling the known affine table point). For doubling,h=2y,U=h^2,V=h^3,
m=3x^2,Q=xU,X=m^2-2Q, deferredY=m(Q-X); the caller subtractsyV.
Zero tests accept both0 and the inherited field representationp.

Each table is built afresh from currentA/2 by small OpenSSL ladders and a GPU
sum. The odd low part is nonzero; all table multipliers are below2^24 and
cannot vanish modulo n. Every built table is spot-checked against independent
OpenSSL multiplication before tuning/search, including all chunk corners.
The audit additionally compares16,384 raw/boundary scalars for every geometry,
including values constructed from individual/final terms, both signs, doubling
and endomorphism multiples, with independent OpenSSL expected coordinates.

The SHA, both recovery ids, hit predicate and candidate enumeration remain the
promoted pinning algorithm. No precomputed hits, instance cache or scorer edit.
