# Regular-prefix proof for grouped GLV

Let A be the nonzero problem-specific `neg_r_inv*G`, n the secp256k1 prime
group order, and phi(x,y)=(beta*x,y), corresponding to lambda modulo n.
Each signed component has magnitude below 2^128. For chunks 0 through c<=6,
the unsigned sum has exact scalar interval

    [2^127 - 2^(17(c+1)-1), 2^127 + 2^(17(c+1)-1) - 1].

This follows from the biased unsigned chunk 0 and the symmetric nonzero odd
digits in each later chunk. Flipping the sign of the next odd digit leaves
the same interval bound, so it also bounds prefix-minus-next. Applying the
component's overall sign only reflects this interval. Let R=2^127+2^118.
All such magnitudes are strictly between zero and R, with R<n.

Accumulate the seven ordinary Q chunks first. Before phi, every potential
equality of the accumulator with plus/minus the next point would require a
nonzero scalar of magnitude below n to be zero modulo n, which is impossible.
This includes the two-point seed. Thus those additions are nonsingular.

Apply phi to the Q accumulator by scaling XYZZ X by beta. Y, ZZ, ZZZ and the
deferred-Y anchor do not change. While adding the seven ordinary P chunks,
a singular mixed addition would imply x+lambda*y=0 mod n, where |x|,|y|<R
and y is the nonzero signed Q prefix. The kernel of this map has index n.
The vectors (a1,-b1) and (a2,a1) lie in the kernel and have determinant
a1^2+a2*b1=n, so they form its full integer lattice basis.

If (x,y) is in that lattice, its two integer coefficients are

    u=(a1*x-a2*y)/n, v=(b1*x+a1*y)/n.

Exact integer checks establish R*(a1+a2)<n and R*(b1+a1)<n. Therefore
|u|<1 and |v|<1, so both integer coefficients must be zero. This would force
y=0, contradicting the Q-prefix interval. Thus no P-prefix addition has a
zero denominator either. No probabilistic argument or omitted rare case is
used for these first fourteen terms.

The final joint window can cancel or otherwise require the exceptional point
law. It is deliberately excluded from the proof and retains that full law.
All result reconstruction, hashing and hit verification remain unchanged.

`prefix-proof.json` records the exact constants, rational coefficient bounds,
and all seven interval endpoints checked by `prepare-regular.py`. The proof
applies to either scalar splitter because it requires only signed 128-bit
component magnitudes. GPU audits must additionally verify the implementation.

## Finite joint term and specialized final exception

The joint term has scalar `(du+lambda*dv)*2^118`, up to a common sign,
where du and dv are nonzero odd integers of magnitude at most 511. Since
511<R, the same lattice rectangle argument shows du+lambda*dv is nonzero
modulo n. Multiplication by 2^118 and the nonzero base A preserves this.
The joint affine term J is therefore finite.

The first fourteen terms also yield a finite accumulator by the rectangle
argument above. In the final addition, equal x coordinates thus mean either
opposite points (return infinity) or equal points (double J). For doubling,
set h=2*yJ, U=h^2, V=h^3, m=3*xJ^2, Q=xJ*U, X=m^2-2*Q, and
Ydeferred=m*(Q-X). The actual Y numerator is Ydeferred-yJ*V, exactly the
affine doubling formula in XYZZ coordinates. This avoids reconstructing the
old projective point, while retaining both exceptional cases.

`tests/glv_prefix_proof.py` checks the exact integer constants and bounds.
`tests/glv_component_audit.cu` tests the actual production recoder and
accumulator against OpenSSL for arbitrary signed 128-bit components,
including lattice cancellation and final-joint doubling. These device tests
are additional evidence for the implementation, separate from the algebra.
