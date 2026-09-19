# SPDX-License-Identifier: GPL-3.0-only
from fractions import Fraction
import json
n=int('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141',16)
lam=int('5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72',16)
a1=int('3086d221a7d46bcde86c90e49284eb15',16)
a2=int('114ca50f7a8e2f3f657c1108d9d44cfd8',16)
b1=int('e4437ed6010e88286f547fa90abfe4c3',16)
R=(1<<127)+(1<<118)
assert a1*a1+a2*b1==n
assert (a1-lam*b1)%n==0 and (a2+lam*a1)%n==0
assert R*(a1+a2)<n and R*(b1+a1)<n
lower=(1<<127)-(1<<16);upper=(1<<127)+(1<<16)-1
for c in range(7):
    if c:
        termmax=((1<<17)-1)*(1<<(17*c-1));lower-=termmax;upper+=termmax
    assert lower==(1<<127)-(1<<(17*(c+1)-1))
    assert upper==(1<<127)+(1<<(17*(c+1)-1))-1
    assert 0<lower<=upper<R<n
assert 511<R
print(json.dumps({'lattice_basis_and_determinant':'pass','prefix_intervals':'pass',
    'coefficient_bounds':[str(Fraction(R*(a1+a2),n)),str(Fraction(R*(b1+a1),n))],
    'finite_joint_term_bound':'pass','scope':'Exact integer checks supporting the argument in PREFIX-PROOF.md; not a proof of CUDA implementation correctness.'},indent=2))
