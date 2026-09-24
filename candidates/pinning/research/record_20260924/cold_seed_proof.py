#!/usr/bin/env python3
"""Integer proof certificate for GLV cold order [4,5,3,2,1,0].

All lattice/reciprocal constants are parsed from the current production
GLVScalar.cuh. The proof concerns exact group arithmetic. Inherited bounded
field approximations can still change rare missed nominations after a reorder.
This script neither runs a GPU nor estimates performance.

For a nonnegative component m, bank0 is b+f0, where
b=(T+1)*2^99-2^17 and 0<=f0<2^18. Other banks are signed odd multiples
of 2^17, 2^36, 2^54, 2^72, 2^99. Their total maximum absolute value is b.

Standalone component: banks4+5 have valuation72; appending banks3,2,1
successively lowers that valuation to54,36,17. Distinct valuations prevent
an addend from equalling either sign of its preceding prefix. For bank0,
P_before=m-bank0. Cancellation requires m=0; doubling requires m=2*bank0,
which exceeds the proven GLV magnitude bound. A first zero component is
skipped by production; zero input is returned before starting the chain.

Mixed second component: the nonzero first component contributes lambda*q.
For any of the five nonbiased banks, an exceptional addition would require
a lattice vector (t,q) with |t|<=b and |q|<=M. The lattice basis is
(-a,-b1),(-c,a), determinant -n. Its inverse gives integer coefficients
with absolute value strictly below1, so both are zero, contradicting q!=0.
For the final biased bank, cancellation means input k=0. Doubling requires
k=+/-2*(b+f0). The rounded reciprocals are monotone. Checking both interval
endpoints proves the coefficients are constant over all f0: (0,1) for
positive k and (a,b1-1) for negative k. In each case the resulting signed
P component has the opposite sign to the doubling requirement.

The rejected [4,5,0,1,2,3] order does double at m=2^73-2^55. The witness
is attainable directly as scalar k=m, whose GLV split is (m,0).
"""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[2]
source=(ROOT/'GLVScalar.cuh').read_text()
entry=(ROOT/'pinning.cu').read_text()

def constant(name,bits):
    raw=re.search(r'const uint'+str(bits)+r'_t '+name+r'\[[0-9]+\]=\{([^}]+)\}',source).group(1)
    return sum(int(v.strip().removesuffix('ULL'),16)<<(bits*i)
               for i,v in enumerate(raw.split(',')))

n=constant('n',64)
a=constant('a1',32)
c=constant('a2',32)
b1=constant('b1',32)
g1=constant('g1',64)
g2=constant('g2',64)
T=int(re.search(r'#define QSB_GT_TOP_CENTER ([0-9]+)u',source).group(1))
M=int('a2a8918ca85bafe22016d0b917e4dd77',16)
bias=(T+1)*(1<<99)-(1<<17)
lambda_value=0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72
assert T==170559769
assert '#define QSB_FOUR_HOT 1' in source and '#define QSB_BIGTBL 1' in source
assert 'return bank < 4u ? 5u - bank : bank - 4u;' in entry
assert a*a+c*b1==n
assert (-a-lambda_value*(-b1))%n==0
assert (-c-lambda_value*a)%n==0
assert n < 1<<256 < 2*n  # one production subtraction reduces every input.

# Prove the magnitude bound from the actual rounded reciprocal constants.
# e_i = round(k*g_i/2^384)-k*d_i/n, for 0<=k<n.
# |e_i| <= 1/2+(n-1)*|g_i*n-d_i*2^384|/(n*2^384).
D=1<<384
delta1=abs(g1*n-a*D)
delta2=abs(g2*n-b1*D)
common=n*D
bound_p=((a+c)*n*(D//2)+(n-1)*(a*delta1+c*delta2))//common
bound_q=((b1+a)*n*(D//2)+(n-1)*(b1*delta1+a*delta2))//common
assert bound_p<=M and bound_q<=M

def coefficients(k):
    assert 0<=k<n
    return ((k*g1+(1<<383))>>384,(k*g2+(1<<383))>>384)

def split(k):
    c1,c2=coefficients(k)
    return k-c1*a-c2*c,c1*b1-c2*a

def digits(m):
    assert 0<=m<=M
    f0=m&((1<<18)-1)
    values=[bias+f0]
    for shift,width in [(18,19),(37,18),(55,18),(73,27)]:
        f=(m>>shift)&((1<<width)-1)
        values.append((2*f-((1<<width)-1))*(1<<(shift-1)))
    values.append((2*(m>>100)-T)*(1<<99))
    assert sum(values)==m
    return values

# Telescoping maximum absolute coefficients for all nonbiased banks.
maximum=[(2**19-1)*2**17,(2**18-1)*2**36,
         (2**18-1)*2**54,(2**27-1)*2**72,T*2**99]
assert sum(maximum)==bias
assert 2*bias>M
assert 4*bias+M<n  # standalone integer equalities cannot wrap modulo n.

# Integer inverse-basis bounds exclude all mixed early exceptions.
assert a*bias+c*M<n
assert b1*bias+a*M<n

# Final mixed doubling: monotonic endpoint checks cover all 262144 f0.
lo=2*bias
hi=lo+2*((1<<18)-1)
assert coefficients(lo)==coefficients(hi)==(0,1)
assert coefficients(n-hi)==coefficients(n-lo)==(a,b1-1)
assert hi<c
assert split(lo)[0]<0 and split(hi)[0]<0
assert split(n-hi)[0]>0 and split(n-lo)[0]>0

# Reachable counterexample that caused the ascending-hot design to be rejected.
witness=(1<<73)-(1<<55)
v=digits(witness)
assert split(witness)==(witness,0)
assert sum(v[i] for i in [4,5,0,1,2])==v[3]

# Directed coefficient checks complement the universal inequalities above.
order=[4,5,3,2,1,0]
vectors={1,M,witness,witness-1,witness+1}
for bit in range(128):
    for offset in [-1,0,1]:
        m=(1<<bit)+offset
        if 0<m<=M:vectors.add(m)
for m in sorted(vectors):
    v=digits(m)
    prefix=v[order[0]]
    for bank in order[1:]:
        assert prefix!=0 and prefix!=v[bank] and prefix!=-v[bank],(m,bank)
        prefix+=v[bank]
    assert prefix==m

print(json.dumps({
    'passed':True,'corrected_order':order,
    'production_glv_sha256':hashlib.sha256(source.encode()).hexdigest(),
    'production_entry_sha256':hashlib.sha256(entry.encode()).hexdigest(),
    'derived_p_bound':hex(bound_p),'derived_q_bound':hex(bound_q),
    'geometry_magnitude_bound':hex(M),'bias':hex(bias),
    'twice_bias_minus_bound':hex(2*bias-M),
    'inverse_basis_i_numerator':str(a*bias+c*M),
    'inverse_basis_j_numerator':str(b1*bias+a*M),
    'inverse_basis_denominator':str(n),
    'positive_endpoint_coefficients':[0,1],
    'negative_endpoint_coefficients':[str(a),str(b1-1)],
    'rejected_ascending_witness':hex(witness),
    'standalone_directed_vectors':len(vectors),
    'gpu_executed':False,
    'scope':'Exact GLV/group identities; inherited field approximations unchanged.'
},indent=2))
