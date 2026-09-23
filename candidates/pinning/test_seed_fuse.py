"""Check the seed sign rewrite, including both deferred-Y conventions.

This checks algebra over secp256k1's field; it does not execute GPU PTX.
"""
from random import Random

p = (1 << 256) - (1 << 32) - 977
rng = Random(20260923)
edge = (0, 1, 2, p//2, p-2, p-1, p, (1 << 256)-1)

def check(x1, y1, x2, y2):
    P = (x2-x1) % p
    if P == 0:
        return
    R = (y2-y1) % p
    pp = P*P % p
    ppp = P*pp % p
    q = x1*pp % p
    x = (R*R-ppp-2*q) % p
    pn = (-P) % p
    rn = (-R) % p
    pppn = pn*pp % p
    xn = (rn*rn+pppn-2*q) % p
    assert x == xn and pppn == (-ppp) % p
    for negative_y in (False, True):
        core = R*((x-q) if negative_y else (q-x)) % p
        coren = rn*((xn-q) if negative_y else (q-xn)) % p
        assert coren == (-core) % p
        # The current XYZZ chain carries a deferred affine Y anchor. Under
        # either convention, both the ordinate numerator and denominator flip.
        old_y = ((-core-y1*ppp) if negative_y else (core-y1*ppp)) % p
        new_y = ((-coren-y1*pppn) if negative_y else (coren-y1*pppn)) % p
        assert new_y == (-old_y) % p
        assert new_y*pow(pppn, p-2, p) % p == old_y*pow(ppp, p-2, p) % p

for x1 in edge:
    for x2 in edge:
        for y1 in edge:
            for y2 in edge:
                check(x1%p, y1%p, x2%p, y2%p)
for _ in range(10000):
    check(*(rng.randrange(p) for _ in range(4)))
print('seed-fuse algebra: 4096 edge tuples and 10000 random tuples passed')
