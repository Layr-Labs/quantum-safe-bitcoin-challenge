"""CPU algebra check of the homogeneous recovery transformation; not a CUDA test."""
import random

P = 2**256 - 2**32 - 977
G = (55066263022277343669578718895168534326250603453777594175500187360389116729240,
     32670510020758816978083085130507043184471273380659243275938904335757337482424)


def affine(a, b):
    if a is None:
        return b
    if b is None:
        return a
    x, y = a
    u, v = b
    if x == u and (y + v) % P == 0:
        return None
    m = ((3*x*x) * pow(2*y, -1, P) if a == b
         else (v-y) * pow(u-x, -1, P)) % P
    z = (m*m-x-u) % P
    return z, (m*(x-z)-y) % P


def mul(k):
    a, b = None, G
    while k:
        if k & 1:
            a = affine(a, b)
        b = affine(b, b)
        k >>= 1
    return a


def mixed(a, b):
    x, y, z = a
    bx, by = b
    u, v = (by*z-y) % P, (bx*z-x) % P
    t = (u*u*z-v*v*v-2*v*v*x) % P
    return v*t % P, (u*(v*v*x-t)-v*v*v*y) % P, v*v*v*z % P


def norm(a):
    x, y, z = a
    zi = pow(z, -1, P)
    return x*zi % P, y*zi % P


def main():
    rng = random.Random(4319)
    for _ in range(128):
        # Same group operations as the production lookup, but with small
        # independent scalar multiples rather than its large CUDA table.
        chunks = [mul(rng.randrange(1, 1 << 20)) for _ in range(16)]
        q = (*chunks[0], 1)
        expected = chunks[0]
        for point in chunks[1:]:
            q = mixed(q, point)
            expected = affine(expected, point)
        assert norm(q) == expected
        r = mul(rng.randrange(1, 1 << 24))
        twice_r = affine(r, r)
        neg_twice_r = twice_r[0], (-twice_r[1]) % P
        q1 = mixed(q, r)
        q2 = mixed(q1, neg_twice_r)
        old1 = mixed((*norm(q), 1), r)
        old2 = mixed(old1, neg_twice_r)
        inv = pow(q1[2]*q2[2] % P, -1, P)
        inv1, inv2 = inv*q2[2] % P, inv*q1[2] % P
        result1 = q1[0]*inv1 % P, q1[1]*inv1 % P
        result2 = q2[0]*inv2 % P, q2[1]*inv2 % P
        assert result1 == norm(old1) == affine(expected, r)
        assert result2 == norm(old2) == affine(expected, (r[0], -r[1] % P))
    print('PASS: 128 deterministic homogeneous recovery cases, both recids')


if __name__ == '__main__':
    main()
