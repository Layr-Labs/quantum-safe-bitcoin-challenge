# Independent Python Jacobian reference; no native or CUDA execution.

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

G = (
    0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
)

def jac_double(Q):
    if Q is None or Q[1] == 0:
        return None
    X, Y, Z = Q
    yy = Y * Y % P
    S = 4 * X * yy % P
    M = 3 * X * X % P
    X3 = (M * M - 2 * S) % P
    Y3 = (M * (S - X3) - 8 * yy * yy) % P
    Z3 = 2 * Y * Z % P
    return X3, Y3, Z3

def jac_add(Q, R):
    if Q is None:
        return R
    if R is None:
        return Q
    X1, Y1, Z1 = Q
    X2, Y2, Z2 = R
    Z1Z1 = Z1 * Z1 % P
    Z2Z2 = Z2 * Z2 % P
    U1 = X1 * Z2Z2 % P
    U2 = X2 * Z1Z1 % P
    S1 = Y1 * Z2 % P * Z2Z2 % P
    S2 = Y2 * Z1 % P * Z1Z1 % P
    if U1 == U2:
        return jac_double(Q) if S1 == S2 else None
    H = (U2 - U1) % P
    I = (2 * H) ** 2 % P
    J = H * I % P
    r = 2 * (S2 - S1) % P
    V = U1 * I % P
    X3 = (r * r - J - 2 * V) % P
    Y3 = (r * (V - X3) - 2 * S1 * J) % P
    Z3 = ((Z1 + Z2) ** 2 - Z1Z1 - Z2Z2) * H % P
    return X3, Y3, Z3

def jac_mul(k, A=G):
    k %= N
    if k == 0:
        return None
    Q = None
    R = (A[0], A[1], 1)
    while k:
        if k & 1:
            Q = jac_add(Q, R)
        R = jac_double(R)
        k >>= 1
    return Q

def to_affine(Q):
    if Q is None:
        return None
    X, Y, Z = Q
    zi = pow(Z, P - 2, P)
    zi2 = zi * zi % P
    return X * zi2 % P, Y * zi2 % P * zi % P
