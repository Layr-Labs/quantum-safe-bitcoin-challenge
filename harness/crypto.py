"""
Standalone secp256k1 + SHA-256 primitives for the QSB grinding benchmark.

Self-contained on purpose: this module must NOT import anything from the QSB
codebase. It implements exactly the operations the grinding workload needs:

  - ECDSA public-key *recovery*  Q = u1*G + u2*R   (u1 = -z*r^-1, u2 = s*r^-1)
  - compressed pubkey encoding
  - SHA-256 / SHA-256d, plus a pure-python intermediate state (after N full blocks)
  - the relaxed validity gate: N-leading-zero-bits

Everything is pure python (no external deps) so the verifier runs anywhere,
including CPU-only CI.
"""
from __future__ import annotations
import hashlib
import struct

# ---- secp256k1 domain parameters ----
P  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
A  = 0
B  = 7
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
G  = (GX, GY)


# ---- field / group arithmetic ----
def modinv(a: int, m: int) -> int:
    return pow(a % m, m - 2, m)


def point_add(p1, p2):
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and (y1 + y2) % P == 0:
        return None
    if p1 == p2:
        lam = (3 * x1 * x1 + A) * modinv(2 * y1, P) % P
    else:
        lam = (y2 - y1) * modinv((x2 - x1) % P, P) % P
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return (x3, y3)


def point_mul(k: int, point=G):
    k %= N
    result = None
    addend = point
    while k:
        if k & 1:
            result = point_add(result, addend)
        addend = point_add(addend, addend)
        k >>= 1
    return result


def y_from_x(x: int, want_odd: bool):
    """Return the y on the curve for x with the requested parity, or None if x
    is not a valid x-coordinate (x^3+7 not a quadratic residue)."""
    if x >= P:
        return None
    alpha = (pow(x, 3, P) + B) % P
    beta = pow(alpha, (P + 1) // 4, P)     # P % 4 == 3, so this is the sqrt
    if (beta * beta) % P != alpha:
        return None                        # x not on curve
    y = beta if (beta & 1) == want_odd else (P - beta)
    return y


def on_curve(x: int) -> bool:
    """True iff x (mod-p reduced) is a valid secp256k1 x-coordinate.

    Computed via the Euler-criterion square root."""
    if x >= P:
        return False
    alpha = (pow(x, 3, P) + B) % P
    beta = pow(alpha, (P + 1) // 4, P)
    return (beta * beta) % P == alpha


# ---- ECDSA public-key recovery (the per-candidate core of the workload) ----
def ecdsa_recover(r: int, s: int, z: int, recid: int):
    """Recover Q = u1*G + u2*R with R.x = r and R.y parity = recid (0=even).

    Returns the point or None if r is not a valid x-coordinate. Uses the
    recovery form u1 = (-z*r^-1) mod N, u2 = (s*r^-1) mod N."""
    y = y_from_x(r, want_odd=bool(recid & 1))
    if y is None:
        return None
    R = (r, y)
    rinv = modinv(r, N)
    u1 = (-z * rinv) % N
    u2 = (s * rinv) % N
    return point_add(point_mul(u1, G), point_mul(u2, R))


def compress_pubkey(point) -> bytes:
    x, y = point
    return bytes([0x02 + (y & 1)]) + x.to_bytes(32, "big")


# ---- hashing ----
def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def sha256d(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


# --- pure-python SHA-256 compression, only to produce the prefix state ---
_K = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
]
_H0 = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]


def _rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF


def sha256_midstate(data: bytes):
    """SHA-256 intermediate state (8 uint32) after absorbing `data`, whose
    length MUST be a multiple of 64 bytes (no padding applied). This is the
    state a grinder can resume from when hashing the variable tail."""
    assert len(data) % 64 == 0, "midstate input must be block-aligned"
    h = list(_H0)
    for off in range(0, len(data), 64):
        w = list(struct.unpack(">16I", data[off:off + 64]))
        for i in range(16, 64):
            s0 = _rotr(w[i-15], 7) ^ _rotr(w[i-15], 18) ^ (w[i-15] >> 3)
            s1 = _rotr(w[i-2], 17) ^ _rotr(w[i-2], 19) ^ (w[i-2] >> 10)
            w.append((w[i-16] + s0 + w[i-7] + s1) & 0xFFFFFFFF)
        a, b, c, d, e, f, g, hh = h
        for i in range(64):
            S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
            ch = (e & f) ^ (~e & g)
            t1 = (hh + S1 + ch + _K[i] + w[i]) & 0xFFFFFFFF
            S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            t2 = (S0 + maj) & 0xFFFFFFFF
            hh, g, f, e, d, c, b, a = g, f, e, (d + t1) & 0xFFFFFFFF, c, b, a, (t1 + t2) & 0xFFFFFFFF
        h = [(x + y) & 0xFFFFFFFF for x, y in zip(h, [a, b, c, d, e, f, g, hh])]
    return h


# ---- the relaxed validity gate ----
def leading_zero_bits(h: bytes) -> int:
    """Number of leading zero *bits* in the 32-byte big-endian hash."""
    v = int.from_bytes(h, "big")
    if v == 0:
        return len(h) * 8
    return len(h) * 8 - v.bit_length()


def is_hit(h: bytes, n_zero_bits: int) -> bool:
    """The benchmark hit condition: N leading zero bits of h. (No follow-up
    EC-point check on h; `on_curve` is kept only for recovery/self-test.)"""
    return leading_zero_bits(h) >= n_zero_bits


if __name__ == "__main__":
    # self-test: recovery round-trip + midstate vs hashlib
    d = 0x1234567890abcdef
    Q = point_mul(d, G)
    k = 0xdeadbeef
    R = point_mul(k, G)
    r = R[0] % N
    z = int.from_bytes(sha256(b"benchmark self test"), "big")
    s = (modinv(k, N) * (z + r * d)) % N
    ok = False
    for recid in (0, 1):
        for rr in (r, r + N):
            if rr >= P:
                continue
            cand = ecdsa_recover(rr, s, z, recid)
            if cand == Q:
                ok = True
    assert ok, "recovery round-trip failed"

    # midstate validation: pad a message to a block multiple per SHA-256, then
    # the full-block midstate must equal hashlib's final digest words.
    import os
    m = os.urandom(50)
    ml = len(m) * 8
    padded = m + b"\x80"
    while len(padded) % 64 != 56:
        padded += b"\x00"
    padded += struct.pack(">Q", ml)
    assert sha256_midstate(padded) == list(struct.unpack(">8I", hashlib.sha256(m).digest())), "midstate mismatch"

    assert compress_pubkey(Q)[0] in (2, 3)
    assert on_curve(GX) and not on_curve(P - 1)
    assert leading_zero_bits(bytes([0x00, 0x00, 0x0f]) + b"\x00" * 29) == 20
    print("crypto self-test: OK  (recovery round-trip, compress, on_curve, midstate, lz)")
