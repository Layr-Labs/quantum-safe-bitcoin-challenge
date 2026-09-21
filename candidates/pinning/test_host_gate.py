#!/usr/bin/env python3
"""Host-only audit of the exact publication-gate algorithm.

Checks, against harness/crypto.py and problems/pinning.json:

  1. Midstate + two-block suffix SHA-256d matches hashlib of the full preimage.
  2. u1 = neg_r_inv * z, Q = u1*G ± u2R matches ecdsa_recover for both recids.
  3. SHA256(compress(Q)) and the leading-zero predicate match the verifier.
  4. pinning.cu actually installs the gate on both hit writers.

No GPU, no PTX. SPDX-License-Identifier: GPL-3.0-only
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "harness"))

import crypto as C  # noqa: E402
import problem as PB  # noqa: E402


def continue_sha256d(midstate, suffix: bytes, seq: int, lt: int,
                     seq_off: int, lt_off: int, total_len: int) -> bytes:
    sl = len(suffix)
    buf = bytearray(128)
    buf[:sl] = suffix
    struct.pack_into("<I", buf, seq_off, seq & 0xFFFFFFFF)
    struct.pack_into("<I", buf, lt_off, lt & 0xFFFFFFFF)
    buf[sl] = 0x80
    nblk = 1 if sl < 56 else 2
    struct.pack_into(">Q", buf, nblk * 64 - 8, total_len * 8)
    h = list(midstate)
    for off in range(0, nblk * 64, 64):
        h = compress_one(h, bytes(buf[off:off + 64]))
    d1 = b"".join(int(x).to_bytes(4, "big") for x in h)
    return hashlib.sha256(d1).digest()


def compress_one(h, block: bytes):
    w = list(struct.unpack(">16I", block))
    for i in range(16, 64):
        s0 = C._rotr(w[i - 15], 7) ^ C._rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
        s1 = C._rotr(w[i - 2], 17) ^ C._rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
        w.append((w[i - 16] + s0 + w[i - 7] + s1) & 0xFFFFFFFF)
    a, b, c, d, e, f, g, hh = h
    for i in range(64):
        S1 = C._rotr(e, 6) ^ C._rotr(e, 11) ^ C._rotr(e, 25)
        ch = (e & f) ^ (~e & g)
        t1 = (hh + S1 + ch + C._K[i] + w[i]) & 0xFFFFFFFF
        S0 = C._rotr(a, 2) ^ C._rotr(a, 13) ^ C._rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (S0 + maj) & 0xFFFFFFFF
        hh, g, f, e, d, c, b, a = g, f, e, (d + t1) & 0xFFFFFFFF, c, b, a, (t1 + t2) & 0xFFFFFFFF
    return [(x + y) & 0xFFFFFFFF for x, y in zip(h, [a, b, c, d, e, f, g, hh])]


def host_zeros(h: bytes) -> int:
    z = 0
    for b in h:
        if b == 0:
            z += 8
            continue
        v, c = b, 0
        while (v & 0x80) == 0:
            c += 1
            v = (v << 1) & 0xFF
        return z + c
    return z


def audit_sha(prob, samples=64):
    prefix = bytes.fromhex(prob["pin_prefix"])
    mid = C.sha256_midstate(prefix)
    suffix0 = bytes.fromhex(prob["suffix"])
    so, lo = prob["seq_offset"], prob["lt_offset"]
    total = prob["total_preimage_len"]
    assert len(prefix) == 155 * 64
    assert len(suffix0) == 75 and so == 31 and lo == 67 and total == 9995
    for i in range(samples):
        seq = 0x80000000 + i * 9973
        lt = 500000000 + i * 104729
        pre = PB.pin_preimage(prob, seq, lt)
        assert len(pre) == total
        expect = C.sha256d(pre)
        got = continue_sha256d(mid, suffix0, seq, lt, so, lo, total)
        assert got == expect, (i, seq, lt, got.hex(), expect.hex())
        assert host_zeros(expect) == C.leading_zero_bits(expect)
    return samples


def audit_bin_layout(prob):
    blob = (ROOT / "problems" / "pinning.bin").read_bytes()
    mid_words = struct.unpack(">8I", blob[:32])
    prefix = bytes.fromhex(prob["pin_prefix"])
    assert tuple(C.sha256_midstate(prefix)) == mid_words
    off = 32
    sl = struct.unpack_from("<I", blob, off)[0]
    off += 4
    suffix = blob[off:off + sl]
    off += sl
    total, so, lo = struct.unpack_from("<III", blob, off)
    off += 12
    nri = int.from_bytes(blob[off:off + 32], "little")
    ux = int.from_bytes(blob[off + 32:off + 64], "little")
    uy = int.from_bytes(blob[off + 64:off + 96], "little")
    assert sl == 75 and total == 9995 and so == 31 and lo == 67
    assert suffix == bytes.fromhex(prob["suffix"])
    assert nri == int(prob["neg_r_inv"], 16)
    assert ux == int(prob["u2r_x"], 16)
    assert uy == int(prob["u2r_y"], 16)
    return True


def audit_recovery(prob):
    seq, lt = 0x80000000, 500000000
    pre = PB.pin_preimage(prob, seq, lt)
    z = int.from_bytes(C.sha256d(pre), "big")
    nri = int(prob["neg_r_inv"], 16)
    u2R = (int(prob["u2r_x"], 16), int(prob["u2r_y"], 16))
    u1 = (nri * z) % C.N
    Qg = C.point_mul(u1, C.G)
    Q0 = C.point_add(Qg, u2R)
    Q1 = C.point_add(Qg, (u2R[0], (C.P - u2R[1]) % C.P))
    E0 = C.ecdsa_recover(int(prob["r"], 16), int(prob["s"], 16), z, 0)
    E1 = C.ecdsa_recover(int(prob["r"], 16), int(prob["s"], 16), z, 1)
    assert Q0 == E0 and Q1 == E1
    h0 = C.sha256(C.compress_pubkey(Q0))
    h1 = C.sha256(C.compress_pubkey(Q1))
    assert h0 == PB.candidate_hash(prob, {"sequence": seq, "locktime": lt}, 0)
    assert h1 == PB.candidate_hash(prob, {"sequence": seq, "locktime": lt}, 1)
    fast = PB.precomputed_points(prob)
    assert PB.recovered_hash_fast(prob, z, 0, nri, *fast) == h0
    assert PB.recovered_hash_fast(prob, z, 1, nri, *fast) == h1
    return True


def audit_source():
    cu = (HERE / "pinning.cu").read_text()
    mathh = (HERE / "GPUMath.h").read_text()
    assert "#define QSB_HOST_GATE 1" in cu
    assert "#define QSB_C31 1" in cu
    assert "#error \"QSB_C31 requires QSB_HOST_GATE" in cu
    assert "qsb_gate_accept" in cu
    assert cu.count("qsb_gate_accept(") >= 3  # definition + two writers
    assert "BN_lebin2bn(pp.neg_r_inv" in cu
    assert "EC_POINT_invert" in cu
    assert mathh.count("QSB_SECOND_FOLD_TAIL") == 7
    assert "sub.u64 t0,t0,k;" in mathh
    assert "add.u64 t0,t0,k;" in mathh
    assert "sub.cc.u32 z0, z0, 0xb73; subc.u32 z1, z1, 3;" in mathh
    # Offset-Y add keeps the t1 correction; dropping it is a ~1/2 error.
    assert "addc.u64 t1,t1,mk;" in mathh
    assert "#define QSB_SAS_Z9SUB_ALL 1" in cu
    assert "#error \"QSB_SAS_Z9SUB_ALL requires QSB_HOST_GATE" in cu
    return True


def main():
    prob = PB.load_problem("pinning")
    sha_n = audit_sha(prob)
    bin_ok = audit_bin_layout(prob)
    rec_ok = audit_recovery(prob)
    src_ok = audit_source()
    result = {
        "test": "exact host publication gate algorithm",
        "sha256d_midstate_samples": sha_n,
        "pinning_bin_layout": bin_ok,
        "recovery_matches_verifier": rec_ok,
        "source_gate_and_c31": src_ok,
        "gpu_executed": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
