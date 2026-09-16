"""Check a proposed one-block SHA tail against hashlib, without using the GPU.

This is a diagnostic experiment, not a replacement for the benchmark verifier.
It tests the byte packing and host-midstate scheme before changing the kernel.
Run from the repository root: python3 candidates/pinning/check_tail_words.py
"""
import ctypes
import ctypes.util
import hashlib
import random
import struct


class SHA256Context(ctypes.Structure):
    _fields_ = [
        ("h", ctypes.c_uint32 * 8),
        ("Nl", ctypes.c_uint32),
        ("Nh", ctypes.c_uint32),
        ("data", ctypes.c_uint32 * 16),
        ("num", ctypes.c_uint),
        ("md_len", ctypes.c_uint),
    ]


crypto = ctypes.CDLL(ctypes.util.find_library("crypto"))
crypto.SHA256_Init.argtypes = [ctypes.POINTER(SHA256Context)]
crypto.SHA256_Transform.argtypes = [ctypes.POINTER(SHA256Context), ctypes.c_void_p]
crypto.SHA256_Transform.restype = None


def compress(ctx, block):
    assert len(block) == 64
    crypto.SHA256_Transform(ctypes.byref(ctx), ctypes.create_string_buffer(block))


def check(seed):
    rng = random.Random(seed)
    prefix = rng.randbytes(9920)
    suffix = bytearray(rng.randbytes(75))
    prefix_state = SHA256Context()
    assert crypto.SHA256_Init(ctypes.byref(prefix_state)) == 1
    for off in range(0, len(prefix), 64):
        compress(prefix_state, prefix[off:off + 64])

    cases = 0
    seqs = [0, 1, 0x80000000, 0xFFFFFFFF, rng.getrandbits(32)]
    locktimes = [0, 1, 255, 256, 65535, 65536, 500000000, 1744599999,
                 0xFFFFFFFF] + [rng.getrandbits(32) for _ in range(20)]
    for seq in seqs:
        suffix[31:35] = seq.to_bytes(4, "little")
        state = SHA256Context.from_buffer_copy(prefix_state)
        compress(state, bytes(suffix[:64]))
        for lt in locktimes:
            # Proposed static GPU message words. Locktime crosses words 0/1.
            w0 = int.from_bytes(suffix[64:67], "big") << 8 | (lt & 255)
            w1 = ((lt >> 8) & 255) << 24 | ((lt >> 16) & 255) << 16
            w1 |= ((lt >> 24) & 255) << 8 | suffix[71]
            w2 = int.from_bytes(suffix[72:75], "big") << 8 | 0x80
            words = [w0, w1, w2] + [0] * 12 + [9995 * 8]
            ctx = SHA256Context.from_buffer_copy(state)
            compress(ctx, struct.pack(">16I", *words))
            got = hashlib.sha256(struct.pack(">8I", *ctx.h)).digest()
            suffix[67:71] = lt.to_bytes(4, "little")
            expected = hashlib.sha256(hashlib.sha256(prefix + suffix).digest()).digest()
            assert got == expected, (seed, seq, lt, got.hex(), expected.hex())
            cases += 1
    return cases


if __name__ == "__main__":
    count = sum(check(seed) for seed in range(16))
    print(f"PASS: {count} SHA256d comparisons across 16 synthetic prefixes; "
          "sequence and locktime boundaries included")
