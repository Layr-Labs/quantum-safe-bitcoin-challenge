#!/usr/bin/env python3
"""Audit lane175 fast-tail constant-midstate routing without a GPU.

This is source-bound by design: it verifies that only the promoted fast-tail
path reads the sequence midstate from constant memory, while the generic path
and hit identity encoding remain on their inherited contracts. The executable
model checks SHA candidate hashes and SHA-gate hit identities against the old
per-sequence midstate dataflow. It also checks the host-side stream/lifetime
shape that protects constant-symbol uploads from in-flight search kernels.
"""

import ctypes
import ctypes.util
import hashlib
import random
import struct
from pathlib import Path


LT_MIN = 500_000_000
TOTAL_LEN = 9995
SEQ_OFFSET = 31
LT_OFFSET = 67


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


def leading_zero_bits(digest):
    count = 0
    for byte in digest:
        if byte == 0:
            count += 8
            continue
        return count + (8 - byte.bit_length())
    return count


def source_audit():
    source = Path(__file__).with_name("pinning.cu").read_text()
    assert "#ifndef QSB_CONST_MIDSTATE" in source
    assert "#define QSB_CONST_MIDSTATE 1" in source
    assert source.count("__device__ __constant__ uint32_t pin_seq_midstate[8];") == 1
    assert "#define QSB_PK_UNROLL 1" in source

    fast = source[source.index("if (FAST_TAIL) {"):source.index("} else {", source.index("if (FAST_TAIL) {"))]
    assert "#if QSB_CONST_MIDSTATE" in fast
    assert "state[i]=pin_seq_midstate[i];" in fast
    assert "#else\n        for (int i=0;i<8;i++) state[i]=d_midstate[i];" in fast

    generic = source[source.index("/* Copy suffix, set sequence + locktime */"):source.index("/* Second SHA-256")]
    assert "pin_seq_midstate" not in generic
    assert "for(int i=0;i<8;i++) state[i]=d_midstate[i];" in generic

    upload = source[source.index("SHA256_Transform(&ctx,block);"):source.index("/* Search all safe locktimes")]
    assert "#if QSB_CONST_MIDSTATE" in upload
    assert "cudaMemcpyToSymbol(pin_seq_midstate,ctx.h,32);" in upload
    assert "cudaMemcpy(d_mid,ctx.h,32,cudaMemcpyHostToDevice);" in upload
    assert source.count("cudaMemcpyToSymbol(pin_seq_midstate,ctx.h,32);") == 1

    select = source[source.index("const bool fast_tail ="):source.index("if (fast_tail) {", source.index("const bool fast_tail ="))]
    for condition in (
        "single_hash && !easy",
        "pp.suffix_len == 75",
        "pp.seq_offset == 31",
        "pp.lt_offset == 67",
        "pp.total_preimage_len == 9995",
    ):
        assert condition in select

    search = source[source.index("for (uint32_t seq = SEQ_MIN + effective_id;"):source.index("/* Progress every 10 sequences */")]
    assert "cudaMemcpyAsync" not in source
    assert "cudaStreamCreate" not in source
    assert search.index("cudaMemcpyToSymbol(pin_seq_midstate,ctx.h,32);") < search.index("for (uint32_t lt_off = 0;")
    assert search.index("cudaMemcpy(d_hit_cnt, &h_hit, 4, cudaMemcpyHostToDevice);") < search.index("launch_pinning_pipeline<true>")
    assert search.index("cudaMemcpy(d_hit_cnt, &h_hit, 4, cudaMemcpyHostToDevice);") < search.index("launch_pinning_pipeline<false>")
    assert search.index("launch_pinning_pipeline<true>") < search.index("#if QSB_HOST_READBACK")
    assert search.index("launch_pinning_pipeline<false>") < search.index("#if QSB_HOST_READBACK")
    readback = search[search.index("#if QSB_HOST_READBACK"):search.index("#endif", search.index("cudaMemcpy(hits"))]
    assert "cudaMemcpy(hit_report, d_hit_cnt, sizeof(hit_report),\n                                         cudaMemcpyDeviceToHost);" in readback
    assert "cudaDeviceSynchronize();" in readback

    assert source.count("launch_pinning_pipeline<true>") == 1
    assert source.count("launch_pinning_pipeline<false>") == 1
    assert 'd_hit_idx[pos]=((uint32_t)idx)|(ri<<30);' in source
    assert 'd_hit_idx[pos]=((uint32_t)idx)|(ri<<30)|(1u<<31);' in source
    assert 'fprintf(f, "sequence=%u\\nlocktime=%u\\nhash_choice=%d\\nrecid=%d\\n",' in source


def prefix_state(prefix):
    ctx = SHA256Context()
    assert crypto.SHA256_Init(ctypes.byref(ctx)) == 1
    for off in range(0, len(prefix), 64):
        compress(ctx, prefix[off : off + 64])
    return ctx


def sequence_midstate(prefix_ctx, suffix, seq):
    block = bytearray(suffix[:64])
    block[SEQ_OFFSET : SEQ_OFFSET + 4] = seq.to_bytes(4, "little")
    ctx = SHA256Context.from_buffer_copy(prefix_ctx)
    compress(ctx, bytes(block))
    return tuple(ctx.h)


def fast_tail_digest(midstate_words, suffix, lt):
    w0 = int.from_bytes(suffix[64:67], "big") << 8 | (lt & 0xFF)
    w1 = ((lt >> 8) & 0xFF) << 24
    w1 |= ((lt >> 16) & 0xFF) << 16
    w1 |= ((lt >> 24) & 0xFF) << 8
    w1 |= suffix[71]
    w2 = int.from_bytes(suffix[72:75], "big") << 8 | 0x80
    words = [w0, w1, w2] + [0] * 12 + [TOTAL_LEN * 8]

    ctx = SHA256Context()
    assert crypto.SHA256_Init(ctypes.byref(ctx)) == 1
    for i, word in enumerate(midstate_words):
        ctx.h[i] = word
    compress(ctx, struct.pack(">16I", *words))
    return hashlib.sha256(struct.pack(">8I", *ctx.h)).digest()


def generic_digest(prefix, suffix, seq, lt):
    candidate = bytearray(prefix + suffix)
    base = len(prefix)
    candidate[base + SEQ_OFFSET : base + SEQ_OFFSET + 4] = seq.to_bytes(4, "little")
    candidate[base + LT_OFFSET : base + LT_OFFSET + 4] = lt.to_bytes(4, "little")
    return hashlib.sha256(hashlib.sha256(candidate).digest()).digest()


def audit_candidate_hashes(seed):
    rng = random.Random(seed)
    prefix = rng.randbytes(9920)
    suffix = bytearray(rng.randbytes(75))
    pctx = prefix_state(prefix)
    cases = 0
    sequences = [
        0,
        1,
        0x7FFFFFFF,
        0x80000000,
        0x80000001,
        0xFFFFFFFE,
        0xFFFFFFFF,
        rng.getrandbits(32),
        (rng.getrandbits(32) + 1) & 0xFFFFFFFF,
    ]
    locktimes = [0, 1, 255, 256, 65_535, 65_536, LT_MIN, 1_744_599_999, 0xFFFFFFFF]
    locktimes.extend(rng.getrandbits(32) for _ in range(16))
    for seq in sequences:
        mid = sequence_midstate(pctx, suffix, seq)
        const_mid = tuple(mid)
        for lt in locktimes:
            old = fast_tail_digest(mid, suffix, lt)
            new = fast_tail_digest(const_mid, suffix, lt)
            expected = generic_digest(prefix, suffix, seq, lt)
            assert old == new == expected, (seed, seq, lt, old.hex(), new.hex(), expected.hex())
            cases += 1
    return cases


def audit_hit_identity():
    rng = random.Random(175)
    prefix = rng.randbytes(9920)
    suffix = bytearray(rng.randbytes(75))
    pctx = prefix_state(prefix)
    seqs = [0xFFFFFFFE, 0xFFFFFFFF, 0, 1]
    old_hits = []
    new_hits = []
    for seq in seqs:
        mid = sequence_midstate(pctx, suffix, seq)
        const_mid = tuple(mid)
        for off in range(4096):
            lt = LT_MIN + off
            old = fast_tail_digest(mid, suffix, lt)
            new = fast_tail_digest(const_mid, suffix, lt)
            if leading_zero_bits(old) >= 8:
                old_hits.append((seq, lt, old.hex()))
            if leading_zero_bits(new) >= 8:
                new_hits.append((seq, lt, new.hex()))
    assert old_hits == new_hits
    assert old_hits, "synthetic N=8 hit set should be non-empty"
    return len(old_hits)


def main():
    source_audit()
    cases = sum(audit_candidate_hashes(seed) for seed in range(32))
    hits = audit_hit_identity()
    print(
        f"PASS: constant-midstate source route and stream lifetime; "
        f"{cases} actual-tail candidate hashes; {hits} synthetic SHA-gate hit identities"
    )


if __name__ == "__main__":
    main()
