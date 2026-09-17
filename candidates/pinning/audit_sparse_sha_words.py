#!/usr/bin/env python3
"""Audit the sparse-pad SHA-256 specializations in pinning.cu against hashlib.

The ranked fast path hashes four structurally sparse blocks per candidate:

1. prepare tail block : W[0..2] live, W[3..14] = 0,     W[15] = 9995*8
2. prepare outer hash : W[0..7] live, W[8] = 0x80000000, W[9..14] = 0,
                        W[15] = 256
3. finish pubkey hash : W[0..8] live, W[9..14] = 0,     W[15] = 0x108 (x2)

Each specialization folds the constant limbs out of the first sixteen rounds
and the first in-place message-schedule mix. This audit derives those folds
independently here and checks, for every site:

  folded(state, live words) == generic(state, full 16-word block)

where generic() is a transcription of GPUHash.h's _SHA256Transform macro
semantics (rounds 0..15 on the original words, then WMIX before rounds
16..31, exactly as the macro expansion orders them), and generic() itself is
anchored to hashlib on the IV. The tail site is additionally anchored end to
end against hashlib.sha256(sha256(preimage)) over full synthetic preimages,
and the pubkey site against hashlib of the 33-byte compressed key, so a
shared mistake between the CUDA transcription and this model cannot pass.

Run from the repository root: python3 candidates/pinning/audit_sparse_sha_words.py
"""
import hashlib
import random
import struct
from pathlib import Path

M = 0xFFFFFFFF
IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
      0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)

K = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1,
    0x923F82A4, 0xAB1C5ED5, 0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174, 0xE49B69C1, 0xEFBE4786,
    0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147,
    0x06CA6351, 0x14292967, 0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85, 0xA2BFE8A1, 0xA81A664B,
    0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A,
    0x5B9CCA4F, 0x682E6FF3, 0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
]


def ror(x, n):
    return ((x >> n) | (x << (32 - n))) & M


def S0(x):
    return ror(x, 2) ^ ror(x, 13) ^ ror(x, 22)


def S1(x):
    return ror(x, 6) ^ ror(x, 11) ^ ror(x, 25)


def s0(x):
    return ror(x, 7) ^ ror(x, 18) ^ (x >> 3)


def s1(x):
    return ror(x, 17) ^ ror(x, 19) ^ (x >> 10)


def Maj(x, y, z):
    return (x & y) | (z & (x | y))


def Ch(x, y, z):
    return z ^ (x & (y ^ z))


def wmix(w):
    """The in-place 16-word schedule mix, sequential lines as in GPUHash.h.

    Later lines read the already-updated earlier limbs; this ordering is the
    whole reason the sparse substitutions below are not the textbook ones.
    """
    for i in range(16):
        w[i] = (w[i] + s1(w[(i + 14) % 16]) + w[(i + 9) % 16]
                + s0(w[(i + 1) % 16])) & M


def rounds16(st, w, base):
    """Sixteen S2Rounds over w[0..15], with the a..h rotation of SHA256_RND.
    Returns the raw working variables; the Davies-Meyer feed-forward happens
    once, after all 64 rounds, exactly as _SHA256Transform's output adds."""
    a, b, c, d, e, f, g, h = st
    for i in range(16):
        t1 = (h + S1(e) + Ch(e, f, g) + K[base + i] + w[i]) & M
        t2 = (S0(a) + Maj(a, b, c)) & M
        d = (d + t1) & M
        h = (t1 + t2) & M
        a, b, c, d, e, f, g, h = h, a, b, c, d, e, f, g
    return (a, b, c, d, e, f, g, h)


def feedforward(st, state):
    return ((st[0] + state[0]) & M, (st[1] + state[1]) & M,
            (st[2] + state[2]) & M, (st[3] + state[3]) & M,
            (st[4] + state[4]) & M, (st[5] + state[5]) & M,
            (st[6] + state[6]) & M, (st[7] + state[7]) & M)


def generic(state, block):
    """Transcription of _SHA256Transform: RND(0) on the original words, then
    WMIX before each later group, as the macro expansion orders them."""
    w = list(block)
    st = rounds16(state, w, 0)
    wmix(w)
    st = rounds16(st, w, 16)
    wmix(w)
    st = rounds16(st, w, 32)
    wmix(w)
    st = rounds16(st, w, 48)
    return feedforward(st, state)


def rounds16_sparse(state, terms):
    """Sixteen S2Rounds where terms[i] is the message word expression; None
    means the word is identically zero, an int means a compile-time constant
    (it is folded into the round-key add), anything else is a live value."""
    a, b, c, d, e, f, g, h = state
    for i in range(16):
        t = terms[i]
        k = K[i]
        if isinstance(t, int):
            k = (k + t) & M
            t = 0
        elif t is None:
            t = 0
        t1 = (h + S1(e) + Ch(e, f, g) + k + t) & M
        t2 = (S0(a) + Maj(a, b, c)) & M
        d = (d + t1) & M
        h = (t1 + t2) & M
        a, b, c, d, e, f, g, h = h, a, b, c, d, e, f, g
    return (a, b, c, d, e, f, g, h)


def tail3(state, w0, w1, w2):
    """Prepare tail block: W[3..14]=0, W[15]=9995*8 (bit length 9995 bytes)."""
    B = 9995 * 8
    a, b, c, d, e, f, g, h = rounds16_sparse(
        state, [w0, w1, w2] + [None] * 12 + [B])
    w = [0] * 16
    w[0] = (w0 + s0(w1)) & M
    w[1] = (w1 + s1(B) + s0(w2)) & M
    w[2] = (w2 + s1(w[0])) & M
    w[3] = s1(w[1])
    w[4] = s1(w[2])
    w[5] = s1(w[3])
    w[6] = (s1(w[4]) + B) & M
    w[7] = (s1(w[5]) + w[0]) & M
    w[8] = (s1(w[6]) + w[1]) & M
    w[9] = (s1(w[7]) + w[2]) & M
    w[10] = (s1(w[8]) + w[3]) & M
    w[11] = (s1(w[9]) + w[4]) & M
    w[12] = (s1(w[10]) + w[5]) & M
    w[13] = (s1(w[11]) + w[6]) & M
    w[14] = (s1(w[12]) + w[7] + s0(B)) & M
    w[15] = (B + s1(w[13]) + w[8] + s0(w[0])) & M
    st = (a, b, c, d, e, f, g, h)
    st = rounds16(st, w, 16)
    wmix(w)
    st = rounds16(st, w, 32)
    wmix(w)
    st = rounds16(st, w, 48)
    return feedforward(st, state)


def pad32(state, v):
    """Prepare outer hash: 32-byte digest message, W[8]=0x80000000,
    W[9..14]=0, W[15]=256."""
    a, b, c, d, e, f, g, h = rounds16_sparse(
        state, list(v) + [0x80000000] + [None] * 6 + [256])
    w = [0] * 16
    w[0] = (v[0] + s0(v[1])) & M
    w[1] = (v[1] + s1(256) + s0(v[2])) & M
    w[2] = (v[2] + s1(w[0]) + s0(v[3])) & M
    w[3] = (v[3] + s1(w[1]) + s0(v[4])) & M
    w[4] = (v[4] + s1(w[2]) + s0(v[5])) & M
    w[5] = (v[5] + s1(w[3]) + s0(v[6])) & M
    w[6] = (v[6] + s1(w[4]) + 256 + s0(v[7])) & M
    w[7] = (v[7] + s1(w[5]) + w[0] + s0(0x80000000)) & M
    w[8] = (0x80000000 + s1(w[6]) + w[1]) & M
    w[9] = (s1(w[7]) + w[2]) & M
    w[10] = (s1(w[8]) + w[3]) & M
    w[11] = (s1(w[9]) + w[4]) & M
    w[12] = (s1(w[10]) + w[5]) & M
    w[13] = (s1(w[11]) + w[6]) & M
    w[14] = (s1(w[12]) + w[7] + s0(256)) & M
    w[15] = (256 + s1(w[13]) + w[8] + s0(w[0])) & M
    st = (a, b, c, d, e, f, g, h)
    st = rounds16(st, w, 16)
    wmix(w)
    st = rounds16(st, w, 32)
    wmix(w)
    st = rounds16(st, w, 48)
    return feedforward(st, state)


def pk33(state, v):
    """Finish pubkey hash: 33-byte compressed key, W[9..14]=0, W[15]=0x108."""
    a, b, c, d, e, f, g, h = rounds16_sparse(
        state, list(v) + [None] * 6 + [0x108])
    w = [0] * 16
    w[0] = (v[0] + s0(v[1])) & M
    w[1] = (v[1] + s1(0x108) + s0(v[2])) & M
    w[2] = (v[2] + s1(w[0]) + s0(v[3])) & M
    w[3] = (v[3] + s1(w[1]) + s0(v[4])) & M
    w[4] = (v[4] + s1(w[2]) + s0(v[5])) & M
    w[5] = (v[5] + s1(w[3]) + s0(v[6])) & M
    w[6] = (v[6] + s1(w[4]) + 0x108 + s0(v[7])) & M
    w[7] = (v[7] + s1(w[5]) + w[0] + s0(v[8])) & M
    w[8] = (v[8] + s1(w[6]) + w[1]) & M
    w[9] = (s1(w[7]) + w[2]) & M
    w[10] = (s1(w[8]) + w[3]) & M
    w[11] = (s1(w[9]) + w[4]) & M
    w[12] = (s1(w[10]) + w[5]) & M
    w[13] = (s1(w[11]) + w[6]) & M
    w[14] = (s1(w[12]) + w[7] + s0(0x108)) & M
    w[15] = (0x108 + s1(w[13]) + w[8] + s0(w[0])) & M
    st = (a, b, c, d, e, f, g, h)
    st = rounds16(st, w, 16)
    wmix(w)
    st = rounds16(st, w, 32)
    wmix(w)
    st = rounds16(st, w, 48)
    return feedforward(st, state)


def byte_perm(a, b, sel):
    """NVIDIA __byte_perm: byte i of the result is source byte sel_nibble(i),
    where nibble 0..3 select a's bytes 0..3 (little-endian) and 4..7 select
    b's bytes 0..3."""
    out = 0
    for i in range(4):
        nib = (sel >> (4 * i)) & 0xF
        src = a if nib < 4 else b
        out |= ((src >> (8 * (nib & 3))) & 0xFF) << (8 * i)
    return out & M


def pubkey_words(x_le_limbs, parity):
    """The kernel's nine pb words: prefix byte then the big-endian x limbs."""
    x0 = x_le_limbs[0] & M
    x1 = (x_le_limbs[0] >> 32) & M
    x2 = x_le_limbs[1] & M
    x3 = (x_le_limbs[1] >> 32) & M
    x4 = x_le_limbs[2] & M
    x5 = (x_le_limbs[2] >> 32) & M
    x6 = x_le_limbs[3] & M
    x7 = (x_le_limbs[3] >> 32) & M
    pb = [0] * 9
    pb[0] = byte_perm(x7, 0x2 + parity, 0x4321)
    pb[1] = byte_perm(x7, x6, 0x0765)
    pb[2] = byte_perm(x6, x5, 0x0765)
    pb[3] = byte_perm(x5, x4, 0x0765)
    pb[4] = byte_perm(x4, x3, 0x0765)
    pb[5] = byte_perm(x3, x2, 0x0765)
    pb[6] = byte_perm(x2, x1, 0x0765)
    pb[7] = byte_perm(x1, x0, 0x0765)
    pb[8] = byte_perm(x0, 0x80, 0x0456)
    return pb


def tail_words(suffix, lt):
    """The kernel's three live tail-block words from the suffix template."""
    w0 = (int.from_bytes(suffix[64:67], "big") << 8) | (lt & 0xFF)
    w1 = (((lt & 0xFF00) << 16) | (lt & 0xFF0000)
          | ((lt >> 16) & 0xFF00) | suffix[71])
    w2 = (int.from_bytes(suffix[72:75], "big") << 8) | 0x80
    return w0, w1, w2


def words_to_block(words):
    return struct.pack(">16I", *words)


def audit_generic_anchor(rng, cases):
    for _ in range(cases):
        n = rng.randrange(1, 56)  # one padded block after the message
        msg = rng.randbytes(n)
        block = bytearray(64)
        block[:n] = msg
        block[n] = 0x80
        block[62] = (n * 8) >> 8
        block[63] = (n * 8) & 0xFF
        w = list(struct.unpack(">16I", bytes(block)))
        got = generic(IV, w)
        assert struct.pack(">8I", *got) == hashlib.sha256(msg).digest()
    return cases


def audit_tail3(rng, cases):
    checked = 0
    edge = [0, 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0x00FFFFFF, 0xFF000000]
    for i in range(cases):
        state = tuple(rng.getrandbits(32) for _ in range(8))
        if i < len(edge):
            lt = edge[i]
        else:
            lt = rng.getrandbits(32)
        w0 = rng.getrandbits(32)
        w1 = rng.getrandbits(32)
        w2 = rng.getrandbits(32) & 0xFFFFFF00  # low byte is the 0x80 pad
        block = [w0, w1, w2] + [0] * 12 + [9995 * 8]
        assert tail3(state, w0, w1, w2) == generic(state, block)
        checked += 1
    return checked


def audit_pad32(rng, cases):
    checked = 0
    for i in range(cases):
        if i < cases // 2:
            # hashlib anchor: a 32-byte message is exactly this block shape.
            msg = rng.randbytes(32)
            v = list(struct.unpack(">8I", msg))
            state = IV
        else:
            v = [rng.getrandbits(32) for _ in range(8)]
            state = tuple(rng.getrandbits(32) for _ in range(8))
        block = list(v) + [0x80000000] + [0] * 6 + [256]
        assert pad32(state, v) == generic(state, block)
        if state == IV:
            digest = struct.pack(">8I", *pad32(state, v))
            assert digest == hashlib.sha256(msg).digest()
        checked += 1
    return checked


def audit_pk33(rng, cases):
    checked = 0
    for i in range(cases):
        if i < cases // 2:
            # hashlib anchor through the kernel's own packing of the key.
            x = rng.randbytes(32)
            parity = rng.getrandbits(1)
            limbs = list(struct.unpack("<4Q", x))
            pb = pubkey_words(limbs, parity)
            state = IV
        else:
            limbs = [rng.getrandbits(64) for _ in range(4)]
            parity = rng.getrandbits(1)
            pb = pubkey_words(limbs, parity)
            state = tuple(rng.getrandbits(32) for _ in range(8))
        block = pb + [0] * 6 + [0x108]
        assert pk33(state, pb) == generic(state, block)
        if state == IV:
            key = bytes([0x2 + parity]) + struct.pack("<4Q", *limbs)[::-1]
            digest = struct.pack(">8I", *pk33(state, pb))
            assert digest == hashlib.sha256(key).digest(), (key.hex(), digest.hex())
        checked += 1
    return checked


def audit_end_to_end_tail(rng, cases):
    """Full preimage double-SHA through midstate + tail3 + pad32, against
    hashlib over the assembled preimage bytes."""
    checked = 0
    lts = [0, 1, 255, 256, 65535, 500000000, 1744599999, 0xFFFFFFFF]
    for i in range(cases):
        prefix = rng.randbytes(9920)
        suffix = bytearray(rng.randbytes(75))
        seq = rng.getrandbits(32)
        suffix[31:35] = seq.to_bytes(4, "little")
        state = IV
        pre = prefix + bytes(suffix)
        for off in range(0, 9984, 64):
            state = generic(state, list(struct.unpack(">16I", pre[off:off + 64])))
        lt = lts[i % len(lts)] if i < len(lts) else rng.getrandbits(32)
        w0, w1, w2 = tail_words(suffix, lt)
        inner = tail3(state, w0, w1, w2)
        z_words = pad32(IV, list(inner))
        suffix[67:71] = lt.to_bytes(4, "little")
        expected = hashlib.sha256(
            hashlib.sha256(prefix + suffix).digest()).digest()
        got = struct.pack(">8I", *z_words)
        assert got == expected, (seq, lt, got.hex(), expected.hex())
        checked += 1
    return checked


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()
    for needle in (
        "_SHA256TransformTail3(state, w0, w1, w2);",
        "_SHA256TransformPad32(s2, state);",
        "_SHA256TransformPk33(hs,pb);",
    ):
        assert needle in source, needle
    # The generic path stays generic for non-fast layouts.
    assert "uint8_t buf[192];" in source
    # Folded sites must not also hand-build the sparse limbs they removed.
    fast = source[source.index("if (FAST_TAIL) {"):
                  source.index("} else {", source.index("if (FAST_TAIL) {"))]
    assert "uint32_t blk[16]" not in fast
    assert "b2[9]=0" not in source[source.index("/* Second SHA-256"):
                                   source.index("/* Scalar from the SHA-256")]
    finish = source[source.index("/* Check both pubkeys"):
                    source.index("template<bool FAST_TAIL>")]
    assert "pb[9]=0;" not in finish
    # Cache-policy hints present at the intended sites. The merged tree routes
    # all pipeline state/tree traffic through the qsb_*_v2/qsb_*_u64 helpers
    # inherited from exact 1a, whose PTX emits .cs (evict-first) stores/loads;
    # this bundle turns the flag on where the promoted crown left it off. The
    # fixed-base table loads stay direct __ldcg (L2-only) reads.
    assert "ulonglong2 x0=__ldcg(&tx[0])" in source
    assert "#define QSB_STREAM 1" in source
    assert source.count("qsb_st_v2(&saved[") == 10
    assert source.count("qsb_ld_v2(&saved[") == 12
    assert "st.global.cs.u64" in source         # qsb_st_u64 checkpoint helper
    assert "ld.global.cs.u64" in source         # qsb_ld_u64 checkpoint helper
    assert "qsb_st_u64(&checkpoint[" in source
    assert "qsb_ld_u64(&checkpoint[" in source


def main():
    rng = random.Random(0x5BAD1DEA)
    anchor = audit_generic_anchor(rng, 200)
    t3 = audit_tail3(rng, 4000)
    p32 = audit_pad32(rng, 4000)
    p33 = audit_pk33(rng, 4000)
    e2e = audit_end_to_end_tail(rng, 40)
    audit_source()
    print(f"PASS: sparse-pad SHA specializations; hashlib anchor={anchor}; "
          f"tail3={t3}; pad32={p32}; pk33={p33}; end-to-end preimages={e2e}; "
          f"source bindings ok")


if __name__ == "__main__":
    main()
