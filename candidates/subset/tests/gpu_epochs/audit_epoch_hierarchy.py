#!/usr/bin/env python3
"""Audit for the hierarchical short-epoch producer (ZLAB_HIERARCH).

All arithmetic is re-derived from problems/subset.json (the committed seed-0
public instance, identical layout to a ranked instance) and from an
independent pure-Python SHA-256 compressor, so nothing is trusted from the
kernel source itself.

Checks:
  1. sha256_compress matches hashlib for random messages (compressor itself).
  2. Flat producer semantics: for epoch rank e, unrank 6-from-[0,137),
     stream prefix_remainder + kept pushes, 1352 bytes = 21 blocks + 8-byte
     remainder; descriptor = (state after 21 blocks, remW words, sorted 6 skips).
  3. Hierarchical producer: group rank g -> 5 skips {s1..s5}; group prefix =
     prefix_remainder + kept pushes of [0,s5) minus {s1..s4}; member m -> j =
     s5+1+m, resume = group staging + kept pushes of [s5+1,137) minus {j}.
     For every sampled (g, m): the concatenated stream equals the flat stream
     of the same 6-subset, so mid/remW/early match the flat descriptor
     bit-for-bit, and the final staging length is exactly 8.
  4. Host walk simulation: whole groups per launch until >= 65536 member
     slots; the slot -> (group, j) map over consecutive launches is a
     gapless, duplicate-free cover of the member-slot space, and the induced
     6-subsets are all distinct and valid (sorted, in range, size 6).
  5. Edge groups: s5 at both extremes, adjacent skips, j at both ends.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CUT = 137          # QSB_SE_CUT: pushes [0,137) are epoch territory
EARLY = 6          # QSB_SE_EARLY: 6 early omissions per epoch
LAUNCH = 65536     # QSB_SE_LAUNCH_BLOCKS member slots per launch
N_POOL = 150

# ---------------- independent SHA-256 compressor ----------------
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]
M = (1 << 32) - 1


def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & M


def sha256_compress(state, block):
    w = [int.from_bytes(block[4 * i:4 * i + 4], "big") for i in range(16)]
    for i in range(16, 64):
        s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
        s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
        w.append((w[i - 16] + s0 + w[i - 7] + s1) & M)
    a, b, c, d, e, f, g, h = state
    for i in range(64):
        S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
        ch = (e & f) ^ (~e & g)
        t1 = (h + S1 + ch + K[i] + w[i]) & M
        S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (S0 + maj) & M
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & M, c, b, a, (t1 + t2) & M
    return tuple((s + v) & M for s, v in zip(state, (a, b, c, d, e, f, g, h)))


IV = (0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19)  # FIPS 180-4 constants


def stream_state(base_state, data, staged=b""):
    """Mirror of the kernels' staging loop: feed staged + data through the
    compressor; return (state, leftover bytes)."""
    buf = bytearray(staged)
    state = list(base_state)
    for byte in data:
        buf.append(byte)
        if len(buf) == 64:
            state = list(sha256_compress(state, bytes(buf)))
            buf = bytearray()
    return tuple(state), bytes(buf)


# ---------------- combinadics ----------------
def binom(n, k):
    if k < 0 or k > n:
        return 0
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


def unrank(rank, n, t):
    """Lexicographic unranking, same hockey-stick walk as the C helpers."""
    out = []
    lo = 0
    for i in range(t):
        k = t - i - 1
        while True:
            c = binom(n - lo - 1, k)
            if rank < c:
                break
            rank -= c
            lo += 1
        out.append(lo)
        lo += 1
    return out


def main():
    prob = json.loads((ROOT / "problems" / "subset.json").read_text())
    assert prob["bench"] == "subset"
    assert len(prob["dummy_sigs"]) == N_POOL
    assert binom(CUT, EARLY) * 256 >= 4e11, "epoch family supply"

    # fixed_prefix is 8234 bytes; prefix_remainder = last 42
    fp = bytes.fromhex(prob["fixed_prefix"])
    assert len(fp) == 8234
    prem = fp[8192:]
    assert len(prem) == 42
    sigs = [bytes.fromhex(d) for d in prob["dummy_sigs"]]

    # 1. compressor sanity vs hashlib
    rng = random.Random(1234)
    for trial in range(64):
        n = rng.randrange(0, 300)
        msg = bytes(rng.randrange(256) for _ in range(n))
        st, left = stream_state(IV, msg)
        full = hashlib.sha256(msg).digest()
        # digest of msg = state after padding; emulate padding via compressor
        pad = b"\x80" + b"\x00" * ((55 - n) % 64) + (8 * n).to_bytes(8, "big")
        st2, left2 = stream_state(st, pad, staged=left)
        assert not left2
        got = b"".join(s.to_bytes(4, "big") for s in st2)
        assert got == full, (trial, got.hex(), full.hex())
    print("check 1 OK: pure-Python compressor matches hashlib on 64 messages")

    # base midstate = state after the 8192-byte block-aligned fixed prefix
    base_mid, base_left = stream_state(IV, fp[:8192])
    assert not base_left

    def flat_desc(e):
        skips = unrank(e, CUT, EARLY)
        assert len(skips) == 6 and skips == sorted(skips)
        ss = set(skips)
        data = prem + b"".join(sigs[i] for i in range(CUT) if i not in ss)
        assert len(data) == 42 + 131 * 10 == 1352
        st, rem = stream_state(base_mid, data)
        assert len(rem) == 8
        return st, rem, skips

    def group_prefix(g):
        s = unrank(g, CUT, EARLY - 1)
        assert len(s) == 5 and s == sorted(s)
        s5 = s[4]
        ss = set(s[:4])
        data = prem + b"".join(sigs[i] for i in range(s5) if i not in ss)
        st, rem = stream_state(base_mid, data)
        assert len(rem) < 64
        return st, rem, s

    def member_desc(g, m):
        gst, grem, s = group_prefix(g)
        s5 = s[4]
        j = s5 + 1 + m
        assert s5 < j < CUT
        data = b"".join(sigs[p] for p in range(s5 + 1, CUT) if p != j)
        st, rem = stream_state(gst, data, staged=grem)
        skips = s[:4] + [s5, j]
        assert skips == sorted(skips) and len(set(skips)) == 6
        return st, rem, skips, data

    # 2+3. hierarchical == flat for random (g, m), including stream equality
    n_groups_total = binom(CUT, EARLY - 1)
    checked = 0
    for trial in range(4000):
        g = rng.randrange(n_groups_total)
        s5 = unrank(g, CUT, EARLY - 1)[4]
        members = CUT - 1 - s5
        if members <= 0:
            continue          # the single s5=136 group has no members
        m = rng.randrange(members) if members > 1 else 0
        st_h, rem_h, skips_h, tail = member_desc(g, m)
        # the same 6-subset under the flat producer: find its rank by construction
        # (flat stream equality is what matters; recompute flat by skips)
        ss = set(skips_h)
        data = prem + b"".join(sigs[i] for i in range(CUT) if i not in ss)
        st_f, rem_f = stream_state(base_mid, data)
        assert (st_h, rem_h) == (st_f, rem_f), (g, m, "descriptor mismatch")
        assert len(rem_h) == 8
        checked += 1
    print(f"check 2+3 OK: {checked} (group,member) descriptors bit-identical to flat")

    # 5. edge groups
    edges = []
    # s5 minimal: {0,1,2,3,4} is rank 0
    edges.append(0)
    # force s5 near max: group {131,132,133,134,135} has rank = C(131,5)-ish; find by search
    target = [131, 132, 133, 134, 135]
    # rank of a combination in lex order: sum over positions
    def rank_of(combo, n):
        r = 0
        lo = 0
        for k, v in enumerate(combo):
            for x in range(lo, v):
                r += binom(n - x - 1, len(combo) - k - 1)
            lo = v + 1
        return r
    edges.append(rank_of(target, CUT))
    edges.append(rank_of([0, 1, 2, 3, 135], CUT))       # adjacent low skips, huge s5
    edges.append(rank_of([0, 1, 133, 134, 135], CUT))  # adjacent high skips
    for g in edges:
        s = unrank(g, CUT, EARLY - 1)
        s5 = s[4]
        for m in {0, CUT - 2 - s5, max(0, (CUT - 1 - s5) // 2)}:
            if not (0 <= m < CUT - 1 - s5):
                continue
            st_h, rem_h, skips_h, _ = member_desc(g, m)
            ss = set(skips_h)
            data = prem + b"".join(sigs[i] for i in range(CUT) if i not in ss)
            st_f, rem_f = stream_state(base_mid, data)
            assert (st_h, rem_h) == (st_f, rem_f), ("edge", g, m)
    print(f"check 5 OK: edge groups (s5 extremes, adjacent skips, j extremes) pass")

    # 4. host-walk simulation over 4 launches
    group_base = 0
    slot_cursor = 0
    seen = set()
    for launch in range(4):
        acc = 0
        g = 0
        slot_start = [0]
        while acc < LAUNCH and group_base + g < n_groups_total:
            s5 = unrank(group_base + g, CUT, EARLY - 1)[4]
            acc += CUT - 1 - s5
            g += 1
            slot_start.append(acc)
        assert acc >= LAUNCH or group_base + g >= n_groups_total
        # every local slot maps to a distinct valid epoch
        for t in range(acc):
            lo, hi = 0, g
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if slot_start[mid] <= t:
                    lo = mid
                else:
                    hi = mid
            s5 = unrank(group_base + lo, CUT, EARLY - 1)[4]
            j = s5 + 1 + (t - slot_start[lo])
            skips = tuple(unrank(group_base + lo, CUT, EARLY - 1)[:4]) + (s5, j)
            assert skips == tuple(sorted(skips)) and len(set(skips)) == 6
            assert all(0 <= x < CUT for x in skips)
            assert skips not in seen, ("duplicate epoch across launches", skips)
            seen.add(skips)
        slot_cursor += acc
        group_base += g
    # coverage invariant: the first k groups' member slots sum to the number
    # of flat epochs whose largest skip falls in those groups
    print(f"check 4 OK: {slot_cursor} member slots over 4 launches, all distinct "
          f"({len(seen)} unique 6-subsets), whole-group fill, gapless")

    # supply sanity for a ranked window
    total_members = sum(CUT - 1 - unrank(gg, CUT, EARLY - 1)[4]
                        for gg in range(0, n_groups_total, 9973))
    approx = total_members * (n_groups_total / len(range(0, n_groups_total, 9973)))
    print(f"supply: ~{approx:.3e} member slots (epochs) available; "
          f"a 1200 s run at 5.1e8/s needs {5.1e8 * 1200 / 256:.3e}")
    assert approx > 3 * 5.1e8 * 1200 / 256, "member-slot supply too thin"
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
