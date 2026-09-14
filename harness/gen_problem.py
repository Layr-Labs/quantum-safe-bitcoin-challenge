#!/usr/bin/env python3
"""
Generate the two SYNTHETIC benchmark problems (pinning + subset-selection).

The problems reproduce the *computational shape* of the QSB grind at its real
dimensions (n=150, t=9, ~10 KB pinning preimage, ~9.9 KB subset preimage,
secp256k1 key-recovery with precomputed neg_r_inv / u2*R) but contain NO real
coin material — the signature (r,s), the preimage bytes, and the dummy-sig pool
are all freshly generated from a fixed RNG seed (reproducible).

Outputs for each bench:
  problems/<bench>.json   authoritative full description (used by the CPU
                          verifier and human-readable)
  problems/<bench>.bin    the same data in a flat binary layout, read unchanged
                          by the reference CUDA kernels.

Run:  python3 harness/gen_problem.py [--seed 0]
"""
from __future__ import annotations
import argparse, json, os, random, struct
from pathlib import Path

import crypto as C   # local standalone crypto

ROOT = Path(__file__).resolve().parent.parent
PROB = ROOT / "problems"
SIG_PUSH_SIZE = 10

# problem dimensions
PIN_MIDSTATE_BLOCKS = 155          # 9920-byte fixed prefix
PIN_SUFFIX_LEN      = 75
PIN_SEQ_OFFSET      = 31           # within the suffix
PIN_LT_OFFSET       = 67
SUB_N, SUB_T        = 150, 9
SUB_MIDSTATE_BLOCKS = 128          # 8192 bytes
SUB_PREFIX_REMAINDER = 42          # fixed_prefix = 8192 + 42 = 8234
SUB_TAIL_LEN        = 218
SUB_TXSUFFIX_LEN    = 44


def le32(x: int) -> bytes:
    return x.to_bytes(32, "little")


def gen_signature(rng: random.Random):
    """A synthetic (r,s): r is guaranteed a valid on-curve x-coord (r = R.x < N)."""
    while True:
        k = rng.randrange(1, C.N)
        R = C.point_mul(k, C.G)
        if R[0] < C.N:                 # so r == R.x, directly recoverable
            r = R[0]
            break
    s = rng.randrange(1, C.N)
    r_inv = C.modinv(r, C.N)
    neg_r_inv = (-r_inv) % C.N
    u2 = (s * r_inv) % C.N
    R_base = (r, C.y_from_x(r, want_odd=False))     # recid-0 base point (even y)
    u2R = C.point_mul(u2, R_base)
    return r, s, neg_r_inv, u2R


def rand_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.getrandbits(8) for _ in range(n))


def gen_pinning(rng: random.Random):
    r, s, neg_r_inv, u2R = gen_signature(rng)
    pin_prefix = rand_bytes(rng, PIN_MIDSTATE_BLOCKS * 64)       # fixed, block-aligned
    suffix = bytearray(rand_bytes(rng, PIN_SUFFIX_LEN))
    # zero the seq/lt slots (kernel patches them per candidate)
    suffix[PIN_SEQ_OFFSET:PIN_SEQ_OFFSET + 4] = b"\x00\x00\x00\x00"
    suffix[PIN_LT_OFFSET:PIN_LT_OFFSET + 4] = b"\x00\x00\x00\x00"
    suffix = bytes(suffix)
    total_preimage_len = len(pin_prefix) + len(suffix)
    midstate = C.sha256_midstate(pin_prefix)

    j = {
        "bench": "pinning",
        "r": hex(r), "s": hex(s),
        "neg_r_inv": hex(neg_r_inv), "u2r_x": hex(u2R[0]), "u2r_y": hex(u2R[1]),
        "pin_prefix": pin_prefix.hex(),
        "suffix": suffix.hex(),
        "seq_offset": PIN_SEQ_OFFSET, "lt_offset": PIN_LT_OFFSET,
        "total_preimage_len": total_preimage_len,
        "midstate_blocks": PIN_MIDSTATE_BLOCKS,
    }
    b = bytearray()
    for v in midstate:
        b += struct.pack(">I", v)
    b += struct.pack("<I", len(suffix)) + suffix
    b += struct.pack("<I", total_preimage_len)
    b += struct.pack("<I", PIN_SEQ_OFFSET)
    b += struct.pack("<I", PIN_LT_OFFSET)
    b += le32(neg_r_inv) + le32(u2R[0]) + le32(u2R[1])
    return j, bytes(b)


def gen_subset(rng: random.Random):
    r, s, neg_r_inv, u2R = gen_signature(rng)
    fixed_prefix = rand_bytes(rng, SUB_MIDSTATE_BLOCKS * 64 + SUB_PREFIX_REMAINDER)
    prefix_remainder = fixed_prefix[SUB_MIDSTATE_BLOCKS * 64:]
    dummy_sigs = [rand_bytes(rng, SIG_PUSH_SIZE) for _ in range(SUB_N)]   # storage order
    tail_section = rand_bytes(rng, SUB_TAIL_LEN)
    tx_suffix = rand_bytes(rng, SUB_TXSUFFIX_LEN)
    total_preimage_len = (len(fixed_prefix) + (SUB_N - SUB_T) * SIG_PUSH_SIZE
                          + SUB_TAIL_LEN + SUB_TXSUFFIX_LEN)
    midstate = C.sha256_midstate(fixed_prefix[:SUB_MIDSTATE_BLOCKS * 64])

    j = {
        "bench": "subset",
        "n": SUB_N, "t": SUB_T, "sig_push_size": SIG_PUSH_SIZE,
        "index_convention": "storage",
        "r": hex(r), "s": hex(s),
        "neg_r_inv": hex(neg_r_inv), "u2r_x": hex(u2R[0]), "u2r_y": hex(u2R[1]),
        "fixed_prefix": fixed_prefix.hex(),
        "dummy_sigs": [d.hex() for d in dummy_sigs],
        "tail_section": tail_section.hex(),
        "tx_suffix": tx_suffix.hex(),
        "midstate_blocks": SUB_MIDSTATE_BLOCKS,
        "prefix_remainder_len": len(prefix_remainder),
        "total_preimage_len": total_preimage_len,
    }
    b = bytearray()
    b += struct.pack("<I", SUB_N)
    b += struct.pack("<I", SUB_T)
    b += struct.pack("<I", total_preimage_len)
    b += struct.pack("<I", SUB_TAIL_LEN)
    b += struct.pack("<I", SUB_TXSUFFIX_LEN)
    b += struct.pack("<I", len(prefix_remainder))
    for v in midstate:
        b += struct.pack(">I", v)
    b += prefix_remainder
    b += b"".join(dummy_sigs)
    b += tail_section
    b += tx_suffix
    b += le32(neg_r_inv) + le32(u2R[0]) + le32(u2R[1])
    return j, bytes(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (reproducible problems)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the instance (default $QSB_PROBLEM_DIR or problems/)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(os.environ.get("QSB_PROBLEM_DIR") or PROB)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for name, gen in (("pinning", gen_pinning), ("subset", gen_subset)):
        j, b = gen(rng)
        j["seed"] = args.seed
        (out_dir / f"{name}.json").write_text(json.dumps(j, indent=2))
        (out_dir / f"{name}.bin").write_bytes(b)
        print(f"  {name}: {(out_dir / f'{name}.json')}  +  {name}.bin ({len(b)} bytes)  "
              f"preimage={j['total_preimage_len']}B")
    print("problems generated (synthetic, seed=%d — no real coin material)" % args.seed)


if __name__ == "__main__":
    main()
