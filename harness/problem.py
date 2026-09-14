"""
Shared re-derivation logic: turn a candidate (a pinning (seq,lt) or a subset
skip-set) into the recovered-key hash `h`, exactly as a grinding kernel must.
Used by BOTH the CPU reference grinder and the verifier, so they agree by
construction.
"""
from __future__ import annotations
import json, os, struct
from pathlib import Path

import crypto as C

ROOT = Path(__file__).resolve().parent.parent


def problem_dir() -> Path:
    """Where the problem instance for this run lives. A ranked run generates a
    fresh instance into a scratch directory and points QSB_PROBLEM_DIR at it;
    the committed seed-0 instance under problems/ is the public example."""
    return Path(os.environ.get("QSB_PROBLEM_DIR") or (ROOT / "problems"))


def load_problem(name: str) -> dict:
    return json.loads((problem_dir() / f"{name}.json").read_text())


def pin_preimage(prob: dict, sequence: int, locktime: int) -> bytes:
    prefix = bytes.fromhex(prob["pin_prefix"])
    suffix = bytearray(bytes.fromhex(prob["suffix"]))
    so, lo = prob["seq_offset"], prob["lt_offset"]
    suffix[so:so + 4] = struct.pack("<I", sequence & 0xFFFFFFFF)
    suffix[lo:lo + 4] = struct.pack("<I", locktime & 0xFFFFFFFF)
    pre = prefix + bytes(suffix)
    assert len(pre) == prob["total_preimage_len"]
    return pre


def sub_preimage(prob: dict, skip) -> bytes:
    """skip = list of `t` STORAGE indices (which dummy sigs to omit)."""
    fp = bytes.fromhex(prob["fixed_prefix"])
    dummies = [bytes.fromhex(d) for d in prob["dummy_sigs"]]
    skipset = set(skip)
    body = b"".join(d for i, d in enumerate(dummies) if i not in skipset)
    pre = fp + body + bytes.fromhex(prob["tail_section"]) + bytes.fromhex(prob["tx_suffix"])
    assert len(pre) == prob["total_preimage_len"], (len(pre), prob["total_preimage_len"])
    return pre


def candidate_preimage(prob: dict, cand: dict) -> bytes:
    if prob["bench"] == "pinning":
        return pin_preimage(prob, cand["sequence"], cand["locktime"])
    return sub_preimage(prob, cand["skip"])


def recovered_hash(prob: dict, z: int, recid: int):
    """h = SHA256(compress(recover(r,s,z,recid)))  or None if r not on curve."""
    r = int(prob["r"], 16)
    s = int(prob["s"], 16)
    Q = C.ecdsa_recover(r, s, z, recid)
    if Q is None:
        return None
    return C.sha256(C.compress_pubkey(Q))


def candidate_hash(prob: dict, cand: dict, recid: int):
    """Independent path (used by the verifier): recompute everything from (r,s)."""
    z = int.from_bytes(C.sha256d(candidate_preimage(prob, cand)), "big")
    return recovered_hash(prob, z, recid)


def precomputed_points(prob: dict):
    """Precomputed offsets from the problem: u2R (recid 0) and neg_2u2R (→ recid 1)."""
    u2R = (int(prob["u2r_x"], 16), int(prob["u2r_y"], 16))
    two = C.point_add(u2R, u2R)
    neg_2u2R = (two[0], (C.P - two[1]) % C.P)
    return u2R, neg_2u2R


def recovered_hash_fast(prob: dict, z: int, recid: int, neg_r_inv: int, u2R, neg_2u2R):
    """Fast path (used by the grinder): u1*G + precomputed offset. Agrees with
    candidate_hash() by construction."""
    u1 = (neg_r_inv * z) % C.N
    Q1 = C.point_add(C.point_mul(u1, C.G), u2R)
    Q = Q1 if recid == 0 else C.point_add(Q1, neg_2u2R)
    return C.sha256(C.compress_pubkey(Q))
