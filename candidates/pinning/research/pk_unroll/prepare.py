#!/usr/bin/env python3
"""Stage exact promoted control and QSB_PK_UNROLL=1 probe."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASE = "067302c"
FILES = ["pinning.cu", "GPUMath.h", "GPUHash.h", "COPYING"]


def git_blob(rel):
    return subprocess.check_output(
        ["git", "show", f"{BASE}:candidates/pinning/{rel}"], cwd=REPO
    )


def write_tree(name, unroll):
    root = HERE / name
    root.mkdir(exist_ok=True)
    hashes = {}
    for rel in FILES:
        data = git_blob(rel)
        if rel == "pinning.cu" and unroll:
            old = b"#define QSB_PK_UNROLL 0       /* 1: unroll the two-recid pubkey SHA loop so both chains interleave */"
            new = b"#define QSB_PK_UNROLL 1       /* 1: unroll the two-recid pubkey SHA loop so both chains interleave */"
            if data.count(old) != 1:
                raise RuntimeError("expected exactly one promoted QSB_PK_UNROLL definition")
            data = data.replace(old, new)
        (root / rel).write_bytes(data)
        hashes[rel] = hashlib.sha256(data).hexdigest()
    return hashes


control = write_tree("control", False)
candidate = write_tree("candidate", True)
(HERE / "provenance.json").write_text(json.dumps({
    "base_commit": BASE,
    "hypothesis": "unrolling the two independent recovered-pubkey SHA chains exposes ILP without changing work",
    "control_sha256": control,
    "candidate_sha256": candidate,
    "changed_files": ["pinning.cu"],
    "production_default_changed": {"QSB_PK_UNROLL": [0, 1]},
}, indent=2) + "\n")
print("prepared", BASE)
