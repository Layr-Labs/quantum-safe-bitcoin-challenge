#!/usr/bin/env python3
"""Stage immutable PR219 control and pending fused-reduction source."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
COMMITS = {
    "control": "f7f588c7a9baacc261f5d6214c13e54130583f31",
    "candidate": "211f74dc72340fa673e4ca0c3924d9af681ef6c0",
}
FILES = [
    "pinning.cu", "GPUMath.h", "GPUHash.h", "LeafRecovery.cuh",
    "RecoveryConstant.h", "COPYING",
]


def blob(commit, rel):
    return subprocess.check_output(
        ["git", "show", f"{commit}:candidates/pinning/{rel}"], cwd=REPO
    )


manifest = {}
for name, commit in COMMITS.items():
    root = HERE / name
    root.mkdir(exist_ok=True)
    hashes = {}
    for rel in FILES:
        data = blob(commit, rel)
        (root / rel).write_bytes(data)
        hashes[rel] = hashlib.sha256(data).hexdigest()
    manifest[name] = {"commit": commit, "sha256": hashes}

(HERE / "provenance.json").write_text(json.dumps({
    "base_submission": "9dbc5b71-c9d1-4a82-a0d1-a928bde91156",
    "candidate_submission": "f297b0f9-d2ec-4b17-964c-703d12226f12",
    "public_prs": {"base": 219, "candidate": 225},
    "trees": manifest,
}, indent=2) + "\n")
print("prepared immutable control and candidate")
