#!/usr/bin/env python3
"""Fingerprint the include closure and optionally compile a fresh CUDA copy.

No submission, benchmark, persistent build cache, or repository mutation.
The isolated build avoids gpu_wrap.py's entry-point-only timestamp cache.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def source_files(root=HERE):
    root = root.resolve()
    seen = set()

    def visit(path):
        path = path.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or out-of-track include: {path.name}")
        if path in seen:
            return
        seen.add(path)
        for relative in re.findall(r'^\s*#include\s+"([^"]+)"', path.read_text(), re.M):
            visit(path.parent / relative)

    visit(root / "subset.cu")
    visit(root / "tests/gpu_epochs/tree_audit.cu")
    return sorted(seen)


def source_identity(root=HERE):
    root = root.resolve()
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in source_files(root)}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"source_fingerprint": fingerprint, "source_sha256": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--note", type=Path, help="also validate a public note's byte limit")
    parser.add_argument("--package-check", action="store_true",
                        help="enforce the track's expanded-file budget on this candidate directory")
    parser.add_argument("--cuda", action="store_true", help="compile fresh production and audit copies with nvcc")
    args = parser.parse_args()
    report = {"include_closure": "pass", **source_identity(),
              "cuda_compile": "not_run", "gpu_executed": False}
    if args.package_check:
        # Yukon archives the whole editable directory, including untracked
        # research. maxSubmissionBytes limits expanded content on the server;
        # a small gzip alone does not satisfy it (a13a9d4b intake rejection).
        manifest = json.loads((HERE.parents[1] / "benchmark.json").read_text())
        track = next(t for t in manifest["tracks"] if t["name"] == "subset")
        limit = track["maxSubmissionBytes"]
        files = [path for path in HERE.rglob("*") if path.is_file()]
        if any(path.is_symlink() for path in HERE.rglob("*")):
            parser.error("package-size check requires a regular-file candidate tree")
        expanded = sum(path.stat().st_size for path in files)
        report["package"] = {"files": len(files), "expanded_file_bytes": expanded,
                             "expanded_limit_bytes": limit,
                             "compressed_size": "separate CLI transport limit"}
        if expanded > limit:
            parser.error(f"candidate expands to {expanded} bytes; at most {limit} are allowed. "
                         "Stage the intended source and checks in an isolated checkout; preserve local research.")
    if args.note:
        data = args.note.read_bytes()
        if not 5120 <= len(data) <= 102400:
            parser.error("public note must be 5–100 KiB")
        report["note"] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if args.cuda:
        nvcc = shutil.which("nvcc")
        if not nvcc:
            parser.error("nvcc is unavailable; CUDA compilation has not been checked")
        with tempfile.TemporaryDirectory(prefix="qsb-subset-clean-build-") as directory:
            root = Path(directory) / "candidates/subset"
            for path in source_files():
                target = root / path.relative_to(HERE)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            for source, output in (("subset.cu", "subset"),
                                   ("tests/gpu_epochs/tree_audit.cu", "tree-audit")):
                subprocess.run([nvcc, "-O3", "-DQSB_ZEROS_N=24", "-o", str(root / output),
                                str(root / source), "-lcrypto", "-lm"],
                               check=True, stdout=sys.stderr, stderr=sys.stderr)
        report["cuda_compile"] = "pass_clean_production_and_audit"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
