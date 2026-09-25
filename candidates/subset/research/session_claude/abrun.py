#!/usr/bin/env python3
"""Local A/B runner for the subset candidate.

Same harness/gpu_wrap.py path as the ranked/local default, except that the
compile flags are selectable so native (sm_86/sm_89) builds and load-qualifier
variants can be compared. Hit collection, argv, and the artifact format are
unchanged; run_benchmark.py still owns the clock and the verifier.

Env:
  QSB_AB_TAG    label used in the binary name and the stdout log
  QSB_AB_FLAGS  extra nvcc flags (space separated), e.g. "-arch=sm_86"
"""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "harness"))
import gpu_wrap

TAG = os.environ.get("QSB_AB_TAG", "default")
EXTRA = os.environ.get("QSB_AB_FLAGS", "").split()

original_compile = gpu_wrap.compile_kernel


def compile_variant(src: Path, zeros: int, no_build: bool = False) -> Path:
    binp = src.parent / f".ab-{TAG}-{src.stem}"
    stamp = src.parent / f".ab-{TAG}-{src.stem}.build"
    want = f"QSB_ZEROS_N={zeros} {EXTRA} {src.stat().st_mtime_ns}"
    if binp.exists() and stamp.exists() and stamp.read_text() == want:
        print(f"  compile: reusing {binp}", flush=True)
        return binp
    cmd = ["nvcc", "-O3", f"-DQSB_ZEROS_N={zeros}", *EXTRA,
           "-o", str(binp), str(src), "-lcrypto", "-lm"]
    print("  compile:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    stamp.write_text(want)
    return binp


original_run = subprocess.run


def logged_run(*args, **kwargs):
    result = original_run(*args, **kwargs)
    if kwargs.get("capture_output") and kwargs.get("text"):
        output = Path(sys.argv[sys.argv.index("--out") + 1])
        text = result.stdout + result.stderr
        output.with_suffix(f".{TAG}.stdout.log").write_text(text)
        for line in text.splitlines():
            if "M/s" in line or "CUDA error" in line or "GTable" in line:
                print(f"  [{TAG}] {line}", flush=True)
    return result


gpu_wrap.compile_kernel = compile_variant
subprocess.run = logged_run
gpu_wrap.main()
