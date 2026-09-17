#!/usr/bin/env python3
"""Run the source-bound audit for the Y/V/H checkpoint pipeline.

The former C/Y/W/ZZZ algebra audit described the superseded recovery. Its
replacement executes the production stage source and root kernels on CPU,
with OpenSSL field primitives and an independent affine-curve oracle.
This is not a substitute for CUDA execution or independent hit verification.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("tests") / "check_leaf_recovery.py"),
                   run_name="__main__")
