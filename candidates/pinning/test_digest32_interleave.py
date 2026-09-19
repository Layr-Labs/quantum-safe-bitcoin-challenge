#!/usr/bin/env python3
"""Compile the actual Digest32 transform and compare it with hashlib.

Host correctness only; this does not measure CUDA performance.
SPDX-License-Identifier: GPL-3.0-only
"""
import ctypes
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tempfile

from test_sha_interleave import extract_function

HERE = Path(__file__).resolve().parent
BASE = "9ef2d74abbbbb1e436b2e11461ba51c11d9cb3b7"


def main():
    source = (HERE / "pinning.cu").read_text()
    function = extract_function(source, "_SHA256TransformDigest32")
    assert function.count("QSB_SHA_INTERLEAVED_16(") == 2
    base_source = subprocess.check_output(
        ["git", "show", f"{BASE}:candidates/pinning/pinning.cu"],
        cwd=HERE, text=True,
    )
    baseline = extract_function(base_source, "_SHA256TransformDigest32")
    assert baseline.count("WMIX();") == 2
    baseline = baseline.replace("_SHA256TransformDigest32", "baseline_digest32")
    macros = (HERE / "GPUHash.h").read_text().split("//Take the last 8 bytes")[0]
    header = (HERE / "sha_schedule_interleaved.cuh").read_text()
    harness = """#include <cstdint>
#define __device__
#define __constant__
#define __forceinline__ inline
""" + macros + header + function + baseline + """
extern "C" void candidate(uint32_t *o, const uint32_t *m) { _SHA256TransformDigest32(o,m); }
extern "C" void control(uint32_t *o, const uint32_t *m) { baseline_digest32(o,m); }
"""
    rng = random.Random(0xD1635732)
    vectors = [bytes([byte]) * 32 for byte in range(256)]
    vectors += [(1 << bit).to_bytes(32, "big") for bit in range(256)]
    vectors += [((1 << 256) - 1 - (1 << bit)).to_bytes(32, "big") for bit in range(256)]
    vectors += [bytes(range(32)), bytes(reversed(range(32)))]
    vectors += [rng.randbytes(32) for _ in range(20000)]
    vectors += [hashlib.sha256(rng.randbytes(length)).digest() for length in range(256)]
    u32 = ctypes.c_uint32
    with tempfile.TemporaryDirectory(prefix="qsb-digest32-host-") as temp:
        cpp, library = Path(temp) / "test.cpp", Path(temp) / "test.so"
        cpp.write_text(harness)
        build = subprocess.run(
            ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wno-unknown-pragmas",
             "-shared", "-fPIC", str(cpp), "-o", str(library)],
            capture_output=True, text=True, check=True,
        )
        lib = ctypes.CDLL(str(library))
        for name in ("candidate", "control"):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.POINTER(u32), ctypes.POINTER(u32)]
            fn.restype = None
        for index, message in enumerate(vectors):
            words = [int.from_bytes(message[i:i + 4], "big") for i in range(0, 32, 4)]
            expected = hashlib.sha256(message).digest()
            for name in ("candidate", "control"):
                inp, out = (u32 * 8)(*words), (u32 * 8)()
                getattr(lib, name)(out, inp)
                actual = b"".join(int(word).to_bytes(4, "big") for word in out)
                assert actual == expected, (index, name)
            alias = (u32 * 8)(*words)
            lib.candidate(alias, alias)
            assert b"".join(int(word).to_bytes(4, "big") for word in alias) == expected, (index, "alias")
    print(json.dumps({
        "base_commit": BASE,
        "test": "actual Digest32 source compiled as host C++ versus hashlib and pinned git control",
        "vectors": len(vectors),
        "digest_comparisons": len(vectors) * 3,
        "candidate_vs_hashlib": "pass",
        "pinned_control_vs_hashlib": "pass",
        "in_place_alias": "pass",
        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "compiler_stderr": build.stderr,
        "function_sha256": hashlib.sha256(function.encode()).hexdigest(),
        "cuda_compiled": False,
        "gpu_executed": False,
        "official_score": None,
        "speedup": None,
    }, indent=2))


if __name__ == "__main__":
    main()
