#!/usr/bin/env python3
"""Execute the current probe body with CPU CUDA shims, then an OpenSSL oracle.

The tree, service and shifted-tail headers are included unchanged. Only CUDA
qualifiers, shared storage and arithmetic primitives are mapped to host code.
This audits whole-worker/control integration, not CUDA PTX or GPU scheduling.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def function_end(source, start):
    brace = source.index("{", start)
    depth = 1
    i = brace + 1
    while depth:
        depth += (source[i] == "{") - (source[i] == "}")
        i += 1
    return i


def main():
    path = HERE / "persistent_affine_probe.cu"
    source = path.read_text()
    start = source.index("#ifndef QSB_AFFINE_MIN_BLOCKS")
    end = source.index("#define CUDA_OK")
    extracted = source[start:end]
    sub_start = extracted.index("    __device__ __forceinline__ static void sub(")
    sub_end = function_end(extracted, sub_start)
    extracted = (extracted[:sub_start] +
                 "    static void sub(uint64_t out[4], const uint64_t a[4], const uint64_t b[4]) {\n"
                 "        cpp_int r=(integer(a)-integer(b))%P; if(r<0)r+=P; limbs(out,r);\n"
                 "    }" + extracted[sub_end:])
    old_shared = "    __shared__ uint64_t products[4][2*AFFINE_N],inverses[4][AFFINE_N];"
    assert extracted.count(old_shared) == 1
    extracted = extracted.replace(old_shared,
        "    auto &products=cpu_block->products; auto &inverses=cpu_block->inverses;")
    old_observer = 'asm volatile(""::"l"(result.x[0][0]),"l"(result.x[1][0]),"r"(result.parities):"memory");'
    assert extracted.count(old_observer) == 1
    extracted = extracted.replace(old_observer, "cpu_record(result,tile,candidate);")
    encode_start = source.index("static bool affine_encode(")
    extracted += source[encode_start:function_end(source, encode_start)]
    with tempfile.TemporaryDirectory(prefix=".audit-persistent-", dir=HERE) as directory:
        temporary = Path(directory)
        include = temporary / "probe.inc"
        include.write_text(extracted)
        binary = temporary / "audit"
        command = ["rtk", "proxy", "g++", "-std=c++20", "-O2", "-pthread",
                   "-Wno-deprecated-declarations", "-I/tmp/qsb-boost-audit/usr/include",
                   f'-DQSB_CPU_PROBE_SOURCE="{include}"',
                   str(HERE / "audit_persistent_cpu.cpp"), "-lcrypto", "-o", str(binary)]
        subprocess.run(command, check=True)
        result = subprocess.run(["rtk", "proxy", str(binary)], check=True,
                                capture_output=True, text=True, timeout=180)
    report = json.loads(result.stdout)
    report["source_sha256"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("persistent_affine_probe.cu", "inverse_tree.cuh",
                     "inverse_service.cuh", "shifted_tail.cuh",
                     "audit_persistent_cpu.cpp", "audit_persistent_cpu.py")
    }
    # The parent may continue editing the probe while this audit is running.
    # Attribute the tested snapshot, never a newer file read after execution.
    report["source_sha256"][path.name] = hashlib.sha256(source.encode()).hexdigest()
    report["extracted_source_sha256"] = hashlib.sha256(extracted.encode()).hexdigest()
    (HERE / "persistent_cpu_result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
