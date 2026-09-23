#!/usr/bin/env python3
"""Build the standalone probe and record native resources; never run GPU work.

Example:
  python3 compile_probe.py --nvcc /tmp/qsb-cuda/nvcc-local \
    --cuobjdump /tmp/qsb-cuda/toolkit/bin/cuobjdump --ptx
Intermediate cubin/SASS/PTX files are removed unless --keep-intermediates is set.
The executable, compiler logs, and JSON report remain in research_affine/.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "persistent_affine_probe.cu"
KERNEL_PREFIX = "_Z27qsb_persistent_affine_probe"


def run(args, log=None):
    result = subprocess.run([str(arg) for arg in args], text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if log is not None:
        log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {args!r}\n{result.stdout[-5000:]}")
    return result.stdout


def resources(log):
    match = re.search(r"Function properties for (" + KERNEL_PREFIX + r"[^\n]+)\n"
                      r"\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\n"
                      r"ptxas info\s*: Used (\d+) registers, used (\d+) barriers,([^\n]*)", log)
    if not match:
        raise RuntimeError("missing persistent-kernel ptxas resource record")
    symbol, stack, stores, loads, registers, barriers, remainder = match.groups()
    shared = re.search(r"(\d+) bytes smem", remainder)
    return dict(symbol=symbol, registers=int(registers), stack_bytes=int(stack),
                spill_store_bytes=int(stores), spill_load_bytes=int(loads),
                barriers=int(barriers), shared_bytes=int(shared.group(1)) if shared else 0)


def local_memory(sass, symbols, kernel):
    # cuobjdump groups callees into the entry's section. ELF symbol intervals
    # distinguish worker instructions from the service helper and its inverse.
    section = sass.split("Function : " + kernel, 1)[1].split("Function : ", 1)[0]
    intervals = []
    for line in symbols.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[3] == "FUNC" and fields[-1].startswith("$" + kernel + "$"):
            start, size = int(fields[1], 16), int(fields[2])
            intervals.append((start, start + size, fields[-1].split("$")[-1]))
    if not any("affine_service_branch" in name for _, _, name in intervals):
        raise RuntimeError("service ELF function interval not found")
    counts = {"worker_entry_and_dispatch": {"LDL": 0, "STL": 0}}
    for _, _, name in intervals:
        counts[name] = {"LDL": 0, "STL": 0}
    addresses = {name: [] for name in counts}
    for line in section.splitlines():
        instruction = re.search(r"/\*([0-9a-f]+)\*/.*?\b(LDL|STL)(?:[.\s])", line)
        if not instruction:
            continue
        offset, op = int(instruction[1], 16), instruction[2]
        owner = next((name for lo, hi, name in intervals if lo <= offset < hi),
                     "worker_entry_and_dispatch")
        counts[owner][op] += 1
        addresses[owner].append(hex(offset))
    return dict(static_instruction_counts=counts, local_instruction_addresses=addresses,
                function_intervals=[dict(symbol=name, start=hex(lo), end_exclusive=hex(hi))
                                    for lo, hi, name in intervals],
                note="Static SASS counts, not dynamic memory traffic. Entry count excludes ELF-identified service and inverse callees.")


def ptx_local_depots(ptx):
    result = {}
    for match in re.finditer(r"(?:\.visible\s+\.entry|\.func)\s+(\S+)\(", ptx):
        name = match.group(1)
        if not any(token in name for token in ("qsb_persistent_affine_probe", "affine_service_branch", "_ModInv")):
            continue
        start = ptx.find("{", match.end())
        depth, end = 1, start + 1
        while depth and end < len(ptx):
            depth += (ptx[end] == "{") - (ptx[end] == "}")
            end += 1
        body = ptx[start:end]
        depots = [int(n) for n in re.findall(r"\.local\s+\.align\s+\d+\s+\.b8\s+\S+\[(\d+)\]", body)]
        result[name] = dict(local_depot_bytes=depots, total_local_depot_bytes=sum(depots))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--readelf", default="readelf")
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--bounds", nargs="+", type=int, default=[3, 4, 6])
    parser.add_argument("--executable-bound", type=int, default=4)
    parser.add_argument("--build-dir", type=Path, default=HERE / "_build")
    parser.add_argument("--report", type=Path, default=HERE / "persistent_resources.json")
    parser.add_argument("--resources-only", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--ptx", action="store_true", help="also inspect PTX local depots at the executable bound")
    args = parser.parse_args()
    if min(args.bounds + [args.executable_bound]) < 1:
        parser.error("launch bounds must be positive")
    args.build_dir.mkdir(parents=True, exist_ok=True)
    common = [args.nvcc, "-O3", "-std=c++17", "-DNDEBUG", "-arch=" + args.arch]
    version = run([args.nvcc, "--version"])
    report = dict(kind="native_compile_resource_screen", gpu_work_executed=False,
                  performance_measured=False, architecture=args.arch,
                  compiler_version=version.strip(), thread_block_size=128,
                  flags=common[1:], probes=[], source_sha256={},
                  limitations=["No GPU correctness or speedup established.",
                               "Actual occupancy must be queried by the probe runtime before a cooperative launch.",
                               "This reference probe uses full multiplication for squares and parity; it is not the optimized candidate."])
    for path in [SOURCE, HERE / "inverse_service.cuh", HERE / "inverse_tree.cuh",
                 HERE / "shifted_tail.cuh", HERE.parent / "pinning.cu", HERE.parent / "GPUMath.h"]:
        report["source_sha256"][str(path.relative_to(HERE.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    cleanup = []
    for bound in args.bounds:
        stem = args.build_dir / f"persistent_b{bound}"
        cubin, log, sass_path = stem.with_suffix(".cubin"), stem.with_suffix(".log"), stem.with_suffix(".sass")
        command = common + [f"-DQSB_AFFINE_MIN_BLOCKS={bound}", "--cubin", "-Xptxas=-v", SOURCE, "-o", cubin]
        compiler_output = run(command, log)
        native = resources(compiler_output)
        sass = run([args.cuobjdump, "--dump-sass", cubin], sass_path)
        symbols = run([args.readelf, "-sW", cubin])
        native.update(min_blocks_per_sm=bound, compiler_log=str(log),
                      local_memory=local_memory(sass, symbols, native["symbol"]),
                      build_command=[str(x) for x in command])
        register_allocation = ((native["registers"] * 32 + 255) // 256) * 256 * 4
        native["register_capacity_check"] = dict(
            assumption_registers_per_sm=65536, assumption_register_allocation_granule_per_warp=256,
            allocated_registers_per_cta=register_allocation,
            register_limited_ctas_per_sm=65536 // register_allocation,
            shared_bytes_at_requested_bound=native["shared_bytes"] * bound,
            runtime_occupancy_still_required=True)
        report["probes"].append(native)
        cleanup.extend([cubin, sass_path])
        print(f"bound={bound}: {native['registers']} registers, {native['shared_bytes']} shared bytes, "
              f"{native['stack_bytes']} stack bytes, spills {native['spill_store_bytes']}/{native['spill_load_bytes']} bytes")
    if args.ptx:
        path = args.build_dir / f"persistent_b{args.executable_bound}.ptx"
        run(common + [f"-DQSB_AFFINE_MIN_BLOCKS={args.executable_bound}", "--ptx", SOURCE, "-o", path],
            args.build_dir / f"persistent_b{args.executable_bound}_ptx.log")
        report["ptx_local_depots"] = ptx_local_depots(path.read_text())
        cleanup.append(path)
    if not args.resources_only:
        executable = args.build_dir / f"persistent_affine_probe_b{args.executable_bound}"
        command = common + [f"-DQSB_AFFINE_MIN_BLOCKS={args.executable_bound}", "-Xptxas=-v", SOURCE,
                            "-lcrypto", "-lm", "-o", executable]
        log = args.build_dir / f"persistent_executable_b{args.executable_bound}.log"
        run(command, log)
        report["executable"] = dict(path=str(executable), min_blocks_per_sm=args.executable_bound,
                                    describe=run([executable, "--describe"]).strip(),
                                    build_command=[str(x) for x in command], compiler_log=str(log))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not args.keep_intermediates:
        for path in cleanup:
            path.unlink()
    print(f"Wrote {args.report}; no GPU workload executed")


if __name__ == "__main__":
    main()
