#!/usr/bin/env python3
"""Reproduce static cold-seed CUDA A/B; leaves compact reports, deletes binaries."""
import argparse
import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def sass_blocks(text):
    result = {}
    for block in text.split("Function : ")[1:]:
        name, body = block.split("\n", 1)
        result[name] = re.findall(r"/\*[0-9a-f]+\*/\s+([^;]+);", body)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    source = out.parents[1] / "pinning.cu"
    reference = out / "coeff_variant" / "pinning.cu"
    report = {"kind": "Static compiler evidence only; no GPU timing", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(), "variants": {}}
    with tempfile.TemporaryDirectory(prefix=".compile_coldseed_", dir=out) as tmp:
        def build(spec):
            name, src, flags = spec
            binary = Path(tmp) / (name + ".cubin")
            command = [args.nvcc, "-O3", "-DQSB_ZEROS_N=24", "--cubin", "--ptxas-options=-v"] + flags + ["-o", str(binary), str(src)]
            proc = subprocess.run(command, text=True, capture_output=True)
            log = proc.stdout + proc.stderr
            (out / ("compile_review_" + name + ".log")).write_text(log.replace(str(source.parents[2]) + "/", ""))
            if proc.returncode:
                return name, {"compile_returncode": proc.returncode, "error": log[-3000:]}, {}
            dump = subprocess.run([args.cuobjdump, "--dump-sass", str(binary)], text=True, capture_output=True, check=True)
            parsed = sass_blocks(dump.stdout)
            item = {"compile_returncode": 0, "flags": ["-O3", "-DQSB_ZEROS_N=24", "--cubin"] + flags, "kernels": {}}
            for symbol, ins in parsed.items():
                if "kernel_pinning_pipeline" not in symbol:
                    continue
                stage = "prepare" if "ILb1ELi0" in symbol else "finish"
                section = log.split("Function properties for " + symbol, 1)[1].split("Compile time", 1)[0]
                count = lambda pattern: int(re.search(pattern, section)[1]) if re.search(pattern, section) else 0
                item["kernels"][stage] = {"instructions": len(ins), "registers": count(r"Used (\d+) registers"), "stack_bytes": count(r"(\d+) bytes stack frame"), "spill_store_bytes": count(r"(\d+) bytes spill stores"), "spill_load_bytes": count(r"(\d+) bytes spill loads"), "shared_bytes": count(r"(\d+) bytes smem"), "instruction_text_sha256": hashlib.sha256("\n".join(ins).encode()).hexdigest(), "mnemonic_counts": dict(sorted(collections.Counter(i.strip().split()[0] for i in ins).items())), "global_loads": [i.strip() for i in ins if "LDG" in i]}
            return name, item, parsed
        specs = [("coldseed_off_sm89", source, ["-arch=sm_89", "-DQSB_GLV_COLD_SEED=0"]), ("coldseed_on_sm89", source, ["-arch=sm_89", "-DQSB_GLV_COLD_SEED=1"]), ("coldseed_off_default", source, ["-DQSB_GLV_COLD_SEED=0"]), ("coldseed_on_default", source, ["-DQSB_GLV_COLD_SEED=1"]), ("coldseed_reference_sm89", reference, ["-arch=sm_89", "-DQSB_GLV_HIGH10_HI=0"])]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(build, specs))
        all_ins = {}
        for name, item, parsed in results:
            report["variants"][name] = item
            all_ins[name] = parsed
        comparisons = [("coldseed_off_sm89", "coldseed_reference_sm89"), ("coldseed_on_sm89", "coldseed_off_sm89"), ("coldseed_on_default", "coldseed_off_default")]
        report["comparisons"] = {}
        for left, right in comparisons:
            entries = {}
            for symbol, seq in all_ins[left].items():
                if "kernel_pinning_pipeline" not in symbol:
                    continue
                stage = "prepare" if "ILb1ELi0" in symbol else "finish"
                other = all_ins[right][symbol]
                a, b = collections.Counter(i.strip().split()[0] for i in seq), collections.Counter(i.strip().split()[0] for i in other)
                changes = [{"instruction_index": i, "left": x.strip(), "right": y.strip()} for i, (x, y) in enumerate(zip(seq, other)) if x != y]
                entries[stage] = {"instruction_text_exact_match": seq == other, "positional_changes": len(changes), "first_changes": changes[:24], "mnemonic_delta": {k: a[k] - b[k] for k in sorted(a.keys() | b.keys()) if a[k] != b[k]}}
            report["comparisons"][left + "_vs_" + right] = entries
    (out / "compile_review_coldseed_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, item in report["variants"].items():
        print(name, {k: {f: v[f] for f in ["instructions", "registers", "spill_store_bytes", "spill_load_bytes"]} for k, v in item.get("kernels", {}).items()}, flush=True)
    for name, item in report["comparisons"].items():
        print(name, {k: {f: v[f] for f in ["instruction_text_exact_match", "positional_changes", "mnemonic_delta"]} for k, v in item.items()}, flush=True)


if __name__ == "__main__":
    main()
