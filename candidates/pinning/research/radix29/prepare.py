#!/usr/bin/env python3
"""Isolated radix29 product with the exact repaired production reduction tail.

Full-width inputs are unpacked into 9 bounded limbs. Each raw product
column fits 64 bits, allowing mad.wide accumulation without per-product carry.
The product is repacked into the unchanged 16x32 reduction input. This tests
whether saving explicit carry work compensates for more products/conversions.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from compile_local import source_closure
from ptx_field_model import extract_ptx, function


def generate():
    lines = ["{", ".reg .u64 acc;", ".reg .u32 tmp;"]
    for stem, count in [("a", 8), ("b", 8), ("u", 9), ("v", 9), ("d", 18), ("x", 16)]:
        lines.append(".reg .u32 " + ",".join(stem + str(i) for i in range(count)) + ";")
    for stem, dest, first in [("a", "u", 4), ("b", "v", 8)]:
        for i in range(4):
            lines.append(f"mov.b64 {{{stem}{2*i},{stem}{2*i+1}}}, %{first+i};")
        for k in range(9):
            j, shift = divmod(29 * k, 32)
            if k == 8:
                lines.append(f"shr.u32 {dest}{k}, {stem}{j}, {shift};")
            elif shift:
                lines.append(f"shf.r.wrap.b32 {dest}{k}, {stem}{j}, {stem}{j+1}, {shift};")
                lines.append(f"and.b32 {dest}{k}, {dest}{k}, 536870911;")
            else:
                lines.append(f"and.b32 {dest}{k}, {stem}{j}, 536870911;")
    lines.append("mov.u64 acc, 0;")
    bound, bounds = 0, []
    limb_max = [(1 << 29) - 1] * 8 + [(1 << 24) - 1]
    for col in range(17):
        pairs = [(i, col-i) for i in range(9) if 0 <= col-i < 9]
        bound += sum(limb_max[i] * limb_max[j] for i, j in pairs)
        assert bound < 1 << 64
        bounds.append(bound)
        for i, j in pairs:
            lines.append(f"mad.wide.u32 acc, u{i}, v{j}, acc;")
        lines.append(f"cvt.u32.u64 d{col}, acc;")
        lines.append(f"and.b32 d{col}, d{col}, 536870911;")
        lines.append("shr.u64 acc, acc, 29;")
        bound >>= 29
    assert bound < 1 << 19
    lines.append("cvt.u32.u64 d17, acc;")
    for k in range(16):
        j, shift = divmod(32 * k, 29)
        lines.append(f"shr.u32 x{k}, d{j}, {shift};")
        if j+1 < 18:
            lines.append(f"shl.b32 tmp, d{j+1}, {29-shift};")
            lines.append(f"or.b32 x{k}, x{k}, tmp;")
        if 58-shift < 32 and j+2 < 18:
            lines.append(f"shl.b32 tmp, d{j+2}, {58-shift};")
            lines.append(f"or.b32 x{k}, x{k}, tmp;")
    return "\n".join(lines) + "\n", bounds


def main():
    out = HERE / "candidate"
    if out.exists():
        raise SystemExit("Refusing to overwrite an existing experiment")
    files = source_closure(BASE.resolve(), "pinning.cu") + [BASE / "COPYING"]
    hashes = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    saved = json.loads((BASE / "research/wide_windows/submitted-source.json").read_text())
    assert hashes == saved["production_sha256"], "Submitted source changed; reassess control"
    for p in files:
        dest = out / p.relative_to(BASE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    header = (out / "GPUMath.h").read_text()
    body = function(header, "__device__ __forceinline__ void _ModMultCore(")
    previous = extract_ptx(body)
    prefix, bounds = generate()
    # The retained reduction uses temporary64-bit t from the old product.
    ptx = prefix + ".reg .u64 t;\n" + previous[previous.index(".reg .u64 r0,r1,r2,r3"):]
    raw = prefix + "\n".join(f"mov.b64 %{i}, {{x{2*i},x{2*i+1}}};" for i in range(8)) + "\n}"
    encoded = "\n".join("        " + json.dumps(line + "\n") for line in ptx.splitlines())
    start, end = body.index("asm("), body.index(': "=l"')
    patched = body[:start] + "asm(\n" + encoded + "\n        " + body[end:]
    (out / "GPUMath.h").write_text(header.replace(body, patched))
    (HERE / "field.ptx").write_text(ptx + "\n")
    (HERE / "product.ptx").write_text(raw + "\n")
    report = {"status": "isolated prototype; not submitted", "base_hashes": hashes,
              "candidate_hashes": {str(p.relative_to(out.resolve())): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in source_closure(out.resolve(), "pinning.cu")},
              "raw_word_products": 81, "reduction_tail": "exact corrected production code",
              "column_upper_bounds": bounds, "maximum_column_bits": max(bounds).bit_length(),
              "host_fallback_unchanged": True, "specialized_square_unchanged": True,
              "gpu_executed": False}
    (HERE / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Generated isolated radix29 product; maximum column bound", max(bounds).bit_length(), "bits")


if __name__ == "__main__":
    main()
