#!/usr/bin/env python3
"""Interpret the actual seeded PTX; this is not CUDA execution or a timing test.

The 512-bit product must equal a*b+c exactly for every tested triple. The
retained reducer is intentionally approximate: it drops the bit-288 first-fold
carry and the second-fold carry above bit 95. Directed boundary mismatches are
reported and accounted for, never relabeled as exact field arithmetic. The
device reducer is also compared with the current promoted multiplier's tail.

Alias checks model overlapping PTX input/output registers and the device's
deferred C++ stores. They do not execute the host-only multiply-then-add shim.
"""
import ast
from collections import Counter
import itertools
import json
from pathlib import Path
import random
import re

B = 1 << 256
K = (1 << 32) + 977
P = B - K
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
MASK96 = (1 << 96) - 1
FOLD_MARKER = ".reg .u64 r0,r1,r2,r3,h0,h1,h2,h3"


def extract_asm(source):
    match = re.search(r"\basm\s*\((.*?)\n\s*:", source, re.S)
    assert match, "assembly template missing"
    return "".join(ast.literal_eval(s) for s in
                   re.findall(r'"(?:\\.|[^"\\])*"', match.group(1)))


def parse(ptx):
    ptx = re.sub(r"/\*.*?\*/", "", ptx, flags=re.S)
    ptx = re.sub(r"\.reg[^;]*;", "", ptx)
    # Scope braces are standalone; mov.b64 operand braces remain intact.
    ptx = re.sub(r"(?m)^\s*[{}]\s*$", "", ptx)
    ptx = re.sub(r";\s*}", ";", ptx)
    ops = []
    for instruction in ptx.split(";"):
        instruction = instruction.strip().lstrip("{").strip()
        if not instruction or instruction == "}":
            continue
        op, args = instruction.split(None, 1)
        operands = re.findall(r"\{[^}]*\}|[^,]+", args.replace(" ", ""))
        assert op in {"mov.b64", "mov.u64", "mov.u32", "mul.wide.u32",
                      "add.cc.u64", "addc.cc.u64", "addc.u64",
                      "add.cc.u32", "addc.cc.u32", "addc.u32"}, op
        ops.append((op, operands))
    return ops


def execute(ops, initial, output_alias=None):
    registers = dict(initial)
    carry = 0

    def name(operand):
        if output_alias is not None and operand in {"%0", "%1", "%2", "%3"}:
            return "%" + str(output_alias + int(operand[1:]))
        return operand

    def get(operand):
        operand = name(operand)
        if operand in registers:
            return registers[operand]
        if re.fullmatch(r"(?:0x[0-9a-fA-F]+|[0-9]+)", operand):
            return int(operand, 0)
        raise AssertionError("uninitialized PTX operand: " + operand)

    def put(operand, value):
        registers[name(operand)] = value

    for op, args in ops:
        if op == "mov.b64":
            if args[0].startswith("{"):
                lo, hi = args[0][1:-1].split(",")
                value = get(args[1])
                put(lo, value & MASK32)
                put(hi, value >> 32)
            elif args[1].startswith("{"):
                lo, hi = args[1][1:-1].split(",")
                put(args[0], get(lo) | (get(hi) << 32))
            else:
                put(args[0], get(args[1]))
        elif op.startswith("mov."):
            put(args[0], get(args[1]))
        elif op == "mul.wide.u32":
            a, b = get(args[1]), get(args[2])
            assert 0 <= a <= MASK32 and 0 <= b <= MASK32
            put(args[0], a * b)
        elif op.startswith("add"):
            bits = int(op.rsplit("u", 1)[1])
            value = get(args[1]) + get(args[2])
            if op.startswith("addc."):
                value += carry
            if ".cc." in op:
                carry = int(value >= (1 << bits))
            put(args[0], value & ((1 << bits) - 1))
        else:
            raise AssertionError(op)
    return registers


def inputs(a, b, c):
    return {"%" + str(base + i): (value >> (64 * i)) & MASK64
            for base, value in ((4, a), (8, b), (12, c)) for i in range(4)}


def result(registers, alias=None):
    base = 0 if alias is None else alias
    return sum(registers["%" + str(base + i)] << (64 * i) for i in range(4))


def reduction_model(product):
    """Independent bigint folds, retaining exactly the documented lost bits."""
    x = [(product >> (32 * i)) & MASK32 for i in range(16)]
    even = sum(x[8 + 2 * i] << (64 * i) for i in range(4))
    odd = sum(x[9 + 2 * i] << (64 * i) for i in range(4))
    f = (product & (B - 1)) + 977 * even
    g = (product >> 256) + 977 * odd
    lost_g = g >> 256
    first = f + ((g & (B - 1)) << 32)
    lost_z = first >> 288
    first &= (1 << 288) - 1
    low = first & (B - 1)
    second = (low & MASK96) + K * (first >> 256)
    lost_tail = second >> 96
    raw = (low & ~MASK96) | (second & MASK96)
    lost = (lost_g + lost_z) * (1 << 288) + lost_tail * (1 << 96)
    assert (product - raw - lost) % P == 0
    return raw, (lost_g, lost_z, lost_tail)


def main():
    here = Path(__file__).resolve().parent
    source = (here / "SeededMAC.cuh").read_text()
    ptx = extract_asm(source)
    product_ptx, reduction_ptx = ptx.split(FOLD_MARKER, 1)
    product_ops = parse(product_ptx)
    reduction_ops = parse(FOLD_MARKER + reduction_ptx)
    full_ops = product_ops + reduction_ops

    baseline_source = (here / "GPUMath.h").read_text()
    baseline_source = baseline_source[baseline_source.index("void _ModMultCore("):]
    # The ranked source selects C31, whose SECOND_FOLD_TAIL is empty. Requiring
    # this explicit default prevents silently auditing the wrong branch.
    assert re.search(r"#define\s+QSB_C31\s+1\b", (here / "pinning.cu").read_text())
    baseline_ptx = extract_asm(baseline_source)
    baseline_tail = parse(FOLD_MARKER + baseline_ptx.split(FOLD_MARKER, 1)[1])
    assert reduction_ops == baseline_tail, "seeded reducer differs from ranked baseline"
    # Device output registers are assigned only after every input operand read.
    first_output = next(i for i, (_, args) in enumerate(full_ops)
                        if args[0] in {"%0", "%1", "%2", "%3"})
    assert not any(re.search(r"%(?:[4-9]|1[0-5])\b", operand)
                   for _, args in full_ops[first_output:] for operand in args[1:])
    assert re.search(r"r\[0\]=r0;\s*r\[1\]=r1;\s*r\[2\]=r2;\s*r\[3\]=r3;", source)

    edge = [0, 1, 2, 976, 977, K - 1, K,
            (1 << 32) - 1, 1 << 32, (1 << 64) - 1, 1 << 64,
            (1 << 96) - 1, 1 << 96, (1 << 128) - 1, 1 << 128,
            B // 2 - 1, B // 2, P - 1, P, P + 1, B - 2, B - 1]
    rng = random.Random(0x5345454445442026)
    cases = itertools.chain(itertools.product(edge, repeat=3),
                            ((rng.getrandbits(256), rng.getrandbits(256),
                              rng.getrandbits(256)) for _ in range(10000)))
    counts = Counter()
    examples = []
    for case_index, (a, b, c) in enumerate(cases):
        corpus = "random" if case_index >= len(edge) ** 3 else "directed"
        initial = inputs(a, b, c)
        product_registers = execute(product_ops, initial)
        product = sum(product_registers["x" + str(i)] << (32 * i) for i in range(16))
        assert product == a * b + c, ("product mismatch", a, b, c)
        raw = result(execute(reduction_ops, product_registers))
        expected_raw, losses = reduction_model(product)
        assert raw == expected_raw, ("unexplained reduction mismatch", a, b, c)
        assert result(execute(baseline_tail, product_registers)) == raw
        counts["exact_product_cases"] += 1
        counts[corpus + "_cases"] += 1
        if raw % P != product % P:
            counts["inherited_reducer_boundary_differences"] += 1
            counts[corpus + "_field_differences"] += 1
            assert any(losses)
            if len(examples) < 2:
                examples.append({"a": hex(a), "b": hex(b), "c": hex(c),
                                 "lost_g288_z288_tail96": losses})
        else:
            counts["exact_field_cases"] += 1
        if counts["exact_product_cases"] % 97 == 0:
            for alias in (4, 8, 12):
                assert result(execute(full_ops, initial, alias), alias) == raw
                counts["device_alias_cases"] += 1

    # A corrupted seed mapping must be caught by the independent integer oracle.
    corrupted = parse(product_ptx.replace("mov.u64 bias, %12;", "mov.u64 bias, 0;", 1))
    bad = execute(corrupted, inputs(3, 5, 7))
    bad_product = sum(bad["x" + str(i)] << (32 * i) for i in range(16))
    assert bad_product != 3 * 5 + 7
    counts["negative_controls"] += 1
    assert counts["inherited_reducer_boundary_differences"] > 0
    counts.setdefault("random_field_differences", 0)
    print(json.dumps({"checks": dict(counts), "unexplained_errors": 0,
                      "GPU_execution": False, "boundary_examples": examples}, indent=2))


if __name__ == "__main__":
    main()
