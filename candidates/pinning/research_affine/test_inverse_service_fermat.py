#!/usr/bin/env python3
"""Symbolic/numerical model of the exact Fermat source chain, not GPU execution."""
from pathlib import Path
import random
import re

P = (1 << 256) - (1 << 32) - 977
source = Path(__file__).with_name("inverse_service_fermat_compile.cu").read_text()
body = source.split("struct FermatInverse", 1)[1].split("norm(t);", 1)[0]
ops = re.findall(r"(cp|sq|qsb_field_mul)\(([^()]*)\);", body)
assert len(ops) == 38, f"unexpected source chain shape: {len(ops)}"


def evaluate(value=None):
    # With value=None, register values are exponents, never reduced modulo p-1.
    r = {"x": 1 if value is None else value % P}
    squares = products = 0
    for op, arguments in ops:
        args = arguments.split(",")
        if op == "cp":
            r[args[0]] = r[args[1]]
        elif op == "sq":
            n = int(args[1]); squares += n
            r[args[0]] = r[args[0]] << n if value is None else pow(r[args[0]], 1 << n, P)
        else:
            products += 1
            a, b = r[args[1]], r[args[2]]
            r[args[0]] = a + b if value is None else a * b % P
    return r["t"], squares, products


exponent, squares, products = evaluate()
assert exponent == P - 2, hex(exponent)
assert (squares, products) == (257, 15), (squares, products)
rng = random.Random(20260923)
values = [0, 1, 2, P - 2, P - 1, P, P + 1, (1 << 256) - 1]
values.extend(rng.randrange(1 << 256) for _ in range(128))
for value in values:
    result, _, _ = evaluate(value)
    assert result == pow(value, P - 2, P), (value, result)
    if value % P:
        assert result * value % P == 1
print(f"PASS: exact exponent p-2, {squares} squares + {products} products, {len(values)} modular reference cases; no GPU execution")
