#!/usr/bin/env python3
"""Source-bound magnitude feasibility screen; no field implementation or timing.

Use the libsecp256k1 magnitude contracts as an abstract interpretation of our
actual deferred point formulas. Multiplication/square are assumed to return
magnitude 1, addition sums magnitudes, and a-b is add(a,negate(b,m_b)), whose
magnitude is m_a+m_b+1. This proves a bound only conditional on those primitive
contracts. It does not validate any proposed CUDA primitive or reduction.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
TRACK = HERE.parents[1]


def extract(source, name):
    start = source.index(name + "(")
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[brace + 1:end - 1]


def operations(body, defer=None):
    body = re.sub(r"/\*.*?\*/|//[^\n]*", "", body, flags=re.S)
    if defer is not None:
        pat = r"if\s*\(DEFER_Y\)\s*\{([^{}]*)\}\s*else\s*\{([^{}]*)\}"
        body, count = re.subn(pat, lambda m: m[1] if defer else m[2], body)
        assert count == 1, "unknown deferred branch"
    body = re.sub(r"\(uint64_t\s*\*\)", "", body)
    body = re.sub(r"uint64_t\s+\w+\[4\]\s*;", "", body)
    ops = []
    pat = r"(_ModMult|_ModSqr|_ModAdd256|_ModSub256|Load256)\(([^()]*)\)\s*;"
    for match in re.finditer(pat, body):
        args = [s.strip() for s in match[2].split(",")]
        assert all(re.fullmatch(r"\w+", s) for s in args)
        ops.append((match[1], args))
    assert not re.sub(pat, "", body).strip(), "unrecognized formula statements"
    return ops


def column_bound(m1, m2, radix, limbs, top_bits):
    """Raw schoolbook convolution bounds, BEFORE reduction/extra fold terms."""
    a = [2 * m1 * ((1 << radix) - 1)] * (limbs - 1)
    b = [2 * m2 * ((1 << radix) - 1)] * (limbs - 1)
    a.append(2 * m1 * ((1 << top_bits) - 1))
    b.append(2 * m2 * ((1 << top_bits) - 1))
    cols = [sum(a[i] * b[j] for i in range(limbs) for j in range(limbs)
                if i + j == k) for k in range(2 * limbs - 1)]
    return {"max_input_limb_bits": max(a + b).bit_length(),
            "max_raw_column_bits": max(cols).bit_length()}


def trace(seed, mixed, windows, weak_seed):
    history, violations, counts = [], [], Counter()
    state = {name: 1 for name in ("X1", "Y1", "X2", "Y2")}
    state.update(ZZ1=1, ZZZ1=1, Yoff=1)

    def execute(ops, label):
        for index, (op, args) in enumerate(ops):
            if weak_seed and label == "seed" and op == "_ModSub256" and args == ["Q", "Q", "T"]:
                history.append({"stage": label, "op": "normalize_weak", "dest": "T",
                                "input_magnitudes": [state["T"]], "output_magnitude": 1})
                counts["normalize_weak"] += 1
                state["T"] = 1
            dest = args[0]
            src = args[1:] if len(args) == 3 or op in ("Load256", "_ModSqr") else args
            mags = [state[s] for s in src]
            if op in ("_ModMult", "_ModSqr"):
                if max(mags) > 8:
                    violations.append({"stage": label, "index": index, "op": op,
                                       "operands": src, "magnitudes": mags})
                result = 1
            elif op == "_ModAdd256":
                result = sum(mags)
            elif op == "_ModSub256":
                result = sum(mags) + 1
            elif op == "Load256":
                result = mags[0]
            else:
                raise AssertionError(op)
            assert result <= 32
            state[dest] = result
            counts[op] += 1
            history.append({"stage": label, "op": op, "dest": dest,
                            "input_magnitudes": mags, "output_magnitude": result})

    execute(seed, "seed")
    for dest, src in (("X1", "X3"), ("Y1", "Y3"), ("ZZ1", "ZZ3"), ("ZZZ1", "ZZZ3")):
        state[dest] = state[src]
    for step in range(2, windows):
        state.update(X2=1, Y2=1, Yoff=1)
        execute(mixed[step != windows - 1], f"add_{step}")
    return {"counts": dict(counts), "violations": violations,
            "final_magnitudes": {x: state[x] for x in ("X1", "Y1", "ZZ1", "ZZZ1")},
            "operations": history}


def main():
    source = (TRACK / "GPUMath.h").read_text()
    seed_body = extract(source, "_PointAddXYZZ_mm")
    mixed_body = extract(source, "_PointAddXYZZ")
    seed = operations(seed_body)
    mixed = {defer: operations(mixed_body, defer) for defer in (False, True)}
    # Check the submitted chain really has ten windows and one final exact add.
    geometry = (TRACK / "wide_geometry.cuh").read_text()
    assert re.search(r"WIDE_CHUNKS\s*=\s*10", geometry), "geometry changed"
    scalar_body = extract((TRACK / "pinning.cu").read_text(), "_FixedBaseSignedXYZZScalar")
    assert scalar_body.count("_PointAddXYZZ_mm(") == 1
    assert scalar_body.count("_PointAddXYZZ<true>(") == 1
    assert scalar_body.count("_PointAddXYZZ<false>(") == 1
    assert re.search(r"c\s*=\s*2\s*;\s*c\s*<\s*WIDE_CHUNKS\s*-\s*1", scalar_body)
    literal = trace(seed, mixed, 10, False)
    bounded = trace(seed, mixed, 10, True)
    assert [(x["stage"], x["magnitudes"]) for x in literal["violations"]] == [
        ("seed", [9, 3]), ("add_2", [9]), ("add_2", [1, 9])]
    assert not bounded["violations"]
    expected = {"_ModMult": 60, "_ModSqr": 18, "_ModAdd256": 16, "_ModSub256": 47}
    assert all(bounded["counts"][k] == v for k, v in expected.items())
    assert bounded["counts"]["normalize_weak"] == 1
    layouts = []
    for name, radix, limbs, top, word in [("5x52", 52, 5, 48, 64),
                                         ("9x29", 29, 9, 24, 32),
                                         ("10x26", 26, 10, 22, 32)]:
        bounds = []
        for x in bounded["operations"]:
            if x["op"] in ("_ModMult", "_ModSqr"):
                mags = x["input_magnitudes"]
                bounds.append(column_bound(mags[0], mags[-1], radix, limbs, top))
        # Raw full products, with schoolbook decomposition to <=32-bit limbs.
        # This is arithmetic accounting, not SASS, issue count or timing.
        components = (radix + 31) // 32
        mul_products = (limbs * components) ** 2
        square_products = limbs * (limbs + 1) // 2 * components ** 2
        layouts.append({"name": name, "physical_storage_words32": limbs * (word // 32),
                        "max_input_limb_bits": max(x["max_input_limb_bits"] for x in bounds),
                        "max_raw_column_bits": max(x["max_raw_column_bits"] for x in bounds),
                        "limb_storage_fits_conservative_contract": max(x["max_input_limb_bits"] for x in bounds) <= word,
                        "schoolbook_raw_narrow_products_mul": mul_products,
                        "limb_triangular_raw_narrow_products_square": square_products,
                        "chain_raw_narrow_products": 60 * mul_products + 18 * square_products})
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {p: hashlib.sha256((TRACK / p).read_bytes()).hexdigest()
                          for p in ("GPUMath.h", "pinning.cu", "wide_geometry.cuh")},
        "function_sha256": {"seed": hashlib.sha256(seed_body.encode()).hexdigest(),
                            "mixed": hashlib.sha256(mixed_body.encode()).hexdigest()},
        "literal_conversion": literal, "one_weak_seed_normalization": bounded,
        "layouts": layouts,
        "current_raw_products": {"mul": 64, "square": 36, "chain": 60 * 64 + 18 * 36},
        "assumptions": ["Canonical affine inputs and anchors are assigned magnitude1 conservatively.",
                        "The input/output bounds of hypothetical primitives are contracts, not implemented GPU code.",
                        "One weak normalization is inserted on the seed T before Q-T; source production is unchanged.",
                        "Radix column bounds exclude reductions, fold terms, intermediate carry and conversions.",
                        "Raw product decompositions assume schoolbook limb multiplication; optimized diagonal squaring or other schedules can differ."],
        "decision": "Defer a GPU rewrite. A bounded formula schedule exists for5x52 and10x26, but substantial net savings are not established.9x29 violates32-bit limb bounds under this contract and needs a different normalization schedule or tighter primitive bounds.",
        "limits": "Source-extracted conditional magnitude/cost screen only; no CUDA implementation, field arithmetic correctness test, GPU execution, timing or measured regression. Recovery/inverse/checkpoint boundaries are not yet analyzed.",
    }
    (HERE / "screen-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"result": "PASS", "literal_violations": literal["violations"],
                      "bounded_counts": bounded["counts"], "layouts": layouts}, indent=2))


if __name__ == "__main__":
    main()
