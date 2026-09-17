#!/usr/bin/env python3
"""Audit the nested tree while preserving qsb_field_mul's raw representatives."""

from pathlib import Path
import random


M = 1 << 256
C = (1 << 32) + 977
P = M - C
WIDTH = 256


def qmul(a, b):
    """Model qsb_field_mul, including its final carry fold but no subtract-p."""
    product = a * b
    first = (product & (M - 1)) + (product >> 256) * C
    second = (first & (M - 1)) + (first >> 256) * C
    low, carry = second & (M - 1), second >> 256
    assert carry <= 1
    result = low + carry * C
    assert result < M
    assert result % P == product % P
    return result


def normalize(value):
    assert 0 <= value < M
    return value - P if value >= P else value


def tree_up(values):
    assert len(values) == WIDTH
    products = list(values) + [0] * (WIDTH - 1)
    offset = 0
    for count in (256, 128, 64, 32, 16, 8, 4, 2):
        half = count // 2
        for tid in range(half):
            products[offset + count + tid] = qmul(
                products[offset + tid], products[offset + half + tid]
            )
        offset += count
    return products[:256], products[256:510], products[510]


def tree_down(leaves, internal, root_inverse):
    products = list(leaves) + list(internal) + [None]
    inverses = [0] * 255
    inverses[254] = root_inverse
    offset = 508
    for count in (2, 4, 8, 16, 32, 64, 128):
        half = count // 2
        for tid in range(count):
            parent = offset + count - WIDTH + (tid & (half - 1))
            inverses[offset - WIDTH + tid] = qmul(
                inverses[parent], products[offset + (tid ^ half)]
            )
        offset -= 2 * count
    return [
        normalize(qmul(inverses[tid & 127], products[tid ^ 128]))
        for tid in range(WIDTH)
    ]


def nested_inverse(values):
    groups = (len(values) + WIDTH - 1) // WIDTH
    saved = []
    roots = []
    for start in range(0, len(values), WIDTH):
        leaves = values[start:start + WIDTH] + [1] * (WIDTH - len(values[start:start + WIDTH]))
        leaves, internal, root = tree_up(leaves)
        saved.append((leaves, internal))
        roots.append(root)

    root_inverses = []
    for start in range(0, groups, WIDTH):
        chunk = roots[start:start + WIDTH]
        n = len(chunk)
        super_leaves = chunk + [1] * (WIDTH - n)
        super_leaves, super_internal, super_root = tree_up(super_leaves)
        assert normalize(super_root) != 0
        super_inverse = pow(normalize(super_root), P - 2, P)
        root_inverses.extend(
            tree_down(super_leaves, super_internal, super_inverse)[:n]
        )

    output = []
    for group, (leaves, internal) in enumerate(saved):
        output.extend(tree_down(leaves, internal, root_inverses[group]))
    return output[:len(values)]


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()
    block = source[source.index("void qsb_block_inverse("):source.index("#define QSB_CHECKPOINT_NODES")]
    assert block.index("qsb_field_normalize(root);") < block.index("_ModInv(root);")
    checkpoint = source[source.index("void qsb_block_inverse_checkpoint("):source.index("/* Batch the per-search-CTA roots")]
    assert "qsb_field_normalize(value);" in checkpoint
    assert "mul.lo.u32 k0, cf, 977" in source
    assert "addc.cc.u32 z1, z1, cf" in source


def main():
    audit_source()
    rng = random.Random(0x524157524F4F5453)
    edge = [1, 2, P - 2, P - 1, P + 1, P + 2, M - 2, M - 1]
    for count in (1, 2, 255, 256, 257, 511, 512, 513, 12055, 65536, 131072):
        values = []
        for i in range(count):
            value = edge[i % len(edge)] if i < 2 * len(edge) else rng.randrange(1, M)
            if value % P == 0:
                value = 1
            values.append(value)
        got = nested_inverse(values)
        assert all(value % P * inverse % P == 1 for value, inverse in zip(values, got))
        assert all(0 <= inverse < P for inverse in got)
        print(f"PASS: {count} raw roots")


if __name__ == "__main__":
    main()
