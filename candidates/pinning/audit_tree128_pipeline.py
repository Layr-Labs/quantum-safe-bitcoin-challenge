#!/usr/bin/env python3
"""Bounded CPU/source audit for the 128-leaf candidate tree pipeline.

The candidate tree is 128 leaves per search CTA while the root-group tree
remains 256 leaves.  This mirrors qsb_field_mul's raw pseudo-Mersenne
representatives, the identity substitution for zero/inactive lanes, the
checkpoint packing, and the independent 256-root super-tree CTAs.
"""

from pathlib import Path
import random
import re


MODULUS = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
WORD = 1 << 256
FOLD = (1 << 32) + 977
CANDIDATE_WIDTH = 128
ROOT_WIDTH = 256


def field_mul(a, b):
    """Model qsb_field_mul, retaining its raw (<2^256) representative."""
    product = a * b
    folded = (product & (WORD - 1)) + (product >> 256) * FOLD
    folded = (folded & (WORD - 1)) + (folded >> 256) * FOLD
    low, carry = folded & (WORD - 1), folded >> 256
    result = low + carry * FOLD
    assert carry <= 1 and result < WORD
    assert result % MODULUS == product % MODULUS
    return result


def normalize(value):
    assert 0 <= value < WORD
    return value - MODULUS if value >= MODULUS else value


def packed_up(leaves, width):
    """Return leaves, packed non-root checkpoints, and the raw root."""
    assert len(leaves) == width and width & (width - 1) == 0
    products = list(leaves) + [0] * (width - 1)
    offset = 0
    count = width
    while count > 1:
        half = count // 2
        for tid in range(half):
            products[offset + count + tid] = field_mul(
                products[offset + tid], products[offset + half + tid]
            )
        offset += count
        count //= 2
    assert offset == 2 * width - 2
    return products[:width], products[width : 2 * width - 2], products[2 * width - 2]


def reference_up(leaves):
    """Independent level-by-level representation of the packed up-tree."""
    level = list(leaves)
    checkpoints = []
    while len(level) > 1:
        half = len(level) // 2
        level = [field_mul(level[i], level[half + i]) for i in range(half)]
        if len(level) > 1:
            checkpoints.extend(level)
    return checkpoints, level[0]


def packed_down(leaves, checkpoints, root_inverse, width):
    """Expand one root inverse and normalize each returned leaf inverse."""
    assert len(leaves) == width and len(checkpoints) == width - 2
    products = list(leaves) + list(checkpoints) + [None]
    inverses = [0] * (width - 1)
    inverses[width - 2] = root_inverse
    offset = 2 * width - 4
    count = 2
    while count < width:
        half = count // 2
        for tid in range(count):
            parent = offset + count - width + (tid & (half - 1))
            inverses[offset - width + tid] = field_mul(
                inverses[parent], products[offset + (tid ^ half)]
            )
        offset -= 2 * count
        count *= 2
    assert offset == 0
    return [
        normalize(
            field_mul(
                inverses[tid & (width // 2 - 1)], products[tid ^ (width // 2)]
            )
        )
        for tid in range(width)
    ]


def usable_factor(value, active):
    # The CUDA caller tests the raw limbs for zero before substituting identity.
    return value if active and value != 0 else 1


def nested_inverse(values, active):
    """Model candidate trees plus one independent root tree per 256 roots."""
    candidate_records = []
    candidate_roots = []
    for start in range(0, len(values), CANDIDATE_WIDTH):
        raw = values[start : start + CANDIDATE_WIDTH]
        flags = active[start : start + CANDIDATE_WIDTH]
        leaves = [usable_factor(v, a) for v, a in zip(raw, flags)]
        leaves += [1] * (CANDIDATE_WIDTH - len(leaves))
        record = packed_up(leaves, CANDIDATE_WIDTH)
        candidate_records.append(record)
        candidate_roots.append(record[2])

    root_records = []
    root_inverses = []
    for start in range(0, len(candidate_roots), ROOT_WIDTH):
        roots = candidate_roots[start : start + ROOT_WIDTH]
        roots += [1] * (ROOT_WIDTH - len(roots))
        record = packed_up(roots, ROOT_WIDTH)
        root_records.append(record)
        super_inverse = pow(normalize(record[2]), MODULUS - 2, MODULUS)
        root_inverses.extend(packed_down(record[0], record[1], super_inverse, ROOT_WIDTH))

    output = []
    for index, (leaves, checkpoints, root) in enumerate(candidate_records):
        root_group = index // ROOT_WIDTH
        root_slot = index % ROOT_WIDTH
        output.extend(
            packed_down(
                leaves,
                checkpoints,
                root_inverses[root_group * ROOT_WIDTH + root_slot],
                CANDIDATE_WIDTH,
            )
        )
    return output[: len(values)], candidate_records, root_records


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()
    for name, value in (
        ("QSB_TREE_N", "128"),
        ("QSB_CAND_STRIDE", "QSB_TREE_N"),
        ("QSB_S0_THREADS", "QSB_TREE_N"),
        ("QSB_S0_BLOCKS", "4"),
        ("QSB_S2_THREADS", "QSB_TREE_N"),
        ("QSB_S2_BLOCKS", "6"),
    ):
        assert re.search(rf"^#define\s+{name}\s+{re.escape(value)}$", source, re.MULTILINE)

    product = source[source.index("template<int N>\n__device__ __forceinline__ void qsb_block_product_checkpoint") : source.index("template<int N>\n__device__ __forceinline__ void qsb_block_inverse_checkpoint")]
    inverse = source[source.index("template<int N>\n__device__ __forceinline__ void qsb_block_inverse_checkpoint") : source.index("/* Batch the per-search-CTA roots")]
    for text in (product, inverse):
        assert "static_assert(N >= 2" in text
        assert "4u*N" in text
        assert "[4][2*N]" in text
    assert "for(int count=N;count>1;count>>=1)" in product
    assert "if(node<2*N-2)" in product
    assert "node-N" in product
    assert "products[k][2*N-2]" in product
    assert "if(tid<N-2)" in inverse
    assert "products[k][N+tid]" in inverse
    assert "inverses[k][N-2]" in inverse
    assert "for(int count=2;count<N;count<<=1)" in inverse
    assert "tid&(N/2-1)" in inverse and "tid^(N/2)" in inverse

    assert source.count("qsb_block_product_checkpoint<256>(") == 1
    assert source.count("qsb_block_inverse_checkpoint<256>(") == 1
    assert source.count("qsb_block_product_checkpoint<QSB_TREE_N>(") == 1
    assert source.count("qsb_block_inverse_checkpoint<QSB_TREE_N>(") == 1
    super_begin = source.index("__global__ void __launch_bounds__(256,1) qsb_invert_super_roots")
    super_end = source.index("__global__ void __launch_bounds__(256,2) qsb_root_group_finish", super_begin)
    assert "blockIdx.x*256u+threadIdx.x" in source[super_begin:super_end]

    launch_begin = source.index("static void launch_pinning_pipeline(")
    launch_end = source.index("/* ============================================================", launch_begin)
    launch = source[launch_begin:launch_end]
    ordered = (
        "kernel_pinning_pipeline<FAST_TAIL,0>",
        "qsb_root_group_prepare<<<root_groups,256>>>",
        "qsb_invert_super_roots<<<(root_groups+255)/256,256>>>",
        "qsb_root_group_finish<<<root_groups,256>>>",
        "kernel_pinning_pipeline<FAST_TAIL,2>",
    )
    positions = [launch.index(name) for name in ordered]
    assert positions == sorted(positions)
    assert "int blocks=(batch_size+QSB_TREE_N-1)/QSB_TREE_N;" in launch
    assert "int blocks0=(batch_size+QSB_S0_THREADS-1)/QSB_S0_THREADS;" in launch
    assert "int blocks2=(batch_size+QSB_S2_THREADS-1)/QSB_S2_THREADS;" in launch
    assert "root_groups>256" not in launch

    assert "int GRDSZ = (BATCH+QSB_TREE_N-1)/QSB_TREE_N;" in source
    assert "GRDSZ*4u*QSB_CAND_STRIDE*sizeof(uint64_t)" in source
    assert "size_t cold = (size_t)gt_entries(0) * 64u;" in source
    assert "size_t skip = (gt_sz > cold + want) ? cold : 0;" in source
    assert "(void *)(d_gt + skip)" in source
    assert "if(count>2)__syncthreads();" in product
    assert inverse.count("__syncthreads();") >= 2


def audit_batch(values, active):
    got, candidate_records, root_records = nested_inverse(values, active)
    expected = []
    for value, is_active in zip(values, active):
        factor = usable_factor(value, is_active)
        expected.append(pow(factor % MODULUS, MODULUS - 2, MODULUS))
    assert got == expected

    for leaves, checkpoints, root in candidate_records + root_records:
        ref_checkpoints, ref_root = reference_up(leaves)
        assert checkpoints == ref_checkpoints
        assert root == ref_root
    return len(candidate_records), len(root_records)


def audit_launch_geometry():
    def geometry(batch_size):
        blocks = (batch_size + CANDIDATE_WIDTH - 1) // CANDIDATE_WIDTH
        root_groups = (blocks + ROOT_WIDTH - 1) // ROOT_WIDTH
        return blocks, root_groups, (root_groups + ROOT_WIDTH - 1) // ROOT_WIDTH

    expected = {
        1: (1, 1, 1),
        127: (1, 1, 1),
        128: (1, 1, 1),
        129: (2, 1, 1),
        32768: (256, 1, 1),
        32769: (257, 2, 1),
        8388608: (65536, 256, 1),
        8388609: (65537, 257, 2),
        16777216: (131072, 512, 2),
    }
    for batch_size, want in expected.items():
        assert geometry(batch_size) == want


def audit_shifted_l2_window():
    table_size = (1 << 20) * 64
    cold = (1 << 17) * 64

    def window(max_persist, max_window):
        want = min(table_size, max_persist)
        if want <= 0 or max_window <= 0:
            return None
        skip = cold if table_size > cold + want else 0
        available = table_size - skip
        return skip, min(want, available, max_window)

    assert window(50 * 1024 * 1024, 128 * 1024 * 1024) == (
        cold,
        50 * 1024 * 1024,
    )
    # When the budget consumes the table's tail, the guard keeps the base at
    # d_gt rather than creating an out-of-range shifted window.
    assert window(60 * 1024 * 1024, 128 * 1024 * 1024) == (0, 60 * 1024 * 1024)
    assert window(50 * 1024 * 1024, 0) is None
    assert window(0, 128 * 1024 * 1024) is None


def main():
    audit_source()
    audit_launch_geometry()
    audit_shifted_l2_window()
    rng = random.Random(0x54524545313238)
    edge = [0, 1, 2, MODULUS - 2, MODULUS - 1, MODULUS + 1, MODULUS + 2, WORD - 1]
    cases = 0
    candidate_trees = root_trees = 0

    # Boundaries cover zero, noncanonical raw representatives, mixed active
    # lanes, a partial 128-leaf CTA, and a partial 256-root super-tree CTA.
    for count in (1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 4097):
        values = []
        for index in range(count):
            value = edge[index % len(edge)] if index < len(edge) else rng.randrange(1, WORD - 1)
            if value != 0 and value % MODULUS == 0:
                value = 1
            values.append(value)
        active = [True] * count
        if count > 7:
            active[3] = False
            active[-1] = False
        ctrees, rtrees = audit_batch(values, active)
        candidate_trees += ctrees
        root_trees += rtrees
        cases += 1

    for _ in range(32):
        count = rng.randrange(1, 2049)
        values = []
        for _ in range(count):
            value = rng.randrange(0, WORD)
            if value != 0 and value % MODULUS == 0:
                value = 1
            values.append(value)
        active = [rng.randrange(8) != 0 for _ in values]
        ctrees, rtrees = audit_batch(values, active)
        candidate_trees += ctrees
        root_trees += rtrees
        cases += 1

    # Exercise the root inverse's multi-CTA partition independently of an
    # 8-million-candidate Python expansion: two 256-root CTAs, with a tail.
    roots = [rng.randrange(1, WORD) for _ in range(ROOT_WIDTH + 1)]
    roots = [1 if value != 0 and value % MODULUS == 0 else value for value in roots]
    for start in (0, ROOT_WIDTH):
        leaves = roots[start : start + ROOT_WIDTH] + [1] * (ROOT_WIDTH - len(roots[start : start + ROOT_WIDTH]))
        packed = packed_up(leaves, ROOT_WIDTH)
        inverse = pow(normalize(packed[2]), MODULUS - 2, MODULUS)
        got = packed_down(packed[0], packed[1], inverse, ROOT_WIDTH)
        assert all((value % MODULUS) * result % MODULUS == 1 for value, result in zip(leaves, got))

    print(
        f"PASS: source geometry; {cases} partial/mixed batches; "
        f"{candidate_trees} candidate checkpoints and {root_trees} root checkpoints; "
        "zero/noncanonical/raw-representative inverses; launch/L2 guard boundaries"
    )


if __name__ == "__main__":
    main()
