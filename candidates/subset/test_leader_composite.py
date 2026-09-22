#!/usr/bin/env python3
"""Static gate for the zero-spill subset leaderboard composite."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
FIELD = (HERE / "hit_filter_field_sc.cuh").read_text()
TREE = (HERE / "tests" / "gpu_epochs" / "tree.cu").read_text()
INV = (HERE / "tests" / "gpu_epochs" / "tree_inverse.cuh").read_text()
PAIR = (HERE / "tests" / "gpu_epochs" / "pair_shared.cuh").read_text()


def enabled(source: str, name: str) -> bool:
    return f"#define {name} 1" in source


assert enabled(TREE, "QSB_ISO_FAST_X")
assert enabled(INV, "QSB_ISO_FUSED_ROOT_SCALE")
assert enabled(FIELD, "QSB_SHORT_CARRY6")
assert enabled(FIELD, "QSB_CHAIN_MUL_LEAN")

# Six lean hot sites use the measured first-fold carry cut.  The default f9
# fallback retains the donor cut, giving seven of nine hot first-fold sites.
for prefix in ("f0", "f2", "f5", "f8", "f13", "f15"):
    assert f"addc.u64 {prefix}_f3" in FIELD, prefix
    assert f"mov.u32 {prefix}_z8, {prefix}_w7" in FIELD, prefix
for prefix in ("f6", "f7"):
    assert f"addc.cc.u64 {prefix}_f3" in FIELD, prefix
    assert f"addc.u32 {prefix}_z8, {prefix}_w7, 0" in FIELD, prefix
assert "#if QSB_CHAIN_MUL_LEAN >= 2" in FIELD
assert "addc.u32 f9_z8, 0, f9_w7" in FIELD

# Exact publication remains separate from the approximate nomination filter.
assert "kernel_verify_pair_hits" in TREE
assert "qsb_pair_verify_candidate" in PAIR
assert "chain_replay_field.cuh" in TREE

for source in (FIELD, TREE, INV, PAIR):
    assert "<<<<<<<" not in source and ">>>>>>>" not in source

base = 623_518_629
exact_and_short6 = 1.00832443
first_fold = 1.00380523
factor = exact_and_short6 * (1 + (first_fold - 1) * 7 / 9)
projection = round(base * factor)
floor = round(base * 1.01)
assert projection > floor
print(
    {
        "defaults": {
            "iso_fast_x": True,
            "iso_fused_root_scale": True,
            "short_carry6": True,
            "chain_mul_lean": True,
        },
        "first_fold_hot_sites": "7/9",
        "projection": projection,
        "one_percent_floor": floor,
        "modeled_margin": projection - floor,
    }
)
