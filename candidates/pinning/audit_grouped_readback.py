#!/usr/bin/env python3
"""Source-shape checks for host-drain / grouped hit readback on tip toggles."""
from pathlib import Path
src = (Path(__file__).resolve().parent / "pinning.cu").read_text()
assert "#define QSB_HOST_READBACK 1" in src
assert "#define QSB_SPARSE_TAIL 1" in src
assert "#define QSB_FINAL_TEMPLATE 1" in src
inner = src.split("Search all safe locktimes for this sequence")[1].split(
    "Progress every 10 sequences"
)[0]
assert "uint32_t hit_report[1 + 64];" in inner
assert "cudaMemcpy(hit_report, d_hit_cnt, sizeof(hit_report)," in inner
assert "fprintf(f, \"sequence=%u\\nlocktime=%u\\nhash_choice=%d\\nrecid=%d\\n\"" in src
assert "printf(\"  seq=0x%08X lt=%u hc=%d recid=%d\\n\"" not in src
assert "__ldg(" in src
assert "overwhelmingly common path" in src
assert "_PointAddXYZZT<true>" in src
assert "_SHA256TransformFastTail11" in src
print("PASS: host drain / grouped readback + tip toggles + hot-path")
