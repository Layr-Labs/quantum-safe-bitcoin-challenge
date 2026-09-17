#!/usr/bin/env python3
"""Source-shape checks for the host-drain cleanup."""
from pathlib import Path
src = (Path(__file__).resolve().parent / "pinning.cu").read_text()
inner = src.split("Search all safe locktimes for this sequence")[1].split(
    "Progress every 10 sequences"
)[0]
assert "cudaDeviceSynchronize()" not in inner
assert "cudaMemset(d_hit_cnt, 0, 4)" in inner
assert "cudaMemcpy(&h_hit, d_hit_cnt, 4, cudaMemcpyDeviceToHost)" in inner
assert 'fprintf(f, "sequence=%u\\nlocktime=%u\\nhash_choice=%d\\nrecid=%d\\n"' in inner
assert "printf(\"  seq=0x%08X lt=%u hc=%d recid=%d\\n\"" not in inner
print("PASS: host drain cleanup source shape")
