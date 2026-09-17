#!/usr/bin/env python3
"""Guard removal of the officially negative dual-Pk33 rider."""

import hashlib
from pathlib import Path


EXPECTED_SOURCE = "ed4339c8ec82bbc438128d100a4ea07cdfda899b8ecbfa14b2080a6c8c63938c"


def main():
    source = Path(__file__).with_name("pinning.cu").read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == EXPECTED_SOURCE

    # PR94's helper, macros, and early FAST_TAIL return must all be absent.
    for forbidden in (
        "_SHA256TransformPk33Dual",
        "QSB_SHA256_DUAL_",
        "uint32_t hs0[8],hs1[8]",
        "gpu_bench_valid_words(hs0)",
        "gpu_bench_valid_words(hs1)",
    ):
        assert forbidden not in source, f"dual-Pk33 residue remains: {forbidden}"

    # Both recovered keys must flow through the established single-message
    # sparse helper; FAST_TAIL then skips only the second-hash iteration.
    assert source.count("__device__ __forceinline__ void _SHA256TransformPk33(") == 1
    assert source.count("uint32_t hs[8];_SHA256TransformPk33(hs,pb);") == 1
    assert source.count("for(int ri=0;ri<2;ri++)") == 1
    assert "uint64_t sx0=ri ? q2x[0] : q1x[0];" in source
    assert "pb[0]=__byte_perm(x7,0x2+(uint8_t)((y_parities>>ri)&1u),0x4321);" in source
    assert "if (FAST_TAIL || single_hash) continue;" in source
    assert "d_hit_idx[pos]=(lt & 0x7FFFFFFFu)|(ri<<31);" in source

    # The two orthogonal measured mechanisms and the original hit reporting
    # remain present.  The unresolved no-hit stdout rider is not bundled.
    assert source.count("__device__ __forceinline__ void _SHA256TransformFastTail11") == 1
    assert source.count("_SHA256TransformFastTail11(state,blk);") == 1
    assert source.count("*** HIT! seq=0x%08X ***") == 1
    assert source.count("seq=0x%08X lt=%u hc=%d recid=%d") == 1

    print("PASS: dual-Pk33 absent; both keys use single sparse-Pk33; Fast11 and original hit reporting remain")


if __name__ == "__main__":
    main()
