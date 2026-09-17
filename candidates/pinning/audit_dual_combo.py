#!/usr/bin/env python3
"""Bind exact PR94 dual code and the sparse-Pk33 + Fast11 contract."""

import hashlib
from pathlib import Path

PR94_HEAD = "3402706d0b7defb80ffe950f2ee0347e29e4379b"
PR94_FULL_PINNING_SHA256 = "85196a48f9b87bb2546688106009ec49be0015aac825253dbd0e8f82da5e3029"
PR94_DUAL_HELPER_SHA256 = "ef0e290d86cc082d50aff5a6217cc157f7c55da3a4418acfc92a39bbdfb738fd"
PR94_DUAL_AUDIT_SHA256 = "12659d9b68e648d9abaf9d63622f1c971a1d9854357538eefdd71b517e8dedd7"
FINAL_PINNING_SHA256 = "ceb288d2eeec49efd86dd3ded61f480a235f8f343b1a70dbfc201052ef622498"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    source = (root / "pinning.cu").read_text()
    audit = (root / "audit_dual_pk33.py").read_bytes()
    assert digest(source.encode()) == FINAL_PINNING_SHA256

    begin = source.index("/* Hash both recovered compressed keys")
    end_token = "#undef QSB_SHA256_DUAL_STEP\n"
    end = source.index(end_token, begin) + len(end_token)
    assert digest(source[begin:end].encode()) == PR94_DUAL_HELPER_SHA256
    assert digest(audit) == PR94_DUAL_AUDIT_SHA256

    assert source.count("__device__ __forceinline__ void _SHA256TransformPk33Dual") == 1
    assert source.count("_SHA256TransformPk33Dual(hs0,hs1,q1x,q2x,y_parities);") == 1
    assert source.count("__device__ __forceinline__ void _SHA256TransformFastTail11") == 1
    assert source.count("_SHA256TransformFastTail11(state,blk);") == 1
    assert "_SHA256TransformSha256d32" not in source
    assert source.count("uint32_t b2[16]") == 1
    assert source.replace(" ", "").count("_SHA256Transform(s2,b2);") == 1
    assert source.count("__device__ __forceinline__ void _SHA256TransformPk33(") == 1
    assert source.count("_SHA256TransformPk33(hs,pb);") == 1

    prepare_tail = source.index("_SHA256TransformFastTail11(state,blk);")
    prepare_inner = source.index("_SHA256Transform(s2,b2);")
    finish_dual = source.index("_SHA256TransformPk33Dual(hs0,hs1,q1x,q2x,y_parities);")
    finish_fallback = source.index("_SHA256TransformPk33(hs,pb);")
    assert prepare_tail < prepare_inner < finish_dual < finish_fallback

    dual_branch = source[finish_dual:source.index("/* Check both pubkeys", finish_dual)]
    assert "if(gpu_bench_valid_words(hs0))ri=0;" in dual_branch
    assert "else if(gpu_bench_valid_words(hs1))ri=1;" in dual_branch
    assert "return;" in dual_branch
    assert "if (FAST_TAIL || single_hash) continue;" in source

    print("PASS: exact PR94 dual helper/audit, generic SHA256d32, Fast11, and dual/fallback contract")


if __name__ == "__main__":
    main()
