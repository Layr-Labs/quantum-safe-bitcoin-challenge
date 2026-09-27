#!/usr/bin/env python3
"""Audit the compile-time single-hash contract of the ranked fast kernel."""

from pathlib import Path


def host_selects_fast(single_hash, easy, suffix_len, seq_offset, lt_offset, total_len):
    return (
        single_hash
        and not easy
        and suffix_len == 75
        and seq_offset == 31
        and lt_offset == 67
        and total_len == 9995
    )


def second_hash_executes(fast_tail, single_hash):
    return not (fast_tail or single_hash)


def audit_truth_table():
    layouts = (
        (75, 31, 67, 9995),
        (74, 31, 67, 9995),
        (75, 30, 67, 9995),
        (75, 31, 66, 9995),
        (75, 31, 67, 9994),
    )
    checked = 0
    for single_hash in (False, True):
        for easy in (False, True):
            for layout in layouts:
                fast = host_selects_fast(single_hash, easy, *layout)
                assert not fast or (single_hash and not easy)
                assert second_hash_executes(fast, single_hash) == (not single_hash)
                checked += 1
    return checked


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()
    select = source[source.index("const bool fast_tail ="):source.index("if (fast_tail) {", source.index("const bool fast_tail ="))]
    for condition in (
        "single_hash && !easy",
        "pp.suffix_len == 75",
        "pp.seq_offset == 31",
        "pp.lt_offset == 67",
        "pp.total_preimage_len == 9995",
    ):
        assert condition in select

    loop = source[source.index("/* Check both pubkeys"):source.index("template<bool FAST_TAIL>")]
    guard = "if (FAST_TAIL || single_hash) continue;"
    assert loop.count(guard) == 1
    assert loop.index("if(vv){") < loop.index(guard)
    assert loop.index(guard) < loop.index("uint8_t pp[64]")
    assert source.count("launch_pinning_pipeline<true>") == 1
    assert source.count("launch_pinning_pipeline<false>") == 1


def main():
    audit_source()
    cases = audit_truth_table()
    print(
        f"PASS: fast-tail single-hash contract; {cases} host selection cases; "
        "second hash executes exactly when single_hash is false"
    )


if __name__ == "__main__":
    main()
