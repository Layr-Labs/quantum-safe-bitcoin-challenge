#!/usr/bin/env python3
"""Guard the deliberate removal of the officially negative SHA256d32 helper."""

import hashlib
from pathlib import Path

EXPECTED_SOURCE = "ceb288d2eeec49efd86dd3ded61f480a235f8f343b1a70dbfc201052ef622498"


def main():
    source = Path(__file__).with_name("pinning.cu").read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == EXPECTED_SOURCE
    compact = source.replace(" ", "")
    assert "_SHA256TransformSha256d32" not in source
    assert source.count("uint32_t b2[16]") == 1
    assert compact.count("for(inti=0;i<8;i++)b2[i]=state[i];") == 1
    assert compact.count("b2[8]=0x80000000u;") == 1
    assert compact.count("for(inti=9;i<15;i++)b2[i]=0;") == 1
    assert compact.count("b2[15]=256;") == 1
    assert compact.count("_SHA256Transform(s2,b2);") == 1
    print("PASS: SHA256d32 specialization absent; exact generic 32-byte pad restored")


if __name__ == "__main__":
    main()
