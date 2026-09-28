#!/usr/bin/env python3
"""Fail loudly when the embedded sm_89 carrier was not built from this source.

Why this exists. The ranked path executes the carrier image in
qsb_carrier_sm89.h, not the PTX beside it, so the carrier is a SECOND COPY of the
kernel and can silently disagree with pinning.cu. QsbCarrier.h guards that with a
geometry stamp -- a compile-time constant folded from the table geometry, compared
host-side against the running binary. That guard is necessary and it works, but it
is BLIND to any source change that does not move the geometry: a behavioural edit
to the finish path, a schedule change, a SHA change. For those, staleness is
invisible at runtime, and the symptom is not a crash but a run that grinds at full
speed and publishes nothing (H79: every post-carrier run published zero hits while
the last pre-carrier run published 133,863 verified).

build_carrier.sh already records the fingerprint of exactly the source it compiled
into the header's first comment lines. This test recomputes that fingerprint and
compares it, so a carrier that is stale for ANY reason -- not just a moved table --
is caught at the only moment it is cheap: before a rental, before a submission.

No GPU, no nvcc, no network. SPDX-License-Identifier: GPL-3.0-only
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import subprocess

HERE = Path(__file__).resolve().parent
CARRIER = HERE / "qsb_carrier_sm89.h"
RECORDED_SRC = re.compile(r"source sha256 ([0-9a-f]{64})")
RECORDED_CUBIN = re.compile(r"cubin sha256 ([0-9a-f]{64})")


def source_fingerprint() -> str:
    """build_carrier.sh's `SRC_SHA`, verbatim, by running its own pipeline.

    Deliberately NOT reimplemented in Python. That pipeline is
    `cat pinning.cu *.cuh $(ls *.h | grep -v qsb_carrier_sm89.h) | LC_ALL=C sort | sha256sum`,
    and `LC_ALL=C sort` over a concatenated byte stream is not what a naive Python
    reimplementation produces -- the first attempt here hashed the files in name order, and a
    second attempt emulated `sort` line-wise; both disagreed with the shell. A guard that
    disagrees with the builder about what "the same source" means is worse than no guard, so
    this shells out and there is exactly one definition.
    """
    out = subprocess.run(
        "cat pinning.cu *.cuh $(ls *.h | grep -v '^qsb_carrier_sm89.h$') "
        "| LC_ALL=C sort | sha256sum",
        shell=True, cwd=HERE, capture_output=True, text=True, check=True)
    return out.stdout.split()[0]


def main() -> int:
    if not CARRIER.exists():
        print("test_carrier_freshness: no qsb_carrier_sm89.h -- the ranked path runs "
              "the plain compute_52 kernels. Nothing to check.")
        return 0

    head = CARRIER.read_text(errors="replace")[:2000]
    src = RECORDED_SRC.search(head)
    cubin = RECORDED_CUBIN.search(head)
    if not src or not cubin:
        print("test_carrier_freshness: FAIL -- qsb_carrier_sm89.h carries no source "
              "fingerprint. Regenerate it with build_carrier.sh; without the "
              "fingerprint this guard cannot run.", file=sys.stderr)
        return 1

    actual = source_fingerprint()
    if actual != src.group(1):
        print(f"test_carrier_freshness: FAIL -- STALE CARRIER.\n"
              f"  header was built from source {src.group(1)}\n"
              f"  this tree hashes to        {actual}\n"
              f"  The ranked path executes the embedded cubin ({cubin.group(1)[:16]}...), "
              f"not the PTX beside it, so this difference is INERT until the carrier is "
              f"rebuilt. The geometry stamp in QsbCarrier.h cannot see it, because a "
              f"behavioural edit does not move the table geometry.\n"
              f"  Fix: cd {HERE} && ./build_carrier.sh   (needs nvcc 12.8 + "
              f"OPENSSL_INC, same toolkit as the runner)", file=sys.stderr)
        return 1

    print(f"test_carrier_freshness: ok -- carrier matches this source ({actual[:16]}...), "
          f"cubin {cubin.group(1)[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
