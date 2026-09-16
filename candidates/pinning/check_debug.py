#!/usr/bin/env python3
"""Compare the pinning CUDA debug pipeline with the public CPU verifier.

The checker runs the candidate's one-point ``debug`` mode for deterministic
inputs, then derives the same externally meaningful values through
``harness/problem.py`` and ``harness/crypto.py``.  Projective coordinates are
required as part of the debug schema, but are only shape-checked: their scale
is an implementation detail.  The final affine points, compressed keys, and
hashes are compared exactly.

Use a temporary ``--problem-dir``/``--work-dir`` when running this command.
The candidate's working directory is where any host-side output would land;
the checker itself only writes per-sample debug logs there.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "harness"
sys.path.insert(0, str(HARNESS))

import crypto as C  # noqa: E402  (the harness is intentionally self-contained)
import problem as PB  # noqa: E402


UINT32_MAX = (1 << 32) - 1
SAMPLE_SEED = 0x515342

DBG_LINE = re.compile(r"^DBG: (?P<name>[A-Za-z0-9_]+) = (?P<value>\S+)\s*$")
HEX_RE = re.compile(r"^[0-9a-fA-F]+$")

# These values are printed by kernel_debug_pin_one_point in pinning.cu.  The
# projective fields are included so a stale or partial debug binary fails
# loudly, even though their values cannot be compared to affine CPU points
# without assuming the candidate's projective representation.
REQUIRED_FIELDS = (
    "seq",
    "lt",
    "suffix_len",
    "total_preimage_len",
    "patched_suffix",
    "midstate",
    "first_sha256",
    "sighash_z",
    "z_scalar",
    "neg_r_inv",
    "u1",
    "u1G_x_affine",
    "u1G_y_affine",
    "u2R_x",
    "u2R_y",
    "q1_proj_x",
    "q1_proj_y",
    "q1_proj_z",
    "q2_proj_x",
    "q2_proj_y",
    "q2_proj_z",
    "q1z_mul_q2z",
    "q1z_q2z_inv",
    "inv1_q1z",
    "inv2_q2z",
    "Q1_aff_x",
    "Q1_aff_y",
    "Q2_aff_x",
    "Q2_aff_y",
    "recid0_pubkey",
    "recid0_sha_pk",
    "recid1_pubkey",
    "recid1_sha_pk",
)

HEX64_FIELDS = {
    "midstate",
    "first_sha256",
    "sighash_z",
    "z_scalar",
    "neg_r_inv",
    "u1",
    "u1G_x_affine",
    "u1G_y_affine",
    "u2R_x",
    "u2R_y",
    "q1_proj_x",
    "q1_proj_y",
    "q1_proj_z",
    "q2_proj_x",
    "q2_proj_y",
    "q2_proj_z",
    "q1z_mul_q2z",
    "q1z_q2z_inv",
    "inv1_q1z",
    "inv2_q2z",
    "Q1_aff_x",
    "Q1_aff_y",
    "Q2_aff_x",
    "Q2_aff_y",
}

PUBKEY_FIELDS = {"recid0_pubkey", "recid1_pubkey"}
HASH_FIELDS = {"recid0_sha_pk", "recid1_sha_pk"}


class CheckError(RuntimeError):
    """A user-actionable checker failure."""


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _hex256(value: int) -> str:
    return f"{value & ((1 << 256) - 1):064x}"


def _point_fields(prefix: str, point: tuple[int, int]) -> dict[str, str]:
    return {
        f"{prefix}_x": f"{point[0]:064x}",
        f"{prefix}_y": f"{point[1]:064x}",
    }


def load_pinning_problem(problem_dir: Path) -> dict:
    """Load the authoritative JSON through harness/problem.py."""
    problem_dir = problem_dir.resolve()
    json_path = problem_dir / "pinning.json"
    bin_path = problem_dir / "pinning.bin"
    if not json_path.is_file():
        raise CheckError(f"missing problem JSON: {json_path}")
    if not bin_path.is_file():
        raise CheckError(f"missing problem binary: {bin_path}")

    previous = os.environ.get("QSB_PROBLEM_DIR")
    os.environ["QSB_PROBLEM_DIR"] = str(problem_dir)
    try:
        problem = PB.load_problem("pinning")
    except Exception as exc:  # surface malformed input as a checker failure
        raise CheckError(f"could not load {json_path}: {exc}") from exc
    finally:
        if previous is None:
            os.environ.pop("QSB_PROBLEM_DIR", None)
        else:
            os.environ["QSB_PROBLEM_DIR"] = previous

    if problem.get("bench") != "pinning":
        raise CheckError(f"problem JSON declares bench={problem.get('bench')!r}")
    return problem


def sample_inputs(count: int) -> list[tuple[int, int]]:
    """Return fixed boundary cases followed by ``count`` seeded random pairs."""
    if count < 0:
        raise CheckError("--samples must be >= 0")

    # Include both uint32 packing extremes and the safe range used by the seed
    # kernel.  The debug entry point accepts all uint32 values, which makes the
    # byte-order/masking check useful independently of the grind loop.
    boundary = [
        (0, 0),
        (0, UINT32_MAX),
        (UINT32_MAX, 0),
        (UINT32_MAX, UINT32_MAX),
        (0x80000000, 500_000_000),
        (0x80000000, 1_744_600_000),
        (UINT32_MAX, 500_000_000),
        (UINT32_MAX, 1_744_600_000),
    ]
    seen = set(boundary)
    result = list(boundary)
    rng = random.Random(SAMPLE_SEED)
    while count:
        pair = (rng.getrandbits(32), rng.getrandbits(32))
        if pair not in seen:
            result.append(pair)
            seen.add(pair)
            count -= 1
    return result


def expected_fields(problem: dict, sequence: int, locktime: int) -> dict[str, str]:
    """Derive expected debug fields via the public CPU harness."""
    # pin_preimage() performs the same uint32 masking and little-endian patch
    # that the verifier uses, while keeping this checker off the CUDA path.
    preimage = PB.pin_preimage(problem, sequence, locktime)
    suffix = bytearray(bytes.fromhex(problem["suffix"]))
    suffix[problem["seq_offset"] : problem["seq_offset"] + 4] = (
        (sequence & UINT32_MAX).to_bytes(4, "little")
    )
    suffix[problem["lt_offset"] : problem["lt_offset"] + 4] = (
        (locktime & UINT32_MAX).to_bytes(4, "little")
    )

    first_sha256 = C.sha256(preimage)
    sighash = C.sha256d(preimage)
    z = int.from_bytes(sighash, "big")
    r = int(problem["r"], 16)
    s = int(problem["s"], 16)
    neg_r_inv = int(problem["neg_r_inv"], 16)
    u1 = (neg_r_inv * z) % C.N
    u1g = C.point_mul(u1, C.G)
    if u1g is None:
        raise CheckError(f"CPU derivation produced point-at-infinity for seq={sequence}")

    recovered = {}
    for recid in (0, 1):
        point = C.ecdsa_recover(r, s, z, recid)
        if point is None:
            raise CheckError(f"CPU recovery failed for recid={recid}")
        recovered[recid] = point

    midstate = C.sha256_midstate(bytes.fromhex(problem["pin_prefix"]))
    result = {
        "seq": f"{sequence & UINT32_MAX:08x}",
        "lt": str(locktime & UINT32_MAX),
        "suffix_len": str(len(suffix)),
        "total_preimage_len": str(problem["total_preimage_len"]),
        "patched_suffix": bytes(suffix).hex(),
        "midstate": "".join(f"{word:08x}" for word in midstate),
        "first_sha256": first_sha256.hex(),
        "sighash_z": sighash.hex(),
        "z_scalar": sighash.hex(),
        "neg_r_inv": _hex256(neg_r_inv),
        "u1": _hex256(u1),
        "u2R_x": _hex256(int(problem["u2r_x"], 16)),
        "u2R_y": _hex256(int(problem["u2r_y"], 16)),
    }
    result.update(_point_fields("u1G", u1g))
    result["u1G_x_affine"] = result.pop("u1G_x")
    result["u1G_y_affine"] = result.pop("u1G_y")

    for recid, point in recovered.items():
        prefix = "Q1_aff" if recid == 0 else "Q2_aff"
        result.update(_point_fields(prefix, point))
        pubkey = C.compress_pubkey(point)
        result[f"recid{recid}_pubkey"] = pubkey.hex()
        result[f"recid{recid}_sha_pk"] = C.sha256(pubkey).hex()
    return result


def parse_debug(output: str) -> dict[str, str]:
    """Parse DBG records and reject a partial or malformed debug stream."""
    fields: dict[str, str] = {}
    malformed = []
    ended = False
    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "DBG: END":
            ended = True
            continue
        if not stripped.startswith("DBG: "):
            continue
        match = DBG_LINE.match(stripped)
        if match is None:
            malformed.append(stripped)
            continue
        name, value = match.group("name"), match.group("value")
        if name in fields:
            raise CheckError(f"duplicate DBG field {name}")
        fields[name] = value

    if malformed:
        raise CheckError(f"malformed DBG line: {malformed[0]}")
    if not ended:
        raise CheckError("missing DBG: END (debug output is incomplete)")
    missing = [name for name in REQUIRED_FIELDS if name not in fields]
    if missing:
        raise CheckError(f"missing DBG field(s): {', '.join(missing)}")
    return fields


def validate_shapes(fields: dict[str, str], expected: dict[str, str]) -> None:
    """Check fixed-width output before comparing values."""
    for name in HEX64_FIELDS:
        value = fields[name]
        if len(value) != 64 or HEX_RE.fullmatch(value) is None:
            raise CheckError(f"DBG field {name} is not 32-byte hex: {value!r}")
    for name in PUBKEY_FIELDS:
        value = fields[name]
        if len(value) != 66 or HEX_RE.fullmatch(value) is None:
            raise CheckError(f"DBG field {name} is not 33-byte hex: {value!r}")
    for name in HASH_FIELDS:
        value = fields[name]
        if len(value) != 64 or HEX_RE.fullmatch(value) is None:
            raise CheckError(f"DBG field {name} is not 32-byte hex: {value!r}")

    for name in ("seq", "lt", "suffix_len", "total_preimage_len"):
        if not fields[name].isdigit():
            if name == "seq":
                # seq is printed with %08x, unlike the decimal fields.
                if len(fields[name]) != 8 or HEX_RE.fullmatch(fields[name]) is None:
                    raise CheckError(f"DBG field {name} is not an integer: {fields[name]!r}")
            else:
                raise CheckError(f"DBG field {name} is not decimal: {fields[name]!r}")

    suffix = fields["patched_suffix"]
    if len(suffix) != len(expected["patched_suffix"]) or HEX_RE.fullmatch(suffix) is None:
        raise CheckError("DBG field patched_suffix has the wrong hex length or characters")


def compare_fields(fields: dict[str, str], expected: dict[str, str]) -> list[str]:
    """Return exact mismatches for all CPU-derived externally meaningful fields."""
    mismatches = []
    for name, want in expected.items():
        got = fields.get(name)
        if got is None:
            mismatches.append(f"{name}: missing")
        elif got.lower() != want.lower():
            mismatches.append(f"{name}: got {got}, expected {want}")
    return mismatches


def run_one(
    binary: Path,
    problem_bin: Path,
    sequence: int,
    locktime: int,
    work_dir: Path,
    timeout: float,
    index: int,
) -> str:
    """Run one explicit binary debug invocation and save its raw output."""
    argv = [
        str(binary),
        str(problem_bin),
        "0",
        "1",
        "0",
        "single_hash",
        "debug",
        f"0x{sequence & UINT32_MAX:08x}",
        str(locktime & UINT32_MAX),
    ]
    try:
        completed = subprocess.run(
            argv,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = _as_text(completed.stdout) + _as_text(completed.stderr)
    except subprocess.TimeoutExpired as exc:
        output = _as_text(exc.stdout) + _as_text(exc.stderr)
        log_path = work_dir / f"check-debug-{index:02d}-{sequence & UINT32_MAX:08x}-{locktime & UINT32_MAX}.log"
        log_path.write_text(output)
        raise CheckError(f"CUDA/debug timeout after {timeout:g}s; raw output: {log_path}") from exc
    except OSError as exc:
        raise CheckError(f"could not execute {binary}: {exc}") from exc

    log_path = work_dir / f"check-debug-{index:02d}-{sequence & UINT32_MAX:08x}-{locktime & UINT32_MAX}.log"
    log_path.write_text(output)
    if completed.returncode != 0:
        raise CheckError(
            f"CUDA/debug failure for seq=0x{sequence & UINT32_MAX:08x} "
            f"lt={locktime & UINT32_MAX}: exit {completed.returncode}; raw output: {log_path}"
        )

    lowered = output.lower()
    error_markers = (
        "cuda error",
        "kernel_debug_pin_one_point failed",
        "no cuda-capable device",
        "invalid device",
        "launch failed",
    )
    marker = next((item for item in error_markers if item in lowered), None)
    if marker is not None:
        raise CheckError(f"CUDA/debug failure ({marker}); raw output: {log_path}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare pinning CUDA debug fields with the independent CPU derivation."
    )
    parser.add_argument("--binary", required=True, type=Path, help="explicit executable pinning binary")
    parser.add_argument(
        "--problem-dir",
        required=True,
        type=Path,
        help="directory containing pinning.json and pinning.bin",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="number of deterministic random pairs, in addition to fixed boundary cases (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="per-sample process timeout in seconds (default: 900)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="candidate cwd and raw-log directory (default: --problem-dir)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples < 0:
        print("check_debug: FAIL: --samples must be >= 0", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("check_debug: FAIL: --timeout must be > 0", file=sys.stderr)
        return 2

    try:
        binary = args.binary.resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise CheckError(f"--binary is not an executable file: {binary}")
        problem_dir = args.problem_dir.resolve()
        problem = load_pinning_problem(problem_dir)
        work_dir = (args.work_dir or problem_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        if not work_dir.is_dir():
            raise CheckError(f"--work-dir is not a directory: {work_dir}")

        samples = sample_inputs(args.samples)
        for index, (sequence, locktime) in enumerate(samples):
            expected = expected_fields(problem, sequence, locktime)
            output = run_one(
                binary,
                problem_dir / "pinning.bin",
                sequence,
                locktime,
                work_dir,
                args.timeout,
                index,
            )
            fields = parse_debug(output)
            validate_shapes(fields, expected)
            mismatches = compare_fields(fields, expected)
            if mismatches:
                log_path = work_dir / f"check-debug-{index:02d}-{sequence & UINT32_MAX:08x}-{locktime & UINT32_MAX}.log"
                raise CheckError(
                    f"sample {index} mismatch for seq=0x{sequence & UINT32_MAX:08x} "
                    f"lt={locktime & UINT32_MAX}; raw output: {log_path}; "
                    + "; ".join(mismatches)
                )
            print(
                f"check_debug: sample {index + 1}/{len(samples)} pass "
                f"seq=0x{sequence & UINT32_MAX:08x} lt={locktime & UINT32_MAX}",
                flush=True,
            )
    except CheckError as exc:
        print(f"check_debug: FAIL: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"check_debug: FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"check_debug: PASS ({len(samples)} samples); logs in {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
