#!/usr/bin/env python3
"""Fail-closed instruction-level audit for the pinning field PTX helpers.

This utility extracts the *actual* inline PTX string literals from
_ModMultCore and _ModSqr in a supplied GPUMath.h, interprets only the PTX
instructions implemented below, and compares the extracted result with exact
Python bigint arithmetic over secp256k1's field.

It is intentionally source-bound and conservative:
* no CUDA compiler or runtime is used;
* unsupported opcodes, malformed source, uninitialised registers, and missing
  outputs are errors rather than silently falling back to a mathematical model;
* Python bigint arithmetic is used only as an independent comparison oracle;
  the returned candidate value always comes from the extracted instruction
  stream.

Examples:
  python3 pinning-field-ptx-audit.py --header GPUMath.h --self-test
  python3 pinning-field-ptx-audit.py --header GPUMath.h --self-test --expect-old
  python3 pinning-field-ptx-audit.py --header corrected/GPUMath.h \
      --self-test --expect-exact
  python3 pinning-field-ptx-audit.py --header GPUMath.h --op mult \
      --a p-65537 --b p-65537
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
MASK256 = (1 << 256) - 1
# secp256k1: p = 2^256 - 2^32 - 977
P = (1 << 256) - (1 << 32) - 977
KNOWN_A = P - 65537
KNOWN_CORRECT = 0x100020001
KNOWN_OLD = 0x1FC30


class AuditError(Exception):
    """Expected fail-closed audit error."""


class UnsupportedPTX(AuditError):
    """Raised when source uses semantics this emulator does not implement."""


@dataclass(frozen=True)
class PTXProgram:
    function: str
    text: str
    sha256: str
    statements: tuple[str, ...]
    declarations: dict[str, int]
    instruction_count: int
    predicate_declarations: frozenset[str] = frozenset()


@dataclass
class Machine:
    regs: dict[str, int]
    widths: dict[str, int]
    specials: dict[str, int]
    outputs: dict[str, int]
    initialized: set[str]
    preds: dict[str, bool] = field(default_factory=dict)
    cc: int = 0

    def _name(self, token: str) -> str:
        token = token.strip()
        if not token:
            raise AuditError("empty operand")
        return token

    def get(self, token: str) -> int:
        token = self._name(token)
        if token.startswith("%"):
            if token not in self.specials:
                raise AuditError(f"read of unavailable special operand {token}")
            return self.specials[token]
        if token in self.regs:
            if token not in self.initialized:
                raise AuditError(f"read of uninitialised register {token}")
            return self.regs[token]
        # A numeric immediate is the only non-register scalar accepted.
        if re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)", token):
            return int(token, 0)
        raise AuditError(f"read of uninitialised/unknown register {token}")

    def get_pred(self, token: str) -> bool:
        token = self._name(token)
        if token not in self.preds:
            raise AuditError(f"read of undeclared predicate {token}")
        return self.preds[token]

    def set_pred(self, token: str, value: bool) -> None:
        token = self._name(token)
        if token not in self.preds:
            raise AuditError(f"write to undeclared predicate {token}")
        self.preds[token] = bool(value)

    def set(self, token: str, value: int, width: int | None = None) -> None:
        token = self._name(token)
        if width is None:
            width = self.widths.get(token, 64)
        mask = MASK32 if width == 32 else MASK64 if width == 64 else None
        if mask is None:
            raise AuditError(f"unsupported destination width {width} for {token}")
        value &= mask
        if token.startswith("%"):
            self.outputs[token] = value
        elif token in self.regs:
            self.regs[token] = value
            self.initialized.add(token)
        else:
            raise AuditError(f"write to undeclared register {token}")


def split_operands(text: str) -> list[str]:
    """Split comma-separated PTX operands while preserving brace pairs."""

    out: list[str] = []
    start = 0
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                raise AuditError("unbalanced PTX braces")
        elif ch == "," and depth == 0:
            item = text[start:i].strip()
            if not item:
                raise AuditError("empty PTX operand")
            out.append(item)
            start = i + 1
    if depth != 0:
        raise AuditError("unbalanced PTX braces")
    item = text[start:].strip()
    if item:
        out.append(item)
    return out


def pair_names(token: str) -> tuple[str, str]:
    token = token.strip()
    if len(token) < 2 or token[0] != "{" or token[-1] != "}":
        raise AuditError(f"expected PTX register pair, got {token!r}")
    parts = split_operands(token[1:-1])
    if len(parts) != 2:
        raise AuditError(f"expected two registers in pair, got {token!r}")
    return parts[0].strip(), parts[1].strip()


def function_body_end(source: str, after_signature: int, function: str) -> int:
    """Return the closing brace of one function, ignoring C string braces."""

    opening = source.find("{", after_signature)
    if opening < 0:
        raise AuditError(f"function body for {function} was not found")
    depth = 0
    in_string = False
    in_char = False
    escaped = False
    line_comment = False
    block_comment = False
    i = opening
    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if line_comment:
            if ch == "\n":
                line_comment = False
        elif block_comment:
            if ch == "*" and nxt == "/":
                block_comment = False
                i += 1
        elif in_string or in_char:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif (in_string and ch == '"') or (in_char and ch == "'"):
                in_string = False if in_string else in_string
                in_char = False if in_char else in_char
        elif ch == "/" and nxt == "/":
            line_comment = True
            i += 1
        elif ch == "/" and nxt == "*":
            block_comment = True
            i += 1
        elif ch == '"':
            in_string = True
        elif ch == "'":
            in_char = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise AuditError(f"unterminated function body for {function}")


def extract_ptx(source: str, function: str) -> str:
    """Extract and decode string literals passed to the named inline asm."""

    # Function names are unique in the supplied header.  Keep the extraction
    # source-bound: do not substitute the host transcription or a hand model.
    match = re.search(r"\b(?:void|static\s+inline\s+void|__device__[^\n]*\bvoid)\s+"
                      + re.escape(function) + r"\s*\(", source)
    if not match:
        raise AuditError(f"function {function} was not found")
    body_end = function_body_end(source, match.end(), function)
    asm_start = source.find("asm(", match.end(), body_end)
    if asm_start < 0:
        raise AuditError(f"inline asm for {function} was not found in its body")

    # Current and corrected headers put the output-constraint colon on a
    # source line outside the C string literals.  Stop there, before parsing
    # constraints or host code as PTX.
    tail = source[asm_start:body_end]
    end_match = re.search(r"\n[ \t]*:\s*\"", tail)
    if not end_match:
        raise AuditError(f"could not locate asm constraints for {function}")
    body = tail[len("asm("):end_match.start()]
    literals = re.findall(r'"(?:\\.|[^"\\])*"', body, flags=re.DOTALL)
    if not literals:
        raise AuditError(f"inline asm for {function} has no string literals")
    try:
        decoded = [ast.literal_eval(literal) for literal in literals]
    except (SyntaxError, ValueError) as exc:
        raise AuditError(f"invalid C string literal in {function}: {exc}") from exc
    if any(not isinstance(value, str) for value in decoded):
        raise AuditError(f"non-string asm literal in {function}")
    ptx = "".join(decoded)
    if not ptx.strip():
        raise AuditError(f"empty PTX for {function}")
    return ptx


def split_statements(ptx: str) -> tuple[str, ...]:
    """Split the extracted PTX at semicolons, retaining no implicit ops."""

    statements: list[str] = []
    for raw in ptx.split(";"):
        # PTX comments are not expected in the current source, but removing a
        # line comment here is safe and avoids treating a comment as an opcode.
        raw = re.sub(r"//[^\n]*", "", raw)
        statement = raw.strip()
        # The asm block's outer braces share the first/last semicolon
        # fragments with a declaration/instruction.  Strip only those outer
        # braces; a trailing operand pair such as ``{z0,z1}`` is data and must
        # remain intact.
        if statement.startswith("{"):
            statement = statement[1:].lstrip()
        statement = re.sub(r"\n[ \t]*}\s*$", "", statement)
        statement = statement.strip()
        if statement:
            statements.append(statement)
    return tuple(statements)


def parse_program(function: str, ptx: str) -> PTXProgram:
    statements = split_statements(ptx)
    declarations: dict[str, int] = {}
    predicate_declarations: set[str] = set()
    executable: list[str] = []
    for statement in statements:
        if statement in {"{", "}"}:
            continue
        if statement.startswith(".reg"):
            match = re.fullmatch(r"\.reg\s+\.u(32|64)\s+(.+)", statement)
            if match:
                width = int(match.group(1))
                for name in match.group(2).split(","):
                    name = name.strip()
                    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                        raise AuditError(f"invalid declared register {name!r}")
                    if name in declarations and declarations[name] != width:
                        raise AuditError(f"register {name} declared at two widths")
                    declarations[name] = width
                continue
            pred_match = re.fullmatch(r"\.reg\s+\.pred\s+(.+)", statement)
            if not pred_match:
                raise UnsupportedPTX(f"unsupported register declaration: {statement}")
            for name in pred_match.group(1).split(","):
                name = name.strip()
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                    raise AuditError(f"invalid declared predicate {name!r}")
                if name in predicate_declarations:
                    raise AuditError(f"predicate {name} declared twice")
                predicate_declarations.add(name)
            continue
        executable.append(statement)
    # The parser below is deliberately closed.  Any unrecognised executable
    # statement will fail during interpretation rather than being ignored.
    if not executable:
        raise AuditError(f"no executable PTX statements for {function}")
    return PTXProgram(function, ptx, hashlib.sha256(ptx.encode()).hexdigest(),
                      tuple(executable), declarations, len(executable),
                      frozenset(predicate_declarations))


def require_operands(mnemonic: str, operands: list[str], count: int) -> None:
    if len(operands) != count:
        raise AuditError(f"{mnemonic} expects {count} operands, got {len(operands)}")


def execute(program: PTXProgram, a: int, b: int | None) -> int:
    """Execute one extracted PTX stream and return its 256-bit output."""

    if not (0 <= a <= MASK256):
        raise AuditError("a must be in [0, 2^256)")
    if b is None:
        b = a
    if not (0 <= b <= MASK256):
        raise AuditError("b must be in [0, 2^256)")
    aw = [(a >> (64 * i)) & MASK64 for i in range(4)]
    bw = [(b >> (64 * i)) & MASK64 for i in range(4)]
    specials = {f"%{4 + i}": value for i, value in enumerate(aw)}
    specials.update({f"%{8 + i}": value for i, value in enumerate(bw)})
    machine = Machine(dict.fromkeys(program.declarations, 0), dict(program.declarations),
                      specials, {}, set(),
                      {name: False for name in program.predicate_declarations})

    def read(token: str) -> int:
        return machine.get(token)

    def write(token: str, value: int, width: int) -> None:
        machine.set(token, value, width)

    for statement in program.statements:
        predicate_match = re.match(
            r"^@(!?)([A-Za-z_][A-Za-z0-9_]*)\s+(.+)$", statement
        )
        if predicate_match:
            negate = bool(predicate_match.group(1))
            predicate = predicate_match.group(2)
            statement = predicate_match.group(3).strip()
            predicate_value = machine.get_pred(predicate)
            if predicate_value == negate:
                continue
        match = re.match(r"^([A-Za-z][A-Za-z0-9_.]*)\s*(.*)$", statement)
        if not match:
            raise UnsupportedPTX(f"unrecognised PTX statement: {statement!r}")
        mnemonic, rest = match.group(1), match.group(2).strip()
        operands = split_operands(rest)
        parts = mnemonic.split(".")

        if mnemonic == "setp.ne.u32":
            require_operands(mnemonic, operands, 3)
            left = read(operands[1]) & MASK32
            right = read(operands[2]) & MASK32
            machine.set_pred(operands[0], left != right)
            continue

        if mnemonic == "mov.b64":
            require_operands(mnemonic, operands, 2)
            dst, src = operands
            dst_is_pair = dst.strip().startswith("{")
            src_is_pair = src.strip().startswith("{")
            if dst_is_pair and src_is_pair:
                raise UnsupportedPTX("mov.b64 pair-to-pair is unsupported")
            if dst_is_pair:
                lo, hi = pair_names(dst)
                value = read(src)
                write(lo, value, 32)
                write(hi, value >> 32, 32)
            elif src_is_pair:
                lo, hi = pair_names(src)
                value = read(lo) | (read(hi) << 32)
                write(dst, value, 64)
            else:
                write(dst, read(src), 64)
            continue

        if mnemonic in {"mov.u32", "mov.u64"}:
            require_operands(mnemonic, operands, 2)
            width = int(parts[-1][1:])
            write(operands[0], read(operands[1]), width)
            continue

        if mnemonic == "cvt.u64.u32":
            require_operands(mnemonic, operands, 2)
            write(operands[0], read(operands[1]) & MASK32, 64)
            continue

        if mnemonic == "mul.wide.u32":
            require_operands(mnemonic, operands, 3)
            write(operands[0], (read(operands[1]) & MASK32) * (read(operands[2]) & MASK32), 64)
            continue

        if mnemonic in {"mul.lo.u32", "mul.hi.u32"}:
            require_operands(mnemonic, operands, 3)
            product = (read(operands[1]) & MASK32) * (read(operands[2]) & MASK32)
            value = product if mnemonic == "mul.lo.u32" else product >> 32
            write(operands[0], value, 32)
            continue

        if mnemonic == "mad.lo.u32":
            require_operands(mnemonic, operands, 4)
            value = ((read(operands[1]) & MASK32) * (read(operands[2]) & MASK32)
                     + (read(operands[3]) & MASK32)) & MASK32
            write(operands[0], value, 32)
            continue

        if mnemonic == "shf.l.wrap.b32":
            require_operands(mnemonic, operands, 4)
            shift = read(operands[3]) & 31
            hi = read(operands[1]) & MASK32
            lo = read(operands[2]) & MASK32
            value = lo if shift == 0 else ((lo << shift) | (hi >> (32 - shift))) & MASK32
            write(operands[0], value, 32)
            continue

        if (len(parts) >= 2 and parts[0] in {"add", "addc"}
                and parts[-1] in {"u32", "u64"}
                and all(part in {"add", "addc", "cc", "u32", "u64"} for part in parts)):
            require_operands(mnemonic, operands, 3)
            width = int(parts[-1][1:])
            mask = MASK32 if width == 32 else MASK64
            total = (read(operands[1]) & mask) + (read(operands[2]) & mask)
            if parts[0] == "addc":
                total += machine.cc
            write(operands[0], total, width)
            if "cc" in parts:
                machine.cc = (total >> width) & 1
            continue

        raise UnsupportedPTX(f"unsupported PTX opcode: {mnemonic}")

    words: list[int] = []
    for i in range(4):
        token = f"%{i}"
        if token not in machine.outputs:
            raise AuditError(f"PTX stream did not produce {token}")
        words.append(machine.outputs[token] & MASK64)
    return sum(word << (64 * i) for i, word in enumerate(words)) & MASK256


def parse_number(text: str) -> int:
    value = text.strip().lower().replace("_", "")
    aliases = {
        "p": P,
        "p-1": P - 1,
        "p-2": P - 2,
        "p-65536": P - 65536,
        "p-65537": P - 65537,
        "p+1": P + 1,
    }
    if value in aliases:
        return aliases[value]
    match = re.fullmatch(r"p([+-])(\d+)", value)
    if match:
        offset = int(match.group(2), 10)
        return P + offset if match.group(1) == "+" else P - offset
    try:
        return int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer or p-relative value: {text}") from exc


def exact(a: int, b: int) -> int:
    return (a * b) % P


def suite_vectors() -> list[tuple[int, int, str]]:
    # The review's exact seed is retained for reproducibility.  The boundary
    # matrix is deliberately explicit and canonical; no host replacement is
    # used to generate the candidate result.
    vectors = [
        (0, 0, "zero"),
        (1, 1, "one"),
        (1, P - 1, "one_times_p_minus_1"),
        (P - 1, 1, "p_minus_1_times_one"),
        (P - 1, P - 1, "p_minus_1_squared"),
        (KNOWN_A, KNOWN_A, "known_p_minus_65537"),
    ]
    rng = random.Random(37207)
    for i in range(100):
        vectors.append((rng.randrange(P), rng.randrange(P), f"random_{i:03d}"))
    return vectors


def check_one(program: PTXProgram, a: int, b: int) -> dict:
    observed = execute(program, a, b)
    wanted = exact(a, b)
    return {
        "a": hex(a),
        "b": hex(b),
        "observed": hex(observed),
        "expected": hex(wanted),
        "match": observed == wanted,
        # Congruence is reported separately so a caller can distinguish a
        # dropped-carry error from a merely noncanonical [0,2^256) residue.
        "congruent": (observed - wanted) % P == 0,
    }


def check_op(program: PTXProgram, op: str, a: int, b: int) -> dict:
    # _ModSqr has only one source operand.  Supplying b is harmless for the
    # generic executor, but square's source must explicitly be a*a.
    left, right = (a, b) if op == "mult" else (a, a)
    result = check_one(program, left, right)
    result["operation"] = op
    return result


def audit_programs(header: Path) -> dict[str, PTXProgram]:
    try:
        source = header.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuditError(f"cannot read header {header}: {exc}") from exc
    programs: dict[str, PTXProgram] = {}
    for function in ("_ModMultCore", "_ModSqr"):
        ptx = extract_ptx(source, function)
        programs[function] = parse_program(function, ptx)
    return programs


def base_report(header: Path, programs: dict[str, PTXProgram]) -> dict:
    return {
        "header": str(header),
        "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
        "ptx": {
            function: {
                "sha256": program.sha256,
                "bytes": len(program.text.encode()),
                "instructions": program.instruction_count,
            }
            for function, program in programs.items()
        },
        "mode": "source-bound-PTX-instruction-emulation",
        "cuda_runtime_validated": False,
        "compiler_validated": False,
        "fail_closed": True,
    }


def self_test(programs: dict[str, PTXProgram]) -> tuple[dict, bool]:
    vectors = suite_vectors()
    by_op: dict[str, dict] = {}
    all_mismatches: list[dict] = []
    for op, function in (("mult", "_ModMultCore"), ("sqr", "_ModSqr")):
        program = programs[function]
        checks = [check_op(program, op, a, b) for a, b, _ in vectors]
        for check, (_, _, label) in zip(checks, vectors):
            check["label"] = label
            if not check["match"]:
                all_mismatches.append(check)
        known = next(check for check in checks if check["label"] == "known_p_minus_65537")
        by_op[op] = {
            "checks": len(checks),
            "matches": sum(1 for check in checks if check["match"]),
            "mismatches": [check for check in checks if not check["match"]],
            "known_vector": known,
        }
    known_observed = {op: by_op[op]["known_vector"]["observed"] for op in by_op}
    known_exact = all(by_op[op]["known_vector"]["match"] for op in by_op)
    known_old = all(by_op[op]["known_vector"]["observed"] == hex(KNOWN_OLD) for op in by_op)
    result = {
        "suite_seed": 37207,
        "vectors_per_operation": len(vectors),
        "operations": by_op,
        "all_mismatches": all_mismatches,
        "classification": (
            "exact-suite-pass" if not all_mismatches else
            "known-old-defect-reproduced" if known_old else
            "mismatch-unclassified"
        ),
        "known_vector_expected": hex(KNOWN_CORRECT),
        "known_vector_old_observed": hex(KNOWN_OLD),
        "known_vector_observed": known_observed,
    }
    # A corrected header is accepted only when every bounded vector matches.
    # An old header is deliberately a failing exact audit, while --expect-old
    # below lets a consumer prove that the old failure is reproduced.
    return result, not all_mismatches


def direct_check(programs: dict[str, PTXProgram], op: str, a: int, b: int) -> tuple[dict, bool]:
    selected = [("mult", "_ModMultCore")] if op == "mult" else \
               [("sqr", "_ModSqr")] if op == "sqr" else \
               [("mult", "_ModMultCore"), ("sqr", "_ModSqr")]
    checks = [check_op(programs[function], label, a, b) for label, function in selected]
    return {"checks": checks}, all(check["match"] for check in checks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--header", required=True, type=Path, help="exact GPUMath.h to extract")
    parser.add_argument("--self-test", action="store_true", help="run the bounded exact suite")
    parser.add_argument("--expect-old", action="store_true", help="success means the known old defect is reproduced")
    parser.add_argument("--expect-exact", action="store_true", help="success means the bounded suite is exact")
    parser.add_argument("--op", choices=("mult", "sqr", "both"), default="both")
    parser.add_argument("--a", type=parse_number, default=KNOWN_A)
    parser.add_argument("--b", type=parse_number, default=None)
    parser.add_argument("--expected", type=parse_number, default=None,
                        help="optional explicit expected bigint for direct mode")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expect_old and args.expect_exact:
        parser.error("--expect-old and --expect-exact are mutually exclusive")
    if args.b is None:
        args.b = args.a
    if not (0 <= args.a <= MASK256 and 0 <= args.b <= MASK256):
        parser.error("--a/--b must be in [0, 2^256)")

    try:
        programs = audit_programs(args.header)
        report = base_report(args.header, programs)
        if args.self_test:
            suite, exact_pass = self_test(programs)
            report["self_test"] = suite
            report["status"] = suite["classification"]
            if args.expect_old:
                ok = suite["classification"] == "known-old-defect-reproduced"
            elif args.expect_exact:
                ok = exact_pass
            else:
                # Default is an exact correctness audit, so the old header
                # fails with status 1 and exposes its mismatches in JSON.
                ok = exact_pass
        else:
            direct, exact_pass = direct_check(programs, args.op, args.a, args.b)
            report["direct"] = direct
            report["status"] = "exact-match" if exact_pass else "exact-mismatch"
            if args.expected is not None:
                for check in direct["checks"]:
                    check["explicit_expected"] = hex(args.expected)
                    check["explicit_match"] = check["observed"] == hex(args.expected)
                ok = all(check["observed"] == hex(args.expected) for check in direct["checks"])
            else:
                ok = exact_pass
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if ok else 1
    except AuditError as exc:
        # Keep unsupported semantics visible to callers and never substitute a
        # host or idealized implementation after an extraction/emulation gap.
        failure = {
            "status": "fail-closed",
            "header": str(args.header),
            "error": str(exc),
            "mode": "source-bound-PTX-instruction-emulation",
            "cuda_runtime_validated": False,
            "compiler_validated": False,
            "fail_closed": True,
        }
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
