#!/usr/bin/env python3
"""Portable host oracle for the corrected zinv32 skip-zero candidate.

The generated C++ executor includes the actual selected-track header and runs
the four-lane cooperative inverse with a host rendezvous model for
__shfl_sync(0xF, value, src).  Python supplies exact secp256k1 modular-inverse
expectations, deterministic tree-shaped product roots, sign-pattern tracing,
and a negative-control build with the old final-high-limb row bug restored.
"""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path


P = (1 << 256) - (1 << 32) - 977
M288 = 1 << 288


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "benchmark.json").is_file() and (path / "candidates/subset/tests/gpu_epochs/zinv32.cuh").is_file():
            return path
    raise RuntimeError("could not locate repository root")


ROOT = find_repo_root(Path(__file__).resolve())
HEADER = ROOT / "candidates/subset/tests/gpu_epochs/zinv32.cuh"


CPP = r'''
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#ifndef QSB_ZINV32_INCLUDE
#define QSB_ZINV32_INCLUDE "candidates/subset/tests/gpu_epochs/zinv32.cuh"
#endif

void zi_trace_neg(int lane, uint32_t neg);
#include QSB_ZINV32_INCLUDE

namespace {

constexpr int kLanes = 4;
thread_local int tls_lane = -1;

struct Rendezvous {
    std::mutex mu;
    std::condition_variable cv;
    uint64_t generation = 0;
    int arrived = 0;
    int departed = 0;
    bool failed = false;
    std::string failure;
    std::array<uint32_t, kLanes> values{};
    std::array<int, kLanes> srcs{};
    std::array<uint32_t, kLanes> returns{};

    void reset() {
        std::lock_guard<std::mutex> lk(mu);
        generation = 0;
        arrived = 0;
        departed = 0;
        failed = false;
        failure.clear();
        values.fill(0);
        srcs.fill(0);
        returns.fill(0);
    }

    void fail_locked(const std::string &msg) {
        if (!failed) {
            failed = true;
            failure = msg;
        }
        cv.notify_all();
    }
};

struct TraceState {
    std::mutex mu;
    std::condition_variable cv;
    uint64_t generation = 0;
    int arrived = 0;
    int departed = 0;
    uint64_t total = 0;
    uint64_t allzero = 0;
    uint64_t allone = 0;
    uint64_t mixed05 = 0;
    uint64_t mixed0a = 0;
    std::array<uint32_t, kLanes> values{};

    void reset() {
        std::lock_guard<std::mutex> lk(mu);
        generation = 0;
        arrived = 0;
        departed = 0;
        total = allzero = allone = mixed05 = mixed0a = 0;
        values.fill(0);
    }
};

Rendezvous g_shuffle;
TraceState g_trace;

unsigned hex_nibble(char c) {
    if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return static_cast<unsigned>(c - 'A' + 10);
    throw std::runtime_error("bad hex digit");
}

std::string strip_hex(std::string s) {
    if (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s.erase(0, 2);
    return s.empty() ? std::string("0") : s;
}

template <size_t N>
std::array<uint32_t, N> parse_hex32(const std::string &hex) {
    const std::string s = strip_hex(hex);
    if (s.size() > N * 8) throw std::runtime_error("hex input too wide");
    std::array<uint32_t, N> out{};
    for (size_t rev = 0; rev < s.size(); ++rev) {
        const unsigned nib = hex_nibble(s[s.size() - 1 - rev]);
        out[rev / 8] |= static_cast<uint32_t>(nib) << ((rev % 8) * 4);
    }
    return out;
}

std::array<uint64_t, 5> parse_hex256(const std::string &hex) {
    auto w = parse_hex32<8>(hex);
    std::array<uint64_t, 5> out{};
    for (int i = 0; i < 4; ++i) {
        out[i] = static_cast<uint64_t>(w[2 * i]) |
                 (static_cast<uint64_t>(w[2 * i + 1]) << 32);
    }
    return out;
}

std::string hex64(const std::array<uint64_t, 5> &x) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::nouppercase;
    for (int i = 3; i >= 0; --i) oss << std::setw(16) << x[i];
    return oss.str();
}

template <size_t N>
std::string hex32(const std::array<uint32_t, N> &x) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::nouppercase;
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) oss << std::setw(8) << x[i];
    return oss.str();
}

std::array<uint64_t, 5> run_inverse(const std::array<uint64_t, 5> &input, uint64_t *shuffles) {
    g_shuffle.reset();
    std::array<std::array<uint64_t, 5>, kLanes> lane_r{};
    for (int lane = 0; lane < kLanes; ++lane) lane_r[lane] = input;
    std::array<std::exception_ptr, kLanes> errors{};
    std::array<std::thread, kLanes> threads;
    for (int lane = 0; lane < kLanes; ++lane) {
        threads[lane] = std::thread([lane, &lane_r, &errors]() {
            tls_lane = lane;
            try {
                zi_inverse_quad(lane_r[lane].data(), lane);
            } catch (...) {
                errors[lane] = std::current_exception();
                std::lock_guard<std::mutex> lk(g_shuffle.mu);
                g_shuffle.fail_locked("lane exception");
            }
            tls_lane = -1;
        });
    }
    for (auto &t : threads) t.join();
    for (const auto &e : errors) if (e) std::rethrow_exception(e);
    {
        std::lock_guard<std::mutex> lk(g_shuffle.mu);
        if (g_shuffle.failed) throw std::runtime_error(g_shuffle.failure);
        if (shuffles) *shuffles = g_shuffle.generation;
    }
    for (int lane = 1; lane < kLanes; ++lane) {
        if (lane_r[lane] != lane_r[0]) throw std::runtime_error("lane inverse mismatch");
    }
    if (lane_r[0][4] != 0) throw std::runtime_error("nonzero fifth inverse limb");
    return lane_r[0];
}

}  // namespace

uint32_t zi_x(uint32_t v, int src) {
    if (tls_lane < 0 || tls_lane >= kLanes) throw std::runtime_error("bad tls lane");
    if (src < 0 || src >= kLanes) throw std::runtime_error("bad source lane");
    std::unique_lock<std::mutex> lk(g_shuffle.mu);
    if (g_shuffle.failed) throw std::runtime_error(g_shuffle.failure);
    const uint64_t gen = g_shuffle.generation;
    g_shuffle.values[tls_lane] = v;
    g_shuffle.srcs[tls_lane] = src;
    if (++g_shuffle.arrived == kLanes) {
        for (int lane = 0; lane < kLanes; ++lane) {
            g_shuffle.returns[lane] = g_shuffle.values[g_shuffle.srcs[lane]];
        }
        g_shuffle.cv.notify_all();
    } else if (!g_shuffle.cv.wait_for(lk, std::chrono::seconds(10), [&] {
                   return g_shuffle.failed || (g_shuffle.generation == gen && g_shuffle.arrived == kLanes);
               })) {
        g_shuffle.fail_locked("shuffle rendezvous timeout");
        throw std::runtime_error(g_shuffle.failure);
    }
    const uint32_t out = g_shuffle.returns[tls_lane];
    if (++g_shuffle.departed == kLanes) {
        g_shuffle.arrived = 0;
        g_shuffle.departed = 0;
        ++g_shuffle.generation;
        g_shuffle.cv.notify_all();
    } else if (!g_shuffle.cv.wait_for(lk, std::chrono::seconds(10), [&] {
                   return g_shuffle.failed || g_shuffle.generation != gen;
               })) {
        g_shuffle.fail_locked("shuffle release timeout");
        throw std::runtime_error(g_shuffle.failure);
    }
    return out;
}

void zi_trace_neg(int lane, uint32_t neg) {
    if (tls_lane != lane) throw std::runtime_error("trace lane mismatch");
    std::unique_lock<std::mutex> lk(g_trace.mu);
    const uint64_t gen = g_trace.generation;
    g_trace.values[lane] = neg ? 1u : 0u;
    if (++g_trace.arrived == kLanes) {
        uint32_t pattern = 0;
        for (int i = 0; i < kLanes; ++i) pattern |= (g_trace.values[i] & 1u) << i;
        ++g_trace.total;
        if (pattern == 0x0) ++g_trace.allzero;
        else if (pattern == 0xf) ++g_trace.allone;
        else if (pattern == 0x5) ++g_trace.mixed05;
        else if (pattern == 0xa) ++g_trace.mixed0a;
        else throw std::runtime_error("unexpected non-pair sign pattern");
        g_trace.cv.notify_all();
    } else if (!g_trace.cv.wait_for(lk, std::chrono::seconds(10), [&] {
                   return g_trace.generation == gen && g_trace.arrived == kLanes;
               })) {
        throw std::runtime_error("trace rendezvous timeout");
    }
    if (++g_trace.departed == kLanes) {
        g_trace.arrived = 0;
        g_trace.departed = 0;
        ++g_trace.generation;
        g_trace.cv.notify_all();
    } else if (!g_trace.cv.wait_for(lk, std::chrono::seconds(10), [&] {
                   return g_trace.generation != gen;
               })) {
        throw std::runtime_error("trace release timeout");
    }
}

int main() {
    try {
        std::string op;
        while (std::cin >> op) {
            if (op == "inv") {
                std::string label, hex;
                std::cin >> label >> hex;
                uint64_t shuffles = 0;
                auto got = run_inverse(parse_hex256(hex), &shuffles);
                std::cout << "inv " << label << " " << hex64(got) << " " << shuffles << "\n";
            } else if (op == "cond") {
                unsigned neg = 0;
                std::string label, hex;
                std::cin >> neg >> label >> hex;
                auto x = parse_hex32<9>(hex);
                zi_condneg(x.data(), neg ? 1u : 0u);
                std::cout << "cond " << neg << " " << label << " " << hex32(x) << "\n";
            } else {
                throw std::runtime_error("unknown command");
            }
        }
        std::cout << "trace " << g_trace.total << " " << g_trace.allzero << " "
                  << g_trace.allone << " " << g_trace.mixed05 << " "
                  << g_trace.mixed0a << "\n";
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
'''


def hx(x: int) -> str:
    return format(x, "x")


def add_case(cases: list[tuple[str, int]], seen: set[int], label: str, x: int) -> None:
    x %= P
    if x not in seen:
        seen.add(x)
        cases.append((label, x))


def tree_root(block: int) -> int:
    level = []
    for lane in range(256):
        digest = hashlib.sha256(f"selected:block:{block}:leaf:{lane}".encode()).digest()
        leaf = int.from_bytes(digest, "big") % P
        level.append(leaf or 1)
    while len(level) > 1:
        level = [(level[i] * level[i + 1]) % P for i in range(0, len(level), 2)]
    return level[0]


def inverse_cases() -> list[tuple[str, int]]:
    cases: list[tuple[str, int]] = []
    seen: set[int] = set()
    for label, x in (
        ("zero", 0),
        ("one", 1),
        ("two", 2),
        ("p-1", P - 1),
        ("p-2", P - 2),
        ("p-half", P >> 1),
        ("rowbug-p-2^27-1", P - (1 << 27) - 1),
        ("rowbug-p-2^51-1", P - (1 << 51) - 1),
        ("2^255", 1 << 255),
        ("2^256-1", (1 << 256) - 1),
    ):
        add_case(cases, seen, label, x)
    for bit in range(0, 256, 5):
        add_case(cases, seen, f"pow2_{bit}", 1 << bit)
        add_case(cases, seen, f"p_minus_pow2_{bit}", P - (1 << bit))
        add_case(cases, seen, f"p_minus_pow2m1_{bit}", P - (1 << bit) - 1)
    rng = random.Random(0x23121219)
    for i in range(160):
        add_case(cases, seen, f"random_{i}", rng.getrandbits(256))
    for block in range(128):
        add_case(cases, seen, f"tree_root_{block}", tree_root(block))
    return cases


def cond_cases() -> list[tuple[str, int]]:
    vals = [0, 1, 2, (1 << 32) - 1, 1 << 32, P, P - 1, 1 << 255, 1 << 287, M288 - 1]
    rng = random.Random(0x231C0A0D)
    vals.extend(rng.getrandbits(288) for _ in range(80))
    out: list[tuple[str, int]] = []
    seen: set[int] = set()
    for i, x in enumerate(vals):
        x %= M288
        if x not in seen:
            seen.add(x)
            out.append((f"c{i}", x))
    return out


def compile_exec(cxx: str, src: Path, out: Path, header: Path, skip: int) -> None:
    cmd = [
        cxx,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-pedantic",
        "-pthread",
        f'-DQSB_ZINV32_INCLUDE="{header}"',
        f"-DQSB_ZI_SKIP_ZERO_NEGATION={skip}",
        str(src),
        "-o",
        str(out),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_exec(binary: Path, commands: list[str]) -> list[str]:
    proc = subprocess.run(
        [str(binary)],
        cwd=ROOT,
        input="\n".join(commands) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return [line for line in proc.stdout.splitlines() if line]


def parse_outputs(lines: list[str]) -> tuple[dict[tuple[str, str], tuple[str, ...]], tuple[int, int, int, int, int]]:
    values: dict[tuple[str, str], tuple[str, ...]] = {}
    trace = (0, 0, 0, 0, 0)
    for line in lines:
        parts = line.split()
        if parts[0] == "inv":
            values[("inv", parts[1])] = (parts[2], parts[3])
        elif parts[0] == "cond":
            values[(f"cond{parts[1]}", parts[2])] = (parts[3],)
        elif parts[0] == "trace":
            trace = tuple(int(x) for x in parts[1:6])  # type: ignore[assignment]
        else:
            raise AssertionError(f"unexpected output: {line}")
    return values, trace


def make_instrumented_header(dst: Path) -> None:
    text = HEADER.read_text()
    needle = "        neg=zi_x(neg,lane&1);\n"
    if needle not in text:
        raise RuntimeError("could not find sign broadcast for instrumentation")
    dst.write_text(text.replace(needle, needle + "        zi_trace_neg(lane,neg);\n", 1))


def make_old_row_header(dst: Path) -> None:
    text = HEADER.read_text()
    fixed = "    X[8]=(uint32_t)(acc>>ZI_B);\n"
    old = "    X[8]=(uint32_t)((int32_t)X[8]>>ZI_B);\n"
    if fixed not in text:
        raise RuntimeError("corrected row assignment not found")
    dst.write_text(text.replace(fixed, old, 1))


def check_exact_inverse(values: dict[tuple[str, str], tuple[str, ...]], cases: list[tuple[str, int]]) -> None:
    failures = []
    for label, x in cases:
        got = int(values[("inv", label)][0], 16)
        want = 0 if x == 0 else pow(x, -1, P)
        if got != want:
            failures.append((label, x, got, want))
    if failures:
        label, x, got, want = failures[0]
        raise AssertionError(f"inverse mismatch {label}: x={x:x} got={got:x} want={want:x}")


def check_condneg(values: dict[tuple[str, str], tuple[str, ...]], cases: list[tuple[str, int]]) -> None:
    for label, x in cases:
        got0 = int(values[("cond0", label)][0], 16)
        got1 = int(values[("cond1", label)][0], 16)
        if got0 != x % M288:
            raise AssertionError(f"condneg(0) mismatch for {label}")
        if got1 != (-x) % M288:
            raise AssertionError(f"condneg(1) mismatch for {label}")


def main() -> int:
    cxx = os.environ.get("CXX", "c++")
    header_sha = hashlib.sha256(HEADER.read_bytes()).hexdigest()
    inv = inverse_cases()
    cond = cond_cases()
    commands = [f"inv {label} {hx(x)}" for label, x in inv]
    for label, x in cond:
        commands.append(f"cond 0 {label} {hx(x)}")
        commands.append(f"cond 1 {label} {hx(x)}")

    with tempfile.TemporaryDirectory(prefix="qsb-zinv32-selected-", dir=ROOT / ".swarm") as td:
        tmp = Path(td)
        src = tmp / "probe.cpp"
        src.write_text(CPP)
        off_bin = tmp / "probe_off"
        on_bin = tmp / "probe_on"
        compile_exec(cxx, src, off_bin, HEADER, 0)
        compile_exec(cxx, src, on_bin, HEADER, 1)
        off_values, _ = parse_outputs(run_exec(off_bin, commands))
        on_values, _ = parse_outputs(run_exec(on_bin, commands))
        if off_values != on_values:
            raise AssertionError("QSB_ZI_SKIP_ZERO_NEGATION=1 changed zinv32 observable output")
        check_exact_inverse(off_values, inv)
        check_condneg(off_values, cond)

        old_header = tmp / "zinv32_old_row.cuh"
        make_old_row_header(old_header)
        old_bin = tmp / "probe_old_row"
        compile_exec(cxx, src, old_bin, old_header, 0)
        bug_cases = [
            ("rowbug-p-2^27-1", P - (1 << 27) - 1),
            ("rowbug-p-2^51-1", P - (1 << 51) - 1),
        ]
        old_values, _ = parse_outputs(run_exec(old_bin, [f"inv {label} {hx(x)}" for label, x in bug_cases]))
        old_mismatches = 0
        for label, x in bug_cases:
            if int(old_values[("inv", label)][0], 16) != pow(x, -1, P):
                old_mismatches += 1
        if old_mismatches == 0:
            raise AssertionError("old-row negative control unexpectedly matched exact inverse")

        inst_header = tmp / "zinv32_instrumented.cuh"
        make_instrumented_header(inst_header)
        trace_bin = tmp / "probe_trace"
        compile_exec(cxx, src, trace_bin, inst_header, 1)
        trace_values, trace = parse_outputs(run_exec(trace_bin, [f"inv {label} {hx(x)}" for label, x in inv]))
        check_exact_inverse(trace_values, inv)

    total, allzero, allone, mixed05, mixed0a = trace
    mixed = mixed05 + mixed0a
    print(f"header_sha256 {header_sha}")
    print(f"equivalence inverse_cases={len(inv)} condneg_cases={len(cond)} off_on_match=1")
    print(f"exact_mod_inverse cases={len(inv)} pass=1")
    print(f"condneg_exact288 cases={len(cond)} flags=0,1 pass=1")
    print(f"old_row_negative_control cases=2 mismatches={old_mismatches} pass=1")
    print(
        "sign_patterns "
        f"total={total} allzero={allzero} allone={allone} "
        f"mixed_pair02={mixed05} mixed_pair13={mixed0a} mixed={mixed}"
    )
    if total:
        print(
            "sign_pattern_rates "
            f"allzero={allzero / total:.6f} allone={allone / total:.6f} "
            f"mixed={mixed / total:.6f}"
        )
    print("host corrected skip-zero oracle: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
