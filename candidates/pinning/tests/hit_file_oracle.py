#!/usr/bin/env python3
"""Execute the hit publisher extracted from pinning.cu without a CUDA device."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def extract(source):
    start = source.index("#if QSB_HIT_FILE_PERSIST\n    ")
    end = source.index("#if QSB_REFILL_BEFORE_GATE\n    uint64_t last_completed_credit", start)
    close = source.index("#if QSB_HIT_FILE_PERSIST\n    if (gpu_hit_file", end)
    close_end = source.index("#endif", close) + len("#endif")
    return source[start:end], source[close:close_end]


PREFIX = r'''
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <cstdarg>
#include <cstdlib>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#define QSB_HOST_GATE 1
#define QSB_CPU_GRIND 0
#define QSB_ASICBOOST 1
#define QSB_AB_B0CONST 1
#define QSB_AB_K 4
#define QSB_COMPACT_READBACK 1
static int opens, closes, flushes, writes, mode;
static int found, rejected;
static uint32_t count;
static uint32_t data[64];
static FILE *tracked;
static std::string read_output() {
    FILE *f = fopen("results/pinning_hit_2.txt", "r");
    if (!f) return "";
    std::string result;
    char line[256];
    while (fgets(line, sizeof(line), f)) result += line;
    fclose(f);
    return result;
}
static FILE *open_file(const char *name, const char *flags) {
    ++opens;
    if (mode == 1) { errno = EACCES; return nullptr; }
    tracked = fopen(name, flags);
    if (tracked && mode == 2) setvbuf(tracked, nullptr, _IONBF, 0);
    return tracked;
}
static int close_file(FILE *f) {
    ++closes;
    int rc = fclose(f);
    tracked = nullptr;
    if (mode == 4) { errno = EIO; return EOF; }
    return rc;
}
static int flush_file(FILE *f) {
    ++flushes;
    return fflush(f);
}
static int write_file(FILE *f, const char *format, ...) {
    ++writes;
    va_list args;
    va_start(args, format);
    int rc = vfprintf(f, format, args);
    va_end(args);
    return rc;
}
static int qsb_gate_accept(void *, uint32_t, uint32_t lt, int ri,
                          int, int, int, int, int) {
    return rejected || lt == 100 ? -1 : ri ^ 1;
}
using cudaError_t = int;
static const int cudaSuccess = 0;
static int cudaEventSynchronize(int) { return mode == 6 && writes > 0 ? 1 : 0; }
static int cudaGetLastError() { return 0; }
static const char *cudaGetErrorString(int) { return "injected CUDA error"; }
struct Readback {
    uint32_t count() const { return ::count; }
    const uint32_t *indices() const { return data; }
};
#define fopen open_file
#define fclose close_file
#define fflush flush_file
#define fprintf write_file
static int exercise() {
    int gpu_index = 2, effective_total = 3;
    int pp = 0, gate_grp = 0, gate_ctx = 0, gate_order = 0, gate_nri = 0, gate_R = 0;
    int slot_busy[1] = {1}, slot_done[1] = {0};
    uint32_t slot_seq[1] = {7}, slot_lt[1] = {100};
    Readback slot_readback[1];
'''
BODY = r'''
    auto publish = [&]() {
#if QSB_REFILL_BEFORE_GATE
        return publish_hits(7, 100, count, data);
#else
        slot_busy[0] = 1;
        return drain_slot(0);
#endif
    };
    if (mode == 7) {
        count = 0;
        assert(publish() == 0 && found == 0 && opens == 0);
        return 0;
    }
    if (mode == 8) rejected = 1;
    count = 4;
    data[0] = 0;
    data[1] = 1;
    data[2] = (1u << 30) | 34;
    data[3] = (1u << 31) | 255;
    int rc = publish();
    if (rc) { assert(!found); return rc; }
    if (mode == 8) {
        assert(found == 0 && writes == 0 && flushes == 0);
    } else if (mode == 0 || mode == 5 || mode == 6 || mode == 9 || mode == 4) {
        const std::string expected = "existing\nsequence=7 locktime=356 recid=1\n"
            "sequence=10 locktime=612 recid=0\nsequence=16 locktime=16228 recid=1\n";
        assert(found == 1 && read_output() == expected);
        if (mode == 5) return 1;
        if (mode == 6) {
#if !QSB_REFILL_BEFORE_GATE
            return publish();
#else
            return 1;
#endif
        }
        if (mode == 9) {
            count = 65;
            for (auto &raw : data) raw = 1;
            assert(publish() == 0);
            assert(writes == 67);
        } else {
            assert(publish() == 0);
            assert(read_output() == expected + expected.substr(9));
        }
#if QSB_HIT_FILE_PERSIST
        assert(opens == 1 && closes == 0 && flushes == 2);
#else
        assert(opens == 2 && closes == 2 && flushes == 0);
#endif
    }
'''
SUFFIX = r'''
    return 0;
}
#undef fopen
#undef fclose
#undef fflush
#undef fprintf
int main(int argc, char **argv) {
    assert(argc == 2);
    mode = atoi(argv[1]);
    mkdir("results", 0755);
    if (mode == 2 || mode == 3) {
        assert(symlink("/dev/full", "results/pinning_hit_2.txt") == 0);
    } else if (mode != 7 && mode != 8 && mode != 1) {
        FILE *f = fopen("results/pinning_hit_2.txt", "w");
        assert(f);
        assert(fputs("existing\n", f) >= 0);
        assert(fclose(f) == 0);
    }
    int rc = exercise();
#if QSB_HIT_FILE_PERSIST
    if (mode >= 1 && mode <= 6) assert(rc == 1);
    else assert(rc == 0);
    assert(closes == (opens && mode != 1 ? 1 : 0));
    assert(!tracked);
#else
    assert(rc == (mode == 5 || mode == 6 ? 1 : 0));
#endif
    if (mode == 7) assert(access("results/pinning_hit_2.txt", F_OK) != 0);
    if (mode == 8) assert(read_output().empty());
    printf("PASS mode=%d opens=%d closes=%d writes=%d flushes=%d\n",
           mode, opens, closes, writes, flushes);
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "pinning.cu")
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args()
    source = args.source.read_text()
    publisher, close = extract(source)
    parent_parts = extract(args.parent.read_text()) if args.parent else None
    compiler = os.environ.get("CXX", "g++")
    checks = 0
    with tempfile.TemporaryDirectory(dir=os.environ.get("QSB_VERIFY_WORK")) as temp:
        root = Path(temp)
        for refill in (0, 1):
            for persist in (0, 1):
                cpp = root / f"publisher-{refill}-{persist}.cpp"
                cpp.write_text(PREFIX + publisher + BODY + close + SUFFIX)
                defines = [f"-DQSB_HIT_FILE_PERSIST={persist}", f"-DQSB_REFILL_BEFORE_GATE={refill}"]
                binary = cpp.with_suffix("")
                subprocess.run([compiler, "-std=c++14", "-O2", *defines, str(cpp), "-o", str(binary)], check=True)
                if not persist and parent_parts:
                    parent = root / "parent.cpp"
                    parent.write_text(PREFIX + parent_parts[0] + BODY + parent_parts[1] + SUFFIX)
                    command = [compiler, "-std=c++14", "-E", "-P", *defines]
                    actual = subprocess.check_output(command + [str(cpp)])
                    expected = subprocess.check_output(command + [str(parent)])
                    assert actual == expected, "OFF publisher differs from the immutable parent"
                    checks += 1
                modes = range(10) if persist else (0, 5, 6, 7, 8, 9)
                for mode in modes:
                    work = root / f"run-{refill}-{persist}-{mode}"
                    work.mkdir()
                    subprocess.run([str(binary), str(mode)], cwd=work, check=True, timeout=5)
                    checks += 1
    print(f"HIT_FILE_ORACLE PASS {checks} checks; output, visibility, append, errors, cleanup, OFF, both publisher paths")


if __name__ == "__main__":
    main()
