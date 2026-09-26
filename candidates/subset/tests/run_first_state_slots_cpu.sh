#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
set -euo pipefail

test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
toolchain_prefix=${QSB_TOOLCHAIN_PREFIX:-}
cxx=${CXX:-c++}
out_dir=${QSB_CPU_OUT_DIR:-/tmp/qsb-first-state-slots-cpu}
mkdir -p "$out_dir"

cxxflags=(-std=c++17 -O2 -pthread)
ldflags=()
if [[ -n "$toolchain_prefix" ]]; then
    cxxflags+=(-I"$toolchain_prefix/include")
    ldflags+=(-L"$toolchain_prefix/lib" -Wl,-rpath,"$toolchain_prefix/lib")
fi
if [[ ${QSB_CPU_SANITIZE:-0} == 1 ]]; then
    cxxflags+=(-fsanitize=undefined -fno-omit-frame-pointer)
    ldflags+=(-fsanitize=undefined)
fi

for windows in 128 256; do
    exe="$out_dir/test_first_state_slots_cpu_$windows"
    "$cxx" "${cxxflags[@]}" -DQSB_SE_WINDOWS="$windows" \
        "${ldflags[@]}" \
        -o "$exe" "$test_dir/test_first_state_slots_cpu.cpp" -lcrypto
    "$exe"
done
