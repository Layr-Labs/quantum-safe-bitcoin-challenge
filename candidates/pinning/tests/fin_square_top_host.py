#!/usr/bin/env python3
"""Generate/run a C++ oracle from the actual before/after square PTX strings."""
import argparse
import ast
import pathlib
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def asm_tokens(path, marker):
    source = path.read_text().split(marker, 1)[1].split('asm(', 1)[1]
    source = source.split(': "=l"', 1)[0]
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|[A-Za-z_]\w*', source)
    return ''.join(ast.literal_eval(t) if t.startswith('"') else '@' + t + '@'
                   for t in tokens)


base = asm_tokens(ROOT / 'GPUMath.h', 'void _ModSqr(')
# The base has a #else arm before the operand list; select its first asm only.
base = base.split('@else@', 1)[0]
new = asm_tokens(ROOT / 'FinSquareTop.cuh', 'void qsb_fin_square_top(')
old_top = ('addc.cc.u32 x14, x14, y14;\n'
           'addc.u32 x15, x15, @QZ@;\n'
           'shf.l.wrap.b32 x15, x14, x15, 1;')
new_top = 'addc.u32 x14, x14, y14;\nshr.u32 x15, x14, 31;'
assert base.count(old_top) == 1
assert base.replace(old_top, new_top) == new, 'PTX changed beyond the top-carry rewrite'


def cpp_function(name, ptx):
    ptx = ptx.split('@QSB_SQR_FOLD_LOW@', 1)[0]
    ptx = ptx.replace('@QZ_DECL@', '').replace('@QZ@', '0')
    for i in range(4):
        ptx = ptx.replace('%' + str(i + 4), f'a[{i}]')
    # Braces only delimit PTX scopes or mov.b64 limb pairs in this fragment.
    ptx = ptx.replace('{', '').replace('}', '')
    lines = [f'Wide {name}(const Input &a) {{', 'bool cc = false; U128 tmp;']
    for inst in ptx.split(';'):
        inst = inst.strip()
        if not inst:
            continue
        if inst.startswith('.reg '):
            _, kind, names = inst.split(None, 2)
            lines.append(f'uint{kind[2:]}_t {names};')
            continue
        op, operands = inst.split(None, 1)
        v = [x.strip() for x in operands.split(',')]
        if op == 'mov.b64' and len(v) == 3:
            # Unpacked destinations are 32-bit; packed destinations are 64-bit.
            # Use the source PTX braces to distinguish the two forms.
            unpack = re.search(r'mov\.b64\s*\{\s*' + re.escape(v[0]) + r'\s*,',
                               base if name == 'old_square' else new)
            if unpack:
                lines += [f'{v[0]} = uint32_t({v[2]});', f'{v[1]} = {v[2]} >> 32;']
            else:
                lines.append(f'{v[0]} = uint64_t({v[1]}) | (uint64_t({v[2]}) << 32);')
        elif op.startswith('mov.'):
            lines.append(f'{v[0]} = {v[1]};')
        elif op == 'mul.wide.u32':
            lines.append(f'{v[0]} = uint64_t({v[1]}) * {v[2]};')
        elif op.startswith(('add.', 'addc.')):
            bits = 64 if op.endswith('u64') else 32
            carry = ' + cc' if op.startswith('addc.') else ''
            lines += [f'tmp = U128({v[1]}) + {v[2]}{carry};',
                      f'{v[0]} = uint{bits}_t(tmp);']
            if '.cc.' in op:
                lines.append(f'cc = tmp >> {bits};')
        elif op == 'shf.l.wrap.b32':
            assert v[3] == '1'
            if v[0] == 'x15':
                lines.append('if (x15 != 0) std::abort();')
            lines.append(f'{v[0]} = ({v[2]} << 1) | ({v[1]} >> 31);')
        elif op == 'shl.b32':
            lines.append(f'{v[0]} = {v[1]} << {v[2]};')
        elif op == 'shr.u32':
            lines.append(f'{v[0]} = {v[1]} >> {v[2]};')
        else:
            raise ValueError(inst)
    lines += ['return {d0,d1,d2,d3,d4,d5,d6,d7};', '}']
    return '\n'.join(lines)


harness = r'''
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
using U128 = __uint128_t;
using Input = std::array<uint64_t, 4>;
using Wide = std::array<uint64_t, 8>;
@FUNCTIONS@
Wide oracle(const Input &a) {
    Wide out{};
    for (unsigned i = 0; i < 4; ++i) {
        U128 carry = 0;
        for (unsigned j = 0; j < 4; ++j) {
            U128 v = U128(a[i]) * a[j] + out[i+j] + carry;
            out[i+j] = uint64_t(v);
            carry = v >> 64;
        }
        out[i+4] = uint64_t(carry);
    }
    return out;
}
void check(const Input &a) {
    if (old_square(a) != new_square(a) || new_square(a) != oracle(a)) {
        std::fprintf(stderr, "square mismatch: %016llx %016llx %016llx %016llx\n",
                     (unsigned long long)a[0], (unsigned long long)a[1],
                     (unsigned long long)a[2], (unsigned long long)a[3]);
        std::abort();
    }
}
int main() {
    uint64_t count = 0;
    const uint32_t limbs[] = {0,1,2,0x7fffffff,0x80000000,0xfffffffe,0xffffffff};
    // Every 8-limb combination of low/high boundary values, plus mixed patterns.
    for (unsigned mask = 0; mask < 6561; ++mask) {
        unsigned m = mask;
        Input a{};
        for (unsigned i = 0; i < 8; ++i, m /= 3)
            a[i/2] |= uint64_t(m % 3 == 0 ? 0 : m % 3 == 1 ? 1 : 0xffffffff) << ((i%2)*32);
        check(a); ++count;
    }
    for (uint32_t x : limbs) for (uint32_t y : limbs) {
        Input a{};
        for (unsigned i = 0; i < 4; ++i) a[i] = uint64_t(x) | (uint64_t(y) << 32);
        check(a); ++count;
    }
    for (unsigned i = 0; i < 256; ++i) {
        Input a{}; a[i/64] = uint64_t(1) << (i%64); check(a); ++count;
        for (auto &v : a) v = ~v;
        check(a); ++count;
    }
    std::mt19937_64 rng(0x53573035544f50ULL);
    for (unsigned i = 0; i < 1000000; ++i) {
        Input a{rng(),rng(),rng(),rng()}; check(a); ++count;
    }
    std::printf("PASS: %llu inputs, original PTX == trimmed PTX == independent 4x64 square; reduction text unchanged\n",
                (unsigned long long)count);
}
'''
source = harness.replace('@FUNCTIONS@', cpp_function('old_square', base) + '\n' +
                         cpp_function('new_square', new))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--emit-cpp', type=pathlib.Path, help='retain the generated C++ oracle')
args = parser.parse_args()
if args.emit_cpp:
    args.emit_cpp.write_text(source)
with tempfile.TemporaryDirectory(prefix='qsb-sw05-host-') as temp:
    cpp = pathlib.Path(temp) / 'fin_square_top_host.cpp'
    cpp.write_text(source)
    binary = pathlib.Path(temp) / 'check'
    subprocess.run(['clang++', '-std=c++17', '-O2', '-fsanitize=undefined',
                    str(cpp), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
