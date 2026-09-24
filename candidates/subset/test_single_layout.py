#!/usr/bin/env python3
"""CPU checks for the experimental layout; no CUDA execution or speed claim.

Compiles the actual source's scalar recoding and digit-staging loop, compares
with unbounded-integer signed recoding, then checks the candidate mapping and
the padded product-tree indices against independent modular inverses.
"""
from pathlib import Path
import ast
import random
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
SRC = (ROOT / 'tests/gpu_epochs/tree.cu').read_text()
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
P = 2**256 - 2**32 - 977
MASK = 2**64 - 1


def function(name):
    start = SRC.index(name + '(')
    start = SRC.rfind('\n', 0, start) + 1
    brace = SRC.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (SRC[end] == '{') - (SRC[end] == '}')
        end += 1
    return SRC[start:end]


def check_digits():
    # Include the literal new staging loop as well as the inherited recoder.
    start = SRC.index('for(int c=0;c<GT_CHUNKS;c++){', SRC.index('void qsb_filter_chain_trial'))
    end = SRC.index('\n    }', start) + len('\n    }')
    stage = SRC[start:end]
    shift = SRC[SRC.index('#define GT_CHUNKS 15'):SRC.index('/* n = secp256k1 group order')]
    shift = shift[:shift.rfind('#endif')]
    source = '''#include <cstdint>
#include <iostream>
#define __device__
#define __host__
#define __forceinline__ inline
const uint64_t GT_ORDER_N[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,
                            0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
'''+shift+'\n'+'\n'.join(function(n) for n in (
        'gt_recode_setup', 'gt_field_bits_v', 'gt_width', 'gt_direct_digit'))+'''
int main(){
    uint64_t k[4];
    while(std::cin >> std::hex >> k[0] >> k[1] >> k[2] >> k[3]){
        uint64_t M[4];int sign;gt_recode_setup(k,M,&sign);
        uint32_t codes[4096]={0},idx;uint64_t neg;unsigned tid=191;
'''+stage+'''
        for(int c=0;c<GT_CHUNKS;c++)std::cout<<std::hex<<codes[c*256+tid]<<' ';
        std::cout<<std::endl;
    }
}
'''
    rng = random.Random(20260924)
    cases = {0, 1, 2, N//2, N//2+1, N-2, N-1, N, N+1, 2**256-1}
    for bit in range(256):
        for d in (-1, 0, 1):
            cases.add(((1 << bit) + d) % 2**256)
    cases.update(rng.getrandbits(256) for _ in range(10000))
    cases = sorted(cases)
    inputs = ''.join(' '.join(f'{(k >> (64*j)) & MASK:x}' for j in range(4))+'\n' for k in cases)
    with tempfile.TemporaryDirectory(prefix='qsb-single-cpu-') as tmp:
        path = Path(tmp)
        (path/'digits.cpp').write_text(source)
        subprocess.run(['c++', '-std=c++11', '-O2', '-fsanitize=undefined',
                        str(path/'digits.cpp'), '-o', str(path/'digits')], check=True)
        result = subprocess.run([str(path/'digits')], input=inputs, text=True,
                                capture_output=True, check=True)
        assert not result.stderr, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == len(cases)
    for k, line in zip(cases, lines):
        # Reference uses the signed odd recurrence, not bit-field extraction.
        value = (2*(k % N)) % N
        sign = 1 if value & 1 else -1
        value = value if sign == 1 else N-value
        expected = []
        for c in range(14):
            width = 18 if c == 0 else 17
            digit = (value % (1 << (width+1))) - (1 << width)
            expected.append(sign*digit)
            value = (value-digit) >> width
        expected.append(sign*value)
        codes = [int(x, 16) for x in line.split()]
        got = [(2*(code & 0x3ffff)+1)*(-1 if code >> 18 else 1) for code in codes]
        assert got == expected, (k, got, expected)
        reconstructed = sum(d << (0 if c == 0 else 17*c+1) for c, d in enumerate(got))
        assert reconstructed % N == (2*k) % N
    return len(cases), len(cases)*15


def check_mapping():
    checks = 0
    for threads in (128, 192, 256):
        for epochs in range(1, 258):
            size = epochs*128
            padded = ((size+threads-1)//threads)*threads
            visits = bytearray(size)
            for index in range(padded):
                ep, lane = divmod(index, 128)
                active = index < padded and ep < epochs
                if not active:
                    continue
                assert not visits[index]
                visits[index] = 1
                for recid in (0, 1):
                    tag = index | recid << 30
                    assert divmod(tag & 0x3fffffff, 128) == (ep, lane)
                    assert tag >> 30 == recid
                checks += 1
            assert all(visits)
    for threads in (128, 192, 256):
        capacity = 262144*((threads+127)//128)
        size = capacity*128
        assert size < 1 << 30
        assert ((size+threads-1)//threads)*threads < 1 << 31
    return checks


def check_tree():
    # A model of the level-packed layout, with independent pow(v,-1,p) oracle.
    # This checks padding/index algebra, not CUDA barriers or PTX field math.
    rng = random.Random(960112)
    checks = 0
    for threads in (128, 192, 256):
        n = 256 if threads == 192 else threads
        for active in (0, 1, 31, 32, 33, threads//2, threads-1, threads):
            values = [rng.randrange(1, P) if j < active else 1 for j in range(n)]
            prod = [None]*512
            inv = [None]*256
            prod[:n] = values
            offset, count = 0, n
            while count > 2:
                half = count//2
                for tid in range(half):
                    prod[offset+count+tid] = prod[offset+tid]*prod[offset+half+tid] % P
                offset += count
                count //= 2
            scale = rng.randrange(1, P)
            root = pow(prod[offset]*prod[offset+1] % P, -1, P)*scale % P
            inv[offset-n] = root*prod[offset+1] % P
            inv[offset-n+1] = root*prod[offset] % P
            offset -= 4
            count = 4
            while count < n:
                half = count//2
                for tid in range(count):
                    inv[offset-n+tid] = inv[offset+count-n+(tid & (half-1))]*prod[offset+(tid ^ half)] % P
                offset -= 2*count
                count *= 2
            for tid in range(threads):
                got = inv[tid & (n//2-1)]*prod[tid ^ (n//2)] % P
                assert got == pow(values[tid], -1, P)*scale % P
                checks += 1
    return checks


def check_point_parking():
    def ptx(park):
        text = subprocess.check_output([
            'clang++', '-E', '-P', '-x', 'c++', '-D__CUDA_ARCH__=890',
            '-DQSB_SINGLE_EPOCH=1', '-DQSB_SINGLE_THREADS=192', f'-DQSB_SINGLE_POINT_PARK={park}',
            str(ROOT/'hit_filter_field_sc.cuh')], text=True, stderr=subprocess.DEVNULL)
        text = text[text.index('void qsb_filter_point_add('):]
        text = text[text.index('asm(')+4:text.index(': "+l"')]
        return ''.join(ast.literal_eval(m[0]) for m in re.finditer(r'"(?:[^"\\]|\\.)*"', text))
    baseline = ptx(0)
    operations = 0
    for mode in (1, 2):
        parked = ptx(mode)
        memory_ops = re.findall(r'(?:st|ld)\.shared\.u64[^;]+;', parked)
        rows = 4 if mode == 1 else 5
        assert len(memory_ops) == 2*rows, memory_ops
        for k in range(rows):
            value = f'ZZZ{k}' if k < 4 else 'ZZ0'
            addr = '%29' + (f'+{1536*k}' if k else '')
            store = f'st.shared.u64 [{addr}],{value};'
            load = f'ld.shared.u64 {value},[{addr}];'
            assert store in memory_ops and load in memory_ops
            between = parked[parked.index(store)+len(store):parked.index(load)]
            assert not re.search(r'\b'+value+r'\b', between), 'parked value used before reload'
        restored = re.sub(r'(?:st|ld)\.shared\.u64[^;]+;', '', parked)
        assert restored.split() == baseline.split(), 'arithmetic changed outside balanced parking'
        addresses = {k*192+tid for k in range(rows) for tid in range(192)}
        assert len(addresses) == rows*192 and max(addresses) < rows*192
        operations += len(memory_ops)
    return operations


if __name__ == '__main__':
    scalars, digits = check_digits()
    print(f'actual-source recoding: {scalars} scalars / {digits} packed digits')
    print(f'mapping and hit tags: {check_mapping()} active candidates')
    print(f'padded inverse-tree model: {check_tree()} lane inverses')
    print(f'actual point PTX: {check_point_parking()} balanced shared operations; all arithmetic identical')
    print('PASS (CPU checks only; no CUDA execution or throughput measurement)')
