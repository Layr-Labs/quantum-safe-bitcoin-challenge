#!/usr/bin/env python3
"""CPU differential audit of the exact production column divstep.

No CUDA or GPU required. Extracts the table and both production routines,
compiles the default-on shortcut against the unchanged four-coefficient
routine, and checks every low-six-bit input over delta=-32..32 plus random
full-width inputs. This proves the tested arithmetic, not GPU execution or
performance. Usage: python3 audit_divstep_noswap.py [--sanitize]
"""
import hashlib
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile

source = Path(__file__).with_name("zinv32.cuh").read_text()
start = source.index("#define ZI_BY_LUT_INIT")
end = source.index("/* ======================= 4-lane")
body = source[start:end]
test = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define ZI_CONST static const
#define ZI_DEV static inline
#define QSB_DIVSTEP_NOSWAP 1
PRODUCTION_BODY
static uint64_t rng = 0x61948b37123045abULL;
static uint32_t random32() {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)rng;
}
static uint64_t checked = 0;
static void check(int32_t delta, uint32_t f, uint32_t g) {
    int32_t a,b,c,d;
    const int32_t want = zi_divstep30_by(delta,f,g,&a,&b,&c,&d);
    for(uint32_t column=0;column<2;column++) {
        int32_t u,q;
        const int32_t got = zi_divstep30_column(delta,f,g,column,&u,&q);
        if(got!=want || u!=(column?b:a) || q!=(column?d:c)) {
            std::fprintf(stderr,"mismatch delta=%d f=%08x g=%08x column=%u\n",
                         delta,f,g,column);
            std::exit(1);
        }
        checked++;
    }
}
int main() {
    unsigned eligible=0;
    for(int dc=-6;dc<=6;dc++) for(unsigned r=0;r<64;r++) {
        const uint64_t formula = (6ULL<<32) | (1ULL<<24)
                               | (uint64_t)((64-r)&63)<<16 | 64;
        const unsigned z=dc<-5?6:(dc>0?0:1-dc);
        const bool fast=((64-r)&63)<(1u<<z);
        if(fast!=(ZI_BY_LUT[(dc+6)*64+r]==formula)) return 2;
        eligible+=fast;
    }
    for(int delta=-32;delta<=32;delta++)
        for(unsigned f=1;f<64;f+=2) for(unsigned g=0;g<64;g++)
            check(delta,f,g);
    for(int i=0;i<1000000;i++) {
        const int delta=(int)(random32()%65)-32;
        const uint32_t f=random32()|1u, g=random32();
        check(delta,f,g);
    }
    std::printf("832 table entries (%u eligible) and %llu column cases matched exactly\n",
                eligible,(unsigned long long)checked);
}
'''.replace("PRODUCTION_BODY", body)
flags = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Wno-unknown-pragmas"]
if "--sanitize" in sys.argv:
    flags += ["-fsanitize=undefined", "-fno-sanitize-recover=all"]
with tempfile.TemporaryDirectory(prefix="qsb-noswap-") as tmp:
    cpp = Path(tmp) / "audit.cpp"
    binary = Path(tmp) / "audit"
    cpp.write_text(test)
    subprocess.run(["g++", *flags, str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
print("zinv32.cuh sha256:", hashlib.sha256(source.encode()).hexdigest())

# Independent arbitrary-precision model: no CUDA limb arithmetic, lane shuffles,
# truncation, or exponent fallback. Compare complete inverses against Python's
# modular inverse and measure shortcut eligibility (an operation count, not speed).
table_text = body[:body.index("ZI_CONST uint64_t ZI_BY_LUT")]
table = [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]+)ULL", table_text)]
assert len(table) == 832
p = 2**256 - 2**32 - 977
inv64 = pow(64, -1, p)
rng = random.Random(61948)
groups = eligible = max_batches = 0
signed_byte = lambda x: (x & 127) - (x & 128)
for _ in range(2000):
    value = rng.randrange(1, p)
    f, g, r, s, delta = p, value, 0, 1, 1
    for batch in range(32):
        for _ in range(5):
            dc = max(-6, min(6, delta))
            ratio = (g * pow(f & 63, -1, 64)) & 63
            entry = table[(dc + 6) * 64 + ratio]
            a, b, c, d = [signed_byte(entry >> i) for i in (0, 8, 16, 24)]
            z = max(0, min(6, 1 - delta))
            fast = ((64 - ratio) & 63) < (1 << z)
            if fast:
                assert (a, b, c, d, entry >> 32) == (64, 0, (64 - ratio) & 63, 1, 6)
            groups += 1
            eligible += fast
            fn, gn = a * f + b * g, c * f + d * g
            assert fn % 64 == gn % 64 == 0
            f, g = fn // 64, gn // 64
            r, s = (a * r + b * s) * inv64 % p, (c * r + d * s) * inv64 % p
            flags = entry >> 32
            delta = (-delta if flags >> 31 else delta) + signed_byte(flags)
        if g == 0:
            break
    else:
        raise AssertionError("inverse model exceeded 32 batches")
    assert abs(f) == 1
    assert (r if f == 1 else -r) % p == pow(value, -1, p)
    max_batches = max(max_batches, batch + 1)
print(f"2000 modular inverses matched; {eligible}/{groups} groups eligible "
      f"({eligible/groups:.3%}); maximum {max_batches} batches")
