#!/usr/bin/env python3
"""Build a host-emulated binary from a subset candidate tree (CPU verification only).

usage: mkemu.py <candidates/subset dir> <out dir> <zeros N> [launch_blocks]

Source transformations (applied to a copy; the candidate is not modified):
  * NAME<<<A,B>>>(ARGS);  ->  QSB_EMU_LAUNCH(A, B, NAME(ARGS));
  * the inline-asm body of tree.cu's field multiply (qsb_field_mul_raw, or
    qsb_field_mul in older trees) -> exact C product (a valid residue < 2^256)
  * GPUMath.h -> generated host port: its PTX carry macros, _CTZ and __clzll are
    replaced by faithful C equivalents; every arithmetic routine is unchanged
    (routines with an #ifdef __CUDA_ARCH__ host path use that path)
  * GPUHash.h's ASSEMBLY_SIGMA rotations -> the equivalent C macros
  * optional: launch-size override (smaller launches for speed)
Everything else is compiled verbatim with g++ against emu/include/cuda_runtime.h.
"""
import re, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src_dir, out_dir, zeros = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
launch_blocks = int(sys.argv[4]) if len(sys.argv) > 4 else None
if out_dir.exists():
    shutil.rmtree(out_dir)
shutil.copytree(src_dir, out_dir)

def transform_launches(text):
    out, pos, n = [], 0, 0
    pat = re.compile(r'(\b[A-Za-z_]\w*(?:<[^<>;()]*>)?)\s*<<<(.*?)>>>\s*\(', re.S)
    while True:
        m = pat.search(text, pos)
        if not m:
            out.append(text[pos:]); break
        out.append(text[pos:m.start()])
        name, dims = m.group(1), m.group(2)
        # find the matching close paren of the argument list
        i, depth = m.end(), 1
        while depth:
            c = text[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            i += 1
        args = text[m.end():i - 1]
        # split dims at the top-level comma
        depth, cut = 0, None
        for j, c in enumerate(dims):
            if c in '([':
                depth += 1
            elif c in ')]':
                depth -= 1
            elif c == ',' and depth == 0:
                cut = j; break
        assert cut is not None, dims
        a, b = dims[:cut].strip(), dims[cut + 1:].strip()
        out.append(f'QSB_EMU_LAUNCH(({a}), ({b}), {name}({args}))')
        pos = i
        n += 1
    return ''.join(out), n

HOST_MACROS = r'''// ---- emulation: faithful C port of the PTX inline-asm carry macros ----
typedef unsigned __int128 fu128; typedef __int128 fs128;
static thread_local uint64_t g_cf;
#define UADDO(c, a, b) { uint64_t _a=(a), _b=(b); fu128 _t=(fu128)_a+_b; (c)=(uint64_t)_t; g_cf=(uint64_t)(_t>>64); }
#define UADDC(c, a, b) { uint64_t _a=(a), _b=(b); fu128 _t=(fu128)_a+_b+g_cf; (c)=(uint64_t)_t; g_cf=(uint64_t)(_t>>64); }
#define UADD(c, a, b)  { uint64_t _a=(a), _b=(b); (c)=_a+_b+g_cf; }
#define UADDO1(c, a) UADDO(c, c, a)
#define UADDC1(c, a) UADDC(c, c, a)
#define UADD1(c, a)  UADD(c, c, a)
#define USUBO(c, a, b) { uint64_t _a=(a), _b=(b); (c)=_a-_b; g_cf=(_a<_b); }
#define USUBC(c, a, b) { uint64_t _a=(a), _b=(b); fs128 _t=(fs128)_a-(fs128)_b-(fs128)g_cf; (c)=(uint64_t)_t; g_cf=(_t<0); }
#define USUB(c, a, b)  { uint64_t _a=(a), _b=(b); (c)=_a-_b-g_cf; }
#define USUBO1(c, a) USUBO(c, c, a)
#define USUBC1(c, a) USUBC(c, c, a)
#define USUB1(c, a)  USUB(c, c, a)
#define UMULLO(lo,a, b) { uint64_t _a=(a), _b=(b); (lo)=_a*_b; }
#define UMULHI(hi,a, b) { uint64_t _a=(a), _b=(b); (hi)=(uint64_t)(((fu128)_a*_b)>>64); }
#define MADDO(r,a,b,c) { uint64_t _h=(uint64_t)(((fu128)(uint64_t)(a)*(uint64_t)(b))>>64); uint64_t _c=(c); fu128 _t=(fu128)_h+_c; (r)=(uint64_t)_t; g_cf=(uint64_t)(_t>>64); }
#define MADDC(r,a,b,c) { uint64_t _h=(uint64_t)(((fu128)(uint64_t)(a)*(uint64_t)(b))>>64); uint64_t _c=(c); fu128 _t=(fu128)_h+_c+g_cf; (r)=(uint64_t)_t; g_cf=(uint64_t)(_t>>64); }
#define MADD(r,a,b,c)  { uint64_t _h=(uint64_t)(((fu128)(uint64_t)(a)*(uint64_t)(b))>>64); uint64_t _c=(c); (r)=_h+_c+g_cf; }
#define MADDS(r,a,b,c) { int64_t _h=(int64_t)(((fs128)(int64_t)(a)*(fs128)(int64_t)(b))>>64); uint64_t _c=(c); (r)=(uint64_t)_h+_c+g_cf; }
'''

def host_port(text):
    lines = text.split('\n')
    start = next(i for i, l in enumerate(lines) if l.startswith('#define UADDO(c, a, b) asm volatile'))
    end = next(i for i, l in enumerate(lines) if l.startswith('#define MADDS(r,a,b,c) asm volatile'))
    block = lines[start:end + 1]
    assert all(l.startswith('#define ') or not l.strip() for l in block), 'unexpected macro block'
    out = '\n'.join(lines[:start] + [HOST_MACROS] + lines[end + 1:])
    ctz = re.compile(r'__device__ __forceinline__ uint32_t _CTZ\(uint64_t x\)\n\{.*?\n\}\n', re.S)
    out, k = ctz.subn('__device__ __forceinline__ uint32_t _CTZ(uint64_t x) { return (uint32_t)__builtin_ctzll(x); }\n', out)
    assert k == 1, '_CTZ not found'
    out = out.replace('__clzll(', '__builtin_clzll(')
    return out

FIELD_MUL_C = r'''__device__ __forceinline__ void QSB_EMU_FIELD_MUL_NAME(uint64_t *out,uint64_t *a,uint64_t *b){
    /* emulation: exact canonical product modulo p (the asm returns the same residue) */
    typedef unsigned __int128 u128;
    const uint64_t C = 0x1000003D1ULL;
    uint64_t r[8] = {0};
    for (int i = 0; i < 4; i++) {
        u128 carry = 0;
        for (int j = 0; j < 4; j++) {
            u128 t = (u128)a[i] * b[j] + r[i + j] + carry;
            r[i + j] = (uint64_t)t; carry = t >> 64;
        }
        r[i + 4] = (uint64_t)carry;
    }
    uint64_t s[4]; u128 acc = 0;
    for (int i = 0; i < 4; i++) { acc += (u128)r[4 + i] * C + r[i]; s[i] = (uint64_t)acc; acc >>= 64; }
    uint64_t top = (uint64_t)acc;
    acc = (u128)top * C + s[0]; s[0] = (uint64_t)acc; acc >>= 64;
    for (int i = 1; i < 4; i++) { acc += s[i]; s[i] = (uint64_t)acc; acc >>= 64; }
    if (acc) {
        acc = (u128)s[0] + C; s[0] = (uint64_t)acc; acc >>= 64;
        for (int i = 1; i < 4; i++) { acc += s[i]; s[i] = (uint64_t)acc; acc >>= 64; }
    }
    if ((s[1] & s[2] & s[3]) == UINT64_MAX && s[0] >= 0xFFFFFFFEFFFFFC2FULL) {
        s[0] -= 0xFFFFFFFEFFFFFC2FULL; s[1] = s[2] = s[3] = 0;
    }
    out[0]=s[0];out[1]=s[1];out[2]=s[2];out[3]=s[3];out[4]=0;
}
'''

total = 0
for p in list(out_dir.rglob('*.cu')) + list(out_dir.rglob('*.cuh')):
    t = p.read_text()
    t2, n = transform_launches(t)
    total += n
    if p.name == 'tree.cu':
        raw = '__device__ __forceinline__ void qsb_field_mul_raw(uint64_t *out,uint64_t *a,uint64_t *b){'
        old = '__device__ __forceinline__ void qsb_field_mul(uint64_t *out,uint64_t *a,uint64_t *b){'
        name = 'qsb_field_mul_raw' if raw in t2 else 'qsb_field_mul'
        start = t2.index(raw if raw in t2 else old)
        end = t2.index('\n}\n', start) + 3
        assert 'asm(' in t2[start:end], 'field multiply body is not the asm one'
        t2 = t2[:start] + FIELD_MUL_C.replace('QSB_EMU_FIELD_MUL_NAME', name) + t2[end:]
        # GPUHash.h's funnel-shift sigma asm equals its C rotation macros
        t2, k = re.subn(r'#define ASSEMBLY_SIGMA 1', '/* emu: ASSEMBLY_SIGMA off (C rotations) */', t2)
        assert k == 1
        if launch_blocks:
            if '#ifndef ZLAB_LAUNCH_BLOCKS' in t2:
                t2, k = re.subn(r'#define ZLAB_LAUNCH_BLOCKS\s+\d+', f'#define ZLAB_LAUNCH_BLOCKS {launch_blocks}', t2)
            else:
                t2, k = re.subn(r'#define QSB_SE_LAUNCH_BLOCKS\s+\d+', f'#define QSB_SE_LAUNCH_BLOCKS {launch_blocks}', t2)
            assert k == 1
    if t2 != t:
        p.write_text(t2)
print(f'launch sites transformed: {total}')
gm = out_dir / 'GPUMath.h'
gm.write_text(host_port(gm.read_text()))
cmd = ['g++', '-O2', '-std=c++17', '-fopenmp', '-w', '-x', 'c++', f'-I{HERE / "include"}',
       f'-DQSB_ZEROS_N={zeros}', '-o', str(out_dir / 'subset'), str(out_dir / 'subset.cu'),
       '-x', 'none', '-lcrypto', '-lm']
print(' '.join(cmd))
subprocess.run(cmd, check=True)
print('built', out_dir / 'subset')
