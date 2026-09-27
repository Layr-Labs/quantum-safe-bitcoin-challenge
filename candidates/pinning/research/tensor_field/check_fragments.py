#!/usr/bin/env python3
"""CPU model of source-extracted fragment indices and warp carry collectives.

This is not CUDA execution. It decodes NVIDIA's per-lane MMA layout and checks
packing, extraction and carry behavior independently against big integers.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import screen

HERE = Path(__file__).resolve().parent
U32 = (1 << 32) - 1


def expression(text, names):
    def walk(node):
        if isinstance(node, ast.Expression): return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int): return node.value
        if isinstance(node, ast.Name): return names[node.id]
        if isinstance(node, ast.BinOp):
            a, b = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add): return a+b
            if isinstance(node.op, ast.Sub): return a-b
            if isinstance(node.op, ast.Mult): return a*b
        raise ValueError('Unknown index expression: '+text)
    return walk(ast.parse(text, mode='eval'))


def window(words, byte):
    q, shift = byte >> 2, (byte & 3) * 8
    lo, hi = words[q & 31], words[(q+1) & 31]
    return ((lo | (hi << 32)) >> shift) & U32


def reverse(value):
    return int.from_bytes(value.to_bytes(4, 'little'), 'big')


def fragments(a, b, index_expressions, second=True):
    aw = [(a >> (32*i)) & U32 if i < 8 else 0 for i in range(32)]
    bw = [(b >> (32*i)) & U32 if i < 8 else 0 for i in range(32)]
    result = [[0]*8 for _ in range(16)]
    for phase in (0,32) if second else (0,):
        aa, bb = [[None]*32 for _ in range(16)], [[None]*8 for _ in range(32)]
        for lane in range(32):
            g,t=lane>>2,lane&3
            names={'g':g,'t':t,'k':phase+4*t}
            for name, formula in index_expressions.items():
                packed=window(aw if name[0]=='a' else bw,expression(formula,names))
                if name[0]=='b':packed=reverse(packed)
                for byte in range(4):
                    i=int(name[1])*4+byte
                    v=(packed>>(8*byte))&255
                    if name[0]=='a':
                        r=g+(8 if i%8>=4 else 0)
                        c=4*t+(i%4)+(16 if i>=8 else 0)
                        assert aa[r][c] is None
                        aa[r][c]=v
                    else:
                        r=4*t+(i%4)+(16 if i>=4 else 0)
                        assert bb[r][g] is None
                        bb[r][g]=v
        for r in range(16):
            for k in range(32): assert aa[r][k]==screen.byte_at(a,phase+k-r)
        for k in range(32):
            for c in range(8): assert bb[k][c]==screen.byte_at(b,7+8*c-phase-k)
        for r in range(16):
            for c in range(8):result[r][c]+=sum(aa[r][k]*bb[k][c] for k in range(32))
    return [(result[lane>>2][2*(lane&3)],result[lane>>2][2*(lane&3)+1]) for lane in range(32)]


def get_coefficient(frags,coefficient):
    column=coefficient>>3
    owner=4*(7-(coefficient&7))+(column>>1)
    return frags[owner][column&1]


def normalize(raw):
    adjusted=[(raw[i]&U32)+((raw[i-1]>>32) if i else 0) for i in range(32)]
    assert max(adjusted) < (1<<33)
    generates=sum((int(v>>32!=0)<<i) for i,v in enumerate(adjusted))
    propagates=sum((int(v&U32==U32)<<i) for i,v in enumerate(adjusted))
    answer=[]
    for lane in range(32):
        before=(1<<lane)-1
        g=generates&before
        stop=(~propagates)&before
        # For uint32 clz, comparing <= equals comparing bit_length >=.
        carry=int(g!=0 and g.bit_length()>=stop.bit_length())
        answer.append(((adjusted[lane]&U32)+carry)&U32)
    return answer


def main():
    source=(HERE/'fragment.cuh').read_text()
    expressions={}
    for m in re.finditer(r'const uint32_t ([ab][0-3])=(.*?);',source):
        expr=re.search(r'qsb_tensor_window\([ab]w,([^()]*)\)',m[2])
        assert expr
        expressions[m[1]]=expr[1]
        if m[1][0]=='b':assert m[2].endswith(',0,0x0123)')
    assert set(expressions)=={'a0','a1','a2','a3','b0','b1'}
    for token in ['const int g=lane>>2, t=lane&3;', 'for(int phase=0;phase<64;phase+=32)',
                  'const int k=phase+4*t;', 'mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32',
                  'const uint32_t carry=(prior_g!=0 && __clz(prior_g)<=__clz(prior_stop));']:
        assert token in source, 'Source/model contract changed: '+token
    counts=0
    for a,b in screen.cases():
        frags=fragments(a,b,expressions)
        coeff=[get_coefficient(frags,t) for t in range(64)]
        assert coeff==screen.reference_coefficients(a,b)
        raw=[sum(coeff[4*i+j]<<(8*j) for j in range(4)) if i<16 else 0 for i in range(32)]
        words=normalize(raw)
        assert sum(word<<(32*i) for i,word in enumerate(words))==a*b
        assert not any(words[16:])
        counts+=1
    # Stress carry propagation independently of products: long propagate runs,
    # competing generate/stop positions, and 8192 random raw-limb vectors.
    carry_cases=[]
    for begin in range(31):
        for end in range(begin+1,32):
            x=[0]*32;x[begin]=(1<<32)
            for i in range(begin+1,end):x[i]=U32
            carry_cases.append(x)
    rng=random.Random(0xC4117)
    carry_cases += [[rng.getrandbits(44) if i<16 else 0 for i in range(32)] for _ in range(8192)]
    for raw in carry_cases:
        result=normalize(raw)
        assert sum(v<<(32*i) for i,v in enumerate(result))==sum(v<<(32*i) for i,v in enumerate(raw))
    # A source-index mutation must fail against the independently decoded tile.
    bad=dict(expressions);bad['a3']='k+16-g'
    # Uniform 0xff limbs hide some wrong offsets; use distinct byte values.
    witness=int.from_bytes(bytes(range(32)),'little')
    try:fragments(witness,screen.MASK,bad)
    except AssertionError:pass
    else:raise AssertionError('Fragment-row mutation survived')
    report={'status':'PASS CPU fragment/carry model',
            'created_at':datetime.now(timezone.utc).isoformat(),
            'source_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                             for name in ('fragment.cuh','micro.cu','control_math.h','check_fragments.py','screen.py')},
            'source_extracted_fragment_expressions':expressions,
            'full_product_cases':counts,'independent_carry_cases':len(carry_cases),
            'mutations_rejected':['wrong_A_fragment_row'],
            'gpu_executed':False,
            'limits':'CPU model with source-extracted fragment indices, documented matrix ownership and separately modeled collectives. Does not execute the C++/CUDA header or validate device intrinsic/runtime behavior.'}
    (HERE/'fragment-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
