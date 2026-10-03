#!/usr/bin/env python3
"""Generate an isolated difference-Karatsuba field PTX experiment.

48 raw 32-bit products, plus the exact corrected production reduction tail.
Host fallback and specialized square are unchanged. No production mutation.
"""
import hashlib,json,re,shutil,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from ptx_field_model import extract_ptx,function
from compile_local import source_closure
BASE=HERE.parent.parent

def declaration(kind,names):return '.reg .'+kind+' '+','.join(names)+';'

def mul128(a,b,out,tag):
    """4x32 even/odd accumulation, mirroring the checked production schedule."""
    e=[tag+'e'+str(i) for i in range(4)];o=[tag+'o'+str(i) for i in range(3)]
    y=[tag+'y'+str(i) for i in range(6)];cy=tag+'cy';lc=tag+'lc';o7=tag+'o7'
    s=[declaration('u64',e+o+[lc]),declaration('u32',y+[cy,o7])]
    def emit(x):s.append(x)
    def mul(dest,i,j):emit(f'mul.wide.u32 {dest}, {a[i]}, {b[j]};')
    mul(e[0],0,0);mul(e[1],0,2)
    mul('t',1,1);emit(f'add.cc.u64 {e[1]}, {e[1]}, t;')
    mul('t',1,3);emit(f'addc.u64 {e[2]}, t, 0;')
    mul('t',2,0);emit(f'add.cc.u64 {e[1]}, {e[1]}, t;')
    mul('t',2,2);emit(f'addc.cc.u64 {e[2]}, {e[2]}, t;')
    emit(f'addc.u32 {cy}, 0, 0; cvt.u64.u32 {lc}, {cy};')
    mul('t',3,1);emit(f'add.cc.u64 {e[2]}, {e[2]}, t;')
    mul('t',3,3);emit(f'addc.u64 {e[3]}, t, {lc};')
    mul(o[0],0,1);mul(o[1],0,3)
    mul('t',1,0);emit(f'add.cc.u64 {o[0]}, {o[0]}, t;')
    mul('t',1,2);emit(f'addc.cc.u64 {o[1]}, {o[1]}, t;')
    emit(f'addc.u32 {cy}, 0, 0; cvt.u64.u32 {lc}, {cy};')
    mul('t',2,1);emit(f'add.cc.u64 {o[1]}, {o[1]}, t;')
    mul('t',2,3);emit(f'addc.u64 {o[2]}, t, {lc};')
    mul('t',3,0);emit(f'add.cc.u64 {o[1]}, {o[1]}, t;')
    mul('t',3,2);emit(f'addc.cc.u64 {o[2]}, {o[2]}, t;')
    emit(f'addc.u32 {o7}, 0, 0;')
    for i in range(4):emit(f'mov.b64 {{{out[2*i]},{out[2*i+1]}}}, {e[i]};')
    for i in range(3):emit(f'mov.b64 {{{y[2*i]},{y[2*i+1]}}}, {o[i]};')
    for i in range(1,7):emit(f'{"add.cc" if i==1 else "addc.cc"}.u32 {out[i]}, {out[i]}, {y[i-1]};')
    emit(f'addc.u32 {out[7]}, {out[7]}, {o7};')
    return s

def main():
    out=HERE/'candidate'
    if out.exists():raise SystemExit('Refusing to overwrite existing experiment')
    files=source_closure(BASE.resolve(),'pinning.cu')+[BASE/'COPYING']
    source_hashes={str(p.relative_to(BASE.resolve())):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    saved=json.loads((BASE/'research/wide_windows/submitted-source.json').read_text())
    assert source_hashes==saved['production_sha256'],'Pending source changed; reassess donor'
    for p in files:
        dst=out/p.relative_to(BASE.resolve());dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    header=(out/'GPUMath.h').read_text();body=function(header,'__device__ __forceinline__ void _ModMultCore(')
    previous=extract_ptx(body)
    a=['a'+str(i) for i in range(8)];b=['b'+str(i) for i in range(8)]
    da=['da'+str(i) for i in range(4)];db=['db'+str(i) for i in range(4)]
    lo=['lo'+str(i) for i in range(8)];hi=['hi'+str(i) for i in range(8)];df=['df'+str(i) for i in range(8)]
    mid=['mid'+str(i) for i in range(9)];x=['x'+str(i) for i in range(16)]
    lines=['{',declaration('u32',a+b+da+db+lo+hi+df+mid+x+['sa','sb','neg','bit']),'.reg .u64 t;']
    for names,first in [(a,4),(b,8)]:
        for i in range(4):lines.append(f'mov.b64 {{{names[2*i]},{names[2*i+1]}}}, %{first+i};')
    for names,diff,sign in [(a,da,'sa'),(b,db,'sb')]:
        for i in range(4):lines.append(f'{"sub.cc" if i==0 else "subc.cc"}.u32 {diff[i]}, {names[i]}, {names[i+4]};')
        lines.append(f'subc.u32 {sign}, 0, 0;')
        for v in diff:lines.append(f'xor.b32 {v}, {v}, {sign};')
        lines.append(f'and.b32 bit, {sign}, 1;')
        for i in range(4):lines.append(f'{"add.cc" if i==0 else "addc.cc"}.u32 {diff[i]}, {diff[i]}, {"bit" if i==0 else "0"};')
    lines+=mul128(a[:4],b[:4],lo,'low')
    lines+=mul128(a[4:],b[4:],hi,'high')
    lines+=mul128(da,db,df,'diff')
    lines+=['xor.b32 neg, sa, sb;','not.b32 neg, neg;','and.b32 bit, neg, 1;']
    for v in df:lines.append(f'xor.b32 {v}, {v}, neg;')
    # Form the signed ninth-limb extension while negating the difference product.
    for i in range(8):lines.append(f'{"add.cc" if i==0 else "addc.cc"}.u32 {df[i]}, {df[i]}, {"bit" if i==0 else "0"};')
    lines.append('addc.u32 neg, neg, 0;')
    for i in range(8):lines.append(f'{"add.cc" if i==0 else "addc.cc"}.u32 {mid[i]}, {lo[i]}, {hi[i]};')
    lines.append('addc.u32 mid8, 0, 0;')
    for i in range(8):lines.append(f'{"add.cc" if i==0 else "addc.cc"}.u32 {mid[i]}, {mid[i]}, {df[i]};')
    lines.append('addc.u32 mid8, mid8, neg;')
    for i in range(8):lines.append(f'mov.u32 x{i}, lo{i}; mov.u32 x{i+8}, hi{i};')
    for i in range(4,16):lines.append(f'{"add.cc" if i==4 else "addc.cc"}.u32 x{i}, x{i}, {"mid"+str(i-4) if i<13 else "0"};')
    product_prefix='\n'.join(lines)+'\n'
    raw=product_prefix+'\n'.join(f'mov.b64 %{i}, {{x{2*i},x{2*i+1}}};' for i in range(8))+'\n}'
    tail=previous[previous.index('.reg .u64 r0,r1,r2,r3'):]
    ptx=product_prefix+tail
    assert ptx.count('mul.wide.u32')==57 # 48 raw products plus9 reduction products
    encoded='\n'.join('        '+json.dumps(line+'\n') for line in ptx.splitlines())
    start=body.index('asm(');end=body.index(': "=l"',start)
    patched=body[:start]+'asm(\n'+encoded+'\n        '+body[end:]
    (out/'GPUMath.h').write_text(header.replace(body,patched))
    (HERE/'product.ptx').write_text(raw+'\n');(HERE/'field.ptx').write_text(ptx+'\n')
    hashes={str(p.relative_to(out.resolve())):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_closure(out.resolve(),'pinning.cu')}
    report={'status':'isolated prototype; not submitted','base_hashes':source_hashes,'candidate_hashes':hashes,'raw_word_products':48,'reduction_mul_wide':9,'extra_reduction_mul_lo':1,'ptx_operations':len([s for s in ptx.split(';') if s.strip() and not s.strip().startswith(('.reg','{','}'))]),'host_fallback_unchanged':True,'specialized_square_unchanged':True,'gpu_executed':False}
    (HERE/'provenance.json').write_text(json.dumps(report,indent=2)+'\n');print('Generated isolated Karatsuba candidate')
if __name__=='__main__':main()
