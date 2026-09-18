#!/usr/bin/env python3
"""Test narrow MAD carry fusion in the exact current even/odd product."""
import hashlib,json,re,shutil,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
sys.path.insert(0,str(HERE.parent))
from compile_local import source_closure
from ptx_field_model import extract_ptx,function

def main():
    out=HERE/'candidate'
    if out.exists():raise SystemExit('Refusing to overwrite experiment')
    files=source_closure(BASE.resolve(),'pinning.cu')+[BASE/'COPYING']
    hashes={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    assert hashes==json.loads((BASE/'research/wide_windows/submitted-source.json').read_text())['production_sha256']
    for p in files:
        dest=out/p.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    header=(out/'GPUMath.h').read_text()
    body=function(header,'__device__ __forceinline__ void _ModMultCore(')
    old=extract_ptx(body);split=old.index('.reg .u64 r0,r1,r2,r3')
    prefix=old[:split];tail=old[split:]
    pattern=r'mul\.wide\.u32 t, (a\d), (b\d);\s*(add(?:c)?(?:\.cc)?)\.u64 (\w+), (\w+), (\w+);'
    replaced=[]
    def rewrite(m):
        a,b,op,dest,c,d=m.groups()
        assert (c=='t')!=(d=='t')
        addend=d if c=='t' else c
        low='madc.lo.cc' if op.startswith('addc') else 'mad.lo.cc'
        high='madc.hi.cc' if '.cc' in op else 'madc.hi'
        replaced.append({'old':m[0],'carry_in':op.startswith('addc'),'carry_out':'.cc' in op})
        return f'mov.b64 {{ml,mh}}, {addend};\n{low}.u32 ml, {a}, {b}, ml;\n{high}.u32 mh, {a}, {b}, mh;\nmov.b64 {dest}, {{ml,mh}};'
    prefix,n=re.subn(pattern,rewrite,prefix)
    assert n==56,n
    # Terminal addc.u64 had no carry-out. The replacement's low instruction
    # changes CC, so ensure every following carry use is preceded by a fresh
    # non-carry-in .cc operation. This check concerns only the product prefix.
    for part in prefix.split('madc.hi.u32')[1:]:
        following=part[part.index(';')+1:]
        flag_ops=re.findall(r'\b(?:addc?(?:\.cc)?|subc?(?:\.cc)?|madc?\.(?:lo|hi)(?:\.cc)?)\.u(?:32|64)\b',following)
        assert flag_ops and flag_ops[0] in ('mad.lo.cc.u32','add.cc.u32','add.cc.u64'),flag_ops[:2]
    prefix=prefix.replace('{','{\n.reg .u32 ml,mh;',1)
    ptx=prefix+tail
    raw=prefix+'\n'.join(f'mov.b64 %{i}, {{x{2*i},x{2*i+1}}};' for i in range(8))+'\n}'
    encoded='\n'.join('        '+json.dumps(line+'\n') for line in ptx.splitlines())
    patched=body[:body.index('asm(')]+'asm(\n'+encoded+'\n        '+body[body.index(': "=l"'):]
    (out/'GPUMath.h').write_text(header.replace(body,patched))
    (HERE/'field.ptx').write_text(ptx+'\n');(HERE/'product.ptx').write_text(raw+'\n')
    report={'status':'isolated prototype; not submitted','base_hashes':hashes,
            'candidate_hashes':{str(p.relative_to(out.resolve())):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_closure(out.resolve(),'pinning.cu')},
            'fused_pairs':n,'raw_products':64,'reduction_tail_unchanged':True,'host_fallback_unchanged':True,
            'specialized_square_unchanged':True,'terminal_flag_liveness_checked':True,'gpu_executed':False}
    (HERE/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Generated isolated narrow MAD experiment; replaced',n,'product/add pairs')

if __name__=='__main__':main()
