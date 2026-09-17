"""Local fail-closed extension of the existing PTX field semantic model.
Only 32-bit subtract/borrow and bitwise AND/XOR are added; no GPU execution.
"""
import inspect,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import ptx_field_model as core
code=inspect.getsource(core.Program)
old="'shf.l.wrap.b32'}, opcode"
assert code.count(old)==1
code=code.replace(old,"'shf.l.wrap.b32','sub.u32','sub.cc.u32','subc.u32','subc.cc.u32','and.b32','xor.b32'}, opcode")
marker="            elif op.startswith('add'):"
assert code.count(marker)==1
code=code.replace(marker,"""            elif op.startswith('sub'):
                assert len(values)==2
                value=values[0]-values[1]-(carry if op.startswith('subc.') else 0)
                if '.cc.' in op:carry=int(value<0)
            elif op in ('and.b32','xor.b32'):
                assert len(values)==2
                value=values[0]&values[1] if op=='and.b32' else values[0]^values[1]
"""+marker)
ns={'re':core.re};exec(compile(code,__file__+'[controlled-extension]','exec'),ns);Program=ns['Program']

def check_semantics():
    core.check_semantics()
    checks=0
    for op in ['sub.u32','sub.cc.u32','subc.u32','subc.cc.u32']:
        for a in [0,1,2,0x7fffffff,0xfffffffe,0xffffffff]:
            for b in [0,1,2,0x7fffffff,0xfffffffe,0xffffffff]:
                for cin in [0,1]:
                    text='''{.reg .u32 seed,d,cf;
                    sub.cc.u32 seed,0,CIN;
                    OP d,A,B;subc.u32 cf,0,0;
                    mov.b64 %0,{d,cf};mov.b64 %1,0;mov.b64 %2,0;mov.b64 %3,0;}'''
                    for x,y in [('CIN',cin),('OP',op),('A',a),('B',b)]:text=text.replace(x,str(y))
                    full=a-b-(cin if op.startswith('subc.') else 0)
                    borrow=int(full<0) if '.cc.' in op else cin
                    assert Program(text).run([])==[(full&0xffffffff)|((-borrow&0xffffffff)<<32),0,0,0]
                    checks+=1
    return checks
