#!/usr/bin/env python3
"""Compile the actual portable CUDA helper bodies and check with exact integers."""
from pathlib import Path
import ast
import re
import ctypes
import random
import subprocess
import tempfile

source = Path(__file__).with_name("GLVScalar.cuh").read_text()
def body(name):
    start = source.index("__device__ __forceinline__ q9_u129 " + name + "(")
    pos = source.index("{", start)
    depth = 1
    end = pos + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end].replace("__device__", "").replace("__forceinline__", "inline")
# Translate the small straight-line PTX block by instruction semantics.
# This audits the actual device source, separately from the portable helper.
def ptx_model():
    device=source[source.index("q9_product129_evenodd("):]
    asm=device[device.index('asm(')+4:device.index(': "=l"')]
    text="".join(ast.literal_eval(q) for q in re.findall(r'"(?:[^"\\]|\\.)*"',asm))
    text=text.strip()[1:-1]
    for i in range(9):text=text.replace("%"+str(i),"p"+str(i))
    lines=["inline q9_u129 emulated_ptx(const uint64_t *x,const uint32_t *d) {",
           "uint64_t p0=0,p1=0,p3=x[0],p4=x[1]; uint32_t p2=0,p5=d[0],p6=d[1],p7=d[2],p8=d[3];",
           "unsigned cc=0; unsigned __int128 wide=0;"]
    for instruction in text.split(";"):
        instruction=instruction.strip()
        if not instruction:continue
        op,args=instruction.split(None,1)
        if op==".reg":
            typ,names=args.split(None,1)
            lines.append(("uint64_t " if typ==".u64" else "uint32_t ")+names+";")
            continue
        if op=="mov.b64":
            match=re.fullmatch(r"\{([^,]+),([^}]+)\},(.+)",args)
            assert match,instruction
            lo,hi,value=match.groups()
            lines.append(f"{lo}=(uint32_t){value}; {hi}=(uint32_t)({value}>>32);")
            continue
        a=[v.strip() for v in args.split(',')];dst=a[0]
        if op.startswith(('mul.wide','mad.wide')):
            expr=f"(uint64_t){a[1]}*{a[2]}"
            if op.startswith('mad'):expr+='+'+a[3]
            lines.append(f"{dst}={expr};")
        elif op.startswith(('add.','addc.')):
            bits=64 if op.endswith('u64') else 32
            extra='+cc' if op.startswith('addc.') else ''
            lines.append(f"wide=(unsigned __int128){a[1]}+{a[2]}{extra}; {dst}=(uint{bits}_t)wide;")
            if '.cc.' in op:lines.append(f"cc=(unsigned)(wide>>{bits});")
        elif op.startswith('cvt.'):
            lines.append(f"{dst}=(uint32_t){a[1]};")
        else:
            symbol={'shl':'<<','shr':'>>','or':'|','and':'&','xor':'^'}[op.split('.')[0]]
            lines.append(f"{dst}={a[1]}{symbol}{a[2]};")
    lines.append('return {p0,p1,p2}; }')
    return "\n".join(lines)
cpp = "#include <stdint.h>\nstruct q9_u129 { uint64_t lo,hi; uint32_t top; };\n"
cpp += body("q9_product129_rows") + "\n" + body("q9_product129_evenodd")+"\n"+ptx_model()
cpp += '''
extern "C" void evaluate(const uint64_t *x,const uint32_t *d,uint64_t *out) {
    q9_u129 a=q9_product129_rows(x,d), b=q9_product129_evenodd(x,d),c=emulated_ptx(x,d);
    out[0]=a.lo;out[1]=a.hi;out[2]=a.top;
    out[3]=b.lo;out[4]=b.hi;out[5]=b.top;
    out[6]=c.lo;out[7]=c.hi;out[8]=c.top;
}
'''
mask=(1<<129)-1
rng=random.Random(0x129E0D)
count=0
with tempfile.TemporaryDirectory() as tmp:
    src=Path(tmp)/"audit.cpp"; lib=Path(tmp)/"audit.so"
    src.write_text(cpp)
    subprocess.run(["g++","-O3","-shared","-fPIC",str(src),"-o",str(lib)],check=True)
    call=ctypes.CDLL(str(lib)).evaluate
    call.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_uint64)]
    def check(x,d):
        global count
        xx=(ctypes.c_uint64*2)(x&((1<<64)-1),x>>64)
        dd=(ctypes.c_uint32*4)(*[(d>>(32*i))&0xffffffff for i in range(4)])
        out=(ctypes.c_uint64*9)();call(xx,dd,out)
        expected=(x*d)&mask
        for offset in (0,3,6):
            got=out[offset]|(out[offset+1]<<64)|(out[offset+2]<<128)
            assert got==expected,(hex(x),hex(d),offset,hex(got),hex(expected))
        count+=1
    edges={0,1,(1<<128)-1}
    for bit in range(128):
        for delta in (-1,0,1):
            edges.add(((1<<bit)+delta)&((1<<128)-1))
    for x in sorted(edges):
        for d in sorted(edges):check(x,d)
    constants=[0x3086d221a7d46bcde86c90e49284eb15,
               0x14ca50f7a8e2f3f657c1108d9d44cfd8,
               0xe4437ed6010e88286f547fa90abfe4c3]
    for _ in range(50000):
        x=rng.getrandbits(128)
        check(x,rng.getrandbits(128))
        for d in constants:check(x,d)
print(f"PASS: {count} exact product comparisons; actual helper bodies and PTX semantics; gpu_executed=false")
