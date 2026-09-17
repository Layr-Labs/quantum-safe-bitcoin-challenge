#!/usr/bin/env python3
"""Model the actual inverse-tree PTX, then execute its actual C++ canonical tail.

No GPU execution. Checks the distinct tree multiplier rather than substituting
hot _ModMultCore evidence for it. Unused predicate declarations alone are elided
from the strict PTX interpreter; any use or unsupported instruction fails closed.
"""
import argparse,ctypes as C,hashlib,itertools,json,random,re,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preflight import source_identity
from ptx_field_model import Program,extract_ptx,function,check_semantics
P=(1<<256)-(1<<32)-977;MASK=(1<<64)-1

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,default=ROOT);ap.add_argument('--report',type=Path,required=True);args=ap.parse_args()
 base=args.source.resolve();identity=source_identity(base);body=function((base/'tests/gpu_epochs/tree.cu').read_text(),'__device__ __forceinline__ void qsb_field_mul(')
 ptx=extract_ptx(body);adapted=ptx;removed=[]
 for declaration,names in re.findall(r'(\.reg \.pred ([^;]+);)',ptx):
  adapted=adapted.replace(declaration,'')
  for name in names.split(','):
   name=name.strip();assert not re.search(r'\b'+re.escape(name)+r'\b',adapted),'Used predicate unsupported';removed.append(name)
 check_semantics();program=Program(adapted)
 tail=body[body.index('    // A 256-bit result can exceed p'):body.rfind('}')]
 code='#include <stdint.h>\nextern "C" void canonical(uint64_t*out,const uint64_t*raw){uint64_t r0=raw[0],r1=raw[1],r2=raw[2],r3=raw[3];\n'+tail+'\n}\n'
 edges=[0,1,2,P-65537,P-65536,P-2,P-1,P,P+1,(1<<256)-1,1<<128,1<<255]
 cases=list(itertools.product(edges,repeat=2));cases += [(P-i,P-j)for i in [1,65535,65536,65537,100000,1<<31]for j in [1,65535,65536,65537,100000,1<<31]]
 rng=random.Random(2026091762);cases += [(rng.getrandbits(256),rng.getrandbits(256))for _ in range(2000)]
 corrected=0
 with tempfile.TemporaryDirectory(prefix='qsb-tree-field-')as td:
  p=Path(td);cpp=p/'tail.cpp';so=p/'tail.so';cpp.write_text(code);subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC',str(cpp),'-o',str(so)],check=True)
  lib=C.CDLL(str(so));lib.canonical.argtypes=[C.POINTER(C.c_uint64)]*2
  for a,b in cases:
   raw=program.run([a>>(64*i)&MASK for i in range(4)]+[b>>(64*i)&MASK for i in range(4)])
   value=sum(int(x)<<(64*i)for i,x in enumerate(raw));assert value%P==a*b%P
   corrected+=value>=P;out=(C.c_uint64*5)(*([0xface]*5));lib.canonical(out,(C.c_uint64*4)(*raw))
   got=sum(int(out[i])<<(64*i)for i in range(4));assert got==a*b%P and out[4]==0
  witness=(P-1,P-(1<<224));mutation=adapted.replace('addc.cc.u32 z7, z7, 0;','addc.u32 z7, z7, 0;');assert mutation!=adapted
  raw=Program(mutation).run([x>>(64*i)&MASK for x in witness for i in range(4)]);assert sum(int(x)<<(64*i)for i,x in enumerate(raw))%P!=witness[0]*witness[1]%P
 assert source_identity(base)==identity
 report={'status':'PASS',**identity,'actual_ptx_and_tail_cases':len(cases),'canonicalized_cases':corrected,'carry_mutation_rejected':True,'unused_predicate_declarations_elided':removed,'function_sha256':hashlib.sha256(body.encode()).hexdigest(),'audit_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'gpu_executed':False,'limitations':__doc__};args.report.write_text(json.dumps(report,indent=2)+'\n');print(report['status'],len(cases))
if __name__=='__main__':main()
