#!/usr/bin/env python3
"""Actual PR217 host M/S and five-output device PTX+cold-correction model."""
import argparse,ctypes as C,hashlib,itertools,json,random,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'research'))
from preflight import source_identity
from ptx_field_model import Program,function,extract_ptx,check_semantics
P=(1<<256)-(1<<32)-977;U=1<<256;K=U-P;MASK=(1<<64)-1
words=lambda x:[x>>(64*i)&MASK for i in range(4)]
integer=lambda v:sum(int(x)<<(64*i) for i,x in enumerate(v))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();src=a.source.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);identity=source_identity(src)
 text=(src/'GPUMath.h').read_text();bodies=[function(text,'__device__ __forceinline__ void _ModMultCore('),function(text,'__device__ __forceinline__ void _ModSqr(')];cold=function(text,'__device__ __noinline__ QsbFieldWords qsb_field_cold_correction(')
 check_semantics()
 # Adapt only PTX operand ABI: five outputs, input operands begin at5. Actual
 # instruction strings stay unchanged. Cold helper retains four-output ABI.
 model=(ROOT/'research/ptx_field_model.py').read_text();assert model.count('i+4')==1 and model.count("for i in range(4)]")==1
 adapted=model.replace('i+4','i+5').replace('for i in range(4)]','for i in range(5)]');ns={'__name__':'operand_abi5'};exec(compile(adapted,'[five-output-PTX-model]','exec'),ns);Five=ns['Program']
 codes=[extract_ptx(b) for b in bodies];programs=[Five(c) for c in codes];coldprog=Program(extract_ptx(cold))
 condition='if ((uint32_t)((uint32_t)(r3 >> 32) + 1u) <= 1u)'
 assert all(condition in b for b in bodies)
 assert 'if (!carry && !((x.b & x.c & x.d) == UINT64_MAX && x.a >= 0xfffffffefffffc2fULL)) return x;' in cold
 def device(which,x,y,mutant=False):
  raw=programs[which].run(words(x)+([] if which else words(y)));limbs,carry=raw[:4],raw[4];assert carry in (0,1)
  z=integer(limbs);prod=x*(x if which else y);first=prod%U+K*(prod//U);second=first%U+K*(first//U);assert second==z+carry*U and second<=U-1+K*K
  selected=(((limbs[3]>>32)+1)&0xffffffff)<=1
  required=bool(carry or z>=P);assert not required or selected
  if selected and (required if not mutant else z>=P):limbs=coldprog.run(limbs)
  return integer(limbs),carry,selected,required
 cpp='#include <stdint.h>\n#define __device__\n#define __forceinline__ inline\n'+ '\n'.join(bodies)+'\nextern "C" void mul(uint64_t*r,const uint64_t*a,const uint64_t*b){_ModMultCore(r,a,b);}\nextern "C" void sq(uint64_t*r,const uint64_t*a){_ModSqr(r,a);}\n'
 (out/'projection.cpp').write_text(cpp);cp=subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC',str(out/'projection.cpp'),'-o',str(out/'projection.so')],capture_output=True,text=True);(out/'compile.log').write_text(cp.stdout+cp.stderr);assert cp.returncode==0,cp.stderr
 lib=C.CDLL(str(out/'projection.so'));u64=C.POINTER(C.c_uint64);lib.mul.argtypes=[u64]*3;lib.sq.argtypes=[u64]*2
 rng=random.Random(21724);edges=[0,1,2,65537,1<<64,1<<128,1<<255,P-65537,P-(1<<64),P-2,P-1,P,P+1,U-2,U-1];cases=list(itertools.product(edges,repeat=2))+[(P-i,P-j) for i in [1,65535,65536,65537,1<<31] for j in [1,65535,65536,65537,1<<31]]+[(rng.getrandbits(256),rng.getrandbits(256)) for _ in range(2000)]
 counts={'host_M':0,'host_S':0,'PTX_M_with_cold':0,'PTX_S_with_cold':0};branches={'carry_repair':0,'normalization':0,'prefilter_false_positive':0};witnesses=[]
 for x,y in cases:
  for square in [0,1]:
   expected=x*(x if square else y)%P
   for alias in range(2 if square else 3):
    aa=(C.c_uint64*4)(*words(x));bb=(C.c_uint64*4)(*words(y));dest=(C.c_uint64*4)() if alias==0 else aa if alias==1 else bb
    (lib.sq(dest,aa) if square else lib.mul(dest,aa,bb));assert integer(dest)==expected,(x,y,square,alias);counts['host_S' if square else 'host_M']+=1
   v,carry,selected,required=device(square,x,y);assert v==expected and v<P,(x,y,square,v,expected);counts['PTX_S_with_cold' if square else 'PTX_M_with_cold']+=1
   if carry:branches['carry_repair']+=1
   elif required:branches['normalization']+=1
   elif selected:branches['prefilter_false_positive']+=1
  if (x,y) in [(P-65537,P-65537),(P-1,P-(1<<64)),(U-1,U-1),(P,1)]:witnesses.append({'a':hex(x),'b':hex(y),'M':hex(device(0,x,y)[0]),'S':hex(device(1,x,y)[0]),'expected_M':hex(x*y%P),'expected_S':hex(x*x%P)})
 # Removing carry-triggered cold correction must fail a canonical witness.
 assert device(0,P-65537,P-65537,True)[0]!=(P-65537)**2%P
 # Canonicalization-alone witness distinguishes dropping normalization.
 raw=programs[0].run(words(P)+words(1));assert integer(raw[:4])==P and raw[4]==0 and device(0,P,1)[0]==0
 assert source_identity(src)==identity
 result={'status':'PASS_ACTUAL_PR217_HOST_AND_DEVICE_COLD_CORRECTION_MODEL',**identity,'checker_sha256':sha(Path(__file__)),'compiled':True,'counts':counts,'branches':branches,'witnesses':witnesses,'mutation_controls':{'drop_carry_trigger_canonical_witness_detected':True,'drop_canonicalization_p_times_one_detected':True},'function_sha256':{name:hashlib.sha256(body.encode()).hexdigest() for name,body in zip(['_ModMultCore','_ModSqr','qsb_field_cold_correction'],bodies+[cold])},'ptx_model_sha256':sha(ROOT/'research/ptx_field_model.py'),'adapted_ABI_model_sha256':hashlib.sha256(adapted.encode()).hexdigest(),'actual_device_programs_sha256':[hashlib.sha256(x.encode()).hexdigest() for x in codes+[extract_ptx(cold)]],'scope':'Actual selected host M/S compiled with alias cases; unchanged actual device PTX with five-output ABI model plus actual cold-helper PTX; source-exact C++ branch predicates modeled in Python. Not CUDA execution or complete kernel validation.','proof':'Second fold S <=2^256−1+C². Carry implies low<C²<2^65, hence top32=0. No carry and low>=p implies top32=0xffffffff. Prefilter covers both. Required cold path addsC modulo2^256, producing canonical residue. False positives return original words.','gpu_executed':False}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','source_fingerprint','counts','branches','mutation_controls']}))
if __name__=='__main__':main()
