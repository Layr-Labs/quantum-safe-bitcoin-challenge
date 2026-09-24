# SPDX-License-Identifier: GPL-3.0-only
"""Exercise the actual counter-reset helper with real CUDA streams.
This is a scheduling/component test, NOT a complete QSB correctness or speed test.
Set QSB_TEST_NVRTC_DIR on Windows; on Linux install CUDA NVRTC in the library path.
"""
import ctypes as C,ctypes.util,json,hashlib,time,os,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
LAB=Path(os.environ.get('QSB_TEST_OUTPUT_DIR') or tempfile.mkdtemp(prefix='qsb-reset-'))
LAB.mkdir(parents=True,exist_ok=True)
header=(HERE/'RootCounterReset.cuh').read_text()
if os.name=='nt':
 nvdir=Path(os.environ['QSB_TEST_NVRTC_DIR']);dll_dir=os.add_dll_directory(str(nvdir))
 builtins=C.WinDLL(str(next(nvdir.glob('nvrtc-builtins64_*.dll'))))
 nv=C.WinDLL(str(next(p for p in sorted(nvdir.glob('nvrtc64_*.dll')) if p.stem.replace('nvrtc64_','').replace('_','').isdigit())));cu=C.WinDLL('nvcuda.dll')
else:
 cu=C.CDLL('libcuda.so.1');nv=C.CDLL(C.util.find_library('nvrtc') or 'libnvrtc.so')
P=C.c_void_p;U=C.c_uint;SZ=C.c_size_t;I=C.c_int;U64=C.c_uint64
def bind(lib,name,args):
 f=getattr(lib,name);f.argtypes=args;f.restype=I;return f
def ok(rc):
 if rc:raise RuntimeError('CUDA/NVRTC error '+str(rc))
for name,args in {
'cuInit':[U],'cuDeviceGet':[C.POINTER(I),I],'cuDeviceGetName':[P,I,I],
'cuCtxCreate_v2':[C.POINTER(P),U,I],'cuCtxDestroy_v2':[P],
'cuModuleLoadData':[C.POINTER(P),P],'cuModuleGetFunction':[C.POINTER(P),P,C.c_char_p],
'cuMemAlloc_v2':[C.POINTER(U64),SZ],'cuMemFree_v2':[U64],
'cuMemcpyDtoH_v2':[P,U64,SZ],
'cuLaunchKernel':[P,U,U,U,U,U,U,U,P,C.POINTER(P),C.POINTER(P)],
'cuCtxSynchronize':[],'cuEventCreate':[C.POINTER(P),U],
'cuEventRecord':[P,P],'cuEventSynchronize':[P]
}.items():bind(cu,name,args)
for name,args in {
'nvrtcCreateProgram':[C.POINTER(P),C.c_char_p,C.c_char_p,I,C.POINTER(C.c_char_p),C.POINTER(C.c_char_p)],
'nvrtcCompileProgram':[P,I,C.POINTER(C.c_char_p)],
'nvrtcGetProgramLogSize':[P,C.POINTER(SZ)],'nvrtcGetProgramLog':[P,P],
'nvrtcGetCUBINSize':[P,C.POINTER(SZ)],'nvrtcGetCUBIN':[P,P],
'nvrtcDestroyProgram':[C.POINTER(P)]
}.items():bind(nv,name,args)
for name,args in {
'cuStreamCreateWithPriority':[C.POINTER(P),U,I], 'cuStreamDestroy_v2':[P],
'cuCtxGetStreamPriorityRange':[C.POINTER(I),C.POINTER(I)],
'cuStreamWaitEvent':[P,P,U], 'cuEventDestroy_v2':[P],
'cuMemsetD32Async':[U64,U,SZ,P], 'cuModuleUnload':[P]
}.items():bind(cu,name,args)
ok(cu.cuInit(0));dev=I();ok(cu.cuDeviceGet(C.byref(dev),0));ctx=P();ok(cu.cuCtxCreate_v2(C.byref(ctx),0,dev))
gpu=C.create_string_buffer(256);ok(cu.cuDeviceGetName(gpu,256,dev))
lo=I();hi=I();ok(cu.cuCtxGetStreamPriorityRange(C.byref(lo),C.byref(hi)))
kernels=r'''
extern "C" __global__ void prepare(uint32_t *p,unsigned tag){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<1024) p[i+1]=0xccccccccu;
 if(i==0) p[1025]=tag;
}
extern "C" __global__ void roots(uint32_t *p){
 unsigned i=blockIdx.x*256u+threadIdx.x;
 qsb_reset_hit_count_in_root(p,i);
}
extern "C" __global__ void finish(uint32_t *p,unsigned n,unsigned tag){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n && i%7u==0u){unsigned pos=atomicAdd(p,1u);if(pos<1024u)p[pos+1]=tag*100000u+i;}
}
'''
def compile(flag):
 src='typedef unsigned int uint32_t;\n#define QSB_ROOT_COUNTER_RESET '+str(flag)+'\n'+header+kernels
 pr=P();ok(nv.nvrtcCreateProgram(C.byref(pr),src.encode(),b'root_reset_test.cu',0,None,None))
 opts=[('--gpu-architecture='+os.environ.get('QSB_TEST_ARCH','sm_89')).encode(),b'--std=c++17']
 rc=nv.nvrtcCompileProgram(pr,len(opts),(C.c_char_p*len(opts))(*opts))
 sz=SZ();ok(nv.nvrtcGetProgramLogSize(pr,C.byref(sz)));log=C.create_string_buffer(sz.value);ok(nv.nvrtcGetProgramLog(pr,log))
 if rc:raise RuntimeError(log.value.decode())
 ok(nv.nvrtcGetCUBINSize(pr,C.byref(sz)));binary=C.create_string_buffer(sz.value);ok(nv.nvrtcGetCUBIN(pr,binary));ok(nv.nvrtcDestroyProgram(C.byref(pr)))
 mod=P();ok(cu.cuModuleLoadData(C.byref(mod),binary));funcs={}
 for name in ['prepare','roots','finish']:
  f=P();ok(cu.cuModuleGetFunction(C.byref(f),mod,name.encode()));funcs[name]=f
 return mod,funcs,hashlib.sha256(src.encode()).hexdigest()
def launch(f,grid,stream,args):
 vals=list(args);av=(P*len(vals))(*(C.addressof(v) for v in vals))
 ok(cu.cuLaunchKernel(f,grid,1,1,256,1,1,0,stream,av,None))
mods={f:compile(f) for f in [0,1]};cases=0;records=0;result_rows=[]
for mode in range(4):
 for fused in [0,1]:
  slots=[]
  for _ in range(2):
   p=P();a=P();ok(cu.cuStreamCreateWithPriority(C.byref(p),1,0));ok(cu.cuStreamCreateWithPriority(C.byref(a),1,0 if mode==3 else hi.value))
   events=[]
   for j in range(3):
    e=P();ok(cu.cuEventCreate(C.byref(e),2));events.append(e)
   d=U64();ok(cu.cuMemAlloc_v2(C.byref(d),1026*4));ok(cu.cuMemsetD32Async(d,0xdead0000,1026,p))
   slots.append(dict(p=p,a=a,e=events,d=d,pending=None))
  def drain(sl):
   global cases,records
   if sl['pending'] is None:return
   n,tag=sl['pending'];ok(cu.cuEventSynchronize(sl['e'][2]));data=(U*1026)();ok(cu.cuMemcpyDtoH_v2(data,sl['d'],1026*4))
   expected=[tag*100000+i for i in range(n) if i%7==0]
   assert data[0]==len(expected),(mode,fused,n,tag,data[0],len(expected))
   assert sorted(data[1:1+data[0]])==expected,(mode,fused,'hit set',n,tag)
   assert data[1025]==tag,'slot attribution changed'
   cases+=1;records+=len(expected);sl['pending']=None
  funcs=mods[fused][1]
  for j in range(80):
   sl=slots[j%2];drain(sl);n=[1,127,128,129,255,256,257,511][j%8];tag=1+j+100*mode
   p,a=sl['p'],sl['a'];ep,er,ed=sl['e'];d=sl['d']
   if not fused:ok(cu.cuMemsetD32Async(d,0,1,p))
   launch(funcs['prepare'],4,p,[d,U(tag)])
   if mode:
    ok(cu.cuEventRecord(ep,p));ok(cu.cuStreamWaitEvent(a,ep,0))
   rs=a if mode else p
   launch(funcs['roots'],2,rs,[d])
   if mode==1:
    ok(cu.cuEventRecord(er,a));ok(cu.cuStreamWaitEvent(p,er,0))
   finish_stream=a if mode in [2,3] else p
   launch(funcs['finish'],(n+255)//256,finish_stream,[d,U(n),U(tag)])
   ok(cu.cuEventRecord(ed,finish_stream));sl['pending']=(n,tag)
  for sl in slots:
   drain(sl)
   for e in sl['e']:ok(cu.cuEventDestroy_v2(e))
   ok(cu.cuStreamDestroy_v2(sl['a']));ok(cu.cuStreamDestroy_v2(sl['p']));ok(cu.cuMemFree_v2(sl['d']))
  result_rows.append(dict(mode=mode,fused=bool(fused),batches=80,passed=True))
# A missing reset must fail: prove the fixture does not silently zero memory.
d=U64();ok(cu.cuMemAlloc_v2(C.byref(d),1026*4));ok(cu.cuMemsetD32Async(d,99,1026,None))
f=mods[0][1];launch(f['prepare'],4,None,[d,U(1)]);launch(f['roots'],2,None,[d]);launch(f['finish'],1,None,[d,U(7),U(1)])
ok(cu.cuCtxSynchronize());v=U();ok(cu.cuMemcpyDtoH_v2(C.byref(v),d,4));assert v.value==100,'negative control unexpectedly passed';ok(cu.cuMemFree_v2(d))
report={'timestamp':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'gpu':gpu.value.decode(),'compiler':'NVRTC 12.6.85 sm_89','source_header_sha256':hashlib.sha256(header.encode()).hexdigest(),'batches_checked':cases,'hit_records_checked':records,'stream_modes':4,'slots_per_mode':2,'cases':result_rows,'missing_reset_negative_control_detected':True,'gpu_executed':True,'actual_candidate_helper_compiled':True,'full_qsb_translation_unit_compiled':False,'cryptographic_pipeline_executed':False,'performance_claim':None,'official_score':None}
(LAB/'gpu-counter-result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
for mod,_,_ in mods.values():ok(cu.cuModuleUnload(mod))
ok(cu.cuCtxDestroy_v2(ctx))
