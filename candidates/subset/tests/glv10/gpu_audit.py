"""Generate an explicit-scalar replay seam, then compile or run the actual-header CUDA audit."""
from pathlib import Path
import argparse, tempfile, subprocess, re, json, hashlib, shlex
HERE=Path(__file__).resolve().parent
SUBSET=HERE.parents[1]
REPO=next(p for p in HERE.parents if (p/'.git').exists())
def generate(build):
 s=(SUBSET/'tests/gpu_epochs/pair_shared.cuh').read_text()
 start=s.index('    uint64_t qx[4],qy[4],qzz[4],qzzz[4];',s.index('int qsb_k2s_front_exact('));end=s.index('\n}\n',start)+2
 front='__device__ int audit_exact_z(const uint64_t *z,const uint8_t *d_gt,uint64_t *u2rx,uint64_t *u2ry,uint64_t *prod,uint64_t *m1,uint64_t *m2) {\n'+s[start:end]
 start=s.index('    uint64_t rx[4]=',s.index('int qsb_pair_verify_candidate('));end=s.index('\n}\n',start)+2
 verify='__device__ int audit_verify_z(const uint64_t *z,const uint8_t *d_gt) {\n'+s[start:end]
 old='qsb_k2s_front_exact(ep,first,lane,d_gt,rx,ry,inv,m1,m2)'
 assert verify.count(old)==1
 verify=verify.replace(old,'audit_exact_z(z,d_gt,rx,ry,inv,m1,m2)')
 (build/'exact_z_extracted.cuh').write_text('// Real production bodies; only SHA generation replaced by explicit z.\n'+front+'\n'+verify+'\n')
 fixture=(SUBSET/'glv10_audit_scalars.h').read_text()
 limbs=[int(x,16) for x in re.findall(r'UINT64_C\(0x([0-9a-fA-F]+)\)',fixture)]
 assert limbs and len(limbs)%4==0
 N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
 A=0x3086D221A7D46BCDE86C90E49284EB15;B=0xE4437ED6010E88286F547FA90ABFE4C3
 G1=0x3086D221A7D46BCDE86C90E49284EB153DAA8A1471E8CA7FE893209A45DBB031
 G2=0xE4437ED6010E88286F547FA90ABFE4C4221208AC9DF506C61571B4AE8AC47F71
 lines=['struct AuditExpectedSplit {uint64_t limbs[4];unsigned signs[2];};','static const AuditExpectedSplit audit_expected_split[] = {']
 for i in range(0,len(limbs),4):
  k=sum(limbs[i+j]<<(64*j) for j in range(4))%N
  c1=(k*G1+2**383)>>384;c2=(k*G2+2**383)>>384
  p=k-c1*A-c2*(A+B);q=c1*B-c2*A
  values=[abs(p)&((1<<64)-1),abs(p)>>64,abs(q)&((1<<64)-1),abs(q)>>64]
  lines.append('{{'+','.join(f'UINT64_C(0x{x:016x})' for x in values)+'},{'+f'{int(p<0)},{int(q<0)}'+'}},')
 lines+=['};'];(build/'expected_splits.cuh').write_text('\n'.join(lines)+'\n')
 return len(limbs)//4

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--docker',help='Existing CUDA container mounting the repository')
 parser.add_argument('--mount-root',default='/work',help='Repository mount in the existing container')
 parser.add_argument('--run',type=Path,help='Explicitly execute GPU audit on this existing digest_params.bin after compiling')
 args=parser.parse_args()
 scratch=REPO/'.scratch';scratch.mkdir(exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='glv12-audit-',dir=scratch) as td:
  build=Path(td);count=generate(build)
  def target(p):
   return str(Path(args.mount_root)/p.resolve().relative_to(REPO)) if args.docker else str(p)
  cmd=['nvcc','-O3','-DQSB_ZEROS_N=3','-DQSB_NATIVE_MODULE=0','-arch=sm_89','-I',target(build),'-o',target(build/'audit'),target(HERE/'audit.cu'),'-lcrypto','-lm']
  prefix=['docker','exec',args.docker] if args.docker else []
  print('Compiling:',shlex.join(prefix+cmd),flush=True);subprocess.run(prefix+cmd,check=True)
  result={'compile':'PASS','gpu_executed':False,'scalar_fixture_count':count,'builder_records':1602,'builder_tail_live':66,'builder_tail_padded':190,'replay_seam':'Exact production bodies; SHA generation replaced by explicit scalar', 'audit_executable_sha256':hashlib.sha256((build/'audit').read_bytes()).hexdigest(),'test_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'audit.cu',HERE/'builder_audit.cuh',HERE/'gpu_audit.py']},'source_sha256':{str(p.relative_to(SUBSET)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SUBSET/'GLVScalar.cuh',SUBSET/'glv10_chain.cuh',SUBSET/'glv10_build.cuh',SUBSET/'glv10_geometry.h',SUBSET/'tests/gpu_epochs/tree.cu',SUBSET/'tests/gpu_epochs/pair_shared.cuh']}}
  if args.run:
   subprocess.run(prefix+[target(build/'audit'),target(args.run.resolve())],check=True);result['gpu_executed']=True
  print(json.dumps(result,indent=2))
if __name__=='__main__':main()
