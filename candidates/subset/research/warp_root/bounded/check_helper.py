#!/usr/bin/env python3
"""Actual bounded HM43 helper: numerical success, uniform capped failure, no publication."""
import argparse,hashlib,importlib.util,json,subprocess,sys,tempfile,time
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research')]
from preflight import source_identity

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=module('original_hm43_check',PARENT/'check_helper.py')
model=module('review_hm43_model',ROOT/'research/pending_sep17/pr189/check_review.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def parse_fixture(s):
 lines=s.splitlines();out=[]
 for line in lines[1:]:
  v=list(map(int,line.split()));out.append((v[0],v[1:33],v[33:65],v[65:]))
 assert int(lines[0])==len(out);return out

def cap_fixture(cases,cap):
 out=[]
 for kind,a,b,e in cases:
  if kind!=4:continue
  root=sum(a[j]<<(64*j) for j in range(4));_,batches=model.inverse(root)
  if batches<=cap:continue
  expected=[]
  for lane in range(32):expected.extend(a[:5] if lane==0 else [0xdeadbeef+lane*31+j for j in range(5)])
  out.append((kind,a,b,expected))
 assert out
 return str(len(out))+'\n'+'\n'.join(' '.join(map(str,[k]+a+b+e)) for k,a,b,e in out)+'\n',len(out)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,default=HERE/'candidate');ap.add_argument('--output',type=Path,default=HERE/'helper-results.json');args=ap.parse_args()
 source=args.source.resolve();identity=source_identity(source)
 hp=source/'tests/gpu_epochs/hm43_warp_inverse.cuh';header=hp.read_text()
 assert 'bool hm43_warp_inverse(' in header and '#define HM43_WARP_MAX_BATCHES 16' in header
 frozen=json.loads((PARENT/'helper-results.json').read_text())
 for filename,key in [('host_warp.hpp','host_warp_sha256'),('host_support.hpp','host_support_sha256'),('field_projection.hpp','field_projection_sha256')]:assert sha(PARENT/filename)==frozen[key]
 assert sha(source/'GPUMath.h')==frozen['corrected_dependency_source_sha256']
 data,counts=base.fixtures();cases=parse_fixture(data);cap0,n0=cap_fixture(cases,0);cap1,n1=cap_fixture(cases,1)
 old='    hm43_warp_inverse(out,lane);'
 assert base.CPP.count(old)==1
 cpp=base.CPP.replace(old,'''    bool succeeded=hm43_warp_inverse(out,lane);
    if(succeeded != (EXPECTED_SUCCESS!=0)) {
     if(!failed.exchange(true))fprintf(stderr,"STATUS_MISMATCH case=%zu lane=%d expected=%d actual=%d\\n",ci,lane,EXPECTED_SUCCESS,(int)succeeded);
    }''')
 # Exact cap-site text is supplied by the bounded source and asserted below.
 import re
 capmatch=re.search(r'if\s*\([^\n]*HM43_WARP_MAX_BATCHES[^\n]*\)\s*return false;',header)
 assert capmatch,'Expected single-line uniform budget check'
 capsite=capmatch.group();assert header.count(capsite)==1
 runs={
 'cap16_positive':(header,16,True,data),
 'cap0_no_publication':(header,0,False,cap0),
 'cap1_no_publication':(header,1,False,cap1),
 'carry_dropped':(header.replace('return sum+((carry>>lane)&1u);','return sum;'),16,True,data),
 'negate_word5_omitted':(header.replace('return ~x+(uint64_t)((zeros&lower)==lower);','return (lane&7)==5 ? x : ~x+(uint64_t)((zeros&lower)==lower);'),16,True,data),
 'guard_propagation_leak':(header.replace('word<5 && sum<a','word<8 && sum<a').replace('word<6 && sum==~0ULL','word<8 && sum==~0ULL'),16,True,data),
 'cap0_false_success':(header.replace(capsite,capsite.replace('return false;','return true;')),0,False,cap0),
 'cap0_publishes_output':(header.replace(capsite,capsite.replace('return false;','{ result[0]^=1; return false; }')),0,False,cap0),
 }
 assert len(set(x[0] for x in runs.values()))==6
 artifacts=HERE/'helper-artifacts';artifacts.mkdir(exist_ok=True);reports={}
 with tempfile.TemporaryDirectory(prefix='qsb-hm43-bounded-') as td:
  td=Path(td)
  for name,(body,cap,success,fixture) in runs.items():
   sub=td/name;sub.mkdir();(sub/'hm43_warp_inverse.cuh').write_text(body);(sub/'check.cpp').write_text(cpp)
   executable=sub/'check'
   cmd=['c++','-std=c++17','-O1','-pthread','-fsanitize=address,undefined','-fno-sanitize-recover=all',f'-DHM43_WARP_MAX_BATCHES={cap}',f'-DEXPECTED_SUCCESS={int(success)}','-I',str(PARENT),str(sub/'check.cpp'),'-o',str(executable)]
   built=subprocess.run(cmd,capture_output=True,text=True,timeout=60);assert built.returncode==0,(name,built.stderr)
   start=time.monotonic()
   with (artifacts/(name+'.stdout')).open('w') as out,(artifacts/(name+'.stderr')).open('w') as err:
    try:r=subprocess.run([str(executable)],input=fixture,text=True,stdout=out,stderr=err,timeout=180);rc=r.returncode;timed=False
    except subprocess.TimeoutExpired:rc=None;timed=True
   out=(artifacts/(name+'.stdout')).read_text();err=(artifacts/(name+'.stderr')).read_text()
   positive=name in ['cap16_positive','cap0_no_publication','cap1_no_publication']
   if positive:assert rc==0,(name,rc,timed,err[-2000:]);result=json.loads(out)
   else:assert rc==2 and ('NUMERIC_MISMATCH' in err or 'STATUS_MISMATCH' in err),(name,rc,timed,err[-2000:]);result=None
   assert 'runtime error:' not in err and 'ERROR: AddressSanitizer' not in err
   reports[name]={'compiled':True,'cap':cap,'exit_code':rc,'timeout':timed,'elapsed_host_seconds_not_performance_proxy':round(time.monotonic()-start,3),'source_sha256':digest(body),'fixture_sha256':digest(fixture),'result':result,'diagnostic':next((s for s in err.splitlines() if 'MISMATCH' in s),None),'stdout_sha256':sha(artifacts/(name+'.stdout')),'stderr_sha256':sha(artifacts/(name+'.stderr'))}
   print(name,rc,result or reports[name]['diagnostic'],flush=True)
 assert source_identity(source)==identity
 report={'status':'PASS_ACTUAL_BOUNDED_HM43_HOST_SUCCESS_FAILURE_AND_MUTATIONS',**identity,'header_sha256':sha(hp),'checker_sha256':sha(Path(__file__)),'original_checker_sha256':sha(PARENT/'check_helper.py'),'scalar_batch_model_sha256':sha(ROOT/'research/pending_sep17/pr189/check_review.py'),'generated_cpp_sha256':digest(cpp),'support_sha256':{f:sha(PARENT/f) for f in ['host_warp.hpp','host_support.hpp','field_projection.hpp']},'cap16_case_counts':counts,'cap0_roots':n0,'cap1_roots_requiring_more_than_one_batch':n1,'forced_failure_lane_returns_checked':32*(n0+n1),'forced_failure_output_words_unchanged_checked':160*(n0+n1),'reports':reports,'scope':'Actual bounded C++ header with32cooperative host lanes and independent bigint arithmetic. Cap0 uses all115root fixtures; cap1 uses only roots independently modeled to need>1batch. Every capped lane returnsfalse and preserves allfive input words. Existing arithmetic/carry/guard unitcases rerun atcap16.','limits':['CPU scalar carry/borrow/CTZ/clz projections, no CUDA or device scheduling execution','Finite roots do not prove universal16batch termination; safe fallback belongs to actual tree integration','Does not exercise retained scalar fallback or verify tree barriers','No throughput/native cost claim'],'candidate_modified':False,'gpu_executed':False}
 args.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
