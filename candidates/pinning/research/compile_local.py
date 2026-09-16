#!/usr/bin/env python3
"""Compile an immutable include closure in the task's local ARM Linux VM.

This checks real nvcc/ptxas code generation, never GPU execution or throughput.
Requires the documented local Lima VM and CUDA 12.8 toolkit; does not install,
start, stop, submit, or change production files.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent

def source_closure(root,entry):
    seen=set()
    def visit(path):
        path=path.resolve()
        if path in seen:return
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('Missing/out-of-source include: '+str(path))
        seen.add(path)
        for name in re.findall(r'^\s*#include\s+"([^"\n]+)"',path.read_text(),re.M):visit(path.parent/name)
    visit(root/entry)
    return sorted(seen)

def parse_resources(text):
    out={}
    for m in re.finditer(r"Compiling entry function '([^']+)' for '([^']+)'\n(.*?)(?=ptxas info\s*: Compiling entry function|\Z)",text,re.S):
        name,arch,body=m.groups();used=re.search(r'Used (\d+) registers[^\n]*',body)
        stack=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',body)
        shared=re.search(r'(\d+) bytes smem',used[0] if used else '')
        out[name]={'target':arch,'registers':int(used[1]) if used else None,
                   'shared_bytes':int(shared[1]) if shared else 0}
        if stack:out[name].update(zip(('stack_bytes','spill_store_bytes','spill_load_bytes'),map(int,stack.groups())))
    return out

def parse_sass(text):
    out={}
    for part in re.split(r'\s*Function : ',text)[1:]:
        name=part.splitlines()[0].strip();ops=[];offsets=[]
        for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+(?:@\S+\s+)?([A-Z][A-Z0-9_.]*)\b',part):
            offsets.append(int(m[1],16));ops.append(m[2])
        if ops:out[name]={'instruction_slots':max(offsets)//16+1,'non_nop':sum(op!='NOP' for op in ops),'opcodes':dict(sorted(Counter(ops).items()))}
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=HERE/'pr24_corrected')
    ap.add_argument('--entry',default='pinning.cu')
    ap.add_argument('--work',type=Path,default=Path('/tmp/qsb-cuda-work'))
    ap.add_argument('--guest-work',type=Path,default=Path('/tmp/qsb-cuda-work'),help='mount point for --work inside the Linux VM')
    ap.add_argument('--lima',default='/tmp/qsb-lima-bin/bin/limactl')
    ap.add_argument('--lima-state',default='/tmp/qsb-lima-state')
    ap.add_argument('--instance',default='qsb-cuda')
    ap.add_argument('--default-build',action='store_true',help='also compile without an explicit architecture, matching official flags')
    ap.add_argument('--report',type=Path)
    args=ap.parse_args();root=args.source.resolve();work=args.work.resolve()
    if not work.is_dir():ap.error('The mounted task work directory does not exist')
    stage=Path(tempfile.mkdtemp(prefix='compile-',dir=work));build=stage/'build';build.mkdir()
    def guest(path):return str(args.guest_work/path.relative_to(work))
    hashes={}
    for src in source_closure(root,args.entry):
        rel=src.relative_to(root);dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        hashes[str(rel)]=hashlib.sha256(src.read_bytes()).hexdigest()
    env=dict(os.environ);env['LIMA_HOME']=args.lima_state
    prefix=[args.lima,'shell',args.instance,'--'];cuda='/usr/local/cuda-12.8/bin/'
    def run(command,output):
        with output.open('w') as f:
            result=subprocess.run(prefix+command,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'CUDA build command failed ({result.returncode}); inspect {output}')
    compiler=subprocess.check_output(prefix+[cuda+'nvcc','--version'],env=env,text=True)
    args89=['-O3','-arch=sm_89','-DQSB_ZEROS_N=24','-Xptxas=-v,--warn-on-spills','--keep','--keep-dir',guest(build),guest(stage/args.entry),'-lcrypto','-lm','-o',guest(build/'candidate')]
    run([cuda+'nvcc',*args89],build/'compile.log')
    run([cuda+'cuobjdump','--dump-sass',guest(build/'candidate')],build/'sass.txt')
    if args.default_build:
        run([cuda+'nvcc','-O3','-DQSB_ZEROS_N=24',guest(stage/args.entry),'-lcrypto','-lm','-o',guest(build/'candidate-default')],build/'default-compile.log')
    resources=parse_resources((build/'compile.log').read_text());sass=parse_sass((build/'sass.txt').read_text())
    for name in resources:resources[name].update(sass.get(name,{}))
    assert resources,'No compiler resource records found'
    assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==h for name,h in hashes.items()),'Input changed during build'
    report={'status':'PASS','validation_level':'real ARM Linux nvcc/ptxas build targeting sm_89',
        'gpu_executed':False,'compiler':compiler.strip(),'source_sha256':hashes,'entry':args.entry,
        'default_flags_build':args.default_build,'kernels':resources,'build_directory':str(build),
        'limits':'Static native codegen only. ARM host binary, no NVIDIA GPU execution, no timing, no runtime/driver-JIT equivalence claim.'}
    data=json.dumps(report,indent=2)+'\n'
    if args.report:args.report.write_text(data)
    print(data)
if __name__=='__main__':main()
