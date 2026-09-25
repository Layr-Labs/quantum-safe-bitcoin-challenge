#!/usr/bin/env python3
"""Compile/test the actual host decision controller only; no CUDA or GPU."""
from pathlib import Path
import argparse,hashlib,json,subprocess,tempfile
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);a=ap.parse_args()
H=Path(__file__).resolve().parents[2]/'native_adapt.cuh';header=H.read_text()
code=r'''
#include "native_adapt.cuh"
#include <assert.h>
#include <limits>
using qsb_adapt::State;
static void drain(State&s,unsigned slot){auto x=s.slots[slot];assert(s.publish(slot,x.base,x.count,x.native));}
static void window(State&s,double gpu,double host){
 assert(s.begin(s.next));bool route=s.backend();unsigned n=s.target();
 for(unsigned j=0;j<n;j++){unsigned slot=j&1;if(s.slots[slot].busy)drain(s,slot);assert(s.launch(slot,s.next,s.capacity,route));}
 assert(s.window_full());drain(s,0);drain(s,1);assert(s.finish(gpu,host));
}
static State run(double ng,double sg,double nh,double sh){State s(8);window(s,1,1);window(s,1,1);window(s,ng,nh);window(s,sg,sh);window(s,sg,sh);window(s,ng,nh);return s;}
int main(){
 State tie=run(100,100,100,100);assert(!tie.sampling()&&tie.backend());
 State win=run(104,100,104,100);assert(!win.backend());
 State margin=run(101,100,104,100);assert(margin.backend());
 State disagree=run(104,100,99,100);assert(disagree.backend());
 State invalid=run(std::numeric_limits<double>::quiet_NaN(),100,104,100);assert(invalid.backend());
 // Exact36-batch measured prefix followed by ordinary steady work; no replay.
 assert(tie.next==36*8&&tie.published==tie.next);assert(tie.launch(0,tie.next,8,true));drain(tie,0);assert(tie.next==37*8);
 State s(8);assert(s.begin(0));assert(!s.launch(0,1,8,true));assert(!s.launch(0,0,9,true));assert(!s.launch(0,0,8,false));assert(s.launch(0,0,8,true));assert(!s.launch(0,8,8,true));assert(!s.begin(8));assert(!s.finish(1,1));assert(s.launch(1,8,8,true));
 assert(!s.publish(1,8,8,true));assert(!s.publish(0,0,7,true));assert(!s.publish(0,0,8,false));drain(s,0);assert(!s.publish(0,0,8,true));drain(s,1);assert(s.finish(1,1));assert(!s.finish(1,1));
 // A final short batch is publishable but never a full price window.
 State tail(8);assert(tail.begin(0));assert(tail.launch(0,0,3,true));drain(tail,0);assert(!tail.window_full()&&!tail.finish(1,1));
 State overflow(8);overflow.next=overflow.published=UINT64_MAX-3;assert(overflow.begin(overflow.next));assert(!overflow.launch(0,overflow.next,8,true));
 State zero(0);assert(!zero.begin(0));
}
'''
faults={'tie':'latched_native=!source;','order':'slots[slot].base!=published','bounds':'count>capacity','route':'native!=backend()'}
replacements={'tie':'latched_native=source;','order':'false','bounds':'false','route':'false'}
records=[]
with tempfile.TemporaryDirectory() as td:
 T=Path(td);(T/'test.cc').write_text(code)
 for name in ['control']+list(faults):
  s=header
  if name!='control':assert s.count(faults[name])==1;s=s.replace(faults[name],replacements[name])
  (T/'native_adapt.cuh').write_text(s)
  build=subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-Werror','-I'+str(T),str(T/'test.cc'),'-o',str(T/'test')],capture_output=True,text=True)
  assert build.returncode==0,(name,build.stderr)
  p=subprocess.run([str(T/'test')],capture_output=True,text=True);assert (p.returncode==0)==(name=='control'),(name,p.returncode,p.stderr)
  records.append(dict(case=name,exit=p.returncode,expected_pass=name=='control'))
r=dict(status='PASS',gpu_executed=False,header_sha256=hashlib.sha256(header.encode()).hexdigest(),tests=records)
if a.out:a.out.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
