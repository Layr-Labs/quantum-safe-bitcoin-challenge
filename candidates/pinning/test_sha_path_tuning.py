#!/usr/bin/env python3
"""Exercise the actual host selection policy, phase drains and accounting.

Artificial rates below test decisions; they are not benchmark measurements.
"""
from pathlib import Path
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent


def main():
    source=r'''
#include "sha_path_tuning.h"
#include <cassert>
#include <cstdio>
#include <limits>
static void check(const double *ratios,bool want,bool invalid=false){
 qsb_sha_path_tuning tune;
 const bool order[]={false,true,false,true,true,false,true,false,false,true};
 unsigned drains=0;uint64_t searched=0,published=0,pending[2]={};
 for(unsigned p=0;p<10;p++){
  assert(!tune.done()&&tune.split()==order[p]);
  unsigned batches=p<2?32:64;uint64_t count=0;
  for(unsigned j=0;j<batches;j++){
   // Include short batches and zero-hit batches. Every batch stays unique.
   uint64_t n=j%13==0?8191:8388608;count+=n;searched+=n;
   unsigned slot=j%2;published+=pending[slot];pending[slot]=n;
   assert(tune.add_batch(n)==(j+1==batches));
  }
  for(unsigned slot=0;slot<2;slot++){published+=pending[slot];pending[slot]=0;}
  assert(published==searched);drains++;
  double rate=4e8;
  if(p>=2&&order[p])rate*=ratios[(p-2)/2];
  double seconds=double(count)/rate;
  if(invalid&&p==4)seconds=std::numeric_limits<double>::quiet_NaN();
  tune.finish_phase(seconds);
 }
 assert(drains==10&&tune.done()&&tune.split()==want);
 assert(published==searched&&searched>0);
 auto phase=tune.phase;assert(!tune.add_batch(1));tune.finish_phase(1);
 assert(tune.phase==phase&&tune.split()==want);
}
int main(){
 const double fast[]={1.04,1.03,1.05,1.04};check(fast,true);
 const double tie[]={1,1,1,1};check(tie,false);
 const double small[]={1.012,1.012,1.012,1.012};check(small,false);
 const double regressed[]={0.96,0.97,0.96,0.97};check(regressed,false);
 const double noisy[]={1.06,0.98,1.06,1.06};check(noisy,false);
 const double two_pairs[]={1.08,1.08,1,1};check(two_pairs,false);
 const double three_pairs[]={1.03,1.03,1.03,1};check(three_pairs,false);
 const double three_strong[]={1.06,1.06,1.06,1};check(three_strong,true);
 const double slow[]={1.02,1.02,1.02,0.996};check(slow,false);
 check(fast,false,true);
 puts("10 selection scenarios, 100 drained cohorts, accounting preserved; synthetic rates only");
}
'''
    with tempfile.TemporaryDirectory(prefix='qsb-sha-tuning-test-') as tmp:
        src=Path(tmp)/'test.cpp';binary=Path(tmp)/'test'
        src.write_text(source)
        subprocess.run(['clang++','-std=c++14','-O1','-fsanitize=undefined,bounds',
                        '-I'+str(HERE),str(src),'-o',str(binary)],check=True)
        subprocess.run([str(binary)],check=True)
    pin=(HERE/'pinning.cu').read_text()
    start=pin.index('if(sha_tuning.add_batch((uint64_t)batch_sz))')
    body=pin[start:pin.index('/* Check if another GPU found it */',start)]
    assert body.index('drain_slot(drain)')<body.index('clock_gettime')<body.index('finish_phase')
    assert body.index('finish_phase')<body.index('qsb_sha_use_producer=sha_tuning.split()')
    print('Source uses full-stream drain before elapsed time and path selection')


if __name__=='__main__':main()
