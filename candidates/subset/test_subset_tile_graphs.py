#!/usr/bin/env python3
"""Exercise the production graph controller with a CUDA API model.
This verifies ordering, cache ownership, failure fallback, and no duplicate retry.
It does not execute a CUDA graph on hardware.
"""
from pathlib import Path
import subprocess,tempfile
HERE=Path(__file__).resolve().parent
CODE=r'''
#include <cassert>
#include <cstdio>
using cudaStream_t=int*;using cudaGraph_t=int*;using cudaGraphExec_t=int*;using cudaError_t=int;
constexpr int cudaSuccess=0,cudaErrorInvalidValue=1,cudaStreamNonBlocking=2,cudaStreamCaptureModeThreadLocal=3;
int fail=0,pending=0,streams=0,graphs=0,execs=0,created=0,captures=0,instantiated=0,launches=0,work=0,captured=0;
bool recording=false;
int cudaGetLastError(){int old=pending;pending=0;return old;}
int cudaStreamCreateWithFlags(cudaStream_t*p,unsigned flags){assert(flags==cudaStreamNonBlocking);if(fail==1)return 9;*p=new int(0);streams++;created++;return 0;}
int cudaStreamDestroy(cudaStream_t p){delete p;streams--;return 0;}
int cudaStreamBeginCapture(cudaStream_t p,int mode){assert(p&&mode==cudaStreamCaptureModeThreadLocal&&!recording);if(fail==2)return 9;recording=true;captures++;captured=0;return 0;}
int cudaStreamEndCapture(cudaStream_t p,cudaGraph_t*g){assert(p&&recording);recording=false;if(fail==3)return 9;*g=new int(captured);graphs++;return 0;}
int cudaGraphInstantiate(cudaGraphExec_t *e,cudaGraph_t g,void*,void*,unsigned long){assert(g);if(fail==4)return 9;*e=new int(*g);execs++;instantiated++;return 0;}
int cudaGraphDestroy(cudaGraph_t g){delete g;graphs--;return 0;}
int cudaGraphExecDestroy(cudaGraphExec_t e){delete e;execs--;return 0;}
int cudaGraphLaunch(cudaGraphExec_t e,cudaStream_t stream){assert(e&&!stream&&!recording);launches++;if(fail==5)return 9;work+=*e;return 0;}
#include "subset_tile_graphs.h"
void reset(){assert(streams==0&&graphs==0&&execs==0&&!recording);fail=pending=created=captures=instantiated=launches=work=captured=0;}
int main(){
 auto two=[](cudaStream_t s){if(s){assert(recording);captured+=2;}else{assert(!recording);work+=2;}};
 auto three=[](cudaStream_t s){if(s){assert(recording);captured+=3;}else{assert(!recording);work+=3;}};
 {
    qsb_subset_tile_graphs g;
    assert(g.run(1,true,two)==0&&work==2);
    assert(g.run(1,true,two)==0&&work==4);
    assert(g.run(2,true,three)==0&&work==7);
    assert(g.run(1,false,two)==0&&work==9); // Tail never replays a full batch.
#if QSB_SUBSET_TILE_GRAPHS
    assert(captures==2&&instantiated==2&&launches==3&&created==1);
#else
    assert(captures==0&&instantiated==0&&launches==0&&created==0);
#endif
    assert(g.run(0,true,two)==cudaErrorInvalidValue&&work==9);
    pending=17;assert(g.run(1,true,two)==17&&work==9);
 }
 reset();
#if QSB_SUBSET_TILE_GRAPHS
 for(int error=1;error<=4;error++){
    {
      qsb_subset_tile_graphs g;fail=error;
      assert(g.run(1,true,two)==0&&work==2); // Failed capture executes no work.
      fail=0;assert(g.run(1,true,two)==0&&work==4&&launches==0); // Disabled route stays direct.
      assert(g.run(2,true,three)==0&&work==7&&launches==1); // Other route can still capture.
    }reset();
 }
 {
    qsb_subset_tile_graphs g;assert(g.run(1,true,two)==0&&work==2);
    fail=5;assert(g.run(1,true,two)==9&&work==2&&launches==2); // No speculative retry.
 }reset();
 {
    qsb_subset_tile_graphs g;
    auto bad_capture=[](cudaStream_t s){if(s){captured++;pending=19;}else work+=2;};
    assert(g.run(1,true,bad_capture)==0&&work==2&&instantiated==0);
    assert(g.run(1,true,two)==0&&work==4&&launches==0);
 }reset();
#endif
 {
    qsb_subset_tile_graphs g;
    auto bad_direct=[](cudaStream_t s){assert(!s);pending=23;};
    assert(g.run(1,false,bad_direct)==23&&work==0);
 }reset();
 puts("Graph controller: route reuse, tail bypass, upstream errors, capture failures, no launch retry, cleanup passed");
}
'''
def main():
    with tempfile.TemporaryDirectory(prefix='qsb-graph-mock-') as tmp:
        p=Path(tmp);(p/'audit.cpp').write_text(CODE)
        for enabled in (0,1):
            subprocess.run(['clang++','-std=c++17','-O2','-fsanitize=undefined,bounds',
                f'-DQSB_SUBSET_TILE_GRAPHS={enabled}','-I'+str(HERE),str(p/'audit.cpp'),'-o',str(p/'audit')],check=True)
            subprocess.run([str(p/'audit')],check=True)
if __name__=='__main__':main()
