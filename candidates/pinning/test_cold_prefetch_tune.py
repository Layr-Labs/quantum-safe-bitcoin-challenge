#!/usr/bin/env python3
"""Execute the production timing helper with an ordered fake CUDA runtime.

Checks timing decisions, event ordering, repeated launch counts, cleanup and
every API failure. This is CPU control-flow evidence, not GPU/cache evidence.
"""
import json
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
CPP = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <vector>
using cudaError_t=int;
using cudaStream_t=int;
struct Event {int position=-1;};
using cudaEvent_t=Event*;
constexpr int cudaSuccess=0,cudaErrorInvalidValue=1,injected=77;
int pin_cold_prefetch_enabled=1;
int calls,fail_at,launches,pending,samples,created,destroyed;
bool ready;
float durations[8];
std::set<Event*> live;
std::vector<int> modes,timed_counts;
int step(){return ++calls==fail_at?injected:cudaSuccess;}
int cudaStreamSynchronize(int stream){
    assert(stream==17);int e=step();if(e)return e;
    ready=true;pending=0;return 0;
}
int cudaEventCreate(Event **p){
    int e=step();if(e)return e;
    *p=new Event;assert(live.insert(*p).second);created++;return 0;
}
int cudaEventDestroy(Event *p){
    assert(live.erase(p)==1);delete p;destroyed++;return step();
}
int cudaMemcpyToSymbol(int &symbol,const int *value,size_t size){
    assert(ready && pending==0 && size==sizeof(int));
    assert(&symbol==&pin_cold_prefetch_enabled);
    int e=step();if(e)return e;
    symbol=*value;modes.push_back(*value);return 0;
}
int cudaEventRecord(Event *p,int stream){
    assert(stream==17 && live.count(p));int e=step();if(e)return e;
    p->position=launches;return 0;
}
int cudaEventSynchronize(Event *p){
    assert(live.count(p) && p->position==launches);
    int e=step();if(e)return e;pending=0;return 0;
}
int cudaEventElapsedTime(float *ms,Event *a,Event *b){
    assert(pending==0 && live.count(a) && live.count(b));
    int e=step();if(e)return e;
    timed_counts.push_back(b->position-a->position);
    assert(samples<8);*ms=durations[samples++];return 0;
}
int launch(){
    assert(ready && !modes.empty());int e=step();if(e)return e;
    launches++;pending++;return 0;
}
#include "ColdPrefetchTune.h"
void reset(const float ms[8],int fail=0){
    assert(live.empty());calls=launches=pending=samples=created=destroyed=0;
    ready=false;fail_at=fail;modes.clear();timed_counts.clear();
    pin_cold_prefetch_enabled=1;std::copy(ms,ms+8,durations);
}
int success_case(const float ms[8],int expected){
    reset(ms);assert(qsb_tune_cold_prefetch(17,launch)==cudaSuccess);
    assert(live.empty() && created==2 && destroyed==2);
    assert(launches==264 && samples==8 && pending==0);
    assert(modes==std::vector<int>({0,1,1,0,1,0,0,1,expected}));
    assert(pin_cold_prefetch_enabled==expected);
    assert(timed_counts==std::vector<int>(8,32));return calls;
}
int main(){
    const float gain[8]={100,95,95,100,95,100,100,95};
    const float flat[8]={100,100,100,100,100,100,100,100};
    const float loss[8]={100,105,105,100,105,100,100,105};
    const float border[8]={100,99,99,100,99,100,100,99};
    const float mixed[8]={100,90,90,100,101,100,100,101};
    const float inconsistent[8]={100,90,90,100,99.6f,100,100,99.6f};
    int total=success_case(gain,1);
    success_case(flat,0);success_case(loss,0);success_case(border,0);
    success_case(mixed,0);success_case(inconsistent,0);
    for(int fault=1;fault<=total;fault++){
        reset(gain,fault);
        assert(qsb_tune_cold_prefetch(17,launch)==injected);
        assert(live.empty() && created==destroyed);
    }
    const float invalid[]={0,-1,std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()};
    for(float value:invalid)for(int sample=0;sample<8;sample++){
        reset(gain);durations[sample]=value;
        assert(qsb_tune_cold_prefetch(17,launch)==cudaErrorInvalidValue);
        assert(samples==sample+1 && live.empty() && created==destroyed);
    }
    printf("{\"decision_cases\":6,\"injected_api_failures\":%d,"
           "\"invalid_timing_cases\":32,\"launches_per_calibration\":264,"
           "\"timed_launches\":256,\"warmups\":8}",total);
}
'''

with tempfile.TemporaryDirectory(prefix='qsb-prefetch-tune-') as td:
    td=Path(td);src=td/'test.cpp';exe=td/'test';src.write_text(CPP)
    subprocess.run(['g++','-O2','-std=c++17','-fsanitize=undefined',
                    '-fno-sanitize-recover=all','-I',str(HERE),str(src),
                    '-o',str(exe)],check=True)
    result=json.loads(subprocess.check_output([str(exe)],text=True))
print(json.dumps(dict(actual_header=True,ubsan='pass',gpu_executed=False,**result),indent=2))
