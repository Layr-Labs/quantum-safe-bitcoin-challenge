#!/usr/bin/env python3
"""Actual producer64 SHA/barrier component CPU projection; not GPU validation."""
import argparse,ctypes as CT,hashlib,importlib.util,json,random,struct,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
parent=HERE.parent/'one_seven/sha-check.py'
spec=importlib.util.spec_from_file_location('producer32_oracle',parent)
O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)
F=O.F;B=O.B

def once(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b,1)

def sha(s):return hashlib.sha256(s.encode()).hexdigest()

HOST=r'''
static uint32_t atomicAdd(uint32_t *p,uint32_t x){
    if(x==0)std::this_thread::sleep_for(std::chrono::microseconds(100));
    return __atomic_fetch_add(p,x,__ATOMIC_SEQ_CST);
}
static uint32_t atomicExch(uint32_t *p,uint32_t x){return __atomic_exchange_n(p,x,__ATOMIC_SEQ_CST);}
static void __threadfence_block(){std::atomic_thread_fence(std::memory_order_seq_cst);}
static uint32_t cache_writes[8][64],cache_owners[8][64];
static void record_cache_write(unsigned owner,unsigned slot,unsigned word){
    require(owner<64 && slot<64 && word<8);
    __atomic_fetch_add(&cache_writes[word][slot],1u,__ATOMIC_SEQ_CST);
    __atomic_fetch_or(&cache_owners[word][slot],owner+1u,__ATOMIC_SEQ_CST);
}
extern "C" void cache_ownership(uint32_t *writes,uint32_t *owners){
    memcpy(writes,cache_writes,sizeof(cache_writes));memcpy(owners,cache_owners,sizeof(cache_owners));
}
'''

wrap=O.WRAP
wrap=once(wrap,'    launch(32,[&](int lane){\n        qsb_producer_prepare_first(&epoch,cache.cache,lane);', '''    uint32_t sync[2]={};
    memset(cache_writes,0,sizeof(cache_writes));memset(cache_owners,0,sizeof(cache_owners));
    launch(64,[&](int tid){
        const int lane=tid&31;
        qsb_producer64_barrier(sync);
        qsb_producer64_prepare_first(&epoch,cache.cache,tid);
        qsb_producer64_barrier(sync);''')
wrap=once(wrap,'for(int packet=0;packet<8;packet++)','for(int packet=tid/32;packet<8;packet+=2)')
wrap=once(wrap,'        require(block_syncs==0);\n    });', '        qsb_producer64_barrier(sync);\n        require(block_syncs==0);\n    });\n    require(sync[0]==0 && sync[1]==3);')
wrap=once(wrap,'    launch(32,[&](int lane){qsb_producer_prepare_first(&epoch,cache,lane);require(block_syncs==0);});', '''    memset(cache_writes,0,sizeof(cache_writes));memset(cache_owners,0,sizeof(cache_owners));
    launch(64,[&](int tid){qsb_producer64_prepare_first(&epoch,cache,tid);require(block_syncs==0);});''')
wrap+=r'''
extern "C" void barrier_lifecycle(int rounds,int *stats){
    uint32_t sync[2]={},payload[64]={};
    launch(64,[&](int tid){
        for(int epoch=0;epoch<rounds;epoch++){
            qsb_producer64_barrier(sync); // retire previous epoch before overwrite
            payload[tid]=(uint32_t)(epoch*64+tid+1);
            qsb_producer64_barrier(sync); // publish all64 owner slots
            for(int owner=0;owner<64;owner++)require(payload[owner]==(uint32_t)(epoch*64+owner+1));
        }
    });
    stats[0]=sync[0];stats[1]=sync[1];
}
'''
O.WRAP=wrap

def project(source,header):
    signature='__device__ __forceinline__ void qsb_producer64_prepare_first('
    prep=F(header,signature)
    instrumented=once(prep,'first_states[j][slot]=initial[j];','{record_cache_write(producer_tid,slot,j);first_states[j][slot]=initial[j];}')
    header=once(header,prep,instrumented)
    cpp,evidence=O.project(source,header)
    cpp=once(cpp,'static void __syncwarp(unsigned mask=0xffffffffu)',HOST+'\nstatic void __syncwarp(unsigned mask=0xffffffffu)')
    return cpp,evidence

def structural(header):
    old=(HERE.parent/'one_seven/candidate/tests/gpu_epochs/window_schedule_shared.cuh').read_text()
    assert header.startswith(old),'Existing API/header prefix changed'
    prep=F(header,'__device__ __forceinline__ void qsb_producer64_prepare_first(')
    expected=F(old,'__device__ __forceinline__ void qsb_producer_prepare_first(').replace('qsb_producer_prepare_first','qsb_producer64_prepare_first').replace('int lane)','int producer_tid)').replace('slot=lane;slot<QSB_FIRST_COUNT;slot+=32','slot=producer_tid;slot<QSB_FIRST_COUNT;slot+=64').replace('    __syncwarp(0xffffffffu);\n','')
    assert prep==expected
    assert '__sync' not in prep
    inv=(HERE.parent/'one_seven/candidate/tests/gpu_epochs/tree_inverse.cuh').read_text()
    barrier=F(header,'__device__ __forceinline__ void qsb_producer64_barrier(')
    expected_barrier=F(inv,'__device__ __forceinline__ void qsb_ec224_barrier(').replace('qsb_ec224_barrier','qsb_producer64_barrier').replace('ticket==6u','ticket==1u')
    assert barrier==expected_barrier
    assert 'if(first_distinct>64)' in header
    return {'original_header_sha256':sha(old),'prepare_sha256':sha(prep),'barrier_sha256':sha(barrier),'hash_tail_sha256':sha(F(header,'__device__ __forceinline__ void qsb_producer_window_hash('))}

def library(cpp,tmp,name):
    lib=O.compile_lib(cpp,tmp,name)
    lib.cache_ownership.argtypes=[B.P32,B.P32]
    lib.barrier_lifecycle.argtypes=[CT.c_int,B.PI32]
    return lib

def boundary(lib):
    rng=random.Random(648);words=[rng.getrandbits(32) for _ in range(64*14)]
    count=0;owned=0
    for n in [1,31,32,33,54,63,64]:
        out=(B.U32*512)();writes=(B.U32*512)();owners=(B.U32*512)()
        lib.cache_boundary(n,(B.U32*len(words))(*words),out);lib.cache_ownership(writes,owners)
        for c in range(64):
            want=B.C.sha256_midstate(bytes(range(1,9))+struct.pack('>14I',*words[c*14:c*14+14])) if c<n else [0xa5a5a5a5]*8
            assert [out[j*64+c] for j in range(8)]==want,(n,c)
            for j in range(8):
                assert writes[j*64+c]==int(c<n),(n,c,j,'write_count')
                assert owners[j*64+c]==(c+1 if c<n else 0),(n,c,j,'owner')
                owned+=1
            count+=1
    return {'cache_class_slots_checked':count,'cache_word_owner_counts_checked':owned,'producer_owner_ids_exercised':list(range(64))}

def barrier_model(last_ticket=1,late_reset=False,rounds=3):
    # Two warp leaders are the scheduling units. A leader reaches this model
    # only after its actual leading warp join. Atomic/fence ordering is SC here.
    # steps0read-generation,1arrive,2reset-or-wait,3publish-or-reset,4exit.
    start=(0,0,((0,0,-1,-1),(0,0,-1,-1)),(0,)*rounds)
    pending=[start];seen={start};bad=[]
    while pending:
        arrivals,generation,leaders,masks=pending.pop();transitions=[]
        if all(x[0]==rounds for x in leaders):continue
        for w,(r,step,saved,ticket) in enumerate(leaders):
            if r==rounds:continue
            aa,gg=arrivals,generation;mm=list(masks);ll=list(leaders)
            if step==0:ll[w]=(r,1,generation,-1)
            elif step==1:
                ticket=arrivals;aa+=1;mm[r]|=1<<w;ll[w]=(r,2,saved,ticket)
            elif step==2:
                if ticket==last_ticket:
                    if late_reset:gg+=1
                    else:aa=0
                    ll[w]=(r,3,saved,ticket)
                elif generation!=saved:ll[w]=(r,4,saved,ticket)
                else:continue
            elif step==3:
                if late_reset:aa=0
                else:gg+=1
                ll[w]=(r,4,saved,ticket)
            else:
                if masks[r]!=3:bad.append('premature_exit');continue
                ll[w]=(r+1,0,-1,-1)
            nxt=(aa,gg,tuple(ll),tuple(mm));transitions.append(nxt)
        if not transitions:bad.append('deadlock')
        for nxt in transitions:
            if nxt not in seen:seen.add(nxt);pending.append(nxt)
    return {'states':len(seen),'rounds':rounds,'violations':sorted(set(bad))}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,default=HERE/'candidate');ap.add_argument('--report',type=Path,default=HERE/'sha-check-results.json');args=ap.parse_args()
    source=args.source.resolve();path=source/'tests/gpu_epochs/window_schedule_shared.cuh';header=path.read_text()
    component=structural(header);cpp,evidence=project(source,header)
    negatives={}
    with tempfile.TemporaryDirectory(prefix='qsb-producer64-') as temp:
        tmp=Path(temp);lib=library(cpp,tmp,'positive');counts=O.hashes(lib);counts.update(boundary(lib))
        stats=(B.I32*2)();lib.barrier_lifecycle(64,stats);assert list(stats)==[0,128]
        counts['actual_barrier_lifecycle_epochs']=64;counts['actual_barrier_generations']=128
        prep=F(header,'__device__ __forceinline__ void qsb_producer64_prepare_first(')
        mutants={
         'missing_upper_producer_warp':header.replace(prep,prep.replace('slot<QSB_FIRST_COUNT;','slot<QSB_FIRST_COUNT && slot<32;')),
         'wrong_packet_choice':header.replace('QSB_FIRST_CLASS[choice]','QSB_FIRST_CLASS[choice&31]'),
         'missing_final_sha':header.replace('digest[j]=s2[j]','digest[j]=state[j]')}
        for name,mut in mutants.items():
            ml=library(project(source,mut)[0],tmp,name)
            try:boundary(ml) if name=='missing_upper_producer_warp' else O.hashes(ml,quick=True)
            except AssertionError:negatives[name]='COMPILED_AND_DETECTED'
            else:raise AssertionError('mutant escaped: '+name)
    model=barrier_model();assert not model['violations']
    for name,kwargs in [('one_warp_threshold',{'last_ticket':0}),('publish_before_reset',{'late_reset':True})]:
        result=barrier_model(**kwargs);assert result['violations'];negatives[name]=result
    for name,mut in [('missing_trailing_warp_join',header.rsplit('__syncwarp();',1)[0]+header.rsplit('__syncwarp();',1)[1]),('wrong_owner_stride',header.replace('slot+=64','slot+=32'))]:
        try:structural(mut)
        except AssertionError:negatives[name]='STRUCTURALLY_DETECTED'
        else:raise AssertionError(name)
    assert path.read_text()==header,'Owned header changed during check'
    # Parent owns concurrently changing kernel/other files. Bind only exact
    # dependencies actually compiled, and verify those did not change.
    assert project(source,header)[1]==evidence,'Extracted frontend dependency changed'
    report={'status':'PASS','validation_level':'CPU actual-source SHA + actual atomic two-warp barrier projection and bounded lifecycle model','component_binding':component,'header_sha256':sha(header),'checker_sha256':sha(Path(__file__).read_text()),'projection_sha256':sha(cpp),'extracted_dependencies':evidence,**counts,'barrier_model':model,'negative_controls':negatives,'cache_capacity':64,'producer_threads':64,'internal_prepare_synchronization':False,'existing_APIs_preserved_exact_prefix':True,'whole_closure_bound':False,'gpu_executed':False,'native_compiled':False,'limits':'Exact SHA helper statements execute with caller publication/retirement barrier and ownership instrumentation. CPU uses sequentially consistent atomic/fence shims, warp barriers and100us atomic-load backoff. Model schedules two leaders for three generations; no GPU memory ordering, ring lifetime, full kernel correctness, performance or achieved occupancy claim. Full closure intentionally unbound during concurrent parent kernel work.'}
    args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
