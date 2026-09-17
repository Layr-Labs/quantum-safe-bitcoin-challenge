#!/usr/bin/env python3
"""Actual producer helper CPU projection; not CUDA, FIFO, or GPU validation."""
import argparse,ctypes as CT,hashlib,importlib.util,itertools,json,math,os,random,shlex,struct,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SUBSET=HERE.parents[2]
sys.path.insert(0,str(SUBSET))
import check_candidate as B
F=B.function
WRAP=r'''
static std::atomic<int> warp_syncs{0};
extern "C" int init_producer(const uint8_t *rows,const uint8_t *windows,const uint32_t *constant,int *stats){
    require(qsb_prepare_constant_schedule(constant,69)==0);
    symbol_writes=0;int old_count=QSB_FIRST_COUNT,old_host=qsb_first_class_count;
    int rc=qsb_prepare_window_schedule(rows,(const uint8_t (*)[3])windows,constant);
    stats[0]=symbol_writes;stats[1]=QSB_FIRST_COUNT;stats[2]=qsb_first_class_count;
    if(rc)require(old_count==QSB_FIRST_COUNT && old_host==qsb_first_class_count && !symbol_writes);
    memset(BINOM_C,0,sizeof(BINOM_C));
    for(int n=0;n<=150;n++)for(int k=0;k<=9;k++){
        if(k==0||k==n)BINOM_C[n][k]=1;
        else if(k<n)BINOM_C[n][k]=BINOM_C[n-1][k]+BINOM_C[n-1][k-1];
    }return rc;
}
extern "C" void producer_epoch(uint64_t rank,uint64_t total,const uint32_t *mid,
 const uint8_t *pre,int prelen,const uint8_t *rows,uint32_t *out,uint32_t *legacy,uint8_t *early){
    epoch_desc_t epoch;threadIdx.x=blockIdx.x=0;blockDim.x=1;
    kernel_build_epochs(rank,total,137,6,mid,pre,prelen,rows,&epoch);
    memcpy(early,epoch.early,6);
    struct Guarded {uint32_t before[8],cache[8][64],after[8];} cache;
    memset(&cache,0xa5,sizeof(cache));
    launch(32,[&](int lane){
        qsb_producer_prepare_first(&epoch,cache.cache,lane);
        for(int packet=0;packet<8;packet++){
            int choice=packet*32+lane;
            qsb_producer_window_hash(out+8*choice,cache.cache,choice);
        }
        require(block_syncs==0);
    });
    for(int i=0;i<8;i++)require(cache.before[i]==0xa5a5a5a5 && cache.after[i]==0xa5a5a5a5);
    launch(256,[&](int lane){
        uint32_t state[8];qsb_scheduled_window_hash(state,&epoch,lane);
        uint32_t b2[16]={},s2[8];memcpy(b2,state,32);b2[8]=0x80000000u;b2[15]=256;
        _SHA256Initialize(s2);_SHA256Transform(s2,b2);memcpy(legacy+8*lane,s2,32);
        require(block_syncs==1);
    });
}
extern "C" void cache_boundary(int count,const uint32_t *words,uint32_t *out){
    QSB_FIRST_COUNT=count;epoch_desc_t epoch={};_SHA256Initialize(epoch.mid);
    epoch.remW[0]=0x01020304;epoch.remW[1]=0x05060708;
    for(int j=0;j<14;j++)for(int c=0;c<64;c++)QSB_FIRST_UNIQUE[j][c]=words[14*c+j];
    uint32_t cache[8][64];memset(cache,0xa5,sizeof(cache));
    launch(32,[&](int lane){qsb_producer_prepare_first(&epoch,cache,lane);require(block_syncs==0);});
    memcpy(out,cache,sizeof(cache));
}
'''

def packed_windows():
    triples=[w for w in itertools.combinations(range(13),3) if not(w[0]>=1 and w[2]<=7 and w[:2]!=(1,2))]
    def key(w):
        kept=[i for i in range(13) if i not in w];return tuple(reversed(kept[-5:])),tuple(kept[:6])
    groups={}
    for w in sorted(triples,key=key):groups.setdefault(key(w)[0],[]).append(tuple(i+137 for i in w))
    bins=[[] for _ in range(8)]
    for _,group in sorted(groups.items(),key=lambda it:(-len(it[1]),it[0])):
        next(b for b in bins if len(b)+len(group)<=32).extend(group)
    assert all(len(b)==32 for b in bins)
    return [w for b in bins for w in b]

def class_windows(n):
    unique={}
    for w in itertools.combinations(range(137,150),3):
        key=tuple(i for i in range(137,150) if i not in w)[:6]
        unique.setdefault(key,w)
    reps=list(unique.values())[:n];return [reps[i%n] for i in range(256)]

def project(source,header):
    tree=(source/'tests/gpu_epochs/tree.cu').read_text()
    sha=(source/'GPUHash.h').read_text().split('//Modified SHA256 function')[0]
    shape=tree[tree.index('#define QSB_SE_N_INC'):tree.index('/* The first 256 lexicographic')]
    backend=B.BACKEND.replace('template<class T> int cudaMemcpyToSymbol','static int symbol_writes=0;\ntemplate<class T> int cudaMemcpyToSymbol').replace('require(n<=sizeof(dst));memcpy(&dst,src,n);','symbol_writes++;require(n<=sizeof(dst));memcpy(&dst,src,n);')
    helpers=[F(tree,x) for x in ['static uint32_t qsb_host_rotr(','static int qsb_prepare_constant_schedule(','__device__ __forceinline__ void qsb_compress_constant_rolled(','__device__ __forceinline__ void unrank_combo(','__global__ void kernel_build_epochs(']]
    projection='\n'.join([backend,sha,'#define MAX_T 16\n#define SIG_PUSH_SIZE 10\nuint64_t BINOM_C[151][10];\nuint32_t QSB_CONST_SCHEDULE[4][64];',shape,'static void __syncwarp(unsigned mask=0xffffffffu){require(mask==0xffffffffu);warp_barriers[threadIdx.x/32]->wait();}',*helpers[:3],header,*helpers[3:],WRAP])
    evidence={'GPUHash_projection':hashlib.sha256(sha.encode()).hexdigest(),'epoch_shape':hashlib.sha256(shape.encode()).hexdigest(),'tree_helpers':[hashlib.sha256(h.encode()).hexdigest() for h in helpers]}
    return projection,evidence

def structural(header,base):
    for name in ['static uint32_t qsb_window_second_key(','static uint32_t qsb_window_first_key(','static int qsb_pack_second_classes(','__device__ __forceinline__ void qsb_scheduled_window_hash_scratch(','__device__ __forceinline__ void qsb_scheduled_window_hash(']:
        assert F(header,name)==F(base,name),name
    prep=F(header,'__device__ __forceinline__ void qsb_producer_prepare_first(')
    consume=F(header,'__device__ __forceinline__ void qsb_producer_window_hash(')
    assert '__syncthreads' not in prep+consume
    assert prep.count('__syncwarp(0xffffffffu);')==1 and prep.rindex('__syncwarp')>prep.rindex('first_states[j][slot]=initial[j]')
    assert 'slot=lane;slot<QSB_FIRST_COUNT;slot+=32' in prep
    old=F(base,'__device__ __forceinline__ void qsb_scheduled_window_hash_scratch(')
    old=old[old.index('    int first_slot='):old.rindex('\n}')].replace('QSB_FIRST_CLASS[lane]','QSB_FIRST_CLASS[choice]').replace('QSB_WINDOW_CLASS[lane]','QSB_WINDOW_CLASS[choice]')
    assert old in consume,'Inherited hash tail changed'
    host=F(header,'static int qsb_prepare_window_schedule(')
    assert host.index('if(first_distinct>64)')<host.index('qsb_first_class_count=first_distinct')<host.index('cudaMemcpyToSymbol')


def compile_lib(cpp,tmp,name):
    source=tmp/(name+'.cpp');libpath=tmp/(name+'.so');source.write_text(cpp)
    prefix=os.environ.get('OPENSSL_PREFIX','/opt/homebrew/opt/openssl@3')
    flags=[f'-I{prefix}/include',f'-L{prefix}/lib'] if Path(prefix).exists() else []
    cmd=shlex.split(os.environ.get('CXX','c++'))+['-std=c++17','-O2','-shared','-fPIC','-pthread','-Wno-deprecated-declarations',*flags,str(source),'-lcrypto','-o',str(libpath)]
    subprocess.run(cmd,check=True,capture_output=True)
    lib=CT.CDLL(str(libpath));lib.init_producer.argtypes=[B.P8,B.P8,B.P32,B.PI32];lib.producer_epoch.argtypes=[B.U64,B.U64,B.P32,B.P8,CT.c_int,B.P8,B.P32,B.P32,B.P8];lib.cache_boundary.argtypes=[CT.c_int,B.P32,B.P32]
    return lib

def boundary(lib):
    rng=random.Random(548);words=[rng.getrandbits(32) for _ in range(64*14)]
    checked=0
    for n in [1,31,32,33,54,63,64]:
        out=(B.U32*(8*64))();lib.cache_boundary(n,(B.U32*len(words))(*words),out)
        for c in range(64):
            want=B.C.sha256_midstate(bytes(range(1,9))+struct.pack('>14I',*words[c*14:c*14+14])) if c<n else [0xa5a5a5a5]*8
            assert [out[j*64+c] for j in range(8)]==want,(n,c)
            checked+=1
    return checked

def hashes(lib,quick=False):
    rng=random.Random(910);count=0;matrix=0;rejections=0
    seeds=[20260916] if quick else [20260916,679162400,715]
    for seed in seeds:
        prob,_=B.GEN.gen_subset(random.Random(seed))
        # Additional random runtime rows deliberately do not rely on DER syntax.
        if seed==715:prob['dummy_sigs']=[rng.randbytes(10).hex() for _ in range(150)]
        prefix=bytes.fromhex(prob['fixed_prefix']);aligned=len(prefix)//64*64
        mid=(B.U32*8)(*B.C.sha256_midstate(prefix[:aligned]));pre=B.buf(prefix[aligned:]);rows=B.buf(b''.join(bytes.fromhex(x) for x in prob['dummy_sigs']))
        suffix=bytes.fromhex(prob['tail_section']+prob['tx_suffix'])
        constant=suffix+b'\x80'+b'\0'*5+(prob['total_preimage_len']*8).to_bytes(8,'big')
        const=(B.U32*69)(*struct.unpack('>69I',constant));total=math.comb(137,6)
        plans=[(packed_windows(),54)]
        if not quick and seed==20260916:plans += [(class_windows(n),n) for n in [1,31,32,33,63,64,65,84]]
        for windows,classes in plans:
            stats=(B.I32*3)();rc=lib.init_producer(rows,B.buf(bytes(itertools.chain.from_iterable(windows))),const,stats)
            if classes>64:
                assert rc==1 and stats[0]==0;rejections+=1;continue
            assert rc==0 and stats[1]==classes,(classes,rc,list(stats));matrix+=1
            ordinals=[0] if quick else ([0,1,32767,32768,total-1,total-2]+[rng.randrange(total) for _ in range(2)] if classes==54 else [total-1])
            for ordinal in ordinals:
                out=(B.U32*(256*8))();old=(B.U32*(256*8))();early=(B.U8*6)()
                lib.producer_epoch(ordinal,total,mid,pre,len(pre),rows,out,old,early)
                assert list(out)==list(old),'Producer differs from original CTA helper/final hash'
                for choice,win in enumerate(windows):
                    want=hashlib.sha256(hashlib.sha256(B.PB.sub_preimage(prob,list(early)+list(win))).digest()).digest()
                    assert struct.pack('>8I',*out[8*choice:8*choice+8])==want,(seed,ordinal,choice)
                    count+=1
    return {'sha256d_compared_to_original_and_hashlib':count,'accepted_host_class_matrices':matrix,'rejected_before_symbol_publication':rejections}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=HERE/'candidate');p.add_argument('--report',type=Path,default=HERE/'sha-check-results.json');args=p.parse_args()
    source=args.source.resolve();identity=B.source_identity(source);path=source/'tests/gpu_epochs/window_schedule_shared.cuh';header=path.read_text()
    basepath=SUBSET/'research/geometry_frontier/shared_thirteen/policy_corrected/cold_prefetch/local_m/candidate/tests/gpu_epochs/window_schedule_shared.cuh';base=basepath.read_text()
    structural(header,base);cpp,evidence=project(source,header)
    negatives={}
    try:structural(header.replace('__syncwarp(0xffffffffu);','/* missing producer fence */'),base)
    except AssertionError:negatives['missing_fence_structural']='DETECTED'
    else:raise AssertionError('fence mutation escaped')
    with tempfile.TemporaryDirectory(prefix='qsb-producer-sha-') as temp:
        tmp=Path(temp);lib=compile_lib(cpp,tmp,'normal');counts=hashes(lib);counts['cache_class_slots_checked']=boundary(lib)
        mutants={'skip_second_cache_pass':header.replace('slot+=32','slot+=64'),'wrong_packet_choice':header.replace('QSB_FIRST_CLASS[choice]','QSB_FIRST_CLASS[choice&31]'),'missing_final_sha':header.replace('digest[j]=s2[j]','digest[j]=state[j]')}
        for name,mutant in mutants.items():
            mlib=compile_lib(project(source,mutant)[0],tmp,name)
            try:boundary(mlib) if name=='skip_second_cache_pass' else hashes(mlib,quick=True)
            except AssertionError:negatives[name]='COMPILED_AND_DETECTED'
            else:raise AssertionError(name+' escaped')
    assert path.read_text()==header,'Owned header changed during audit'
    assert B.source_identity(source)==identity,'Candidate closure changed during audit; rerun on stable source'
    report={'status':'PASS','level':'CPU actual-source projection','source_identity':identity,'closure_binding_scope':'Full closure snapshot bound; only extracted SHA/frontend component executed by this checker','header_sha256':hashlib.sha256(header.encode()).hexdigest(),'baseline_header_sha256':hashlib.sha256(base.encode()).hexdigest(),'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'extracted_dependencies':evidence,**counts,'negative_controls':negatives,'cache_capacity':64,'producer_threads':32,'producer_CTA_joins':0,'preserved_legacy_APIs_and_hostpacker':'byte-identical extracted functions','PR156_attribution':{'author':'anamdongparkjinhyeong','head':'6f6410dc7393c48c6814d016703fb404a310a3a1','scope':'64-class capacity mechanism; new producer API implemented here'},'gpu_executed':False,'native_compiled':False,'limits':'Full closure and extracted dependencies bound; only SHA/frontend component executed. CPU emulates warp/CTA joins and CUDA symbol copies; SHA uses actual header C fallback. Does not test FIFO memory ordering, GPU races, producer service rate, integrated identity, or resource residency.'}
    args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
