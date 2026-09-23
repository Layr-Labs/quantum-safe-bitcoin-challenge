#!/usr/bin/env python3
"""Execute the actual host mask-drain/publication loop with a controlled gate.

Tests selection/packing/output only. Cryptographic gate semantics are audited
separately; false GPU nominations must not leak through this loop.
"""
from pathlib import Path
import ctypes,hashlib,json,random,re,subprocess,tempfile

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'affine_driver.cuh'


def main():
    source=SOURCE.read_text()
    start=source.index('            FILE* output=nullptr;')
    end=source.index('            searched+=count;',start)
    loop=source[start:end]
    body=r'''
#include <cstdint>
#include <cstdio>
struct Gate{int group,ctx,order,nri,recovery;};
static const uint8_t* accepted;static unsigned accepted_base,gate_calls;
static int qsb_host_exact_hit(const int*,uint32_t,uint32_t lt,int ri,int,int,int,int,int){
 ++gate_calls;return (accepted[lt-accepted_base]>>ri)&1;
}
extern "C" int drain(unsigned count,uint32_t start_lt,const uint32_t* h_hits,
 const uint32_t* h_exceptions,const uint8_t* exact,const char* filename,
 uint64_t totals[3]){
 accepted=exact;accepted_base=start_lt;gate_calls=0;
 int pp=0;Gate gate{};uint64_t seq=0x80000013u,published=0,exceptions=0;
'''+loop+r'''
 totals[0]=published;totals[1]=exceptions;totals[2]=gate_calls;return 0;
}
'''
    rng=random.Random(0x5055424c495348)
    U32,U64,U8=ctypes.c_uint32,ctypes.c_uint64,ctypes.c_uint8
    checks={'cases':0,'candidate_slots':0,'published_recids':0,'exception_slots':0,
            'false_nominations':0,'double_hit_candidates':0,'exception_only_hits':0}
    with tempfile.TemporaryDirectory(prefix='.audit-affine-publish-',dir=HERE) as directory:
        tmp=Path(directory);cpp=tmp/'audit.cpp';so=tmp/'audit.so';output=tmp/'hits.txt';cpp.write_text(body)
        subprocess.run(['rtk','proxy','g++','-std=c++17','-O2','-shared','-fPIC',str(cpp),'-o',str(so)],check=True)
        lib=ctypes.CDLL(str(so));lib.drain.argtypes=[U32,U32,ctypes.POINTER(U32),ctypes.POINTER(U32),ctypes.POINTER(U8),ctypes.c_char_p,ctypes.POINTER(U64)]
        for count in [1,2,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,513]:
            for trial in range(5):
                hits=(U32*((count+15)//16))();exceptions=(U32*((count+31)//32))();exact=(U8*count)()
                nominated=[];exceptional=[]
                expected=[];want_exceptions=want_calls=0
                start_lt=0xffffffff-count+1
                for i in range(count):
                    # Each position cycles through all four true-recipient
                    # masks, including no recipient and both recipients.
                    exact[i]=(i+trial)%4
                    nomination=rng.randrange(4)
                    fallback=(i+2*trial)%5==0
                    hits[i//16]|=nomination<<(2*(i%16))
                    exceptions[i//32]|=int(fallback)<<(i%32)
                    if nomination or fallback:
                        want_calls+=2;want_exceptions+=fallback
                        for recid in range(2):
                            if exact[i]&(1<<recid):expected.append((0x80000013,start_lt+i,recid))
                        checks['false_nominations']+=bool(nomination and not exact[i])
                        checks['double_hit_candidates']+=exact[i]==3
                        checks['exception_only_hits']+=bool(fallback and not nomination and exact[i])
                # Bits beyond the last valid candidate must be ignored.
                if count%16:hits[-1]|=((1<<32)-1)^((1<<(2*(count%16)))-1)
                if count%32:exceptions[-1]|=((1<<32)-1)^((1<<(count%32))-1)
                if output.exists():output.unlink()
                totals=(U64*3)()
                assert lib.drain(count,start_lt,hits,exceptions,exact,str(output).encode(),totals)==0
                got=[]
                if output.exists():
                    for line in output.read_text().splitlines():
                        match=re.fullmatch(r'sequence=(\d+) locktime=(\d+) recid=([01])',line)
                        assert match;got.append(tuple(map(int,match.groups())))
                assert got==expected,(count,trial)
                assert len(set(got))==len(got)
                assert list(totals)==[len(expected),want_exceptions,want_calls]
                checks['cases']+=1;checks['candidate_slots']+=count
                checks['published_recids']+=len(got);checks['exception_slots']+=want_exceptions
    report={'checks':checks,'mismatches':0,'gpu_execution':False,
            'scope':'Actual host mask-drain, both-recid checking and text writer; controlled exact-gate predicate; all partial-word edges and false nominations.',
            'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'extracted_loop_sha256':hashlib.sha256(loop.encode()).hexdigest()}
    (HERE/'affine_publish_result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
