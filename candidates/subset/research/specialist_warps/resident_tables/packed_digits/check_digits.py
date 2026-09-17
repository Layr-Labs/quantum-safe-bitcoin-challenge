#!/usr/bin/env python3
"""Execute actual source helpers as ordinary C++ against independent Python integers."""
import argparse
import hashlib
import json
import random
import re
import struct
import subprocess
from pathlib import Path
import check_model as model

HERE = Path(__file__).resolve().parent

CPP = r'''
#include <cstdint>
#include <cstdio>
#include <cstring>
#define __device__
#define __constant__
#define __forceinline__ inline
ACTUAL_HELPERS
static volatile uint64_t arena[4][768];
int main() {
  uint64_t record[8]; unsigned long long cases=0,aliases=0,digit_cases=0;
  while(fread(record,sizeof(record),1,stdin)==1) {
    const uint64_t *expected=record+4;
    uint64_t result[4]; qsb_pack_scalar16(record,result);
    if(memcmp(result,expected,32)){fprintf(stderr,"DIGIT_MISMATCH normal case=%llu\n",cases);return 3;}
    for(int delta=-3;delta<=3;delta++) {
      uint64_t buffer[12],before[12];
      for(int i=0;i<12;i++)buffer[i]=0x1234567890abcdefULL+i;
      memcpy(buffer+4,record,32); memcpy(before,buffer,sizeof(buffer));
      qsb_pack_scalar16(buffer+4,buffer+4+delta);
      if(memcmp(buffer+4+delta,expected,32)){fprintf(stderr,"ALIAS_MISMATCH case=%llu delta=%d\n",cases,delta);return 4;}
      for(int i=0;i<12;i++)if((i<4+delta||i>=8+delta)&&buffer[i]!=before[i]){
        fprintf(stderr,"ALIAS_CANARY case=%llu delta=%d\n",cases,delta);return 5;}
      aliases++;
    }
    uint32_t digest[8];
    for(int j=0;j<4;j++){digest[6-2*j]=record[j]>>32;digest[7-2*j]=record[j];}
    qsb_pack_digest16(digest);
    for(int j=0;j<4;j++)result[j]=((uint64_t)digest[6-2*j]<<32)|digest[7-2*j];
    if(memcmp(result,expected,32)){fprintf(stderr,"WIRE_MISMATCH case=%llu\n",cases);return 6;}
    unsigned tid=cases%192;
    for(int j=0;j<4;j++)arena[j][tid]=result[j];
    for(unsigned c=0;c<16;c++) {
      uint32_t idx=~0u;uint64_t neg=~0ULL;
      qsb_ec192_packed_digit(arena,tid,c,&idx,&neg);
      uint32_t cell=(expected[c/4]>>(16*(c%4)))&65535;
      if(idx!=(cell&32767)||neg!=(cell>>15)){
        fprintf(stderr,"EXTRACT_MISMATCH case=%llu c=%u idx=%u neg=%llu\n",cases,c,idx,(unsigned long long)neg);return 7;}
      digit_cases++;
    }
    cases++;
  }
  if(ferror(stdin)||!cases){fprintf(stderr,"INPUT_ERROR\n");return 8;}
  unsigned long long exhaustive=0;
  for(unsigned c=0;c<16;c++)for(unsigned cell=0;cell<65536;cell++) {
    unsigned tid=cell%192;
    for(int j=0;j<4;j++)arena[j][tid]=0xa55af00fdeadbeefULL;
    arena[c/4][tid]=(arena[c/4][tid]&~(65535ULL<<(16*(c%4))))|((uint64_t)cell<<(16*(c%4)));
    uint32_t idx;uint64_t neg;qsb_ec192_packed_digit(arena,tid,c,&idx,&neg);
    if(idx!=(cell&32767)||neg!=(cell>>15)){fprintf(stderr,"EXTRACT_EXHAUSTIVE_MISMATCH c=%u cell=%u\n",c,cell);return 9;}
    exhaustive++;
  }
  printf("PASS cases=%llu aliases=%llu digit_cases=%llu exhaustive=%llu\n",cases,aliases,digit_cases,exhaustive);
}
'''

def extract(text, name):
    return '__device__ __forceinline__ void '+model.body(text,name)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=HERE.parents[3])
    p.add_argument('--output',type=Path,default=HERE/'digits-results')
    a=p.parse_args();root=a.source.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    hashes,fp=model.closure(root)
    assert fp=='cd3d11f24e6fd42459eed3ba3a431c7f952de506cb1d22dfddc9b291e4c98667'
    tree=(root/'tests/gpu_epochs/tree.cu').read_text()
    device=(root/'tests/gpu_epochs/compact_table_device.cuh').read_text()
    constant=re.search(r'__device__ __constant__ uint64_t GT_ORDER_N\[4\] = \{.*?\};',tree,re.S).group()
    parts={'GT_ORDER_N':constant,'gt_recode_setup':extract(tree,'gt_recode_setup')}
    for name in ['qsb_pack_scalar16','qsb_pack_digest16','qsb_ec192_packed_digit']:
        parts[name]=extract(device,name)
    helpers='\n'.join(parts.values())
    rng=random.Random(0xAB116)
    cases={0,1,2,model.N-1,model.N,model.N+1,model.LIMIT-1}
    sha_byte_vectors = [hashlib.sha256(message).digest() for message in (b'', b'abc', bytes(range(32)))]
    for digest_bytes in sha_byte_vectors:
        digest_words = struct.unpack('>8I', digest_bytes)
        scalar = int.from_bytes(digest_bytes, 'big')
        assert model.integer([(digest_words[6-2*j]<<32)|digest_words[7-2*j] for j in range(4)]) == scalar
        cases.add(scalar)
    for center in [model.N,model.N//2,(model.N+1)//2,model.LIMIT//2]+[1<<b for b in range(256)]:
        cases.update(x for x in range(center-3,center+4) if 0<=x<model.LIMIT)
    fixture=HERE.parent/'small32/field-check/chain_recovery/exception-scalars.json'
    cases.update(int(x,0) for x in json.loads(fixture.read_text()))
    cases.update(rng.getrandbits(256) for _ in range(12000))
    # All 16-bit fields crossing the first limb boundary, for both setup signs.
    inv2=pow(2,-1,model.N)
    for f in range(65536):
        M=1+(f<<49)
        for sign in (1,-1):cases.add((sign*M*inv2)%model.N)
    payload=bytearray()
    for k in sorted(cases):
        M,s=model.setup_reference(k)
        es=model.serial_reference(M,s)
        ps=[((abs(e)-1)//2)|((e<0)<<15) for e in es]
        packed=[sum(ps[4*j+i]<<(16*i) for i in range(4)) for j in range(4)]
        payload.extend(struct.pack('<8Q',*model.limbs(k),*packed))
    original=CPP.replace('ACTUAL_HELPERS',helpers)
    mutations={
      'descending_groups':('for(int k=0;k<4;k++){\n        const uint64_t lo=M[k]',
                           'for(int k=3;k>=0;k--){\n        const uint64_t lo=M[k]','DIGIT_MISMATCH'),
      'omit_crossing_bit':('if(j==3)f|=(uint32_t)(hi&1ULL)<<15;','if(false)f|=(uint32_t)(hi&1ULL)<<15;','DIGIT_MISMATCH'),
      'top_nonlast':('const bool last=k==3&&j==3;','const bool last=false;','DIGIT_MISMATCH'),
      'drop_global_sign':('const uint32_t sflag=(uint32_t)(sign<0);','const uint32_t sflag=0;','DIGIT_MISMATCH'),
      'wire_halves_swapped':('digest[6-2*k]=(uint32_t)(words[k]>>32);','digest[6-2*k]=(uint32_t)words[k];','WIRE_MISMATCH'),
      'unmasked_cell':('const uint32_t cell=(uint32_t)(arena[c>>2][tid]>>(16*(c&3u)))&0xffffu;',
                       'const uint32_t cell=(uint32_t)(arena[c>>2][tid]>>(16*(c&3u)));','EXTRACT_MISMATCH'),
    }
    results={}
    for name in ['positive']+list(mutations):
        source=original
        if name!='positive':
            before,after,diagnostic=mutations[name]
            assert source.count(before)==1,(name,source.count(before))
            source=source.replace(before,after)
        cpp=out/(name+'.cpp');exe=out/(name+'.bin');cpp.write_text(source)
        command=['clang++','-std=c++17','-O2','-Wall','-Wextra',str(cpp),'-o',str(exe)]
        build=subprocess.run(command,capture_output=True,text=True,timeout=60)
        (out/(name+'.compile.log')).write_text(build.stdout+build.stderr)
        assert build.returncode==0,(name,build.stderr)
        run=subprocess.run([str(exe)],input=payload,capture_output=True,timeout=60)
        stdout,stderr=run.stdout.decode(),run.stderr.decode()
        (out/(name+'.stdout.log')).write_text(stdout)
        (out/(name+'.stderr.log')).write_text(stderr)
        if name=='positive':assert run.returncode==0 and stdout.startswith('PASS '),(stdout,stderr)
        else:assert run.returncode!=0 and diagnostic in stderr,(name,run.returncode,stderr)
        results[name]={'compile_exit':build.returncode,'run_exit':run.returncode,'stdout':stdout,'stderr':stderr,
                       'generated_cpp_sha256':model.sha(source.encode()),'binary_sha256':model.sha(exe.read_bytes())}
        exe.unlink() # Reproducible from source, no platform binary in public evidence.
    report={'status':'PASS','source_fingerprint':fp,'source_sha256':hashes,
            'actual_helper_sha256':{name:model.sha(src.encode()) for name,src in parts.items()},
            'checker_sha256':model.sha(Path(__file__).read_bytes()),'model_dependency_sha256':model.sha(Path(model.__file__).read_bytes()),
            'scope':'Extracted actual C++ scalar setup, packers and shared digit helper executed on native Mac CPU; Python modular/serial-digit oracle. No field arithmetic, CUDA execution, curve or ring scheduling claim.',
            'input_cases':len(cases),'binary_fixture_sha256':model.sha(payload),'binary_fixture_bytes':len(payload),
            'fixture_generation':'Deterministic in checker; sent over stdin, not persisted.',
            'overlap_offsets_words':list(range(-3,4)),'all_consumer_owners_exercised':list(range(192)),
            'exhaustive_extract_cases':16*65536,'crossing_field_sign_cases_generated':2*65536,
            'known_SHA256_byte_ABI_vectors':[v.hex() for v in sha_byte_vectors],
            'wire_protocol':'Actual prototype preserves reversed high/low SHA word wiring; digest[6-2*k]=high32(packed[k]), digest[7-2*k]=low32(packed[k]).',
            'results':results}
    (out/'results.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'PASS','source_fingerprint':fp,'input_cases':len(cases),'positive':results['positive']['stdout'].strip(),'negative_controls':len(mutations)}))

if __name__=='__main__':main()
